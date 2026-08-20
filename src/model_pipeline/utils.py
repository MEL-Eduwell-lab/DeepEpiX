import os.path as op
from pathlib import Path
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mne
import json
import pandas as pd

from typing import Tuple

RAW_READERS = {
    ".ds": mne.io.read_raw_ctf,
    ".fif": mne.io.read_raw_fif,
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
}


# read and write pickle files
def save_obj(obj, name, path):
    with open(op.join(path, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    with open(op.join(path, name), "rb") as f:
        return pickle.load(f)


# center scale each window using all window mean and std
def standardize(X, mean=None, std=None):
    if not mean:
        mean = np.mean(X, axis=(1, 2))
    if not std:
        std = np.std(X, axis=(1, 2))
    X_stand = np.zeros(X.shape)
    nb_data = X.shape[0]
    for i in range(0, nb_data, 1):
        X_stand[i, :, :] = (X[i, :, :] - mean[i]) / std[i]
    return X_stand


# read raw from different acquisition systems
def read_raw(data_path, preload, verbose, bad_channels=None):
    data_path = Path(data_path)
    reader = RAW_READERS.get(data_path.suffix)

    if reader is not None:
        raw = reader(str(data_path), preload=preload, verbose=verbose)

    elif data_path.is_dir():
        # Assume BTi/4D format: folder must contain 3 specific files
        files = list(data_path.glob("*"))
        # Try to identify the correct files by names
        raw_fname = next(
            (f for f in files if "rfDC" in f.name and f.suffix == ""), None
        )
        config_fname = next((f for f in files if "config" in f.name.lower()), None)
        hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

        if not all([raw_fname, config_fname, hs_fname]):
            raise ValueError(
                "Could not identify raw, config, or hs file in BTi folder."
            )

        raw = mne.io.read_raw_bti(
            pdf_fname=str(raw_fname),
            config_fname=str(config_fname),
            head_shape_fname=str(hs_fname),
            preload=preload,
            verbose=verbose,
        )

    else:
        raise ValueError("Unrecognized file or folder type for M/EEG data.")

    if bad_channels:
        raw.drop_channels(bad_channels)

    return raw

def load_raw_from_parquet(parquet_path: str, json_path: str) -> Tuple[mne.io.RawArray, dict]:
    """
    Reconstruct an MNE RawArray using .parquet & .jsons files.

    Parquet file must contain channels as columns & samples as rows.
    Json file must contain : sfreq, ch_names, ch_types & bads.

    Parameters
    ----------
    parquet_path: str
        Path to the .parquet file containing signal values.
    json_path: str
        Path to the .json file containing mne metadata.

    Returns
    -------
    mne.io.RawArray
        MNE instance of the reconstructed signal.
    metadata: dict
        Dictionnary containing signal metadatas.
    """
    df = pd.read_parquet(parquet_path)
    data = df.values.T  # shape: (n_channels, n_times)

    with open(str(json_path), "r") as f:
        metadata = json.load(f)

    ch_names = metadata["ch_names"]
    channel_types_dict = metadata["channel_types"]
    ch_types = [channel_types_dict[ch] for ch in ch_names]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=metadata["sfreq"],
        ch_types=ch_types, #type: ignore
    )
    info["bads"] = metadata.get("bads", [])

    if "lowpass" in metadata:
        dict.__setitem__(info, "lowpass", float(metadata["lowpass"]))
    
    if "highpass" in metadata:
        dict.__setitem__(info, "highpass", float(metadata["highpass"]))

    return mne.io.RawArray(data, info, verbose=False), metadata


def interpolate_missing_channels(raw, good_channels):
    """
    Interpolate missing M/EEG channels from a known good channel layout.

    Parameters
    ----------
    raw : mne.io.RawArray
        The raw M/EEG data object with potentially missing channels.
    good_channels : dict
        Dictionary mapping channel base names to their 3D sensor locations
        (as numpy arrays), representing the full expected channel layout.

    Returns
    -------
    mne.io.RawArray
        Raw object with missing channels interpolated and reordered
        to match the full good_channels layout.
    """

    def get_base(name):
        return name.split()[0].split("-")[0].strip()

    raw.rename_channels(get_base)
    existing_channels = set(raw.info["ch_names"])
    good_basenames = list(good_channels.keys())

    # Figure out missing channels by base name
    missing_basenames = [name for name in good_basenames if name not in existing_channels]
    extra_channels = [ch for ch in raw.info["ch_names"] if ch not in good_channels]
    new_raw = raw.copy()

    if extra_channels:
        new_raw.drop_channels(extra_channels)

    if missing_basenames:
        valid_channels = [ch for ch in new_raw.info["ch_names"] if ch not in missing_basenames]
        if not valid_channels:
            raise ValueError("Can't find any valid channel to serve as template")
        template_channel = valid_channels[0]

        for miss in missing_basenames:
            new_channel = new_raw.copy().pick([template_channel])
            new_channel.rename_channels({template_channel: miss})

            new_channel._data[:] = 0.0

            new_raw.add_channels([new_channel], force_update_info=True)

        for ch in new_raw.info["chs"]:
            if ch["ch_name"] in good_channels:
                ch["loc"] = good_channels[ch["ch_name"]]

        new_raw.reorder_channels(good_basenames)
        new_raw.info["bads"] = missing_basenames
        new_raw.interpolate_bads(origin=(0, 0, 0.04), reset_bads=True)

    else:
        new_raw.reorder_channels(good_basenames)

    return new_raw


def fill_missing_channels(raw, target_channel_count):
    """
    Fill missing channels by duplicating existing ones at regular intervals.

    Parameters
    ----------
    raw : mne.io.Raw
        The original raw object.
    target_channel_count : int
        Desired total number of channels.

    Returns
    -------
    numpy.ndarray
        Data array with inserted channels, shape (target_channel_count, n_times).
        Returns original data unchanged if current count >= target.
    """
    data = raw.get_data()
    current_count = data.shape[0]

    if current_count >= target_channel_count:
        return data

    n_missing = target_channel_count - current_count

    # Get evenly spaced indices from the existing channels to duplicate
    duplicate_indices = np.linspace(0, current_count - 1, n_missing, dtype=int)

    new_data = []

    for i in range(current_count):
        new_data.append(data[i])  # Original channel
        if i in duplicate_indices:
            new_data.append(data[i])  # Insert duplicate right after

    return np.stack(new_data, axis=0)


def compute_gfp(window):
    """Compute Global Field Power (GFP) as standard deviation across channels."""
    return np.std(window, axis=0)


def find_peak_gfp(gfp, times, smoothing_sigma=2, percentile=90):
    """Find the peak GFP time within a window after smoothing."""
    gfp_smooth = gaussian_filter1d(gfp, sigma=smoothing_sigma)

    # Thresholding: Find first peak above percentile threshold
    threshold = np.percentile(gfp_smooth, percentile)    #type: ignore
    peak_indices = np.where(gfp_smooth >= threshold)[0]

    if len(peak_indices) > 0:
        return times[peak_indices[0]]  # First peak above threshold
    else:
        return times[np.argmax(gfp_smooth)]  # Default to max if no peak found
