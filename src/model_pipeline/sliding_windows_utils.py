from tqdm import tqdm
import numpy as np
import mne
from model_pipeline.utils import (
    interpolate_missing_channels,
    fill_missing_channels,
    save_obj,
    load_obj,
    standardize,
)

from typing import Union

from pathlib import Path

def save_data_matrices(
    raw: mne.io.RawArray,
    output_dir: str,
    channel_groups: dict,
    good_channels: Union[dict, list],
    channel_type: str = "mag",
) -> None:
    """
    Extract MEG/EEG data from a RawArray and save it as a pickle file.

    Depending on whether the raw channel layout matches the reference
    ``good_channels`` layout, this function either:

    - Interpolates missing channels using spherical spline interpolation
      (when channel names overlap with ``good_channels``), or
    - Pads the data to the target channel count by duplicating existing
      channels at regular intervals (fallback path).

    The resulting data array of shape ``(n_channels, n_times)`` is saved
    under the key ``"MEG/EEG"`` as a pickle file named ``data_raw``.

    Parameters
    ----------
    raw : mne.io.RawArray
        Preprocessed MNE Raw object. Channel names may include suffixes
        (e.g. ``"MLC21-4408"``), which are stripped to base names
        (e.g. ``"MLC21"``) before comparison with ``good_channels``.
    output_dir : dict[str, list[str]]
        Directory where processed data will be stored.
    channel_groups : dict
        Mapping from group name to list of channel names belonging to that
        group. Groups named ``"bad"`` and ``"EEG"`` are excluded when
        building the channel order for the ``mag`` fallback path.
        The ``"bad"`` group is excluded for the ``eeg`` fallback path.
    good_channels : dict[str, numpy.ndarray]
        Reference channel layout mapping base channel names to their sensor
        position vectors of shape (12,), as expected by MNE (position +
        coil orientation). Used both to detect missing channels and to assign
        sensor locations before interpolation.
    channel_type : str, optional
        Type of channels to process. Must be either ``"mag"`` (default) or
        ``"eeg"``. Controls which groups are excluded when building the
        fallback channel order.

    Returns
    -------
    None
        Data is saved to disk as a pickle file. The saved object is a dict
        of the form {"MEG/EEG": [ndarray]}, where the array has shape
        (n_channels, n_times) with n_channels == len(good_channels).

    Raises
    ------
    ValueError
        If channel_type is not "mag" or "eeg".

    Notes
    -----
    The interpolation path calls :func:`interpolate_missing_channels`, which
    uses MNE's spherical spline interpolation centered at origin=(0, 0, 0.04). 
    The fallback path calls fill_missing_channels(), which duplicates existing 
    channels and does not preserve spatial topology — prefer the interpolation path when possible.
    """
    if channel_type not in ("mag", "eeg"):
        raise ValueError(f"Unsupported channel_type: {channel_type}")
    
    def get_base(name):
        return name.split()[0].split("-")[0].strip()
       
    current_basenames = {get_base(ch) for ch in raw.info["ch_names"]}
    can_interpolate = bool(current_basenames & set(good_channels.keys()))

    if channel_type == "mag":
        if can_interpolate:
            raw = interpolate_missing_channels(raw, good_channels)
            data = {"MEG/EEG": [raw.get_data()]}

        else:
            channels_order = [
                ch
                for group, chans in channel_groups.items()
                if group not in ("bad", "EEG")
                for ch in chans
            ]
            raw.reorder_channels(channels_order)
            meg_data = fill_missing_channels(raw, len(good_channels))
            data = {"MEG/EEG": [meg_data]}

    elif channel_type == "eeg":
        if can_interpolate:
            raw = interpolate_missing_channels(raw, good_channels)
            data = {"MEG/EEG": [raw.get_data()]}
        else:
            channels_order = [
                ch
                for group, chans in channel_groups.items()
                if group != "bad"
                for ch in chans
            ]
            raw.reorder_channels(channels_order)
            meg_data = fill_missing_channels(raw, len(good_channels))
            data = {"MEG/EEG": [meg_data]}

    save_obj(data, "data_raw", output_dir)


def create_windows(
    output_dir: str,
    window_size_s: int,
    stand: bool,
    sfreq: int,
    spike_spacing_from_border_s: float,
) -> int:
    """
    Crop windows from the pickle file and save them in a binary file.

    Parameters
    ----------
    output_dir : str
        Directory where processed data is stored.
    window_size_s : int
        Window size in seconds.
    stand : bool
        If True, standardize the data before saving.
    sfreq : int
        Sampling frequency in Hz.
    spike_spacing_from_border_s : float
        Minimum spacing from window borders in seconds,
        used to compute the stride between window centers.

    Returns
    -------
    int
        Total number of windows created.

    Raises
    ------
    RuntimeError
        If no valid windows could be created with the given parameters.
    """
    output_dir = Path(output_dir)

    window_size = int(window_size_s * sfreq)
    window_spacing = int((window_size_s - 2 * spike_spacing_from_border_s) * sfreq)

    data = load_obj("data_raw.pkl", output_dir)

    all_windows = []
    window_centers_all = []
    block_indices_all = []

    for block_idx, block_data in enumerate(data["MEG/EEG"]):
        window_centers = np.arange(window_size / 2, block_data.shape[1], window_spacing)

        block_windows = []
        for center in tqdm(window_centers, desc=f"Block {block_idx}"):
            if window_size / 2 <= center <= block_data.shape[1] - window_size / 2:
                low = int(center - window_size / 2)
                high = int(center + window_size / 2 + 0.1)  # Handle odd sizes
                block_windows.append(block_data[:, low:high])

                window_centers_all.append(center)
                block_indices_all.append(block_idx)

        if block_windows:
            all_windows.extend(block_windows)

    if not all_windows:
        raise RuntimeError("No valid windows were created. Check your parameters.")

    X_all = np.stack(all_windows).astype("float32")

    if stand:
        X_all = standardize(X_all)

    (output_dir / "data_raw_windows_bi").write_bytes(X_all.tobytes())

    # Save metadata
    save_obj(np.array(window_centers_all), "data_raw_timing", output_dir)
    save_obj(np.array(block_indices_all), "data_raw_blocks", output_dir)

    return len(X_all)

def generate_database(total_nb_windows: int) -> np.ndarray:
    """
    Generate a database of test window IDs.

    Args:
        total_nb_windows: Total number of windows.

    Returns:
        Array of shape (N, 1), for window index
    """
    X_test_ids = np.arange(total_nb_windows, dtype=int)
    return X_test_ids


def get_win_data_signal(f, win, dim):
    """
    Load and normalize a single window from a binary MEG data file.

    Parameters
    ----------
    f : file object
        Opened binary file containing MEG windows in float32 format.
    win : int
        Index of the window to retrieve.
    dim : tuple of int
        Shape of a single window as (n_channels, n_times).

    Returns
    -------
    numpy.ndarray
        Normalized window of shape (1, n_channels, n_times, 1).
    """
    f.seek(dim[0] * dim[1] * win * 4) 
    sample = np.fromfile(f, dtype="float32", count=dim[0] * dim[1])
    sample = sample.reshape(dim[1], dim[0])
    sample = np.swapaxes(sample, 0, 1)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)

    mean = np.mean(sample)
    std = np.std(sample)
    sample_norm = (sample - mean) / std

    return sample_norm
