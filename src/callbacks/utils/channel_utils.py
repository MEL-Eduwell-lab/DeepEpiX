import re
from collections import defaultdict
import mne
import config
from pathlib import Path
import json


def get_grouped_channels_meg(grouped_channels, ch_names):
    prefix_pattern = re.compile(r"^[A-Z]{3}$")

    if "MEG" in ch_names[int(len(ch_names) / 2)]:

        with open(Path(config.MONTAGES_DIR / "montage_MEG...123.json"), "r") as f:
            CHANNEL_GROUPS = json.load(f)

        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [ch for ch in channels if ch in ch_names]
            grouped_channels[region] = filtered_channels

    elif "M" in ch_names[int(len(ch_names) / 2)]:
        for ch_name in ch_names:
            prefix = ch_name.split("-")[0][:3]
            if prefix_pattern.match(prefix):
                grouped_channels[prefix].append(ch_name)

    elif "A" in ch_names[int(len(ch_names) / 2)]:

        with open(Path(config.MONTAGES_DIR / "montage_A1...json"), "r") as f:
            CHANNEL_GROUPS = json.load(f)

        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [ch for ch in channels if ch in ch_names]
            grouped_channels[region] = filtered_channels

    return grouped_channels


def get_scalp_eeg_picks(info, montage=None, exclude="bads"):
    """
    Return picks for "eeg"-typed channels that have a known position in a
    standard scalp montage (default: standard_1020).

    Acquisition systems and file readers (e.g. EDF, CTF) frequently label
    auxiliary physiological channels (ECG, EMG, respiration belt, markers...)
    as channel type "eeg" even though they are not part of the scalp
    electrode layout. Restricting to channels present in the montage avoids
    both crashing raw.set_montage() and contaminating EEG-only computations
    (topomaps, PSD, ICA) with these non-scalp signals.
    """
    if montage is None:
        montage = mne.channels.make_standard_montage("standard_1020")
    montage_ch_names = set(montage.ch_names)
    eeg_picks = mne.pick_types(
        info, meg=False, eeg=True, stim=False, eog=False, exclude=exclude
    )
    return [i for i in eeg_picks if info["ch_names"][i] in montage_ch_names]


def _split_eeg_channels_by_montage(info, ch_names):
    """
    Split EEG-typed channel names into "real" scalp electrodes (present in
    the standard 10-20 montage) and the rest (auxiliary channels such as
    ECG/EMG/markers that acquisition systems/readers often label as type
    "eeg" too). Order within each list follows `ch_names`.
    """
    scalp_names = {info["ch_names"][i] for i in get_scalp_eeg_picks(info)}
    montage_names = [ch for ch in ch_names if ch in scalp_names]
    other_names = [ch for ch in ch_names if ch not in scalp_names]
    return montage_names, other_names


def get_grouped_channels_by_prefix(raw, modality, bad_channels=None):
    """
    Load channels from raw data and group them by their 3-letter prefix.

    Returns:
        dict: Dictionary where keys are 3-letter prefixes and values are lists of channel names.
    """
    grouped_channels = defaultdict(list)

    if modality == "meg":
        # Get only MEG channels (both magnetometers and gradiometers)
        ch_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        grouped_channels = get_grouped_channels_meg(grouped_channels, ch_names)

    elif modality == "eeg":
        ch_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        montage_names, other_names = _split_eeg_channels_by_montage(raw.info, ch_names)
        if montage_names:
            grouped_channels["EEG"] = montage_names
        if other_names:
            grouped_channels["EEG (Other)"] = other_names

    elif modality == "mixed":
        # Get only MEG channels (both magnetometers and gradiometers)
        ch_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        grouped_channels = get_grouped_channels_meg(grouped_channels, ch_names)
        eeg_ch_picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False
        )
        eeg_ch_names = [raw.info["ch_names"][i] for i in eeg_ch_picks]
        montage_names, other_names = _split_eeg_channels_by_montage(raw.info, eeg_ch_names)
        if montage_names:
            grouped_channels["EEG"] = montage_names
        if other_names:
            grouped_channels["EEG (Other)"] = other_names

    elif modality == "unkown":
        raise Exception(
            "Cannot determine the modality of the raw data: no EEG or MEG channels found."
        )

    if bad_channels:
        if isinstance(bad_channels, str):
            bad_channels_list = [
                ch.strip() for ch in bad_channels.split(",") if ch.strip()
            ]
        else:
            bad_channels_list = list(bad_channels)
        grouped_channels["bad"] = bad_channels_list

    return grouped_channels
