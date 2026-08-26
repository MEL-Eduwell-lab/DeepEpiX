import re
from collections import defaultdict
import mne
import config
from pathlib import Path
import json


# Standard bipolar "double banana" montage used in clinical EEG review,
# grouped by chain. Each entry is (derived_channel_name, anode, cathode);
# the derived channel's signal is anode - cathode.
TCP_AR_BIPOLAR_MONTAGE = {
    "Left Temporal": [
        ("Fp1-F7", "Fp1", "F7"),
        ("F7-T3", "F7", "T3"),
        ("T3-T5", "T3", "T5"),
        ("T5-O1", "T5", "O1"),
    ],
    "Right Temporal": [
        ("Fp2-F8", "Fp2", "F8"),
        ("F8-T4", "F8", "T4"),
        ("T4-T6", "T4", "T6"),
        ("T6-O2", "T6", "O2"),
    ],
    "Central": [
        ("Fz-Cz", "Fz", "Cz"),
        ("Cz-Pz", "Cz", "Pz"),
    ],
    "Left Parasagittal": [
        ("Fp1-F3", "Fp1", "F3"),
        ("F3-C3", "F3", "C3"),
        ("C3-P3", "C3", "P3"),
        ("P3-O1", "P3", "O1"),
    ],
    "Right Parasagittal": [
        ("Fp2-F4", "Fp2", "F4"),
        ("F4-C4", "F4", "C4"),
        ("C4-P4", "C4", "P4"),
        ("P4-O2", "P4", "O2"),
    ],
}


def apply_bipolar_montage(raw, montage=TCP_AR_BIPOLAR_MONTAGE):
    """
    Re-reference EEG data to a predefined bipolar montage (anode - cathode
    pairs), replacing the original channels with the derived ones.

    Pairs whose anode or cathode channel is not present in `raw` are
    skipped rather than raising, so a partial montage (e.g. missing mastoid
    electrodes) still yields the chains that can be computed.
    """
    available = set(raw.ch_names)
    pairs = [
        (name, anode, cathode)
        for chain_pairs in montage.values()
        for name, anode, cathode in chain_pairs
        if anode in available and cathode in available
    ]
    if not pairs:
        return raw

    names, anodes, cathodes = zip(*pairs)
    return mne.set_bipolar_reference(
        raw, anode=list(anodes), cathode=list(cathodes), ch_name=list(names)
    )


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
            grouped_channels["Non 10-20 EEG"] = other_names

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
            grouped_channels["Non 10-20 EEG"] = other_names

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


def get_post_reference_channel_groups(prep_raw, modality, reference, bad_channels=None):
    """
    Build the channel-selection groups for the raw-signal-viewer UI, matching
    the ACTUAL channels of `prep_raw` after preprocessing/re-referencing.

    For "none" or "average" reference, channel identity is unchanged, so
    this matches the scalp-space groups used for preprocessing/reordering.
    For the bipolar montage, derived channels (e.g. "Fp1-F7") replace scalp
    electrodes; since they don't match any standard_1020 name, they would
    otherwise be lumped into "Non 10-20 EEG" -- group them by montage chain
    (e.g. "Left Temporal") instead, matching TCP_AR_BIPOLAR_MONTAGE.
    """
    grouped_channels = get_grouped_channels_by_prefix(
        prep_raw, modality, bad_channels=bad_channels
    )
    if reference != "bipolar_tcp_ar":
        return grouped_channels

    remaining_other = grouped_channels.pop("Non 10-20 EEG", [])
    for chain_name, chain_pairs in TCP_AR_BIPOLAR_MONTAGE.items():
        chain_ch_names = {name for name, _, _ in chain_pairs}
        chain_channels = [ch for ch in remaining_other if ch in chain_ch_names]
        if chain_channels:
            grouped_channels[chain_name] = chain_channels
            remaining_other = [ch for ch in remaining_other if ch not in chain_ch_names]

    if remaining_other:
        grouped_channels["Non 10-20 EEG"] = remaining_other

    return grouped_channels
