import os
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog
import mne
from dash import html
import dash_bootstrap_components as dbc
import config

# Single-file MEG/EEG formats recognized by the app, mapped to their MNE reader.
# ".ds" (CTF) is a folder but is keyed by suffix like the others; BTi/4D is
# handled separately in read_raw() since it is a folder without a fixed suffix.
RAW_READERS = {
    ".ds": mne.io.read_raw_ctf,
    ".fif": mne.io.read_raw_fif,
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
}


def get_data_path_options(data_dir=Path(config.DATA_DIR)):
    all_data = get_valid_paths(str(data_dir))  # recursive search

    return (
        [{"label": d.name, "value": str(d.resolve())} for d in all_data]
        if all_data
        else [{"label": "No data available", "value": ""}]
    )


def browse_folder():
    root = tk.Tk()
    root.withdraw()  # Hide root window
    root.attributes("-topmost", True)  # Make sure dialog appears on top
    data_path = filedialog.askdirectory(title="Select a folder", parent=root)
    root.destroy()  # Destroy the root window after selection
    return data_path


def test_valid_path(path: str) -> bool:
    """
    Recursively check if a directory tree contains a valid MEG/EEG dataset:
    - Any file with a recognized MEG/EEG suffix (see RAW_READERS)
    - Any folder containing an 'hs_file'
    """
    p = Path(path)

    # Direct match
    if p.suffix in RAW_READERS:
        return True
    if p.is_dir() and (p / "hs_file").exists():
        return True

    # Recursive search
    for root, dirs, files in os.walk(p):
        # Check for recognized MEG/EEG files
        if any(Path(f).suffix in RAW_READERS for f in files):
            return True
        # Check if any subfolder contains hs_file
        for d in dirs:
            if (Path(root) / d / "hs_file").exists():
                return True

    return False


# def get_valid_paths(path):
#     parts = path.split(os.sep)
#     for part in reversed(parts):
#         if part.endswith((".ds", ".fif")):
#             return part
#     return None


def get_valid_paths(path) -> list:
    """
    Recursively collect all valid dataset paths under a directory:
    - Folders ending with .ds (do not recurse inside)
    - Files with a recognized MEG/EEG suffix (see RAW_READERS)
    - Folders containing 'hs_file'
    Returns a list of Path objects.
    """
    path = Path(path)
    valid_paths = []

    if path.is_file() and path.suffix in RAW_READERS:
        valid_paths.append(path)
        return valid_paths

    if path.is_dir():
        # Add current path if it's a .ds folder or contains hs_file
        if path.name.endswith(".ds") or (path / "hs_file").exists():
            valid_paths.append(path)
        else:
            # Recurse into subdirectories. Skip directories we cannot read
            # (permission-restricted or special folders, common on macOS).
            try:
                subs = list(path.iterdir())
            except (PermissionError, OSError):
                subs = []
            for sub in subs:
                valid_paths.extend(get_valid_paths(sub))

    return valid_paths


def get_bad_channels(raw, new_bad_channels):
    bad_channels = raw.info.get("bads", [])
    all_bad_channels = []
    if new_bad_channels:
        if isinstance(new_bad_channels, str):
            new_bad_channels_list = [
                ch.strip() for ch in new_bad_channels.split(",") if ch.strip()
            ]
        else:
            new_bad_channels_list = list(new_bad_channels)
        all_bad_channels = list(set(bad_channels + new_bad_channels_list))
    return all_bad_channels


def get_raw_modality(raw):
    n_eeg = len(mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False))
    n_meg = len(mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False))

    if n_eeg > 0 and n_meg > 0:
        return "mixed"
    elif n_eeg > 0:
        return "eeg"
    elif n_meg > 0:
        return "meg"
    else:
        return "unknown"


def read_raw(data_path, preload, verbose, bad_channels=None):
    data_path = Path(data_path)
    reader = RAW_READERS.get(data_path.suffix)

    if reader is not None:
        raw = reader(str(data_path), preload=preload, verbose=verbose)

    elif data_path.is_dir():
        # Assume BTi/4D format: folder must contain 3 specific files
        files = list(data_path.glob("*"))
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
        raise ValueError("Unrecognized file or folder type for MEG/EEG data.")

    if bad_channels:
        raw.drop_channels(bad_channels)

    return raw


def build_table_raw_info(data_path):

    raw = read_raw(data_path, preload=False, verbose=False)
    info = raw.info

    table = [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    [
                        html.Strong("File: "),
                        html.Span(
                            f"{raw.filenames[0] if raw.filenames else 'Unknown'}"
                        ),
                    ]
                ),
                dbc.ListGroupItem(
                    [html.Strong("Number of Channels: "), html.Span(f"{info['nchan']}")]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Sampling Frequency: "),
                        html.Span(f"{info['sfreq']} Hz"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("High-pass Filter: "),
                        html.Span(f"{info['highpass']} Hz"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Low-pass Filter: "),
                        html.Span(f"{info['lowpass']} Hz"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Duration: "),
                        html.Span(f"{round(raw.times[-1], 2)} seconds"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Channel Names: "),
                        html.Span(
                            "..."
                            + f"{', '.join(info['ch_names'][30:35]) + '...' if len(info['ch_names']) > 35 else ', '.join(info['ch_names'])}"
                        ),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Bad Channels: "),
                        html.Span(
                            f"{', '.join(info['bads']) if info['bads'] else 'None'}"
                        ),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Measurement Date: "),
                        html.Span(f"{str(info['meas_date'])}"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Experimenter: "),
                        html.Span(f"{info.get('experimenter', 'Unknown')}"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("SSP/ICA Components: "),
                        html.Span(f"{len(info.get('comps', []))} components"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Projections (SSP): "),
                        html.Span(f"{len(info.get('projs', []))} projections"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Digitized Points: "),
                        html.Span(
                            f"{len(info.get('dig', []))} points"
                            if info.get("dig")
                            else "None"
                        ),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("CTF Head Transform: "),
                        html.Span("Available" if info.get("ctf_head_t") else "None"),
                    ]
                ),
                dbc.ListGroupItem(
                    [
                        html.Strong("Device to Head Transform: "),
                        html.Span("Available" if info.get("dev_head_t") else "None"),
                    ]
                ),
            ]
        )
    ]

    return table


def build_table_events_statistics(data_path):

    raw = read_raw(data_path, preload=False, verbose=False)
    annotations = raw.annotations

    if len(annotations) == 0:
        return html.P("No annotations found in this recording.")

    description_counts = Counter(annotations.description)

    table_header = [html.Thead(html.Tr([html.Th("Event Name"), html.Th("Count")]))]
    table_body = [
        html.Tr([html.Td(desc), html.Td(count)])
        for desc, count in description_counts.items()
    ]
    annotation_table = dbc.Table(
        table_header + [html.Tbody(table_body)],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
    )

    stats_summary = [
        html.Span(
            [
                html.I(
                    className="bi bi-bar-chart-line",
                    style={"marginRight": "10px", "fontSize": "1.2em"},
                ),
                "Event Summary",
            ],
            className="card-title",
        ),
        dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Total annotations: {len(annotations)}"),
                dbc.ListGroupItem(f"Unique event types: {len(description_counts)}"),
                dbc.ListGroupItem(
                    f"First event starts at {annotations.onset[0]:.2f} s"
                ),
                dbc.ListGroupItem(
                    f"Last event ends at {(annotations.onset[-1] + annotations.duration[-1]):.2f} s"
                ),
            ],
            style={"marginBottom": "15px"},
        ),
        annotation_table,
    ]

    return stats_summary
