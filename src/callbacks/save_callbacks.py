import os
import dash
from dash import Input, Output, State, callback
from datetime import datetime
import mne
import pandas as pd
from collections import defaultdict
from callbacks.utils import markerfile_utils as mu
from callbacks.utils import annotation_utils as au
from callbacks.utils import path_utils as dpu


def register_display_annotations_to_save_checkboxes():
    @callback(
        Output("annotations-to-save-checkboxes", "options"),
        Output("annotations-to-save-checkboxes", "value"),
        Input("annotation-store", "data"),
        prevent_initial_call=False,
    )
    def _display_annotations_to_save_checkboxes(annotations_store):
        """Populate the checklist options and default value dynamically"""
        if not annotations_store:
            return dash.no_update, dash.no_update

        description_counts = au.get_annotation_descriptions(annotations_store)
        options = [
            {"label": f"{name} ({count})", "value": f"{name}"}
            for name, count in description_counts.items()
        ]
        value = [f"{name}" for name in description_counts.keys()]
        return options, value  # Set all annotations as default selected


def register_display_bad_channels_to_save_checkboxes():
    @callback(
        Output("bad-channels-to-save-checkboxes", "options"),
        Output("bad-channels-to-save-checkboxes", "value"),
        Input("channel-store", "data"),
        prevent_initial_call=False,
    )
    def _display_bad_channels_to_save_checkboxes(channel_store):
        """Populate the checklist options and default value dynamically"""
        if not channel_store:
            return dash.no_update, dash.no_update

        bad_channels = channel_store.get("bad", [])
        options = [
            {"label": f"{bad_chan}", "value": f"{bad_chan}"}
            for bad_chan in bad_channels
        ]
        value = [f"{bad_chan}" for bad_chan in bad_channels]
        return options, value


def register_save_modifications():
    @callback(
        Output("saving-mrk-status", "children"),
        Input("save-annotation-button", "n_clicks"),
        State("data-path-store", "data"),
        State("saving-format-radio", "value"),
        State("annotations-to-save-checkboxes", "value"),
        State("annotation-store", "data"),
        State("bad-channels-to-save-checkboxes", "value"),
        State("csv-filename-input", "value"),
    )
    def _save_modifications(
        n_clicks, data_path, format, annotations_to_save, annotations, bad_channels, csv_filename
    ):
        """
        Export modified annotations and bad channel metadata to disk.

        Parameters
        ----------
        n_clicks : int
            Trigger count from the save button.
        data_path : str
            Source path of the M/EEG data.
        save_format : str
            Desired export format ("original" or "fif").
        annotations_to_save : list of str
            List of annotation descriptions selected for export.
        annotations : list of dict
            The full annotation data from the store.
        bad_channels : list of str
            List of channel names marked as bad.
        csv_filename : str
            Name to be used when saving CSV file.

        Returns
        -------
        str
            Status message to be displayed in the UI.
        """
        if n_clicks > 0:
            if not data_path:
                return "⚠️ Error: No folder path selected."
            if not annotations:
                return "⚠️ Error: No annotations found."
            if annotations_to_save == []:
                return dash.no_update

            try:
                is_ds = data_path.endswith(".ds")
                is_fif = data_path.endswith(".fif")
                is_dir = os.path.isdir(data_path)

                if format == "csv":
                    # Filter annotations
                    filtered = [a for a in annotations if a["description"] in annotations_to_save]

                    if not filtered:
                        return "⚠️ Error: No matching annotations to save."

                    grouped_annotations = defaultdict(list)
                    for a in filtered:
                        grouped_annotations[a["description"]].append(a)

                    rows = []
                    for desc in sorted(grouped_annotations, key=lambda d: grouped_annotations[d][0]["onset"]):
                        group_sorted = sorted(grouped_annotations[desc], key=lambda x: x["onset"])
                        for a in group_sorted:
                            rows.append({
                                "onset (s)": a["onset"],
                                "duration (s)": a["duration"],
                                "description": a["description"],
                            })

                    df = pd.DataFrame(rows)

                    csv_name = (csv_filename or "annotations").strip() or "annotations"
                    if is_fif or is_ds or is_dir or os.path.isfile(data_path):
                        fname = os.path.join(os.path.dirname(data_path), f"{csv_name}.csv")
                    else:
                        return "⚠️ Error: Unsupported folder path format."

                    df.to_csv(fname, index=False, float_format="%.4f")

                elif format == "original":
                    if is_ds:
                        # Rename old marker file
                        old_mrk_name = (
                            f"OldMarkerFile_{datetime.now().strftime('%d.%m.%H.%M')}"
                        )
                        mu.modify_name_oldmarkerfile(data_path, old_mrk_name)
                        mu.save_mrk_file(
                            data_path, "MarkerFile", annotations_to_save, annotations
                        )
                        return "File saved successfully !"
                    elif is_dir:
                        return "This action is impossible. Please select FIF saving format."

                # Save as a new .fif for every supported input format. With
                # "original", the ".ds" and 4D folder cases have already
                # returned above, so this only runs for single-file inputs.
                if format == "fif" or format == "original":
                    raw = dpu.read_raw(data_path, verbose=False, preload=True)

                    # Filter annotations
                    filtered = [
                        (a["onset"], a["duration"], a["description"])
                        for a in annotations
                        if a["description"] in annotations_to_save
                    ]
                    if not filtered:
                        return "⚠️ Error: No matching annotations to save."

                    # Apply annotations and bad channels
                    onsets, durations, descriptions = zip(*filtered)
                    annot = mne.Annotations(
                        onset=onsets, duration=durations, description=descriptions
                    )
                    raw.set_annotations(annot)
                    raw.info["bads"] = bad_channels

                    # Determine save path: always a new .fif with a dated suffix
                    # next to the source, so the original file is never overwritten.
                    date_suffix = datetime.now().strftime("%d.%m.%H.%M")
                    if is_ds:
                        stem = os.path.basename(data_path)[: -len(".ds")]
                        fname = os.path.join(
                            os.path.dirname(data_path), f"{stem}_{date_suffix}.fif"
                        )
                    elif is_dir:
                        fname = os.path.join(
                            os.path.dirname(data_path),
                            f"{os.path.basename(data_path)}_{date_suffix}.fif",
                        )
                    elif os.path.isfile(data_path):
                        stem = os.path.splitext(data_path)[0]
                        fname = f"{stem}_{date_suffix}.fif"
                    else:
                        return "⚠️ Error: Unsupported folder path format."

                    raw.save(fname, overwrite=True)
                return "File saved successfully !"
            except Exception as e:
                return f"⚠️ Error saving the file: {str(e)}"

        return dash.no_update
    
def register_csv_name():
    @callback(
        Output("csv-filename-div", "style"),
        Input("saving-format-radio", "value"),
    )
    def toggle_csv_filename_div(format_value):
        if format_value == "csv":
            return {"marginBottom": "20px", "display": "block"}
        return {"marginBottom": "20px", "display": "none"}