# Dash & Plotly & other librairies
import dash
from dash import Input, Output, State, callback
from pathlib import Path

# Local Imports
from callbacks.utils import path_utils as dpu
from callbacks.utils import channel_utils as chu
from layout.config_layout import FLEXDIRECTION


def register_update_dropdown():
    @callback(
        Output("data-path-dropdown", "options"),
        Output("data-path-warning", "children"),  # Optional: warning display
        Input("open-folder-button", "n_clicks"),
        State("data-path-dropdown", "options"),
        prevent_initial_call=True,
    )
    def update_dropdown(n_clicks, data_path_list):
        """Update dropdown when a folder is selected via file explorer."""
        if n_clicks > 0:
            data_path = dpu.browse_folder()

            # Init options if None
            if not data_path:
                return dash.no_update, dash.no_update

            if data_path:
                if not dpu.test_valid_path(data_path):
                    return (
                        dash.no_update,
                        "Selected folder is not a valid MEG/EEG folder "
                        "(.ds, .fif, .edf, .bdf, .vhdr, .set, or 4D).",
                    )

            dropdown = dpu.get_data_path_options(Path(data_path))
            return dropdown, ""

        return dash.no_update, dash.no_update


def register_handle_valid_data_path():
    @callback(
        Output("load-button", "disabled"),
        Output("preprocess-display-button", "disabled", allow_duplicate=True),
        Output("frequency-container", "style"),
        Output("data-path-warning", "children", allow_duplicate=True),
        Output("pending-has-scalp-eeg", "data"),
        Input("data-path-dropdown", "value"),
        prevent_initial_call=True,
    )
    def handle_valid_data_path(data_path):
        """Validate folder path and show warning if invalid."""
        if data_path:
            if not dpu.test_valid_path(data_path):
                return (
                    True,
                    True,
                    {"display": "none"},
                    "Path must end with '.ds', '.fif', '.edf', '.bdf', '.vhdr', '.set', "
                    "or contain 3 files for 4D neuroimaging to be a valid raw MEG/EEG object.",
                    False,
                )

            try:
                raw = dpu.read_raw(data_path, preload=False, verbose=False)
                has_scalp_eeg = bool(chu.get_scalp_eeg_picks(raw.info))
                return (
                    False,
                    True,
                    {"display": "none"},
                    "",
                    has_scalp_eeg,
                )  # Valid: enable button and clear warning
            except Exception as e:
                return (
                    True,
                    True,
                    {"display": "none"},
                    f"Invalid MEG/EEG path: {str(e)}",
                    False,
                )

        # No path selected yet (including on initial page load): stay quiet,
        # the disabled Load button already makes the next step clear.
        return True, True, {"display": "none"}, "", False


def register_store_data_path_and_clear_data():
    @callback(
        Output("frequency-container", "style", allow_duplicate=True),
        Output("preprocess-display-button", "disabled"),
        Output("data-path-store", "data"),
        Output("chunk-limits-store", "clear_data"),
        Output("frequency-store", "clear_data"),
        Output("annotation-store", "clear_data"),
        Output("channel-store", "clear_data"),
        Output("display-channel-store", "clear_data"),
        Output("model-probabilities-store", "clear_data"),
        Output("sensitivity-analysis-store", "clear_data"),
        Output("raw-modality", "clear_data"),
        Output("ica-store", "clear_data"),
        Input("load-button", "n_clicks"),
        State("data-path-dropdown", "value"),
        prevent_initial_call=True,
    )
    def store_data_path_and_clear_data(n_clicks, data_path):
        """Clear all stores and display frequency section on load."""
        if not data_path:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        return (
            {"display": "flex", **FLEXDIRECTION["row-flex"]},
            False,
            data_path,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )


def register_populate_tab_contents():
    @callback(
        Output("raw-info-container", "children"),
        Output("event-stats-container", "children"),
        Input("tabs", "active_tab"),
        Input("data-path-store", "data"),
        prevent_initial_call=True,
    )
    def populate_tab_contents(selected_tab, data_path):
        """Populate tab content based on selected tab and stored folder path."""
        if not data_path or not selected_tab:
            return dash.no_update, dash.no_update

        raw_info_content = dash.no_update
        event_stats_content = dash.no_update

        if selected_tab == "raw-info-tab":
            raw_info_content = dpu.build_table_raw_info(data_path)

        if selected_tab == "events-tab":
            event_stats_content = dpu.build_table_events_statistics(data_path)

        return raw_info_content, event_stats_content
