# --- Dash ---
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# --- Local Utilities ---
from callbacks.utils import path_utils as dpu

# --- Layout Config ---
from layout.config_layout import INPUT_STYLES, FLEXDIRECTION

# --- Callbacks ---
from callbacks.storage_callbacks import register_populate_memory_tab_contents
from callbacks.data_path_callbacks import (
    register_handle_valid_data_path,
    register_update_dropdown,
    register_store_data_path_and_clear_data,
    register_populate_tab_contents,
)
from callbacks.preprocessing_callbacks import (
    register_handle_frequency_parameters,
    register_toggle_eeg_reference,
    register_preprocess_meg_data,
)
from callbacks.psd_callbacks import register_display_psd

dash.register_page(__name__, path="/")

layout = html.Div(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H5(
                                [
                                    html.I(
                                        className="bi bi-database-check",
                                        style={
                                            "marginRight": "10px",
                                            "fontSize": "1.2em",
                                        },
                                    ),
                                    "Database",
                                ]
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "20px",
                            "margin": "30px",
                        },
                    ),
                    html.Div(
                        id="subject-memory",
                        children=[
                            dbc.Tabs(
                                id="subject-tabs-memory",
                                active_tab="subject-tab-memory",
                                children=[
                                    dbc.Tab(
                                        label="General",
                                        tab_id="subject-tab-memory",
                                        children=[
                                            html.Div(
                                                id="subject-container-memory",
                                                children=[
                                                    html.Span(
                                                        "No subject in memory. Please choose one below."
                                                    )
                                                ],
                                                style={"padding": "10px"},
                                            )
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Raw Info",
                                        tab_id="raw-info-tab-memory",
                                        children=[
                                            html.Div(
                                                id="raw-info-container-memory",
                                                children=[
                                                    html.Label(
                                                        "No subject in memory. Please choose one below."
                                                    )
                                                ],
                                                style={"padding": "10px"},
                                            )
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Event Statistics",
                                        tab_id="events-tab-memory",
                                        children=[
                                            html.Div(
                                                id="event-stats-container-memory",
                                                children=[
                                                    html.Label(
                                                        "No subject in memory. Please choose one below."
                                                    )
                                                ],
                                                style={"padding": "10px"},
                                            )
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="History",
                                        tab_id="history-tab-memory",
                                        children=[
                                            html.Div(
                                                id="history-container-memory",
                                                children=[
                                                    html.Label(
                                                        "No history in memory. Please do further analysis."
                                                    )
                                                ],
                                                style={"padding": "10px"},
                                            )
                                        ],
                                    ),
                                ],
                                style={**FLEXDIRECTION["row-tabs"], "width": "100%"},
                            )
                        ],
                    ),
                ]
            ),
            className="mb-5",
            style={
                "width": "100%",
                "boxShadow": "0px 12px 24px rgba(0, 0, 0, 0.15)",  # smooth grey shadow
                "borderRadius": "30px",
            },
        ),
        html.Div(
            [
                html.H5(
                    [
                        html.I(
                            className="bi bi-person-plus-fill",
                            style={"marginRight": "10px", "fontSize": "1.2em"},
                        ),
                        "Select MEG/EEG Data",
                        dbc.Tooltip(
                            "This dropdown lists all MEG/EEG data files located in the /data folder. "
                            "To use a different folder, change the path in the /src/config.py file.",
                            target="data-path-dropdown",
                            autohide=True,
                            placement="top",
                            class_name="custom-tooltip",
                            trigger="focus",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "📂 Open Folder",
                                id="open-folder-button",
                                color="secondary",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="data-path-dropdown",
                                options=dpu.get_data_path_options(),
                                placeholder="Select ...",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Button(
                                [
                                    html.I(
                                        className="bi bi-1-circle-fill",
                                        style={"marginRight": "10px"},
                                    ),
                                    "Load",
                                ],
                                id="load-button",
                                color="success",
                                disabled=True,
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                [
                                    html.I(
                                        className="bi bi-2-circle-fill",
                                        style={"marginRight": "10px"},
                                    ),
                                    "Preprocess & Display",
                                ],
                                id="preprocess-display-button",
                                color="warning",
                                disabled=True,
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading",
                                type="default",
                                children=[html.Div(id="preprocess-status")],
                            ),
                            width="auto",
                        ),
                    ],
                    className="gy-2 align-items-center",
                ),
                html.Div(
                    id="data-path-warning",
                    className="text-danger",
                    style={"marginTop": "10px"},
                ),
                dcc.Location(id="url", refresh=True),
            ],
            style={"padding": "10px"},
        ),
        html.Div(
            id="frequency-container",
            children=[
                html.Div(
                    [
                        html.H6(
                            [
                                html.I(
                                    className="bi bi-sliders2",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Set Frequency Parameters for Signal Preprocessing",
                            ],
                            className="card-title",
                        ),
                        html.Hr(style={"width": "50%"}),
                        html.Div(
                            [
                                html.Label("Resampling Frequency (Hz): "),
                                dbc.Input(
                                    id="resample-freq",
                                    type="number",
                                    step=1,
                                    min=1,
                                    persistence=True,
                                    value=150,
                                    persistence_type="session",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("High-pass Frequency (Hz): "),
                                dbc.Input(
                                    id="high-pass-freq",
                                    type="number",
                                    step=0.1,
                                    min=0.1,
                                    persistence=True,
                                    value=0.5,
                                    persistence_type="session",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Low-pass Frequency (Hz): "),
                                dbc.Input(
                                    id="low-pass-freq",
                                    type="number",
                                    step=1,
                                    min=1,
                                    persistence=True,
                                    value=50,
                                    persistence_type="session",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Notch filter Frequency (Hz): "),
                                dbc.Input(
                                    id="notch-freq",
                                    type="number",
                                    step=1,
                                    min=0,
                                    persistence=True,
                                    persistence_type="session",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                            ],
                        ),
                        html.Div(
                            [
                                html.Label("EEG Reference: "),
                                dcc.Dropdown(
                                    id="eeg-reference",
                                    options=[
                                        {"label": "Average", "value": "average"},
                                        {
                                            "label": "Bipolar (double banana)",
                                            "value": "double_banana_bipolar",
                                        },
                                    ],
                                    value="average",
                                    clearable=False,
                                    disabled=True,
                                    style={"width": "200px"},
                                ),
                                dbc.Tooltip(
                                    "Re-reference EEG channels before filtering. "
                                    "Bipolar applies the standard temporal "
                                    "'double banana' montage; pairs whose electrodes "
                                    "are missing from the recording are skipped. "
                                    "Disabled for MEG recordings, which don't need "
                                    "re-referencing.",
                                    target="eeg-reference",
                                    placement="right",
                                    class_name="custom-tooltip",
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.H6(
                            [
                                html.I(
                                    className="bi bi-heart-pulse-fill",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Give Hint on Channel for ECG Peak Detection",
                            ],
                            className="card-title",
                        ),
                        html.Hr(style={"width": "50%"}),
                        html.Div(
                            [
                                html.Label("Channel Name:"),
                                dbc.Input(
                                    id="heartbeat-channel",
                                    type="text",
                                    placeholder="None",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                                dbc.Tooltip(
                                    """ch_name : None | str

            - The name of the channel to use for MNE ECG peak detection. 
            - If None (default), ECG channel is used if present. 
            - If None and no ECG channel is present, a synthetic ECG channel is created from the cross-channel average. 
            - This synthetic channel can only be created from MEG channels.

            The software uses mne.preprocessing.find_ecg_events() function which might be inacurate. Annotations should be considered as optional helper markers rather than validated ground truth.
            """,
                                    target="heartbeat-channel",
                                    placement="right",
                                    class_name="custom-tooltip",
                                ),
                            ]
                        ),
                        html.H6(
                            [
                                html.I(
                                    className="bi bi-eye-slash-fill",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Drop New Bad Channels",
                            ],
                            className="card-title",
                        ),
                        html.Hr(style={"width": "50%"}),
                        html.Div(
                            [
                                html.Label("Bad Channels:"),
                                dbc.Input(
                                    id="bad-channels",
                                    type="text",
                                    placeholder="None",
                                    style=INPUT_STYLES["number-in-box"],
                                ),
                                dbc.Tooltip(
                                    """ch_name : None | str | list of str

            - Enter one or more channel names separated by commas (e.g., "MEG 001, MEG 002").
            - If left empty (default), no channels are dropped.
            - Dropped channels will be excluded from topomaps, ICA, and model predictions, but may still be visible in plots.
            """,
                                    target="bad-channels",
                                    placement="right",
                                    class_name="custom-tooltip",
                                ),
                            ]
                        ),
                    ],
                    style={"width": "50%"},
                ),
                html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Tabs(
                                    id="tabs",
                                    active_tab="raw-info-tab",
                                    children=[
                                        dbc.Tab(
                                            label="Raw Info",
                                            tab_id="raw-info-tab",
                                            children=[
                                                html.Div(
                                                    id="raw-info-container",
                                                    style={"padding": "10px"},
                                                ),
                                            ],
                                        ),
                                        dbc.Tab(
                                            label="Event Statistics",
                                            tab_id="events-tab",
                                            children=[
                                                html.Div(
                                                    id="event-stats-container",
                                                    style={"padding": "10px"},
                                                ),
                                            ],
                                        ),
                                        dbc.Tab(
                                            label="Power Spectral Density",
                                            tab_id="psd-tab",
                                            children=[
                                                html.Div(
                                                    id="psd-container",
                                                    children=[
                                                        dbc.Button(
                                                            "Compute & Display",
                                                            id="compute-display-psd-button",
                                                            color="secondary",
                                                            n_clicks=0,
                                                            style={"marginTop": "15px"},
                                                        ),
                                                        dcc.Loading(
                                                            id="loading",
                                                            type="default",
                                                            children=[
                                                                html.Div(
                                                                    id="psd-status"
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                    style={"padding": "10px"},
                                                )
                                            ],
                                        ),
                                    ],
                                    style={**FLEXDIRECTION["row-tabs"]},
                                ),
                            ]
                        ),
                        className="mb-5",  # Adds margin below the card
                        style={
                            "width": "100%",
                            "boxShadow": "0px 12px 24px rgba(0, 0, 0, 0.15)",  # smooth grey shadow
                            "borderRadius": "30px",
                        },
                    ),
                    style={"width": "50%"},
                ),
            ],
            style={"display": "none"},
        ),
    ]
)

register_populate_memory_tab_contents()

register_update_dropdown()

register_handle_valid_data_path()

register_store_data_path_and_clear_data()

register_populate_tab_contents()

register_handle_frequency_parameters()

register_toggle_eeg_reference()

register_display_psd()

register_preprocess_meg_data()