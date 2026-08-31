# Dash & Plotly
import dash
from dash import html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Local Imports
from callbacks.utils import path_utils as dpu
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu
from layout.config_layout import ICON


def register_populate_memory_tab_contents():
    @callback(
        Output("subject-container-memory", "children"),
        Output("raw-info-container-memory", "children"),
        Output("event-stats-container-memory", "children"),
        Output("history-container-memory", "children"),
        Input("url", "pathname"),
        Input("subject-tabs-memory", "active_tab"),
        State("data-path-store", "data"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("annotation-store", "data"),
        State("history-store", "data"),
        prevent_initial_call=False,
    )
    def populate_memory_tab_contents(
        pathname,
        selected_tab,
        data_path,
        chunk_limits,
        freq_data,
        annotations_data,
        history_data,
    ):
        """
        Populate the Memory tab UI components based on stored session data.

        Parameters
        ----------
        pathname : str
            The current URL pathname (trigger).
        selected_tab : str
            The ID of the currently active tab in the memory view.
        data_path : str
            Path to the currently loaded dataset.
        chunk_limits : list
            List of time boundaries for data processing chunks.
        freq_data : dict
            Dictionary containing 'resample_freq', 'low_pass_freq', 
            'high_pass_freq', and 'notch_freq'.
        annotations_data : list of dict
            List of dictionaries representing MEG/EEG annotations/markers.
        history_data : list of dict
            Logged history of actions performed during the session.

        Returns
        -------
        tuple of dash.development.base_component.Component
            A 4-element tuple containing the HTML/Dash components for 
            (Subject, Raw Info, Event Stats, History).
        """
        if not data_path or not chunk_limits or not freq_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        subject_content = dash.no_update
        raw_info_content = dash.no_update
        event_stats_content = dash.no_update
        history_content = dash.no_update

        if selected_tab == "subject-tab-memory":
            subject_content = html.Div(
                [
                    html.Span(
                        [
                            html.I(
                                className="bi bi-person-rolodex",
                                style={"marginRight": "10px", "fontSize": "1.2em"},
                            ),
                            "Subject",
                        ],
                        className="card-title",
                    ),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem([html.Strong(data_path)]),
                        ],
                        style={"marginBottom": "15px"},
                    ),
                    html.Span(
                        [
                            html.I(
                                className="bi bi-sliders",
                                style={"marginRight": "10px", "fontSize": "1.2em"},
                            ),
                            "Frequency Parameters",
                        ],
                        className="card-title",
                    ),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem(
                                [
                                    html.Strong("Resample Frequency: "),
                                    html.Span(
                                        f"{freq_data.get('resample_freq', 'N/A')} Hz"
                                    ),
                                ]
                            ),
                            dbc.ListGroupItem(
                                [
                                    html.Strong("Low-pass Filter: "),
                                    html.Span(
                                        f"{freq_data.get('low_pass_freq', 'N/A')} Hz"
                                    ),
                                ]
                            ),
                            dbc.ListGroupItem(
                                [
                                    html.Strong("High-pass Filter: "),
                                    html.Span(
                                        f"{freq_data.get('high_pass_freq', 'N/A')} Hz"
                                    ),
                                ]
                            ),
                            dbc.ListGroupItem(
                                [
                                    html.Strong("Notch Filter: "),
                                    html.Span(
                                        f"{freq_data.get('notch_freq', 'N/A')} Hz"
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            )

        if selected_tab == "raw-info-tab-memory":
            raw_info_content = dpu.build_table_raw_info(data_path)

        if selected_tab == "events-tab-memory":
            event_stats_content = au.build_table_events_statistics(annotations_data)

        if selected_tab == "history-tab-memory":

            history_content = dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Span(
                                    [
                                        html.I(
                                            className=f"bi {ICON[category]}",
                                            style={
                                                "marginRight": "10px",
                                                "fontSize": "1.2em",
                                            },
                                        ),
                                        category.capitalize(),
                                    ],
                                    className="card-title",
                                ),
                                html.Hr(),
                                (
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(entry)
                                            for entry in hu.read_history_data_by_category(
                                                history_data, category
                                            )
                                        ]
                                    )
                                    if hu.read_history_data_by_category(
                                        history_data, category
                                    )
                                    else html.P(
                                        "No entries yet.", className="text-muted"
                                    )
                                ),
                            ],
                            style={"marginBottom": "10px"},
                        )
                        for category in ["annotations", "models", "ICA"]
                    ]
                )
            )

        return subject_content, raw_info_content, event_stats_content, history_content
