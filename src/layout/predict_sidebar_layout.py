from dash import html, dcc
from layout.config_layout import (
    INPUT_STYLES,
    BUTTON_STYLES,
    LABEL_STYLES,
    FLEXDIRECTION,
    BOX_STYLES
)
import dash_bootstrap_components as dbc
from callbacks.utils import predict_utils as pu


def create_predict():
    layout = (
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select signal to use for prediction:",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.RadioItems(
                            id="signal-version-predict",
                            options=[{"label": "Filtered signal", "value": "__raw__"}],
                            value="__raw__",
                            inline=False,
                            style={"margin": "10px 0", "fontSize": "12px"},
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                    style=BOX_STYLES["classic"],
                ),
                html.Div(
                    [
                        html.Label(
                            "Preprocessing to use:",
                            style={**LABEL_STYLES["classic"]},
                        ),
                        dbc.RadioItems(
                            id="predict-preprocess-choice",
                            options=[
                                {"label": "Custom", "value": "custom"},
                                {"label": "Same as training", "value": "same_as_training"},
                            ],
                            value="custom",  # Default selection
                            inline=True,  # Display buttons in a row
                            style={"margin-left": "10px"},
                        ),
                        dbc.Tooltip(
                            """
                        Signal's preprocessing to use before applying prediction models:

                        - custom: Uses the signal as it has been processed during your current session (keeps your filters and changes in progress).
                        - Same as training: Ignore your current changes and reprocess the raw signal strictly following the specific configuration of the model.

                        Novice users are advised to select the “Same as training” option. Please note that if you select this option, any preprocessing performed earlier in your session will not be taken into account.
                            """,
                            target="predict-preprocess-choice",
                            placement="left",
                            class_name="custom-tooltip",
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Available Models:", style={**LABEL_STYLES["classic"]}
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=pu.get_model_options(),
                            placeholder="Select ...",
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Environment:", style={**LABEL_STYLES["classic"]}),
                        dbc.Input(
                            id="venv",
                            type="text",
                            value="",
                            disabled=True,
                            style={**INPUT_STYLES["small-number"]},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Sensitivity Analysis (smoothGrad):",
                            style={**LABEL_STYLES["classic"]},
                        ),
                        dbc.RadioItems(
                            id="sensitivity-analysis",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=False,  # Default selection
                            inline=True,  # Display buttons in a row
                            style={"margin-left": "10px"},
                        ),
                        dbc.Tooltip(
                            "SmoothGrad is an explainability method that computes the gradients of the loss with respect to the input across several noisy versions of the same input and then averages them.\n" 
                            "This process reduces noise and better highlights the regions of the time series that most influence the model's predictions.\n\n"
                            "Currently only available for CNN model.",
                            target="sensitivity-analysis",
                            placement="left",
                        ),
                    ],
                    id="sensitivity-analysis-div",
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label("SmoothGrad Threshold:", style={**LABEL_STYLES["classic"]}),
                        dbc.Input(
                            id="smoothgrad-threshold",
                            type="number",
                            value=0.5,
                            step=0.01,
                            min=0,
                            max=1,
                            style=INPUT_STYLES["small-number"],
                        ),
                        dbc.Tooltip(
                            "Indicates the minimum spike probability predicted by the model over a window to calculate the explainability method.\n"
                            "Used as a balance between computation time and fine interpretability of results.",
                            target="smoothgrad-threshold",
                            placement="left",
                        ),
                    ],
                    id="smoothgrad-threshold-div",
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Adjust onset (Global Field Power):",
                            style={**LABEL_STYLES["classic"]},
                        ),
                        dbc.RadioItems(
                            id="adjust-onset",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=True,
                            inline=True,
                            style={"margin-left": "10px"},
                        ),
                        dbc.Tooltip(
                            "Instead of setting the spike prediction to the middle of the window, this method adjusts the onset to the exact timestep where the Global Field Power (GFP) is at its maximum. This ensures more precise alignment with the probable spikes.",
                            target="adjust-onset",
                            placement="left",
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Run Prediction",
                            id="run-prediction-button",
                            color="warning",
                            outline=True,
                            size="sm",
                            n_clicks=0,
                            disabled=False,
                            style=BUTTON_STYLES["big"],
                        ),
                    ]
                ),
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        html.Div(
                            id="prediction-status",
                            style={"marginTop": "10px", "marginBottom": "20px"},
                        )
                    ],
                ),
                # Prediction Output
                html.Div(
                    id="store-display-div",
                    children=[
                        html.Div(
                            [
                                html.Label(
                                    "Threshold Refinement:",
                                    style={**LABEL_STYLES["classic"]},
                                ),
                                dbc.Input(
                                    id="adjusted-threshold",
                                    type="number",
                                    value=0.5,
                                    step=0.01,
                                    min=0,
                                    max=1,
                                    style=INPUT_STYLES["small-number"],
                                ),
                                dbc.Tooltip(
                                    "Controls how confident the model must be to detect a spike.",
                                    target="threshold",
                                    placement="left",
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            id="prediction-output",
                            children=[
                                dbc.Tabs(
                                    id="prediction-output-tabs",
                                    active_tab="prediction-output-summary",
                                    children=[
                                        dbc.Tab(
                                            label="Summary",
                                            tab_id="prediction-output-summary",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-summary-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "100%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                        dbc.Tab(
                                            label="Distribution",
                                            tab_id="prediction-output-distribution",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-distribution-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "100%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                        dbc.Tab(
                                            label="Table",
                                            tab_id="prediction-output-table",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-table-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "95%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                                "overflowX": "hidden",
                                                "maxWidth": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        **FLEXDIRECTION["row-tabs"],
                                        "width": "100%",
                                    },
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Detected Spike Name:",
                                    style={**LABEL_STYLES["classic"]},
                                ),
                                dbc.Input(
                                    id="model-spike-name",
                                    type="text",
                                    value="detected_spikes_name",
                                    style={**INPUT_STYLES["small-number"]},
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        dbc.Button(
                            "Store & Display",
                            id="store-display-button",
                            color="success",
                            outline=True,
                            disabled=False,
                            n_clicks=0,
                            style=BUTTON_STYLES["big"],
                        ),
                    ],
                    style={"display": "none"},
                ),
            ]
        ),
    )
    return layout
