import pandas as pd

import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from layout.config_layout import INPUT_STYLES, FLEXDIRECTION
from callbacks.utils import annotation_utils as au
from callbacks.utils import performance_utils as pu
import callbacks.utils.model_utils as mu

dash.register_page(__name__, name="Model performance", path="/model/performance")

layout = html.Div(
    [
        dcc.Location(id="url", refresh=True),
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    "Your Models",
                                    html.I(
                                        className="bi bi-info-circle-fill",
                                        id="models-help-icon",
                                        style={
                                            "fontSize": "0.8em",
                                            "cursor": "pointer",
                                            "verticalAlign": "middle",
                                            "marginLeft": "15px",
                                        },
                                    ),
                                ],
                                style={"margin": "0px"},
                            ),
                            dbc.Tooltip(
                                "Here you can see your pretrained models and, if applicable, compute performance using already stored predictions.",
                                target="models-help-icon",
                                placement="right",
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
                    mu.render_pretrained_models_table(),
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
            id="model-playground",
            children=[
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Button(
                                    html.I(
                                        className="bi bi-arrow-clockwise",
                                        style={"fontSize": "0.9rem"},
                                    ),
                                    id="refresh-performances",
                                    className="p-0 border-0",
                                    color="link",
                                    n_clicks=0,
                                    style={"marginRight": "12px"},
                                ),
                                dbc.Progress(
                                    [
                                        dbc.Progress(
                                            value=34,
                                            color="primary",
                                            bar=True,
                                            striped=True,
                                            label="1. Model Prediction",
                                            id="step-1-bar",
                                            className="progress-step",
                                        ),
                                        dbc.Progress(
                                            value=0,
                                            color="info",
                                            bar=True,
                                            striped=True,
                                            label="2. Ground Truth",
                                            id="step-2-bar",
                                            className="progress-step",
                                        ),
                                        dbc.Progress(
                                            value=0,
                                            color="warning",
                                            bar=True,
                                            striped=True,
                                            label="3. Settings",
                                            id="step-3-bar",
                                            className="progress-step",
                                        ),
                                    ],
                                    style={
                                        "height": "30px",
                                        "flexGrow": 1,
                                        "borderRadius": "8px",
                                    },
                                    className="d-flex w-100",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "marginBottom": "20px",
                            },
                        ),
                        # Conditionally displayed panels
                        html.Div(
                            id="model-prediction",
                            children=[
                                dbc.RadioItems(
                                    id="model-prediction-radio",
                                    inline=False,
                                    style={"marginLeft": "60px", "fontSize": "14px"},
                                )
                            ],
                        ),
                        html.Div(
                            id="ground-truth",
                            children=[
                                dbc.Checklist(
                                    id="ground-truth-checkboxes",
                                    inline=False,
                                    style={"margin": "10px 0", "fontSize": "14px"},
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-chevron-double-right"),
                                    id="ok-ground-truth-button",  # Unique ID for each button
                                    color="warning",
                                    outline=True,
                                    size="sm",
                                    n_clicks=0,
                                    disabled=False,
                                    style={"border": "none", "margin": "20px"},
                                ),
                            ],
                            style={"display": "none"},
                        ),
                        html.Div(
                            id="settings",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Tolerance (ms):",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "14px",
                                                        "marginRight": "10px",
                                                        "width": "150px",
                                                    },
                                                ),
                                                dbc.Input(
                                                    id="performance-tolerance",
                                                    type="number",
                                                    value=200,
                                                    step=10,
                                                    min=0,
                                                    max=1000,
                                                    style={
                                                        **INPUT_STYLES["number"],
                                                        "width": "200px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Threshold:",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "14px",
                                                        "marginRight": "10px",
                                                        "width": "150px",
                                                    },
                                                ),
                                                dbc.Input(
                                                    id="performance-threshold",
                                                    type="number",
                                                    value=0.5,
                                                    step=0.01,
                                                    min=0,
                                                    max=1,
                                                    style={
                                                        **INPUT_STYLES["number"],
                                                        "width": "200px",
                                                        "marginBottom": "30px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                            },
                                        ),
                                    ],
                                    style={"display": "block"},
                                ),
                            ],
                            style={"display": "none"},
                        ),
                    ],
                    style={"width": "40%"},
                ),
                html.Div(
                    id="performance-results-div",
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="performance-results-title-panel",
                                        children=[html.Small("Performance Results")],
                                        style={"marginBottom": "10px"},
                                    ),
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                label="Performance Metrics",
                                                tab_id="metrics-tab",
                                                children=[
                                                    html.Div(
                                                        id="performance-metrics-table"
                                                    )
                                                ],
                                            ),
                                            dbc.Tab(
                                                label="Confusion Matrix",
                                                tab_id="confusion-tab",
                                                children=[
                                                    html.Div(
                                                        id="confusion-matrix-table"
                                                    )
                                                ],
                                            ),
                                            dbc.Tab(
                                                label="Distance Statistics",
                                                tab_id="stats-tab",
                                                children=[
                                                    html.I(
                                                        className="bi bi-info-circle-fill",
                                                        id="distance-help-icon",
                                                        style={
                                                            "fontSize": "0.8em",
                                                            "cursor": "pointer",
                                                            "verticalAlign": "middle",
                                                            "marginLeft": "15px",
                                                        },
                                                    ),
                                                    dbc.Tooltip(
                                                        "This panel shows distances between model predictions and ground truth. "
                                                        "• TP: Time between matched prediction and true event. Smaller values mean more accurate predictions. "
                                                        "• FP: Time to nearest true event (unmatched prediction). Useful for spotting near-misses or totally spurious predictions. "
                                                        "• FN: Time to nearest prediction (missed true event). A small distance means the model nearly detected the event.",
                                                        target="distance-help-icon",
                                                        placement="top",
                                                        style={"maxWidth": "600px"},
                                                    ),
                                                    html.Div(
                                                        id="distance-statistics-table"
                                                    ),
                                                ],
                                            ),
                                            dbc.Tab(
                                                label="F1 vs Tolerance",
                                                tab_id="f1-tolerance",
                                                children=[
                                                    dcc.Graph(
                                                        id="f1-vs-tolerance-graph"
                                                    )
                                                ],
                                            ),
                                            dbc.Tab(
                                                label="F1 vs Threshold",
                                                tab_id="f1-threshold",
                                                children=[
                                                    dcc.Graph(
                                                        id="f1-vs-threshold-graph"
                                                    )
                                                ],
                                            ),
                                        ],
                                        id="performance-tabs",
                                        active_tab="metrics-tab",
                                        style={"marginTop": "10px"},
                                    ),
                                ]
                            ),
                            id="performance-panel",
                            style={
                                "marginBottom": "30px",
                                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
                                "borderRadius": "1rem",
                            },
                        )
                    ],
                    style={"width": "60%", "marginLeft": "60px"},
                ),
            ],
            style={**FLEXDIRECTION["row-flex"], "display": "flex"},
        ),
    ]
)


@callback(
    Output("model-prediction", "style"),
    Output("ground-truth", "style"),
    Output("step-2-bar", "value"),
    Input("model-prediction-radio", "value"),
    prevent_initial_call=True,
)
def progress_bar_step_1(mp_val):
    if mp_val is not None:
        return (
            {"display": "none"},
            {"display": "flex", "justifyContent": "center", "marginLeft": "60px"},
            33,
        )
    raise PreventUpdate


@callback(
    Output("ground-truth", "style", allow_duplicate=True),
    Output("settings", "style"),
    Output("step-3-bar", "value"),
    Input("ok-ground-truth-button", "n_clicks"),
    State("ground-truth-checkboxes", "value"),
    prevent_initial_call=True,
)
def progress_bar_step_2(n_clicks, gt_val):
    if n_clicks and gt_val is not None:
        return (
            {"display": "none"},
            {"display": "flex", "justifyContent": "flex-end", "marginLeft": "60px"},
            33,
        )
    raise PreventUpdate


@callback(
    Output("url", "href", allow_duplicate=True),
    Input("refresh-performances", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_page(n_clicks):
    return (
        "/model/performance"  # or ctx.triggered_id or url.pathname to simulate refresh
    )


@callback(
    Output("model-prediction-radio", "options"),
    Output("ground-truth-checkboxes", "options"),
    Input("annotation-store", "data"),
    prevent_initial_call=False,
)
def display_model_names_checklist(annotations_store):
    description_counts = au.get_annotation_descriptions(annotations_store)
    options = [
        {"label": f"{name} ({count})", "value": f"{name}"}
        for name, count in description_counts.items()
    ]
    return options, options  # dash.no_update  # Set all annotations as default selected


@callback(
    Output("performance-results-title-panel", "children"),
    Output("confusion-matrix-table", "children"),
    Output("performance-metrics-table", "children"),
    Output("distance-statistics-table", "children"),
    Output("f1-vs-tolerance-graph", "figure"),
    Output("f1-vs-threshold-graph", "figure"),
    Input("performance-tolerance", "value"),
    Input("performance-threshold", "value"),
    Input("ground-truth-checkboxes", "value"),
    State("model-prediction-radio", "value"),
    State("annotation-store", "data"),
    State("model-csv-store", "data"),
    prevent_initial_call=True,
)
def compute_performance(
    tolerance, threshold, ground_truth, model_prediction, annotations, model_csv_store
):
    if not model_prediction or not ground_truth or tolerance is None:
        error = "⚠️ Error: Missing inputs. Please select model predictions, ground truth, tolerance and threshold."
        return dash.no_update, error, error, error, None, None

    # Convert annotations to DataFrame
    annotations_df = pd.DataFrame(annotations).set_index("onset")

    # Get annotation onsets
    model_onsets = au.get_annotations(model_prediction, annotations_df)
    gt_onsets = au.get_annotations(ground_truth, annotations_df)

    # Compute matches
    delta = tolerance / 1000
    tp, fp, fn, tp_dists, fp_dists, fn_dists = pu.compute_matches(
        model_onsets, gt_onsets, delta
    )

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Confusion matrix table
    conf_matrix_data = [
        {"": "Actual Negative", "Predicted Negative": "–", "Predicted Positive": fp},
        {"": "Actual Positive", "Predicted Negative": fn, "Predicted Positive": tp},
    ]
    conf_matrix = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(""),
                        html.Th("Predicted Negative"),
                        html.Th("Predicted Positive"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row[""]),
                            html.Td(row["Predicted Negative"]),
                            html.Td(row["Predicted Positive"]),
                        ]
                    )
                    for row in conf_matrix_data
                ]
            ),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    # Performance metrics
    perf_metrics = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody(
                [
                    html.Tr([html.Td("Precision"), html.Td(f"{precision:.3f}")]),
                    html.Tr([html.Td("Recall"), html.Td(f"{recall:.3f}")]),
                    html.Tr([html.Td("F1 Score"), html.Td(f"{f1:.3f}")]),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    # Distance stats
    stats_data = [
        {"Type": "TP", **pu.get_distance_stats(tp_dists)},
        {"Type": "FP", **pu.get_distance_stats(fp_dists)},
        {"Type": "FN", **pu.get_distance_stats(fn_dists)},
    ]
    dist_stats = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Type"),
                        html.Th("Mean"),
                        html.Th("Std"),
                        html.Th("Min"),
                        html.Th("Max"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row["Type"]),
                            html.Td(row["mean"]),
                            html.Td(row["std"]),
                            html.Td(row["min"]),
                            html.Td(row["max"]),
                        ]
                    )
                    for row in stats_data
                ]
            ),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    # Title panel
    title = html.Small(
        [
            html.I(className="bi bi-flag-fill me-2 text-primary"),
            "Ground Truth: ",
            html.Span([f"{gt}/" for gt in ground_truth], className="fw-bold text-dark"),
            html.I(className="bi bi-stars mx-3 text-success"),
            "Prediction: ",
            html.Span(model_prediction, className="fw-bold text-dark"),
            html.I(className="bi bi-sliders2-vertical mx-3 text-warning"),
            "Tolerance: ",
            html.Span(f"{tolerance} ms", className="fw-bold text-dark"),
            html.Span(" | ", className="mx-2"),
            "Threshold: ",
            html.Span(f"{threshold}", className="fw-bold text-dark"),
        ],
        className="text-muted",
    )

    # F1 vs. Tolerance
    tolerances = list(range(10, 300, 10))  # ms
    f1_by_tol = []
    for tol in tolerances:
        delta = tol / 1000
        tp, fp, fn, *_ = pu.compute_matches(model_onsets, gt_onsets, delta)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_by_tol.append(f1)

    f1_tol_fig = {
        "data": [
            {
                "x": tolerances,
                "y": f1_by_tol,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": "#007BFF"},
                "name": "F1 Score",
            }
        ],
        "layout": {
            "title": "F1 Score vs. Tolerance",
            "xaxis": {"title": "Tolerance (ms)"},
            "yaxis": {"title": "F1 Score", "range": [0, 1]},
            "height": 300,
        },
    }

    # F1 vs. Threshold
    delta = tolerance / 1000

    csv_path = model_csv_store.get(model_prediction) if model_csv_store else None

    if csv_path and "uiednet" in csv_path.lower():
        # UIEDNET "probas" are segmentation-peak prominences, not calibrated
        # probabilities, so a threshold sweep is not meaningful.
        f1_thresh_fig = {
            "data": [],
            "layout": {
                "title": "F1 Score vs. Threshold",
                "xaxis": {"title": "Threshold"},
                "yaxis": {"title": "F1 Score", "range": [0, 1]},
                "height": 300,
                "annotations": [
                    {
                        "text": "Not available for UIEDNET predictions",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                    }
                ],
            },
        }
    else:
        thresholds = [round(x * 0.05, 2) for x in range(1, 21)]
        f1_by_thresh = []

        for th in thresholds:
            if csv_path:
                df = pd.read_csv(csv_path)
                model_onsets_th = df[df["probas"] > th]["onset"].values
            else:
                model_onsets_th = model_onsets

            tp, fp, fn, *_ = pu.compute_matches(model_onsets_th, gt_onsets, delta)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_by_thresh.append(f1)

        f1_thresh_fig = {
            "data": [
                {
                    "x": thresholds,
                    "y": f1_by_thresh,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "line": {"color": "#28A745"},
                    "name": "F1 Score",
                }
            ],
            "layout": {
                "title": "F1 Score vs. Threshold",
                "xaxis": {"title": "Threshold"},
                "yaxis": {"title": "F1 Score", "range": [0, 1]},
                "height": 300,
            },
        }

    # Return all outputs (title, tables, graphs)
    return title, conf_matrix, perf_metrics, dist_stats, f1_tol_fig, f1_thresh_fig
