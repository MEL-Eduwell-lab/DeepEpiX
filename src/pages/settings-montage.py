import io
import random
import base64
import itertools

import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc

import mne
import matplotlib.pyplot as plt
from layout.config_layout import REGION_COLOR_PALETTE, BOX_STYLES, FLEXDIRECTION
from callbacks.utils import path_utils as dpu
from callbacks.utils import channel_utils as chu

dash.register_page(__name__, name="Settings", path="/settings/montage")

layout = html.Div(
    [
        dcc.Location(id="url", refresh=True),
        html.Div(
            [
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                id="your-montage-container",
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "justifyContent": "center",
                                            "alignItems": "center",
                                            "gap": "20px",
                                            "margin": "30px",
                                        },
                                        children=[
                                            html.H4(
                                                [
                                                    "Your Montage ",
                                                    html.I(
                                                        className="bi bi-info-circle-fill",
                                                        id="montage-help-icon",
                                                        style={
                                                            "fontSize": "0.8em",
                                                            "cursor": "pointer",
                                                            "verticalAlign": "middle",
                                                        },
                                                    ),
                                                ],
                                                style={"margin": 0},
                                            ),
                                            dbc.Button(
                                                html.I(className="bi bi-trash-fill"),
                                                id="delete-all-button",
                                                color="danger",
                                                style={
                                                    "marginLeft": "5px",
                                                    "fontSize": "1.2em",
                                                },
                                                title="Delete all",
                                            ),
                                        ],
                                    ),
                                    dbc.Tooltip(
                                        "Here you can see which montage you have already created.",
                                        target="montage-help-icon",
                                        placement="right",
                                    ),
                                    html.Div(
                                        id="saved-montages-table",
                                        style={
                                            "overflowX": "auto",
                                            "width": "60%",
                                            "margin": "20px auto",
                                            "padding": "10px",
                                        },
                                    ),
                                ],
                            )
                        ]
                    ),
                    className="mb-5",
                    style={
                        "width": "100%",
                        "boxShadow": "0px 12px 24px rgba(0, 0, 0, 0.15)",  # smooth grey shadow
                        "borderRadius": "30px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    id="montage-name-container",
                    children=[
                        html.H4(
                            [
                                html.I(
                                    className="bi bi-1-circle-fill",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Create new montage",
                            ],
                            style={"margin-bottom": "15px"},
                        ),
                        dbc.Input(
                            id="new-montage-name",
                            type="text",
                            placeholder="Montage name...",
                            style={"marginBottom": "10px", "width": "100%"},
                        ),
                        dbc.Button(
                            "Create",
                            id="create-button",
                            color="success",
                            disabled=True,
                            n_clicks=0,
                            style={"marginTop": "5px", "width": "100%"},
                        ),
                    ],
                    style={**BOX_STYLES["classic"], "width": "20%"},
                ),
                html.Div(
                    id="montage-selection-container",
                    children=[
                        html.H4(
                            [
                                html.I(
                                    className="bi bi-2-circle-fill",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Select channels",
                            ],
                            style={"margin-bottom": "15px"},
                        ),
                        dcc.Dropdown(
                            id="selection-method-dropdown",
                            options=[
                                {"label": "Checklist", "value": "checklist"},
                                {"label": "Random Pick", "value": "random"},
                            ],
                            value="checklist",
                            clearable=False,
                            style={"width": "100%", "marginBottom": "20px"},
                        ),
                        html.Div(
                            id="checklist-method-container", style={"display": "none"}
                        ),
                        html.Div(
                            id="random-pick-method-container",
                            children=[
                                html.Div(
                                    [
                                        html.H5("Apply % to each group:"),
                                        dbc.Input(
                                            id="random-pick-count-%",
                                            type="number",
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=0,
                                            placeholder="e.g. 10 for 10%",
                                            style={"width": "80px", "fontSize": "14px"},
                                        ),
                                    ],
                                    style={"marginBottom": "20px", "padding": "10px"},
                                ),
                                html.Hr(),
                                html.Div(
                                    id="random-pick-method-regions",
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "10px",
                                        "padding": "10px",
                                    },
                                ),
                            ],
                            style={"display": "none"},
                        ),
                    ],
                    style={**BOX_STYLES["classic"], "width": "60%", "display": "none"},
                ),
                html.Div(
                    id="channels-layout-container",
                    children=[
                        html.H4(
                            [
                                html.I(
                                    className="bi bi-3-circle-fill",
                                    style={"marginRight": "10px", "fontSize": "1.2em"},
                                ),
                                "Save montage",
                            ],
                            style={"margin-bottom": "15px"},
                        ),
                        html.Div(
                            id="channels-layout-display",
                            children=html.Img(
                                id="channels-layout-img", style={"width": "100%"}
                            ),
                        ),
                        dbc.Button(
                            "Save",
                            id="save-button-ica",
                            color="success",
                            disabled=False,
                            n_clicks=0,
                            style={"marginTop": "5px", "width": "100%"},
                        ),
                    ],
                    style={**BOX_STYLES["classic"], "width": "20%", "display": "none"},
                ),
            ],
            style={**FLEXDIRECTION["row-flex"], "display": "flex"},
        ),
    ],
)


@callback(
    Output("saved-montages-table", "children"),
    Input("montage-store", "data"),
    State("channel-store", "data"),
    prevent_initial_call=False,
)
def update_montage_table(data, folder_loaded):
    if not data and not folder_loaded:
        return html.Div(
            "No montages saved yet. No subject in memory. Please load one on Home page.",
            style={"textAlign": "center", "color": "#888"},
        )
    if not data:
        return html.Div(
            "No montages saved yet.", style={"textAlign": "center", "color": "#888"}
        )

    table_header = html.Thead(
        html.Tr(
            [
                html.Th("Montage Name"),
                html.Th("Channels"),
                html.Th("Delete"),
            ]
        )
    )

    table_body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(montage_name),
                    html.Td(
                        ", ".join(channels),
                        style={
                            "whiteSpace": "nowrap",  # Prevent wrapping
                            "overflowX": "auto",  # Enable horizontal scrolling
                            "maxWidth": "200px",  # Adjust the max width
                            "padding": "10px",  # Padding for readability
                        },
                    ),
                    html.Td(
                        dbc.Button(
                            html.I(className="bi bi-trash"),
                            id={"type": "delete-montage-btn", "index": montage_name},
                            color="danger",
                            size="sm",
                        )
                    ),
                ]
            )
            for montage_name, channels in data.items()
        ]
    )

    return dbc.Table(
        children=[table_header, table_body],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        size="sm",
        class_name="table-light text-center",
    )


@callback(
    Output("montage-store", "data", allow_duplicate=True),
    Input({"type": "delete-montage-btn", "index": ALL}, "n_clicks"),
    State("montage-store", "data"),
    prevent_initial_call=True,
)
def delete_montage(n_clicks_list, montage_data):
    triggered = dash.ctx.triggered_id

    if (
        not triggered
        or not n_clicks_list
        or all(n is None or n <= 0 for n in n_clicks_list)
    ):
        return dash.no_update

    montage_to_delete = triggered["index"]

    if montage_to_delete in montage_data:
        del montage_data[montage_to_delete]

    return montage_data


@callback(
    Output("montage-store", "data", allow_duplicate=True),
    Input("delete-all-button", "n_clicks"),
    prevent_initial_call=True,
)
def delete_all_montage(n_clicks):
    if n_clicks and n_clicks > 0:
        return {}
    return dash.no_update


@callback(
    Output("create-button", "disabled"),
    Input("new-montage-name", "value"),
    State("montage-store", "data"),
    State("channel-store", "data"),
    prevent_initial_call=True,
)
def handle_valid_montage_name(name, montage_store_data, channel_data):
    """Validate montage name"""
    if name:
        if name not in montage_store_data:
            if channel_data:
                return False
    return True


@callback(
    Output("montage-selection-container", "style"),
    Output("channels-layout-container", "style"),
    Output("new-montage-name", "disabled"),
    Input("create-button", "n_clicks"),
    State("new-montage-name", "value"),
    State("montage-store", "data"),
    prevent_initial_call=True,
)
def handle_create_button(n_clicks, new_montage_name, montage_store_data):
    if n_clicks > 0:
        return (
            {**BOX_STYLES["classic"], "width": "60%"},
            {**BOX_STYLES["classic"], "width": "20%"},
            True,
        )


@callback(
    Output("checklist-method-container", "style"),
    Output("random-pick-method-container", "style"),
    Input("selection-method-dropdown", "value"),
    prevent_initial_call=False,
)
def update_selection_method_ui(method):

    if method == "checklist":
        return {
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "space-between",
        }, {"display": "none"}

    elif method == "random":
        return {"display": "none"}, {
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "space-between",
        }

    return dash.no_update, dash.no_update  # fallback


@callback(
    Output("checklist-method-container", "children"),
    Input("checklist-method-container", "style"),
    State("channel-store", "data"),
    prevent_initial_call=True,
)
def update_checklist_method_container(style, channel_data):
    if style == {"display": "none"}:
        return dash.no_update

    if not channel_data:
        return []

    rainbow_colors = itertools.cycle(REGION_COLOR_PALETTE)

    children = []
    for group, channels in channel_data.items():
        group_color = next(rainbow_colors)
        group_div = html.Div(
            [
                html.H5(
                    group,
                    style={
                        "fontSize": "14px",
                        "fontWeight": "bold",
                        "marginBottom": "5px",
                        "color": group_color,
                    },
                ),
                dbc.Button(
                    "Check All",
                    id={"type": "check-all-btn", "group": group},
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "10px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dbc.Button(
                    "Clear All",
                    id={"type": "clear-all-btn", "group": group},
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "10px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Checklist(
                    id={"type": "montage-checklist", "group": group},
                    options=[{"label": ch, "value": ch} for ch in channels],
                    value=channels if group == "EEG" else [],
                    style={"marginTop": "10px", "fontSize": "10px"},
                ),
            ],
            style={"flex": "1 0 120px", "padding": "5px"},
        )

        children.append(group_div)

    return children


@callback(
    Output("random-pick-method-regions", "children"),
    Input("random-pick-method-container", "style"),
    State("channel-store", "data"),
    prevent_initial_call=True,
)
def update_random_pick_inputs(style, channel_groups):
    if style == {"display": "none"}:
        return dash.no_update

    if not channel_groups:
        return []

    rainbow_colors = itertools.cycle(REGION_COLOR_PALETTE)

    layout = [
        html.Div(
            children=[
                html.H5(
                    group,
                    style={
                        "fontSize": "14px",
                        "fontWeight": "bold",
                        "marginBottom": "5px",
                        "color": next(rainbow_colors),
                    },
                ),
                dbc.Input(
                    id={"type": "random-pick-count", "group": group},
                    type="number",
                    min=0,
                    step=1,
                    value=0,
                    max=len(channels),
                    style={"width": "50%", "fontSize": "14px", "padding": "5px"},
                ),
            ],
            style={"flex": "1 0 170px", "padding": "8px"},
        )
        for group, channels in channel_groups.items()
    ]

    return layout


@callback(
    Output({"type": "random-pick-count", "group": MATCH}, "value"),
    Input("random-pick-count-%", "value"),
    State("channel-store", "data"),
    State({"type": "random-pick-count", "group": MATCH}, "id"),
    prevent_initial_call=True,
)
def apply_percentage_to_group(global_percent, channel_store, triggered_id):
    if not global_percent or global_percent <= 0:
        return dash.no_update

    group = triggered_id["group"]
    channels = channel_store.get(group, [])
    total = len(channels)

    computed_value = round((global_percent / 100.0) * total)
    return computed_value


@callback(
    Output("montage-store", "data", allow_duplicate=True),
    Output("url", "href"),
    Input("save-button-ica", "n_clicks"),
    State("new-montage-name", "value"),
    State("selection-method-dropdown", "value"),
    State("montage-store", "data"),
    State({"type": "montage-checklist", "group": ALL}, "value"),
    State({"type": "random-pick-count", "group": ALL}, "value"),
    State("channel-store", "data"),
    prevent_initial_call=True,
)
def update_montage_store(
    n_clicks,
    new_montage_name,
    selection_method,
    montage_store_data,
    checked_values,
    pick_values,
    channel_groups,
):
    if n_clicks <= 0 or not new_montage_name:
        return dash.no_update, dash.no_update

    if not montage_store_data:
        montage_store_data = {}

    region_keys = list(channel_groups.keys())
    selected_channels = []

    if selection_method == "checklist":
        for checked in checked_values:
            if checked:
                selected_channels.extend(checked)

    elif selection_method == "random":
        for i, group in enumerate(region_keys):
            available = channel_groups[group]
            try:
                pick_count = int(pick_values[i]) if pick_values[i] else 0
            except ValueError:
                pick_count = 0

            if pick_count > 0:
                selected_channels.extend(
                    random.sample(available, min(pick_count, len(available)))
                )

    if not selected_channels:
        return dash.no_update, dash.no_update

    # Save to store
    montage_store_data[new_montage_name] = selected_channels

    return montage_store_data, "/settings/montage"


@callback(
    Output("channels-layout-img", "src"),
    Input({"type": "montage-checklist", "group": ALL}, "value"),
    Input({"type": "random-pick-count", "group": ALL}, "value"),
    State("channel-store", "data"),
    State("data-path-store", "data"),
    State("selection-method-dropdown", "value"),
    State("raw-modality", "data"),
    prevent_initial_call=True,
)
def update_meg_layout(
    checked_values, pick_values, channel_groups, data_path, selection_method, modality
):

    region_keys = list(channel_groups.keys())
    selected_channels_by_group = [[] for _ in region_keys]

    if selection_method == "checklist":
        for i, checked in enumerate(checked_values):
            if checked:
                selected_channels_by_group[i].extend(checked)

    elif selection_method == "random":
        for i, group in enumerate(region_keys):
            available = channel_groups[group]
            try:
                pick_count = int(pick_values[i]) if pick_values[i] else 0
            except ValueError:
                pick_count = 0

            if pick_count > 0:
                selected = random.sample(available, min(pick_count, len(available)))
                selected_channels_by_group[i].extend(selected)

    if not any(selected_channels_by_group):
        return dash.no_update

    raw = dpu.read_raw(data_path, preload=False, verbose=False)

    if modality == "eeg":
        # Drop "eeg"-typed channels without a scalp position (e.g. ECG, EMG,
        # markers mislabeled as "eeg" by the file reader) before set_montage,
        # which otherwise raises on their missing position.
        montage = mne.channels.make_standard_montage("standard_1020")
        eeg_picks = set(mne.pick_types(raw.info, meg=False, eeg=True))
        scalp_picks = set(chu.get_scalp_eeg_picks(raw.info, montage=montage))
        non_scalp = [raw.ch_names[i] for i in eeg_picks - scalp_picks]
        if non_scalp:
            raw.drop_channels(non_scalp)
        raw.set_montage(montage)

    info = raw.info

    highlighted = [
        mne.pick_channels(info.ch_names, group)
        for group in selected_channels_by_group
        if group
    ]

    fig = mne.viz.plot_sensors(
        info,
        kind="topomap",
        ch_groups=highlighted,
        show_names=False,
        pointsize=200,
        linewidth=0,
        show=True,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


@callback(
    Output({"type": "montage-checklist", "group": MATCH}, "value"),
    Input({"type": "check-all-btn", "group": MATCH}, "n_clicks"),
    State("channel-store", "data"),
    prevent_initial_call=True,
)
def check_all_channels(n_clicks, channel_store):
    if not n_clicks or n_clicks <= 0:
        return dash.no_update

    group = dash.callback_context.triggered_id["group"]
    return channel_store.get(group, [])


@callback(
    Output(
        {"type": "montage-checklist", "group": MATCH}, "value", allow_duplicate=True
    ),
    Input({"type": "clear-all-btn", "group": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_channels(n_clicks):
    if not n_clicks or n_clicks <= 0:
        return dash.no_update
    return []
