import dash
from dash import html, dcc, Input, Output, clientside_callback, callback
from dash_extensions import Keyboard
import os

# Layout imports
import layout.graph_layout as gl
from layout.sidebar_layout import create_sidebar
from layout.config_layout import ERROR
import dash_bootstrap_components as dbc

# Callback imports

# --- Selection ---
from callbacks.selection_callbacks import (
    register_toggle_sidebar,
    register_navigate_tabs_raw,
    register_update_channels_checklist_options,
    register_cancel_or_confirm_annotation_suppression,
    register_annotation_checkboxes_options,
    register_annotation_dropdown_options,
    register_callbacks_montage_names,
    register_callbacks_sensivity_analysis,
    register_hide_channel_selection_when_montage,
    register_clear_check_all_annotation_checkboxes,
    register_manage_channels_checklist,
    register_offset_display,
    register_page_buttons_display,
    register_modal_annotation_suppression,
    register_toggle_intersection_modal,
    register_create_intersection,
)

# --- Graph ---
from callbacks.graph_callbacks import register_update_graph_raw_signal

# --- Annotation ---
from callbacks.annotation_callbacks import (
    register_move_to_next_annotation,
    register_update_annotation_graph,
    register_update_annotations_on_graph,
    register_move_with_keyboard,
)

# --- Topomap ---
from callbacks.topomap_callbacks import (
    register_activate_deactivate_topomap_button,
    register_display_topomap_on_click,
)

# --- Spikes ---
from callbacks.spike_callbacks import (
    register_add_event_to_annotation,
    register_add_event_onset_duration_on_click,
    register_delete_selected_spike,
    register_enable_add_event_button,
    register_enable_delete_event_button,
)

# --- History ---
from callbacks.history_callbacks import (
    register_clean_ica_history,
    register_clean_annotation_history,
    register_update_annotation_history,
)

# --- Save ---
from callbacks.save_callbacks import (
    register_display_annotations_to_save_checkboxes,
    register_display_bad_channels_to_save_checkboxes,
    register_save_modifications,
    register_csv_name,
)

# --- Predict ---
from callbacks.predict_callbacks import (
    register_fill_signal_versions_predict,
    register_fill_model_options,
    register_execute_predict_script,
    register_store_display_prediction,
    register_update_selected_model,
    register_smoothgrad_threshold,
)

dash.register_page(__name__, name="Data Viz & Analyze", path="/viz/raw-signal")

layout = html.Div(
    [
        Keyboard(id="keyboard", captureKeys=["ArrowRight", "ArrowLeft", "+", "-"]),
        dcc.Location(id="url", refresh=False),
        # Main container
        html.Div(
            [
                # Sidebar container
                html.Div(
                    [
                        # Collapsible sidebar
                        dbc.Collapse(
                            create_sidebar(),
                            id="sidebar-collapse",
                            is_open=True,
                            dimension="width",
                            className="sidebar-collapse",
                        ),
                        # Button stack on the left
                        html.Div(
                            [
                                dbc.Button(
                                    html.I(
                                        id="sidebar-toggle-icon", className="bi bi-x-lg"
                                    ),
                                    id="toggle-sidebar",
                                    color="danger",
                                    size="sm",
                                    className="mb-2 shadow-sm",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-hand-index-thumb"),
                                    id="nav-select",
                                    color="primary",
                                    size="sm",
                                    className="mb-2",
                                    title="Select",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-activity"),
                                    id="nav-analyze",
                                    color="success",
                                    size="sm",
                                    className="mb-2",
                                    title="Analyze",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-stars"),
                                    id="nav-predict",
                                    color="warning",
                                    size="sm",
                                    className="mb-2",
                                    title="Spike Prediction",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-floppy"),
                                    id="nav-save",
                                    color="secondary",
                                    size="sm",
                                    className="mb-2",
                                    title="Save",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",  # button -> collapse (left to right)
                        "alignItems": "flex-start",
                        "zIndex": 1000,
                    },
                ),
                html.Div(id="dummy-output", style={"display": "none"}),
                html.Div(
                    [
                        # Error message overlaying the graph
                        html.Div(
                            [
                                "Please make a ",
                                html.I(
                                    className="bi bi-arrow-left-square-fill",
                                    style={"color": "#0d6efd"},
                                ),
                                " selection and click on ",
                                html.I(
                                    className="bi bi-arrow-clockwise",
                                    style={"color": "orange"},
                                ),
                                " Refresh button.",
                            ],
                            id="python-error",
                            style=ERROR,
                        ),
                        # Graph container
                        gl.create_graph_container(
                            update_button_id="update-button",
                            page_selector_id="page-selector",
                            next_spike_buttons_container_id="next-spike-buttons-container",
                            prev_spike_id="prev-spike",
                            next_spike_id="next-spike",
                            loading_id="loading-graph",
                            signal_graph_id="signal-graph",
                            annotation_graph_id="annotation-graph",
                        ),
                    ],
                    style={
                        "position": "relative",  # <-- key to overlay positioning
                        "flex": 1,
                    },
                ),
            ],
            style={
                "display": "flex",  # Horizontal layout
                "flexDirection": "row",
                "height": "90vh",  # Use the full height of the viewport
                "width": "96vw",  # Use the full width of the viewport
                "overflow": "hidden",  # Prevent overflow in case of resizing
                "boxSizing": "border-box",
                "gap": "20px",
            },
        ),
    ]
)

clientside_callback(
    """
    function(relayoutData) {
        // This function runs after relayout (zoom/pan/plotly_afterplot) events
        console.log("clientside callback triggered");
        // Get container and scroll it to bottom
        var container = document.getElementById('graph-container');
        if(container){
            container.scrollTop = container.scrollHeight;
        }
        console.log("clientside callback triggered");
        return '';
    }
    """,
    Output("dummy-output", "children"),
    Input("signal-graph", "figure"),
    prevent_initial_call=False,
)

# --- Siderbar dynamic ---
register_toggle_sidebar(
    collapse_id="sidebar-collapse",
    icon_id="sidebar-toggle-icon",
    toggle_id="toggle-sidebar",
)

register_navigate_tabs_raw(
    collapse_id="sidebar-collapse",
    sidebar_tabs_id="sidebar-tabs",
    icon_id="sidebar-toggle-icon",
)

# --- Page Navigation ---
register_page_buttons_display(page_selector_id="page-selector")

# --- Channels checklist ---
register_update_channels_checklist_options(checkboxes_id="channel-region-checkboxes")

register_manage_channels_checklist(checkboxes_id="channel-region-checkboxes")

register_hide_channel_selection_when_montage()

# --- Annotation Management ---
register_annotation_checkboxes_options(
    checkboxes_id="annotation-checkboxes",
)

register_annotation_dropdown_options(
    dropdown_id="annotation-dropdown", checkboxes_id="annotation-checkboxes"
)

register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-btn",
    clear_all_btn_id="clear-all-annotations-btn",
    checkboxes_id="annotation-checkboxes",
)

register_update_annotations_on_graph(
    graph_id="signal-graph",
    checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector",
)

register_update_annotation_graph(
    update_button_id="update-button",
    page_selector_id="page-selector",
    checkboxes_id="annotation-checkboxes",
    annotation_graph_id="annotation-graph",
)

register_display_annotations_to_save_checkboxes()

register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-to-save-btn",
    clear_all_btn_id="clear-all-annotations-to-save-btn",
    checkboxes_id="annotations-to-save-checkboxes",
)

register_modal_annotation_suppression(
    btn_id="delete-annotations-btn",
    checkboxes_id="annotation-checkboxes",
    modal_id="delete-confirmation-modal",
    modal_body_id="delete-modal-body",
)

register_cancel_or_confirm_annotation_suppression(
    confirm_btn_id="confirm-delete-btn",
    cancel_btn_id="cancel-delete-btn",
    checkboxes_id="annotation-checkboxes",
    modal_id="delete-confirmation-modal",
)

register_toggle_intersection_modal(
    btn_id="create-intersection-btn",
    checkboxes_id="annotation-checkboxes",
    modal_id="create-intersection-modal",
    modal_body_id="create-intersection-modal-body",
)

register_create_intersection(
    confirm_btn_id="confirm-intersection-btn",
    cancel_btn_id="cancel-intersection-btn",
    checkboxes_id="annotation-checkboxes",
    tolerance_id="intersection-tolerance",
    modal_id="create-intersection-modal",
)

# --- Graph & Channel Handling ---
register_update_graph_raw_signal()
register_offset_display(
    offset_decrement_id="offset-decrement",
    offset_increment_id="offset-increment",
    offset_display_id="offset-display",
    keyboard_id="keyboard",
)

# --- Topomap Interactions ---
register_display_topomap_on_click()
register_activate_deactivate_topomap_button()

# --- Spike Handling ---
register_add_event_onset_duration_on_click()
register_add_event_to_annotation()
register_delete_selected_spike()
register_enable_add_event_button()
register_enable_delete_event_button()


register_move_with_keyboard(
    keyboard_id="keyboard",
    graph_id="signal-graph",
    page_selector_id="page-selector",
)

register_move_to_next_annotation(
    prev_spike_id="prev-spike",
    next_spike_id="next-spike",
    graph_id="signal-graph",
    dropdown_id="annotation-dropdown",
    checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector",
)

# --- History ---
register_update_annotation_history()
register_clean_ica_history()
register_clean_annotation_history()

# --- Predict ---
register_fill_signal_versions_predict()
register_fill_model_options()
register_execute_predict_script()
register_store_display_prediction()
register_update_selected_model()
register_smoothgrad_threshold()

# --- Save ---
register_display_bad_channels_to_save_checkboxes()
register_save_modifications()
register_csv_name()

# --- Analysis ---
register_callbacks_sensivity_analysis()

# --- Montage ---
register_callbacks_montage_names(radio_id="montage-radio")
