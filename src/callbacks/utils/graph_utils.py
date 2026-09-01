import time
import itertools
import numpy as np
import pandas as pd
import dash
from dash import Patch
from pathlib import Path
import base64
import plotly.graph_objects as go
from layout.config_layout import (
    DEFAULT_FIG_LAYOUT,
    REGION_COLOR_PALETTE,
    COLOR_PALETTE,
    ERROR,
)
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import dataframe_utils as du
from callbacks.utils import smoothgrad_utils as su


def calculate_channel_offset_std(signal_df, scale_factor, min_offset=1, max_offset=10):
    # per-channel quantile clipping to reduce artifact effect
    q_low = signal_df.quantile(0.1)
    q_high = signal_df.quantile(0.9)
    trimmed = signal_df.clip(lower=q_low, upper=q_high, axis=1)

    stds = trimmed.std(skipna=True)
    scale_norm = (max_offset - min_offset) / scale_factor * 5
    return stds.mean() * scale_norm


def get_y_axis_ticks_with_gap(channel_names, base_offset, group_gap=2):
    """
    Compute y-axis ticks with additional spacing when the side (L/R/Z) changes.

    Parameters:
    - channel_names: list of channel names (e.g., ['MRC61-2805', 'MLC23-2805'])
    - base_offset: float, base vertical spacing between channels
    - group_gap: multiplier to insert larger gap when group changes

    Returns:
    - np.array of y-axis positions with extra group separation
    """
    y_ticks = []
    current_y = 0
    previous_side = None

    for name in channel_names:
        side = name[1] if len(name) > 1 else ""

        if previous_side is not None and side != previous_side:
            current_y += base_offset * group_gap  # Add extra space

        y_ticks.append(current_y)
        current_y += base_offset
        previous_side = side

    return np.array(y_ticks)


def apply_default_layout(
    fig, xaxis_range, time_range, selected_channels, y_axis_ticks, channel_offset
):
    """
    Apply default layout to a Plotly figure with dynamic values for axis range, title, etc.

    Parameters:
    - fig: The Plotly figure to update.
    - xaxis_range: Range for the x-axis (list or tuple).
    - time_range: Time range for the x-axis (tuple or list with two values [start, end]).
    - data_path: Title for the plot.

    Returns:
    - Updated Plotly figure with applied layout.
    """
    layout = DEFAULT_FIG_LAYOUT.copy()
    layout["xaxis"]["range"] = xaxis_range          #type: ignore
    layout["xaxis"]["minallowed"] = time_range[0]   #type: ignore
    layout["xaxis"]["maxallowed"] = time_range[1]   #type: ignore

    height_per_channel = 35  # if compact_view else 35
    layout["height"] = max(500, len(selected_channels) * height_per_channel)  #type: ignore

    ymin = min(y_axis_ticks) - 2 * channel_offset
    ymax = max(y_axis_ticks) + 2 * channel_offset
    layout["yaxis"]["range"] = [ymin, ymax]         #type: ignore

    fig.update_layout(layout)
    return fig


def generate_graph_time_channel(
    selected_channels,
    offset_selection,
    time_range,
    data_path,
    freq_data,
    color_selection,
    xaxis_range,
    channels_region,
    display_channels_region,
    filter={},
    excluded_ica_components=None,
):
    """
    Handles the preprocessing and figure generation for the MEG/EEG signal
    visualization.

    `channels_region` is the scalp-space channel grouping, used to
    reprocess/reorder the raw signal on a cache miss. `display_channels_region`
    matches the channels actually shown (e.g. bipolar-derived ones) and
    drives the "rainbow" per-group coloring of traces.
    """
    import time

    # Get recording from cache
    start_time = time.time()
    raw_ddf, _ = pu.get_preprocessed_dataframe_dask(
        data_path, freq_data, time_range[0], time_range[1], channels_region, excluded_ica_components=excluded_ica_components,
    )

    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    shifted_times = du.get_shifted_time_axis_dask(time_range, raw_ddf)
    print(
        f"Step 2: Time shifting completed in {time.time() - filter_start_time:.2f} seconds."
    )

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    try:
        filtered_raw_df = raw_ddf[selected_channels].compute()

        clean_variance = filtered_raw_df.var().mean()
        print(f"Variance : {clean_variance}")
        print(type(filtered_raw_df))
        print(filtered_raw_df.shape)
    except Exception:
        return (
            dash.no_update,
            "⚠️ Error: Selected channels are invalid. This may be due to choosing a montage that is incompatible with the current data format.",
            ERROR,
        )

    print(
        f"Step 3: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds."
    )

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset_std(filtered_raw_df, offset_selection)
    y_axis_ticks = get_y_axis_ticks_with_gap(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(
        y_axis_ticks, (len(filtered_raw_df), 1)
    )
    print(
        f"Step 4: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds."
    )

    # Create a dictionary mapping channels to their colors
    if color_selection == "rainbow":
        region_to_color = {
            region: REGION_COLOR_PALETTE[i % len(REGION_COLOR_PALETTE)]
            for i, region in enumerate(display_channels_region.keys())
        }

        # Build channel-to-color mapping
        channel_to_color = {
            channel: region_to_color[region]
            for region, channels in display_channels_region.items()
            for channel in channels
        }

        # Final color map only for selected channels
        color_map = {
            channel: channel_to_color[channel]
            for channel in selected_channels
            if channel in channel_to_color
        }
    elif color_selection == "blue":
        color_map = {channel: "blue" for channel in selected_channels}
    elif color_selection == "white":
        color_map = {channel: "white" for channel in selected_channels}
    elif "smoothGrad" in color_selection:
        color_map = {channel: "blue" for channel in selected_channels}

    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = (
        shifted_times  # Add time as a column for Plotly Express
    )

    fig = go.Figure()

    for col in shifted_filtered_raw_df.columns[:-1]:  # Exclude Time
        fig.add_trace(
            go.Scattergl(
                x=shifted_filtered_raw_df["Time"],
                y=shifted_filtered_raw_df[col],
                mode="lines",
                name=col,
                line=dict(color=color_map.get(col, None), width=1),
            )
        )

    if "smoothGrad" in color_selection:
        fig = su.add_smoothgrad_scatter(
            fig,
            shifted_filtered_raw_df,
            time_range,
            selected_channels,
            filter=filter,
            all_channels=sum(channels_region.values(), []),
        )

    print(
        f"Step 5: Figure creation completed in {time.time() - fig_start_time:.2f} seconds."
    )

    layout_start_time = time.time()
    fig = apply_default_layout(
        fig, xaxis_range, time_range, selected_channels, y_axis_ticks, channel_offset
    )
    print(
        f"Step 6: Layout update completed in {time.time() - layout_start_time:.2f} seconds."
    )

    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    return fig, None, {"display": "none"}


def generate_graph_time_ica(
    offset_selection,
    time_range,
    data_path,
    ica_result_path,
    color_selection,
    xaxis_range,
    excluded_indices
):
    """
    Generate a Plotly figure representing ICA component time courses.

    This function coordinates the following visualization pipeline:
    1. Retrieves resampled ICA sources using Dask/Parquet.
    2. Computes vertical offsets for trace stacking.
    3. Maps colors and opacities (dimming excluded components).
    4. Renders the figure using WebGL (Scattergl) for smooth interaction.

    Parameters
    ----------
    offset_selection : float
        Scalar multiplier used to determine the vertical spacing between 
        component traces.
    time_range : tuple of float
        The (start, end) timestamps in seconds for the data chunk to load.
    data_path : str
        Path to the original raw MEG/EEG data file.
    ica_result_path : str
        Path to the pre-computed MNE ICA solution file (.fif).
    color_selection : {'rainbow', 'blue', 'white'}
        The color scheme applied to the signal traces.
    xaxis_range : list of float
        The [min, max] values for the initial x-axis zoom level.
    excluded_indices : list of int or set
        Indices of ICA components (e.g., [1, 5]) that should be rendered 
        with lower opacity to indicate they are marked for exclusion.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting Plotly figure containing the styled ICA traces.
    error : str or None
        A descriptive error message if the process fails, otherwise None.
    """
    
    import time

    start_time = time.time()
    try:
        raw_ddf = pu.get_ica_components_dask(
            data_path, time_range[0], time_range[1], ica_result_path
        )
    except Exception:
        return (
            dash.no_update,
            "⚠️ You must select an ICA result before trying to display",
        )
    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    shifted_times = du.get_shifted_time_axis_dask(time_range, raw_ddf)
    print(
        f"Step 2: Time shifting completed in {time.time() - filter_start_time:.2f} seconds."
    )

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    selected_channels = [col for col in raw_ddf.columns if col.startswith("ICA")]
    try:
        filtered_raw_df = raw_ddf[selected_channels].compute()
    except Exception:
        return (
            dash.no_update,
            "⚠️ Error: Selected channels are invalid. This may be due to choosing a montage that is incompatible with the current data format.",
        )

    print(
        f"Step 3: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds."
    )

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset_std(filtered_raw_df, offset_selection)
    y_axis_ticks = np.arange(len(selected_channels)) * channel_offset
    shifted_filtered_raw_df = filtered_raw_df + np.tile(
        y_axis_ticks, (len(filtered_raw_df), 1)
    )
    print(
        f"Step 4: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds."
    )

    if color_selection == "rainbow":
        num_colors = len(REGION_COLOR_PALETTE)
        color_map = {
            channel: REGION_COLOR_PALETTE[i % num_colors]
            for i, channel in enumerate(selected_channels)
        }
    elif color_selection == "blue":
        color_map = {channel: "blue" for channel in selected_channels}
    elif color_selection == "white":
        color_map = {channel: "white" for channel in selected_channels}

    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = (
        shifted_times  # Add time as a column for Plotly Express
    )

    fig = go.Figure()

    for col in shifted_filtered_raw_df.columns[:-1]:  # Exclude Time
        try:
            col_idx = int(col.replace("ICA", ""))
        except (ValueError, TypeError):
            col_idx = None

        opacity = 0.2 if (excluded_indices and col_idx in excluded_indices) else 1.0
        
        fig.add_trace(
            go.Scattergl(
                x=shifted_filtered_raw_df["Time"],
                y=shifted_filtered_raw_df[col],
                opacity=opacity,
                mode="lines",
                name=col,
                line=dict(color=color_map.get(col, None), width=1),
            )
        )

    print(
        f"Step 5: Figure creation completed in {time.time() - fig_start_time:.2f} seconds."
    )

    layout_start_time = time.time()
    fig = apply_default_layout(
        fig, xaxis_range, time_range, selected_channels, y_axis_ticks, channel_offset
    )
    print(
        f"Step 6: Layout update completed in {time.time() - layout_start_time:.2f} seconds."
    )

    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    return fig, None


def update_annotations_on_graph(
    fig_dict, annotations_to_show, page_selection, annotations, chunk_limits
):
    """Update annotations visibility based on the checklist selection."""

    start_time = time.time()

    fig_patch = Patch()
    y_min, y_max = fig_dict["layout"]["yaxis"].get("range", [0, 1])

    # Filter annotations based on the current time range
    annotations_start_time = time.time()
    time_range = chunk_limits[int(page_selection)]
    annotations_df = pd.DataFrame(annotations).set_index("onset")
    filtered_annotations_df = du.get_annotations_df_filtered_on_time(
        time_range, annotations_df
    )

    annotations_end_time = time.time()
    print(
        f"Time to filter annotations: {annotations_end_time - annotations_start_time:.4f} seconds"
    )

    # Prepare the shapes and annotations for the selected annotations
    new_shapes = []
    new_annotations = []

    description_colors = {}
    color_cycle = itertools.cycle(COLOR_PALETTE)
    for desc in annotations_to_show:
        description_colors[desc] = next(color_cycle)

    shapes_start_time = time.time()

    for _, row in filtered_annotations_df.iterrows():
        description = row["description"]

        if str(description) in annotations_to_show:
            duration = row["duration"]
            color = description_colors[description]  # Get assigned color
            # Check the duration and add either a vertical line or a rectangle
            if duration == 0:
                # Vertical line if duration is 0
                new_shapes.append(
                    dict(
                        type="line",
                        x0=row.name,
                        x1=row.name,
                        y0=y_min,
                        y1=y_max,
                        xref="x",
                        yref="y",
                        line=dict(color=color, width=1, dash="line"),
                        opacity=0.9,
                    )
                )
            else:
                # Rectangle if duration > 0
                new_shapes.append(
                    dict(
                        type="rect",
                        x0=row.name,
                        x1=row.name + duration,
                        y0=y_min,
                        y1=y_max,
                        xref="x",
                        yref="y",
                        line=dict(color=color, width=2),
                        fillcolor=color,
                        opacity=0.3,
                    )
                )
            # Add the label in the margin
            new_annotations.append(
                dict(
                    x=row.name,
                    y=0,
                    xref="x",
                    yref="paper",
                    text=description,
                    showarrow=False,
                    font=dict(size=10, color=color),
                    align="center",
                    xanchor="center",
                    textangle=-90,
                    bgcolor="white",
                    borderwidth=1,
                    opacity=1,
                )
            )

    shapes_end_time = time.time()
    print(
        f"Time to generate shapes and annotations: {shapes_end_time - shapes_start_time:.4f} seconds"
    )

    fig_patch["layout"]["shapes"] = new_shapes
    fig_patch["layout"]["annotations"] = new_annotations

    end_time = time.time()
    print(
        f"Total execution time for update_annotations_on_graph: {end_time - start_time:.4f} seconds"
    )

    return fig_patch

def get_ica_components_figures(components_dir):
    components_dir = Path(components_dir)
    images_b64 = []

    for png_path in sorted(components_dir.glob("page_*.png")):
        with open(png_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images_b64.append(f"data:image/png;base64,{encoded}")

    return images_b64