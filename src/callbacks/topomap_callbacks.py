import time
import dash
from dash import html, Input, Output, State, callback
from callbacks.utils import path_utils as dpu
from callbacks.utils import topomap_utils as tu
from callbacks.utils import preprocessing_utils as pu


def register_display_topomap_on_click():
    @callback(
        Output("topomap-result", "children"),
        Output("topomap-picture", "children"),
        Output("history-store", "data", allow_duplicate=True),
        Input("signal-graph", "clickData"),
        State("data-path-store", "data"),
        State("plot-topomap-button", "outline"),
        State("page-selector", "value"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("channel-store", "data"),
        State("raw-modality", "data"),
        prevent_initial_call=True,
    )
    def display_clicked_content(
        click_info,
        data_path,
        button,
        page_selection,
        chunk_limits,
        freq_data,
        channel_store,
        modality,
    ):
        """
        Generate and display a sensor topomap based on a clicked time point.

        Parameters
        ----------
        click_info : dict
            Data from the Plotly click event containing coordinates (x is time).
        data_path : str
            Path to the raw MEG/EEG data file.
        is_button_active : bool
            State of the topomap toggle button (False usually indicates 'active' outline).
        page_selection : int or str
            Index of the current time chunk being viewed.
        chunk_limits : list of tuples
            Start and end times for each data chunk.
        freq_data : dict
            Preprocessing parameters (resample_freq, filters).
        channel_store : dict
            Mapping of channel types and names, including 'bad' channels.
        modality : str
            The data modality (e.g., 'meg', 'eeg').

        Returns
        -------
        tuple
            (Time label string, html.Img component for topomap, dash.no_update).
        """
        if button is False:
            try:
                start_time = time.time()  # Start timing
                t = click_info["points"][0]["x"]
                print(
                    f"Time to extract time from click info: {time.time() - start_time:.4f} seconds"
                )

                # Load raw data (metadata only)
                load_start_time = time.time()
                raw = dpu.read_raw(data_path, preload=False, verbose=False)
                time_range = chunk_limits[int(page_selection)]
                # Topomaps are computed on the original scalp channels, not
                # the bipolar montage used for display (a bipolar channel
                # has no single meaningful scalp position).
                scalp_freq_data = pu.get_original_signal_freq_data(freq_data)
                raw_ddf, _ = pu.get_preprocessed_dataframe_dask(
                    data_path, scalp_freq_data, time_range[0], time_range[1], channel_store
                )
                print(
                    f"Time to load raw and preprocessed data: {time.time() - load_start_time:.4f} seconds"
                )

                img_str_start_time = time.time()
                img_str = tu.create_topomap_from_preprocessed(
                    raw,
                    raw_ddf,
                    freq_data["resample_freq"],
                    time_range[0],
                    t,
                    channel_store.get("bad", []),
                    modality,
                )  # Returns base64-encoded string
                print(
                    f"Time to generate topomap image: {time.time() - img_str_start_time:.4f} seconds"
                )

                img_src = f"data:image/png;base64,{img_str}"
                topomap_image = html.Img(
                    src=img_src,
                    style={
                        "width": "100%",
                    },
                )

                return (
                    f"Time: {t:.3f} s",
                    topomap_image,
                    dash.no_update,
                )

            except Exception as e:
                print(f"⚠️ Error in plot_topomap: {str(e)}")
                return None, dash.no_update, dash.no_update

        return dash.no_update, dash.no_update, dash.no_update


def register_activate_deactivate_topomap_button():
    @callback(
        [Output("plot-topomap-button", "outline"),
         Output("plot-topomap-button", "children")],
        Input("plot-topomap-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _activate_deactivate_topomap_button(n_clicks):
        if n_clicks % 2 == 0:
            return True, "Activate"
        return False, "Deactivate"
