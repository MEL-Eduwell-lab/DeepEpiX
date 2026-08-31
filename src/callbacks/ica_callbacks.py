import os
import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
import config
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import history_utils as hu
from callbacks.utils import graph_utils as gu


def register_compute_ica():
    @callback(
        Output("ica-status", "children"),
        Output("compute-ica-button", "n_clicks"),
        Output("ica-store", "data"),
        Output("history-store", "data", allow_duplicate=True),
        Output("ica-components-dir-store", "data"),
        Input("compute-ica-button", "n_clicks"),
        State("data-path-store", "data"),
        State("chunk-limits-store", "data"),
        State("n-components", "value"),
        State("ica-method", "value"),
        State("max-iter", "value"),
        State("decim", "value"),
        State("history-store", "data"),
        State("ica-store", "data"),
        State("channel-store", "data"),
        State("ica-components-dir-store", "data"),
        prevent_initial_call=True,
    )
    def _compute_ica(
        n_clicks,
        data_path,
        chunk_limits,
        n_components,
        ica_method,
        max_iter,
        decim,
        history_data,
        ica_store,
        channel_store,
        components_dir_store,
    ):
        """
        Decompose MEG/EEG signals into independent components using ICA.

        This function validates input parameters, executes the ICA decomposition 
        (or retrieves it from cache), save computed ICA in cache, and triggers the Dask-based generation
        of component time-courses for the visualization interface.

        Parameters
        ----------
        n_clicks : int
            Trigger count from the compute button.
        data_path : str
            File system path to the raw data.
        chunk_limits : list of tuples
            Time boundaries (start, end) for data segments.
        n_components : int or float
            Number of principal components to keep, or the fraction of 
            variance to explain.
        ica_method : {'fastica', 'infomax', 'picard'}
            The algorithm used to perform ICA decomposition.
        max_iter : int
            Maximum number of iterations for the algorithm to converge.
        decim : int
            Temporal decimation factor to speed up computation.
        history_data : dict
            Session history log for tracking user actions.
        ica_store : list
            Storage for ICA object paths.
        channel_store : dict
            Channel groupings and bad channel information.

        Returns
        -------
        tuple
            (Status message, button reset, ICA path list, updated history, components directory).
        """
        if n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Validation: Check if all required fields are filled
        if not data_path:
            error_message = "⚠️ Please choose a subject to display on Home page."
            return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not chunk_limits:
            error_message = "⚠️ You have a subject in memory but its recording has not been preprocessed yet. Please go back on Home page to reprocess the signal."
            return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        missing_fields = []
        if not n_components:
            missing_fields.append("N components")
        if not ica_method:
            missing_fields.append("ICA method")
        if not max_iter:
            missing_fields.append("Max iterations")
        if not decim:
            missing_fields.append("Temporal decimation")
        if missing_fields:
            error_message = (
                f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
            )
            return (
                error_message,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        try: 
            ica_path, components_dir, is_from_cache, n_components, explained_var = pu.run_ica_processing(
                data_path, n_components, ica_method, max_iter, decim, 
                channel_store, config.CACHE_DIR, ica_store
            )

            if is_from_cache:
                status_msg = "✅ Reusing existing ICA results"
                return status_msg, 0, ica_store, history_data, components_dir_store

            else:
                status_msg = "✅ ICA computation complete"
                for start_time, end_time in chunk_limits:
                    pu.get_ica_components_dask(data_path, start_time, end_time, ica_path)

                action = f"Computed ICA with <n_components = {n_components}, method: {ica_method}, max_iter: {max_iter}, decim: {decim}> as parameters.\n"
                history_data = hu.fill_history_data(history_data, "ICA", action, n_components, explained_var, ica_key=str(ica_path))
                if str(ica_path) not in ica_store:
                    ica_store.append(str(ica_path))

                if not isinstance(components_dir_store, dict):
                    components_dir_store = {}

                components_dir_store[str(ica_path)] = str(components_dir)

                return status_msg, 0, ica_store, history_data, components_dir_store

        except Exception as e:
            return f"Erreur : {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

def register_apply_ica_exclusion():
    @callback(
        Output("history-store", "data", allow_duplicate=True),
        Output("exclusion-status", "children"),
        Output("ica-components-selection", "value"),
        Input("apply-ica-exclusion-button", "n_clicks"),
        State("ica-components-selection", "value"),
        State("history-store", "data"),
        State("data-path-store", "data"),
        State("chunk-limits-store", "data"),
        State("ica-result-radio", "value"),
        State("frequency-store", "data"),
        State("channel-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_ica_exclusion(
        n_clicks, selected, history_data, data_path, 
        chunk_limits, ica_result_path, freq_data, channel_store):
        """
        Apply ICA component exclusion and trigger signal reconstruction caching.

        This callback takes the user-selected independent components (ICs), 
        merges them with previously excluded components, and reconstructs 
        the MEG signal. The cleaned signal is then cached on disk in Parquet 
        format for each data chunk.

        Parameters
        ----------
        n_clicks : int
            Number of times the apply button has been clicked.
        selected : list of int
            Component indices currently selected in the UI for exclusion.
        history_data : dict
            Store containing the session's processing history and current 
            excluded components.
        data_path : str
            Path to the raw MEG/EEG data file.
        chunk_limits : list of tuples
            Time windows (start, end) used for partitioned processing.
        ica_result_path : str
            Path to the .fif file containing the ICA solution.
        freq_data : dict
            Frequency filter and resampling settings.
        channel_store : dict
            Metadata about channel types and names.

        Returns
        -------
        history_data : dict
            Updated history containing the new cumulative list of 
            excluded components.
        status : dbc.Alert
            Status message indicating success or failure of the operation.
        clear_selection : list
            An empty list used to reset the component selection dropdown/checklist.
        """
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update

        if not selected:
            return (
                dash.no_update,
                dbc.Alert("No components selected.", color="warning", duration=3000),
                [],
            )
    
        history_data = history_data or {}
        meta = history_data.setdefault("metadata", {})
        ica_results = meta.setdefault("ica_results", {})
        ica_entry = ica_results.setdefault(ica_result_path, {
            "n_components": None,
            "explained_var": None,
            "excluded_components": [],
        })

        already_excluded = set(ica_entry.get("excluded_components", []))
        all_excluded = sorted(already_excluded | set(selected))

        # ICA is fit/applied on the original scalp channels (see
        # run_ica_processing), so build `prep_raw` the same way here,
        # regardless of the bipolar display montage; get_reconstructed_signal_dask
        # re-derives the bipolar montage on the cleaned signal when needed.
        scalp_freq_data = pu.get_original_signal_freq_data(freq_data)
        prep_raw = pu.sort_filter_resample(data_path, scalp_freq_data, channel_store)
        for start_time, end_time in chunk_limits:

            pu.get_reconstructed_signal_dask(
                data_path,
                freq_data,
                start_time,
                end_time,
                ica_result_path,
                all_excluded,
                prep_raw,
            )

        ica_entry["excluded_components"] = all_excluded
        action = f"Excluded components {all_excluded} from signal (ICA computed with {ica_entry['n_components']} components).\n"
        history_data = hu.fill_history_data(history_data, "ICA", action, ica_key=ica_result_path)

        status = dbc.Alert(
            f"{len(all_excluded)} component(s) permanently excluded "
            f"({len(all_excluded) - len(already_excluded)} new).",
            color="danger",
            duration=4000,
        )

        return history_data, status, []


def register_fill_ica_results(ica_result_radio_id):
    @callback(
        Output(ica_result_radio_id, "options"),
        Output(ica_result_radio_id, "value"),
        Input("sidebar-tabs-ica", "active_tab"),
        Input("ica-store", "data"),
        prevent_initial_call=False,
    )
    def fill_ica_results(pathname, ica_store):
        if not ica_store:
            return dash.no_update, dash.no_update

        options = [{"label": os.path.basename(k), "value": k} for k in ica_store]
        selected = ica_store[-1]

        return options, selected
    
def register_plot_ica_maps(ica_result_radio_id):
    @callback(
        Output("components-window-ica", "style"),
        Output("components-content-ica", "children"),
        Input("nav-components-ica", "n_clicks"),
        Input("close-components-ica", "n_clicks"),
        Input(ica_result_radio_id, "value"),
        State("components-window-ica", "style"),
        State("ica-components-dir-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_components_window(
        open_clicks, close_clicks, ica_result_path, current_style, components_dir
    ):
        """
        Manage the display state and content of the ICA topographic components side window.

        Parameters
        ----------
        open_clicks : int
            Trigger count from the navigation menu button.
        close_clicks : int
            Trigger count from the window's close (X) button.
        current_style : dict
            Current CSS style dictionary of the component window container.
        components_dir : str
            The filesystem path where ICA component images (PNGs/SVGs) 
            were saved during the ICA fitting process.

        Returns
        -------
        dict
            Updated CSS style for the window container (toggling 'display').
        dash.development.base_component.Component
            HTML content containing either the gallery of component images 
            or a warning message if no directory is found.
        """
        triggered = dash.ctx.triggered_id

        base_style = {
            "position": "fixed",
            "top": "70px",
            "right": "20px",
            "zIndex": 2000,
        }

        if triggered == "close-components-ica":
            return {**base_style, "display": "none"}, dash.no_update

        resolved_dir = None
        if isinstance(components_dir, dict):
            resolved_dir = components_dir.get(ica_result_path)
        else:
            resolved_dir = components_dir

        def build_content(resolved_dir):
            if not resolved_dir:
                return html.Div(
                    "⚠️ Aucune composante disponible. Veuillez d'abord calculer l'ICA.",
                    style={"padding": "12px", "color": "orange"},
                )
            images = gu.get_ica_components_figures(resolved_dir)
            return html.Div(
                [html.Img(src=src, style={"width": "100%", "marginBottom": "8px"}) for src in images],
                style={"overflowY": "auto", "maxHeight": "60vh", "padding": "8px"},
            )


        if triggered == "nav-components-ica":
            return {**base_style, "display": "block"}, build_content(resolved_dir)
        
        if triggered == ica_result_radio_id:
            window_is_open = current_style and current_style.get("display") == "block"
            if window_is_open:
                return current_style, build_content(resolved_dir)
            
        return current_style, dash.no_update
    