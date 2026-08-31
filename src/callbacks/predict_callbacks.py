import os
import time
import subprocess

import pandas as pd
import dash
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc
import json
import hashlib

import config
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import predict_utils as pru

from config import STATIC_DIR


def register_update_selected_model():
    @callback(
        Output("venv", "value"),
        Input("model-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_selected_model(selected_value):
        """Update the selected model path and detect the environment."""
        if not selected_value:
            return dash.no_update

        # Detect environment
        if selected_value.endswith((".keras", ".h5")):
            environment = "TensorFlow (.tfenv)"
        elif selected_value.endswith((".pth", ".ckpt")):
            environment = "PyTorch (.torchenv)"
        else:
            environment = "Unknown"

        return environment

def register_fill_signal_versions_predict():
    @callback(
        Output("signal-version-predict", "options"),
        Input("history-store", "data"),
        Input("url", "pathname"),
        prevent_initial_call=False,
    )
    def _fill_signal_versions_predict(history_data, url):
        """
        This function populate signal version options for prediction.

        Parameters
        ----------
        history_data : dict or None
            The data dictionary from the 'history-store' component. 
            Expected to have structure: 
            `metadata -> ica_results -> {path: {excluded_components: [...]}}`.
        url : str
            The current page path from the 'url' component.

        Returns
        -------
        list of dict
            A list of option dictionaries for the dcc.Dropdown component.
        """
        options = [{"label": "Filtered signal", "value": "__raw__"}]
        ica_results = (
            (history_data or {})
            .get("metadata", {})
            .get("ica_results", {})
        )
        for ica_path, ica_meta in ica_results.items():
            excluded = ica_meta.get("excluded_components", [])
            if not excluded:
                continue
            name = os.path.basename(ica_path).removesuffix("-ica.fif")
            options.append({
                "label": f"ICA · {name}_{excluded}",
                "value": ica_path,
            })
        return options 

def register_fill_model_options():
    @callback(
        Output("model-dropdown", "options"),
        Output("model-dropdown", "value"),
        Input("raw-modality", "data"),
        Input("url", "pathname"),
        State("model-dropdown", "value"),
        prevent_initial_call=False,
    )
    def _fill_model_options(modality, url, current_value):
        """Restrict the model list to those adapted to the recording modality.

        Models live in per-modality sub-folders (``models/MEG`` and
        ``models/EEG``); only the ones matching ``raw-modality`` are proposed.
        """
        options = pru.get_model_options(modality)
        valid_values = {opt["value"] for opt in options}
        value = current_value if current_value in valid_values else None
        return options, value


def register_execute_predict_script():
    @callback(
        Output("prediction-status", "children"),
        Output("run-prediction-button", "n_clicks"),
        Output("store-display-div", "style"),
        Output("model-probabilities-store", "data", allow_duplicate=True),
        Output("sensitivity-analysis-store", "data", allow_duplicate=True),
        Input("run-prediction-button", "n_clicks"),
        State("data-path-store", "data"),
        State("frequency-store", "data"),
        State("channel-store", "data"),
        State("chunk-limits-store", "data"),
        State("history-store", "data"),
        State("signal-version-predict", "value"),
        State("model-dropdown", "value"),
        State("venv", "value"),
        State("smoothgrad-threshold", "value"),
        State("sensitivity-analysis", "value"),
        State("adjust-onset", "value"),
        State("model-probabilities-store", "data"),
        State("sensitivity-analysis-store", "data"),
        State("channel-store", "data"),
        State("predict-preprocess-choice", "value"),
        prevent_initial_call=True,
    )
    def _execute_predict_script(
        n_clicks,
        data_path,
        freq_data,
        channels_dict,
        chunk_limits,
        history_data,
        signal_version,
        model_path,
        venv,
        smoothgrad_threshold,
        sensitivity_analysis,
        adjust_onset,
        model_probabilities_store,
        sensitivity_analysis_store,
        channel_store,
        preprocessing_option,
    ):
        """
        Execute model inference and sensitivity analysis via external subprocesses.

        This callback reconstructs the signal of interest by aggregating the packets 
        (Parquet files) associated with the previously used time windows in the app 
        or reuses an existing global cache. It then retrieves the MNE metadata (JSON format)
        before launching the prediction and SmoothGrad scripts in the appropriate virtual environment.

        Parameters
        ----------
        n_clicks : int
            The number of times the 'run-prediction-button' has been clicked.
        data_path : str
            File system path to the input dataset or subject directory.
        freq_data : dict
            Frequency-related parameters for signal preprocessing.
        channels_dict : dict
            Mapping of channel names and indices to be used for extraction.
        chunk_limits : list of int
            Start and end indices for data slicing.
        history_data : dict
            Application session history containing metadata.
        signal_version : str
            The specific ICA version to use or '__raw__' for 
            filtered data.
        model_path : str
            Path to the selected model file (e.g., .h5 or .pth).
        venv : str
            The selected environment type (e.g., "TensorFlow" or "PyTorch").
        smoothgrad_threshold : float
            Indicated the minimum model's probability to compute SmoothGrad on the window
        sensitivity_analysis : bool
            Whether to execute the SmoothGrad attribution script.
        adjust_onset : bool
            Whether to apply post-processing logic to adjust predicted onset 
            times.
        model_probabilities_store : list
            Cached paths to existing prediction CSV files.
        sensitivity_analysis_store : list
            Cached paths to existing SmoothGrad pickle files.
        channel_store : str
            Serialized string representation of the selected channels.
        preprocessing_option : str
            Version of the signal to use:
                - 'custom' will use the signal preprocessed during the session;
                - 'same_as_training' will preprocess the signal from scratch using model's specific configuration.
        
        Returns
        -------
        prediction_status : str or bool
            Feedback message for the user, or True if the process completed 
            successfully.
        n_clicks : int
            Resets the button click count (returns 0 or dash.no_update).
        display_style : dict
            CSS style dictionary (e.g., {'display': 'block'}) to show/hide 
            result containers.
        model_probabilities_store : list
            Updated list containing the path to the new prediction CSV.
        sensitivity_analysis_store : list
            Updated list containing the path to the new SmoothGrad .pkl file.

        Notes
        -----
        The function follows a strict execution pipeline:
        1. **Cache Check**: If a prediction with the same model already exists 
           in the store, it returns early.
        2. **Signal Management**: The function checks for the existence of a global cache 
           via an MD5 hash (including data_path, freq_data, and excluded ICA components).
           If absent, it recreates the signal from the preprocessed time windows 
           and saves it in Parquet format.
        3. **MNE Metadata**: Retrieves signal structure information 
           from a JSON file (`_mne_meta.json`).
        4. **Inference**: Runs `main.py` using the `subprocess` module in the 
           environment specified by `venv`.
        5. **Attribution**: If `sensitivity_analysis` is True, runs 
           `run_smoothgrad.py`.
        """
        if not n_clicks or n_clicks == 0:
            return None, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not data_path:
            return "⚠️ Please choose a subject to display on Home page.", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        missing_fields = [f for f, v in [("Model", model_path), ("Environment", venv)] if not v]

        if missing_fields:
            return f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        excluded_ica_comp = None
        if signal_version != "__raw__":
            excluded_ica_comp = history_data["metadata"]["ica_results"][signal_version]["excluded_components"]

        signal_name = "raw" if signal_version == "__raw__" else os.path.basename(signal_version).split("-ica.fif")[0]

        cache_dir = config.CACHE_DIR
        signal_name_with_ica = f"{signal_name}_{excluded_ica_comp}" if excluded_ica_comp is not None else signal_name
        signal_name_with_preprocess = "same_as_training" if preprocessing_option == "same_as_training" else f"{signal_name_with_ica}_{preprocessing_option}"

        predictions_csv_path = cache_dir / f"{os.path.basename(model_path)}_{signal_name_with_preprocess}_predictions.csv"
        smoothgrad_path = cache_dir / f"{os.path.basename(model_path)}_{signal_name_with_preprocess}_smoothGrad.pkl"

        # When same_as_training is selected, ICA component info embedded in the signal name is irrelevant because preprocessing 
        # is redone from scratch using the model's config.
        if preprocessing_option == "same_as_training":
            model_basename = os.path.basename(model_path)
            existing_csv_candidates = list(cache_dir.glob(f"{model_basename}_*_same_as_training_predictions.csv"))

            if existing_csv_candidates:
                # All same_as_training runs for a given model produce identical results, so any existing file is a valid cache hit.
                predictions_csv_path = existing_csv_candidates[0]

            existing_pkl_candidates = list(cache_dir.glob(f"{model_basename}_*_same_as_training_smoothGrad.pkl"))

            if existing_pkl_candidates:
                smoothgrad_path = existing_pkl_candidates[0]

        need_predictions = not predictions_csv_path.exists()
        need_smoothgrad = sensitivity_analysis and not smoothgrad_path.exists()

        if need_smoothgrad and os.path.basename(model_path) != "model_CNN.keras":
            return "⚠️ SmoothGrad is only available for model_CNN model", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not need_predictions and not need_smoothgrad:
            reuse_message = (
                "✅ Reusing existing model predictions (same_as_training preprocessing is signal-independent)"
                if preprocessing_option == "same_as_training"
                else "✅ Reusing existing model predictions"
            )
            return (
                reuse_message,
                0,
                {"display": "block"},
                [str(predictions_csv_path)],
                [str(smoothgrad_path)] if smoothgrad_path.exists() else dash.no_update,
            )
        mne_info_path = "None"
        if preprocessing_option =="custom":
            meta = (history_data or {}).get("metadata", {})
            ica_results = meta.get("ica_results", {})
            excluded_ica = (
                ica_results[signal_version].get("excluded_components", [])
                if signal_version and signal_version != "__raw__" and signal_version in ica_results
                else None
            )
            # Always produce the prediction signal in the referential channel
            # space (get_original_signal_freq_data drops a bipolar display
            # reference); the model's own montage_type, kept from training,
            # drives any derivation downstream. Only the session filtering,
            # notch and removed ICA components are carried over.
            predict_freq_data = pu.get_original_signal_freq_data(freq_data)

            signal_cache_path = os.path.join(
                cache_dir,
                f"signal_{hashlib.md5(f'{data_path}_{json.dumps(predict_freq_data, sort_keys=True)}_{sorted(excluded_ica) if excluded_ica else []}_{preprocessing_option}'.encode()).hexdigest()}.parquet"
            )
            mne_info_path = signal_cache_path.replace(".parquet", "_mne_meta.json")

            # prep_raw is the referential signal ICA was fit on; build it once if
            # either the ICA segments or the sidecar still need it.
            prep_raw = None
            if excluded_ica or not os.path.exists(mne_info_path):
                prep_raw = pu.sort_filter_resample(data_path, predict_freq_data, channels_dict)

            if excluded_ica:
                # extract_preprocess_signal only *reads* ICA-cleaned segments, it
                # never applies ICA. Regenerate any missing segment here (in the
                # referential scalp space) so we don't silently fall back to the
                # non-ICA signal. get_reconstructed_signal_dask is a no-op when
                # the segment is already cached.
                for start_time, end_time in chunk_limits:
                    pu.get_reconstructed_signal_dask(
                        data_path, predict_freq_data, start_time, end_time,
                        signal_version, excluded_ica, prep_raw,
                    )

            if not os.path.exists(mne_info_path):
                pu.save_mne_sidecar(signal_cache_path, prep_raw)

            if not os.path.exists(signal_cache_path):
                signal = pru.extract_preprocess_signal(
                    data_path, predict_freq_data, channels_dict, chunk_limits, excluded_ica
                )
                print(f"Variance : {signal.var().mean()} | Shape : {signal.shape}")
                signal.to_parquet(signal_cache_path)
            else:
                print(f"✅ Signal already in cache — skip extraction ({signal_cache_path})")

        else:
            print("🔄 Applying training preprocessing to signal...")
            signal_cache_path = os.path.join(
                cache_dir,
                f"signal_train_preproc_{hashlib.md5(f'{data_path}_{os.path.basename(model_path)}'.encode()).hexdigest()}.parquet"
            )
            model_config = f"{STATIC_DIR}/model_config.json"

            print(f"Model config : {model_config}")
            mne_info_path, df = pu.preprocess_same_as_training(model_config, model_path, data_path, channels_dict, signal_cache_path)
        
        # Otherwise, execute model
        if "TensorFlow" in venv:
            ACTIVATE_ENV = str(config.TENSORFLOW_ENV / "bin/python")
        elif "PyTorch" in venv:
            ACTIVATE_ENV = str(config.TORCH_ENV / "bin/python")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(config.APP_ROOT)

        if need_predictions:
            command = [
                ACTIVATE_ENV,
                str(config.MODEL_PIPELINE_DIR / "main.py"),
                str(model_path),
                str(venv),
                str(data_path),
                str(cache_dir),
                str(adjust_onset),
                str(channel_store),
                str(signal_cache_path),
                str(mne_info_path),
                str(signal_name_with_preprocess),
            ]

            try:
                start_time = time.time()
                subprocess.run(
                    command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
                )  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Model testing executed in {time.time()-start_time:.2f} seconds")

            except Exception as e:
                return f"⚠️ Error running model: {e}", dash.no_update, dash.no_update, dash.no_update, dash.no_update,
    
        if not predictions_csv_path.exists():
                return "⚠️ Error running prediction.", 0, {"display": "block"}, model_probabilities_store, dash.no_update
    
        model_probabilities_store = [str(predictions_csv_path)]

        if need_smoothgrad:
            command = [
                ACTIVATE_ENV,
                str(config.MODEL_PIPELINE_DIR / "run_smoothgrad.py"),
                str(model_path),
                str(venv),
                str(cache_dir),
                str(predictions_csv_path),
                str(smoothgrad_threshold),
                str(mne_info_path),
                str(signal_name_with_preprocess),
            ]

            try:
                start_time = time.time()
                subprocess.run(
                    command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
                )
                print(f"Smoothgrad executed in {time.time()-start_time:.2f} seconds")

            except Exception as e:
                return f"⚠️ Error running smoothgrad: {e}", 0, {"display": "block"}, model_probabilities_store, dash.no_update

            if not smoothgrad_path.exists():
                return "⚠️ Error running smoothgrad.", 0, {"display": "block"}, model_probabilities_store, dash.no_update

        return (
            True,
            0,
            {"display": "block"},
            model_probabilities_store,
            sensitivity_analysis_store,
        )


@callback(
    Output("prediction-output-summary-div", "children"),
    Output("prediction-output-distribution-div", "children"),
    Output("prediction-output-table-div", "children"),
    Input("store-display-div", "style"),
    Input("adjusted-threshold", "value"),
    State("model-probabilities-store", "data"),
    prevent_initial_call=True,
)
def update_prediction_table(style, threshold, prediction_csv_path):
    if style["display"] == "none":
        return None, None, None
    if not prediction_csv_path or threshold is None:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df = pd.read_csv(prediction_csv_path[0])
        df_filtered = df[df["probas"] > threshold]
        df_filtered["probas"] = df_filtered["probas"].round(2)

        if df_filtered.empty:
            msg = html.P("No events found in this recording.")
            return msg, msg, msg

        table = dbc.Table.from_dataframe( #type: ignore
            df_filtered.copy(),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            dark=False,
        )

        # UIEDNET "probas" are segmentation-peak prominences, not calibrated
        # probabilities, so the probability histogram is not meaningful.
        if "uiednet" in os.path.basename(prediction_csv_path[0]).lower():
            distribution = html.P(
                "Probability distribution is not available for UIEDNET predictions."
            )
        else:
            distribution = au.build_prediction_distribution_statistics(df, threshold)

        return (
            au.build_table_prediction_statistics(df_filtered, len(df)),
            distribution,
            table,
        )
    except Exception as e:
        error_msg = html.P(f"⚠️ Error loading predictions: {e}")
        return error_msg, error_msg, error_msg


@callback(
    Output("model-spike-name", "value"),
    Input("model-dropdown", "value"),
    Input("signal-version-predict", "value"),
    Input("predict-preprocess-choice", "value"),
    State("history-store", "data"),
    prevent_initial_call=True,
)
def update_spike_name(model_path, signal_version, preprocessing, history_data):
    """ Update predicted spike annotation's name"""
    if model_path is None:
        return dash.no_update
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    if preprocessing == "same_as_training":
        return f"{model_name}_same_as_training"
    
    if signal_version == "__raw__":
        signal_name = "raw"
        return f"{model_name}_raw"
    else:
        signal_name = os.path.basename(signal_version).split("-ica.fif")[0]
        excluded = history_data["metadata"]["ica_results"][signal_version]["excluded_components"]
        return f"{model_name}_{signal_name}_{excluded}"


def register_store_display_prediction():
    @callback(
        Output("annotation-store", "data", allow_duplicate=True),
        Output("sidebar-tabs", "active_tab"),
        Output("store-display-div", "style", allow_duplicate=True),
        Output("history-store", "data"),
        Output("model-csv-store", "data"),
        Input("store-display-button", "n_clicks"),
        State("annotation-store", "data"),
        State("model-probabilities-store", "data"),
        State("adjusted-threshold", "value"),
        State("model-spike-name", "value"),
        State("history-store", "data"),
        State("model-csv-store", "data"),
        prevent_initial_call=True,
    )
    def store_display_prediction(
        n_clicks,
        annotation_data,
        prediction_csv_path,
        threshold,
        spike_name,
        history_data,
        model_csv_store,
    ):
        if not n_clicks or n_clicks == 0 or prediction_csv_path is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        if not annotation_data:
            annotation_data = []

        unique_spike_name = spike_name
        counter = 1
        while unique_spike_name in model_csv_store:
            unique_spike_name = f"{spike_name} ({counter})"
            counter += 1

        df = pd.read_csv(prediction_csv_path[0])
        prediction_df = df[df["probas"] > threshold]

        new_annotations = prediction_df[["onset", "duration"]].copy()
        new_annotations["description"] = unique_spike_name  # Set spike name as description
        new_annotations_dict = new_annotations.to_dict(orient="records") #type: ignore

        annotation_data.extend(new_annotations_dict)

        model_csv_store[unique_spike_name] = prediction_csv_path[0]

        action = f"Tested model with <{unique_spike_name}> as the predicted event name.\n"
        history_data = hu.fill_history_data(history_data, "models", action)

        # Return updated annotations and switch tab
        return annotation_data, "selection-tab", {"display": "none"}, history_data, model_csv_store

def register_smoothgrad_threshold():
    @callback(
    Output("smoothgrad-threshold-div", "style"),
    Input("sensitivity-analysis", "value"),
    )
    def toggle_smoothgrad_threshold(sensitivity_analysis):
        if sensitivity_analysis:
            return {"marginBottom": "20px", "display": "block"}
        return {"marginBottom": "20px", "display": "none"}