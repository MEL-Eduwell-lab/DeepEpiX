# Dash & Plotly
import dash
from dash import Input, Output, State, callback
import config

# External Libraries

# Local Imports
from callbacks.utils import path_utils as dpu
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import annotation_utils as au
from callbacks.utils import channel_utils as chu
from callbacks.utils import cache_utils as cu

def register_handle_frequency_parameters():
    @callback(
        Output("preprocess-status", "children"),
        Input("resample-freq", "value"),
        Input("high-pass-freq", "value"),
        Input("low-pass-freq", "value"),
        Input("notch-freq", "value"),
        prevent_initial_call=True,
    )
    def handle_frequency_parameters(
        resample_freq, high_pass_freq, low_pass_freq, notch_freq
    ):
        """Retrieve frequency parameters and store them."""
        if not low_pass_freq or not high_pass_freq:
            return "⚠️ Please fill high and low pass frequency parameters."

        elif high_pass_freq >= low_pass_freq:
            return "⚠️ High-pass frequency must be less than low-pass frequency."

        return dash.no_update


def register_toggle_eeg_reference():
    @callback(
        Output("eeg-reference", "disabled"),
        Output("eeg-reference", "value"),
        Input("pending-has-scalp-eeg", "data"),
        prevent_initial_call=True,
    )
    def toggle_eeg_reference(has_scalp_eeg):
        """
        Enable EEG re-referencing only when the recording has real 10-20
        scalp electrodes (see chu.get_scalp_eeg_picks) -- not just any
        "eeg"-typed channel, since a MEG recording with a few auxiliary
        channels (ECG, EOG...) mislabeled as "eeg" shouldn't count.
        """
        if has_scalp_eeg:
            return False, dash.no_update
        # MEG recording: re-referencing is irrelevant ("average" is a no-op with
        # no scalp electrodes), so just disable the control on a valid value.
        return True, "average"


def register_preprocess_meg_data():
    @callback(
        Output("preprocess-status", "children", allow_duplicate=True),
        Output("frequency-store", "data"),
        Output("annotation-store", "data"),
        Output("channel-store", "data", allow_duplicate=True),
        Output("display-channel-store", "data"),
        Output("raw-modality", "data"),
        Output("chunk-limits-store", "data"),
        Output("url", "pathname"),
        Output("history-store", "data", allow_duplicate=True),
        Output("ica-store", "data", allow_duplicate=True),
        Output("ica-components-dir-store", "data", allow_duplicate=True),
        Output("model-csv-store", "data", allow_duplicate=True),
        Input("preprocess-display-button", "n_clicks"),
        State("data-path-store", "data"),
        State("resample-freq", "value"),
        State("high-pass-freq", "value"),
        State("low-pass-freq", "value"),
        State("notch-freq", "value"),
        State("eeg-reference", "value"),
        State("heartbeat-channel", "value"),
        State("bad-channels", "value"),
        running=[(Output("compute-display-psd-button", "disabled"), True, False)],
        prevent_initial_call=True,
    )
    def preprocess_meg_data(
        n_clicks,
        data_path,
        resample_freq,
        high_pass_freq,
        low_pass_freq,
        notch_freq,
        eeg_reference,
        heartbeat_ch_name,
        bad_channels,
    ):
        """
        Execute M/EEG preprocessing pipeline, save mne info & update application state.
        Preprocessing a novel example resets all history and ICA related components.

        Parameters
        ----------
        n_clicks : int
            Number of times the preprocess button has been clicked.
        data_path : str
            Path to the raw M/EEG file or directory.
        resample_freq : float
            Target sampling rate in Hz.
        high_pass_freq : float
            Lower bound of the bandpass filter in Hz.
        low_pass_freq : float
            Upper bound of the bandpass filter in Hz.
        notch_freq : float
            Frequency to be removed by the notch filter (e.g., 50 or 60 Hz).
        eeg_reference : {"average", "double_banana_bipolar"}
            EEG re-referencing scheme to apply. Ignored for MEG recordings.
        heartbeat_ch_name : str
            Name of the ECG/heartbeat channel for artifact detection.
        bad_channels : list of str
            List of channel names to be excluded from processing.

        Returns
        -------
        tuple
            Updated states for Dash components including status messages, 
            frequency settings, annotations, and navigation path.
        """
        NO_UPDATE = (dash.no_update,) * 12

        if n_clicks > 0:
            try:
                cu.clear_old_cache_files_and_csv(f"{config.CACHE_DIR}")
                raw = dpu.read_raw(
                    data_path, preload=True, verbose=False, bad_channels=None
                )

                modality = dpu.get_raw_modality(raw)

                all_bad_channels = dpu.get_bad_channels(raw, bad_channels)
                if all_bad_channels:
                    raw.drop_channels(all_bad_channels)
                annotations_dict = au.get_annotations_dataframe(
                    raw, heartbeat_ch_name, modality
                )

                # #--- Find .mrk file if data_path is a directory ---
                # annotations_dict = au.get_mrk_annotations_dataframe(
                #     data_path, annotations_dict
                # )

                channels_dict = chu.get_grouped_channels_by_prefix(
                    raw, modality, bad_channels=all_bad_channels
                )
                max_length = pu.get_max_length(raw, resample_freq)
                chunk_limits = pu.update_chunk_limits(max_length)

                freq_data = {
                    "resample_freq": resample_freq,
                    "low_pass_freq": low_pass_freq,
                    "high_pass_freq": high_pass_freq,
                    "notch_freq": pu.get_notch_harmonics(notch_freq, raw.info["sfreq"]),
                    "reference": eeg_reference,
                }

                prep_raw = pu.sort_filter_resample(data_path, freq_data, channels_dict)
                
                for chunk_idx in chunk_limits:
                    start_time, end_time = chunk_idx
                    _, _ = pu.get_preprocessed_dataframe_dask(
                        data_path,
                        freq_data,
                        start_time,
                        end_time,
                        channels_dict,
                        prep_raw=prep_raw,
                    )
                cache_file = pu.get_cache_filename(data_path, freq_data)
                pu.save_mne_sidecar(cache_file, prep_raw)

                display_channels_dict = chu.get_post_reference_channel_groups(
                    prep_raw, modality, eeg_reference, bad_channels=all_bad_channels
                )

                return (
                    "Preprocessed and saved data",
                    freq_data,
                    annotations_dict,
                    channels_dict,
                    display_channels_dict,
                    modality,
                    chunk_limits,
                    "/viz/raw-signal",
                    [],
                    [],
                    None,
                    {}
                )

            except Exception as e:
                return (
                    f"⚠️ Error during preprocessing : {str(e)}",) + NO_UPDATE[1:]

        return NO_UPDATE
