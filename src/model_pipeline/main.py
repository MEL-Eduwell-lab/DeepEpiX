import sys
import ast
import importlib
import os


def run_model_pipeline(
    model_name,
    model_type,
    subject,
    output_path,
    adjust_onset,
    channel_groups,
    signal_cache_path,
    mne_info_cache_path,
    signal_name,
):

    # === Determine module based on model_name ===
    MODEL_MODULES = {
        "model_CNN.keras": "model_pipeline.run_CNN_features_models",
        "model_features_only.keras": "model_pipeline.run_CNN_features_models",
        "transformer.ckpt": "model_pipeline.run_hbiot_meg",
        "hbiot.ckpt": "model_pipeline.run_hbiot_meg",
        "uiednet.ckpt": "model_pipeline.run_uiednet",
        # "new_model": "model_pipeline.run_new_model",  # example
    }

    module_name = MODEL_MODULES.get(os.path.basename(model_name))
    if module_name is None:
        raise ValueError(f"Cannot determine backend for model '{model_name}'")

    if os.path.basename(model_name) in ("hbiot.ckpt", "transformer.ckpt"):
        modality_dir = os.path.basename(os.path.dirname(model_name)).upper()
        if modality_dir == "EEG":
            module_name = "model_pipeline.run_hbiot_eeg"

    model_module = importlib.import_module(module_name)
    test_model = getattr(model_module, "test_model")

    kwargs = dict(
        model_name=model_name,
        output_path=output_path,
        signal_cache_path=signal_cache_path,
        mne_info_cache_path=mne_info_cache_path,
        adjust_onset=adjust_onset,
        channel_groups=channel_groups,
        signal_name=signal_name,
    )

    # === Run the model ===
    return test_model(**kwargs)


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2]  # TensorFlow or Pytorch
    data_path = sys.argv[3]
    results_path = sys.argv[4]
    adjust_onset = str(sys.argv[5]).lower() == "true"  # Bool
    channel_groups = ast.literal_eval(sys.argv[6])
    signal_cache_path = sys.argv[7]
    mne_info_cache_path = sys.argv[8]
    signal_name = sys.argv[9]


    if not os.path.exists(signal_cache_path):
        raise FileNotFoundError(f"Singal file not found: {signal_cache_path}")
    if not os.path.exists(mne_info_cache_path):
        raise FileNotFoundError(f"MNE Info file not found: {mne_info_cache_path}")

    run_model_pipeline(
        model_path,
        model_type,
        data_path,
        results_path,
        adjust_onset,
        channel_groups,
        signal_cache_path,
        mne_info_cache_path,
        signal_name,
    )
