import copy
import os
import pandas as pd
import numpy as np
import lightning as L #type: ignore
from scipy.signal import find_peaks

from lightning_datamodules.datamodules import PredictionDataModule, SPikeDetector
from utils import load_config


_PREDICTION_FILTER_OVERRIDES = {"l_freq": None, "h_freq": None, "notch_freq": 0}


def get_callback_params(config, callback_name):
    """Return the parameter dict of a named callback entry from the config."""
    for entry in config.get("callbacks", []):
        if entry.get("name") == callback_name:
            return entry.get(callback_name, {})
    return {}


def resolve_dataset_config(train_dataset_config):
    """Build the prediction dataset_config from the training one.

    The montage, sampling rate, normalization and windowing are kept verbatim;
    only the band-pass / notch filters are disabled because the signal is
    already filtered upstream (see ``apply_standard_filters``).
    """
    cfg = copy.deepcopy(train_dataset_config)
    cfg.update(_PREDICTION_FILTER_OVERRIDES)
    return cfg

def extract_paths(model_name):
    """Extract configuration, checkpoint & reference channels path"""
    if os.path.basename(model_name) == "uiednet.ckpt":
        config_path = "./utils_uiednet/config/hparams.yaml"
        checkpoint_path = model_name
        reference_channels_path = "./utils_uiednet/config/reference_channels.pkl"

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if reference_channels_path is not None and not os.path.exists(
        reference_channels_path
    ):
        raise FileNotFoundError(
            f"Reference channels file not found: {reference_channels_path}"
        )

    return config_path, checkpoint_path, reference_channels_path

def load_model(checkpoint_path, config):
    """Load Model from checkpoint"""
    input_shape = tuple(config["model"][config["model"]["name"]]["input_shape"])

    model = SPikeDetector.load_from_checkpoint(
        checkpoint_path, config=config, input_shape=input_shape, log_dir=None,
        weights_only=False, strict=False
    )

    return model

def extract_peaks_from_mask(
    frequency_sampling: float,
    mask_1d: np.ndarray,
    min_distance_ms: float,
    min_prominence: float,
) -> np.ndarray:
    """
    Extract peaks from a 1D masks using min prominence param
    """
    min_distance_samples = max(1, int(min_distance_ms / 1000.0 * frequency_sampling))

    kwargs = dict(
        prominence=min_prominence,
        distance=min_distance_samples,
    )

    peaks, _ = find_peaks(mask_1d, **kwargs)

    return peaks

def predict_file(
    predictions: list,
    model,
    frequency_sampling: int = 256,
    window_overlap: float = 0.5,
    min_distance_ms: float = 100.0,
) -> list[dict]:
    """Turn raw UIEDNET predictions into a list of detected IED events.

    Each positive window contributes its segmentation probability trace to a
    recording-long signal; IED onsets are then the peaks of that signal
    (``min_prominence`` from the model, ``min_distance_ms`` from the config).

    Returns a list of ``{"onset", "duration", "probas"}`` dicts, the format the
    software consumes downstream (same as ``run_hbiot``).
    """
    assert predictions is not None, "No predictions returned from the model."

    min_prominence = model.min_prominence

    seg_by_window: dict[int, np.ndarray] = {}
    for batch_predictions in predictions:
        if not isinstance(batch_predictions, dict):
            continue

        seg_probs = batch_predictions.get("seg_probs")
        if seg_probs is None:
            continue

        batch_metadata = batch_predictions.get("metadata", [])
        preds = batch_predictions["predictions"]
        mask = batch_predictions.get("window_mask", batch_predictions.get("mask"))

        probs = batch_predictions["probs"]
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        elif probs.dim() == 3:
            probs = probs.squeeze(-1)
        batch_size, n_windows = probs.shape[0], probs.shape[1]

        for i in range(batch_size):
            if i >= len(batch_metadata):
                continue

            start_win_idx = batch_metadata[i].get(
                "start_window_idx", batch_metadata[i].get("start_position", 0)
            )
            sample_preds = preds[i]
            sample_mask = (
                mask[i] if mask is not None and getattr(mask, "ndim", 0) == 2 else None
            )
            sample_seg = seg_probs[i].cpu().numpy()

            for j in range(n_windows):
                if sample_mask is not None and sample_mask[j] == 0:
                    continue
                pred = (
                    int(sample_preds[j].item())
                    if hasattr(sample_preds, "__len__")
                    else int(sample_preds.item())
                )
                if pred == 1:
                    seg_by_window[start_win_idx + j] = sample_seg[j]

    if not seg_by_window:
        return []

    T = len(next(iter(seg_by_window.values())))
    window_step = max(1, int(T * (1 - window_overlap)))
    full_length = max(seg_by_window) * window_step + T

    signal = np.zeros(full_length, dtype=np.float32)
    for win_idx, trace in seg_by_window.items():
        start = win_idx * window_step
        end = min(start + T, full_length)
        signal[start:end] = np.maximum(signal[start:end], trace[: end - start])

    peaks = extract_peaks_from_mask(
        frequency_sampling=frequency_sampling,
        mask_1d=signal,
        min_distance_ms=min_distance_ms,
        min_prominence=min_prominence,
    )

    return [
        {
            "onset": float(peak) / frequency_sampling,
            "duration": 0,
            "probas": float(signal[peak]),
        }
        for peak in peaks
    ]

def save_predictions(output_path, signal_name, model_name, df):
    """Save predictions into a CSV file compatible with MNE annotations."""
    if signal_name is not None:
        output_file = os.path.join(
            output_path, f"{os.path.basename(model_name)}_{signal_name}_predictions.csv"
        )
    else:
        output_file = os.path.join(
            output_path, f"{os.path.basename(model_name)}_predictions.csv"
        )
    df.to_csv(output_file, index=False)
    return output_file


def test_model(
    model_name,
    output_path,
    signal_cache_path,
    mne_info_cache_path,
    adjust_onset=True,
    signal_name=None,
    channel_groups=None,
) -> str:
    """
    Predict spikes in an EEG file using UIEDNET model
    """
    config_path, checkpoint_path, reference_channels_path = extract_paths(model_name)
    config = load_config(config_path)
    model = load_model(checkpoint_path, config)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    prediction_config = {
        "signal_path": signal_cache_path,
        "mne_info_path": mne_info_cache_path,
        "modality": "EEG",
        "reference_channels_path": reference_channels_path,
        "dataset_config": resolve_dataset_config(
            config["data"][config["data"]["name"]]["dataset_config"]
        ),
        "dataloader_config": config["data"][config["data"]["name"]][
            "dataloader_config"
        ],
    }

    datamodule = PredictionDataModule(**prediction_config)
    datamodule.setup(stage="predict")

    predictions = trainer.predict(model, datamodule=datamodule)

    # adjust_onset is not used here: UIEDNET onsets come from the segmentation peak itself.
    results = predict_file(
        predictions=predictions,  #type: ignore
        model=model,
        frequency_sampling=config["data"][config["data"]["name"]]["dataset_config"]["sampling_rate"],
        window_overlap=config["data"][config["data"]["name"]]["dataset_config"]["window_overlap"],
        min_distance_ms=get_callback_params(config, "MetricsEvaluationCallback").get("min_distance_ms", 100.0),
    )

    return save_predictions(
        output_path,
        signal_name,
        model_name,
        pd.DataFrame(results, columns=["onset", "duration", "probas"]),
    )