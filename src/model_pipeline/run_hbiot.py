#!/usr/bin/env python3
"""
Prediction function for MEG spike detection.
Can be used as a standalone script or imported as a module.
"""

import os
import logging

import pandas as pd
import lightning as L #type: ignore

from lightning_datamodules.datamodules import PredictionDataModule, SPikeDetector
from utils import load_config

logger = logging.getLogger(__name__)

def extract_paths(model_name):
    """Extract configuration, checkpoint & reference channels path"""
    if os.path.basename(model_name) == "transformer.ckpt":
        config_path = "./utils_biot/config/hparams.yaml"
        checkpoint_path = model_name
        reference_channels_path = "./utils_biot/config/reference_channels.pkl"

    elif os.path.basename(model_name) == "hbiot.ckpt":
        config_path = "./utils_biot/config_hbiot/hparams.yaml"
        checkpoint_path = model_name
        reference_channels_path = "./utils_biot/config_hbiot/reference_channels.pkl"

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
        checkpoint_path, config=config, input_shape=input_shape, log_dir=None
    )

    return model

def process_predictions(predictions, adjust_onset):
    """
    Process raw model predictions and metadata into a list of formatted results.

    This function iterates through batches of model outputs, handles various
    tensor shapes (1D to 3D), filters overlapping windows, and extracts
    event-based timings (onsets) and probabilities.

    Parameters
    ----------
    predictions : list of dict
        A list where each element is a dictionary containing:
        - "probs" (torch.Tensor): The raw probability scores. 
          Supported shapes: (n_windows,), (batch, n_windows), or 
          (batch, n_windows, n_classes).
        - "metadata" (list of dict, optional): List of metadata for each 
          sample in the batch. Must contain "window_times" and "n_windows".
        - "mask" (torch.Tensor, optional): Boolean or binary mask to skip 
          specific windows.
    adjust_onset : bool
        If True, the "onset" time is set to the 'peak_time' from metadata.
        If False, the "onset" time is set to the 'center_time'.

    Returns
    -------
    results : list of dict
        A list of detected events, where each dictionary contains:
        - "onset" (float): The calculated timestamp of the event.
        - "duration" (int): Hardcoded to 0, representing point events.
        - "probas" (float): The probability score for that specific window.
    """
    results = []
    assert predictions is not None, "No predictions returned from the model."

    for batch_predictions in predictions:
        if not isinstance(batch_predictions, dict):
            continue
        batch_metadata = batch_predictions.get("metadata", [])
        probs = batch_predictions["probs"]
        mask = batch_predictions.get("mask", None)

        # Handle different tensor shapes
        if probs.dim() == 1:
            # Single sample case: reshape to [1, n_windows]
            probs = probs.unsqueeze(0)
        elif probs.dim() == 3:
            # Remove last dimension if it's singleton (n_classes=1)
            probs = probs.squeeze(-1)

        # Process each sample in the batch
        batch_size = probs.shape[0]
        for i in range(batch_size):
            if i >= len(batch_metadata):
                continue

            metadata = batch_metadata[i]

            if "window_times" in metadata:
                # Chunked prediction using unified metadata naming
                window_times = metadata["window_times"]
                n_windows = metadata["n_windows"]  # Unified naming convention

                # Get probabilities for this sample
                # Assume binary classification: squeeze to 1D array
                sample_probs = probs[i].squeeze()
                sample_mask = mask[i] if mask is not None and mask.ndim == 2 else None

                for j in range(n_windows):
                    # Skip odd indexes due to 50% overlap
                    if j % 2 == 1:
                        continue

                    if j >= len(window_times):
                        continue

                    # Skip masked windows
                    if sample_mask is not None and sample_mask[j] == 0:
                        continue

                    seg_time = window_times[j]
                    prob = (
                        float(sample_probs[j].item())
                        if sample_probs.ndim == 1
                        else float(sample_probs[j, 0].item())
                    )  # Assuming binary classification

                    # Calculate onset (use GFP peak for precise timing if requested)
                    onset = (
                        seg_time["center_time"]
                        if not adjust_onset
                        else seg_time["peak_time"]
                    )
                    results.append(
                        {
                            "onset": onset,
                            "duration": 0,  # Duration is 0 for point events
                            "probas": prob,
                        }
                    )
    return results

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
    channel_groups=None,
    signal_name=None,
) -> str:
    """
    Predict spikes in a MEG file.
    """
    # Extract data and configuration files
    config_path, checkpoint_path, reference_channels_path = extract_paths(model_name)
    config = load_config(config_path)
    model = load_model(checkpoint_path, config)

    # Create trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # Create prediction datamodule
    prediction_config = {
        "signal_path": signal_cache_path,
        "mne_info_path": mne_info_cache_path,
        "modality": "MEG",
        "reference_channels_path": reference_channels_path,
        "dataset_config": config["data"][config["data"]["name"]]["dataset_config"],
        "dataloader_config": config["data"][config["data"]["name"]][
            "dataloader_config"
        ],
    }

    datamodule = PredictionDataModule(**prediction_config)
    datamodule.setup(stage="predict")

    # Run prediction
    predictions = trainer.predict(model, datamodule=datamodule)

    # Process predictions
    results = process_predictions(predictions, adjust_onset)

    # Save predictions to CSV
    return save_predictions(output_path, signal_name, model_name, pd.DataFrame(results))
