import os

import lightning as L #type: ignore

from lightning_datamodules.datamodules import PredictionDataModule, SPikeDetector
from utils import load_config

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
        weights_only=False
    )

    return model

def test_model(
    model_name,
    output_path,
    signal_cache_path,
    mne_info_cache_path,
    adjust_onset=True,
    signal_name=None,
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
        "dataset_config": config["data"][config["data"]["name"]]["dataset_config"],
        "dataloader_config": config["data"][config["data"]["name"]][
            "dataloader_config"
        ],
    }

    datamodule = PredictionDataModule(**prediction_config)
    datamodule.setup(stage="predict")

    predictions = trainer.predict(model, datamodule=datamodule)

    print(predictions)

    ls