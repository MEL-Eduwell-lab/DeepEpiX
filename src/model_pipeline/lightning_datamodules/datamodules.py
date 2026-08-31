from typing import Any
import os
import torch # type: ignore
import logging
import lightning as L # type: ignore


from utils_biot.models import BIOTClassifier, BIOTHierarchicalClassifier
from model_pipeline.utils_uiednet.model import UIEDNET
from model_pipeline.lightning_datamodules.predict_dataset_registry import (
    PREDICT_DATASET_BUILDERS,
)

logger = logging.getLogger(__name__)


def infer_n_patches_per_channel(checkpoint_path: str) -> int | None:
    """Read the trained ``positional_embedding`` length from a BIOT checkpoint.

    Some checkpoints were trained with an earlier patch-count formula, so the
    value recomputed from ``token_size`` / ``overlap`` / window length no longer
    matches the stored weights. Reading the length straight from the checkpoint
    keeps every model (EEG/MEG, hierarchical or not) loadable.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    for key in (
        "model.window_encoder.positional_embedding",
        "model.biot.positional_embedding",
    ):
        if key in state_dict:
            return state_dict[key].shape[0]
    return None


class PredictionDataModule(L.LightningDataModule):
    """Lightning DataModule for prediction on a single MEG or EEG file.

    The modality-specific dataset is selected via ``modality`` and looked up in
    ``PREDICT_DATASET_BUILDERS``; all downstream logic (shapes, dataloader,
    collate) is shared.

    Note: At inference time, channel selection is handled automatically by the
    PredictDataset based on the available channels in the file.
    """

    def __init__(
        self,
        signal_path: str,
        mne_info_path: str,
        dataset_config: dict[str, Any],
        dataloader_config: dict[str, Any],
        modality: str = "MEG",
        reference_channels_path: str | None = None,
        num_workers_ratio: float = 0.5,
        **kwargs
    ):
        """Initialize prediction data module.

        Args:
            signal_path: Path to the preprocessed signal file
            mne_info_path: Path to the .json information file (MEG only)
            dataset_config: Configuration for data processing
            dataloader_config: Configuration for data loaders
            modality: Which PredictDataset to build ("MEG" or "EEG")
            reference_channels_path: Path to reference channels pickle file
            num_workers_ratio: Ratio of CPU cores to use for workers (default: 0.5)
            **kwargs: Additional parameters for compatibility (unused)
        """
        super().__init__()
        if modality not in PREDICT_DATASET_BUILDERS:
            raise ValueError(
                f"Unknown modality {modality!r}; expected one of "
                f"{sorted(PREDICT_DATASET_BUILDERS)}"
            )
        self.signal_path = signal_path
        self.mne_info_path = mne_info_path
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.modality = modality
        self.reference_channels_path = reference_channels_path
        self.num_workers_ratio = num_workers_ratio

        self.predict_dataset: torch.utils.data.Dataset | None = None
        self.input_shape: torch.Size | None = None
        self.output_shape: torch.Size | None = None
        
    def prepare_data(self):
        """Prepare data - verify file exists."""
        if not os.path.exists(self.signal_path):
            raise FileNotFoundError(f"Preprocessed signal not found: {self.signal_path}")
            
    def setup(self, stage: str | None = None):
        """Set up the prediction dataset."""
        if stage == 'predict' or stage is None:
            self.predict_dataset = PREDICT_DATASET_BUILDERS[self.modality](
                signal_path=self.signal_path,
                mne_info_path=self.mne_info_path,
                dataset_config=self.dataset_config,
                reference_channels_path=self.reference_channels_path,
            )
           
            # Set shapes
            if len(self.predict_dataset) > 0:
                sample = self.predict_dataset[0]
                data = sample[0]  # chunk data
                self.input_shape = data.shape
                self.output_shape = torch.Size([data.shape[0]])  # n_windows

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Create the prediction dataloader with dynamic num_workers."""
        if self.predict_dataset is None:
            raise RuntimeError("Call setup() before getting prediction dataloader")

        predict_config = self.dataloader_config.get('predict', self.dataloader_config.get('test', {})).copy()
        
        # Check that shuffle is False for prediction
        if predict_config.get('shuffle', True):
            predict_config['shuffle'] = False

        if 'num_workers' not in predict_config or predict_config['num_workers'] == 0:
            optimal_workers = get_optimal_num_workers(
                ratio=self.num_workers_ratio,
                min_workers=0,
                max_workers=None
            )
            predict_config['num_workers'] = optimal_workers

        return torch.utils.data.DataLoader(
            self.predict_dataset,
            **predict_config,
            collate_fn=predict_collate_fn,
        )
    
    def get_input_shape(self) -> torch.Size:
        """Get the input shape for model initialization."""
        if self.input_shape is None:
            raise RuntimeError("Call setup() before getting input shape")
        return self.input_shape
    
    def get_output_shape(self) -> torch.Size:
        """Get the output shape for model initialization."""
        if self.output_shape is None:
            raise RuntimeError("Call setup() before getting output shape")
        return self.output_shape

class SPikeDetector(L.LightningModule):
    """Lightning module for spike detection in MEG data.

    This module handles training, validation, and testing of MEG spike detection models.
    All metrics computation and reporting is handled by the MetricsEvaluationCallback.

    Attributes:
        config: Configuration dictionary containing all component settings
        model: The neural network model for spike detection
        loss_fn: The loss function for training
        threshold: Classification threshold for binary predictions (updated by callback)
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_shape: tuple[int, int, int],
        log_dir: str,
        **_kwargs,
    ) -> None:
        """Initialize the Lightning module with configuration.

        Args:
            config: Configuration dictionary containing model, loss, optimizer settings
            input_shape: Shape of the input data (channels, time_points)
            log_dir: Directory for logging
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If required configuration keys are missing
            TypeError: If input_shape is not a tuple
        """
        # Input validation
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got {type(config)}")

        required_keys = ["model", "data", "evaluation"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a tuple of length 3, got {input_shape}"
            )
        super().__init__()
        logger.info("Initializing ConfigurableLightningModule")
        self.config = config
        self.log_dir = log_dir
        self.input_shape = input_shape
        config["model"][config["model"]["name"]]["input_shape"] = list(input_shape)
        n_patches_per_channel = _kwargs.get("n_patches_per_channel")
        if n_patches_per_channel is not None:
            config["model"][config["model"]["name"]]["n_patches_per_channel"] = n_patches_per_channel
        if log_dir is not None:
            config["model"][config["model"]["name"]]["log_dir"] = log_dir
            self.save_hyperparameters(config)

        # Create model and processing flags
        self.contextual = config["model"][config["model"]["name"]].get(
            "contextual", False
        )
        self.segmentation = config["data"]["OnTheFlyDataModule"]["dataset_config"].get("segmentation", False)
        self.sequential_processing = config["model"][config["model"]["name"]].get(
            "sequential_processing", False
        )
        if config["model"]["name"] == "BIOT":
            self.model = BIOTClassifier(**config["model"]["BIOT"])    
        elif config["model"]["name"] == "BIOTHierarchical":
            self.model = BIOTHierarchicalClassifier(
                **config["model"]["BIOTHierarchical"]
            )
        elif config["model"]["name"] == "UIEDNET":
            self.model = UIEDNET(
                **config["model"]["UIEDNET"]
            )
        else:
            raise ValueError(f"Unsupported model name: {config['model']['name']}")

        # Temperature scaling configuration and validation
        self.temperature_scaling_enabled = config["evaluation"].get(
            "temperature_scaling", False
        )
        # Classification threshold (can be updated by MetricsEvaluationCallback if threshold_optimization=True)
        self.clf_threshold = config["evaluation"].get("default_threshold", 0.5)
        self.min_prominence = config["evaluation"].get("default_min_prominence", 0.7)

        # Temperature scaling for calibrated predictions (1.0 = no scaling)
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        self.temperature.requires_grad = (
            False  # Only optimized during temperature scaling phase
        )

        self.epoch_spike_counts = []

    @staticmethod
    def _extract_logits(output) -> torch.Tensor:
        """Extract clf_logits & clf_probs from model output (tensor or dict)."""
        if isinstance(output, dict):
            if 'clf_logits' not in output:
                raise KeyError(f"Dict output missing 'clf_logits'. Got: {list(output.keys())}")
            return output['clf_logits']
        return output
    
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore threshold and temperature from checkpoint if available."""
        super().on_load_checkpoint(checkpoint)
        # Legacy checkpoints trained before ClassificationHead always included an
        # nn.Dropout layer used indices 1/3 (no dropout) instead of 2/5 (with dropout).
        # Remap those keys so strict loading still works regardless of the dropout config.
        state_dict = checkpoint.get("state_dict", {})
        legacy_to_current = {
            ".classifier.clshead.1.": ".classifier.clshead.2.",
            ".classifier.clshead.3.": ".classifier.clshead.5.",
        }
        for legacy, current in legacy_to_current.items():
            for key in list(state_dict.keys()):
                if legacy in key:
                    state_dict[key.replace(legacy, current)] = state_dict.pop(key)
        if "hyper_parameters" in checkpoint:
            if "threshold" in checkpoint["hyper_parameters"]:
                self.clf_threshold = checkpoint["hyper_parameters"]["threshold"]
                print(f"Restored threshold from checkpoint: {self.clf_threshold:.4f}")
            if "clf_threshold" in checkpoint["hyper_parameters"]:
                self.clf_threshold = checkpoint["hyper_parameters"]["clf_threshold"]
                logger.info(f"Restored threshold from checkpoint: {self.clf_threshold:.4f}")

            if "min_prominence" in checkpoint["hyper_parameters"]:
                self.min_prominence = checkpoint["hyper_parameters"]["min_prominence"]
                logger.info(f"Restored min_prominence from checkpoint: {self.min_prominence:.4f}")

            if "temperature" in checkpoint["hyper_parameters"]:
                temp_value = checkpoint["hyper_parameters"]["temperature"]
                if isinstance(temp_value, torch.Tensor):
                    self.temperature.data = temp_value.to(self.temperature.device)
                else:
                    self.temperature.data = torch.tensor(
                        [temp_value], device=self.temperature.device
                    )
                print(
                    f"Restored temperature from checkpoint: {self.temperature.item():.4f}"
                )

    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits for calibrated predictions.

        Temperature scaling divides logits by a learned temperature parameter T:
        - T > 1: Makes predictions less confident (smoother probabilities)
        - T = 1: No scaling (default)
        - T < 1: Makes predictions more confident (sharper probabilities)

        Args:
            logits: Raw model logits [batch_size, n_windows, n_classes] or [batch_size, n_windows]

        Returns:
            Temperature-scaled logits of the same shape
        """
        return logits / self.temperature

    def _stack_window_outputs(
        self, window_outputs: list, batch_size: int, n_windows: int
    ):
        """Stack per-window outputs along window dimension."""
        first = window_outputs[0]
        if isinstance(first, dict):
            return {
                key: torch.stack([o[key] for o in window_outputs], dim=1)
                if isinstance(first[key], torch.Tensor) else first[key]
                for key in first
            }
        # Tensor: (B, n_classes) -> (B, N, n_classes) -> squeeze if n_classes=1
        stacked = torch.stack(window_outputs, dim=1)   # (B, N, n_classes)
        return stacked.squeeze(-1) 

    def _reshape_flat_output(
        self, result: torch.Tensor | dict[str, torch.Tensor], batch_size: int, n_windows: int
    ):
        """Reshape flat (B*N, *) output back to (B, N, *)."""
        if isinstance(result, dict):
            reshaped = {}
            for key, val in result.items():
                if isinstance(val, torch.Tensor) and val.shape[0] == batch_size * n_windows:
                    # (B*N, *extra) -> (B, N, *extra) -> squeeze last dim if 1
                    reshaped[key] = val.view(batch_size, n_windows, *val.shape[1:]).squeeze(-1)
                else:
                    reshaped[key] = val
            return reshaped
        return result.view(batch_size, n_windows, -1).squeeze(-1)

    @staticmethod
    def _extract_seg_mask(metadata: list[dict], device: torch.device) -> torch.Tensor | None:
        """Stack per-sample seg_masks from metadata into a batch tensor.
        
        Returns None if no sample has a seg_mask (segmentation disabled).
        """
        masks = [m.get('seg_mask') for m in metadata]
        
        if all(m is None for m in masks):
            return None
        
        ref_shape = next(m for m in masks if m is not None).shape
        masks = [
            m if m is not None else torch.zeros(ref_shape)
            for m in masks
        ]
        
        return torch.stack(masks, dim=0).to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        window_mask: torch.Tensor | None = None,
        force_sequential: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the model with contextual and sequential processing support.

        Handles different processing modes:
        - Contextual: Pass full sequence [batch_size, n_windows, n_channels, n_timepoints] to model
        - Non-contextual + batch mode: Reshape to [BxN_window, n_channels, n_timepoints]
        - Non-contextual + sequential: Loop through windows individually

        Args:
            x: Input tensor of shape [batch_size, n_windows, n_channels, n_timepoints]
            channel_mask: Optional channel mask tensor (B, C) where True=valid, False=masked.
            window_mask: Optional window mask tensor (B, N) where True=valid, False=masked.
            force_sequential: Whether to force sequential processing mode.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, n_windows, n_classes]
        """
        if self.contextual:
            # Contextual models process the full sequence with temporal context
            return self.model(x, channel_mask, window_mask, *args, **kwargs)

        # Non-contextual processing for window-level models
        batch_size, n_windows, n_channels, n_timepoints = x.shape

        if self.sequential_processing or force_sequential:
            # Sequential mode: Process each window individually in a loop
            window_outputs = [
                self.model(x[:, i], channel_mask, **kwargs)   # (B, *) per window
                for i in range(n_windows)
            ]
            return self._stack_window_outputs(window_outputs, batch_size, n_windows)
        
        else:
            # Batch mode: Reshape to process all windows simultaneously
            x_flat = x.view(batch_size * n_windows, n_channels, n_timepoints)  # [B×N_window, n_channels, n_timepoints]
            if channel_mask is not None:
                kwargs['channel_mask'] = (
                    channel_mask.unsqueeze(1)
                    .expand(-1, n_windows, -1)
                    .reshape(batch_size * n_windows, n_channels)
                )
            if "unknown_mask" in kwargs and kwargs["unknown_mask"] is not None:
                kwargs['unknown_mask'] = (
                    kwargs['unknown_mask'].unsqueeze(1)
                    .expand(-1, n_windows, -1)
                    .reshape(batch_size * n_windows, n_channels)
                )
            result = self.model(x_flat, *args, **kwargs)  # [B×N_window, n_classes] or dict

            return self._reshape_flat_output(result, batch_size, n_windows)

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step.

        Args:
            batch: Batch data (X, window_mask, channel_mask, metadata) where:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata for result export
            batch_idx: Index of the batch

        Returns:
            Dictionary containing predictions, probabilities, and metadata
        """
        X, window_mask, channel_mask, metadata = batch

        if not self.segmentation:

            if not metadata[0]["USE_REFERENCE_CHANNELS"] and channel_mask is not None:
                # Channel mask is actually true everywhere but for padded channels
                # We actually don't know if good channels are really good at inference time, we just know that this is real data
                # So we use an unknown mask that is all True where channel_mask is given
                unknown_mask = torch.ones_like(channel_mask, dtype=torch.bool)

            # Forward pass with batch-aware channel mask
            force_sequential = self.config["model"]["name"] != "BIOTHierarchical"
            output = self.forward(
                X,
                channel_mask=channel_mask,
                window_mask=window_mask,
                unknown_mask=unknown_mask,
                force_sequential=force_sequential,
            )

        else:
            output = self.forward(X, channel_mask=channel_mask, window_mask=window_mask, force_sequential=True)

        # Apply temperature scaling and compute calibrated probabilities
        clf_logits = self.apply_temperature_scaling(self._extract_logits(output))
        probs = torch.sigmoid(clf_logits).cpu().detach()

        # Prepare outputs
        outputs = {
            "logits": clf_logits.cpu().detach(),
            "probs": probs,
            "predictions": (probs >= self.clf_threshold).float(),
            "batch_size": X.shape[0],
            "n_windows": X.shape[1] if len(X.shape) > 2 else 1,
            "batch_idx": batch_idx,
            "metadata": metadata if metadata else {},
            "channel_mask": (
                channel_mask.cpu().detach().float().numpy()
                if channel_mask is not None
                else None
            ),
            "window_mask": (
                window_mask.cpu().detach().float().numpy()
                if window_mask is not None
                else None
            ),
        }
        if isinstance(output, dict) and 'seg_logits' in output:
            outputs['seg_logits'] = output['seg_logits'].cpu().detach()
            outputs['seg_probs'] = output['seg_probs'].cpu().detach()
        return outputs

def predict_collate_fn(batch):
    """Collate function for prediction batches with padding and masking.

    Handles batches with (data, metadata) tuples from PredictDataset.
    Pads variable-length sequences and extracts channel masks from metadata.

    Args:
        batch: List of (data, metadata) tuples from dataset

    Returns:
        Tuple of (batch_data, batch_window_mask, batch_channel_mask, metadata_list)
    """
    # For chunked prediction: (data, metadata) - use padded collate for consistency with training
    data_list = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]

    # Pad data to same length as training (handles variable chunk sizes)
    seg_counts = [d.shape[0] for d in data_list]
    max_segs = max(seg_counts)

    padded_data, window_mask_list = [], []
    channel_mask_list = []

    for i, data in enumerate(data_list):
        n = data.shape[0]
        pad = max_segs - n
        padded_data.append(torch.cat([data, torch.zeros(pad, *data.shape[1:])]))
        window_mask_list.append(torch.cat([torch.ones(n), torch.zeros(pad)]))

        # Extract channel mask from metadata
        if metadata_list and i < len(metadata_list):
            ch_mask = metadata_list[i].get('channel_mask', None)
            if ch_mask is not None:
                if isinstance(ch_mask, list):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                elif not isinstance(ch_mask, torch.Tensor):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                channel_mask_list.append(ch_mask)
            else:
                n_channels = data.shape[1] if len(data.shape) > 1 else 1
                channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))
        else:
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))

    batch_data = torch.stack(padded_data, dim=0)
    batch_window_mask = torch.stack(window_mask_list, dim=0)  # 1=real, 0=padded
    batch_channel_mask = torch.stack(channel_mask_list, dim=0) if channel_mask_list else None

    return batch_data, batch_window_mask, batch_channel_mask, metadata_list

def get_optimal_num_workers(ratio: float = 0.5, min_workers: int = 0, max_workers: int | None = None) -> int:
    """Dynamically determine the optimal number of workers for data loading.

    Args:
        ratio: Conservative ratio to multiply CPU count by (default: 0.5 for 50% of CPUs)
        min_workers: Minimum number of workers (default: 0)
        max_workers: Maximum number of workers (default: None, no limit)

    Returns:
        Optimal number of workers as an integer

    Example:
        # Use 50% of available CPUs
        num_workers = get_optimal_num_workers(ratio=0.5)

        # Use 75% of available CPUs, but at least 2 and at most 8
        num_workers = get_optimal_num_workers(ratio=0.75, min_workers=2, max_workers=8)
    """
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    # Calculate optimal workers with conservative ratio
    optimal_workers = max(min_workers, int(cpu_count * ratio))

    # Apply maximum limit if specified
    if max_workers is not None:
        optimal_workers = min(optimal_workers, max_workers)

    logger.info(f"Dynamically determined num_workers: {optimal_workers} "
                f"(CPU count: {cpu_count}, ratio: {ratio}, "
                f"min: {min_workers}, max: {max_workers})")

    return optimal_workers