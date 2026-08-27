"""Registry mapping a modality to its prediction dataset builder.

Both modalities share :class:`PredictionDataModule`; only the dataset that is
instantiated in ``setup()`` differs. Each builder accepts the same normalized
kwargs (forwarded by the data module) and picks the ones it needs.
"""

import pickle
from typing import Any, Callable, Dict

import torch  # type: ignore

from model_pipeline.utils_biot.data import PredictDataset as MEGPredictDataset
from model_pipeline.utils_uiednet.data import PredictDataset as EEGPredictDataset


def _load_channels(path: str | None) -> list[str] | None:
    if path is None:
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _build_meg(
    *,
    signal_path: str,
    mne_info_path: str,
    dataset_config: Dict[str, Any],
    reference_channels_path: str | None = None,
    **_ignored: Any,
) -> torch.utils.data.Dataset:
    return MEGPredictDataset(
        signal_path=signal_path,
        mne_info_path=mne_info_path,
        dataset_config=dataset_config,
        reference_channels_path=reference_channels_path,
    )


def _build_eeg(
    *,
    signal_path: str,
    dataset_config: Dict[str, Any],
    reference_channels_path: str | None = None,
    **_ignored: Any,
) -> torch.utils.data.Dataset:
    return EEGPredictDataset(
        file_path=signal_path,
        dataset_config=dataset_config,
        good_channels=_load_channels(reference_channels_path),
        contextual=dataset_config.get('contextual', True),
    )


PredictDatasetBuilder = Callable[..., torch.utils.data.Dataset]

PREDICT_DATASET_BUILDERS: Dict[str, PredictDatasetBuilder] = {
    "MEG": _build_meg,
    "EEG": _build_eeg,
}
