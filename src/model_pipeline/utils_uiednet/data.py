import numpy as np
from typing import Any
from mne.io.base import BaseRaw
import torch # type: ignore
from scipy import stats
import logging
import lightning as L # type: ignore
import pydantic
import mne
import os

from utils import find_gfp_peak_in_window, load_raw_from_parquet

logger = logging.getLogger(__name__)

RAW_READERS = {
    ".ds": mne.io.read_raw_ctf,
    ".fif": mne.io.read_raw_fif,
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
}


class MontageChannel(pydantic.BaseModel):
    """Represents a single bipolar montage channel (ch1 - ch2)."""
    index: int
    name: str
    channel_1: str
    channel_2: str


DOUBLE_BANANA_BIPOLAR: list[MontageChannel] = [
    # Left temporal chain
    MontageChannel(index=0,  name="Fp1-F7", channel_1="Fp1", channel_2="F7"),
    MontageChannel(index=1,  name="F7-T3",  channel_1="F7",  channel_2="T3"),
    MontageChannel(index=2,  name="T3-T5",  channel_1="T3",  channel_2="T5"),
    MontageChannel(index=3,  name="T5-O1",  channel_1="T5",  channel_2="O1"),
    # Right temporal chain
    MontageChannel(index=4,  name="Fp2-F8", channel_1="Fp2", channel_2="F8"),
    MontageChannel(index=5,  name="F8-T4",  channel_1="F8",  channel_2="T4"),
    MontageChannel(index=6,  name="T4-T6",  channel_1="T4",  channel_2="T6"),
    MontageChannel(index=7,  name="T6-O2",  channel_1="T6",  channel_2="O2"),
    # Central/transverse chain
    MontageChannel(index=8,  name="Fz-Cz",  channel_1="Fz",  channel_2="Cz"),
    MontageChannel(index=9,  name="Cz-Pz",  channel_1="Cz",  channel_2="Pz"),
    # Left parasagittal chain
    MontageChannel(index=10, name="Fp1-F3", channel_1="Fp1", channel_2="F3"),
    MontageChannel(index=11, name="F3-C3",  channel_1="F3",  channel_2="C3"),
    MontageChannel(index=12, name="C3-P3",  channel_1="C3",  channel_2="P3"),
    MontageChannel(index=13, name="P3-O1",  channel_1="P3",  channel_2="O1"),
    # Right parasagittal chain
    MontageChannel(index=14, name="Fp2-F4", channel_1="Fp2", channel_2="F4"),
    MontageChannel(index=15, name="F4-C4",  channel_1="F4",  channel_2="C4"),
    MontageChannel(index=16, name="C4-P4",  channel_1="C4",  channel_2="P4"),
    MontageChannel(index=17, name="P4-O2",  channel_1="P4",  channel_2="O2"),
]

MONTAGE_REGISTRY: dict[str, list[MontageChannel]] = {
    "double_banana_bipolar": DOUBLE_BANANA_BIPOLAR,
}

MISSING_REFERENCE_CHANNELS = {"A1", "A2", "Oz", "Fpz"}


def get_bipolar_montage(
    montage: list[MontageChannel], use_missing_references: bool = True
) -> list[MontageChannel]:
    """Return a bipolar montage, optionally excluding derivations referencing A1/A2.

    When ``use_missing_references`` is False, derivations involving an ear channel
    (e.g. ``A1-T3``, ``T4-A2``) are dropped and the remaining channels are
    re-indexed contiguously from 0, preserving their relative order.
    """
    if use_missing_references:
        return montage

    kept = [
        mc for mc in montage
        if MISSING_REFERENCE_CHANNELS.isdisjoint((mc.channel_1, mc.channel_2))
    ]
    return [
        mc.model_copy(update={"index": i})
        for i, mc in enumerate(sorted(kept, key=lambda x: x.index))
    ]

class PredictDataset(torch.utils.data.Dataset):
    """Dataset for prediction using sequential chunk/window extraction.

    Returns:
        Tuple of (chunk_data, metadata) with unified metadata convention including
        chunk_onset_sample, chunk_idx, window_times, etc.
    """

    def __init__(
        self,
        file_path: str,
        dataset_config: dict[str, Any],
        good_channels : list[str],
        contextual: bool = True,
        mne_info_path: str | None = None,
    ):
        """Initialize prediction dataset with sequential chunk extraction.

        Args:
            file_path: Path to the EEG file.
            dataset_config: Configuration for data processing.
            good_channels: Channels to use for prediction
            contextual: Using contextual model or not
        """
        self.file_path = file_path
        self.mne_info_path = mne_info_path
        self.dataset_config = dataset_config
        self.good_channels = good_channels
        self.contextual = contextual

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PredictDataset for {file_path}")

        # Load and preprocess the recording once
        self.eeg_data = None
        self.channel_info = None
        self.sampling_rate = None
        self.n_chunks = 0

        self._load_recording()

    def _load_recording(self):
        """Load and preprocess the EEG recording once."""
        try:
            raw, self.eeg_data, self.channel_info = load_and_process_eeg_data(
                self.file_path,
                self.dataset_config,
                good_channels=self.good_channels,
                mne_info_path=self.mne_info_path,
            )

            self.sampling_rate = raw.info['sfreq']
            raw.close()

            self.all_windows = create_windows(
                self.eeg_data,
                self.sampling_rate,
                self.dataset_config['window_duration_s'],
                self.dataset_config.get('window_overlap', 0.0),
            )

            num_context_windows = self.dataset_config['n_windows']
            total_windows = len(self.all_windows)

            if self.contextual:
                self.n_chunks = (total_windows + num_context_windows - 1) // num_context_windows

            self.logger.info(f"Loaded recording: {self.eeg_data.shape[1]} samples, "
                           f"{total_windows} windows, {self.n_chunks} chunks")

        except Exception as e:
            self.logger.error(f"Error loading file {self.file_path}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return number of chunks."""
        if self.contextual:
            return self.n_chunks
        return len(self.all_windows)
    
    def _build_window_time_entry(self, global_idx: int, local_idx: int) -> dict[str, Any]:
        """Compute timing and GFP metadata for a single window.

        Args:
            global_idx: Index of the window across the full recording.
            local_idx:  Index of the window within its parent chunk (0 for non-contextual).

        Returns:
            Dictionary of window timing/GFP metadata.
        """

        window_duration_samples = int(
            self.dataset_config['window_duration_s'] * self.sampling_rate
        )
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        window_start = global_idx * window_step
        window_end = window_start + window_duration_samples
        window_center = window_start + window_duration_samples // 2

        peak_sample, peak_time = find_gfp_peak_in_window(
            self.eeg_data, window_start, window_end, self.sampling_rate # type: ignore
        )

        return {
            'start_sample': int(window_start),                          # type: ignore
            'end_sample': int(window_end),                              # type: ignore
            'center_sample': int(window_center),                        # type: ignore
            'peak_sample': int(peak_sample),                            # type: ignore
            'start_time': float(window_start / self.sampling_rate),     # type: ignore
            'end_time': float(window_end   / self.sampling_rate),       # type: ignore
            'center_time': float(window_center / self.sampling_rate),   # type: ignore
            'peak_time': float(peak_time),
            'window_idx_in_chunk': local_idx,
            'global_window_idx':   global_idx,
        }
    
    def _common_metadata(
        self,
        onset_sample: int,
        offset_sample: int,
        n_windows: int,
        window_times: list,
        chunk_idx: int,
        start_window_idx: int,
        end_window_idx: int,
    ) -> dict[str, Any]:
        """Build the shared metadata dict used by both modes."""
        window_duration_samples = int(
            self.dataset_config['window_duration_s'] * self.sampling_rate
        )
        return {
            'chunk_onset_sample': onset_sample,
            'chunk_offset_sample': offset_sample,
            'chunk_duration_samples': offset_sample - onset_sample,
            'chunk_idx': chunk_idx,
            'start_window_idx': start_window_idx,
            'end_window_idx': end_window_idx,
            'n_windows': n_windows,
            'window_times': window_times,
            'window_duration_s': self.dataset_config['window_duration_s'],
            'window_duration_samples': window_duration_samples,
            'file_name': self.file_path,
            'patient_id': self.file_path.split('/')[-2]
                                     if '/' in self.file_path else 'unknown',
            'USE_REFERENCE_CHANNELS': self.channel_info.get('USE_REFERENCE_CHANNELS', False)
                                     if self.channel_info else False,
            'channel_mask': self.channel_info.get('channel_mask', None)
                                     if self.channel_info else None,
            'selected_channels': self.channel_info.get('selected_channels', [])
                                     if self.channel_info else [],
            'n_selected_channels': len(self.channel_info.get('selected_channels', []))
                                     if self.channel_info else 0,
            'sampling_rate': self.sampling_rate,
            'is_test_set': False,
            'extraction_mode': 'sequential',
            'contextual': self.contextual,
        }
    
    def _getitem_contextual(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a chunk of n_windows windows for contextual models."""
        num_context_windows = self.dataset_config['n_windows']
        window_duration_samples = int(
            self.dataset_config['window_duration_s'] * self.sampling_rate
        )
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        start_window_idx = idx * num_context_windows
        end_window_idx = min(start_window_idx + num_context_windows, len(self.all_windows))
        windows = self.all_windows[start_window_idx:end_window_idx]

        chunk_onset_sample = start_window_idx * window_step
        chunk_offset_sample = (
            chunk_onset_sample
            + len(windows) * window_step
            + (window_duration_samples - window_step)
        )

        window_times = [
            self._build_window_time_entry(global_idx, local_idx)
            for local_idx, global_idx in enumerate(range(start_window_idx, end_window_idx))
        ]

        metadata = self._common_metadata(
            onset_sample=chunk_onset_sample,
            offset_sample=chunk_offset_sample,
            n_windows=len(windows),
            window_times=window_times,
            chunk_idx=idx,
            start_window_idx=start_window_idx,
            end_window_idx=end_window_idx,
        )

        return torch.tensor(windows, dtype=torch.float32), metadata
    
    def _getitem_non_contextual(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a single window for non-contextual models.

        The window is wrapped in a leading dimension of 1 so that downstream
        code can treat both modes uniformly: (n_windows, n_channels, window_samples).
        """
        window_duration_samples = int(
            self.dataset_config['window_duration_s'] * self.sampling_rate
        )
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        window = self.all_windows[idx]          # (n_channels, window_samples)

        wt_entry = self._build_window_time_entry(global_idx=idx, local_idx=0)

        onset_sample  = idx * window_step
        offset_sample = onset_sample + window_duration_samples

        metadata = self._common_metadata(
            onset_sample=onset_sample,
            offset_sample=offset_sample,
            n_windows=1,
            window_times=[wt_entry],
            chunk_idx=idx,          # chunk_idx == window_idx in non-contextual mode
            start_window_idx=idx,
            end_window_idx=idx + 1,
        )

        # unsqueeze(0) → (1, n_channels, window_samples) for shape consistency
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0), metadata

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Dispatch to contextual or non-contextual extraction."""
        if self.contextual:
            return self._getitem_contextual(idx)
        return self._getitem_non_contextual(idx)


def load_and_process_eeg_data(
    file_path: str,
    config: dict[str, Any],
    good_channels: list[str],
    mne_info_path: str | None = None,
) -> tuple[BaseRaw, np.ndarray, dict[str, Any]]:
    """
    Load and process EEG data with comprehensive preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the EEG data file. Currently supports
        FIF format (.fif extension) only. Other formats raise ``ValueError``.
    config : dict of {str: Any}
        Configuration dictionary containing preprocessing parameters:
        
        - 'montage_type' : str
            Either ``'referential'`` (default) or ``'bipolar'``.
        - 'sampling_rate' : float
            Target sampling rate in Hz for resampling (e.g., 250.0).
        - 'l_freq' : float, optional
            High-pass filter cutoff frequency in Hz (default: 0.5).
        - 'h_freq' : float, optional
            Low-pass filter cutoff frequency in Hz (default: 80.0).
        - 'notch_freq' : float, optional
            Fundamental frequency for notch filtering in Hz (default: 50.0).
            Set to 0 or negative to disable. Harmonics up to Nyquist are
            automatically included.
        - 'bipolar_montage' : str, optional
            Key into ``MONTAGE_REGISTRY`` selecting the bipolar derivation set
            (default: ``'double_banana_bipolar'``). Only used when
            ``montage_type='bipolar'``.
        - 'normalization' : dict, optional
            Normalization configuration:
            
            - 'method' : str
                Normalization approach to apply (default is "robust_zscore")
            - 'axis' : int or None
                Axis along which to normalise (``None`` for global).

        - is_transformer : bool, optional
            Controls missing-channel strategy:

                - ``False`` (default): missing channels are **interpolated** via spherical
                splines (referential) or reconstructed before derivation (bipolar),
                yielding a fully-populated, fixed-size output array.
                - ``True``: missing channels are **zero-filled**; their positions in
                ``channel_mask`` are set to ``False``. Use for transformer architectures
                that handle variable channel availability via attention masking.
            
    good_channels : list of str
        Ordered reference list of **referential** channel names. Defines the
        canonical channel ordering and size for the output array in referential
        mode. In bipolar mode this parameter is unused — channel ordering is
        determined by ``MontageChannel.index`` in the montage definition.
        
    Returns
    -------
    raw : mne.io.RawArray
        MNE RawArray after normalisation and soft clipping. Channel set and
        ordering reflect the chosen montage. Annotations from the original
        recording are preserved.
    data : numpy.ndarray, shape (n_channels, n_timepoints)
        Preprocessed EEG array with fixed channel ordering:

        - **Referential**: ``n_channels = max(23, len(good_channels))``.
          Index ``i`` always corresponds to ``good_channels[i]``.
        - **Bipolar**: ``n_channels = max(22, len(montage))``.
          Index ``i`` always corresponds to ``MontageChannel.index == i``.

        Missing or zeroed channels are represented as rows of zeros.
        ``dtype`` matches the input data (typically ``float64``).
    channel_info : dict
        Channel metadata dictionary with the following keys:

        - 'ch_info' : list
            MNE channel info objects (``raw.info['chs']``) for the processed
            channels.
        - 'selected_channels' : list of str
            Names of channels carrying valid (non-zeroed) data, in canonical
            order.
        - 'channel_mask' : torch.Tensor, shape (n_channels,), dtype=torch.bool
            ``True`` for positions with valid data, ``False`` for zero-filled
            positions (missing or zeroed-out channels). Used for attention
            masking in neural networks.
        - 'montage_type' : str
            Either ``'referential'`` or ``'bipolar'``, echoing the config value.
        - 'effective_channels' : list of str
            Full ordered list of channel/derivation names for the chosen montage
            (``good_channels`` for referential, bipolar derivation names for
            bipolar). Includes both valid and missing positions.
    """
    try:
        file_ext = os.path.splitext(str(file_path))[1].lower()
        if file_ext == ".parquet":
            if not mne_info_path or not os.path.exists(str(mne_info_path)):
                raise ValueError(
                    f"Cached parquet signal '{file_path}' requires a companion "
                    f"MNE metadata JSON; got mne_info_path={mne_info_path!r}."
                )
            raw, _ = load_raw_from_parquet(str(file_path), str(mne_info_path))
            # The cache is written via mne ``to_data_frame`` which scales EEG to
            # µV; reconstructing a RawArray from it makes MNE treat those values
            # as volts. Rescale back to volts for a consistent downstream unit.
            raw.apply_function(lambda x: x * 1e-6, picks="all", channel_wise=False)
        else:
            reader = RAW_READERS.get(file_ext)
            if reader is None:
                raise ValueError(
                    f"Unsupported file type '{file_ext}' for {file_path}. "
                    f"Supported extensions: {list(RAW_READERS.keys()) + ['.parquet']}"
                )
            raw = reader(str(file_path), preload=True, verbose=False)
        raw.pick(picks=["eeg"])

        montage_type: str = config.get("montage_type", "referential")
        is_transformer: str = config.get("is_transformer", False)

        if montage_type not in {"referential", "bipolar"}:
            raise ValueError(
                f"Unknown montage_type '{montage_type}'. "
                f"Expected 'referential' or 'bipolar'."
            )
        
        # Check for invalid values at the end of the recording
        raw_data_temp = raw.get_data()
        is_invalid = ~np.isfinite(raw_data_temp) | (np.abs(raw_data_temp) > 2.0)
        invalid_indices = np.where(is_invalid.any(axis=0))[0]

        if len(invalid_indices) > 0:
            first_nan_idx = invalid_indices[0]
            t_stop = (first_nan_idx / raw.info['sfreq']) - 0.1
            
            if t_stop > 0:
                raw.crop(tmin=0, tmax=t_stop)
            else:
                raise ValueError(f"File {file_path} starts with NaNs. Skipping.")

        # Resample and filter signal
        apply_standard_filters(raw, config)

        # Montage definition & channel interpolation if needed
        if montage_type == "bipolar":
            bipolar_montage_key = config.get("bipolar_montage", "double_banana_bipolar")
            base_montage = MONTAGE_REGISTRY.get(bipolar_montage_key)
            if base_montage is None:
                raise ValueError(
                    f"Unknown bipolar montage '{bipolar_montage_key}'. "
                    f"Available: {list(MONTAGE_REGISTRY.keys())}"
                )
            bipolar_montage_def = get_bipolar_montage(base_montage, config.get("use_missing_references", True))
            bipolar_missing_strategy = "zero" if is_transformer else "interpolate"
            raw = apply_bipolar_montage(
                    raw,
                    bipolar_montage_def,
                    missing_channel_strategy=bipolar_missing_strategy,
                )
            effective_channels = [mc.name for mc in sorted(bipolar_montage_def, key=lambda x: x.index)]

            logger.info(
                f"Bipolar montage '{bipolar_montage_key}' applied "
                f"({len(bipolar_montage_def)} derivations, "
                f"missing_strategy='{bipolar_missing_strategy}')."
            )

        else:
            if not is_transformer and len(raw.ch_names) < len(good_channels):
                raw = interpolate_missing_channels(raw, good_channels)
            raw.set_eeg_reference("average", projection=False)
            effective_channels = good_channels

        # Normalize extracted data if scop is global (skip if scop is window)
        info = raw.info.copy()
        raw_data = np.array(raw.get_data())  # Shape: (n_selected_channels, n_timepoints)
        n_timepoints = raw_data.shape[1]


        normalized_data = normalize_data(
            raw_data,
            config.get('normalization', {'method': 'robust_zscore', 'axis': None})
        )
        signal_processed = exponential_soft_clipping(normalized_data, threshold=15, hardness=0.1)

        # Reconstruct a preprocessed mne instance
        raw_clipped = mne.io.RawArray(signal_processed, info)
        raw_clipped.set_annotations(raw.annotations)

        # Define the maximum number of channels depending on selected montage
        if montage_type == "bipolar":
            n_channels = max(mc.index for mc in bipolar_montage_def) + 1
        else:
            n_channels = len(good_channels)

        num_channels = max(n_channels, len(effective_channels))
        data = np.zeros((num_channels, n_timepoints), dtype=raw_data.dtype)
        channel_mask = torch.zeros(num_channels, dtype=torch.bool)

        # Extract bad channels and construct channel ordering
        if montage_type == "bipolar":
            zeroed_derivations = set(raw_clipped.info.get('bads', []))
            processed_data = raw_clipped.get_data()

            for mc in bipolar_montage_def:
                if mc.name not in zeroed_derivations:
                    data[mc.index, :] = processed_data[mc.index, :]
                    channel_mask[mc.index] = True

            selected_channels = [
                mc.name for mc in sorted(bipolar_montage_def, key=lambda x: x.index)
                if mc.name not in zeroed_derivations
            ]
            channel_info = {
                'ch_info': raw_clipped.info['chs'],
                'selected_channels': selected_channels,
            }

            if zeroed_derivations:
                logger.info(
                    f"Bipolar: {len(zeroed_derivations)} derivation(s) zeroed and masked "
                    f"(missing referential channels): {sorted(zeroed_derivations)}"
                )

        # Create index to reorder channels
        else:
            raw_clipped, channel_info = select_channels(raw_clipped, effective_channels)
            raw_data = np.array(raw_clipped.get_data())

            # Create index mapping for efficiency
            effective_channels_index = {ch: i for i, ch in enumerate(effective_channels)}

            # Place each channel's data at its correct position
            for ch_idx, ch_name in enumerate(channel_info["selected_channels"]):
                if ch_name in effective_channels_index:
                    target_idx = effective_channels_index[ch_name]
                    data[target_idx, :] = raw_data[ch_idx, :]
                    channel_mask[target_idx] = True
                else:
                    logger.warning(f"Channel {ch_name} not in good_channels reference - skipping")

        n_valid = channel_mask.sum().item()
        logger.debug(
            f"Channel masking (file: {os.path.basename(file_path)}): "
            f"{n_valid}/{len(good_channels)} valid channels"
            + (" [transformer mode, no interpolation]" if is_transformer else "")
        )

        channel_info['channel_mask'] = channel_mask
        channel_info["montage_type"] = montage_type
        channel_info["effective_channels"] = effective_channels
        
        return raw_clipped, data, channel_info
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise

def apply_standard_filters(raw: BaseRaw, config: dict[str, Any]) -> None:
    """Apply the standard resampling, bandpass, and notch filters in-place.

    Args:
        raw: MNE Raw object (modified in-place).
        config: Configuration dictionary with keys ``sampling_rate``,
            ``l_freq``, ``h_freq``, and ``notch_freq``.
    """
    if raw.info['sfreq'] != config['sampling_rate']:
        raw.resample(sfreq=config['sampling_rate'])

    # In prediction the signal is already band-pass / notch filtered upstream
    # (by the session or by preprocess_same_as_training), so re-filtering here
    # would stack a second pass. The caller signals this by nulling l_freq/h_freq;
    # only the resample above (needed to match the model's window length) is kept.
    if config.get('l_freq') is None and config.get('h_freq') is None:
        return

    raw.filter(
        l_freq=config.get('l_freq', 0.5),
        h_freq=config.get('h_freq', 95.0)
    )

    if config.get('notch_freq', 50.0) > 0:
        freqs = np.arange(config['notch_freq'], config['sampling_rate'] / 2, config['notch_freq']).tolist()
        raw.notch_filter(freqs=freqs)

def exponential_soft_clipping(data, threshold, hardness):
    abs_data = np.abs(data)
    
    mask = abs_data > threshold
    
    #threshold + (1 - exp(-(x-threshold)))
    compressed_data = np.copy(data)
    
    diff = abs_data[mask] - threshold
    compressed_values = threshold + (1 / hardness) * (1 - np.exp(-hardness * diff))
    
    compressed_data[mask] = np.sign(data[mask]) * compressed_values
    
    return compressed_data

def apply_bipolar_montage(
    raw: BaseRaw,
    montage: list[MontageChannel],
    missing_channel_strategy: str = "zero", # "zero" | "interpolate"
) -> BaseRaw:
    """
    Convert a referential MNE Raw object to a bipolar montage.

    Each output channel is the difference (channel_1 - channel_2) defined
    in the montage specification.  Channels referenced in the montage but
    absent from ``raw`` are silently filled with zeros and flagged as bad
    so downstream interpolation can handle them if needed.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Referential EEG recording (preloaded).
    montage : list of MontageChannel
        Ordered bipolar derivation definitions.
     missing_channel_strategy : {'zero', 'interpolate'}
        Strategy when a referential channel required by a derivation is absent:

        * 'zero' - the derivation is set to zero and flagged as bad.
          Fast, no assumption on spatial position, conservative.
        * 'interpolate' - the missing referential channel is reconstructed
          via spherical spline interpolation (standard_1020 montage) before
          computing the derivation. Requires enough neighbouring channels.

    Returns
    -------
    raw_bipolar : mne.io.RawArray
        New Raw object whose channels correspond 1-to-1 with ``montage``.
    """
    data = raw.get_data()                        # (n_ref_ch, n_times)
    ch_index = {ch: i for i, ch in enumerate(raw.ch_names)}
    sfreq = raw.info["sfreq"]
    n_times = data.shape[1]

    missing_refs = list(dict.fromkeys(
        ch
        for mc in montage
        for ch in (mc.channel_1, mc.channel_2)
        if ch not in ch_index
    ))

    if missing_refs and missing_channel_strategy == "interpolate":
        all_required_refs = list(dict.fromkeys(
                ch for mc in montage for ch in (mc.channel_1, mc.channel_2)
            ))
        logger.info(
            f"Bipolar montage: interpolating {len(missing_refs)} missing "
            f"referential channel(s): {missing_refs}"
        )
        raw = interpolate_missing_channels(raw, all_required_refs)

        # Rebuild index after interpolation
        data = raw.get_data()
        ch_index = {ch: i for i, ch in enumerate(raw.ch_names)}
        missing_refs = []

    bipolar_data = np.zeros((len(montage), n_times), dtype=data.dtype)
    bipolar_names: list[str] = []
    zeroed_derivations: list[str] = []

    for mc in montage:
        bipolar_names.append(mc.name)
        i1 = ch_index.get(mc.channel_1)
        i2 = ch_index.get(mc.channel_2)

        if i1 is None or i2 is None:
            zeroed_derivations.append(mc.name)
            # Leave the row as zeros → will be treated as a bad channel
        else:
            bipolar_data[mc.index, :] = data[i1, :] - data[i2, :]

    info = mne.create_info(
        ch_names=bipolar_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(bipolar_names),
    )
    raw_bipolar = mne.io.RawArray(bipolar_data, info, verbose=False)
    raw_bipolar.set_annotations(raw.annotations)
    raw_bipolar.info["bads"] = zeroed_derivations

    return raw_bipolar

def interpolate_missing_channels(raw, good_channels):
    """
    Reconstruct missing EEG channels using spatial spherical spline interpolation.

    This function identifies channels that are present in a reference list but
    absent from the current recording, creates placeholder channels at their
    theoretical electrode positions (based on the standard 10-20 system), and
    uses MNE's spherical spline interpolation to estimate their electrical
    activity from spatially neighboring channels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing partial EEG recordings. Must have been loaded
        with `preload=True` to allow in-place modifications. The object should
        already have EEG channel types set correctly.
    good_channels : list of str
        Complete reference list of expected channel names in standard EEG nomenclature.
        This typically represents the full channel set from your recording system or the channels required by a
        pre-trained model. Channel names must match those in the standard_1020 montage.

    Returns
    -------
    new_raw : mne.io.BaseRaw
        Copy of the input Raw object containing all channels from `good_channels`.
        Channels present in the original recording retain their measured values,
        while previously missing channels contain interpolated estimates. The
        channel order matches `good_channels` exactly. The 'bads' list in info
        still contains the names of interpolated channels for tracking purposes.

    Raises
    ------
    KeyError
        If any channel name in `good_channels` is not found in the standard_1020
        montage. Ensure all channel names follow standard nomenclature.
    ValueError
        If the raw object is not preloaded or if interpolation fails due to
        insufficient spatial information.
    RuntimeError
        If MNE's interpolate_bads method fails (e.g., too few channels for
        reliable interpolation, typically <4 channels).
        
    Notes
    -----
    **Algorithm Steps:**
    
    1. Identify missing channels
    2. Create a copy of the input Raw object to avoid in-place modification
    3. Apply standard_1020 montage to establish 3D electrode positions
    4. For each missing channel:
       
       a. Copy data from an arbitrary existing channel (values will be overwritten)
       b. Rename the copied channel to the missing channel name
       c. Add this placeholder channel to the Raw object
       d. Set its 3D location from the montage (first 3 values of 'loc' field)
       
    5. Reorder all channels to match `good_channels` order
    6. Mark all missing (now placeholder) channels as 'bad'
    7. Run spherical spline interpolation with origin at (0, 0, 0.04) meters
    8. Keep channels marked as 'bad' for tracking (reset_bads=False)
    """
    existing_channels = raw.info['ch_names']
    missing_channels = list(set(good_channels) - set(existing_channels))
    new_raw = raw.copy()

    mont = mne.channels.make_standard_montage('standard_1020')
    new_raw.set_montage(mont)

    # creates fake channels and set them to "bad channels", rename them with the name of the missing channels, 
    # then mne is supposed to be able to reconstruct bad channels with "interpolate_bads" 
    for miss in missing_channels:
        to_copy = raw.info['ch_names'][0]
        new_channel = new_raw.copy().pick([to_copy])
        new_channel.rename_channels({to_copy: miss})
        new_raw.add_channels([new_channel], force_update_info=True)

        #specifies the location of the missing channel
        for i in range(len(new_raw.info['chs'])):
            if new_raw.info['chs'][i]['ch_name'] == miss:
                new_raw.info['chs'][i]['loc'][:3] = mont.get_positions()['ch_pos'][miss]

    new_raw.reorder_channels(good_channels)
    new_raw.info['bads'] = missing_channels

    if np.isnan(new_raw.get_data()).any():
        logger.warning("Detected NaNs before interpolation")

    new_raw.interpolate_bads(origin=(0, 0, 0.04), reset_bads=True)
    if np.isnan(new_raw.get_data()).any():
        logger.warning("Detected NaNs after interpolation")

    return new_raw

def normalize_data(data: np.ndarray, norm_config: dict, eps: float | None = None) -> np.ndarray:
    """Normalize data using specified method."""
    if eps is None:
        eps = norm_config.get('epsilon', 1e-20)
    
    method = norm_config.get('method', 'robust_zscore')
    axis = norm_config.get('axis', None)
    
    if method == 'percentile':
        percentile = norm_config.get('percentile', 95)
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        q = np.percentile(np.abs(data), percentile, axis=axis, keepdims=True)
        return data / (q + eps)
    
    elif method == 'robust_normalize':
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + eps)
    
    elif method == 'robust_zscore':
        median = np.median(data, axis=axis, keepdims=True)
        mad = stats.median_abs_deviation(data, axis=axis, scale='normal')  # type: ignore
        if axis is not None:
            mad = np.expand_dims(mad, axis=axis)
        return (data - median) / (mad + eps)  # type: ignore
    
    elif method == 'zscore':
        return (data - np.mean(data, axis=axis, keepdims=True)) / (np.std(data, axis=axis, keepdims=True) + eps)
    
    elif method == 'minmax':
        min_v = np.min(data, axis=axis, keepdims=True)
        max_v = np.max(data, axis=axis, keepdims=True)
        return (data - min_v) / (max_v - min_v + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Supported methods: percentile, zscore, minmax, robust_normalize, robust_zscore")

def select_channels(raw: BaseRaw, good_channels: list[str]) -> tuple[BaseRaw, dict[str, Any]]:
    """Select channels ensuring consistent ordering across all samples for batch compatibility.

    Args:
        raw: MNE Raw object containing EEG data
        good_channels: ORDERED list of reference channel names (defines canonical ordering)

    Returns:
        Tuple of (processed_raw, channel_info) where channel_info contains:
            - 'loc': Dictionary mapping selected channel names to coordinates (legacy)
            - 'selected_channels': ORDERED list of channel names matching good_channels order
            - 'n_selected': Number of channels actually present in raw data
            - 'n_with_coordinates': Number of channels with coordinate info (legacy)
    """
    logger.debug(f"Raw channels available: {len(raw.ch_names)} channels")
    logger.debug(f"Good channels reference: {len(good_channels)} channels")
    
    available_channels = set(raw.ch_names)  # Use set for O(1) lookup
    logger.debug(f"Available EEG channels: {len(available_channels)}")

    # Ensure batch consistency - all samples have same channel ordering
    selected_channels = [ch for ch in good_channels if ch in available_channels]

    if len(selected_channels) == 0:
        raise ValueError(f"No channels from good_channels found in raw data! "
                        f"Raw has: {list(raw.ch_names)[:10]}..., "
                        f"Expected: {good_channels[:10]}...")

    logger.debug(f"Selected {len(selected_channels)}/{len(good_channels)} channels from reference list")

    # Pick only the selected channels in the raw object
    raw = raw.pick_channels(selected_channels)
    channel_info = {
        'ch_info': raw.info['chs'],  # Full channel info from MNE},
        'selected_channels': selected_channels,
    }
    return raw, channel_info

def create_windows(
    eeg_data: np.ndarray,
    sampling_rate: float,
    window_duration_s: float,
    window_overlap: float,
) -> np.ndarray:
    """Create windows from EEG data.
    
    Args:
        eeg_data: EEG data array (n_channels, n_timepoints)
        sampling_rate: Sampling rate in Hz
        window_duration_s: Duration of each window in seconds
        window_overlap: Overlap between windows (0.0 to 1.0)
        
    Returns:
        Array of windows with shape (n_windows, n_channels, n_samples_per_window)
    """
    window_duration_samples = int(window_duration_s * sampling_rate)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    
    windows = []
    seg_start = 0
    
    while seg_start + window_duration_samples <= eeg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(eeg_data[:, seg_start:seg_end])
        seg_start += window_step
    
    return np.array(windows)