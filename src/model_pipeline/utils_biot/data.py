import numpy as np
from typing import Any
from mne.io.base import BaseRaw
from scipy.ndimage import median_filter
import torch # type: ignore
from scipy import stats
import logging
import lightning as L # type: ignore
import pickle

from model_pipeline.utils import load_raw_from_parquet
from utils import find_gfp_peak_in_window

logger = logging.getLogger(__name__)

class PredictDataset(torch.utils.data.Dataset):
    """Dataset for prediction using sequential chunk extraction.

    Returns:
        Tuple of (chunk_data, metadata) with unified metadata convention including
        chunk_onset_sample, chunk_idx, window_times, etc.
    """

    def __init__(
        self,
        signal_path: str,
        mne_info_path: str,
        dataset_config: dict[str, Any],
        n_channels: int = 275,
        reference_channels_path: str | None = None,
    ):
        """Initialize prediction dataset with sequential chunk extraction.

        Args:
            signal_path: Path to the preprocessed signal file (.parquet).
            mne_info_path: Path to the .json information file
            dataset_config: Configuration for data processing.
            n_channels: Number of MEG channels (default: 275) for consistent input size.
        """
        self.signal_path = signal_path
        self.mne_info_path = mne_info_path
        self.dataset_config = dataset_config
        self.n_channels = n_channels
        if reference_channels_path is not None:
            with open(reference_channels_path, 'rb') as f:
                self.reference_channels = pickle.load(f)
        else:
            self.reference_channels = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PredictDataset for {signal_path}")

        # Load and preprocess the recording once
        self.channel_info = None
        self.n_chunks = 0

        self._load_recording()

    def _load_recording(self):
        """Load and preprocess the MEG recording once."""
        try:
            raw, self.meg_data, self.channel_info = load_and_process_meg_data(
                self.signal_path,
                self.mne_info_path,
                self.dataset_config,
                good_channels=self.reference_channels,
                n_channels=self.n_channels,
            )

            self.sampling_rate = raw.info['sfreq']
            raw.close()

            self.all_windows = create_windows(
                self.meg_data,
                self.sampling_rate,
                self.dataset_config['window_duration_s'],
                self.dataset_config.get('window_overlap', 0.0),
            )

            num_context_windows = self.dataset_config['n_windows']
            total_windows = len(self.all_windows)
            self.n_chunks = (total_windows + num_context_windows - 1) // num_context_windows

            print(f"Loaded recording: {self.meg_data.shape[1]} samples, "
                           f"{total_windows} windows, {self.n_chunks} chunks")

        except Exception as e:
            self.logger.error(f"Error loading file {self.signal_path}: {e}")
            raise

    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Extract a chunk sequentially for prediction.

        Args:
            idx: Chunk index (0-based).

        Returns:
            Tuple of (chunk_data, metadata) with chunk_data as tensor of shape
            (n_windows, n_channels, window_samples) and metadata dictionary.
        """
        num_context_windows = self.dataset_config['n_windows']
        window_duration_samples = int(self.dataset_config['window_duration_s'] * self.sampling_rate)
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        start_window_idx = idx * num_context_windows
        end_window_idx = min(start_window_idx + num_context_windows, len(self.all_windows))

        windows = self.all_windows[start_window_idx:end_window_idx]

        chunk_onset_sample = start_window_idx * window_step

        window_times = []
        for local_idx, global_idx in enumerate(range(start_window_idx, end_window_idx)):
            window_start = global_idx * window_step
            window_end = window_start + window_duration_samples
            window_center = window_start + window_duration_samples // 2

            peak_sample, peak_time = find_gfp_peak_in_window(
                self.meg_data, window_start, window_end, self.sampling_rate
            )

            window_times.append({
                'start_sample': int(window_start),
                'end_sample': int(window_end),
                'center_sample': int(window_center),
                'peak_sample': int(peak_sample),
                'start_time': float(window_start / self.sampling_rate),
                'end_time': float(window_end / self.sampling_rate),
                'center_time': float(window_center / self.sampling_rate),
                'peak_time': float(peak_time),
                'window_idx_in_chunk': local_idx,
                'global_window_idx': global_idx,
            })

        metadata = {
            'chunk_onset_sample': chunk_onset_sample,
            'chunk_offset_sample': chunk_onset_sample + len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_duration_samples': len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_idx': idx,
            'start_window_idx': start_window_idx,
            'end_window_idx': end_window_idx,
            'n_windows': len(windows),
            'window_times': window_times,
            'window_duration_s': self.dataset_config['window_duration_s'],
            'window_duration_samples': window_duration_samples,
            'signal_path': self.signal_path,
            'channel_mask': self.channel_info.get('channel_mask', None) if self.channel_info else None,
            'selected_channels': self.channel_info.get('selected_channels', []) if self.channel_info else [],
            'n_selected_channels': len(self.channel_info.get('selected_channels', [])) if self.channel_info else 0,
            'USE_REFERENCE_CHANNELS': self.channel_info.get('USE_REFERENCE_CHANNELS', False) if self.channel_info else False,
            'sampling_rate': self.sampling_rate,
            'is_test_set': False,
            'extraction_mode': 'sequential',
        }

        return torch.tensor(windows, dtype=torch.float32), metadata
    
def load_and_process_meg_data(
    signal_cache_path: str,
    mne_info_cache_path: str,
    config: dict[str, Any],
    good_channels: list[str] | None = None,
    n_channels: int = 275,
    close_raw: bool = True
) -> tuple[BaseRaw, np.ndarray, dict[str, Any]]:
    """Load and process MEG data for prediction.
    
    Args:
        file_path: Path to the MEG data file.
        config: Configuration dictionary with preprocessing parameters.
        good_channels: List of channels that should be present. If None, use all available channels (useful for inference on new systems).
        n_channels: Number of MEG channels to use (default: 275) to enforce consistent input size
        close_raw: Whether to close the MNE Raw object after processing to free memory.
            
    Returns:
        Tuple containing:
            - raw: MNE Raw object after processing.
            - data: Processed MEG data array (n_channels, n_timepoints).
            - channel_info: loc information and channel mask.
    """
    USE_REFERENCE_CHANNELS = False
    try:
        raw, _ = load_raw_from_parquet(signal_cache_path, mne_info_cache_path)

        if raw.info['sfreq'] != config['sampling_rate']:
            raw.resample(sfreq=config['sampling_rate'])

        if good_channels is None or not USE_REFERENCE_CHANNELS:
            good_channels = list(raw.ch_names)  # Use all available channels if no reference provided
            # sample n_channels from good_channels if more than n_channels are available

            if len(good_channels) > n_channels:
                good_channels = good_channels[:n_channels]

        # Select channels based on good channels and location information
        raw, channel_info = select_channels(raw, good_channels)

        # Get raw data from MNE (in order of selected_channels)
        raw_data = np.array(raw.get_data())  # Shape: (n_selected_channels, n_timepoints)
        n_timepoints = raw_data.shape[1]

        # Now normalize and filter
        raw_data = normalize_data(raw_data, config.get('normalization', {'method': 'robust_zscore', 'axis': None}))

        if config.get('median_filter_temporal_window_ms', 0) > 0:
            raw_data = apply_median_filter(raw_data, config['sampling_rate'], config['median_filter_temporal_window_ms'])

        if close_raw:
            raw.close()
        
        # Reorder data to match good_channels exactly
        # This ensures all samples in batch have data at same positions
        # Position i in data array ALWAYS represents good_channels[i]
        num_channels = max(n_channels, len(good_channels))
        data = np.zeros((num_channels, n_timepoints), dtype=raw_data.dtype)
        channel_mask = torch.zeros(num_channels, dtype=torch.bool)

        # Create index mapping for efficiency
        good_channels_index = {ch: i for i, ch in enumerate(good_channels)}

        # Place each channel's data at its correct position
        for ch_idx, ch_name in enumerate(channel_info['selected_channels']):
            if ch_name in good_channels_index:
                target_idx = good_channels_index[ch_name]
                data[target_idx, :] = raw_data[ch_idx, :]
                channel_mask[target_idx] = True
            else:
                logger.warning(f"Channel {ch_name} not in good_channels reference - skipping")

        # Store channel mask for batch collation
        channel_info['channel_mask'] = channel_mask
        channel_info['USE_REFERENCE_CHANNELS'] = USE_REFERENCE_CHANNELS and (good_channels is not None)

        return raw, data, channel_info

    except Exception as e:
        logger.error(f"Error processing {signal_cache_path}: {e}")
        raise


def select_channels(raw: BaseRaw, good_channels: list[str]) -> tuple[BaseRaw, dict[str, Any]]:
    """Select channels ensuring consistent ordering across all samples for batch compatibility.

    Args:
        raw: MNE Raw object containing MEG data
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
    
    # Get available MEG channels from the raw data
    available_channels = set(raw.ch_names)  # Use set for O(1) lookup
    logger.debug(f"Available MEG channels: {len(available_channels)}")

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


def apply_median_filter(data: np.ndarray, sfreq: float, temporal_window_ms: float) -> np.ndarray:
    """Apply median filter with adaptive kernel size based on sampling frequency.
    
    Args:
        data: MEG data array of shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        temporal_window_ms: Temporal smoothing window in milliseconds
        
    Returns:
        Filtered data with same shape as input
    """
    if temporal_window_ms <= 0:
        return data
    
    # Calculate kernel size based on sampling frequency and temporal window
    kernel_samples = int(temporal_window_ms * sfreq / 1000)
    # Ensure odd kernel size for symmetric filtering
    kernel_size = kernel_samples if kernel_samples % 2 == 1 else kernel_samples + 1
    
    # Apply median filter along time axis (axis=1) for each channel
    return median_filter(data, size=(1, kernel_size))


def create_windows(
    meg_data: np.ndarray,
    sampling_rate: float,
    window_duration_s: float,
    window_overlap: float,
) -> np.ndarray:
    """Create windows from MEG data.
    
    Args:
        meg_data: MEG data array (n_channels, n_timepoints)
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
    
    while seg_start + window_duration_samples <= meg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(meg_data[:, seg_start:seg_end])
        seg_start += window_step
    
    return np.array(windows)
