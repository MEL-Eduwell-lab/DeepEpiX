import torch
import torch.nn as nn
from x_transformers import Encoder
import logging

def log_tensor_statistics(tensor: torch.Tensor, name: str, logger_obj: logging.Logger | None =None) -> None:
    """Log detailed statistics about a tensor for debugging NaN/inf issues.

    Args:
        tensor: Tensor to analyze
        name: Name/description of the tensor
        logger_obj: Logger to use (defaults to module logger)
    """
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__)
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    n_total = tensor.numel()
    if n_nan > 0 or n_inf > 0:
        logger_obj.error(f'ALERT {name}: NaN={n_nan}/{n_total} ({100 * n_nan / n_total:.2f}%), Inf={n_inf}/{n_total} ({100 * n_inf / n_total:.2f}%)')
    if n_nan == 0 and n_inf == 0:
        logger_obj.debug(f'OK {name}: shape={tuple(tensor.shape)}, mean={tensor.float().mean():.4f}, std={tensor.float().std():.4f}, min={tensor.float().min():.4f}, max={tensor.float().max():.4f}, NaN=0, Inf=0')

@staticmethod
def _compute_n_patches(n_samples: int, patch_size: int, overlap: float) -> int:
    stride = max(1, int(patch_size * (1 - overlap)))
    return (n_samples - patch_size) // stride + 1

def stft(sample: torch.Tensor, n_fft, hop_length) -> torch.Tensor:
    """Compute Short-Time Fourier Transform for spectral patch embedding.

    Transforms raw temporal EEG data into frequency-domain representation for
    spectral processing mode. Uses rectangular window for simple frequency analysis.

    Args:
        sample (torch.Tensor): Input temporal data.
            Shape: (batch_size, 1, time_samples)
        n_fft (int): Number of FFT points, determines frequency resolution.
        hop_length (int): Number of samples between successive frames.

    Returns:
        torch.Tensor: Magnitude spectrogram.
            Shape: (batch_size, freq_bins, time_frames)
    """
    spectral = torch.stft(
        input=sample.squeeze(1),  # from shape (batch_size, 1, ts) to (batch_size, ts)
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.ones(n_fft, device=sample.device),  # this is a rectangular window
        center=False,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(spectral)


class PatchFrequencyEmbedding(nn.Module):
    """Embedding module for spectral domain EEG data processing.

    Transforms STFT frequency representations into fixed-size embeddings for
    transformer processing. Used in spectral mode of BIOT encoder.

    Attributes:
        projection (nn.Linear): Projects frequency vectors to embedding space.
            Input size: n_freq (number of frequency bins)
            Output size: emb_size (embedding dimension)
    """

    def __init__(self, emb_size: int=256, n_freq: int=101):
        """Initialize the patch frequency embedding.

        Args:
            emb_size: Size of the embedding vector.
            n_freq: Number of frequency components.
        """
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)
        self.logger = logging.getLogger(__name__ + '.PatchFrequencyEmbedding')
        self.logger.debug(f'Initialized with emb_size={emb_size}, n_freq={n_freq}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frequency-to-embedding projection.

        Args:
            x (torch.Tensor): STFT magnitude spectrogram.
                Shape: (batch_size, freq_bins, time_frames)

        Returns:
            torch.Tensor: Temporal sequence of frequency embeddings.
                Shape: (batch_size, time_frames, emb_size)
        """
        # Permute to (batch, time, freq) for linear projection along freq dimension
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PatchTimeEmbedding(nn.Module):
    """Embedding module for raw temporal MEG data with overlapping patches.

    Processes raw time-series MEG data by creating overlapping temporal patches
    and projecting them to fixed-size embeddings. Used in raw mode of BIOT encoder.

    PROCESSING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: Raw temporal data (single channel)                               │
    │        Shape: (batch_size, 1, time_samples)                             │
    │        Example: (800, 1, 80) [400ms at 200Hz]                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Remove Channel Dimension: (batch, 1, time) → (batch, time)              │
    │                          Prepare for patch extraction                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Unfold Operation: Create overlapping patches                            │
    │ • patch_size=40, overlap=0.5 → stride=20                                │
    │ • (batch, 80) → (batch, n_patches, 40)                                  │
    │ • Example: (800, 80) → (800, 5, 40)                                     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Linear Projection: (batch, n_patches, patch_size) →                     │
    │                   (batch, n_patches, emb_size)                          │
    │ Each 40-sample patch → 256-dimensional embedding                        │
    └─────────────────────────────────────────────────────────────────────────┘

    EXAMPLE PATCH CALCULATION:
    For 80 samples, patch_size=40, overlap=0.5:
    • stride = 40 x (1 - 0.5) = 20 samples
    • n_patches = (80 - 40) / 20 + 1 = 3 patches
    • Patch positions: [0:40], [20:60], [40:80]

    Attributes:
        patch_size (int): Number of samples per temporal patch.
            Example: 40 (400ms at 200Hz sampling rate)
        overlap (float): Overlap ratio between adjacent patches (0.0-1.0).
            Example: 0.5 (50% overlap for smooth temporal coverage)
        projection (nn.Linear): Projects raw patches to embedding space.
            Input size: patch_size, Output size: emb_size
    """

    def __init__(self, emb_size: int=256, patch_size: int=100, overlap: float=0.0):
        """Initialize the patch time embedding.

        Args:
            emb_size: Size of the embedding vector.
            patch_size: Size of each time patch.
            overlap: Amount of overlap between adjacent patches (0.0-1.0).
        """
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.projection = nn.Linear(patch_size, emb_size)
        stride = int(self.patch_size * (1 - self.overlap))
        self.stride = max(1, stride)
        self.logger = logging.getLogger(__name__ + '.PatchTimeEmbedding')
        self.logger.debug(f'Initialized with emb_size={emb_size}, patch_size={patch_size}, overlap={overlap}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with overlapping patch extraction and embedding.

        PATCH CALCULATION EXAMPLE:
        Input: 80 samples, patch_size=40, overlap=0.5
        • stride = 40 x (1 - 0.5) = 20 samples
        • Patches: [0:40], [20:60], [40:80]
        • Total patches: (80-40)/20 + 1 = 3 patches

        SHAPE TRANSFORMATIONS:
        (batch, 1, time) → (batch, time) → (batch, n_patches, patch_size) → 
        (batch, n_patches, emb_size)

        Args:
            x (torch.Tensor): Raw temporal MEG data for single channel.
                Shape: (batch_size, 1, time_samples)
                Example: (800, 1, 80)
                - 800 = batch_size x n_windows (e.g., 32 x 25)
                - 1 = single channel (processed one at a time)
                - 80 = temporal samples (400ms at 200Hz)

        Returns:
            torch.Tensor: Sequence of temporal patch embeddings.
                Shape: (batch_size, n_patches, emb_size)
                Example: (800, 3, 256)
                - 3 patches with 50% overlap
                - Each patch embedded to 256 dimensions
                - Ready for transformer sequence processing

        Raises:
            ValueError: If input length is less than patch_size
        """
        time_steps = x.shape[2]
        if time_steps < self.patch_size:
            error_msg = f'Input length ({time_steps}) must be >= patch_size ({self.patch_size})'
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        x = x.squeeze(1)
        x = x.unfold(1, self.patch_size, self.stride)
        x = self.projection(x)
        return x


class XTransformerEncoder(nn.Module):
    """
    Wrapper for x-transformers Encoder to replace FullAttentionTransformer.
    
    Provides a drop-in replacement for the current full attention implementation
    with advanced features while maintaining compatibility with BIOT's encoding process.
    
    Key improvements over standard attention:
    - Flash Attention for memory efficiency with long sequences
    - RMSNorm for improved training stability
    - Proper parameter mapping for drop-in replacement
    
    IMPORTANT: This wrapper is designed to work with BIOT's manual positional embeddings.
    RoPE and memory tokens are disabled to avoid conflicts with the existing encoding pipeline.
    
    Architecture Benefits for MEG data:
    - Efficient processing of 321-token sequences per window
    - Maintains exact sequence length for downstream processing
    - Compatible with existing channel and positional embedding strategy
    - Reduced memory usage with Flash Attention
    """

    def __init__(self, dim: int, depth: int, heads: int=8, max_seq_len: int=1024, dim_head: int | None =None, attn_dropout: float=0.0, ff_dropout: float=0.0, use_flash_attn: bool=True, use_rmsnorm: bool=True, **kwargs):
        """
        Initialize the X-Transformers Encoder wrapper with BIOT-compatible settings.
        
        Args:
            dim: Embedding dimension size
            depth: Number of transformer layers  
            heads: Number of attention heads
            max_seq_len: Maximum sequence length (matches BIOT's calculated sequence length)
            dim_head: Dimension per attention head (default: dim // heads)
            attn_dropout: Attention dropout probability
            ff_dropout: Feed-forward dropout probability
            use_flash_attn: Whether to use Flash Attention for memory efficiency
            use_rotary_pos_emb: DISABLED - BIOT uses manual positional embeddings
            use_rmsnorm: Whether to use RMSNorm for improved training stability
            **kwargs: Additional arguments for compatibility with LinearAttentionTransformer
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        if dim_head is None:
            dim_head = dim // heads
        xtransformer_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['causal', 'ff_chunks', 'attn_layer_dropout', 'local_heads', 'local_window_size']:
                xtransformer_kwargs[key] = value
        self.encoder = Encoder(dim=dim, depth=depth, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_flash=use_flash_attn, use_rmsnorm=use_rmsnorm, rotary_pos_emb=False, rel_pos_bias=False, **xtransformer_kwargs)
        self.config = {'dim': dim, 'depth': depth, 'heads': heads, 'max_seq_len': max_seq_len, 'use_flash_attn': use_flash_attn, 'use_rmsnorm': use_rmsnorm, 'attn_dropout': attn_dropout, 'ff_dropout': ff_dropout}

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None =None, **kwargs) -> torch.Tensor:
        """
        Forward pass through the x-transformers encoder with BIOT compatibility.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
                For BIOT: (batch_size, 321, emb_size) where 321 = 1 CLS + 320 patches
            mask: Optional attention mask of shape (batch_size, seq_len)
                Compatible with BIOT's input_mask parameter
            **kwargs: Additional arguments for compatibility with LinearAttentionTransformer
                
        Returns:
            torch.Tensor: Encoded representations of shape (batch_size, seq_len, dim)
                Exact same shape as input - no memory tokens added
                Compatible with BIOT's downstream processing expectations
        """
        return self.encoder(x, mask=mask)

    def get_config(self) -> dict:
        """Return the configuration used for this encoder."""
        return self.config.copy()


class FullAttentionTransformer(nn.Module):
    """
    Enhanced Full Attention Transformer using x-transformers library.
    
    This replaces the original PyTorch MultiheadAttention implementation with
    state-of-the-art attention mechanisms including Flash Attention, Rotary 
    Positional Embeddings, and memory tokens.
    
    Key improvements:
    - Flash Attention for memory efficiency
    - RMSNorm for improved training stability
    """

    def __init__(self, dim, depth, max_seq_len, heads=8, ff_dropout=0.2, attn_dropout=0.2, use_flash_attn=True, use_rmsnorm=True, **kwargs):
        super().__init__()
        self.transformer = XTransformerEncoder(dim=dim, depth=depth, heads=heads, max_seq_len=max_seq_len, attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_flash_attn=use_flash_attn, use_rmsnorm=use_rmsnorm, **kwargs)

    def forward(self, x, mask=None):
        """
        Forward pass through the enhanced transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            mask: Optional attention mask

        Returns:
            Transformed tensor of same shape as input (excluding memory tokens)
        """
        output = self.transformer(x, mask=mask)
        return output


class BIOTEncoder(nn.Module):
    """Modified BIOT Encoder with multi-representation output for window processing.

    This encoder implements the core window processing logic for the hierarchical BIOT model.
    It processes individual MEG windows through patch embeddings, channel embeddings,
    transformer attention, and multiple output representations.

    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ INPUT: (batch_size x n_windows, n_channels, n_samples_per_window)│
    │        Example: (800, 64, 80) [treating each window independently]          │
    └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 1: Per-Channel Patch Embedding                                        │
    │ • Raw mode: Split temporal data into overlapping patches                    │
    │ • Spectral mode: STFT → frequency domain patches                            │
    │ • Each channel: (1, 80) → (5, 256) [5 patches per channel]                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 2: Channel Token Addition                                             │
    │ • Add learnable channel identity embeddings to each patch                   │
    │ • Enables model to distinguish between different MEG channels               │
    │ • Shape: (BxN, 64x5, 256) = (BxN, 320, 256)                                 │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 3: CLS Token Addition & Positional Encoding                           │
    │ • Prepend learnable CLS token for global window representation             │
    │ • Add positional encodings for temporal patch order                         │
    │ • Shape: (BxN, 321, 256) [1 CLS + 320 patches]                              │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 4: Transformer Processing                                             │
    │ • Linear attention for efficient processing of long sequences               │
    │ • Models inter-channel and intra-temporal dependencies                      │
    │ • Shape: (BxN, 321, 256) → (BxN, 321, 256)                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 5: Multi-Representation Output                                        │
    │ • CLS Token: Global window summary                                         │
    │ • Mean: Average all token emb                                               │
    │ • Max: Max of all token emb                                                 │
    │ • Selected: Top-k most informative patches via attention                    │
    │ • Combined: (BxN, variable, 256) [configurable token types]                │
    └─────────────────────────────────────────────────────────────────────────────┘

    MULTI-REPRESENTATION STRATEGY:
    1. CLS Token: Learned global representation optimized through attention
    2. Pooled Representation: Statistical summary (mean and max) of all patches
    3. Selected Tokens: Attention-weighted selection of most discriminative patches
    
    This approach provides configurable views of window information:
    - Global context (CLS) - if use_cls_token=True
    - Statistical summary (pooled) - if use_mean_pool/use_max_pool=True
    - Local discriminative features (selected) - if n_selected_tokens > 0

    GRADIENT FLOW:
    • CLS token receives gradients from final classification loss
    • Selected tokens receive attention-weighted gradients based on importance
    • Pooled representation receives uniform gradients from all patches
    • Channel embeddings shared across all patches from same channel

    Attributes:
        training (bool): Training mode flag for augmentations
        patch_frequency_embedding (PatchFrequencyEmbedding): Processes spectral domain data
        patch_time_embedding (PatchTimeEmbedding): Processes raw temporal data with overlap
        cls_token (nn.Parameter): Learnable global window representation token
        transformer (Transformer): Linear/full attention transformer for sequence processing
        positional_embedding (nn.Parameter): Learnable positional encodings for patch order
        channel_tokens (nn.Embedding): Learnable channel identity embeddings
        spatial_channel_embedding (SpatialChannelEmbedding): Location-based channel embeddings
        token_selector (nn.Sequential): MLP for scoring patch importance
        cls_proj (nn.Linear): Projects CLS token to final embedding space
        pool_proj (nn.Linear): Projects pooled representation to final embedding space
        selected_proj (nn.Linear): Projects selected tokens to final embedding space
    """

    def __init__(self, emb_size: int=256, heads: int=8, depth: int=4, n_selected_tokens: int=3, selection_temperature: float=1.0, n_channels: int=275, n_samples_per_window: int=40, token_size: int=200, overlap: float=0.0, mode: str='spec', sfreq: float=200.0, linear_attention: bool=True, attn_dropout: float=0.2, ff_dropout: float=0.2, use_cls_token: bool=True, use_mean_pool: int=0, use_max_pool: bool=False, use_min_pool: bool=False, n_patches_per_channel: int | None=None, **kwargs):
        """Initialize the modified BIOT encoder.

        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_selected_tokens: Number of tokens to select and return. (instead of a single CLS token for example)
            use_cls_token: Whether to include CLS token in output.
            use_mean_pool: Whether to include moments in output. It is an integer indicating that the first k moments should be used.
            use_max_pool: Whether to include max pooling token in output.
            use_min_pool: Whether to include min pooling token in output.
            selection_temperature: Temperature for token selection.
            n_samples_per_window: Number of samples in any given raw window.
            n_channels: Number of input channels.
            token_size: Number of FFT points for Spectral mode / Number of samples for Raw mode.
            overlap: Overlap between patches for raw data mode.
            raw: Whether to use raw time-domain processing.
            linear_attention: Whether to use linear attention or full attention.
            reference_coordinates: Mapping from channel names to coordinate positions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + '.ModifiedBIOTEncoder')
        self.n_fft = token_size
        self.hop_length = int(token_size * (1 - overlap))
        self.mode = mode
        self.sfreq = sfreq
        valid_modes = ['raw', 'spec', 'features']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        self.logger.info('Modified BIOT encoder initialized')
        self.logger.info(f'Processing mode: {mode}')
        if mode == 'spec':
            self.patch_embedding = PatchFrequencyEmbedding(emb_size=emb_size, n_freq=self.n_fft // 2 + 1)
        elif mode == 'raw':
            self.patch_embedding = PatchTimeEmbedding(emb_size=emb_size, patch_size=token_size, overlap=overlap)
        else:
            self.logger.warning(f"Invalid mode '{mode}', defaulting to 'raw'")
            self.patch_embedding = PatchTimeEmbedding(emb_size=emb_size, patch_size=token_size, overlap=overlap)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.use_cls_token = use_cls_token
        self.use_mean_pool = use_mean_pool
        self.use_max_pool = use_max_pool
        self.use_min_pool = use_min_pool
        self.n_selected_tokens = n_selected_tokens
        self.selection_temperature = selection_temperature
        n_tokens_per_window = 0
        if use_cls_token:
            n_tokens_per_window += 1
            self.cls_proj = nn.Linear(emb_size, emb_size)
        if use_mean_pool:
            for k in range(1, use_mean_pool + 1):
                setattr(self, f'mean_pool_proj_{k}', nn.Linear(emb_size, emb_size))
                n_tokens_per_window += 1
        if use_max_pool:
            n_tokens_per_window += 1
            self.max_pool_proj = nn.Linear(emb_size, emb_size)
        if use_min_pool:
            n_tokens_per_window += 1
            self.min_pool_proj = nn.Linear(emb_size, emb_size)
        if n_tokens_per_window == 0:
            raise ValueError('At least one token type must be enabled (CLS, mean_pool, max_pool, or selected_tokens > 0)')
        if n_selected_tokens > 0:
            n_tokens_per_window += n_selected_tokens
            self.token_selector = nn.Sequential(nn.Linear(emb_size, emb_size // 2), nn.ReLU(), nn.Linear(emb_size // 2, 1))
            self.selected_proj = nn.Linear(emb_size, emb_size)
        self.n_tokens_per_window = n_tokens_per_window
        self.logger.info(f'Token configuration: CLS={use_cls_token}, Mean={use_mean_pool}, Max={use_max_pool}, Selected={n_selected_tokens}')
        self.logger.info(f'Total tokens per window: {n_tokens_per_window}')
        # A patch count passed explicitly (e.g. read from a checkpoint trained with
        # a different formula) takes precedence over the computed value.
        if n_patches_per_channel is None:
            n_patches_per_channel = _compute_n_patches(n_samples_per_window, token_size, overlap)
        self.n_patches_per_channel = n_patches_per_channel
        n_tokens = n_patches_per_channel * n_channels
        self.logger.info(f'Max sequence length for transformer: {n_tokens + 1}')
        self.transformer = FullAttentionTransformer(dim=emb_size, heads=heads, depth=depth, max_seq_len=n_tokens + 1, attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_flash_attn=True, use_rmsnorm=True)
        self.logger.info(f'Patches per channel: {self.n_patches_per_channel}')
        self.positional_embedding = nn.Parameter(torch.randn(self.n_patches_per_channel, emb_size))
        self.channel_embedding = nn.Parameter(torch.randn(n_channels, emb_size))
        self.unk_channel_embedding = nn.Parameter(torch.randn(1, emb_size))
        self.missing_channel_embedding = nn.Parameter(torch.randn(1, emb_size))
        self.training = True

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None, unk_augment: float=0.0, unknown_mask: torch.Tensor | None =None) -> torch.Tensor:
        """Forward pass with detailed multi-stage processing and shape tracking.

        DATA FLOW:

        Input → Per-Channel Processing → Channel Tokens → CLS Token →
        Positional Encoding → Transformer → Multi-Representation Output

        SHAPE TRANSFORMATIONS:
        (BxN, C, T) → Per-channel: (BxN, 1, T) → (BxN, P, E) →
        Concatenated: (BxN, C*P, E) → With CLS: (BxN, C*P+1, E) →
        Final: (BxN, n_tokens_per_window, E)

        PROCESSING STAGES:
        1. Channel-wise patch embedding (raw or spectral)
        2. Channel identity token addition
        3. CLS token prepending and positional encoding
        4. Transformer attention processing
        5. Multi-representation extraction

        Args:
            x (torch.Tensor): Input MEG window data.
                Shape: (batch_size x n_windows, n_channels, n_samples_per_window)
                Example: (800, 275, 80)
                - 800 = batch_size x n_windows (e.g., 32 x 25)
                - 275 = number of MEG channels
                - 80 = samples per window (400ms at 200Hz)

            loc: Mapping from channel names to positions.
            channel_mask: Channel mask (BxN, C) where 1=valid, 0=masked.
            unk_augment: Percentage of channels to randomly replace with [UNK] token during training (0.0-1.0)
            unknown_mask: Optional mask (BxN, C) where 1=unknown, 0=known. Overrides unk_augment if provided. This is for inference-time unknown channel handling.

        Returns:
            torch.Tensor: Multi-representation window embeddings.
                Shape: (batch_size x n_windows, n_tokens_per_window, emb_size)
                Example: (800, 6, 256) where 6 depends on token configuration
                Contains configurable representations per window based on enabled token types:
                - CLS token (if use_cls_token=True): global window summary
                - Mean pooling (if use_mean_pool>0): k-th moment statistics
                - Max pooling (if use_max_pool=True): max embeddings
                - Min pooling (if use_min_pool=True): min embeddings
                - Selected tokens (if n_selected_tokens > 0): discriminative local features
        """
        batch_size = x.shape[0]
        n_channels = x.shape[1]
        log_tensor_statistics(x, f'BIOTEncoder input (batch_size={batch_size}, n_channels={n_channels})', self.logger)
        
        if channel_mask is not None:
            n_valid_channels = channel_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f'BIOTEncoder channel_mask: avg valid channels={n_valid_channels:.1f}/{n_channels}')
        
        emb_seq = []
        for i in range(n_channels):
            channel_data = x[:, i:i + 1, :]
            if self.mode == 'spec':
                channel_data = stft(channel_data, n_fft=self.n_fft, hop_length=self.hop_length)
            channel_tokens = self.patch_embedding(channel_data)
            if i == 0:
                log_tensor_statistics(channel_tokens, f'BIOTEncoder channel {i} after patch_embedding', self.logger)
                log_tensor_statistics(self.positional_embedding, 'BIOTEncoder positional_embedding', self.logger)
            channel_tokens = channel_tokens + self.positional_embedding[:channel_tokens.size(1)].unsqueeze(0)
            if i == 0:
                log_tensor_statistics(channel_tokens, f'BIOTEncoder channel {i} after adding positional', self.logger)
            emb_seq.append(channel_tokens)
        emb = torch.cat(emb_seq, dim=1)
        log_tensor_statistics(emb, 'BIOTEncoder after patch embedding concatenation', self.logger)
        
        channel_embs = self.channel_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        if channel_mask is None:
            channel_mask = torch.ones((batch_size, n_channels), dtype=torch.bool, device=x.device)
        missing_mask = ~channel_mask
        channel_embs = torch.where(missing_mask.unsqueeze(-1), self.missing_channel_embedding.expand(batch_size, n_channels, -1), channel_embs)
        # Unknown channel augmentation during training
        if self.training and unk_augment > 0.0:
            valid_mask = channel_mask
            aug_mask = torch.rand(batch_size, n_channels, device=x.device) < unk_augment
            unk_mask = valid_mask & aug_mask
            channel_embs = torch.where(unk_mask.unsqueeze(-1), self.unk_channel_embedding.expand(batch_size, n_channels, -1), channel_embs)
        # Unknown channel handling during inference
        elif unknown_mask is not None:
            channel_embs = torch.where(unknown_mask.unsqueeze(-1), self.unk_channel_embedding.expand(batch_size, n_channels, -1), channel_embs)
        
        # Patch-level expansion of channel embeddings
        channel_embs_expanded = channel_embs.unsqueeze(2).expand(-1, -1, self.n_patches_per_channel, -1)
        channel_embs_flat = channel_embs_expanded.reshape(batch_size, -1, channel_embs.size(-1))
        emb = emb + channel_embs_flat
        log_tensor_statistics(emb, 'BIOTEncoder after adding channel embeddings', self.logger)
        
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        emb_with_cls = torch.cat([cls_tokens, emb], dim=1)
        padded_mask = None
        if channel_mask is not None:
            # Create patch-level mask from channel-level mask
            patch_mask = channel_mask.unsqueeze(-1).expand(-1, -1, self.n_patches_per_channel).reshape(batch_size, -1)
            # Add CLS token mask
            padded_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=x.device), patch_mask], dim=1)
        output = self.transformer(emb_with_cls, mask=padded_mask)
        log_tensor_statistics(output, 'BIOTEncoder after transformer', self.logger)
        
        representations = []
        if self.use_cls_token:
            cls_output = self.cls_proj(output[:, 0, :])
            representations.append(cls_output.unsqueeze(1))
        sequence_output = output[:, 1:, :]
        sequence_mask = padded_mask[:, 1:] if padded_mask is not None else None
        
        if self.use_mean_pool:
            if sequence_mask is not None:
                masked_seq = sequence_output * sequence_mask.unsqueeze(-1).float()
                sum_mask = sequence_mask.sum(dim=1, keepdim=True).float().unsqueeze(-1)
                mean = masked_seq.sum(dim=1, keepdim=True) / sum_mask.clamp(min=1e-06)
            else:
                mean = sequence_output.mean(dim=1, keepdim=True)
            for k in range(1, self.use_mean_pool + 1):
                if k == 1:
                    kth_moment = mean.squeeze(1)
                else:
                    centered = sequence_output - mean
                    if sequence_mask is not None:
                        centered = centered * sequence_mask.unsqueeze(-1).float()
                        sum_mask = sequence_mask.sum(dim=1, keepdim=True).float()
                        kth_moment = centered.pow(k).sum(dim=1) / sum_mask.clamp(min=1e-06)
                    else:
                        kth_moment = centered.pow(k).mean(dim=1)
                moment_proj = getattr(self, f'mean_pool_proj_{k}')(kth_moment)
                representations.append(moment_proj.unsqueeze(1))
        
        if self.use_max_pool:
            if sequence_mask is not None:
                masked_seq = sequence_output.masked_fill(~sequence_mask.unsqueeze(-1), float('-inf'))
                max_pool = self.max_pool_proj(masked_seq.max(dim=1).values)
            else:
                max_pool = self.max_pool_proj(sequence_output.max(dim=1).values)
            representations.append(max_pool.unsqueeze(1))
        
        if self.use_min_pool:
            if sequence_mask is not None:
                masked_seq = sequence_output.masked_fill(~sequence_mask.unsqueeze(-1), float('inf'))
                min_pool = self.min_pool_proj(masked_seq.min(dim=1).values)
            else:
                min_pool = self.min_pool_proj(sequence_output.min(dim=1).values)
            representations.append(min_pool.unsqueeze(1))
        
        if self.n_selected_tokens > 0:
            token_scores = self.token_selector(sequence_output).squeeze(-1)
            selection_weights = torch.softmax(token_scores / self.selection_temperature, dim=-1)
            _, topk_indices = torch.topk(selection_weights, self.n_selected_tokens, dim=-1)
            emb_size = sequence_output.size(-1)
            selected_tokens = []
            for i in range(self.n_selected_tokens):
                indices = topk_indices[:, i].unsqueeze(-1).unsqueeze(-1)
                indices = indices.expand(-1, -1, emb_size)
                selected = torch.gather(sequence_output, 1, indices).squeeze(1)
                selected_tokens.append(self.selected_proj(selected).unsqueeze(1))
            selected_tokens = torch.cat(selected_tokens, dim=1)
            representations.append(selected_tokens)
        
        if len(representations) == 1:
            log_tensor_statistics(representations[0], 'BIOTEncoder final output (single token)', self.logger)
            return representations[0]
        combined = torch.cat(representations, dim=1)
        log_tensor_statistics(combined, f'BIOTEncoder final output (combined {len(representations)} token types)', self.logger)
        return combined


class ClassificationHead(nn.Sequential):
    """Module for classification head.

    This module takes the embeddings and produces class probabilities.

    Attributes:
        clshead (nn.Sequential): Sequential module with the classification layers.
    """

    def __init__(self, emb_size: int, n_classes: int, dropout: float | None = None, **kwargs):
        """Initialize the classification head.

        Args:
            emb_size: Size of the input embedding.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.dropout = dropout
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
            nn.Linear(emb_size, emb_size // 2),
            nn.ELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
            nn.Linear(emb_size // 2, n_classes),
        )
        self.logger = logging.getLogger(__name__ + '.ClassificationHead')
        self.logger.debug(f'Initialized with emb_size={emb_size}, n_classes={n_classes}')

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            x: Input embedding tensor
        Returns:
            Class logits.
        """
        return self.clshead(x).squeeze(-1)


class BIOTClassifier(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) Classifier.
    
    This model uses the BIOT encoder for feature extraction and adds a classification head.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        classifier (ClassificationHead): Classification head.
    """

    def __init__(self, emb_size:int=256, heads: int=8, depth: int=4, n_classes: int=1, mode: str='spec', overlap: float=0.0, log_dir: str | None =None, n_selected_tokens: int=3, use_cls_token: bool=True, use_mean_pool: bool=True, use_max_pool: bool=True, classifier_dropout: float | None = None, input_shape: list | tuple | None = None, **kwargs):
        """Initialize the BIOT classifier.

        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_classes: Number of output classes.
            mode: Processing mode ("raw", "spec", or "features").
            overlap: Overlap between patches.
            log_dir: Optional directory for log files. If None, logs to console only.
            n_selected_tokens: Number of tokens to select and return (for multi-token classification).
            use_cls_token: Whether to include CLS token in output.
            use_mean_pool: Whether to include mean pooling token in output.
            use_max_pool: Whether to include max pooling token in output.
            classifier_dropout: Dropout probability for the classification head. None disables dropout.
            input_shape: (n_windows, n_channels, n_samples_per_window). When given, n_channels and
                n_samples_per_window are derived from it, overriding any BIOTEncoder defaults.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + '.BIOTClassifier')
        self.logger.info(f'Initializing BIOT classifier with emb_size={emb_size}, heads={heads}, depth={depth}, n_classes={n_classes}, mode={mode}')
        if input_shape is not None:
            kwargs['n_channels'] = input_shape[1]
            kwargs['n_samples_per_window'] = input_shape[-1]
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, mode=mode, overlap=overlap, n_selected_tokens=n_selected_tokens, use_cls_token=use_cls_token, use_mean_pool=use_mean_pool, use_max_pool=use_max_pool, **kwargs)
        self.n_classes = n_classes
        self.classifier = ClassificationHead(emb_size=emb_size, n_classes=n_classes, dropout=classifier_dropout)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None =None, *args,**kwargs) -> torch.Tensor:
        """Forward pass of the BIOT classifier.

        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            channel_mask: Optional batch-aware mask (B, C) where True=valid, False=masked.

        Returns:
            Logits for each class.
        """
        biot_output = self.biot(x, channel_mask=channel_mask, *args, **kwargs)
        logits = self.classifier(biot_output)
        return logits


class BIOTHierarchicalClassifier(nn.Module):
    """BIOT Hierarchical Encoder for MEG spike detection.

    This encoder implements a sophisticated two-stage hierarchical attention mechanism
    specifically designed for processing long sequences of high-dimensional MEG data:
    
    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │ INPUT: (batch_size, n_windows, n_channels, n_samples_per_window)               │
    │       Default: (B, 25, 275, 80) - 25 windows of 400ms at 200Hz, 275 MEG channels│
    └──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 1: Intra-window Attention (Per window Processing)                    │
    │ • Process each window independently using BIOTEncoder                       │
    │ • Extract multiple representations: CLS + Pooled + Selected tokens (e.g. 6t.)│
    │ • Shape: (BxN, 275, 80) → (B, N, 6, emb_size)                                │
    └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 2: window Positional Encoding                                        │
    │ • Add learnable position embeddings for temporal window order              │
    │ • Shape: (B, N, 6, emb_size) → (B, N, 6, emb_size)                          │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 3: Inter-window Attention                                            │
    │ • Model long-range temporal dependencies between windows                   │
    │ • Shape: (B, Nx6, emb_size) → (B, Nx6, emb_size)                            │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 4: Classification Head with Attention                                 │
    │ • Attention-based aggregation of tokens per window                         │
    │ • Shape: (B, Nx6, emb_size) → (B, N, n_classes)                             │
    └─────────────────────────────────────────────────────────────────────────────┘

    INFORMATION EXTRACTION HIERARCHY:
    1. Local temporal patterns via patch embeddings per channel
    2. Inter-channel interactions through window-level attention
    3. window-level feature extraction via multiple representation types
    4. Long-range temporal context through inter-window attention
    5. Classification via attention-based token aggregation

    Attributes:
        window_encoder (BIOTEncoder): Processes individual windows with intra-attention
        window_pos_embedding (nn.Parameter): Learnable positional embeddings for windows
        inter_window_transformer (Transformer): Models dependencies between windows
        classifier (AttentionClassificationHead): Attention-based classification head
        n_windows (int): Number of temporal windows in input
        n_tokens_per_window (int): Number of tokens extracted per window (CLS + pooled + selected)
    """

    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            window_encoder_depth: int = 4,
            inter_window_depth: int = 4,
            token_size: int = 40,
            overlap: float = 0.5,
            mode: str = "raw",
            linear_attention: bool = True,
            input_shape: tuple[int, int, int] | None = None,
            transformer: dict | None = None,
            token_selection: dict | None = None,
            classifier: dict | None = None,
            log_dir: str | None = None,
            n_classes: int = 1,
            reference_coordinates: dict[str, list[float]] = {},
            max_virtual_batch_size: int = 640,
            n_patches_per_channel: int | None = None,
            **kwargs
    ):
        """Initialize the BIOT hierarchical encoder.

        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            window_encoder_depth: Number of transformer layers for intra-window attention.
            inter_window_depth: Number of transformer layers for inter-window attention.
            token_size: Size of each token window.
            overlap: Overlap percentage between windows.
            mode: Processing mode ("raw", "spec", or "features").
            linear_attention: Whether to use linear attention for transformers.
            input_shape: Shape of the input data non batched (n_windows, n_channels, n_samples_per_window)
            transformer: Parameters for transformer layers.
                Must include 'attn_dropout' and 'ff_dropout' keys.
            token_selection: Parameters for token selection.
                Can include 'n_selected_tokens', 'use_cls_token', 'use_mean_pool', 'use_max_pool' keys.
            classifier: Parameters for the classification head.
            log_dir: Directory for logging (optional).
            n_classes: Number of output classes for classification.
            reference_coordinates: Dictionary mapping channel names to their spatial coordinates.
            max_virtual_batch_size: Maximum virtual batch size (BxN) to process in a single forward pass. Defaults to 640 = 32x20 (tested empirically on h100 with 80GB RAM).
                When BxN exceeds this value, processing is done in chunks. Default: 640.
            **kwargs: Additional keyword arguments for flexibility.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTHierarchicalEncoder")
        self.max_virtual_batch_size = max_virtual_batch_size

        assert input_shape is not None, "Input shape must be provided for the encoder."
        assert transformer is not None, "Transformer parameters must be provided."
        assert token_selection is not None, "Token selection parameters must be provided."
        assert classifier is not None, "Classifier parameters must be provided."

        print(f"Initializing BIOTHierarchicalClassifier with input shape: {input_shape}")
        n_windows, n_channels, n_samples_per_window = input_shape

        self.logger.info(f"Max virtual batch size set to {max_virtual_batch_size}. "
                        f"Expected virtual batch size for input shape: {input_shape[0]} windows = "
                        f"B x {input_shape[0]} = variable virtual batch")

        # Save channel dimensions
        self.n_channels = n_channels
        self.n_windows = n_windows
        
        # Calculate n_tokens_per_window based on token selection configuration
        use_cls_token = token_selection.get("use_cls_token", True)
        use_mean_pool = token_selection.get("use_mean_pool", 1)     # 1: mean, 2: mean+variance, etc.
        use_max_pool = token_selection.get("use_max_pool", True)
        use_min_pool = token_selection.get("use_min_pool", True)
        n_selected_tokens = token_selection.get("n_selected_tokens", 3)
        
        self.n_tokens_per_window = 0
        if use_cls_token:
            self.n_tokens_per_window += 1
        if use_mean_pool:
            self.n_tokens_per_window += use_mean_pool
        if use_max_pool:
            self.n_tokens_per_window += 1
        if use_min_pool:
            self.n_tokens_per_window += 1
        self.n_tokens_per_window += n_selected_tokens
        
        if self.n_tokens_per_window == 0:
            raise ValueError("At least one token type must be enabled in token_selection config")

        # Modified BIOT encoder for window-level processing with configurable tokens
        self.window_encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=window_encoder_depth,
            n_selected_tokens=n_selected_tokens,          # Number of selected tokens per window
            use_cls_token=use_cls_token,                  # Whether to use CLS token
            use_mean_pool=use_mean_pool,                  # Whether to use mean pooling token
            use_max_pool=use_max_pool,                    # Whether to use max pooling token
            use_min_pool=use_min_pool,                    # Whether to use min pooling token
            n_channels=n_channels,                        # Use full channel count
            n_samples_per_window=n_samples_per_window,    # Number of tokens depends on this, token_size, and overlap
            token_size=token_size,
            overlap=overlap,
            mode=mode,                                    # Processing mode: "raw", "spec", or "features"
            linear_attention=linear_attention,            # Use linear attention or full attention for window encoder
            reference_coordinates=reference_coordinates,  # Pass spatial coordinates to window encoder
            n_patches_per_channel=n_patches_per_channel,  # Override computed patch count (e.g. from checkpoint)
        )

        # Learnable embeddings for each window position
        # Each window gets the same positional encoding for all its tokens
        self.window_pos_embedding = nn.Parameter(
            torch.randn(n_windows, emb_size)
        )  # (N, emb_size)

        # Inter-window transformer
        self.logger.info(f"Using {'linear' if linear_attention else 'full'} attention for transformers")
        max_seq_len = n_windows * self.n_tokens_per_window  # Total sequence length for inter-window transformer
        self.inter_window_transformer= FullAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=inter_window_depth,
            max_seq_len=max_seq_len,
            attn_dropout=transformer.get("attn_dropout", 0.2),
            ff_dropout=transformer.get("ff_dropout", 0.2),
            use_flash_attn=True,
            use_rmsnorm=True
        )

        # Classification head - choose based on number of tokens
        self.n_classes = n_classes
        if self.n_tokens_per_window > 1:
            # Multi-token classification with attention aggregation
            self.classifier = AttentionClassificationHead(
                emb_size=emb_size,
                n_classes=n_classes,
                n_tokens_per_window=self.n_tokens_per_window,
                **classifier
            )
        else:
            # Single token classification with simple MLP
            self.classifier = ClassificationHead(emb_size=emb_size, n_classes=n_classes)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None, window_mask: torch.Tensor | None = None, unk_augment: float = 0.0, unknown_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the BIOT hierarchical encoder with detailed shape tracking.

        DETAILED DATA FLOW:
        
        Input → window Processing → Position Encoding → Inter-window Attention → Classification
        
        SHAPE TRANSFORMATIONS:
        (B, N, Nch, Ns) → (B, N, n_tokens_per_window, E) → (B, Nxn_tokens_per_window, E) → (B, N, 1)
        
        Where:
            B = batch_size, N = n_windows, E = emb_size, Ns = n_samples_per_window, Nch = n_channels
            n_tokens_per_window = configurable (CLS + mean_pool + max_pool + selected tokens)
            1 = n_classes (binary spike detection)

        Args:
            x (torch.Tensor): Input MEG data tensor.
                Shape: (batch_size, n_windows, n_channels, n_samples_per_window)
                Default shape: (B, 25, 275, 80)
                - 25 windows of 400ms duration at 200Hz each
                - 275 MEG channels
                - 80 samples per window (at 200Hz sampling rate)
            channel_mask (Optional[torch.Tensor]): Channel mask (B, C) where True=valid, False=padded.
            window_mask (Optional[torch.Tensor]): Window mask (B, N) where True=valid, False=padded.
            unk_augment (float): Probability of augmenting unknown tokens during training.
                Default: 0.0 (no augmentation)
                Only concerns BIOTEncoder stage.
            unknown_mask (Optional[torch.Tensor]): Mask indicating which channels are unknown (B, C).
                Only concerns BIOTEncoder stage for inference time.
        Returns:
            torch.Tensor: Classification logits for each window.
                Shape: (batch_size, n_windows, n_classes)
                Default shape: (B, 25, 1)
                - Binary classification logits for each of 25 windows
                - Values are raw logits (pre-softmax/sigmoid)
        """
        batch_size, n_windows, n_channels, n_samples = x.shape  # (B, N, Nch, Ns)

        # Log input statistics
        log_tensor_statistics(x, f"HBIOT input (B={batch_size}, N={n_windows}, C={n_channels}, S={n_samples})", self.logger)
        if channel_mask is not None:
            n_valid_channels = channel_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f"HBIOT channel_mask: avg valid channels={n_valid_channels:.1f}/{n_channels}")
        if window_mask is not None:
            n_valid_windows = window_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f"HBIOT window_mask: avg valid windows={n_valid_windows:.1f}/{n_windows}")

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 1: Intra-window Processing (Independent window Encoding)
        # Purpose: Extract rich representations from each window independently
        # ═══════════════════════════════════════════════════════════════════════════
        # Reshape for batch processing: treat each window as independent sample
        virtual_batch_size = batch_size * n_windows
        x_reshaped = x.view(virtual_batch_size, n_channels, n_samples)
        # Shape: (B, N, Nch, Ns) → (BxN, Nch, Ns)
        log_tensor_statistics(x_reshaped, "HBIOT after reshaping to process windows independently", self.logger)

        # same for channel mask if provided
        channel_mask_reshaped = None
        if channel_mask is not None:
            channel_mask_reshaped = channel_mask.unsqueeze(1).repeat(1, n_windows, 1)
            # Shape: (B, C) → (B, 1, C) → (B, N, C)
            channel_mask_reshaped = channel_mask_reshaped.view(virtual_batch_size, n_channels)
            # Shape: (B, N, C) → (BxN, C)
        unknown_mask_reshaped = None
        if unknown_mask is not None:
            unknown_mask_reshaped = unknown_mask.unsqueeze(1).repeat(1, n_windows, 1)
            # Shape: (B, C) → (B, 1, C) → (B, N, C)
            unknown_mask_reshaped = unknown_mask_reshaped.view(virtual_batch_size, n_channels)
            # Shape: (B, N, C) → (BxN, C)

        # Process windows through BIOT encoder with optional chunking for large virtual batches
        if virtual_batch_size <= self.max_virtual_batch_size:
            # Fast path: single forward pass for all windows
            self.logger.debug(f"HBIOT processing virtual_batch_size={virtual_batch_size} in single pass")
            window_tokens = self.window_encoder(x_reshaped, channel_mask_reshaped, unk_augment=unk_augment, unknown_mask=unknown_mask_reshaped)
            # Shape: (BxN, Nch, Ns) → (BxN, n_tokens_per_window, emb_size)
        else:
            # Slow path: chunk processing when virtual batch exceeds maximum
            self.logger.info(f"HBIOT chunking virtual_batch_size={virtual_batch_size} into chunks of {self.max_virtual_batch_size}")
            chunks = []
            for chunk_start in range(0, virtual_batch_size, self.max_virtual_batch_size):
                chunk_end = min(chunk_start + self.max_virtual_batch_size, virtual_batch_size)
                self.logger.debug(f"HBIOT processing chunk [{chunk_start}:{chunk_end}]")

                # Extract chunk
                chunk_x = x_reshaped[chunk_start:chunk_end]
                chunk_mask = channel_mask_reshaped[chunk_start:chunk_end] if channel_mask_reshaped is not None else None
                chunk_unknown_mask = unknown_mask_reshaped[chunk_start:chunk_end] if unknown_mask_reshaped is not None else None

                # Process chunk through encoder
                chunk_tokens = self.window_encoder(chunk_x, chunk_mask, unk_augment=unk_augment, unknown_mask=chunk_unknown_mask)
                chunks.append(chunk_tokens)

            # Concatenate all chunks
            window_tokens = torch.cat(chunks, dim=0)
            # Shape: (BxN, n_tokens_per_window, emb_size)
            self.logger.debug(f"HBIOT concatenated {len(chunks)} chunks into shape {window_tokens.shape}")

        log_tensor_statistics(window_tokens, "HBIOT after window_encoder (flat)", self.logger)

        # Reshape back to batch structure with windows
        window_tokens = window_tokens.view(batch_size, n_windows, self.n_tokens_per_window, -1)
        # Shape: (BxN, n_tokens_per_window, emb_size) → (B, N, n_tokens_per_window, emb_size)
        log_tensor_statistics(window_tokens, "HBIOT after reshaping window tokens back", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 2: window Positional Encoding
        # Purpose: Add temporal order information to windows
        # ═══════════════════════════════════════════════════════════════════════════ 
        # Create positional encoding matrix for all tokens in all windows
        window_pos_matrix = self.window_pos_embedding[:n_windows].unsqueeze(1).repeat(1, self.n_tokens_per_window, 1)
        # Shape: (N, emb_size) → (N, 1, emb_size) → (N, n_tokens_per_window, emb_size)
        
        # Add positional encodings to window tokens
        window_tokens = window_tokens + window_pos_matrix.unsqueeze(0)
        # Shape: (B, N, n_tokens_per_window, emb_size) + (1, N, n_tokens_per_window, emb_size) → (B, N, n_tokens_per_window, emb_size)
        log_tensor_statistics(window_tokens, "HBIOT after adding positional encoding", self.logger)


        # Stage 2bis : Masking tokens of padded windows if mask is provided
        token_mask = None
        if window_mask is not None:
            token_mask = window_mask.view(batch_size, n_windows, 1).repeat(1, 1, self.n_tokens_per_window)
            # Shape: (B, N) → (B, N, 1) → (B, N, n_tokens_per_window)
            # Example: (32, 25) → (32, 25, 1) → (32, 25, 6)
            window_tokens = window_tokens * token_mask.unsqueeze(-1).float()
            # Shape: (B, N, n_tokens_per_window, emb_size) * (B, N, n_tokens_per_window, 1) → (B, N, n_tokens_per_window, emb_size)
            log_tensor_statistics(window_tokens, "HBIOT after applying window mask", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 3: Inter-window Attention
        # Purpose: Model long-range temporal dependencies between windows
        # ═══════════════════════════════════════════════════════════════════════════
        # Flatten windows and tokens for sequence processing
        all_embeddings = window_tokens.view(batch_size, n_windows * self.n_tokens_per_window, -1)
        # Shape: (B, N, n_tokens_per_window, emb_size) → (B, Nxn_tokens_per_window, emb_size)

        # Flatten token mask for transformer
        token_mask_flat = None
        if token_mask is not None:
            token_mask_flat = token_mask.view(batch_size, n_windows * self.n_tokens_per_window).bool()
            # Shape: (B, N, n_tokens_per_window) → (B, Nxn_tokens_per_window)
            # Example: (32, 25, 6) → (32, 150)

        # Process through inter-window transformer for temporal context
        # The transformer library expects mask where 1=valid, 0=ignore (same as our convention)
        output_embeddings = self.inter_window_transformer(all_embeddings, mask=token_mask_flat)
        # Shape: (B, Nxn_tokens_per_window, emb_size) → (B, Nxn_tokens_per_window, emb_size)
        log_tensor_statistics(output_embeddings, "HBIOT after inter_window_transformer", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 4: Classification with Masked Attention
        # Purpose: Generate predictions for each window using attention-based aggregation
        # ═══════════════════════════════════════════════════════════════════════════
        logits = self.classifier(output_embeddings, mask=token_mask_flat)
        # Shape: (B, Nxn_tokens_per_window, emb_size) → (B, N, n_classes)
        log_tensor_statistics(logits, "HBIOT final output logits", self.logger)
        return logits


class AttentionClassificationHead(nn.Module):
    """Attention-based classification head for hierarchical token aggregation.

    This module implements an attention mechanism to intelligently aggregate multiple
    tokens per window into a single classification decision. Instead of simple pooling,
    it learns to focus on the most informative tokens for spike detection.

    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: (batch_size, n_windows x n_tokens_per_window, emb_size)          │
    │        Example: (32, 150, 256) [25 windows x 6 tokens each]             │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Reshape: (batch_size x n_windows, n_tokens_per_window, emb_size)        │
    │          Example: (800, 6, 256) [process each window independently]     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Multi-Head Attention:                                                   │
    │ • Query: Learnable classification query (1, 1, emb_size)                │
    │ • Key/Value: All tokens from window (n_tokens_per_window, emb_size)     │
    │ • Output: Attended representation (1, emb_size)                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Classification MLP:                                                     │
    │ • LayerNorm + Dropout + Linear + GELU + Dropout + Linear                │
    │ • Output: (batch_size x n_windows, n_classes)                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Reshape: (batch_size, n_windows, n_classes) or (batch_size, n_windows)  │
    │          Example: (32, 25) for binary classification [n_classes=1]      │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, emb_size: int, n_classes: int, n_tokens_per_window: int, num_heads: int = 4, dropout: float = 0.1):
        """Initialize attention-based classification head.

        Args:
            emb_size (int): Embedding dimension size.
                Example: 256 (standard BIOT embedding size)
            n_classes (int): Number of output classes.
                Example: 1 (binary spike detection: spike vs. non-spike)
            n_tokens_per_window (int): Number of tokens per window to aggregate.
                Example: 6 (1 CLS + 2 pooled + 3 selected tokens from window encoder)
            dropout (float, optional): Dropout probability for regularization.
                Default: 0.1 (10% dropout)
        """
        super().__init__()
        self.n_tokens_per_window = n_tokens_per_window
        self.n_classes = n_classes

        # Learnable classification query - optimized during training to focus on
        # discriminative features for spike detection
        self.classification_query = nn.Parameter(torch.randn(1, 1, emb_size))
        # Shape: (1, 1, emb_size) - single query vector shared across all windows
        # This query learns to "ask" for the most relevant information for classification

        # Multi-head attention for intelligent token aggregation
        self.token_attention = nn.MultiheadAttention(
            embed_dim=emb_size,          # 256-dimensional embeddings
            num_heads=num_heads,         # Configurable number of attention heads for diverse attention patterns
            dropout=dropout,             # Attention dropout for regularization
            batch_first=True,            # Batch dimension first in input tensors
        )

        # Classification MLP with progressive dimension reduction
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),                      # Normalize inputs for stable training
            nn.Dropout(dropout),                                        # Input dropout for regularization
            nn.Linear(emb_size, emb_size // 2),  # 256 → 128: Feature compression
            nn.GELU(),                                                    # Smooth activation function (better than ReLU)
            nn.Dropout(dropout),                                        # Hidden dropout for additional regularization
            nn.Linear(emb_size // 2, n_classes)  # 128 → n_classes: Final classification layer
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            """Forward pass with attention-based token aggregation per window.

            DETAILED PROCESSING STEPS:

            1. RESHAPE: Group tokens by window for independent processing
            2. IDENTIFY: Detect fully masked windows to skip in attention (production approach)
            3. ATTENTION: Use learnable query to attend to valid windows only
            - Skips fully masked windows entirely (no wasted computation, no NaN)
            - Properly masks out padded tokens within valid windows
            4. CLASSIFICATION: Process attended representation through MLP
            5. RESHAPE: Return per-window predictions with masked windows zeroed

            SHAPE TRANSFORMATIONS:
            - Multi-class (C > 1): (B, NxT, E) → (BxN, T, E) → (BxN_valid, 1, E) → (BxN, C) → (B, N, C)
            - Binary (C = 1):      (B, NxT, E) → (BxN, T, E) → (BxN_valid, 1, E) → (BxN, 1) → (B, N)

            Where:
                B = batch_size, N = n_windows, T = n_tokens_per_window,
                E = emb_size, C = n_classes, N_valid = number of valid windows

            Args:
                x (torch.Tensor): Token embeddings from inter-window transformer.
                    Shape: (batch_size, n_windows * n_tokens_per_window, emb_size)
                    Example: (32, 150, 256) for n_tokens_per_window=6, or (32, 25, 256) for n_tokens_per_window=1

                mask (Optional[torch.Tensor]): Token validity mask (1=valid, 0=masked/padded).
                    Shape: (batch_size, n_windows * n_tokens_per_window)
                    Example: (32, 150) where masked tokens should be ignored in attention

            Returns:
                torch.Tensor: Classification logits for each window.
                    Shape: (batch_size, n_windows, n_classes) if n_classes > 1
                           (batch_size, n_windows) if n_classes == 1
                    Example: (32, 25) for binary classification
                    Raw logits for spike classification per window
            """
            batch_size = x.size(0)
            n_windows = x.size(1) // self.n_tokens_per_window
            emb_size = x.size(2)

            # Log input
            logger = logging.getLogger(__name__)
            log_tensor_statistics(x, f"AttentionClassificationHead input (B={batch_size}, N={n_windows})", logger)
            if mask is not None:
                n_valid_tokens = mask.sum(dim=1).float().mean().item()
                logger.debug(f"AttentionClassificationHead mask: avg valid tokens={n_valid_tokens:.1f}/{x.size(1)}")

            # Handle single token case - no attention needed
            if self.n_tokens_per_window == 1:
                x = x.view(batch_size * n_windows, -1)  # (B*N, E)
                logits = self.classifier(x)  # (B*N, n_classes)

                # Zero out invalid windows
                if mask is not None:
                    window_valid_mask = mask.view(batch_size * n_windows)  # (B*N,)
                    logits = logits * window_valid_mask.unsqueeze(-1).float()

                logits = logits.view(batch_size, n_windows, -1)

                # For binary classification (n_classes=1), squeeze last dimension to get (B, N)
                if self.n_classes == 1:
                    logits = logits.squeeze(-1)

                log_tensor_statistics(logits, "AttentionClassificationHead output (single token)", logger)
                return logits

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 1: Reshape to Process Each Window Independently (Multi-token case)
            # ═══════════════════════════════════════════════════════════════════════════
            x = x.view(batch_size * n_windows, self.n_tokens_per_window, emb_size)
            # Shape: (B, NxT, E) → (BxN, T, E)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 2: Identify Valid Windows (Skip Fully Masked - Production Approach)
            # ═══════════════════════════════════════════════════════════════════════════
            valid_windows_mask = None
            key_padding_mask_valid = None
            
            if mask is not None:
                # Reshape mask to per-window: (B, N*T) → (B*N, T)
                mask_reshaped = mask.view(batch_size * n_windows, self.n_tokens_per_window)
                
                # Identify windows with at least one valid token
                valid_windows_mask = mask_reshaped.any(dim=1)  # (BxN,) True where window has >=1 valid token
                n_valid = valid_windows_mask.sum().item()
                n_total = batch_size * n_windows
                
                if n_valid < n_total:
                    logger.debug(f"AttentionClassificationHead: skipping {n_total - n_valid}/{n_total} fully masked windows")
                
                # Prepare key_padding_mask only for valid windows
                # PyTorch MultiheadAttention: True = ignore, False = attend
                # Our convention: 1 = valid, 0 = masked → invert
                if n_valid > 0:
                    key_padding_mask_valid = ~mask_reshaped[valid_windows_mask].bool()  # (N_valid, T)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 3: Attention-based Token Aggregation (Valid Windows Only)
            # ═══════════════════════════════════════════════════════════════════════════
            # Initialize output for all windows (zeros for masked windows)
            attended_output = torch.zeros(
                batch_size * n_windows, 1, emb_size,
                device=x.device, dtype=x.dtype
            )  # (BxN, 1, E)
            
            # Process only valid windows through attention (skip fully masked)
            if valid_windows_mask is None or valid_windows_mask.all():
                # Fast path: all windows valid
                query = self.classification_query.expand(batch_size * n_windows, 1, emb_size).to(dtype=x.dtype)
                attended_output, _ = self.token_attention(
                    query, x, x,
                    key_padding_mask=key_padding_mask_valid
                )
            elif valid_windows_mask.any():              
                # Slow path: some windows masked, skip them entirely
                n_valid = int(valid_windows_mask.sum().item())
                query_valid = self.classification_query.expand(n_valid, 1, emb_size).to(dtype=x.dtype)
                x_valid = x[valid_windows_mask]  # (N_valid, T, E)

                attended_output_valid, _ = self.token_attention(
                    query_valid, x_valid, x_valid,
                    key_padding_mask=key_padding_mask_valid
                )  # (N_valid, 1, E)

                # Place valid outputs back into full tensor
                attended_output[valid_windows_mask] = attended_output_valid.to(dtype=attended_output.dtype)
            # else: all windows masked, keep zeros
                
            log_tensor_statistics(attended_output, "AttentionClassificationHead after attention", logger)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 4: Classification Through MLP
            # ═══════════════════════════════════════════════════════════════════════════
            logits = self.classifier(attended_output.squeeze(1))
            # Shape: (BxN, 1, E) → (BxN, E) → (BxN, n_classes)
            
            # Zero out logits for fully masked windows (redundant but explicit)
            if valid_windows_mask is not None and not valid_windows_mask.all():
                logits[~valid_windows_mask] = 0.0
                
            log_tensor_statistics(logits, "AttentionClassificationHead after classifier MLP", logger)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 5: Reshape to Window Structure
            # ═══════════════════════════════════════════════════════════════════════════
            logits = logits.view(batch_size, n_windows, -1)
            # Shape: (BxN, n_classes) → (B, N, n_classes)

            # For binary classification (n_classes=1), squeeze last dimension to get (B, N)
            # This ensures consistency with ground truth labels shape
            if self.n_classes == 1:
                logits = logits.squeeze(-1)
                # Shape: (B, N, 1) → (B, N)

            log_tensor_statistics(logits, "AttentionClassificationHead final output", logger)
            return logits
