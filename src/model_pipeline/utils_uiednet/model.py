"""
U-IEDNet
========
Deep learning framework for classification and segmentation of interictal
epileptiform discharges (IEDs) in multichannel EEG.

Reference
---------
Sun et al., "Deep learning-based classification and segmentation of interictal
epileptiform discharges using multichannel electroencephalography",
Epilepsia, 2025;66:3398-3410.  https://doi.org/10.1111/epi.18463
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """
    Strided 1-D encoder block.
    Conv1d(stride=2) → BatchNorm1d → LeakyReLU(0.1)

    Halves the temporal dimension on each call.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=2, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class UpConvBlock1D(nn.Module):
    """
    Strided 1-D decoder block (x2 upsampling).
    ConvTranspose1d(stride=2) → BatchNorm1d → LeakyReLU(0.1)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel,
                               stride=2, padding=kernel // 2,
                               output_padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class TransformerBlock(nn.Module):
    """
    Pre-LN transformer block.
    LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual

    Args:
        d_model : embedding dimension.
        n_heads : number of attention heads.
        mlp_ratio : hidden-layer expansion ratio inside the MLP.
        dropout : dropout rate in attention and MLP.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                "for nn.MultiheadAttention."
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x : (B, L, D)
            key_padding_mask : (B, L) bool — True for positions to ignore.

        Returns:
            (B, L, D)
        """
        h, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask,
        )
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
    
class TemporalEncoder(nn.Module):
    """
    Hierarchical temporal encoder with intra-channel self-attention.

    All operations use shared weights across EEG channels via the B·C trick:
    channels are merged into the batch dimension so a single set of parameters
    is applied identically to every channel.

    Downsampling path
    -----------------
    Conv1 (k=7, s=2)  :  (1,   T)    → (64,  T/2)   → skip1
    Conv2 (k=5, s=2)  :  (64,  T/2)  → (128, T/4)   → skip2
    Conv3 (k=3, s=2)  :  (128, T/4)  → (256, T/8)   → skip3
    BiGRU x gru_layers  :  (256, T/8) → (256, T/8)   sequence-to-sequence
    Conv4 (k=3, s=2)  :  (256, T/8)  → (d_model, T/16)  → bottleneck (for decoder)

    Intra-channel temporal self-attention
    --------------------------------------
    For each channel's temporal sequence (T/16 tokens of dim d_model):
      1. Prepend one learnable CLS token  →  length T/16 + 1
      2. Apply tf_layers x TransformerBlock  (shared across channels)
      3. LayerNorm
      4. Extract token[0]  →  per-channel CLS embedding (d_model-d)


    Skip connections are saved after each ConvBlock (before BiGRU / ConvBlock4)
    for the U-Net decoder.

    Returns
    -------
    bottleneck : (B, C, T//16, d_model) — raw Conv4 output saved for the decoder.
    cls_tokens : (B, C, d_model) — per-channel CLS embeddings.
    skips : (skip1, skip2, skip3)
    """

    def __init__(
        self,
        d_model: int = 512,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        tf_heads: int = 4,
        tf_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.conv1 = ConvBlock1D(1, 64,  kernel=7)
        self.conv2 = ConvBlock1D(64, 128, kernel=5)
        self.conv3 = ConvBlock1D(128, 256, kernel=3)
        self.bigru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,   # output size = 2 * gru_hidden
        )
        self.conv4 = ConvBlock1D(gru_hidden * 2, d_model, kernel=3)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_transformers = nn.ModuleList([
            TransformerBlock(d_model, tf_heads, dropout=dropout)
            for _ in range(tf_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _to_btf(h: torch.Tensor, B: int, C: int) -> torch.Tensor:
        """(B·C, features, time) → (B, C, time, features)"""
        F_dim, T_dim = h.shape[1], h.shape[2]
        return h.reshape(B, C, F_dim, T_dim).permute(0, 1, 3, 2)
    
    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        B, C, T = x.shape
        h = x.reshape(B * C, 1, T)             # (B·C, 1,   T)

        h = self.conv1(h)                       # (B·C, 64,  T/2)
        skip1 = self._to_btf(h, B, C)          # (B, C, T/2,  64)

        h = self.conv2(h)                       # (B·C, 128, T/4)
        skip2 = self._to_btf(h, B, C)          # (B, C, T/4,  128)

        h = self.conv3(h)                       # (B·C, 256, T/8)
        skip3 = self._to_btf(h, B, C)          # (B, C, T/8,  256)

        # BiGRU operates over the temporal dimension
        h_seq, _ = self.bigru(h.permute(0, 2, 1))  # (B·C, T/8, 2·gru_hidden)
        h = h_seq.permute(0, 2, 1)             # (B·C, 2·gru_hidden, T/8)

        h = self.conv4(h)
        bottleneck = self._to_btf(h, B, C)

        seq = h.permute(0, 2, 1)
        cls = self.cls_token.expand(B * C, 1, self.d_model)
        seq = torch.cat([cls, seq], dim=1)

        for layer in self.temporal_transformers:
            seq = layer(seq)

        cls_tokens = seq[:, 0, :].reshape(B, C, self.d_model)

        return bottleneck, cls_tokens, (skip1, skip2, skip3)
    
class SpatialEncoder(nn.Module):
    """
    Inter-channel transformer encoder.

    The sequence fed to the transformer has length C: one CLS token per
    electrode.  This lets every attention head model pairwise interactions
    between all channels simultaneously.

    Zero-padded channels are excluded from attention via a key-padding mask
    so they cannot corrupt the representations of real channels.

    Args:
        d_model : token dimension (must equal TemporalEncoder output = 512).
        n_heads : attention heads.
        n_layers : stacked transformer blocks.
        max_channels : size of the learnable channel positional embedding.
        dropout : dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 3,
        n_layers: int = 2,
        max_channels: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_channels, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cls_tokens: torch.Tensor,
                channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            cls_tokens   : (B, C, 512) — one CLS token per channel.
            channel_mask : (B, C) bool — True for zero-padded channels.

        Returns:
            (B, C, 512) — spatially enriched per-channel representations.
        """
        B, C, _ = cls_tokens.shape

        x = cls_tokens + self.pos_embed[:, :C, :]

        for layer in self.layers:
            x = layer(x, key_padding_mask=channel_mask)

        return self.norm(x)
    
class TemporalDecoder(nn.Module):
    """
    U-Net temporal decoder with skip connections.

    Upsampling path  (in_ch → out_ch  |  time resolution)
    ──────────────────────────────────────────────────────
    UpConv1 :  512 → 256  |  T/16 → T/8
    cat skip3 (+256) → 512
    UpConv2 :  512 → 128  |  T/8  → T/4
    cat skip2 (+128) → 256
    UpConv3 :  256 →  64  |  T/4  → T/2
    cat skip1  (+64) → 128
    OutConv :  128 →   1  |  T/2  → T   (TransposeConv + Sigmoid)

    Output is gated element-wise by the per-channel classification confidence.
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.up1 = UpConvBlock1D(d_model, 256)
        self.up2 = UpConvBlock1D(256 + 256, 128)
        self.up3 = UpConvBlock1D(128 + 128, 64)
        self.out_seg = nn.Sequential(
            nn.ConvTranspose1d(64 + 64, 1,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        clf_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bottleneck : (B, C, T//16, 512)
            skips : skip1 (B,C,T/2,64), skip2 (B,C,T/4,128), skip3 (B,C,T/8,256)
            clf_confidence : (B, C) in [0,1] — gates the output. None = no gating.

        Returns:
            seg_prob : (B, C, T)
        """
        skip1, skip2, skip3 = skips
        B, C = bottleneck.shape[:2]

        def _to_1d(t: torch.Tensor) -> torch.Tensor:
            """(B, C, time, feat) → (B·C, feat, time)"""
            return t.permute(0, 1, 3, 2).reshape(B * C, t.shape[3], t.shape[2])

        h  = _to_1d(bottleneck)   # (B·C, 512,  T/16)
        s3 = _to_1d(skip3)        # (B·C, 256,  T/8)
        s2 = _to_1d(skip2)        # (B·C, 128,  T/4)
        s1 = _to_1d(skip1)        # (B·C, 64,   T/2)

        h = self.up1(h)                        # (B·C, 256, T/8)
        h = torch.cat([h, s3], dim=1)         # (B·C, 512, T/8)

        h = self.up2(h)                        # (B·C, 128, T/4)
        h = torch.cat([h, s2], dim=1)         # (B·C, 256, T/4)

        h = self.up3(h)                        # (B·C, 64,  T/2)
        h = torch.cat([h, s1], dim=1)         # (B·C, 128, T/2)

        h = self.out_seg(h)                    # (B·C, 1,   T)
        seg_logit = h.reshape(B, C, -1)         # (B, C, T)

        # Gate segmentation by per-channel classification confidence
        if clf_confidence is not None:
            seg_logit = seg_logit * clf_confidence.unsqueeze(-1)   # (B, C, 1) broadcast

        return seg_logit
    
class UIEDNET(nn.Module):
    """
    UIEDNET: simultaneous classification and segmentation of IEDs in
    multichannel EEG.

    Args
    ----
    input_shape : list
        Input shape of the given data.
    gru_hidden : int
        Hidden size *per direction* of the BiGRU (total output = 2 x gru_hidden).
    gru_layers : int
        Number of stacked BiGRU layers.
    tf_heads : int
        Attention heads in the spatial transformer.
    tf_layers : int
        Number of stacked transformer blocks in the spatial encoder.
    use_segmentation : bool
        When True (default), the temporal decoder is instantiated and
        ``forward`` returns a ``'seg'`` key in its output dict.
    dropout : float
        Dropout rate used in transformer blocks.

    Forward inputs
    --------------
    x : (B, C, T) float — multichannel EEG segment.
    channel_mask : (B, C) bool, optional — True for zero-padded channels.
                   Auto-detected from all-zero channels when not provided.

    Forward outputs  (dict)
    -----------------------
    'clf' : (B, C) or (B,)  — classification probabilities.
    'seg' : (B, C, T) — segmentation probabilities (only if use_segmentation=True).
    """

    def __init__(
        self,
        input_shape: list = [32, 23, 512],
        gru_hidden: int = 128,
        gru_layers: int = 2,
        tf_heads: int = 4,
        tf_layers: int = 2,
        tf_emb_size: int = 512,
        use_segmentation: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_shape[-1] % 16 == 0, "n_time must be divisible by 16."

        self.input_shape = input_shape

        self.use_segmentation = use_segmentation

        self.temporal_encoder = TemporalEncoder(
            d_model= tf_emb_size,
            gru_hidden=gru_hidden, gru_layers=gru_layers,
            tf_heads=tf_heads, tf_layers=tf_layers, dropout=dropout,
        )

        self.spatial_encoder = SpatialEncoder(
            d_model=tf_emb_size, n_heads=tf_heads, n_layers=tf_layers,
            max_channels=input_shape[1], dropout=dropout,
        )
        self.clf_head_per_chan = nn.Linear(tf_emb_size, 1)       # per-channel classification

        if use_segmentation:
            self.temporal_decoder = TemporalDecoder(d_model=tf_emb_size)

    @staticmethod
    def _auto_channel_mask(x: torch.Tensor) -> torch.Tensor:
        """
        Detect zero-padded channels by checking whether every sample is zero.

        Returns
        -------
        (B, C) bool — True for padded / inactive channels.
        """
        return x.abs().sum(dim=-1) != 0
    
    def _collapse_seg_channels(
        self,
        seg: torch.Tensor,            # (B, C, T)
        chan_probs: torch.Tensor,     # (B, C) — poids d'attention
    ) -> torch.Tensor:
        """
        Collapse (B, C, T) → (B, T) pondéré par chan_probs.
        """
        weights = chan_probs / (chan_probs.sum(dim=1, keepdim=True) + 1e-8)
        # Somme pondérée sur les canaux
        collapsed = (seg * weights.unsqueeze(-1)).sum(dim=1)               # (B, T)
        return collapsed

    def forward(self, x: torch.Tensor,
                channel_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        if channel_mask is None:
            channel_mask = self._auto_channel_mask(x)   # (B, C)

        # 1. Temporal Encoder
        bottleneck, cls_tokens, skips = self.temporal_encoder(x)    # (B, C, T/16, 512)

        # 2. Spatial Encoder
        spatial_feats = self.spatial_encoder(cls_tokens, ~channel_mask)
        chan_logits = self.clf_head_per_chan(spatial_feats).squeeze(-1)  # (B, C)
        chan_logits = chan_logits.masked_fill(~channel_mask, -1e4)
        chan_probs = torch.sigmoid(chan_logits)                          # (B, C)

        clf_logit = chan_logits.max(dim=1).values                       # (B,)
        clf_prob = torch.sigmoid(clf_logit)                             # (B,)

        out = {"clf_logits": clf_logit, "clf_probs": clf_prob}

        # 4. Segmentation output  (optional)
        if self.use_segmentation:

            seg_logit_2d = self.temporal_decoder(
                bottleneck, skips, clf_confidence=chan_probs,
            )                                                            # (B, C, T)

            seg_logit_2d = seg_logit_2d.masked_fill(
                (~channel_mask).unsqueeze(-1), 0.0
            )

            seg_logit_1d = self._collapse_seg_channels(
                seg_logit_2d, chan_probs,
            )                                                   # (B, T)

            out["seg_logits"] = seg_logit_1d                    # (B, T)
            out["seg_probs"]  = torch.sigmoid(seg_logit_1d)     # (B, T)

        return out

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)