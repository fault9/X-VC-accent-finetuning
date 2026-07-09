"""Tiny streaming-safe AccentBridge: causal residual feature editor for X-VC.

Purpose (see docs/streaming_accentbridge_plan.md): the decode diagnostic proved
X-VC's content streams stay accent-neutral for native input, so accent must be
injected INTO the content pathway. This module edits the post-semantic_adapter
stream — (B, 1024, T) at 50 Hz / 20 ms frames — with a residual delta:

    edited = x + delta(x),   delta zero-init  =>  exact identity at init
                                                  (same philosophy as LoRA-B=0).

Streaming properties, by construction:

  * length-stable: T in == T out, no insert/delete, no drift;
  * causal with an EXPLICIT lookahead budget: every output frame t depends on
    input frames <= t + lookahead_frames, giving exactly
    lookahead_frames * 20 ms algorithmic latency — this maps 1:1 onto the
    existing `future_ms` window knob of bins/infer_utils.run_streaming;
  * stateless conv: works with X-VC's re-encode-per-chunk streaming (the chunk's
    history region supplies the left context), no KV cache needed;
  * tiny: default config ~0.6M params, sub-millisecond per 320 ms hop on GPU.

Left context comes from `n_layers` dilated causal convolutions (receptive field
`receptive_field_frames`); lookahead is applied once around the whole stack by
right-padding the input L frames and cropping the first L output frames, so the
budget is a single auditable number rather than a per-layer accounting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_MS = 20.0  # post-semantic_adapter stream rate: 50 Hz


class _CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.pw = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.left_pad, 0))
        y = self.pw(self.act(self.conv(y)))
        return x + y


class AccentBridge(nn.Module):
    """Residual causal editor over (B, dim, T) features."""

    def __init__(
        self,
        dim: int = 1024,
        hidden: int = 192,
        n_layers: int = 4,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        lookahead_frames: int = 0,
    ):
        super().__init__()
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0")
        self.dim = dim
        self.lookahead_frames = int(lookahead_frames)
        self.proj_in = nn.Conv1d(dim, hidden, 1)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(hidden, kernel_size, dilation_growth ** i)
            for i in range(n_layers)
        ])
        self.proj_out = nn.Conv1d(hidden, dim, 1)
        # Zero-init the output projection: the bridge is the identity map until
        # trained (mirrors LoRA's zero-init B).
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self._receptive = 1 + sum(
            (kernel_size - 1) * dilation_growth ** i for i in range(n_layers))

    @property
    def lookahead_ms(self) -> float:
        return self.lookahead_frames * FRAME_MS

    @property
    def receptive_field_frames(self) -> int:
        """Left-context frames each output frame can see (excludes lookahead)."""
        return self._receptive

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Residual edit for input (B, dim, T); output frame t saw x[.. t+L]."""
        L = self.lookahead_frames
        y = F.pad(x, (0, L)) if L else x
        y = self.proj_in(y)
        for blk in self.blocks:
            y = blk(y)
        y = self.proj_out(y)
        return y[:, :, L:] if L else y

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        d = self.delta(x)
        return (x + d, d) if return_delta else x + d

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, lookahead={self.lookahead_frames} frames "
                f"({self.lookahead_ms:.0f} ms), receptive_field="
                f"{self.receptive_field_frames} frames, params={self.n_params()/1e6:.2f}M")
