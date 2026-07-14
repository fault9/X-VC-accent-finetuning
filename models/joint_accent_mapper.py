"""Source-agnostic joint semantic/acoustic accent mapper for X-VC.

The mapper sits between X-VC's frozen source encoders and frozen prenet.  It
never receives a source-speaker id or embedding: target voice remains the job
of X-VC's existing reference/speaker conditioning.  Both output heads share a
causal trunk so semantic and acoustic edits cannot drift independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        # Per-frame normalization avoids making a short utterance depend on
        # batch padding length (GroupNorm would pool across time as well).
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = F.pad(y, (self.left_pad, 0))
        y = self.depthwise(y)
        y = self.pointwise(F.gelu(y))
        return x + y


class JointAccentMapper(nn.Module):
    """Joint residual editor over X-VC's two 50 Hz source streams.

    Parameters are deliberately independent of a particular source speaker.
    ``semantic`` and ``acoustic_zq`` are ``(B, 1024, T)`` by default;
    ``acoustic_code`` is the frozen quantizer's low-dimensional code embedding
    ``(B, 8, T)``.  Zero-initialized heads make both outputs exact identities at
    initialization.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        acoustic_dim: int = 1024,
        code_dim: int = 8,
        hidden: int = 256,
        layers: int = 6,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        lookahead_frames: int = 4,
        input_dropout: float = 0.05,
    ):
        super().__init__()
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0")
        if hidden < 16:
            raise ValueError("hidden must be >= 16")
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.code_dim = code_dim
        self.lookahead_frames = int(lookahead_frames)
        branch = hidden // 2
        self.semantic_in = nn.Conv1d(semantic_dim, branch, 1)
        self.acoustic_in = nn.Conv1d(acoustic_dim, hidden - branch, 1)
        self.input_dropout = nn.Dropout1d(input_dropout)
        self.blocks = nn.ModuleList(
            [
                _CausalResidualBlock(
                    hidden, kernel_size, dilation_growth ** index
                )
                for index in range(layers)
            ]
        )
        self.semantic_delta = nn.Conv1d(hidden, semantic_dim, 1)
        self.code_delta = nn.Conv1d(hidden, code_dim, 1)
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.semantic_delta.bias)
        nn.init.zeros_(self.code_delta.weight)
        nn.init.zeros_(self.code_delta.bias)

    @property
    def lookahead_ms(self) -> int:
        return self.lookahead_frames * 20

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _trunk(self, semantic: torch.Tensor, acoustic_zq: torch.Tensor) -> torch.Tensor:
        if semantic.shape[0] != acoustic_zq.shape[0] or semantic.shape[-1] != acoustic_zq.shape[-1]:
            raise ValueError(
                f"stream shape mismatch: semantic={tuple(semantic.shape)}, "
                f"acoustic={tuple(acoustic_zq.shape)}"
            )
        x = torch.cat(
            [self.semantic_in(semantic), self.acoustic_in(acoustic_zq)], dim=1
        )
        x = self.input_dropout(x)
        lookahead = self.lookahead_frames
        if lookahead:
            x = F.pad(x, (0, lookahead))
        for block in self.blocks:
            x = block(x)
        return x[:, :, lookahead:] if lookahead else x

    def forward(
        self,
        semantic: torch.Tensor,
        acoustic_zq: torch.Tensor,
        acoustic_code: torch.Tensor,
        *,
        return_deltas: bool = False,
    ):
        if acoustic_code.shape[1] != self.code_dim:
            raise ValueError(
                f"expected code_dim={self.code_dim}, got {acoustic_code.shape[1]}"
            )
        if acoustic_code.shape[-1] != semantic.shape[-1]:
            raise ValueError("acoustic-code and semantic timelines must match")
        hidden = self._trunk(semantic, acoustic_zq)
        semantic_delta = self.semantic_delta(hidden)
        code_delta = self.code_delta(hidden)
        edited_semantic = semantic + semantic_delta
        edited_code = acoustic_code + code_delta
        if return_deltas:
            return edited_semantic, edited_code, semantic_delta, code_delta
        return edited_semantic, edited_code

    @staticmethod
    def code_logits(
        code_query: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Cosine logits ``(B,T,K)`` against the frozen X-VC codebook."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        query = F.normalize(code_query.transpose(1, 2), dim=-1)
        keys = F.normalize(codebook, dim=-1)
        return torch.matmul(query, keys.transpose(0, 1)) / temperature

    @classmethod
    def nearest_codes(
        cls, code_query: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        return cls.code_logits(code_query, codebook).argmax(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"semantic_dim={self.semantic_dim}, acoustic_dim={self.acoustic_dim}, "
            f"code_dim={self.code_dim}, lookahead={self.lookahead_frames} frames "
            f"({self.lookahead_ms} ms), params={self.n_params()/1e6:.2f}M"
        )
