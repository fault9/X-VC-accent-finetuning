"""Causal, source-agnostic pronunciation editor for X-VC post-prenet features.

This v1 keeps X-VC's 50 Hz runtime timeline fixed.  A recurrent sequence trunk
can use the full deployment history plus a bounded right lookahead, while an
auxiliary duration head learns phone-level shorten/keep/lengthen tendencies.
The duration head is deliberately diagnostic-only: hard repeat/drop actions
are not applied until they have passed a separate intelligibility gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPronunciationEditor(nn.Module):
    """Sequence editor over X-VC's unified post-prenet content stream."""

    ACTIONS = ("shorten", "keep", "lengthen")

    def __init__(
        self,
        input_dim: int = 1024,
        hidden: int = 256,
        layers: int = 2,
        lookahead_frames: int = 4,
        required_history_frames: int = 62,
        input_dropout: float = 0.05,
    ):
        super().__init__()
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0")
        if required_history_frames < 0:
            raise ValueError("required_history_frames must be >= 0")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.lookahead_frames = int(lookahead_frames)
        self.required_history_frames = int(required_history_frames)
        self.input_proj = nn.Conv1d(input_dim, hidden, 1)
        self.lookahead_proj = nn.Conv1d(
            hidden * (self.lookahead_frames + 1), hidden, 1
        )
        self.input_dropout = nn.Dropout(input_dropout)
        self.sequence = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=input_dropout if layers > 1 else 0.0,
        )
        self.delta = nn.Conv1d(hidden, input_dim, 1)
        self.duration_logits = nn.Conv1d(hidden, len(self.ACTIONS), 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.duration_logits.weight)
        with torch.no_grad():
            self.duration_logits.bias.copy_(torch.tensor([-2.0, 2.0, -2.0]))

    @property
    def lookahead_ms(self) -> int:
        return self.lookahead_frames * 20

    @property
    def receptive_history_frames(self) -> int:
        # The GRU is recurrent, but training and deployment both provide this
        # much explicit left context before any output frame is supervised.
        return self.required_history_frames

    @property
    def receptive_history_ms(self) -> int:
        return self.receptive_history_frames * 20

    def validate_stream_window(
        self, history_ms: int, smooth_ms: int, future_ms: int
    ) -> None:
        if self.receptive_history_ms > int(history_ms):
            raise ValueError(
                f"editor needs {self.receptive_history_ms} ms history, "
                f"but the stream provides {history_ms} ms"
            )
        available_right_ms = int(smooth_ms) + int(future_ms)
        if self.lookahead_ms > available_right_ms:
            raise ValueError(
                f"editor needs {self.lookahead_ms} ms lookahead, but the "
                f"stream provides only {available_right_ms} ms"
            )

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _lookahead_context(self, hidden: torch.Tensor) -> torch.Tensor:
        frames = [hidden]
        for offset in range(1, self.lookahead_frames + 1):
            frames.append(F.pad(hidden[:, :, offset:], (0, offset)))
        return self.lookahead_proj(torch.cat(frames, dim=1))

    def forward(self, hidden: torch.Tensor, *, return_aux: bool = False):
        if hidden.ndim != 3 or hidden.shape[1] != self.input_dim:
            raise ValueError(
                f"expected (batch,{self.input_dim},frames), got {tuple(hidden.shape)}"
            )
        x = self.input_proj(hidden)
        x = self._lookahead_context(x)
        x = self.input_dropout(x.transpose(1, 2))
        x, _ = self.sequence(x)
        x = x.transpose(1, 2)
        delta = self.delta(x)
        edited = hidden + delta
        actions = self.duration_logits(x)
        if return_aux:
            return edited, delta, actions
        return edited
