"""Small differentiable monotonic-alignment losses for paired speech streams."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_cost(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance for channel-first sequences ``(C,T)``."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must both be (channels, frames)")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target channel counts differ")
    source_n = F.normalize(source.transpose(0, 1), dim=-1)
    target_n = F.normalize(target.transpose(0, 1), dim=-1)
    return 1.0 - source_n @ target_n.transpose(0, 1)


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Differentiable monotonic path cost for a precomputed ``(N,M)`` matrix.

    This aligns predictions to untouched target sequences inside matched phone
    spans.  It does not resample either input stream or create warped training
    features.
    """
    if cost.ndim != 2 or not cost.numel():
        raise ValueError("cost must be a non-empty 2-D tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    rows, columns = cost.shape
    infinity = cost.new_tensor(float("inf"))
    zero = cost.new_zeros(())
    previous = [zero] + [infinity] * columns
    for row in range(rows):
        current = [infinity]
        for column in range(columns):
            predecessors = torch.stack(
                [previous[column + 1], current[column], previous[column]]
            )
            soft_min = -gamma * torch.logsumexp(-predecessors / gamma, dim=0)
            current.append(cost[row, column] + soft_min)
        previous = current
    return previous[-1] / (rows + columns)


def phonewise_dual_stream_loss(
    predicted_semantic: torch.Tensor,
    predicted_code: torch.Tensor,
    target_semantic: torch.Tensor,
    target_code: torch.Tensor,
    phone_segments: list[list[dict]],
    *,
    gamma: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Soft-align semantic and acoustic-code sequences inside matched phones."""
    semantic_total = predicted_semantic.new_zeros(())
    code_total = predicted_semantic.new_zeros(())
    weight_total = 0.0
    phones = 0
    for batch_index, segments in enumerate(phone_segments):
        for segment in segments:
            source_start, source_end = segment["src"]
            target_start, target_end = segment["tgt"]
            if source_end <= source_start or target_end <= target_start:
                continue
            weight = float(segment.get("confidence", 1.0))
            semantic_total = semantic_total + weight * soft_dtw(
                cosine_cost(
                    predicted_semantic[batch_index, :, source_start:source_end],
                    target_semantic[batch_index, :, target_start:target_end],
                ),
                gamma,
            )
            code_total = code_total + weight * soft_dtw(
                cosine_cost(
                    predicted_code[batch_index, :, source_start:source_end],
                    target_code[batch_index, :, target_start:target_end],
                ),
                gamma,
            )
            weight_total += weight
            phones += 1
    if phones == 0:
        raise ValueError("batch contains no usable matched phone spans")
    return semantic_total / weight_total, code_total / weight_total, phones
