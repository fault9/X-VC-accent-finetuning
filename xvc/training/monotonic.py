"""Small differentiable monotonic-alignment losses for paired speech streams."""

from __future__ import annotations

import math

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


def phonewise_single_stream_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    phone_segments: list[list[dict]],
    *,
    gamma: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Phone-local monotonic loss for a unified post-prenet representation."""
    total = predicted.new_zeros(())
    weight_total = 0.0
    phones = 0
    for batch_index, segments in enumerate(phone_segments):
        for segment in segments:
            source_start, source_end = segment["src"]
            target_start, target_end = segment["tgt"]
            if source_end <= source_start or target_end <= target_start:
                continue
            weight = float(segment.get("confidence", 1.0))
            total = total + weight * soft_dtw(
                cosine_cost(
                    predicted[batch_index, :, source_start:source_end],
                    target[batch_index, :, target_start:target_end],
                ),
                gamma,
            )
            weight_total += weight
            phones += 1
    if phones == 0:
        raise ValueError("batch contains no usable matched phone spans")
    return total / weight_total, phones


def phonewise_discrete_code_loss(
    code_logits: torch.Tensor,
    target_code_indices: torch.Tensor,
    phone_segments: list[list[dict]],
    *,
    gamma: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Phone-local monotonic NLL over the frozen acoustic codebook.

    ``code_logits`` is ``(B, T_source, K)`` and must be computed with the same
    frozen codebook geometry used for inference.  For every source/target phone
    pair, the cost at ``(i, j)`` is the negative log probability assigned by
    source query ``i`` to the real target-persona code id at target frame ``j``.
    Soft-DTW supplies duration-tolerant monotonic supervision without warping
    either sequence.

    The NLL is divided by ``log(K)``.  A uniform K-way prediction is therefore
    approximately one, keeping conservative loss weights interpretable across
    codebook sizes.
    """
    if code_logits.ndim != 3:
        raise ValueError("code_logits must be (batch, source_frames, codes)")
    if target_code_indices.ndim != 2:
        raise ValueError("target_code_indices must be (batch, target_frames)")
    if code_logits.shape[0] != target_code_indices.shape[0]:
        raise ValueError("code-logit and target-index batch sizes differ")
    if code_logits.shape[-1] < 2:
        raise ValueError("codebook must contain at least two entries")

    log_probabilities = F.log_softmax(code_logits, dim=-1)
    normalizer = math.log(code_logits.shape[-1])
    total = code_logits.new_zeros(())
    weight_total = 0.0
    phones = 0
    for batch_index, segments in enumerate(phone_segments):
        for segment in segments:
            source_start, source_end = segment["src"]
            target_start, target_end = segment["tgt"]
            if source_end <= source_start or target_end <= target_start:
                continue
            local_targets = target_code_indices[
                batch_index, target_start:target_end
            ].long()
            if not local_targets.numel():
                continue
            if int(local_targets.min()) < 0 or int(local_targets.max()) >= code_logits.shape[-1]:
                raise ValueError("target code index lies outside the frozen codebook")
            local_log_probabilities = log_probabilities[
                batch_index, source_start:source_end
            ]
            cost = -local_log_probabilities[:, local_targets] / normalizer
            weight = float(segment.get("confidence", 1.0))
            total = total + weight * soft_dtw(cost, gamma)
            weight_total += weight
            phones += 1
    if phones == 0:
        raise ValueError("batch contains no usable matched phone spans")
    return total / weight_total, phones


def phonewise_target_code_margin_loss(
    code_logits: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    source_code: torch.Tensor,
    target_code: torch.Tensor,
    phone_segments: list[list[dict]],
    *,
    margin: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Make an aligned target-persona code beat the original source code.

    Alignment is computed once from detached, untouched source/target code
    embeddings. No waveform or feature is resampled. Positions whose aligned
    source and target ids are already equal are ignored: this loss asks for
    target-directed substitutions, not indiscriminate code churn.
    """
    if code_logits.ndim != 3:
        raise ValueError("code_logits must be (batch, source_frames, codes)")
    if margin <= 0:
        raise ValueError("margin must be positive")
    total = code_logits.new_zeros(())
    weight_total = 0.0
    positions = 0
    source_ids_cpu = source_indices.detach().cpu()
    target_ids_cpu = target_indices.detach().cpu()
    for batch_index, segments in enumerate(phone_segments):
        for segment in segments:
            source_start, source_end = segment["src"]
            target_start, target_end = segment["tgt"]
            if source_end <= source_start or target_end <= target_start:
                continue
            path = hard_dtw_path(
                cosine_cost(
                    source_code[batch_index, :, source_start:source_end],
                    target_code[batch_index, :, target_start:target_end],
                )
            )
            confidence = float(segment.get("confidence", 1.0))
            for source_local, target_local in path:
                source_position = source_start + source_local
                target_position = target_start + target_local
                source_id = int(source_ids_cpu[batch_index, source_position])
                target_id = int(target_ids_cpu[batch_index, target_position])
                if source_id == target_id:
                    continue
                logits = code_logits[batch_index, source_position]
                total = total + confidence * F.relu(
                    logits[source_id] - logits[target_id] + margin
                )
                weight_total += confidence
                positions += 1
    if positions == 0:
        # A batch with no true substitutions should be a safe no-op rather
        # than an error or pressure to alter already-correct codes.
        return code_logits.sum() * 0.0, 0
    return total / weight_total, positions


def hard_dtw_path(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Return a deterministic monotonic minimum-cost path for diagnostics."""
    if cost.ndim != 2 or not cost.numel():
        raise ValueError("cost must be a non-empty 2-D tensor")
    values = cost.detach().float().cpu()
    rows, columns = values.shape
    infinity = float("inf")
    scores = [[infinity] * (columns + 1) for _ in range(rows + 1)]
    back = [[None] * (columns + 1) for _ in range(rows + 1)]
    scores[0][0] = 0.0
    # Prefer a diagonal predecessor on exact ties, then vertical, then
    # horizontal.  This avoids gratuitous repeats in equal-cost regions.
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            predecessors = (
                (scores[row - 1][column - 1], row - 1, column - 1, 0),
                (scores[row - 1][column], row - 1, column, 1),
                (scores[row][column - 1], row, column - 1, 2),
            )
            best = min(predecessors, key=lambda item: (item[0], item[3]))
            scores[row][column] = float(values[row - 1, column - 1]) + best[0]
            back[row][column] = (best[1], best[2])
    path = []
    row, column = rows, columns
    while row and column:
        path.append((row - 1, column - 1))
        row, column = back[row][column]
    if row or column:
        raise RuntimeError("hard-DTW path did not reach the origin")
    path.reverse()
    return path


def phonewise_aligned_code_agreement(
    predicted_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    source_code: torch.Tensor,
    target_code: torch.Tensor,
    phone_segments: list[list[dict]],
) -> dict[str, float]:
    """Compare native and edited ids to ASI ids on one fixed phone-DTW path.

    The path is derived from the untouched source/target code embeddings, so a
    mapper cannot improve this metric merely by changing the alignment.  The
    returned gain measures target-aligned edits rather than arbitrary code
    churn.
    """
    for name, value in (
        ("predicted_indices", predicted_indices),
        ("source_indices", source_indices),
        ("target_indices", target_indices),
    ):
        if value.ndim != 2:
            raise ValueError(f"{name} must be (batch, frames)")
    if source_code.ndim != 3 or target_code.ndim != 3:
        raise ValueError("source_code and target_code must be (batch, channels, frames)")

    predicted_matches = 0.0
    source_matches = 0.0
    weight_total = 0.0
    positions = 0
    for batch_index, segments in enumerate(phone_segments):
        for segment in segments:
            source_start, source_end = segment["src"]
            target_start, target_end = segment["tgt"]
            if source_end <= source_start or target_end <= target_start:
                continue
            path = hard_dtw_path(
                cosine_cost(
                    source_code[batch_index, :, source_start:source_end],
                    target_code[batch_index, :, target_start:target_end],
                )
            )
            confidence = float(segment.get("confidence", 1.0))
            for source_local, target_local in path:
                source_position = source_start + source_local
                target_position = target_start + target_local
                target_id = target_indices[batch_index, target_position]
                predicted_matches += confidence * float(
                    predicted_indices[batch_index, source_position] == target_id
                )
                source_matches += confidence * float(
                    source_indices[batch_index, source_position] == target_id
                )
                weight_total += confidence
                positions += 1
    if not positions or not weight_total:
        raise ValueError("batch contains no usable aligned code positions")
    predicted = predicted_matches / weight_total
    source = source_matches / weight_total
    return {
        "source": source,
        "predicted": predicted,
        "gain": predicted - source,
        "positions": float(positions),
    }
