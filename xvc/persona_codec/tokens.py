"""Pure token/sequence utilities for the StreamVoice-style persona codec path.

Everything in this module is CPU-only numpy/stdlib so the geometry rules the
persona-codec generator relies on can be unit-tested without torch, the X-VC
checkpoint, or a GPU.  No function here reads or writes audio; the persona
codec path never warps waveforms or features, it only re-groups and compares
already-extracted discrete token IDs.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# One WhisperVQ semantic token spans this many acoustic-code frames when the
# X-VC config upsamples 12.5 Hz semantic tokens to the 50 Hz fusion rate.
# The extraction script verifies this value against both freshly encoded
# audio and the live semantic adapter's measured upsample ratio, and the
# fail-closed checkpoint loader re-checks it against the adapter at load
# time; nothing downstream may assume it without those checks.
EXPECTED_GROUP_SIZE = 4


def validate_token_ids(tokens: np.ndarray, vocab_size: int, name: str) -> None:
    """Fail closed when token IDs cannot belong to the stated vocabulary."""
    if tokens.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D token sequence, got {tokens.shape}")
    if tokens.size == 0:
        raise ValueError(f"{name}: empty token sequence")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"{name}: token dtype must be integer, got {tokens.dtype}")
    lo, hi = int(tokens.min()), int(tokens.max())
    if lo < 0 or hi >= vocab_size:
        raise ValueError(
            f"{name}: token id range [{lo}, {hi}] outside vocabulary of {vocab_size}"
        )


def trim_to_group_ratio(
    semantic_len: int, acoustic_len: int, group_size: int
) -> tuple[int, int, int]:
    """Trim stream lengths so acoustic frames group exactly under semantic steps.

    Returns ``(semantic_len', acoustic_len', dropped_acoustic_frames)`` where
    ``acoustic_len' == group_size * semantic_len'``.  Encoder edge behaviour can
    differ by one frame between the two frozen encoders, so a sub-group trim
    (< ``group_size`` dropped frames) is expected; a full dropped group — one
    whole semantic step of unaccounted audio — means the utterance geometry is
    wrong, so that fails instead.
    """
    if semantic_len < 1 or acoustic_len < 1:
        raise ValueError("both streams must be non-empty")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    usable_semantic = min(semantic_len, acoustic_len // group_size)
    if usable_semantic < 1:
        raise ValueError(
            f"cannot group {acoustic_len} acoustic frames under "
            f"{semantic_len} semantic steps with group size {group_size}"
        )
    if semantic_len - usable_semantic > 1:
        raise ValueError(
            f"semantic/acoustic lengths {semantic_len}/{acoustic_len} disagree by "
            f"more than one semantic step at group size {group_size}"
        )
    dropped = acoustic_len - usable_semantic * group_size
    if dropped >= group_size:
        raise ValueError(
            f"trimming would drop {dropped} acoustic frames "
            f"(>= one full group / semantic step) for lengths "
            f"{semantic_len}/{acoustic_len}"
        )
    return usable_semantic, usable_semantic * group_size, dropped


def group_acoustic_codes(codes: np.ndarray, group_size: int) -> np.ndarray:
    """Reshape a flat 50 Hz code sequence into ``[T_semantic, group_size]``."""
    if codes.ndim != 1:
        raise ValueError(f"expected flat code sequence, got {codes.shape}")
    if codes.size == 0 or codes.size % group_size != 0:
        raise ValueError(
            f"{codes.size} acoustic codes do not group by {group_size}; "
            "trim with trim_to_group_ratio first"
        )
    return codes.reshape(-1, group_size)


def levenshtein_alignment(
    source: Sequence[int], target: Sequence[int]
) -> dict[str, Any]:
    """Token-level edit distance with operation counts and matched positions.

    Returns dict with ``distance``, ``normalized`` (by max length), operation
    counts (``matches``/``substitutions``/``insertions``/``deletions``, where
    insertions add target tokens absent from the source), and ``matched_pairs``
    — (source_index, target_index) tuples for exactly matched tokens, used to
    associate acoustic groups across the pair.
    """
    src = list(int(x) for x in source)
    tgt = list(int(x) for x in target)
    rows, cols = len(src) + 1, len(tgt) + 1
    dist = np.zeros((rows, cols), dtype=np.int32)
    dist[:, 0] = np.arange(rows)
    dist[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dist[i, j] = min(
                dist[i - 1, j] + 1,        # deletion (source token dropped)
                dist[i, j - 1] + 1,        # insertion (target token added)
                dist[i - 1, j - 1] + cost, # match / substitution
            )
    matches = substitutions = insertions = deletions = 0
    matched_pairs: list[tuple[int, int]] = []
    i, j = len(src), len(tgt)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i, j] == dist[i - 1, j - 1] + (
            0 if src[i - 1] == tgt[j - 1] else 1
        ):
            if src[i - 1] == tgt[j - 1]:
                matches += 1
                matched_pairs.append((i - 1, j - 1))
            else:
                substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dist[i, j] == dist[i - 1, j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    matched_pairs.reverse()
    longest = max(len(src), len(tgt))
    return {
        "distance": int(dist[len(src), len(tgt)]),
        "normalized": float(dist[len(src), len(tgt)]) / longest if longest else 0.0,
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "source_len": len(src),
        "target_len": len(tgt),
        "matched_pairs": matched_pairs,
    }


def prompt_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    ids = set()
    for row in rows:
        prompt = str(row.get("prompt_id", "")).casefold()
        if not prompt:
            raise ValueError(f"manifest row without prompt_id: {row!r}")
        ids.add(prompt)
    return ids


def assert_prompt_disjoint(
    train_rows: Iterable[Mapping[str, Any]], val_rows: Iterable[Mapping[str, Any]]
) -> None:
    """Fail closed when train/val prompts overlap (leaked validation)."""
    overlap = sorted(prompt_ids(train_rows) & prompt_ids(val_rows))
    if overlap:
        raise ValueError(
            f"train/val prompt overlap ({len(overlap)} prompts): {overlap[:10]}"
        )


def length_ratio_stats(ratios: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(ratios), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no length ratios to summarize")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }
