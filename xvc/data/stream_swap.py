"""Pure helpers for causal X-VC semantic/acoustic stream-swap audits.

MFA phone intervals are used only to construct a diagnostic time map.  The
helpers in this module never alter training audio, manifests, or model weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def resolve_audio_path(
    recorded_path: str | Path, search_roots: Sequence[str | Path]
) -> tuple[Path, bool]:
    """Resolve a stale manifest path without silently choosing another clip.

    The original path wins when it exists. Otherwise, exact-basename matches
    are collected below ``search_roots``. Multiple byte-identical copies are
    safe; conflicting copies fail closed because choosing one would invalidate
    a causal audio comparison.
    """

    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded, False

    candidates: list[Path] = []
    for root_value in search_roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        candidates.extend(path for path in root.rglob(recorded.name) if path.is_file())
    candidates = sorted(set(path.resolve() for path in candidates))
    if not candidates:
        roots = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            f"audio path is stale and {recorded.name!r} was not found under: {roots}"
        )
    if len(candidates) == 1:
        return candidates[0], True

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        return hasher.hexdigest()

    hashes = {digest(path) for path in candidates}
    if len(hashes) != 1:
        formatted = "\n  ".join(str(path) for path in candidates)
        raise ValueError(
            f"ambiguous non-identical copies for {recorded.name!r}:\n  {formatted}"
        )
    return candidates[0], True


def build_phone_frame_map(
    source_frames: int,
    target_frames: int,
    phone_segments: Sequence[Mapping[str, Any]],
    *,
    source_annotation_frames: int | None = None,
    target_annotation_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Map every source frame to a (fractional) target-frame position.

    ``phone_segments`` contains matched, monotonic phone intervals with
    half-open ``src=[start, end]`` and ``tgt=[start, end]`` frame ranges.  The
    segment boundaries and centres become interpolation anchors.  Regions not
    covered by matched phones are filled monotonically between neighbouring
    anchors; the first and last frames are always pinned.

    Annotation frame counts may differ slightly from freshly encoded stream
    lengths.  In that case anchors are scaled before interpolation.
    """

    if source_frames < 2 or target_frames < 2:
        raise ValueError("source_frames and target_frames must both be >= 2")
    if not phone_segments:
        raise ValueError("at least one matched phone segment is required")

    src_ann = int(source_annotation_frames or source_frames)
    tgt_ann = int(target_annotation_frames or target_frames)
    if src_ann < 2 or tgt_ann < 2:
        raise ValueError("annotation frame counts must both be >= 2")

    def scale_src(value: float) -> float:
        return value * (source_frames - 1) / (src_ann - 1)

    def scale_tgt(value: float) -> float:
        return value * (target_frames - 1) / (tgt_ann - 1)

    anchors: list[tuple[float, float]] = [
        (0.0, 0.0),
        (float(source_frames - 1), float(target_frames - 1)),
    ]
    covered = np.zeros(source_frames, dtype=bool)
    accepted = 0

    for segment in phone_segments:
        try:
            src_start, src_end = (int(x) for x in segment["src"])
            tgt_start, tgt_end = (int(x) for x in segment["tgt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid phone segment: {segment!r}") from exc

        if src_end <= src_start or tgt_end <= tgt_start:
            continue

        src_last = src_end - 1
        tgt_last = tgt_end - 1
        anchors.extend(
            [
                (scale_src(src_start), scale_tgt(tgt_start)),
                (
                    scale_src((src_start + src_last) / 2.0),
                    scale_tgt((tgt_start + tgt_last) / 2.0),
                ),
                (scale_src(src_last), scale_tgt(tgt_last)),
            ]
        )

        cov_start = max(0, min(source_frames, int(round(scale_src(src_start)))))
        cov_end = max(
            cov_start + 1,
            min(source_frames, int(round(scale_src(src_end - 1))) + 1),
        )
        covered[cov_start:cov_end] = True
        accepted += 1

    if accepted == 0:
        raise ValueError("no positive-duration matched phone segments")

    anchors.sort(key=lambda pair: (pair[0], pair[1]))
    grouped: list[tuple[float, float]] = []
    duplicate_source_anchors = 0
    for src_pos, tgt_pos in anchors:
        src_pos = float(np.clip(src_pos, 0, source_frames - 1))
        tgt_pos = float(np.clip(tgt_pos, 0, target_frames - 1))
        if grouped and abs(grouped[-1][0] - src_pos) < 1e-7:
            old_src, old_tgt = grouped[-1]
            grouped[-1] = (old_src, max(old_tgt, tgt_pos))
            duplicate_source_anchors += 1
        else:
            grouped.append((src_pos, tgt_pos))

    src_anchor = np.asarray([pair[0] for pair in grouped], dtype=np.float64)
    tgt_anchor_raw = np.asarray([pair[1] for pair in grouped], dtype=np.float64)
    # Endpoint pins are controls, not suggestions.  A phone that begins at
    # source frame zero may begin after target leading silence; it must not
    # displace the explicit (0, 0) audit anchor during duplicate grouping.
    tgt_anchor_raw[0] = 0.0
    tgt_anchor_raw[-1] = float(target_frames - 1)
    tgt_anchor = np.maximum.accumulate(tgt_anchor_raw)
    monotonic_repairs = int(np.count_nonzero(tgt_anchor != tgt_anchor_raw))
    tgt_anchor = np.clip(tgt_anchor, 0, target_frames - 1)

    positions = np.interp(
        np.arange(source_frames, dtype=np.float64), src_anchor, tgt_anchor
    )
    positions = np.clip(positions, 0, target_frames - 1).astype(np.float32)
    if np.any(np.diff(positions) < -1e-6):
        raise AssertionError("constructed phone map is not monotonic")

    slopes = np.diff(positions)
    diagnostics: dict[str, float | int] = {
        "matched_phones": accepted,
        "anchors": int(len(grouped)),
        "duplicate_source_anchors": duplicate_source_anchors,
        "monotonic_anchor_repairs": monotonic_repairs,
        "source_phone_coverage": float(covered.mean()),
        "slope_min": float(slopes.min()),
        "slope_median": float(np.median(slopes)),
        "slope_p99": float(np.quantile(slopes, 0.99)),
        "slope_max": float(slopes.max(initial=0.0)),
    }
    return positions, diagnostics
