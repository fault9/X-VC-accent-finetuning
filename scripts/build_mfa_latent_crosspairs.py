#!/usr/bin/env python3
"""Build an X-VC latent cross-pair dataset from MFA TextGrid alignments.

This is the second half of the MFA-guided latent-alignment experiment.  Unlike
``align_crosspairs.py --warp-method latent``, it does not use global acoustic
DTW to create the target-frame -> source-time map.  It uses transcript-derived
MFA word/phone boundaries as monotonic guardrails, preserving both pristine
waveforms and aligning training losses through a text-pivot.

Expected sequence:

  1. Run ``prepare_mfa_crosspairs.py``.
  2. Run MFA over ``mfa_corpus/source`` and ``mfa_corpus/target``.
  3. Run this script with the resulting TextGrid directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from align_crosspairs import (
    EVAL_PROMPTS,
    HOP,
    SR,
    clean_reference_path,
    load_manifest,
    mel_dist,
    prompt_id,
    raw_target_path,
    read_wav,
    source_speaker,
    target_key,
    target_speaker,
    uniform_resample_len,
    wav_duration,
)


SILENCE_LABELS = {
    "",
    "sil",
    "sp",
    "spn",
    "<eps>",
    "<unk>",
    "{sil}",
    "silence",
    "noise",
    "breath",
}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"').strip()


def read_textgrid(path: Path) -> dict[str, list[Interval]]:
    """Parse common long/short TextGrid interval tiers without dependencies."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[Interval]] = {}

    # Long TextGrid emitted by Praat/MFA.
    blocks = re.split(r"\n\s*item \[\d+\]:", text)
    for block in blocks:
        if "IntervalTier" not in block:
            continue
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        name = name_match.group(1)
        intervals: list[Interval] = []
        for match in re.finditer(
            r"xmin\s*=\s*([0-9.eE+-]+)\s*"
            r"xmax\s*=\s*([0-9.eE+-]+)\s*"
            r"text\s*=\s*(\"(?:[^\"]|\"\")*\"|[^\n\r]+)",
            block,
            flags=re.S,
        ):
            start = float(match.group(1))
            end = float(match.group(2))
            label = _unquote(match.group(3))
            intervals.append(Interval(start, end, label))
        if intervals:
            tiers[name] = intervals

    if tiers:
        return tiers

    # Short TextGrid fallback: class/name followed by triples.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    idx = 0
    while idx < len(lines):
        if _unquote(lines[idx]) != "IntervalTier":
            idx += 1
            continue
        if idx + 4 >= len(lines):
            break
        name = _unquote(lines[idx + 1])
        idx += 4  # class, name, xmin, xmax
        intervals: list[Interval] = []
        while idx + 2 < len(lines):
            if _unquote(lines[idx]) in {"IntervalTier", "TextTier"}:
                break
            try:
                start = float(lines[idx])
                end = float(lines[idx + 1])
            except ValueError:
                break
            label = _unquote(lines[idx + 2])
            intervals.append(Interval(start, end, label))
            idx += 3
        if intervals:
            tiers[name] = intervals
    if not tiers:
        raise ValueError(f"could not parse TextGrid intervals: {path}")
    return tiers


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def choose_tier(tiers: dict[str, list[Interval]], preference: str) -> tuple[str, list[Interval]]:
    candidates = []
    pref = preference.lower()
    for name, intervals in tiers.items():
        lname = name.lower()
        if pref in lname:
            candidates.append((name, intervals))
    if not candidates and pref == "phones":
        for name, intervals in tiers.items():
            lname = name.lower()
            if "phone" in lname or "segment" in lname:
                candidates.append((name, intervals))
    if not candidates and pref == "words":
        for name, intervals in tiers.items():
            if "word" in name.lower():
                candidates.append((name, intervals))
    if not candidates:
        raise ValueError(f"no {preference!r} tier found; tiers={sorted(tiers)}")
    # Prefer the densest matching interval tier.
    return max(candidates, key=lambda item: len(item[1]))


def voiced_intervals(path: Path, preference: str) -> tuple[str, list[Interval]]:
    tiers = read_textgrid(path)
    name, intervals = choose_tier(tiers, preference)
    kept = [
        interval for interval in intervals
        if interval.end > interval.start
        and normalize_label(interval.label) not in SILENCE_LABELS
    ]
    if not kept:
        raise ValueError(f"{path}: tier {name!r} has no non-silence intervals")
    return name, kept


def aligned_intervals_with_fallback(
    source_path: Path,
    target_path: Path,
    tier: str,
    fallback_tier: str,
) -> tuple[str, str, str, list[Interval], list[Interval]]:
    """Load source/target intervals, falling back before pairing if needed.

    MFA can produce a phone tier that is technically present but all ``spn`` if
    the dictionary did not cover the utterance well enough.  In that case the
    word tier is still useful and should be used instead of skipping the pair.
    """
    errors: list[str] = []
    tier_order = [tier]
    if fallback_tier != "none" and fallback_tier not in tier_order:
        tier_order.append(fallback_tier)
    for candidate in tier_order:
        try:
            source_tier, source_intervals = voiced_intervals(source_path, candidate)
            target_tier, target_intervals = voiced_intervals(target_path, candidate)
            return (
                candidate,
                source_tier,
                target_tier,
                source_intervals,
                target_intervals,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
    raise ValueError("; ".join(errors))


def choose_anchor_mode(tier_used: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "centers" if tier_used == "words" else "boundaries"


def paired_intervals(
    source: list[Interval],
    target: list[Interval],
    min_label_match: float,
) -> tuple[list[tuple[Interval, Interval]], float]:
    """Pair intervals by order, requiring labels to mostly agree."""
    n = min(len(source), len(target))
    if n == 0:
        return [], 0.0
    pairs = [(source[i], target[i]) for i in range(n)]
    matches = [
        normalize_label(a.label) == normalize_label(b.label)
        for a, b in pairs
    ]
    match_rate = sum(matches) / len(matches)
    if match_rate < min_label_match:
        raise ValueError(
            f"label match rate {match_rate:.1%} < {min_label_match:.1%}"
        )
    return pairs, match_rate


def mfa_time_map(
    source_intervals: list[Interval],
    target_intervals: list[Interval],
    source_len: int,
    target_len: int,
    min_label_match: float,
    anchor_mode: str,
    min_local_stretch: float,
    max_local_stretch: float,
) -> tuple[list[tuple[int, int]], dict]:
    """Return target-sample -> source-sample anchors from paired MFA intervals."""
    interval_pairs, match_rate = paired_intervals(
        source_intervals, target_intervals, min_label_match
    )
    anchors: list[tuple[int, int]] = [(0, 0)]
    for src, tgt in interval_pairs:
        if anchor_mode == "centers":
            time_pairs = (
                ((tgt.start + tgt.end) / 2.0, (src.start + src.end) / 2.0),
            )
        elif anchor_mode == "boundaries":
            time_pairs = ((tgt.start, src.start), (tgt.end, src.end))
        else:
            raise ValueError(f"unsupported MFA anchor mode: {anchor_mode!r}")
        for tgt_time, src_time in time_pairs:
            target_sample = int(round(tgt_time * SR))
            source_sample = int(round(src_time * SR))
            target_sample = int(np.clip(target_sample, 0, target_len))
            source_sample = int(np.clip(source_sample, 0, source_len))
            if target_sample > anchors[-1][0] and source_sample > anchors[-1][1]:
                anchors.append((target_sample, source_sample))
    if anchors[-1] != (target_len, source_len):
        if target_len > anchors[-1][0] and source_len > anchors[-1][1]:
            anchors.append((target_len, source_len))
        else:
            # Last phone/word can legitimately reach the file end. Force the
            # endpoint after dropping any non-monotonic trailing duplicate.
            while len(anchors) > 1 and (
                anchors[-1][0] >= target_len or anchors[-1][1] >= source_len
            ):
                anchors.pop()
            anchors.append((target_len, source_len))
    if len(anchors) < 4:
        raise ValueError(f"too few MFA anchors ({len(anchors)})")
    ratios = local_ratios(anchors)
    if not np.all(np.isfinite(ratios)):
        raise ValueError("non-finite MFA local stretch ratios")
    if np.min(ratios) < min_local_stretch or np.max(ratios) > max_local_stretch:
        raise ValueError(
            "pathological MFA local stretch: "
            f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f}, "
            f"allowed=[{min_local_stretch:.3f}, {max_local_stretch:.3f}], "
            f"anchor_mode={anchor_mode}"
        )
    return anchors, {
        "interval_pairs": len(interval_pairs),
        "label_match_rate": match_rate,
        "anchors": len(anchors),
        "anchor_mode": anchor_mode,
        "local_stretch_min": float(np.min(ratios)),
        "local_stretch_max": float(np.max(ratios)),
        "local_stretch_median": float(np.median(ratios)),
    }


def local_ratios(time_map: list[tuple[int, int]]) -> np.ndarray:
    points = np.asarray(time_map, dtype=np.float64)
    delta = np.diff(points, axis=0)
    safe = np.maximum(delta[:, 0], 1.0)
    return delta[:, 1] / safe


def latent_positions_from_time_map(
    time_map: list[tuple[int, int]],
    source_len: int,
    target_len: int,
    frame_hop: int = 320,
) -> np.ndarray:
    target_frames = max(1, target_len // frame_hop)
    target_centres = (np.arange(target_frames, dtype=np.float64) + 0.5) * frame_hop
    points = np.asarray(time_map, dtype=np.float64)
    source_samples = np.interp(target_centres, points[:, 0], points[:, 1])
    positions = source_samples / float(max(source_len - 1, 1))
    return np.maximum.accumulate(np.clip(positions, 0.0, 1.0)).astype(np.float32)


def index_textgrids(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.TextGrid")) + list(root.rglob("*.textgrid"))
    index: dict[str, Path] = {}
    for path in paths:
        index[path.stem] = path
    if not index:
        raise ValueError(f"no TextGrid files under {root}")
    return index


def resolve_source_path(row: dict, source_root: Path) -> Path:
    path = Path(row["source_wav_path"])
    return path if path.is_absolute() else source_root / path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--source-align-dir", required=True)
    parser.add_argument("--target-align-dir", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--l2-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", choices=("phones", "words"), default="phones")
    parser.add_argument("--fallback-tier", choices=("none", "words"), default="words")
    parser.add_argument(
        "--mfa-anchor-mode",
        choices=("auto", "boundaries", "centers"),
        default="auto",
        help=(
            "how to convert MFA intervals into anchors; auto uses phone "
            "boundaries for phone tiers and word centres for word-tier fallback"
        ),
    )
    parser.add_argument("--min-label-match", type=float, default=0.80)
    parser.add_argument("--min-local-stretch", type=float, default=1 / 8)
    parser.add_argument("--max-local-stretch", type=float, default=8.0)
    parser.add_argument("--min-duration", type=float, default=2.6)
    parser.add_argument("--prompts-file")
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    source_root = Path(args.source_root)
    l2_root = Path(args.l2_root)
    out = Path(args.out)
    source_tg = index_textgrids(Path(args.source_align_dir))
    target_tg = index_textgrids(Path(args.target_align_dir))
    splits = {
        split: load_manifest(str(prepared_root / "selected_manifests" / f"{split}.jsonl"))
        for split in ("train", "val")
    }

    forbidden_reference_prompts = (
        {prompt_id(row) for rows in splits.values() for row in rows}
        | EVAL_PROMPTS
    )
    reference_pools: dict[str, list[Path]] = {}
    rows_out: dict[str, list[dict]] = {"train": [], "val": []}
    qc_rows: list[dict] = []
    skipped: dict[str, int] = {
        "missing_textgrid": 0,
        "bad_alignment": 0,
        "short": 0,
    }
    local_values: list[float] = []
    diagnostic_mfa: list[float] = []
    diagnostic_resample: list[float] = []
    tier_counts: dict[str, int] = {}

    for split, rows in splits.items():
        for row in rows:
            source_id = row.get("mfa_source_id", row["source_utt"])
            target_id = row.get(
                "mfa_target_id",
                row["target_utt"][:-3] if row["target_utt"].endswith("_ft") else row["target_utt"],
            )
            if source_id not in source_tg or target_id not in target_tg:
                skipped["missing_textgrid"] += 1
                print(f"  [skip] {source_id}: missing source/target TextGrid")
                continue
            src_path = resolve_source_path(row, source_root)
            raw_tgt_path = raw_target_path(row, l2_root)
            src = read_wav(str(src_path))
            tgt = read_wav(str(raw_tgt_path))
            if len(src) / SR < args.min_duration or len(tgt) / SR < args.min_duration:
                skipped["short"] += 1
                continue
            try:
                (
                    tier_used,
                    source_tier,
                    target_tier,
                    src_intervals,
                    tgt_intervals,
                ) = aligned_intervals_with_fallback(
                    source_tg[source_id],
                    target_tg[target_id],
                    args.tier,
                    args.fallback_tier,
                )
                anchor_mode = choose_anchor_mode(tier_used, args.mfa_anchor_mode)
                try:
                    time_map, stats = mfa_time_map(
                        src_intervals,
                        tgt_intervals,
                        source_len=len(src),
                        target_len=len(tgt),
                        min_label_match=args.min_label_match,
                        anchor_mode=anchor_mode,
                        min_local_stretch=args.min_local_stretch,
                        max_local_stretch=args.max_local_stretch,
                    )
                except ValueError:
                    if args.fallback_tier != "words" or args.tier == "words":
                        raise
                    source_tier, src_intervals = voiced_intervals(source_tg[source_id], "words")
                    target_tier, tgt_intervals = voiced_intervals(target_tg[target_id], "words")
                    tier_used = "words"
                    anchor_mode = choose_anchor_mode(tier_used, args.mfa_anchor_mode)
                    time_map, stats = mfa_time_map(
                        src_intervals,
                        tgt_intervals,
                        source_len=len(src),
                        target_len=len(tgt),
                        min_label_match=args.min_label_match,
                        anchor_mode=anchor_mode,
                        min_local_stretch=args.min_local_stretch,
                        max_local_stretch=args.max_local_stretch,
                    )
            except Exception as exc:
                skipped["bad_alignment"] += 1
                print(f"  [skip] {source_id}: {exc}")
                continue

            positions = latent_positions_from_time_map(time_map, len(src), len(tgt))
            pair_ratios = local_ratios(time_map)
            local_values.extend(pair_ratios.tolist())
            tier_counts[tier_used] = tier_counts.get(tier_used, 0) + 1

            # Fitted diagnostic only. We compare source mel against target mel
            # sampled through the MFA source timeline to sanity-check mapping
            # coherence. This is not used as an independent quality score.
            target_centres = (np.arange(max(1, len(tgt) // HOP), dtype=np.float64) + 0.5) * HOP
            points = np.asarray(time_map, dtype=np.float64)
            source_samples = np.interp(target_centres, points[:, 0], points[:, 1])
            warped_len = len(src)
            source_frame_positions = np.arange(max(1, int(math.ceil(warped_len / HOP)))) * HOP
            target_at_source = np.interp(source_frame_positions, source_samples, target_centres)
            # Convert frame-position diagnostic to a rough waveform via nearest
            # target samples. It is intentionally crude and metadata-only.
            sample_positions = np.interp(
                np.arange(warped_len),
                source_frame_positions[: len(target_at_source)],
                target_at_source,
                left=0,
                right=len(tgt) - 1,
            )
            diagnostic = np.interp(sample_positions, np.arange(len(tgt)), tgt).astype(np.float32)
            diagnostic_mfa.append(mel_dist(src, diagnostic))
            diagnostic_resample.append(mel_dist(src, uniform_resample_len(tgt, len(src))))

            src_out = out / "wavs" / "src" / src_path.name
            tgt_out = out / "wavs" / "tgt" / Path(row["target_wav_path"]).name
            raw_src_out = out / "wavs" / "raw_src" / src_path.name
            raw_tgt_out = (
                out / "wavs" / "raw_tgt"
                / f"{raw_tgt_path.parent.parent.name}_{raw_tgt_path.name}"
            )
            for source_file, dest in (
                (src_path, src_out),
                (raw_tgt_path, tgt_out),
                (src_path, raw_src_out),
                (raw_tgt_path, raw_tgt_out),
            ):
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(source_file, dest)

            reference = clean_reference_path(
                row,
                l2_root,
                reference_pools,
                args.min_duration,
                forbidden_reference_prompts,
            )
            ref_out = out / "wavs" / "ref" / f"{reference.parent.parent.name}_{reference.name}"
            ref_out.parent.mkdir(parents=True, exist_ok=True)
            if not ref_out.exists():
                shutil.copyfile(reference, ref_out)

            align_out = out / "alignment" / f"{row['source_utt']}.npy"
            align_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(align_out, positions, allow_pickle=False)

            rr = dict(row)
            rr["source_wav_path"] = str(src_out).replace("\\", "/")
            rr["target_wav_path"] = str(tgt_out).replace("\\", "/")
            rr["raw_source_wav_path"] = str(raw_src_out).replace("\\", "/")
            rr["raw_target_wav_path"] = str(raw_tgt_out).replace("\\", "/")
            rr["target_reference_wav_path"] = str(ref_out).replace("\\", "/")
            rr["latent_alignment_path"] = str(align_out).replace("\\", "/")
            rr["mfa_source_textgrid"] = str(source_tg[source_id]).replace("\\", "/")
            rr["mfa_target_textgrid"] = str(target_tg[target_id]).replace("\\", "/")
            rr["mfa_alignment_tier"] = tier_used
            rr["raw_source_duration"] = round(len(src) / SR, 6)
            rr["raw_target_duration"] = round(len(tgt) / SR, 6)
            rows_out[split].append(rr)
            qc_rows.append({
                "split": split,
                "source_utt": row["source_utt"],
                "target_key": target_key(row),
                "prompt_id": prompt_id(row),
                "alignment_method": "mfa_latent",
                "tier": tier_used,
                "anchor_mode": stats["anchor_mode"],
                "source_tier": source_tier,
                "target_tier": target_tier,
                "raw_source_duration": round(len(src) / SR, 6),
                "raw_target_duration": round(len(tgt) / SR, 6),
                "global_stretch_ratio": round(len(src) / len(tgt), 6),
                "interval_pairs": stats["interval_pairs"],
                "label_match_rate": round(stats["label_match_rate"], 6),
                "anchors": stats["anchors"],
                "local_stretch_min": round(stats["local_stretch_min"], 6),
                "local_stretch_max": round(stats["local_stretch_max"], 6),
                "local_stretch_median": round(stats["local_stretch_median"], 6),
                "diagnostic_mel_dist_mfa": round(diagnostic_mfa[-1], 6),
                "diagnostic_mel_dist_resample": round(diagnostic_resample[-1], 6),
                "diagnostic_mel_improvement": round(
                    diagnostic_resample[-1] - diagnostic_mfa[-1], 6
                ),
            })

    if not rows_out["train"] or not rows_out["val"]:
        raise RuntimeError(
            "MFA latent build produced an empty train or validation split"
        )

    write_jsonl(out / "manifests" / "train.jsonl", rows_out["train"])
    write_jsonl(out / "manifests" / "val.jsonl", rows_out["val"])
    write_jsonl(out / "alignment_qc.jsonl", qc_rows)
    if args.prompts_file:
        shutil.copyfile(args.prompts_file, out / "PROMPTS")

    prompts_train = {prompt_id(row) for row in rows_out["train"]}
    prompts_val = {prompt_id(row) for row in rows_out["val"]}
    meta = {
        "pairs": sum(len(rows) for rows in rows_out.values()),
        "train_pairs": len(rows_out["train"]),
        "val_pairs": len(rows_out["val"]),
        "train_prompts": len(prompts_train),
        "val_prompts": len(prompts_val),
        "prompt_overlap": len(prompts_train & prompts_val),
        "warp_method": "latent",
        "alignment_method": "mfa",
        "mfa_tier_requested": args.tier,
        "mfa_tier_counts": tier_counts,
        "minimum_raw_duration_seconds": args.min_duration,
        "skipped": skipped,
        "local_stretch_ratio": ({
            "min": round(float(np.min(local_values)), 4),
            "p01": round(float(np.percentile(local_values, 1)), 4),
            "median": round(float(np.median(local_values)), 4),
            "p99": round(float(np.percentile(local_values, 99)), 4),
            "max": round(float(np.max(local_values)), 4),
            "allowed_min": args.min_local_stretch,
            "allowed_max": args.max_local_stretch,
        } if local_values else None),
        "diagnostic_mel_dist_mfa_mean": round(float(np.mean(diagnostic_mfa)), 3),
        "diagnostic_mel_dist_resample_mean": round(float(np.mean(diagnostic_resample)), 3),
        "diagnostic_improvement": round(
            float(np.mean(diagnostic_resample) - np.mean(diagnostic_mfa)), 3
        ),
        "source_speakers": {
            split: {
                speaker: sum(source_speaker(row) == speaker for row in rows)
                for speaker in sorted({source_speaker(row) for row in rows})
            }
            for split, rows in rows_out.items()
        },
        "target_speakers": {
            split: {
                speaker: sum(target_speaker(row) == speaker for row in rows)
                for speaker in sorted({target_speaker(row) for row in rows})
            }
            for split, rows in rows_out.items()
        },
    }
    (out / "align_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nMFA latent dataset written under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
