#!/usr/bin/env python
"""Annotate AccentBridge shards with genuine phone-tier supervision metadata.

MFA is used only to locate corresponding phone intervals. Audio and feature
streams are never warped or resampled. The output retains each original shard
item and adds ``phone_segments`` containing native-source and target frame
spans, confidence, and a mean-one realization weight.

This is deliberately fail-closed: both TextGrids must contain a real phone
tier. Word-tier fallback is forbidden because that experiment was already run
and performed worse than DTW.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from phone_supervision import align_phone_sequences, index_textgrids, phone_tier


def _frame_range(interval, duration, frames):
    lo = max(0, min(frames, int(round(interval.start / duration * frames))))
    hi = max(lo + 1, min(frames, int(round(interval.end / duration * frames))))
    return lo, hi


def _trim_range(lo, hi, fraction, max_frames, min_frames):
    """Adaptively remove uncertain boundaries without erasing short phones."""
    n = hi - lo
    if n < min_frames:
        return None
    margin = min(max_frames, int(round(n * fraction)))
    if n - 2 * margin < min_frames:
        margin = 0
    return lo + margin, hi - margin


def _phase_stats(segment):
    """Duration-invariant onset/middle/offset means plus global std."""
    import torch
    import torch.nn.functional as F

    if segment.shape[-1] >= 3:
        phase = F.adaptive_avg_pool1d(segment.unsqueeze(0), 3)[0]
    else:
        phase = segment.mean(dim=-1, keepdim=True).expand(-1, 3)
    std = segment.std(dim=-1, unbiased=False)
    return phase, std


def _resolve_textgrid(index, meta, side):
    """Resolve both raw L2-ARCTIC and derived cross-pair naming schemes."""
    candidates = [Path(meta[f"{side}_wav_path"]).stem,
                  str(meta.get(f"{side}_utt", ""))]
    prompt = str(meta.get("prompt", ""))
    speaker = str(meta.get(f"{side}_speaker", ""))
    if prompt and speaker:
        if side == "target":
            speaker = speaker.split("__", 1)[0]
        prompt = prompt if prompt.startswith("arctic_") else f"arctic_{prompt}"
        candidates.append(f"{speaker}_{prompt}")
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    raise ValueError(f"missing {side} TextGrid; tried {candidates}")


def annotate_item(item, src_grid: Path, tgt_grid: Path, args):
    import torch

    src_tier, src_phones, src_duration = phone_tier(src_grid, args.tier)
    tgt_tier, tgt_phones, tgt_duration = phone_tier(tgt_grid, args.tier)
    pairs, match_rate = align_phone_sequences(src_phones, tgt_phones)
    if match_rate < args.min_label_match:
        raise ValueError(f"phone label match {match_rate:.1%} < {args.min_label_match:.1%}")

    source = item["sem_adapted_src"].float()
    target = item["sem_adapted_tgt"].float()
    segments = []
    rejected_duration = rejected_short = 0

    for src_phone, tgt_phone, label in pairs:
        duration_ratio = ((src_phone.end - src_phone.start)
                          / max(tgt_phone.end - tgt_phone.start, 1e-8))
        if not args.min_duration_ratio <= duration_ratio <= args.max_duration_ratio:
            rejected_duration += 1
            continue
        src_range = _trim_range(
            *_frame_range(src_phone, src_duration, source.shape[-1]),
            args.boundary_trim_fraction, args.max_boundary_trim_frames,
            args.min_phone_frames)
        tgt_range = _trim_range(
            *_frame_range(tgt_phone, tgt_duration, target.shape[-1]),
            args.boundary_trim_fraction, args.max_boundary_trim_frames,
            args.min_phone_frames)
        if src_range is None or tgt_range is None:
            rejected_short += 1
            continue
        s0, s1 = src_range
        t0, t1 = tgt_range

        src_phase, src_std = _phase_stats(source[:, s0:s1])
        tgt_phase, tgt_std = _phase_stats(target[:, t0:t1])
        phase_gap = (src_phase - tgt_phase).pow(2).mean()
        std_gap = (src_std - tgt_std).pow(2).mean()
        gap = torch.sqrt(phase_gap + args.std_gap_weight * std_gap + 1e-8)
        confidence = min(duration_ratio, 1.0 / duration_ratio)
        segments.append({
            "phone": label,
            "src": [s0, s1],
            "tgt": [t0, t1],
            "duration_ratio": round(duration_ratio, 6),
            "confidence": round(confidence, 6),
            "gap": round(float(gap), 6),
        })

    if len(segments) < args.min_matched_phones:
        raise ValueError(f"only {len(segments)} usable matched phones")
    raw = torch.tensor([x["gap"] * x["confidence"] for x in segments])
    raw = (raw / raw.mean().clamp(min=1e-8)).clamp(args.min_weight, args.max_weight)
    raw /= raw.mean().clamp(min=1e-8)
    for segment, weight in zip(segments, raw.tolist()):
        segment["weight"] = round(weight, 6)

    out = dict(item)
    out["phone_segments"] = segments
    out["phone_meta"] = {
        "source_textgrid": str(src_grid),
        "target_textgrid": str(tgt_grid),
        "source_tier": src_tier,
        "target_tier": tgt_tier,
        "matched_phones": len(segments),
        "label_match_rate": round(match_rate, 6),
        "rejected_duration": rejected_duration,
        "rejected_short": rejected_short,
        "source_was_warped": False,
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-root", required=True,
                    help="extracted AccentBridge root containing train/ and val/")
    ap.add_argument("--source-align-dir", required=True)
    ap.add_argument("--target-align-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", choices=("phones", "phone"), default="phones")
    ap.add_argument("--min-label-match", type=float, default=0.90)
    ap.add_argument("--min-pair-coverage", type=float, default=0.80)
    ap.add_argument("--boundary-trim-fraction", type=float, default=0.15)
    ap.add_argument("--max-boundary-trim-frames", type=int, default=1)
    ap.add_argument("--min-phone-frames", type=int, default=2)
    ap.add_argument("--min-matched-phones", type=int, default=5)
    ap.add_argument("--min-duration-ratio", type=float, default=0.5)
    ap.add_argument("--max-duration-ratio", type=float, default=2.0)
    ap.add_argument("--std-gap-weight", type=float, default=0.25)
    ap.add_argument("--min-weight", type=float, default=0.25)
    ap.add_argument("--max-weight", type=float, default=4.0)
    ap.add_argument("--shard-size", type=int, default=32)
    args = ap.parse_args(argv)

    import torch

    source_grids = index_textgrids(Path(args.source_align_dir))
    target_grids = index_textgrids(Path(args.target_align_dir))
    pairs_root, out_root = Path(args.pairs_root), Path(args.out)
    if out_root.exists() and any(out_root.rglob("shard_*.pt")):
        raise SystemExit(f"[error] {out_root} already contains output shards; "
                         "choose a new --out directory to avoid mixing runs")
    qc_rows, failures, total, kept = [], [], 0, 0
    split_counts = {}
    phone_gaps = defaultdict(list)

    for split in ("train", "val"):
        in_dir, out_dir = pairs_root / split, out_root / split
        if not in_dir.is_dir():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        output, shard_idx = [], 0
        split_total, split_kept = 0, 0

        def flush():
            nonlocal output, shard_idx
            if output:
                torch.save(output, out_dir / f"shard_{shard_idx:04d}.pt")
                output, shard_idx = [], shard_idx + 1

        with (out_dir / "pairs_manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for shard in sorted(in_dir.glob("shard_*.pt")):
                for item in torch.load(shard, map_location="cpu"):
                    total += 1
                    split_total += 1
                    meta = item["meta"]
                    try:
                        source_grid = _resolve_textgrid(source_grids, meta, "source")
                        target_grid = _resolve_textgrid(target_grids, meta, "target")
                        annotated = annotate_item(item, source_grid, target_grid, args)
                        for phone in annotated["phone_segments"]:
                            phone_gaps[phone["phone"]].append(phone["gap"])
                        pm = annotated["phone_meta"]
                        qc_rows.append({"split": split, "source_utt": meta["source_utt"],
                                        "target_utt": meta["target_utt"], **pm})
                        manifest.write(json.dumps({
                            "shard": f"shard_{shard_idx:04d}.pt", "index": len(output),
                            **meta, **pm,
                        }) + "\n")
                        output.append(annotated)
                        kept += 1
                        split_kept += 1
                        if len(output) >= args.shard_size:
                            flush()
                    except Exception as exc:
                        failures.append({"split": split, "source_utt": meta.get("source_utt"),
                                         "target_utt": meta.get("target_utt"), "error": str(exc)})
            flush()
        split_counts[split] = {"seen": split_total, "kept": split_kept,
                               "coverage": round(split_kept / max(split_total, 1), 6)}

    coverage = kept / max(total, 1)
    splits_pass = all(v["seen"] and v["coverage"] >= args.min_pair_coverage
                      for v in split_counts.values()) and set(split_counts) == {"train", "val"}
    summary = {
        "status": "pass" if total and coverage >= args.min_pair_coverage and splits_pass else "fail",
        "pairs_seen": total, "pairs_kept": kept, "pairs_failed": len(failures),
        "pair_coverage": round(coverage, 6),
        "minimum_pair_coverage": args.min_pair_coverage,
        "splits": split_counts,
        "tier_policy": "genuine phone tier only; word fallback forbidden",
        "alignment_policy": "MFA spans only; source audio/features are never warped",
        "phone_gap_mean": {k: round(sum(v) / len(v), 6)
                           for k, v in sorted(phone_gaps.items())},
        "failures": failures[:50],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "phone_supervision_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if qc_rows:
        with (out_root / "phone_supervision_qc.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=qc_rows[0].keys())
            writer.writeheader()
            writer.writerows(qc_rows)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        print("[error] phone-supervision coverage gate failed; do not train", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
