#!/usr/bin/env python
"""Score, filter, and optionally gate self-distillation teacher renders.

The historical mode remains available::

    python scripts/gate_teacher_renders.py data/sd_probe_*/wavs

That mode prints MOS/accent summaries for quick listening probes.  Production
rebuilds should also pass ``--candidate-root`` and ``--filtered-root``.  In
that mode the script:

* scores the candidate and baseline on the same carriers with the evaluation
  stack (UTMOS, CommonAccent, Whisper content agreement, and ERes2Net speaker
  similarity);
* rejects individual candidate renders below the configured quality/content/
  identity floors;
* writes filtered train/val manifests that still point at the original wavs;
* writes ``per_clip.csv``, ``summary.csv``, ``rejected.csv`` and ``gate.json``;
* exits non-zero if the retained set fails an aggregate gate.

WER uses the source clip's sidecar transcript when one exists.  Otherwise it
uses ASR(source) as the reference and reports ``asr_source_proxy``.  This does
not estimate linguistic WER against ground truth, but it does detect content
that the deeper teacher changed relative to the native carrier--the relevant
failure mode before distillation.

The accent classifier is a noisy proxy.  For the paired depth gate, labels are
mapped to the project's established ordinal diagnostic: indian=2, england=1,
other=0.  The gate and raw labels are persisted so this assumption is visible;
human listening remains the final accent-depth decision.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob as globlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

ACCENT_DEPTH = {"indian": 2.0, "england": 1.0}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _read_manifests(root: Path) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    splits: Dict[str, List[dict]] = {}
    by_target: Dict[str, dict] = {}
    for split in ("train", "val"):
        path = root / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"[error] missing candidate manifest: {path}")
        rows = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_split"] = split
            row["_manifest_line"] = lineno
            target_name = Path(row["target_wav_path"]).name
            if target_name in by_target:
                raise SystemExit(
                    f"[error] duplicate target basename {target_name!r} in {root}"
                )
            by_target[target_name] = row
            rows.append(row)
        splits[split] = rows
    return splits, by_target


def _source_reference(row: dict, whisper, cache: Dict[str, Tuple[List[str], str]]):
    from eval_checkpoints import norm_text

    source = Path(row["source_wav_path"])
    key = str(source)
    if key in cache:
        return cache[key]
    if not source.is_file():
        raise FileNotFoundError(f"source wav missing: {source}")
    sidecar = source.with_suffix(".txt")
    if sidecar.is_file():
        result = (norm_text(sidecar.read_text(encoding="utf-8")), "ref_text")
    else:
        result = (norm_text(whisper.transcribe(str(source))), "asr_source_proxy")
    cache[key] = result
    return result


def _load_similarity_scorer(args):
    if not (args.reference and args.config and args.ckpt):
        return None

    import torch
    from bins.infer_utils import load_pair_as_tensors, load_xvc, precompute_conditions

    cfg, model, device = load_xvc(args.config, args.ckpt, args.xvc_device, False)
    hop = int(cfg["latent_hop_length"])
    _, target, target_cond = load_pair_as_tensors(
        args.reference, args.reference, cfg, device, hop, True
    )
    with torch.inference_mode():
        reference_embedding, _ = precompute_conditions(model, target, target_cond)
    return cfg, model, device, reference_embedding


def _similarity(path: Path, scorer) -> float:
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from eval_checkpoints import maybe_resample

    cfg, model, device, reference_embedding = scorer
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav, _ = maybe_resample(wav, sr, int(cfg["sample_rate"]))
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(wav)).float().to(device)
        embedding, _ = model.speaker_encoder(tensor.view(1, 1, -1))
        return float(
            F.cosine_similarity(
                embedding.flatten(1), reference_embedding.flatten(1)
            ).item()
        )


def _summarize(label: str, rows: List[dict]) -> dict:
    labels = collections.Counter(r["accent_label"] for r in rows)
    return {
        "set": label,
        "n": len(rows),
        "mos_mean": _round(_mean(r["mos_pred"] for r in rows)),
        "mos_min": _round(min((r["mos_pred"] for r in rows), default=None)),
        "wer_mean": _round(_mean(r["wer"] for r in rows)),
        "sim_mean": _round(_mean(r["sim_cosine"] for r in rows)),
        "accent_depth_mean": _round(
            _mean(ACCENT_DEPTH.get(r["accent_label"], 0.0) for r in rows)
        ),
        "indian_frac": _round(
            _mean(1.0 if r["accent_label"] == "indian" else 0.0 for r in rows)
        ),
        "accent_hist": " ".join(f"{k}:{v}" for k, v in labels.most_common()),
    }


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_manifest_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _write_filtered_manifests(
    filtered_root: Path,
    splits: Dict[str, List[dict]],
    accepted_names: set,
    report: dict,
) -> Dict[str, int]:
    manifest_dir = filtered_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in splits.items():
        kept = [r for r in rows if Path(r["target_wav_path"]).name in accepted_names]
        counts[split] = len(kept)
        with (manifest_dir / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in kept:
                handle.write(json.dumps(_clean_manifest_row(row)) + "\n")
    (filtered_root / "teacher_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+", help="directories of rendered wavs")
    ap.add_argument("--candidate-root", default=None,
                    help="dataset root containing candidate manifests; enables filtering")
    ap.add_argument("--filtered-root", default=None,
                    help="output dataset root for filtered manifests")
    ap.add_argument("--out", default="exp/teacher_render_gate")
    ap.add_argument("--candidate-dir", default=None,
                    help="candidate wav dir (default: final positional dir)")
    ap.add_argument("--baseline-dir", default=None,
                    help="baseline wav dir (default: first positional dir when >1)")
    ap.add_argument("--reference", default=None, help="pinned target reference wav")
    ap.add_argument("--config", default=None, help="X-VC config for similarity")
    ap.add_argument("--ckpt", default=None, help="X-VC checkpoint for similarity")
    ap.add_argument("--limit", type=int, default=40,
                    help="max wavs per dir; 0 scores all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--xvc-device", type=int, default=0)
    ap.add_argument("--wer-model", default="small")
    ap.add_argument("--worst", type=int, default=3)

    ap.add_argument("--clip-mos-min", type=float, default=2.8)
    ap.add_argument("--clip-wer-max", type=float, default=0.10)
    ap.add_argument("--clip-sim-min", type=float, default=0.65)
    ap.add_argument("--aggregate-mos-min", type=float, default=2.8)
    ap.add_argument("--aggregate-wer-max", type=float, default=0.06)
    ap.add_argument("--aggregate-sim-min", type=float, default=0.65)
    ap.add_argument("--min-accent-depth-gain", type=float, default=0.05)
    ap.add_argument("--min-retained-count", type=int, default=100)
    ap.add_argument("--min-retained-fraction", type=float, default=0.50)
    ap.add_argument("--min-retained-val", type=int, default=5)
    args = ap.parse_args(argv)

    import soundfile as sf
    from eval_checkpoints import (
        AccentClassifier, MOSPredictor, Whisper, norm_text, word_error_rate,
    )

    candidate_dir = Path(args.candidate_dir or args.dirs[-1])
    baseline_dir = Path(
        args.baseline_dir or (args.dirs[0] if len(args.dirs) > 1 else args.dirs[-1])
    )
    production_mode = bool(args.candidate_root or args.filtered_root)
    if production_mode and not (args.candidate_root and args.filtered_root):
        raise SystemExit("[error] --candidate-root and --filtered-root are required together")
    if production_mode and not (args.reference and args.config and args.ckpt):
        raise SystemExit(
            "[error] production gate requires --reference, --config, and --ckpt "
            "so speaker similarity cannot be silently omitted"
        )

    splits: Dict[str, List[dict]] = {}
    by_target: Dict[str, dict] = {}
    if production_mode:
        splits, by_target = _read_manifests(Path(args.candidate_root))

    # Match eval_checkpoints.py's load order.  Loading X-VC first avoids asking
    # the largest model to find one contiguous allocation after three metric
    # models have fragmented the GPU allocator.
    sim_scorer = _load_similarity_scorer(args) if production_mode else None
    whisper = Whisper(args.wer_model, args.device) if production_mode else None
    clf = AccentClassifier(args.device)
    mos = MOSPredictor(args.device)
    source_cache: Dict[str, Tuple[List[str], str]] = {}
    all_rows: List[dict] = []

    dir_labels: Dict[str, str] = {}
    for directory in args.dirs:
        path = Path(directory)
        if path.resolve() == candidate_dir.resolve():
            dir_labels[directory] = "candidate"
        elif path.resolve() == baseline_dir.resolve():
            dir_labels[directory] = "baseline"
        else:
            dir_labels[directory] = path.parent.name or path.name

    for directory in args.dirs:
        label = dir_labels[directory]
        wavs = sorted(Path(p) for p in globlib.glob(str(Path(directory) / "*.wav")))
        if args.limit > 0:
            wavs = wavs[: args.limit]
        if not wavs:
            raise SystemExit(f"[error] {directory}: no wavs")
        scored = []
        for path in wavs:
            row = by_target.get(path.name)
            source = split = wer_mode = None
            wer = sim = None
            errors = []
            if production_mode:
                if row is None:
                    errors.append("missing_manifest_row")
                else:
                    source = row["source_wav_path"]
                    split = row["_split"]
                    try:
                        ref_words, wer_mode = _source_reference(row, whisper, source_cache)
                        hyp_words = norm_text(whisper.transcribe(str(path)))
                        wer = word_error_rate(ref_words, hyp_words)
                    except Exception as exc:
                        errors.append(f"wer_error:{type(exc).__name__}")
                    try:
                        sim = _similarity(path, sim_scorer)
                    except Exception as exc:
                        errors.append(f"sim_error:{type(exc).__name__}")

            accent_label, accent_conf = clf.classify(str(path))
            wav, sr = sf.read(str(path), always_2d=False)
            mos_pred = float(mos.score(wav, sr))
            result = {
                "set": label,
                "clip": path.name,
                "source": source,
                "split": split,
                "mos_pred": mos_pred,
                "wer": wer,
                "wer_mode": wer_mode,
                "sim_cosine": sim,
                "accent_label": accent_label,
                "accent_conf": float(accent_conf),
                "accent_depth": ACCENT_DEPTH.get(accent_label, 0.0),
                "score_errors": ";".join(errors),
            }
            scored.append(result)
            all_rows.append(result)

        summary = _summarize(label, scored)
        print(f"\n== {directory} ({len(scored)} wavs; {label}) ==")
        print(f"  labels: {summary['accent_hist']}")
        print(
            f"  MOS mean {summary['mos_mean']} min {summary['mos_min']}  "
            f"WER {summary['wer_mean']}  sim {summary['sim_mean']}  "
            f"depth {summary['accent_depth_mean']}"
        )
        for item in sorted(scored, key=lambda r: r["mos_pred"])[: args.worst]:
            print(
                f"    worst {item['mos_pred']:.2f}  "
                f"{item['accent_label']:10s}  {item['clip']}"
            )

    if not production_mode:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidate_rows = [r for r in all_rows if r["set"] == "candidate"]
    baseline_rows = [r for r in all_rows if r["set"] == "baseline"]
    if len(candidate_rows) != len(by_target):
        raise SystemExit(
            f"[error] scored {len(candidate_rows)} candidate wavs but manifests "
            f"contain {len(by_target)} rows; use --limit 0 and ensure the render is complete"
        )

    rejected = []
    accepted = []
    for row in candidate_rows:
        reasons = []
        if row["score_errors"]:
            reasons.extend(row["score_errors"].split(";"))
        if row["mos_pred"] < args.clip_mos_min:
            reasons.append(f"mos<{args.clip_mos_min:g}")
        if row["wer"] is None:
            reasons.append("missing_wer")
        elif row["wer"] > args.clip_wer_max:
            reasons.append(f"wer>{args.clip_wer_max:g}")
        if row["sim_cosine"] is None:
            reasons.append("missing_similarity")
        elif row["sim_cosine"] < args.clip_sim_min:
            reasons.append(f"sim<{args.clip_sim_min:g}")
        if reasons:
            rejected.append({**row, "rejection_reasons": ";".join(sorted(set(reasons)))})
        else:
            accepted.append(row)

    accepted_names = {r["clip"] for r in accepted}
    baseline_by_name = {r["clip"]: r for r in baseline_rows}
    missing_baseline_names = sorted(accepted_names - set(baseline_by_name))
    paired_baseline = [baseline_by_name[n] for n in accepted_names if n in baseline_by_name]
    accepted_summary = _summarize("candidate_retained", accepted)
    baseline_summary = _summarize("baseline_paired_to_retained", paired_baseline)
    raw_summaries = [
        _summarize("baseline_all", baseline_rows),
        _summarize("candidate_all", candidate_rows),
        baseline_summary,
        accepted_summary,
    ]
    depth_gain = None
    if (
        accepted_summary["accent_depth_mean"] is not None
        and baseline_summary["accent_depth_mean"] is not None
    ):
        depth_gain = (
            accepted_summary["accent_depth_mean"]
            - baseline_summary["accent_depth_mean"]
        )

    retained_fraction = len(accepted) / len(candidate_rows) if candidate_rows else 0.0
    val_retained = sum(1 for r in accepted if r["split"] == "val")
    failures = []
    if missing_baseline_names:
        failures.append(
            f"baseline_missing {len(missing_baseline_names)} retained candidate clip(s)"
        )
    if len(accepted) < args.min_retained_count:
        failures.append(
            f"retained_count {len(accepted)} < {args.min_retained_count}"
        )
    if retained_fraction < args.min_retained_fraction:
        failures.append(
            f"retained_fraction {retained_fraction:.3f} < {args.min_retained_fraction:.3f}"
        )
    if val_retained < args.min_retained_val:
        failures.append(f"retained_val {val_retained} < {args.min_retained_val}")
    if (
        accepted_summary["mos_mean"] is None
        or accepted_summary["mos_mean"] < args.aggregate_mos_min
    ):
        failures.append(
            f"retained_mos {accepted_summary['mos_mean']} < {args.aggregate_mos_min:g}"
        )
    if (
        accepted_summary["wer_mean"] is None
        or accepted_summary["wer_mean"] > args.aggregate_wer_max
    ):
        failures.append(
            f"retained_wer {accepted_summary['wer_mean']} > {args.aggregate_wer_max:g}"
        )
    if (
        accepted_summary["sim_mean"] is None
        or accepted_summary["sim_mean"] < args.aggregate_sim_min
    ):
        failures.append(
            f"retained_sim {accepted_summary['sim_mean']} < {args.aggregate_sim_min:g}"
        )
    if depth_gain is None or depth_gain < args.min_accent_depth_gain:
        failures.append(
            f"paired_accent_depth_gain {_round(depth_gain)} < "
            f"{args.min_accent_depth_gain:g}"
        )

    report = {
        "status": "pass" if not failures else "fail",
        "candidate_root": args.candidate_root,
        "filtered_root": args.filtered_root,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "thresholds": {
            "clip_mos_min": args.clip_mos_min,
            "clip_wer_max": args.clip_wer_max,
            "clip_sim_min": args.clip_sim_min,
            "aggregate_mos_min": args.aggregate_mos_min,
            "aggregate_wer_max": args.aggregate_wer_max,
            "aggregate_sim_min": args.aggregate_sim_min,
            "min_accent_depth_gain": args.min_accent_depth_gain,
            "min_retained_count": args.min_retained_count,
            "min_retained_fraction": args.min_retained_fraction,
            "min_retained_val": args.min_retained_val,
        },
        "n_candidate": len(candidate_rows),
        "n_retained": len(accepted),
        "n_rejected": len(rejected),
        "retained_fraction": round(retained_fraction, 4),
        "retained_val": val_retained,
        "baseline_missing_for_retained": missing_baseline_names,
        "paired_accent_depth_gain": _round(depth_gain),
        "summaries": raw_summaries,
        "failures": failures,
        "accent_depth_mapping": ACCENT_DEPTH,
        "note": (
            "Accent depth is a noisy classifier proxy; a passing gate permits "
            "training but does not replace blinded listening."
        ),
    }

    counts = _write_filtered_manifests(
        Path(args.filtered_root), splits, accepted_names, report
    )
    report["filtered_manifest_counts"] = counts
    (Path(args.filtered_root) / "teacher_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    fields = [
        "set", "clip", "source", "split", "mos_pred", "wer", "wer_mode",
        "sim_cosine", "accent_label", "accent_conf", "accent_depth", "score_errors",
    ]
    _write_csv(out / "per_clip.csv", all_rows, fields)
    _write_csv(out / "summary.csv", raw_summaries, list(raw_summaries[0].keys()))
    _write_csv(out / "rejected.csv", rejected, fields + ["rejection_reasons"])
    (out / "gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n== production teacher gate ==")
    print(
        f"  retained {len(accepted)}/{len(candidate_rows)} "
        f"({retained_fraction:.1%}); val={val_retained}; rejected={len(rejected)}"
    )
    print(
        f"  retained MOS={accepted_summary['mos_mean']} "
        f"WER={accepted_summary['wer_mean']} sim={accepted_summary['sim_mean']} "
        f"paired depth gain={_round(depth_gain)}"
    )
    print(f"  reports: {out}")
    print(f"  filtered manifests: {args.filtered_root}/manifests")
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}", file=sys.stderr)
        candidate_name = Path(args.candidate_root).name
        raise SystemExit(
            f"[error] {candidate_name} teacher gate failed; refusing to train student"
        )

    candidate_name = Path(args.candidate_root).name
    print(f"  PASS: filtered {candidate_name} teacher set is eligible for student training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
