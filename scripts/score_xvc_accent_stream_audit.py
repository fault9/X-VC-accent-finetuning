#!/usr/bin/env python
"""Score every condition from ``audit_xvc_accent_streams.py`` in one pass."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

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
            raise SystemExit(f"[error] missing render manifest: {path}")
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


def _source_reference(row: dict, whisper, cache: Dict[str, tuple]):
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
    wav, sample_rate = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav, _ = maybe_resample(wav, sample_rate, int(cfg["sample_rate"]))
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(wav)).float().to(device)
        embedding, _ = model.speaker_encoder(tensor.view(1, 1, -1))
        return float(
            F.cosine_similarity(
                embedding.flatten(1), reference_embedding.flatten(1)
            ).item()
        )


def _summarize(label: str, rows: List[dict]) -> dict:
    labels = collections.Counter(row["accent_label"] for row in rows)
    return {
        "set": label,
        "n": len(rows),
        "mos_mean": _round(_mean(row["mos_pred"] for row in rows)),
        "mos_min": _round(min((row["mos_pred"] for row in rows), default=None)),
        "wer_mean": _round(_mean(row["wer"] for row in rows)),
        "sim_mean": _round(_mean(row["sim_cosine"] for row in rows)),
        "accent_depth_mean": _round(
            _mean(ACCENT_DEPTH.get(row["accent_label"], 0.0) for row in rows)
        ),
        "indian_frac": _round(
            _mean(1.0 if row["accent_label"] == "indian" else 0.0 for row in rows)
        ),
        "indian_prob_mean": _round(_mean(row["indian_prob"] for row in rows)),
        "accent_hist": " ".join(f"{k}:{v}" for k, v in labels.most_common()),
    }


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--xvc-device", type=int, default=0)
    parser.add_argument("--wer-model", default="small")
    args = parser.parse_args(argv)

    import soundfile as sf

    from eval_checkpoints import (
        AccentClassifier,
        MOSPredictor,
        Whisper,
        norm_text,
        word_error_rate,
    )
    audit_root = Path(args.audit_root)
    meta_path = audit_root / "audit_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"[error] missing audit metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    conditions = list(meta["conditions"])
    manifest_condition = meta.get("manifest_condition", conditions[0])
    if manifest_condition not in conditions:
        raise SystemExit(
            f"[error] manifest_condition {manifest_condition!r} is not in conditions"
        )
    manifest_root = audit_root / manifest_condition
    _, by_target = _read_manifests(manifest_root)

    # Match the established evaluator's memory-safe model load order.
    sim_scorer = _load_similarity_scorer(args)
    whisper = Whisper(args.wer_model, args.device)
    classifier = AccentClassifier(args.device)
    mos = MOSPredictor(args.device)
    source_cache = {}
    rows = []
    summaries = []

    for condition in conditions:
        wavs = sorted((audit_root / condition / "wavs").glob("*.wav"))
        if len(wavs) != len(by_target):
            raise SystemExit(
                f"[error] {condition}: {len(wavs)} wavs but {len(by_target)} manifest rows"
            )
        condition_rows = []
        for wav_path in wavs:
            manifest_row = by_target.get(wav_path.name)
            if manifest_row is None:
                raise SystemExit(f"[error] no manifest row for {wav_path.name}")
            reference_words, wer_mode = _source_reference(
                manifest_row, whisper, source_cache
            )
            hypothesis_words = norm_text(whisper.transcribe(str(wav_path)))
            accent_label, accent_conf, accent_posterior = classifier.classify_detailed(
                str(wav_path)
            )
            wav, sample_rate = sf.read(str(wav_path), always_2d=False)
            row = {
                "set": condition,
                "clip": wav_path.name,
                "source": manifest_row["source_wav_path"],
                "split": manifest_row["_split"],
                "mos_pred": float(mos.score(wav, sample_rate)),
                "wer": word_error_rate(reference_words, hypothesis_words),
                "wer_mode": wer_mode,
                "sim_cosine": _similarity(wav_path, sim_scorer),
                "accent_label": accent_label,
                "accent_conf": float(accent_conf),
                "indian_prob": float(accent_posterior["indian"]),
                "accent_depth": ACCENT_DEPTH.get(accent_label, 0.0),
            }
            condition_rows.append(row)
            rows.append(row)
        summary = _summarize(condition, condition_rows)
        summaries.append(summary)
        print(
            f"{condition:30s} n={summary['n']:>2} MOS={summary['mos_mean']} "
            f"WER={summary['wer_mean']} sim={summary['sim_mean']} "
            f"Indian={summary['indian_frac']} p={summary['indian_prob_mean']} "
            f"({summary['accent_hist']})"
        )

    out = Path(args.out)
    fields = [
        "set", "clip", "source", "split", "mos_pred", "wer", "wer_mode",
        "sim_cosine", "accent_label", "accent_conf", "indian_prob", "accent_depth",
    ]
    _write_csv(out / "per_clip.csv", rows, fields)
    _write_csv(out / "condition_summary.csv", summaries, list(summaries[0]))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "audit_root": str(audit_root),
                "reference": args.reference,
                "config": args.config,
                "checkpoint": args.ckpt,
                "wer_note": "sidecar reference text where available; otherwise ASR-on-source proxy",
                "accent_note": (
                    "indian_prob is the continuous CommonAccent posterior; hard labels "
                    "are secondary and all selection requires listening"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[metrics] {out / 'condition_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
