#!/usr/bin/env python
"""Supervision-quality diagnostic for latent-aligned accent training — decode
what the loss actually sees, through the REAL training code path, no training.

Context / why this exists
-------------------------
Calibration (scripts/calibrate_eval_floor.py) showed the raw L2 corpus is clean
(MOS ~3.8-3.9) while every fine-tune bottoms out at ~2.2-2.5 regardless of rank,
target set, LR, or recon anchor. So the artifact is introduced between "clean
corpus clip" and "model output". In THIS codebase the DTW map warps the
SOURCE-side frozen streams (semantic embedding + quantized acoustic latents of
the pristine native clip, resampled onto the L2 timeline inside
`XVC._latent_aligned_source`); the loss target is the pristine L2 waveform.
The untested link is therefore the WARPED-INPUT REGIME: can the frozen stock
stack even produce clean audio from time-warped input streams, or does training
chase an unreachable target through corrupted inputs?

IMPORTANT prior result (do not treat alignment fixes as pre-validated): a
phone-level MFA alignment was already attempted (`crosspair_hindi_mfa_latent_200`).
It silently fell back to WORD-tier alignment (final mfa_tier_counts were words
only), its mel-dist QC was WORSE than uniform resampling (~13.77 vs 12.84), and
its training eval was poor (MOS ~1.8-2.34, degraded WER, almost no Indian-accent
signal). So if this diagnostic implicates alignment, the conclusion is "invest
in a genuinely better alignment method (proper phone-tier MFA, soft/attention
alignment, confidence-weighted maps)" — not "re-run the failed word-tier MFA".

What it does
------------
Runs the STOCK (or any given) checkpoint over real manifest pairs using the
actual training dataset class + collate + `XVC.forward` (the same code path the
trainer uses — nothing reimplemented), under two forced conditions:

  * ``identity_recon``   (reconstruction_ratio=1.0): the dataloader's clean
    reconstruction branch — the L2 clip's OWN latents with an identity map.
    This is the unwarped encode/decode control.
  * ``warped_crosspair`` (reconstruction_ratio=0.0): the accent-pressure branch
    — native-source latents DTW-resampled onto the L2 timeline, exactly the
    inputs the fine-tunes train on.
  * ``target_segment``: the pristine L2 target segment itself (the supervision
    audio), scored directly as the anchor.

Each decoded waveform is scored with the calibration metric stack (UTMOS MOS,
CommonAccent label, ERes2Net cosine to the run's own reference embedding) plus
``mel_dist`` to the target segment — the SAME 40-mel L1 metric as the dataset
QC diagnostics in align_crosspairs.py, so values are comparable to the
diagnostic_mel_dist numbers in dataset meta (e.g. 12.84 resample / 13.77 MFA).

Built-in verification (since conditions are forced via config, not reimplemented):
the first batch of each condition hard-asserts `role_assignment_mode`, asserts
identity-condition alignment positions are linear (identity map), and reports
how far warped-condition positions deviate from a straight line.

Interpretation
--------------
  * identity_recon HIGH MOS, warped_crosspair LOW MOS
      -> the alignment/warped-input regime is the artifact source. Fine-tuning
         cannot fix this from any config knob; alignment must improve first
         (proper phone-tier MFA or another method — word-tier MFA already
         failed, see above).
  * BOTH high MOS (trained outputs still low)
      -> inputs and round trip are clean; the artifact emerges from training
         dynamics/objective (e.g. regression-only drift with the adversarial
         term off) -> move to objective/feature-matching fixes.
  * identity_recon ITSELF low MOS
      -> decoder/codec round-trip or segment/collate machinery degrades audio
         (unlikely: base conversions score ~3.7 through model.inference — but
         if seen, the latent-training machinery itself is implicated).

Usage (container, conda xvc, repo root):
    python scripts/decode_latent_targets.py \
        --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml \
        --ckpt ckpts/xvc.pt \
        --data-root data/crosspair_hindi_latent_400 \
        --data-root data/crosspair_hindi_latent_800_strict \
        --data-root data/crosspair_hindi_mfa_latent_200 \
        --split val --max-pairs 24 \
        --out exp/decode_latent_targets

Missing --data-root dirs are skipped with a warning (800_strict / mfa_latent_200
are optional). WER is deliberately not computed: items are fixed 2.4 s training
segments cut mid-utterance, so no full reference text applies.

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # scripts/ -> eval_checkpoints, align_crosspairs
sys.path.insert(0, str(_HERE.parent))   # repo root -> bins, models, utils

from eval_checkpoints import AccentClassifier, MOSPredictor, sha256_file  # noqa: E402

CONDITIONS = (("warped_crosspair", 0.0), ("identity_recon", 1.0))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _std(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 4)


def _linear_deviation(pos) -> float:
    """Max |positions - straight line through endpoints| — 0 for an identity map."""
    import numpy as np
    p = np.asarray(pos, dtype=np.float64)
    if p.size < 3:
        return 0.0
    line = np.linspace(p[0], p[-1], p.size)
    return float(np.abs(p - line).max())


def _build_dataset(base_cfg, manifest: str, split: str, recon_ratio: float):
    """The trainer's own dataset/collate, with the condition forced via config."""
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "datasets": {"train": [manifest], "val": [manifest]},
            "dataloader": {
                "reconstruction_ratio": float(recon_ratio),
                "reversed_ratio": 0.0,
                # determinism / single-process hygiene for a diagnostic:
                "partition": False,
                "list_shuffle": False,
                "shuffle": False,
                "cycle": 1,
                "static": {"batch_size": 1},
            },
        },
    )
    sampler = hydra.utils.instantiate(cfg["dataloader"], cfg, mode=split)
    return sampler.sample()


def _verify_batch(batch, condition: str, utt: str) -> Dict:
    """Hard-assert the forced condition actually took the intended branch."""
    expected_role = "reconstruction" if condition == "identity_recon" else "standard"
    roles = batch["role_assignment_mode"]
    if any(r != expected_role for r in roles):
        raise RuntimeError(
            f"[verify] condition {condition!r} produced role(s) {roles} "
            f"(expected {expected_role!r}) — dataloader branch mismatch")
    pos = batch["latent_alignment_positions"]
    if pos is None:
        raise RuntimeError(f"[verify] condition {condition!r} has no alignment positions")
    dev = _linear_deviation(pos[0].cpu().numpy())
    if condition == "identity_recon" and dev > 1e-3:
        raise RuntimeError(
            f"[verify] identity map deviates from linear by {dev:.5f} (utt={utt}) "
            "— identity-alignment branch is not producing an identity map")
    return {"utt": utt, "role": expected_role, "linear_deviation": round(dev, 6)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="a latent-alignment training config (defines dataloader/model)")
    ap.add_argument("--ckpt", default="ckpts/xvc.pt",
                    help="checkpoint to decode through (default: stock ckpts/xvc.pt)")
    ap.add_argument("--data-root", action="append", required=True,
                    help="dataset root containing manifests/<split>.jsonl; repeatable; "
                         "missing roots are skipped with a warning")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--max-pairs", type=int, default=24, help="pairs per dataset per condition")
    ap.add_argument("--out", default="exp/decode_latent_targets")
    ap.add_argument("--skip-accent-clf", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    from align_crosspairs import mel_dist  # same 40-mel L1 as dataset QC diagnostics
    from bins.infer_utils import load_xvc, to_numpy_audio
    from utils.file import load_config

    cfg_model, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    sr = int(cfg_model["sample_rate"])
    cuda = "cuda" if torch.cuda.is_available() else "cpu"
    mos_model = MOSPredictor(cuda)
    accent_clf = None if args.skip_accent_clf else AccentClassifier(cuda)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    verify_info: Dict[str, dict] = {}
    fieldnames = ["dataset", "condition", "utt", "mos_pred", "sim_cosine",
                  "mel_dist_to_target", "accent_label", "accent_conf", "dur_s"]

    def spk_sim(wav_np, ref_emb) -> float:
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(wav_np)).float().to(device)
            emb, _ = model.speaker_encoder(t.view(1, 1, -1))
            return round(float(F.cosine_similarity(
                emb.flatten(1), ref_emb.flatten(1)).item()), 4)

    def score(wav_np, wav_path: Path, ref_emb, target_np):
        sf.write(str(wav_path), np.asarray(wav_np, dtype="float32"), sr)
        alabel = aconf = None
        if accent_clf:
            alabel, aconf = accent_clf.classify(str(wav_path))
            aconf = round(aconf, 4)
        return {
            "mos_pred": round(mos_model.score(wav_np, sr), 3),
            "sim_cosine": spk_sim(wav_np, ref_emb),
            "mel_dist_to_target": (
                round(mel_dist(wav_np.astype(np.float32),
                               target_np.astype(np.float32)), 3)
                if target_np is not None else None),
            "accent_label": alabel, "accent_conf": aconf,
            "dur_s": round(len(wav_np) / sr, 3),
        }

    for root in args.data_root:
        root = Path(root)
        name = root.name
        manifest = root / "manifests" / f"{args.split}.jsonl"
        if not manifest.is_file():
            print(f"[skip] {manifest} not found — skipping dataset {name}", file=sys.stderr)
            continue
        base_cfg = load_config(args.config)

        for condition, ratio in CONDITIONS:
            ds = _build_dataset(base_cfg, str(manifest), args.split, ratio)
            sample_dir = out_root / "samples" / name / condition
            sample_dir.mkdir(parents=True, exist_ok=True)
            tgt_dir = out_root / "samples" / name / "target_segment"
            tgt_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== {name} / {condition} (ratio={ratio}, split={args.split}) ===")
            n_done = 0
            for batch in ds:
                if n_done >= args.max_pairs:
                    break
                utt = batch["index"][0]
                if n_done == 0:
                    verify_info[f"{name}/{condition}"] = _verify_batch(batch, condition, utt)
                    print(f"  [verify] role={verify_info[f'{name}/{condition}']['role']} "
                          f"linear_deviation={verify_info[f'{name}/{condition}']['linear_deviation']}")

                model_batch = {k: (v.to(device) if hasattr(v, "to") else v)
                               for k, v in batch.items()}
                # Mirror the trainer's batch preparation (see bins/smoke_test.py).
                model_batch["source_wav"] = model_batch["source_wav"].unsqueeze(1)
                model_batch["target_wav"] = model_batch["target_wav"].unsqueeze(1)
                model_batch["step"] = 0
                with torch.no_grad():
                    outputs = model(model_batch)

                recons_np = to_numpy_audio(outputs["recons"])
                target_np = to_numpy_audio(model_batch["target_wav"])
                ref_emb = outputs["sim_feat"]  # the run's own reference embedding

                rows.append({"dataset": name, "condition": condition, "utt": utt,
                             **score(recons_np, sample_dir / f"{utt}.wav",
                                     ref_emb, target_np)})
                # Score the pristine supervision segment once (during warped pass).
                if condition == "warped_crosspair":
                    rows.append({"dataset": name, "condition": "target_segment",
                                 "utt": utt,
                                 **score(target_np, tgt_dir / f"{utt}.wav",
                                         ref_emb, None)})
                r = rows[-2] if condition == "warped_crosspair" else rows[-1]
                print(f"  {utt:<28} mos={r['mos_pred']} sim={r['sim_cosine']} "
                      f"mel_dist={r['mel_dist_to_target']} accent={r['accent_label']}")
                n_done += 1
            if n_done == 0:
                raise RuntimeError(f"no usable pairs from {manifest} ({condition})")

    if not rows:
        raise SystemExit("[error] no dataset produced rows (all --data-root missing?)")

    with open(out_root / "per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summaries = []
    keys = sorted({(r["dataset"], r["condition"]) for r in rows})
    for ds_name, cond in keys:
        sub = [r for r in rows if r["dataset"] == ds_name and r["condition"] == cond]
        hist: Dict[str, int] = {}
        for r in sub:
            if r["accent_label"]:
                hist[r["accent_label"]] = hist.get(r["accent_label"], 0) + 1
        summaries.append({
            "dataset": ds_name, "condition": cond, "n": len(sub),
            "mos_mean": _mean([r["mos_pred"] for r in sub]),
            "mos_std": _std([r["mos_pred"] for r in sub]),
            "sim_mean": _mean([r["sim_cosine"] for r in sub]),
            "mel_dist_mean": _mean([r["mel_dist_to_target"] for r in sub]),
            "indian_frac": _mean([1.0 if r["accent_label"] == "indian" else 0.0
                                  for r in sub if r["accent_label"]]),
            "accent_hist": " ".join(f"{k}:{v}" for k, v in
                                    sorted(hist.items(), key=lambda kv: -kv[1])[:5]),
        })
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "ckpt_sha256": sha256_file(Path(args.ckpt)),
            "verification": verify_info,
            "interpretation": {
                "identity_high_warped_low":
                    "alignment/warped-input regime is the artifact source -> "
                    "better alignment needed (word-tier MFA already failed; "
                    "phone-tier or another method)",
                "both_high":
                    "supervision inputs are clean -> artifact comes from training "
                    "dynamics/objective -> feature-matching / objective fixes",
                "identity_low":
                    "latent-training round trip itself degrades audio -> inspect "
                    "codec/decoder + segment machinery",
            },
        }, f, indent=2, default=str)

    print(f"\n[decode-latent-targets] wrote {out_root}/per_clip.csv, summary.csv")
    for s in summaries:
        print(f"  {s['dataset']:<32} {s['condition']:<18} n={s['n']:<3} "
              f"mos={s['mos_mean']} (std {s['mos_std']}) sim={s['sim_mean']} "
              f"mel_dist={s['mel_dist_mean']} [{s['accent_hist']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
