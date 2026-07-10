#!/usr/bin/env python
"""Render a fine-tuned X-VC checkpoint (the accented "teacher") over native wavs
into a standard-format distillation dataset: SAME-TIMELINE (source ->
accented-target) pairs, no DTW.

Self-distill counterpart of make_distill_dataset.py: instead of inserting an
AccentBridge delta, the teacher is a LoRA fine-tune checkpoint (e.g. the
early-stopped latent-objective model that acquired the accent) doing its normal
offline conversion against the pinned persona reference. The student then
trains on these pairs through the EXISTING plain (non latent-alignment)
dataloader path -- input and target are frame-aligned by construction.

Reserved eval prompts (arctic_b0002-b0012) are excluded from train; --val-frac
prompts are held out to val. Persona mode: one dataset per target speaker.

Usage (container, conda xvc):
    python scripts/make_selfdistill_dataset.py \
        --run-dir exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5 \
        --step 100 \
        --config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml \
        --source-glob "data/distill_sources_asi/*.wav" \
        --reference data/eval_targets/ASI.wav \
        --out data/selfdistill_hindi_asi --device 0

GATE before training a student: run the accent classifier + your ears over a
few wavs in <out>/wavs -- if the teacher renders lost the accent shift, do not
spend a student run on them.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

EVAL_PROMPTS = {f"arctic_b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="fine-tune run dir containing ckpt/<step>.pt")
    ap.add_argument("--step", type=int, required=True,
                    help="teacher checkpoint step (e.g. 100)")
    ap.add_argument("--config", required=True,
                    help="the run's finetune config (defines the LoRA topology "
                         "so the checkpoint loads by exact key match)")
    ap.add_argument("--source-glob", required=True, help="native wavs to render")
    ap.add_argument("--reference", required=True,
                    help="persona reference wav (conditioning + manifest field)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf

    from bins.infer_utils import (load_pair_as_tensors, load_xvc, run_offline,
                                  to_numpy_audio)

    ckpt_path = Path(args.run_dir) / "ckpt" / f"{args.step}.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"[error] teacher checkpoint not found: {ckpt_path}")

    sources = sorted(Path(p) for p in globlib.glob(args.source_glob))
    sources = [s for s in sources
               if not any(ep in s.stem for ep in EVAL_PROMPTS)]
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("[error] no source wavs (after eval-prompt exclusion)")

    cfg, model, device = load_xvc(args.config, str(ckpt_path), args.device, False)
    print(f"[teacher] {ckpt_path} (LoRA topology from {args.config})")

    def synth(source_path: str):
        src, tgt, tgt_cond = load_pair_as_tensors(
            source_path, args.reference, cfg, device,
            int(cfg["latent_hop_length"]), True)
        return to_numpy_audio(run_offline(model, src, tgt, tgt_cond))

    out = Path(args.out)
    wav_dir = out / "wavs"
    man_dir = out / "manifests"
    wav_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["sample_rate"])

    # Prompt-level val holdout (never split one prompt across train/val).
    prompts = sorted({m.group(1) for s in sources
                      for m in [re.search(r"(arctic_[ab]\d{4})", s.stem)] if m})
    random.Random(args.seed).shuffle(prompts)
    val_prompts = set(prompts[: max(1, int(len(prompts) * args.val_frac))])

    rows = {"train": [], "val": []}
    for i, s in enumerate(sources):
        wav = synth(str(s))
        tgt_path = wav_dir / f"{s.stem}__teacher.wav"
        sf.write(str(tgt_path), np.asarray(wav, dtype="float32"), sr)
        m = re.search(r"(arctic_[ab]\d{4})", s.stem)
        split = "val" if (m and m.group(1) in val_prompts) else "train"
        rows[split].append({
            "source_utt": s.stem,
            "source_wav_path": str(s).replace("\\", "/"),
            "target_utt": f"{s.stem}__teacher_ft",   # dataloader strips last 3
            "target_wav_path": str(tgt_path).replace("\\", "/"),
            "target_reference_wav_path": str(args.reference).replace("\\", "/"),
        })
        if (i + 1) % 25 == 0:
            print(f"  rendered {i + 1}/{len(sources)}")

    for split, rs in rows.items():
        with open(man_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    meta = {"teacher_run": str(args.run_dir), "teacher_step": args.step,
            "teacher_config": args.config, "reference": args.reference,
            "n_train": len(rows["train"]), "n_val": len(rows["val"]),
            "eval_prompts_excluded": sorted(EVAL_PROMPTS)}
    with open(out / "distill_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[selfdistill-data] train={len(rows['train'])} val={len(rows['val'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
