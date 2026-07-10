#!/usr/bin/env python
"""Render a frozen AccentBridge teacher over native wavs into a standard-format
distillation dataset: SAME-TIMELINE (source -> accented-target) pairs, no DTW.

For each native clip: target = stock X-VC rendering of its bridge-edited
features (the one-line insertion), conditioned on the persona reference. The
resulting manifest trains the LoRA student through the EXISTING plain (non
latent-alignment) dataloader path — input and target are frame-aligned by
construction, so the old input-vs-target contradiction cannot occur.

Reserved eval prompts (arctic_b0002-b0012) are excluded from train; --val-frac
prompts are held out to val. Persona mode: one dataset per target speaker.

Usage (container, conda xvc):
    python scripts/make_distill_dataset.py \
        --source-glob "data/finetuning_audio/eval_sources_native/*.wav" \
        --bridge-ckpt exp/accentbridge_asi_l0plus5k/bridge.pt \
        --reference data/eval_targets/ASI.wav \
        --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
        --out data/distill_hindi_asi --device 0
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
    ap.add_argument("--source-glob", required=True, help="native wavs to render")
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--reference", required=True,
                    help="persona reference wav (conditioning + manifest field)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="ckpts/xvc.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch

    from bins.infer_utils import load_pair_as_tensors, load_xvc, to_numpy_audio
    from models.accentbridge import AccentBridge

    sources = sorted(Path(p) for p in globlib.glob(args.source_glob))
    sources = [s for s in sources
               if not any(ep in s.stem for ep in EVAL_PROMPTS)]
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("[error] no source wavs (after eval-prompt exclusion)")

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    ck = torch.load(args.bridge_ckpt, map_location="cpu")
    bridge = AccentBridge(**ck["config"])
    bridge.load_state_dict(ck["state_dict"])
    bridge.eval().to(device)
    print(f"[teacher] {bridge.extra_repr()} delta_scale={args.delta_scale}")

    @torch.inference_mode()
    def synth(source_path: str):
        src, tgt, tgt_cond = load_pair_as_tensors(
            source_path, args.reference, cfg, device,
            int(cfg["latent_hop_length"]), True)
        spk, _ = model.speaker_encoder(tgt)
        frame_cond = model.mel_extractor(tgt_cond)
        feat = model.semantic_encoder.extract_and_encode(
            src.squeeze(1))["speech_tokens"]
        sem = model.semantic_encoder.embed_ids(feat)
        sem = model.semantic_adapter(sem.transpose(1, 2))
        sem = (sem + args.delta_scale * bridge.delta(sem)).transpose(1, 2)
        z = model.acoustic_encoder(src)
        zq = model.acoustic_quantizer(z)[0]
        combined = torch.cat([sem, zq.transpose(1, 2)], dim=2)
        x = model.prenet(combined.transpose(1, 2), spk)
        x = model.acoustic_converter(x, frame_cond, spk)
        return to_numpy_audio(model.acoustic_decoder(x))

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
    meta = {"teacher": args.bridge_ckpt, "delta_scale": args.delta_scale,
            "reference": args.reference, "n_train": len(rows["train"]),
            "n_val": len(rows["val"]), "eval_prompts_excluded": sorted(EVAL_PROMPTS)}
    with open(out / "distill_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[distill-data] train={len(rows['train'])} val={len(rows['val'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
