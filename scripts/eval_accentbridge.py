#!/usr/bin/env python
"""Evaluate a trained AccentBridge at the feature level, and optionally
synthesize a few wavs through the real X-VC stack with the bridge inserted.

Feature metrics (val shards, no audio, CPU-fine):
  * pre/post distance to the L2 target (cosine + L2), gap-closed fraction
  * identity drift: bridge(L2 features) vs L2 features — must stay ~0
  * per speaker-pair breakdown
  * delta magnitude stats (how hard the bridge works)

Optional synthesis (--synthesize N, needs --config/--ckpt): runs the model's
own inference path with ONE change — `sem_emb = bridge(sem_emb)` after the
semantic_adapter, i.e. exactly the runtime insertion point documented in
docs/streaming_accentbridge_plan.md (bins/infer_utils.py run_stream_chunk_forward).
Sources are the pair's ORIGINAL native wavs on their natural timeline (the true
deployment regime); the reference is the pair's L2 clip. Wavs land in
--out/samples/ for listening and for the standard eval stack.

Usage:
    python scripts/eval_accentbridge.py \
        --val-dir data/accentbridge_pairs/val \
        --bridge-ckpt exp/accentbridge_l0/bridge.pt \
        --out exp/accentbridge_l0/eval \
        [--synthesize 6 --config configs/xvc.yaml --ckpt ckpts/xvc.pt --device 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # scripts/ -> train_accentbridge
sys.path.insert(0, str(_HERE.parent))   # repo root -> bins, models


def _mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthesize", type=int, default=0,
                    help="also render N val pairs to wav through X-VC + bridge")
    ap.add_argument("--config", default=None, help="X-VC config (for --synthesize)")
    ap.add_argument("--ckpt", default=None, help="X-VC checkpoint (for --synthesize)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from models.accentbridge import AccentBridge

    ck = torch.load(args.bridge_ckpt, map_location="cpu")
    bridge = AccentBridge(**ck["config"])
    bridge.load_state_dict(ck["state_dict"])
    bridge.eval()
    print(f"[bridge] {bridge.extra_repr()}")

    from train_accentbridge import load_pairs  # same shard reader
    items = load_pairs(Path(args.val_dir), args.limit)
    if not items:
        raise SystemExit("[error] no val pairs")

    rows = []
    with torch.inference_mode():
        for it in items:
            s = it["sem_adapted_src_warped"].float().unsqueeze(0)
            t = it["sem_adapted_tgt"].float().unsqueeze(0)
            e, d = bridge(s, return_delta=True)
            idd = (bridge(t) - t).norm(dim=1).mean().item() / \
                (t.norm(dim=1).mean().item() + 1e-8)
            m = it["meta"]
            rows.append({
                "pair": f"{m['source_utt']}->{m['target_utt']}",
                "speaker_pair": f"{m['source_speaker']}->{m['target_speaker']}",
                "pre_cos": round(F.cosine_similarity(s, t, dim=1).mean().item(), 4),
                "post_cos": round(F.cosine_similarity(e, t, dim=1).mean().item(), 4),
                "pre_l2": round((s - t).norm(dim=1).mean().item(), 4),
                "post_l2": round((e - t).norm(dim=1).mean().item(), 4),
                "delta_rel": round(d.norm(dim=1).mean().item()
                                   / (s.norm(dim=1).mean().item() + 1e-8), 4),
                "identity_drift": round(idd, 5),
            })

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "per_pair.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups["ALL"].append(r)
        groups[r["speaker_pair"]].append(r)
    summary = {}
    print(f"\n{'group':<14} {'n':>3} {'pre_cos':>8} {'post_cos':>9} "
          f"{'gap_closed':>10} {'id_drift':>9} {'delta':>7}")
    for g, sub in groups.items():
        pre, post = _mean([r["pre_l2"] for r in sub]), _mean([r["post_l2"] for r in sub])
        summary[g] = {
            "n": len(sub),
            "pre_cos": _mean([r["pre_cos"] for r in sub]),
            "post_cos": _mean([r["post_cos"] for r in sub]),
            "pre_l2": pre, "post_l2": post,
            "gap_closed_l2": round(1 - post / max(pre, 1e-8), 4),
            "identity_drift": _mean([r["identity_drift"] for r in sub]),
            "delta_rel": _mean([r["delta_rel"] for r in sub]),
        }
        s = summary[g]
        print(f"{g:<14} {s['n']:>3} {s['pre_cos']:>8} {s['post_cos']:>9} "
              f"{s['gap_closed_l2']:>10} {s['identity_drift']:>9} {s['delta_rel']:>7}")
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.synthesize > 0:
        if not (args.config and args.ckpt):
            raise SystemExit("[error] --synthesize needs --config and --ckpt")
        from bins.infer_utils import load_pair_as_tensors, load_xvc, to_numpy_audio
        import numpy as np
        import soundfile as sf

        cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
        bridge_d = bridge.to(device)
        sample_dir = out_root / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        @torch.inference_mode()
        def synth(source_path, ref_path, use_bridge: bool):
            src, tgt, tgt_cond = load_pair_as_tensors(
                source_path, ref_path, cfg, device, int(cfg["latent_hop_length"]), True)
            spk, _ = model.speaker_encoder(tgt)
            frame_cond = model.mel_extractor(tgt_cond)
            feat = model.semantic_encoder.extract_and_encode(
                src.squeeze(1))["speech_tokens"]
            sem = model.semantic_encoder.embed_ids(feat)
            sem = model.semantic_adapter(sem.transpose(1, 2))       # (B, 1024, T50)
            if use_bridge:
                sem = bridge_d(sem)                                  # THE insertion
            sem = sem.transpose(1, 2)
            z = model.acoustic_encoder(src)
            zq = model.acoustic_quantizer(z)[0]
            combined = torch.cat([sem, zq.transpose(1, 2)], dim=2)
            x = model.prenet(combined.transpose(1, 2), spk)
            x = model.acoustic_converter(x, frame_cond, spk)
            return to_numpy_audio(model.acoustic_decoder(x))

        for it in items[: args.synthesize]:
            m = it["meta"]
            for tag, ub in (("bridged", True), ("plain", False)):
                wav = synth(m["source_wav_path"], m["target_wav_path"], ub)
                sf.write(str(sample_dir / f"{m['source_utt']}__{tag}.wav"),
                         np.asarray(wav, dtype="float32"), int(cfg["sample_rate"]))
            print(f"[synth] {m['source_utt']}: bridged + plain written")
        print(f"[synth] wavs in {sample_dir} — score them with the standard "
              "eval/calibration stack for MOS/accent/sim")

    print(f"[eval] wrote {out_root}/per_pair.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
