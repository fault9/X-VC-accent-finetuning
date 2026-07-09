#!/usr/bin/env python
"""Extract frame-aligned native/L2 representation pairs from the REAL X-VC
encoders — the training data (and go/no-go evidence) for the AccentBridge.

For every cross-pair manifest row this script encodes BOTH sides with the
model's own modules (no hand-rolled approximations):

  * `tokens`       : WhisperVQ speech tokens (12.5 Hz discrete ids)
  * `sem_adapted`  : post-semantic_adapter stream, (1024, T@50Hz) — the
                     AccentBridge insertion point (bins/infer_utils.py:121)
  * `zq`           : quantized acoustic latents (1024, T@50Hz), optional

and aligns the NATIVE side onto the L2 timeline with the dataset's own stored
DTW map, using the model's own samplers (`XVC._sample_bct_at_normalized_positions`
for continuous streams, nearest-index for ids) — i.e. exactly the machinery
training uses in `_latent_aligned_source`.

Output: .pt shards (fp16) + pairs_manifest.jsonl under --out/<split>/.
Each shard item holds: sem_adapted_src (native timeline), sem_adapted_src_warped
(L2 timeline), sem_adapted_tgt, tokens_src / tokens_src_warped / tokens_tgt,
optional zq_src_warped / zq_tgt, positions, and meta (utts, speakers, prompt,
wav paths — so downstream tools can synthesize).

Usage (container, conda xvc, repo root — CPU works, GPU faster):
    python scripts/extract_accentbridge_pairs.py \
        --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml \
        --ckpt ckpts/xvc.pt \
        --data-root data/crosspair_hindi_latent_400 \
        --split val --limit 40 --device 0 --out data/accentbridge_pairs

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def prompt_of(name: str):
    m = re.search(r"arctic_([ab]\d{4})", name)
    return m.group(1) if m else None


def speaker_of(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_arctic_")[0] if "_arctic_" in stem else Path(path).parent.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="ckpts/xvc.pt")
    ap.add_argument("--data-root", action="append", required=True,
                    help="dataset root with manifests/<split>.jsonl; repeatable")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--limit", type=int, default=None, help="max pairs per dataset")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="data/accentbridge_pairs")
    ap.add_argument("--shard-size", type=int, default=32)
    ap.add_argument("--no-zq", action="store_true", help="skip acoustic latents")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from bins.infer_utils import load_xvc
    from models.codec.sac.model import XVC
    from models.codec.sac.utils import process_audio

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    hop = int(cfg["latent_hop_length"])

    @torch.inference_mode()
    def encode(wav_path: str):
        wav_np = process_audio(wav_path, cfg, hop)
        wav = torch.from_numpy(wav_np).view(1, 1, -1).float().to(device)
        tokens = model.semantic_encoder.extract_and_encode(
            wav.squeeze(1))["speech_tokens"]                      # (1, T12.5)
        emb = model.semantic_encoder.embed_ids(tokens)            # (1, T12.5, 1280)
        adapted = model.semantic_adapter(
            emb.transpose(1, 2))                                  # (1, 1024, T50)
        zq = None
        if not args.no_zq:
            z = model.acoustic_encoder(wav)
            zq = model.acoustic_quantizer(z)[0]                   # (1, 1024, T50)
        return tokens, adapted, zq

    def warp_ids(ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Nearest-index resample of discrete ids at normalized positions."""
        n = ids.shape[-1]
        idx = (pos.clamp(0, 1) * (n - 1)).round().long()
        return ids[:, idx]

    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = open(out_root / "pairs_manifest.jsonl", "w", encoding="utf-8")

    shard, shard_idx, n_done, n_skipped = [], 0, 0, 0

    def flush():
        nonlocal shard, shard_idx
        if shard:
            torch.save(shard, out_root / f"shard_{shard_idx:04d}.pt")
            shard = []
            shard_idx += 1

    for root in args.data_root:
        root = Path(root)
        mpath = root / "manifests" / f"{args.split}.jsonl"
        if not mpath.is_file():
            print(f"[skip] {mpath} not found", file=sys.stderr)
            continue
        rows = [json.loads(l) for l in open(mpath, encoding="utf-8") if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        print(f"=== {root.name}/{args.split}: {len(rows)} pair(s) ===")

        for row in rows:
            try:
                apath = row.get("latent_alignment_path")
                if not apath:
                    raise ValueError("row has no latent_alignment_path")
                pos_np = np.load(apath, allow_pickle=False).astype(np.float32)
                pos = torch.from_numpy(pos_np).view(1, -1).to(device)

                tok_s, sem_s, zq_s = encode(row["source_wav_path"])
                tok_t, sem_t, zq_t = encode(row["target_wav_path"])

                # Trim the map and target streams to a common L2-timeline length.
                T = min(pos.shape[-1], sem_t.shape[-1],
                        zq_t.shape[-1] if zq_t is not None else pos.shape[-1])
                pos_T = pos[:, :T]
                sem_sw = XVC._sample_bct_at_normalized_positions(sem_s, pos_T)
                tok_sw = warp_ids(tok_s, pos_T[0])
                zq_sw = (XVC._sample_bct_nearest(zq_s, pos_T)
                         if zq_s is not None else None)

                item = {
                    "sem_adapted_src": sem_s[0].half().cpu(),
                    "sem_adapted_src_warped": sem_sw[0, :, :T].half().cpu(),
                    "sem_adapted_tgt": sem_t[0, :, :T].half().cpu(),
                    "tokens_src": tok_s[0].cpu(),
                    "tokens_src_warped": tok_sw[0].cpu(),
                    "tokens_tgt": tok_t[0].cpu(),
                    "positions": pos_T[0].cpu(),
                    "meta": {
                        "dataset": root.name, "split": args.split,
                        "source_utt": row["source_utt"],
                        "target_utt": row["target_utt"],
                        "source_wav_path": row["source_wav_path"],
                        "target_wav_path": row["target_wav_path"],
                        "source_speaker": speaker_of(row["source_wav_path"]),
                        "target_speaker": speaker_of(row["target_wav_path"]),
                        "prompt": prompt_of(row["source_wav_path"]),
                    },
                }
                if zq_sw is not None:
                    item["zq_src_warped"] = zq_sw[0, :, :T].half().cpu()
                    item["zq_tgt"] = zq_t[0, :, :T].half().cpu()

                manifest.write(json.dumps({
                    "shard": f"shard_{shard_idx:04d}.pt", "index": len(shard),
                    "T_frames": int(T), **item["meta"],
                }) + "\n")
                shard.append(item)
                if len(shard) >= args.shard_size:
                    flush()
                n_done += 1
            except Exception as e:  # noqa: BLE001 — skip bad rows, keep going
                n_skipped += 1
                print(f"[skip] {row.get('source_utt')}: {e}", file=sys.stderr)

    flush()
    manifest.close()
    print(f"[extract] {n_done} pair(s) -> {out_root} ({shard_idx} shard(s), "
          f"{n_skipped} skipped)")
    if n_done == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
