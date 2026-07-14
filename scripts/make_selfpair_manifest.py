#!/usr/bin/env python
"""Build a SELF-PAIR dataset (source == target == a real recording) from the
target side of existing cross-pair manifests.

Two uses (see CHANGES.md):
  * recon-only baseline: train a LoRA on nothing but real persona audio -- the
    "no pairs at all" arm (supervisor hypothesis: normal LoRA on the acoustic
    should transfer accent).
  * real-texture anchor: mix a capped set of self-pairs into a distill student
    so its reconstruction anchor is REAL audio instead of teacher renders --
    the first texture lever that does not route through the renderer.

Self-pairs are trivially same-timeline, so they train through the plain
(non latent-alignment) dataloader path; with source == target every batch is a
clean reconstruction regardless of reconstruction_ratio.

Rows are deduped by wav (many cross-pairs share one target rendition), the
reserved eval prompts (arctic_b0002-b0012) are excluded, and every referenced
wav must exist -- missing files fail the build loudly.

Usage (container, conda xvc):
    python scripts/make_selfpair_manifest.py \
        --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/train.jsonl \
        --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/val.jsonl \
        --reference data/eval_targets/ASI.wav \
        --out data/asi_selfpairs_wide
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

EVAL_PROMPTS = {f"b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}
_PROMPT = re.compile(r"arctic_([ab]\d{4})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-manifest", required=True, action="append",
                    help="cross-pair train/val.jsonl to harvest target wavs "
                         "from; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="data/eval_targets/ASI.wav",
                    help="pinned persona reference wav (manifest field)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap unique utterances after dedup+filtering "
                         "(0 = keep all); use to control the mix ratio when "
                         "unioning with a distill set")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--min-duration", type=float, default=2.0,
                    help="drop utterances shorter than this (seconds); "
                         "nothing downstream re-checks non-crosspair datasets")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    if not Path(args.reference).is_file():
        raise SystemExit(f"[error] reference wav missing: {args.reference}")

    # Harvest: prefer the raw (unprocessed) target rendition when present.
    pool = {}  # wav path -> known duration (or None)
    for mpath in args.from_manifest:
        for line in open(mpath, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            wav = row.get("raw_target_wav_path") or row["target_wav_path"]
            dur = row.get("raw_target_duration")
            if wav not in pool or pool[wav] is None:
                pool[wav] = dur

    missing = [w for w in pool if not Path(w).is_file()]
    if missing:
        for w in sorted(missing)[:10]:
            print(f"  MISSING {w}", file=sys.stderr)
        raise SystemExit(
            f"[error] {len(missing)}/{len(pool)} harvested wavs do not exist "
            "on disk (run from the repo root of the machine that holds the "
            "dataset)")

    kept, dropped_eval, dropped_short = [], 0, 0
    for wav in sorted(pool):
        m = _PROMPT.search(Path(wav).stem)
        if m and m.group(1) in EVAL_PROMPTS:
            dropped_eval += 1
            continue
        dur = pool[wav]
        if dur is None and args.min_duration > 0:
            import soundfile as sf
            info = sf.info(wav)
            dur = info.frames / info.samplerate
            pool[wav] = dur
        if args.min_duration > 0 and dur is not None and dur < args.min_duration:
            dropped_short += 1
            continue
        kept.append(wav)

    if not kept:
        raise SystemExit("[error] no utterances survived filtering")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.limit > 0:
        kept = kept[: args.limit]
    n_val = max(1, int(len(kept) * args.val_frac))
    splits = {"val": kept[:n_val], "train": kept[n_val:]}

    out = Path(args.out)
    man_dir = out / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    minutes = 0.0
    for split in ("train", "val"):
        with open(man_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for wav in sorted(splits[split]):
                stem = Path(wav).stem
                row = {
                    "source_utt": stem,
                    "source_wav_path": wav.replace("\\", "/"),
                    "target_utt": f"{stem}_ft",   # dataloader strips last 3
                    "target_wav_path": wav.replace("\\", "/"),
                    "target_reference_wav_path":
                        str(args.reference).replace("\\", "/"),
                }
                if pool[wav] is not None:
                    row["raw_source_duration"] = pool[wav]
                    row["raw_target_duration"] = pool[wav]
                    minutes += pool[wav] / 60.0
                f.write(json.dumps(row) + "\n")

    meta = {"from_manifests": args.from_manifest, "reference": args.reference,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "minutes_known": round(minutes, 2), "limit": args.limit,
            "min_duration": args.min_duration, "seed": args.seed,
            "dropped_eval_prompts": dropped_eval,
            "dropped_short": dropped_short}
    with open(out / "selfpair_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[selfpairs] {len(pool)} harvested -> {len(kept)} kept "
          f"(dropped {dropped_eval} eval-prompt, {dropped_short} short) "
          f"~{minutes:.1f} min")
    print(f"[selfpairs] train={len(splits['train'])} val={len(splits['val'])} "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
