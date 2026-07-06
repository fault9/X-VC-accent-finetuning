#!/usr/bin/env python3
"""
Build parallel cross-pair training data for accent conversion (Option B pilot).

Direction being taught: NATIVE speech in -> ACCENTED speech out.
Each training example pairs a native speaker's recording of an ARCTIC prompt
(source) with an L2 speaker's recording of the SAME prompt (target), with the
target uniformly time-stretched (pitch-preserving WSOLA) to the source's
duration so X-VC's frame-aligned losses (mel / SSL-feature / sim) see
approximately aligned frames. Phone-level misalignment remains (+-100ms) —
that is the accepted crudeness of the B1 pilot; B2 upgrades to phone-level
forced alignment if B1 shows an accent signal.

Pairs are gender-matched by construction (pass matching speakers).

    python scripts/build_crosspairs.py \
        --pair bdl:ASI --pair rms:ASI --pair clb:TNI --pair slt:TNI \
        --out data/crosspair_hindi --per-source 500

Outputs:
    out/wavs/src/<src>_<prompt>.wav          (copied source)
    out/wavs/tgt/<tgt>__<src>_<prompt>.wav   (stretched target)
    out/manifests/{train,val}.jsonl          (X-VC pair manifests)
    out/build_meta.json                      (counts, stretch stats, sha256s)

Part of the X-VC accent fine-tuning pipeline (upstream Jerrister/X-VC, MIT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Raw corpus roots (same layout as configs/data_groups.yaml sources).
CMU_GLOB = "cmu_us_{spk}_arctic/wav"
DEFAULT_CMU_ROOT = "D:/datasets/arctic-audio/cmu"
DEFAULT_L2_ROOT = "D:/datasets/arctic-audio/l2arctic"

# Prompts reserved for evaluation sources (data/eval_sources) — never train on
# their text content so eval stays unseen-content.
EVAL_PROMPTS = {f"arctic_b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}

SR = 16000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def wsola_stretch(x: np.ndarray, alpha: float, win_ms: int = 50, tol_ms: int = 7) -> np.ndarray:
    """Pitch-preserving uniform time-stretch by factor alpha (output ~= alpha*len).

    Plain WSOLA: overlap-add hann windows at fixed synthesis hop; each analysis
    frame may shift +-tol to best continue the previous frame (max correlation).
    """
    if abs(alpha - 1.0) < 1e-3:
        return x.copy()
    n = int(SR * win_ms / 1000)
    n -= n % 2
    hs = n // 2                      # synthesis hop (50% hann OLA)
    ha = hs / alpha                  # analysis hop
    tol = int(SR * tol_ms / 1000)
    win = np.hanning(n).astype(np.float32)

    out_len = int(len(x) * alpha) + n
    y = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32) + 1e-8

    prev_start = 0
    k = 0
    while True:
        pos_out = k * hs
        ideal = int(round(k * ha))
        if pos_out + n > out_len or ideal + n + tol > len(x):
            break
        if k == 0:
            start = 0
        else:
            # natural continuation of the previously chosen frame
            nat_start = prev_start + hs
            if nat_start + n > len(x):
                break
            nat = x[nat_start:nat_start + n]
            lo = max(0, ideal - tol)
            hi = min(len(x) - n, ideal + tol)
            seg = x[lo:hi + n]
            corr = np.correlate(seg, nat, mode="valid")
            start = lo + int(np.argmax(corr))
        y[pos_out:pos_out + n] += x[start:start + n] * win
        norm[pos_out:pos_out + n] += win
        prev_start = start
        k += 1
    y = y / norm
    return y[: int(len(x) * alpha)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="src_native:tgt_l2, e.g. bdl:ASI (repeatable; gender-match them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmu-root", default=DEFAULT_CMU_ROOT)
    ap.add_argument("--l2-root", default=DEFAULT_L2_ROOT)
    ap.add_argument("--per-source", type=int, default=500,
                    help="prompts per source speaker (deterministic shuffle, seed 1234)")
    ap.add_argument("--val-prompts", "--val-per-pair", dest="val_prompts",
                    type=int, default=20,
                    help="number of ARCTIC prompt IDs held out globally; the "
                         "legacy --val-per-pair spelling is accepted")
    ap.add_argument("--alpha-range", default="0.6,1.7",
                    help="skip pairs whose stretch factor falls outside this range")
    args = ap.parse_args()

    out = Path(args.out)
    lo_a, hi_a = (float(v) for v in args.alpha_range.split(","))
    rng = random.Random(1234)

    # Build every pair first, then split *prompt groups* globally.  Splitting
    # inside each source/target pair leaks the same ARCTIC sentence across
    # train and validation (for example BDL/a0123 in val and RMS/a0123 in
    # train), which makes the validation loss optimistic.
    all_rows = []
    stats = {"pairs": {}, "skipped_alpha": 0, "alphas": []}

    for pair in args.pair:
        src_spk, tgt_spk = pair.split(":")
        src_dir = Path(args.cmu_root) / CMU_GLOB.format(spk=src_spk)
        tgt_dir = Path(args.l2_root) / tgt_spk / "wav"
        src_stems = {p.stem for p in src_dir.glob("*.wav")}
        tgt_stems = {p.stem for p in tgt_dir.glob("*.wav")}
        shared = sorted((src_stems & tgt_stems) - EVAL_PROMPTS)
        rng.shuffle(shared)
        shared = shared[: args.per_source]

        rows = []
        for stem in shared:
            src_wav = read_wav(src_dir / f"{stem}.wav")
            tgt_wav = read_wav(tgt_dir / f"{stem}.wav")
            if len(src_wav) < SR or len(tgt_wav) < SR:
                continue
            alpha = len(src_wav) / len(tgt_wav)
            if not (lo_a <= alpha <= hi_a):
                stats["skipped_alpha"] += 1
                continue
            stretched = wsola_stretch(tgt_wav, alpha)
            # exact length match for the frame-aligned losses
            if len(stretched) < len(src_wav):
                stretched = np.pad(stretched, (0, len(src_wav) - len(stretched)))
            stretched = stretched[: len(src_wav)]

            src_out = out / "wavs" / "src" / f"{src_spk}_{stem}.wav"
            tgt_out = out / "wavs" / "tgt" / f"{tgt_spk}__{src_spk}_{stem}.wav"
            if not src_out.exists():
                src_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_dir / f"{stem}.wav", src_out)
            write_wav(tgt_out, stretched)
            stats["alphas"].append(round(alpha, 3))

            rows.append({
                "source_utt": f"{src_spk}_{stem}",
                "source_wav_path": str(src_out).replace("\\", "/"),
                # loader indexes target_utt[:-3]; keep the _ft suffix convention
                "target_utt": f"{tgt_spk}__{src_spk}_{stem}_ft",
                "target_wav_path": str(tgt_out).replace("\\", "/"),
            })
        for row in rows:
            row["_prompt_id"] = Path(row["source_wav_path"]).stem.split("_", 1)[1]
            all_rows.append(row)
        stats["pairs"][pair] = len(rows)
        print(f"{pair}: {len(rows)} pairs "
              f"(alpha mean {np.mean([a for a in stats['alphas']][-len(rows):]):.3f})")

    prompt_ids = sorted({r["_prompt_id"] for r in all_rows})
    split_rng = random.Random(1234)
    split_rng.shuffle(prompt_ids)
    val_prompt_ids = set(prompt_ids[: args.val_prompts])
    val_rows = [r for r in all_rows if r["_prompt_id"] in val_prompt_ids]
    train_rows = [r for r in all_rows if r["_prompt_id"] not in val_prompt_ids]
    for row in all_rows:
        row.pop("_prompt_id", None)

    train_prompts = {Path(r["source_wav_path"]).stem.split("_", 1)[1] for r in train_rows}
    val_prompts = {Path(r["source_wav_path"]).stem.split("_", 1)[1] for r in val_rows}
    overlap = train_prompts & val_prompts
    if overlap:
        raise RuntimeError(f"prompt leakage after split: {sorted(overlap)[:5]}")

    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        with open(mdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    alphas = np.array(stats["alphas"])
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": stats["pairs"],
        "train": len(train_rows), "val": len(val_rows),
        "train_prompts": len(train_prompts), "val_prompts": len(val_prompts),
        "prompt_overlap": 0,
        "skipped_alpha": stats["skipped_alpha"],
        "alpha": {"mean": round(float(alphas.mean()), 3),
                  "p5": round(float(np.percentile(alphas, 5)), 3),
                  "p95": round(float(np.percentile(alphas, 95)), 3)},
        "eval_prompts_excluded": sorted(EVAL_PROMPTS),
        "manifest_sha256": {n: sha256_file(mdir / f"{n}.jsonl") for n in ("train", "val")},
    }
    with open(out / "build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\ntrain {len(train_rows)} | val {len(val_rows)} | "
          f"alpha mean {meta['alpha']['mean']} [{meta['alpha']['p5']}..{meta['alpha']['p95']}] "
          f"| skipped {stats['skipped_alpha']}")
    print(f"manifests + build_meta.json under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
