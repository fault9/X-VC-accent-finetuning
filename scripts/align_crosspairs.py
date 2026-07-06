#!/usr/bin/env python3
"""
DTW-align parallel cross pairs for accent conversion (Option B, phase B2).

The B1 pilot used UNIFORM time-stretch (global duration match only) and
collapsed into garbled output (WER > 1.0), whether or not semantic_adapter was
frozen.  That rules out the adapter choice as the sole cause; it does not by
itself prove that alignment is the only cause.

For each (native source, L2 target) pair reading the
SAME ARCTIC prompt, it DTW-aligns the two on speaker-normalised MFCCs (pairs are
gender-matched to reduce speaker mismatch), then warps the target audio onto
the source timeline with gap-free overlap-add driven by the DTW path.
Frame t of the warped target is then (approximately) the SAME phoneme as frame t
of the source, so the losses become coherent.

It reports a diagnostic mel-distance after DTW warp and uniform stretch. Since
DTW is fitted on related spectral features, this is not an independent proof
of alignment quality. Validate and listen to the warped targets before training.

    python scripts/align_crosspairs.py \
        --train-manifest data/crosspair_hindi/manifests/train.jsonl \
        --val-manifest data/crosspair_hindi/manifests/val.jsonl \
        --resplit-val-prompts 40 \
        --out data/crosspair_hindi_dtw

Copies source WAVs and writes warped targets, split-preserving manifests, and
align_meta.json. Part of the X-VC pipeline (upstream Jerrister/X-VC, MIT).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import wave
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 160          # 10 ms frames
N_MFCC = 13


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def mfcc_cmn(x: np.ndarray) -> np.ndarray:
    """MFCCs + delta, cepstral-mean-normalised (speaker-robust-ish). (frames, dim)."""
    m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    d = librosa.feature.delta(m)
    f = np.vstack([m, d]).T
    f = f - f.mean(axis=0, keepdims=True)
    f = f / (f.std(axis=0, keepdims=True) + 1e-8)
    return f


def dtw_path(A: np.ndarray, B: np.ndarray, band: float = 0.2):
    """Classic DTW with a Sakoe-Chiba band. Returns list of (i,j) source->target."""
    na, nb = len(A), len(B)
    w = max(int(band * max(na, nb)), abs(na - nb) + 1)
    INF = np.inf
    D = np.full((na + 1, nb + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    # cost = euclidean; compute per-cell within band
    for i in range(1, na + 1):
        jlo = max(1, int(i * nb / na) - w)
        jhi = min(nb, int(i * nb / na) + w)
        a = A[i - 1]
        for j in range(jlo, jhi + 1):
            cost = np.linalg.norm(a - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = na, nb
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def _dense_mapping(path, n_source_frames: int) -> np.ndarray:
    """Interpolate a monotonic target-frame position for every source frame."""
    from collections import defaultdict

    s2t = defaultdict(list)
    for i, j in path:
        s2t[i].append(j)
    known_i = np.asarray(sorted(s2t), dtype=np.float64)
    if len(known_i) == 0:
        raise ValueError("empty DTW path")
    known_j = np.asarray([np.median(s2t[int(i)]) for i in known_i], dtype=np.float64)
    return np.interp(np.arange(n_source_frames), known_i, known_j)


def warp_target_to_source(src: np.ndarray, tgt: np.ndarray, path, win_ms: int = 40):
    """DTW-guided, gap-free overlap-add; output length equals source length.

    This is OLA, not WSOLA: the DTW path supplies the target position and no
    secondary waveform-similarity search is performed. Reflect-padding and a
    dense mapping ensure boundary frames are covered.
    """
    n = int(SR * win_ms / 1000)
    n -= n % 2
    half = n // 2
    win = np.hanning(n).astype(np.float32)
    n_source_frames = max(1, int(np.ceil(len(src) / HOP)) + 1)
    mapping = _dense_mapping(path, n_source_frames)

    tgt_pad = np.pad(tgt, (half, half), mode="reflect")
    out = np.zeros(len(src) + 2 * half + HOP, dtype=np.float32)
    norm = np.zeros_like(out)
    for i, target_frame in enumerate(mapping):
        c_out = half + i * HOP
        c_in = half + int(round(target_frame * HOP))
        c_in = int(np.clip(c_in, half, len(tgt_pad) - half))
        segment = tgt_pad[c_in - half:c_in + half]
        if len(segment) != n:
            continue
        out[c_out - half:c_out + half] += segment * win
        norm[c_out - half:c_out + half] += win

    crop = slice(half, half + len(src))
    if np.any(norm[crop] <= 1e-6):
        raise RuntimeError("DTW OLA left uncovered output samples")
    return (out[crop] / norm[crop]).astype(np.float32)


def uniform_resample_len(tgt: np.ndarray, target_len: int) -> np.ndarray:
    if len(tgt) == target_len:
        return tgt
    idx = np.linspace(0, len(tgt) - 1, target_len)
    return np.interp(idx, np.arange(len(tgt)), tgt).astype(np.float32)


def mel_dist(a: np.ndarray, b: np.ndarray) -> float:
    L = min(len(a), len(b))
    ma = librosa.feature.melspectrogram(y=a[:L], sr=SR, hop_length=HOP, n_mels=40)
    mb = librosa.feature.melspectrogram(y=b[:L], sr=SR, hop_length=HOP, n_mels=40)
    ma = librosa.power_to_db(ma); mb = librosa.power_to_db(mb)
    k = min(ma.shape[1], mb.shape[1])
    return float(np.mean(np.abs(ma[:, :k] - mb[:, :k])))


def prompt_id(row: dict) -> str:
    stem = Path(row["source_wav_path"]).stem
    if "_arctic_" not in stem:
        raise ValueError(f"cannot derive ARCTIC prompt from {stem!r}")
    return "arctic_" + stem.split("_arctic_", 1)[1]


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-manifest", required=True,
                    help="training manifest from build_crosspairs.py")
    ap.add_argument("--val-manifest", required=True,
                    help="validation manifest from build_crosspairs.py")
    ap.add_argument("--resplit-val-prompts", type=int, default=None,
                    help="repair legacy leaky manifests by combining them and "
                         "holding out this many prompt IDs globally")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l2-root", default="D:/datasets/arctic-audio/l2arctic",
                    help="raw L2 root — align the RAW target (single warp), not the "
                         "build's pre-stretched one (avoids double time-warp artifacts)")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    splits = {
        "train": load_manifest(args.train_manifest),
        "val": load_manifest(args.val_manifest),
    }
    train_prompts = {prompt_id(r) for r in splits["train"]}
    val_prompts = {prompt_id(r) for r in splits["val"]}
    overlap = train_prompts & val_prompts
    if overlap:
        if args.resplit_val_prompts is None:
            raise SystemExit(
                "[error] input manifests leak prompt IDs across train/val: "
                + ", ".join(sorted(overlap)[:10])
                + "; rebuild with fixed build_crosspairs.py or pass "
                  "--resplit-val-prompts N to repair legacy manifests"
            )
        combined = splits["train"] + splits["val"]
        prompt_ids = sorted({prompt_id(row) for row in combined})
        random.Random(1234).shuffle(prompt_ids)
        held_out = set(prompt_ids[:args.resplit_val_prompts])
        splits = {
            "train": [row for row in combined if prompt_id(row) not in held_out],
            "val": [row for row in combined if prompt_id(row) in held_out],
        }
        train_prompts = {prompt_id(r) for r in splits["train"]}
        val_prompts = {prompt_id(r) for r in splits["val"]}
        if train_prompts & val_prompts:
            raise RuntimeError("global prompt re-split failed")
        print(f"[repair] re-split legacy manifests by prompt: "
              f"train={len(splits['train'])}, val={len(splits['val'])}")
    if args.limit:
        splits = {name: rows[:args.limit] for name, rows in splits.items()}

    rows_out = {"train": [], "val": []}
    d_dtw, d_uni = [], []
    total = sum(len(rows) for rows in splits.values())
    completed = 0
    for split, rows in splits.items():
        for r in rows:
            src_path = Path(r["source_wav_path"])
            src = read_wav(str(src_path))
            # Re-derive the raw L2 target; never warp the already stretched B1 file.
            utt = r["target_utt"][:-3] if r["target_utt"].endswith("_ft") else r["target_utt"]
            tgt_spk, rest = utt.split("__", 1)
            prompt = rest.split("_", 1)[1]
            raw_tgt_path = Path(args.l2_root) / tgt_spk / "wav" / f"{prompt}.wav"
            tgt_raw = read_wav(str(raw_tgt_path))
            path = dtw_path(mfcc_cmn(src), mfcc_cmn(tgt_raw))
            warped = warp_target_to_source(src, tgt_raw, path)

            # Fitted diagnostic only; independent QC happens in validate_crosspairs.py.
            d_dtw.append(mel_dist(src, warped))
            d_uni.append(mel_dist(src, uniform_resample_len(tgt_raw, len(src))))

            src_out = out / "wavs" / "src" / src_path.name
            src_out.parent.mkdir(parents=True, exist_ok=True)
            if not src_out.exists():
                shutil.copy2(src_path, src_out)
            tgt_out = out / "wavs" / "tgt" / Path(r["target_wav_path"]).name
            write_wav(tgt_out, warped)
            rr = dict(r)
            rr["source_wav_path"] = str(src_out).replace("\\", "/")
            rr["target_wav_path"] = str(tgt_out).replace("\\", "/")
            rows_out[split].append(rr)

            completed += 1
            if completed % 200 == 0:
                print(f"  {completed}/{total}  diagnostic mel-dist "
                      f"dtw {np.mean(d_dtw):.2f} vs resample {np.mean(d_uni):.2f}")

    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_out.items():
        with open(mdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    meta = {
        "pairs": sum(len(rows) for rows in rows_out.values()),
        "train_pairs": len(rows_out["train"]),
        "val_pairs": len(rows_out["val"]),
        "train_prompts": len({prompt_id(r) for r in rows_out["train"]}),
        "val_prompts": len({prompt_id(r) for r in rows_out["val"]}),
        "prompt_overlap": 0,
        "diagnostic_mel_dist_dtw_mean": round(float(np.mean(d_dtw)), 3),
        "diagnostic_mel_dist_resample_mean": round(float(np.mean(d_uni)), 3),
        "diagnostic_improvement": round(float(np.mean(d_uni) - np.mean(d_dtw)), 3),
    }
    with open(out / "align_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nFitted diagnostic mel-dist: DTW {meta['diagnostic_mel_dist_dtw_mean']} "
          f"vs resample {meta['diagnostic_mel_dist_resample_mean']} "
          f"(improvement {meta['diagnostic_improvement']}; not an independent QC metric)")
    print(f"manifests + align_meta.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
