#!/usr/bin/env python3
"""
DTW-align parallel cross pairs for accent conversion (Option B, phase B2).

The B1 pilot used UNIFORM time-stretch (global duration match only) and
collapsed: X-VC's frame-aligned losses got contradictory phoneme targets and
the model dissolved into garble (WER > 1.0), whether or not semantic_adapter
was frozen. Root cause = alignment, not modules.

This tool aligns properly: for each (native source, L2 target) pair reading the
SAME ARCTIC prompt, it DTW-aligns the two on speaker-normalised MFCCs (pairs are
gender-matched, so timbre doesn't fool the path), then warps the target audio
onto the source's timeline with a variable-rate WSOLA driven by the DTW path.
Frame t of the warped target is then (approximately) the SAME phoneme as frame t
of the source, so the losses become coherent.

It also reports an alignment-quality metric per pair: mean framewise mel-distance
after DTW-warp vs after plain uniform-stretch. Lower = better aligned; if DTW
isn't clearly beating uniform, don't bother training on it.

    python scripts/align_crosspairs.py \
        --in-manifest data/crosspair_hindi/manifests/train.jsonl \
        --out data/crosspair_hindi_dtw

Reuses the copied source wavs; only re-warps targets. Writes new manifests +
align_meta.json. Part of the X-VC pipeline (upstream Jerrister/X-VC, MIT).
"""
from __future__ import annotations

import argparse
import json
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


def warp_target_to_source(src: np.ndarray, tgt: np.ndarray, path, win_ms: int = 40):
    """Place windowed target frames at their DTW-mapped source positions (OLA).

    For each source frame i, gather the target frame(s) j mapped to it and OLA a
    target window centred at j into output position i. Output length == len(src).
    """
    out = np.zeros(len(src) + SR, dtype=np.float32)
    norm = np.zeros_like(out) + 1e-8
    n = int(SR * win_ms / 1000); n -= n % 2
    win = np.hanning(n).astype(np.float32)
    # source frame -> list of target frames
    from collections import defaultdict
    s2t = defaultdict(list)
    for i, j in path:
        s2t[i].append(j)
    for i, js in s2t.items():
        j = int(np.median(js))
        c_out = i * HOP
        c_in = j * HOP
        a0, a1 = c_in - n // 2, c_in + n // 2
        o0, o1 = c_out - n // 2, c_out + n // 2
        if a0 < 0 or a1 > len(tgt) or o0 < 0:
            continue
        out[o0:o1] += tgt[a0:a1] * win
        norm[o0:o1] += win
    out = out / norm
    return out[: len(src)]


def uniform_stretch_len(tgt: np.ndarray, target_len: int) -> np.ndarray:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-manifest", action="append", required=True,
                    help="cross-pair manifest(s) from build_crosspairs.py (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l2-root", default="D:/datasets/arctic-audio/l2arctic",
                    help="raw L2 root — align the RAW target (single warp), not the "
                         "build's pre-stretched one (avoids double time-warp artifacts)")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    rows_out = []
    d_dtw, d_uni = [], []

    rows = []
    for mf in args.in_manifest:
        rows += [json.loads(l) for l in open(mf, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    for n, r in enumerate(rows):
        src = read_wav(r["source_wav_path"])
        # Re-derive the RAW L2 target from target_utt = "TGT__SRC_arctic_XXXX_ft"
        # and align it in a single warp (no build-time uniform stretch underneath).
        utt = r["target_utt"][:-3] if r["target_utt"].endswith("_ft") else r["target_utt"]
        tgt_spk, rest = utt.split("__", 1)
        prompt = rest.split("_", 1)[1]          # drop the src-speaker prefix
        raw_tgt_path = Path(args.l2_root) / tgt_spk / "wav" / f"{prompt}.wav"
        tgt_raw = read_wav(str(raw_tgt_path))
        A, B = mfcc_cmn(src), mfcc_cmn(tgt_raw)
        path = dtw_path(A, B)
        warped = warp_target_to_source(src, tgt_raw, path)

        # quality: DTW-warp vs plain uniform-stretch of the RAW target
        d_dtw.append(mel_dist(src, warped))
        d_uni.append(mel_dist(src, uniform_stretch_len(tgt_raw, len(src))))

        stem = Path(r["target_wav_path"]).stem
        tgt_out = out / "wavs" / "tgt" / f"{stem}.wav"
        write_wav(tgt_out, warped)
        rr = dict(r); rr["target_wav_path"] = str(tgt_out).replace("\\", "/")
        rows_out.append(rr)
        if (n + 1) % 200 == 0:
            print(f"  {n+1}/{len(rows)}  mel-dist dtw {np.mean(d_dtw):.2f} vs uni {np.mean(d_uni):.2f}")

    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    # keep the same train/val split proportion as input order
    val = rows_out[:80]; train = rows_out[80:] if len(rows_out) > 80 else rows_out
    for name, rs in (("train", train), ("val", val)):
        with open(mdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    meta = {
        "pairs": len(rows_out),
        "mel_dist_dtw_mean": round(float(np.mean(d_dtw)), 3),
        "mel_dist_uniform_mean": round(float(np.mean(d_uni)), 3),
        "improvement": round(float(np.mean(d_uni) - np.mean(d_dtw)), 3),
    }
    with open(out / "align_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDTW mel-dist {meta['mel_dist_dtw_mean']} vs uniform {meta['mel_dist_uniform_mean']} "
          f"(improvement {meta['improvement']}; positive = DTW better)")
    print(f"manifests + align_meta.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
