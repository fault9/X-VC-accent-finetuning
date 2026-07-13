#!/usr/bin/env python
"""Build a clean persona reference wav.

X-VC conditions on the FIRST `reference_duration` seconds of the reference
(bins/infer_utils.py truncates it), and the frame-level conditioning is a mel
of that clip -- so the recording's noise floor is part of "what the target
sounds like" and gets cloned into every conversion. Serving-side loudness
normalization (16k->24k + EBU R128 in the HMO path) then amplifies the hiss.

This scans candidate recordings of the persona speaker, ranks files by an SNR
proxy (90th-percentile frame RMS vs 10th-percentile = speech level over noise
floor), picks the densest-speech window of the requested duration inside the
best file, optionally spectral-gates it (noisereduce, if installed), and
writes a mono 16 kHz reference.

Usage (container, conda xvc, repo root):
    python scripts/make_clean_reference.py \
        --in-glob 'data/<asi target wavs>/*.wav' \
        --out data/eval_targets/ASI_clean.wav --duration 4.0 --denoise

Then A/B against the old reference (same source, both references) BEFORE
swapping it in as canonical -- a cleaned reference changes the base eval row,
so old summary tables stop being comparable and the base must be re-run.
"""

from __future__ import annotations

import argparse
import glob as globlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

FRAME = 0.025
HOP = 0.010


def frame_rms(wav, sr):
    import numpy as np
    frame, hop = int(FRAME * sr), int(HOP * sr)
    if len(wav) < frame:
        return np.zeros(0)
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((wav[idx] ** 2).mean(axis=1) + 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-glob", required=True,
                    help="candidate recordings of the persona speaker")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=4.0,
                    help="output length in seconds (>= configs' "
                         "reference_duration; 3.0 is what gets consumed)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--pick", default=None,
                    help="skip ranking and use this file")
    ap.add_argument("--denoise", action="store_true",
                    help="spectral-gate the chosen window (needs `pip install "
                         "noisereduce`; skipped with a warning if missing)")
    ap.add_argument("--top", type=int, default=8, help="files to report")
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf

    from utils.audio import load_audio

    paths = sorted(globlib.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"[error] no wavs match {args.in_glob}")

    sr = args.sample_rate
    need = int(args.duration * sr)

    ranked = []
    for p in paths:
        wav = load_audio(p, sr)
        if len(wav) < need:
            continue
        rms = frame_rms(wav, sr)
        noise = np.percentile(rms, 10)
        speech = np.percentile(rms, 90)
        snr_db = 20 * np.log10(speech / max(noise, 1e-9))
        ranked.append((snr_db, p, wav))
    if not ranked:
        raise SystemExit(f"[error] no candidate is >= {args.duration}s")
    ranked.sort(key=lambda t: -t[0])

    print(f"[rank] SNR proxy (speech level over noise floor), top {args.top}:")
    for snr_db, p, _ in ranked[: args.top]:
        print(f"  {snr_db:6.1f} dB  {p}")

    if args.pick:
        match = [t for t in ranked if t[1] == args.pick]
        if not match:
            raise SystemExit(f"[error] --pick {args.pick} not among usable candidates")
        snr_db, path, wav = match[0]
    else:
        snr_db, path, wav = ranked[0]

    # Densest-speech window of the requested duration (max mean frame RMS).
    rms = frame_rms(wav, sr)
    hop = int(HOP * sr)
    win_frames = max(1, (need - int(FRAME * sr)) // hop)
    kernel = np.ones(win_frames) / win_frames
    density = np.convolve(rms, kernel, mode="valid")
    start = int(np.argmax(density)) * hop
    clip = wav[start: start + need].astype("float32")
    print(f"[pick] {path} ({snr_db:.1f} dB) window {start / sr:.2f}s..{(start + need) / sr:.2f}s")

    if args.denoise:
        try:
            import noisereduce as nr
            clip = nr.reduce_noise(y=clip, sr=sr, stationary=True).astype("float32")
            print("[denoise] noisereduce stationary spectral gate applied")
        except ImportError:
            print("[denoise] SKIPPED -- `pip install noisereduce` to enable")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), clip, sr)
    print(f"[out] {out}  ({args.duration:.1f}s mono @ {sr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
