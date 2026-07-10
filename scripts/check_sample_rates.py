#!/usr/bin/env python3
"""Assert every wav in the given directories is mono at one expected sample rate.

Why this exists: every consumer in this repo RESAMPLES on load
(``utils/audio.py:load_audio`` -> soxr VHQ) -- the training dataloader loads
source/target/reference wavs through it and ``scripts/eval_checkpoints.py``
loads eval sources and pinned references the same way -- so a wav at the wrong
rate is silently tolerated rather than flagged. ``scripts/validate_crosspairs.py``
already hard-fails off-rate wavs INSIDE crosspair datasets, but the eval
source/target dirs are only counted by the guarded runner, never rate-checked.
This script closes that gap; ``scripts/run_lowrank_lora_sweep.sh`` runs it
before touching the GPU.

Usage:
    python scripts/check_sample_rates.py --dirs data/eval_sources data/eval_targets
    python scripts/check_sample_rates.py --dirs data/distill_sources_asi --expect 16000

Exit 0 with a one-line summary when every wav matches; exit 1 listing every
offending file (wrong rate, not mono, unreadable, or an empty/missing dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="directories scanned non-recursively for *.wav")
    parser.add_argument("--expect", type=int, default=16000,
                        help="expected sample rate in Hz (default 16000)")
    args = parser.parse_args()

    try:
        import soundfile
    except ImportError as exc:
        print(f"[sr-check] soundfile is required: {exc}", file=sys.stderr)
        return 1

    failures = []
    checked = 0
    for d in args.dirs:
        root = Path(d)
        if not root.is_dir():
            failures.append(f"{d}: not a directory")
            continue
        wavs = sorted(root.glob("*.wav"))
        if not wavs:
            failures.append(f"{d}: contains no .wav files")
            continue
        for wav in wavs:
            try:
                info = soundfile.info(str(wav))
            except Exception as exc:
                failures.append(f"{wav}: unreadable ({exc})")
                continue
            checked += 1
            if info.samplerate != args.expect:
                failures.append(f"{wav}: sample rate {info.samplerate} != {args.expect}")
            if info.channels != 1:
                failures.append(f"{wav}: {info.channels} channels (expected mono)")

    if failures:
        print(f"[sr-check] FAILED ({len(failures)} problem(s), {checked} wavs read):",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"[sr-check] ok: {checked} wavs mono @ {args.expect} Hz "
          f"across {len(args.dirs)} dir(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
