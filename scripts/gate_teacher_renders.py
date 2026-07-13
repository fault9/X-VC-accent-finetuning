#!/usr/bin/env python
"""Gate candidate self-distill teacher renders: accent-label counts + predicted
MOS per directory, side by side. Run from the repo root (the accent classifier
uses a relative savedir).

    python scripts/gate_teacher_renders.py data/sd_probe_*/wavs

The classifier is a noisy proxy: 'england' has tracked mild Hindi-English,
'indian' the stronger shift. The MOS floor tells you how much texture damage
a student would have to absorb. Ears decide -- this just ranks what to listen to.
"""

from __future__ import annotations

import argparse
import collections
import glob as globlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="directories of rendered wavs")
    ap.add_argument("--limit", type=int, default=40,
                    help="max wavs per dir (sorted order, so the same clips "
                         "are compared across dirs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worst", type=int, default=3,
                    help="how many lowest-MOS clips to list per dir")
    args = ap.parse_args(argv)

    import soundfile as sf

    from eval_checkpoints import AccentClassifier, MOSPredictor

    clf = AccentClassifier(args.device)
    mos = MOSPredictor(args.device)

    for d in args.dirs:
        wavs = sorted(globlib.glob(str(Path(d) / "*.wav")))[: args.limit]
        if not wavs:
            print(f"\n== {d}: NO WAVS ==")
            continue
        labels = collections.Counter()
        scored = []
        for p in wavs:
            lab, _ = clf.classify(p)
            labels[lab] += 1
            wav, sr = sf.read(p)
            scored.append((mos.score(wav, sr), lab, p))
        scores = [s for s, _, _ in scored]
        print(f"\n== {d} ({len(wavs)} wavs) ==")
        print("  labels:", dict(labels.most_common()))
        print(f"  MOS mean {sum(scores) / len(scores):.3f}  min {min(scores):.3f}")
        scored.sort()
        for s, lab, p in scored[: args.worst]:
            print(f"    worst {s:.2f}  {lab:10s}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
