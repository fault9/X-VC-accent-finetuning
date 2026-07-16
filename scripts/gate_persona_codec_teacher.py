#!/usr/bin/env python
"""Gate the offline persona codec teacher (Phase 1 falsification gate).

Same thresholds and calibrated Indian-posterior semantics as
`scripts/gate_joint_persona_mapper.py` (both CLIs call the shared
`xvc.evaluation.run_accent_gate`), applied to the render conditions written
by `scripts/render_persona_codec_teacher.py`.  This is the hard stop of the
StreamVoice-style plan: if the *offline, full-context* teacher cannot close
>= 25% of the calibrated genuine-ASI/native posterior gap without breaking
MOS/WER/similarity, the streaming student must not be built.  Passing still
requires matched blinded listening.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from xvc.evaluation import run_accent_gate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stock-set", default="stock_xvc")
    parser.add_argument("--candidate-set", default="persona_codec_teacher")
    parser.add_argument("--max-mos-drop", type=float, default=0.25)
    parser.add_argument("--max-wer-increase", type=float, default=0.05)
    parser.add_argument("--max-sim-drop", type=float, default=0.03)
    parser.add_argument(
        "--min-indian-prob-gain",
        type=float,
        default=0.02,
        help="fallback absolute gain when --calibration is not supplied",
    )
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--calibration-target-set", default="genuine_asi")
    parser.add_argument("--calibration-native-set", default="native_unseen")
    parser.add_argument("--min-accent-gap-closed", type=float, default=0.25)
    args = parser.parse_args(argv)

    return run_accent_gate(
        summary=args.summary,
        out=args.out,
        stock_set=args.stock_set,
        candidate_set=args.candidate_set,
        max_mos_drop=args.max_mos_drop,
        max_wer_increase=args.max_wer_increase,
        max_sim_drop=args.max_sim_drop,
        min_indian_prob_gain=args.min_indian_prob_gain,
        calibration=args.calibration,
        calibration_target_set=args.calibration_target_set,
        calibration_native_set=args.calibration_native_set,
        min_accent_gap_closed=args.min_accent_gap_closed,
        interpretation=(
            "Indian posterior is the primary classifier canary; hard indian_frac "
            "is reported but does not gate. Passing still requires blinded "
            "listening. A failing offline teacher is a stop condition: do not "
            "build the streaming student and do not assume more data fixes it."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
