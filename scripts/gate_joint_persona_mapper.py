#!/usr/bin/env python
"""Gate a joint mapper on accent, target voice, quality, and intelligibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-mos-drop", type=float, default=0.25)
    parser.add_argument("--max-wer-increase", type=float, default=0.05)
    parser.add_argument("--max-sim-drop", type=float, default=0.03)
    parser.add_argument(
        "--min-indian-prob-gain",
        type=float,
        default=0.02,
        help="minimum paired mean gain in CommonAccent Indian posterior",
    )
    args = parser.parse_args(argv)

    with Path(args.summary).open(encoding="utf-8") as handle:
        rows = {row["set"]: row for row in csv.DictReader(handle)}
    required = {"stock_xvc", "joint_persona_mapper"}
    if not required <= rows.keys():
        raise SystemExit(f"[error] missing summary rows: {sorted(required - rows.keys())}")
    stock = rows["stock_xvc"]
    candidate = rows["joint_persona_mapper"]

    def number(row, key):
        value = row.get(key, "")
        if value in {"", "None", None}:
            raise ValueError(f"missing numeric metric {key!r} for {row.get('set')}")
        return float(value)

    deltas = {
        "mos": number(candidate, "mos_mean") - number(stock, "mos_mean"),
        "wer": number(candidate, "wer_mean") - number(stock, "wer_mean"),
        "sim": number(candidate, "sim_mean") - number(stock, "sim_mean"),
        "indian_frac": number(candidate, "indian_frac") - number(stock, "indian_frac"),
        "indian_prob": (
            number(candidate, "indian_prob_mean")
            - number(stock, "indian_prob_mean")
        ),
    }
    failures = []
    if deltas["mos"] < -args.max_mos_drop:
        failures.append(f"MOS delta {deltas['mos']:.4f} < {-args.max_mos_drop:.4f}")
    if deltas["wer"] > args.max_wer_increase:
        failures.append(f"WER delta {deltas['wer']:.4f} > {args.max_wer_increase:.4f}")
    if deltas["sim"] < -args.max_sim_drop:
        failures.append(
            f"target-speaker similarity delta {deltas['sim']:.4f} < {-args.max_sim_drop:.4f}"
        )
    if deltas["indian_prob"] < args.min_indian_prob_gain:
        failures.append(
            f"Indian posterior gain {deltas['indian_prob']:.4f} "
            f"< {args.min_indian_prob_gain:.4f}"
        )
    result = {
        "status": "pass" if not failures else "fail",
        "summary": args.summary,
        "stock": stock,
        "candidate": candidate,
        "deltas": {key: round(value, 6) for key, value in deltas.items()},
        "thresholds": {
            "max_mos_drop": args.max_mos_drop,
            "max_wer_increase": args.max_wer_increase,
            "max_target_speaker_similarity_drop": args.max_sim_drop,
            "min_indian_probability_gain": args.min_indian_prob_gain,
        },
        "failures": failures,
        "interpretation": (
            "Indian posterior is the primary classifier canary; hard indian_frac "
            "is reported but does not gate. Passing still requires blinded listening."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
