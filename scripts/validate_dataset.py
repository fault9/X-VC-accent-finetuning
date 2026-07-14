#!/usr/bin/env python3
"""Validate an X-VC cross-pair dataset (canonical entry point).

    python scripts/validate_dataset.py data/crosspair_hindi_latent_400_asi

Same engine and thresholds as ``scripts/validate_crosspairs.py`` (kept for
compatibility — the guarded runner calls it by name); this entry point takes
the dataset root as a positional argument and always saves the JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xvc.data.validation import Thresholds, validate_crosspair_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path,
                        help="dataset containing manifests/{train,val}.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-internal-zero-ms", type=float, default=100.0)
    parser.add_argument("--min-rms-dbfs", type=float, default=-45.0)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.0001)
    parser.add_argument("--max-abs-dc", type=float, default=0.01)
    parser.add_argument("--report-json", type=Path, default=None,
                        help="default: <data_root>/validation_report.json")
    args = parser.parse_args()

    result = validate_crosspair_dataset(
        args.data_root,
        Thresholds(
            sample_rate=args.sample_rate,
            max_internal_zero_ms=args.max_internal_zero_ms,
            min_rms_dbfs=args.min_rms_dbfs,
            min_duration=args.min_duration,
            max_clipped_fraction=args.max_clipped_fraction,
            max_abs_dc=args.max_abs_dc,
        ),
    )
    print(json.dumps(result.report, indent=2))
    result.write_report_json(args.report_json or args.data_root / "validation_report.json")
    if not result.passed:
        print("\nFirst failures:")
        for failure in result.failures[:30]:
            print(" -", failure)
        return 1
    print("\nPASS: cross-pair dataset is structurally safe for a training smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
