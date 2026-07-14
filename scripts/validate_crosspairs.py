#!/usr/bin/env python3
"""Hard preflight checks for native-to-accent X-VC cross-pair datasets.

Compatibility wrapper: the checks live in ``xvc.data.validation`` (same
thresholds, same report fields, same exit codes). One behavioral fix over the
historical script: a legacy ``alignment_qc.jsonl`` row missing a gated field
(e.g. ``anchor_removal_fraction``) is now a readable validation failure
instead of a raw ``KeyError`` crash; gates apply only when ``align_meta.json``
configures them (previously they were evaluated with vacuous defaults).

New: ``--report-json`` saves the aggregate statistics (including the dataset
schema version) as JSON; default ``<data-root>/validation_report.json``.
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
    parser.add_argument("--data-root", required=True,
                        help="dataset containing manifests/{train,val}.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-internal-zero-ms", type=float, default=100.0)
    parser.add_argument("--min-rms-dbfs", type=float, default=-45.0)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.0001)
    parser.add_argument("--max-abs-dc", type=float, default=0.01,
                        help="maximum absolute waveform mean on the [-1,1] scale")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="where to save the aggregate report "
                             "(default: <data-root>/validation_report.json)")
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
    report_path = args.report_json or Path(args.data_root) / "validation_report.json"
    try:
        result.write_report_json(report_path)
    except OSError as exc:
        print(f"warning: could not save report to {report_path}: {exc}")

    if not result.passed:
        print("\nFirst failures:")
        for failure in result.failures[:30]:
            print(" -", failure)
        raise SystemExit(1)
    print("\nPASS: cross-pair dataset is structurally safe for a training smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
