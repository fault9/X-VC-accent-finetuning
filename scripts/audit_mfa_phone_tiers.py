#!/usr/bin/env python
"""Audit MFA output directories and fail unless every TextGrid has phones."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from phone_supervision import index_textgrids, normalize_phone, phone_tier, read_textgrid


def audit(root: Path):
    index = index_textgrids(root)
    phone_counts = Counter()
    tier_counts = Counter()
    failures = []
    n_intervals = 0
    for stem, path in sorted(index.items()):
        try:
            name, phones, _ = phone_tier(path)
            tier_counts[name] += 1
            n_intervals += len(phones)
            phone_counts.update(normalize_phone(x.label) for x in phones)
        except Exception as exc:
            tiers = []
            try:
                tiers = sorted(read_textgrid(path))
            except Exception:
                pass
            failures.append({"stem": stem, "path": str(path),
                             "tiers": tiers, "error": str(exc)})
    return {
        "root": str(root), "textgrids": len(index),
        "phone_intervals": n_intervals,
        "tier_counts": dict(tier_counts),
        "phone_counts": dict(sorted(phone_counts.items())),
        "failures": failures,
        "status": "pass" if index and not failures and n_intervals else "fail",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-align-dir", required=True)
    parser.add_argument("--target-align-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = {
        "source": audit(Path(args.source_align_dir)),
        "target": audit(Path(args.target_align_dir)),
    }
    result["status"] = "pass" if all(
        result[x]["status"] == "pass" for x in ("source", "target")) else "fail"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        print("[error] genuine phone-tier audit failed; do not train", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
