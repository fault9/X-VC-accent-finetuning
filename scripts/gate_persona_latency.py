#!/usr/bin/env python
"""Fail a streaming persona arm that exceeds the existing HMO latency budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-p95-overhead-ms", type=float, default=10.0)
    parser.add_argument("--realtime-budget-ms", type=float, default=120.0)
    args = parser.parse_args(argv)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    if not meta.get("streaming"):
        raise SystemExit("[error] latency gate requires streaming evaluation")
    latency = meta.get("latency_ms", {})
    stock = latency.get("stock_xvc", {}).get("p95")
    candidate = latency.get("joint_persona_mapper", {}).get("p95")
    if stock is None or candidate is None:
        raise SystemExit("[error] missing p95 latency measurements")
    overhead = float(candidate) - float(stock)
    failures = []
    if overhead > args.max_p95_overhead_ms:
        failures.append(
            f"p95 mapper overhead {overhead:.2f}ms > {args.max_p95_overhead_ms:.2f}ms"
        )
    if float(candidate) > args.realtime_budget_ms:
        failures.append(
            f"candidate p95 {float(candidate):.2f}ms > {args.realtime_budget_ms:.2f}ms"
        )
    result = {
        "status": "fail" if failures else "pass",
        "stock_p95_ms": stock,
        "candidate_p95_ms": candidate,
        "p95_overhead_ms": round(overhead, 4),
        "max_p95_overhead_ms": args.max_p95_overhead_ms,
        "realtime_budget_ms": args.realtime_budget_ms,
        "failures": failures,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
