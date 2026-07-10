#!/usr/bin/env python3
"""Compare eval_compare results across sweep arms, side by side.

Reads ``<run_dir>/eval_compare/summary.csv`` (and ``metrics.csv`` for accent
labels, which summary.csv does not carry) for every run dir given, and prints
one table per metric with a column per run, rows ordered base-first then by
step. Runs whose eval has not finished are skipped with a warning, so this
works mid-sweep.

Usage:
    python scripts/compare_sweep_evals.py exp/run_a exp/run_b [exp/run_c ...]
    python scripts/compare_sweep_evals.py --metrics mos_pred_mean wer_mean exp/run_a exp/run_b
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

DEFAULT_METRICS = ["mos_pred_mean", "wer_mean", "sim_cosine_mean"]


def step_key(step):
    if step == "base":
        return (0, 0, "")
    if step.isdigit():
        return (1, int(step), "")
    return (2, 0, step)


def trim_labels(names):
    """Strip the shared prefix/suffix so columns read e.g. 'r1' not the full
    config name; fall back to the full name if trimming empties it."""
    if len(names) < 2:
        return list(names)
    prefix = os.path.commonprefix(names)
    suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    out = []
    for n in names:
        end = len(n) - len(suffix) if suffix else len(n)
        trimmed = n[len(prefix):end] if end > len(prefix) else ""
        out.append(trimmed or n)
    return out


def load_run(run_dir: Path):
    """(summary rows by step, accent Counter by step) or None if not evaluable."""
    summary_path = run_dir / "eval_compare" / "summary.csv"
    if not summary_path.is_file():
        return None
    with open(summary_path, newline="") as f:
        summary = {row["step"]: row for row in csv.DictReader(f)}
    accents = {}
    metrics_path = run_dir / "eval_compare" / "metrics.csv"
    if metrics_path.is_file():
        with open(metrics_path, newline="") as f:
            for row in csv.DictReader(f):
                label = (row.get("accent_label") or "").strip()
                if label:
                    accents.setdefault(row["step"], Counter())[label] += 1
    return summary, accents


def fmt(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="experiment dirs containing eval_compare/")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help=f"summary.csv columns to tabulate (default: {' '.join(DEFAULT_METRICS)})")
    args = parser.parse_args()

    runs = []
    for d in args.run_dirs:
        loaded = load_run(Path(d))
        if loaded is None:
            print(f"[compare] skipping {d}: no eval_compare/summary.csv (eval not finished?)",
                  file=sys.stderr)
            continue
        runs.append((Path(d).name.rstrip("/"), *loaded))
    if not runs:
        print("[compare] nothing to compare", file=sys.stderr)
        return 1

    labels = trim_labels([name for name, _, _ in runs])
    steps = sorted({s for _, summary, _ in runs for s in summary}, key=step_key)
    width = max(10, max(len(l) for l in labels) + 2)

    for metric in args.metrics:
        print(f"\n=== {metric} ===")
        print(f"{'step':<8}" + "".join(f"{l:>{width}}" for l in labels))
        for step in steps:
            cells = [fmt(summary.get(step, {}).get(metric)) for _, summary, _ in runs]
            print(f"{step:<8}" + "".join(f"{c:>{width}}" for c in cells))

    print("\n=== accent_label counts (from metrics.csv) ===")
    for (name, _, accents), label in zip(runs, labels):
        print(f"[{label}]" if label != name else f"[{name}]")
        if not accents:
            print("  (no accent labels found)")
            continue
        for step in sorted(accents, key=step_key):
            counts = ", ".join(f"{k}={v}" for k, v in accents[step].most_common())
            print(f"  {step:<8}{counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
