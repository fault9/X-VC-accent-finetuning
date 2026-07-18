#!/usr/bin/env python3
"""Summarize X-VC naturalness evaluations overall and per target voice."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def summarize(metrics_path: Path, output_path: Path) -> list[dict]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"empty metrics file: {metrics_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["step"], row["target"])].append(row)
        grouped[(row["step"], "__overall__")].append(row)

    baseline = {
        target: rows for (step, target), rows in grouped.items() if step == "base"
    }
    output_rows = []
    for (step, target), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0] != "base", int(item[0][0]) if item[0][0].isdigit() else -1, item[0][1])
    ):
        mos = _mean(rows, "mos_pred")
        wer = _mean(rows, "wer")
        sim = _mean(rows, "sim_cosine")
        base_rows = baseline.get(target, [])
        base_mos = _mean(base_rows, "mos_pred")
        base_wer = _mean(base_rows, "wer")
        base_sim = _mean(base_rows, "sim_cosine")
        output_rows.append(
            {
                "step": step,
                "target": target,
                "n": len(rows),
                "mos_mean": _fmt(mos),
                "mos_delta_vs_base": _fmt(None if mos is None or base_mos is None else mos - base_mos),
                "wer_mean": _fmt(wer),
                "wer_delta_vs_base": _fmt(None if wer is None or base_wer is None else wer - base_wer),
                "sim_mean": _fmt(sim),
                "sim_delta_vs_base": _fmt(None if sim is None or base_sim is None else sim - base_sim),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    # ASCII labels keep the report usable in Windows cp1252 terminals too.
    print("step target n MOS dMOS WER dWER SIM dSIM")
    for row in output_rows:
        print(
            f"{row['step']:>5} {row['target']:<34} {row['n']:>3} "
            f"{row['mos_mean']:>6} {row['mos_delta_vs_base']:>7} "
            f"{row['wer_mean']:>6} {row['wer_delta_vs_base']:>7} "
            f"{row['sim_mean']:>6} {row['sim_delta_vs_base']:>7}"
        )
    print(f"\n[summary] {output_path}")
    return output_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    metrics = Path(args.metrics)
    output = Path(args.out) if args.out else metrics.with_name("naturalness_by_target.csv")
    summarize(metrics, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
