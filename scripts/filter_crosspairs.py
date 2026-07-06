#!/usr/bin/env python3
"""Create quality-filtered cross-pair manifests without deleting any audio."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def target_key(row: dict) -> str:
    utt = row["target_utt"][:-3] if row["target_utt"].endswith("_ft") else row["target_utt"]
    speaker, rest = utt.split("__", 1)
    prompt = rest.split("_", 1)[1]
    return f"{speaker}:{prompt}"


def read_exclusions(path: str | None) -> tuple[set[str], set[str]]:
    pairs, targets = set(), set()
    if not path:
        return pairs, targets
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        (targets if ":" in value else pairs).add(value)
    return pairs, targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metrics", required=True,
                        help="JSONL from audit_warped_pairs.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-file")
    parser.add_argument("--manual-decisions",
                        help="JSONL: source_utt, decision keep/exclude, optional reason")
    parser.add_argument("--max-wer-delta", type=float, default=0.10)
    parser.add_argument("--max-added-errors", type=int, default=1)
    parser.add_argument("--review-warped-wer", type=float, default=0.35)
    parser.add_argument("--review-utmos-drop", type=float, default=0.30)
    parser.add_argument("--max-utmos-drop", type=float, default=0.50)
    parser.add_argument("--exclude-unreviewed", action="store_true",
                        help="exclude borderline/missing-metric rows lacking manual keep")
    args = parser.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    metrics = {row["source_utt"]: row for row in read_jsonl(Path(args.metrics))}
    qc_path = root / "alignment_qc.jsonl"
    qc = {row["source_utt"]: row for row in read_jsonl(qc_path)}
    excluded_pairs, excluded_targets = read_exclusions(args.exclude_file)
    manual = {}
    if args.manual_decisions:
        manual = {
            row["source_utt"]: row for row in read_jsonl(Path(args.manual_decisions))
        }

    kept_by_split = {"train": [], "val": []}
    kept_qc = []
    exclusions = []
    review_queue = []

    for split in ("train", "val"):
        rows = read_jsonl(root / "manifests" / f"{split}.jsonl")
        for row in rows:
            source = row["source_utt"]
            hard_reasons = []
            review_reasons = []
            if source in excluded_pairs:
                hard_reasons.append("manual pair exclusion")
            if target_key(row) in excluded_targets:
                hard_reasons.append("manual raw-target exclusion")

            metric = metrics.get(source)
            if metric is None:
                review_reasons.append("missing ASR/UTMOS metrics")
            else:
                if float(metric["wer_delta"]) > args.max_wer_delta:
                    hard_reasons.append(
                        f"WER delta {float(metric['wer_delta']):.3f}"
                    )
                if int(metric["added_errors"]) > args.max_added_errors:
                    hard_reasons.append(
                        f"added word errors {int(metric['added_errors'])}"
                    )
                if float(metric["warped_wer"]) > args.review_warped_wer:
                    review_reasons.append(
                        f"warped WER {float(metric['warped_wer']):.3f}"
                    )
                drop = metric.get("utmos_drop")
                if drop is not None:
                    if float(drop) > args.max_utmos_drop:
                        hard_reasons.append(f"UTMOS drop {float(drop):.3f}")
                    elif float(drop) > args.review_utmos_drop:
                        review_reasons.append(f"UTMOS drop {float(drop):.3f}")

            alignment = qc.get(source)
            if alignment is None:
                hard_reasons.append("missing alignment QC")
            elif float(alignment["diagnostic_mel_improvement"]) < 0:
                review_reasons.append("negative fitted mel-distance improvement")

            decision = manual.get(source, {}).get("decision", "").lower()
            manual_reason = manual.get(source, {}).get("reason")
            if decision not in ("", "keep", "exclude"):
                raise ValueError(f"{source}: invalid manual decision {decision!r}")
            if decision == "exclude":
                hard_reasons.append(manual_reason or "manual listening exclusion")
            elif decision == "keep":
                # Human comparison against the raw accented target can override
                # ASR/MOS false positives, but never an explicit exclusion or
                # missing structural QC.
                non_overridable = {
                    "manual pair exclusion",
                    "manual raw-target exclusion",
                    "missing alignment QC",
                }
                hard_reasons = [r for r in hard_reasons if r in non_overridable]
                review_reasons = []

            exclude = bool(hard_reasons) or (
                args.exclude_unreviewed and bool(review_reasons)
            )
            audit = {
                "split": split,
                "source_utt": source,
                "target_key": target_key(row),
                "hard_reasons": hard_reasons,
                "review_reasons": review_reasons,
                "manual_decision": decision or None,
            }
            if exclude:
                exclusions.append(audit)
            else:
                kept_by_split[split].append(row)
                if alignment:
                    kept_qc.append(alignment)
                if review_reasons:
                    review_queue.append(audit)

    train_prompts = {
        Path(row["source_wav_path"]).stem.split("_arctic_", 1)[1]
        for row in kept_by_split["train"]
    }
    val_prompts = {
        Path(row["source_wav_path"]).stem.split("_arctic_", 1)[1]
        for row in kept_by_split["val"]
    }
    if train_prompts & val_prompts:
        raise RuntimeError("quality filtering introduced prompt leakage")
    if not kept_by_split["train"] or not kept_by_split["val"]:
        raise RuntimeError("quality filtering emptied train or validation split")

    write_jsonl(out / "manifests" / "train.jsonl", kept_by_split["train"])
    write_jsonl(out / "manifests" / "val.jsonl", kept_by_split["val"])
    write_jsonl(out / "alignment_qc.jsonl", kept_qc)
    write_jsonl(out / "exclusions.jsonl", exclusions)
    write_jsonl(out / "review_queue.jsonl", review_queue)
    shutil.copyfile(root / "align_meta.json", out / "align_meta.json")
    report = {
        "source_data_root": str(root),
        "train_kept": len(kept_by_split["train"]),
        "val_kept": len(kept_by_split["val"]),
        "excluded": len(exclusions),
        "review_queue": len(review_queue),
        "thresholds": {
            "max_wer_delta": args.max_wer_delta,
            "max_added_errors": args.max_added_errors,
            "review_warped_wer": args.review_warped_wer,
            "review_utmos_drop": args.review_utmos_drop,
            "max_utmos_drop": args.max_utmos_drop,
            "exclude_unreviewed": args.exclude_unreviewed,
        },
    }
    (out / "filter_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
