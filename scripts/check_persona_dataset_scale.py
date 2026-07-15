#!/usr/bin/env python
"""Fail closed unless a target-persona dataset is a genuine scale-up."""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path


PROMPT_RE = re.compile(r"(arctic_[ab]\d{4})", re.IGNORECASE)
SPEAKER_RE = re.compile(r"([^_]+)_arctic_", re.IGNORECASE)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def prompt(value: str) -> str | None:
    match = PROMPT_RE.search(value)
    return match.group(1).casefold() if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--eval-source-dir", type=Path, required=True)
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--min-train-pairs", type=int, default=600)
    parser.add_argument("--min-source-speakers", type=int, default=4)
    parser.add_argument("--min-unique-target-minutes", type=float, default=45.0)
    parser.add_argument("--min-eval-speakers", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    train = rows(args.dataset_root / "manifests" / "train.jsonl")
    validation = rows(args.dataset_root / "manifests" / "val.jsonl")
    train_speakers = {str(row.get("source_speaker", "")).casefold() for row in train}
    target_speakers = {
        str(row.get("target_speaker", "")).casefold() for row in train + validation
    }
    unique_targets = sorted({Path(row["target_wav_path"]) for row in train})
    missing_targets = [str(path) for path in unique_targets if not path.is_file()]
    target_seconds = 0.0 if missing_targets else sum(wav_seconds(path) for path in unique_targets)
    train_prompts = {str(row.get("prompt_id", "")).casefold() for row in train}
    val_prompts = {str(row.get("prompt_id", "")).casefold() for row in validation}

    eval_paths = sorted(args.eval_source_dir.glob("*.wav"))
    eval_speakers = set()
    eval_prompts = set()
    for path in eval_paths:
        match = SPEAKER_RE.match(path.stem)
        eval_speakers.add((match.group(1) if match else path.stem.split("_", 1)[0]).casefold())
        value = prompt(path.stem)
        if value:
            eval_prompts.add(value)

    failures = []
    if len(train) < args.min_train_pairs:
        failures.append(f"train pairs {len(train)} < {args.min_train_pairs}")
    if len(train_speakers) < args.min_source_speakers:
        failures.append(
            f"native training speakers {sorted(train_speakers)} < {args.min_source_speakers}"
        )
    if target_seconds / 60 < args.min_unique_target_minutes:
        failures.append(
            f"unique target minutes {target_seconds/60:.2f} < {args.min_unique_target_minutes:.2f}"
        )
    if target_speakers != {args.target_speaker.casefold()}:
        failures.append(f"unexpected target speakers: {sorted(target_speakers)}")
    if train_prompts & val_prompts:
        failures.append("train/validation prompt overlap")
    if train_speakers & eval_speakers:
        failures.append(
            f"evaluation speakers seen in training: {sorted(train_speakers & eval_speakers)}"
        )
    if len(eval_speakers) < args.min_eval_speakers:
        failures.append(f"evaluation speakers {sorted(eval_speakers)} < {args.min_eval_speakers}")
    if train_prompts & eval_prompts:
        failures.append(f"evaluation prompt overlap: {sorted(train_prompts & eval_prompts)[:20]}")
    if missing_targets:
        failures.append(f"missing target wavs: {missing_targets[:5]}")

    report = {
        "status": "fail" if failures else "pass",
        "dataset_root": str(args.dataset_root),
        "train_pairs": len(train),
        "val_pairs": len(validation),
        "train_source_speakers": sorted(train_speakers),
        "unique_train_target_recordings": len(unique_targets),
        "unique_train_target_minutes": round(target_seconds / 60, 3),
        "eval_wavs": len(eval_paths),
        "eval_speakers": sorted(eval_speakers),
        "failures": failures,
        "note": "Target minutes count unique ASI recordings, never repeated pair exposure.",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
