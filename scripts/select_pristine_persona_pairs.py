#!/usr/bin/env python
"""Select prompt-matched pristine native-to-persona recordings for MFA.

Each target utterance is used once by default. Native speakers are assigned in
a deterministic balanced rotation, so source diversity does not inflate the
amount of unique target-persona speech. The output is a pair-index manifest;
audio remains untouched at its original path.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import wave
from pathlib import Path


PROMPT_RE = re.compile(r"(arctic_[ab]\d{4})", re.IGNORECASE)


def prompt_id(path: Path) -> str:
    match = PROMPT_RE.search(path.stem)
    if not match:
        raise ValueError(f"cannot parse ARCTIC prompt id: {path}")
    return match.group(1).casefold()


def audio_seconds(path: Path, min_duration: float) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000:
            raise ValueError(f"expected mono 16 kHz WAV: {path}")
        duration = handle.getnframes() / handle.getframerate()
    if duration < min_duration:
        raise ValueError(f"duration {duration:.3f}s < {min_duration:.3f}s: {path}")
    return duration


def indexed(pattern: str, min_duration: float) -> dict[str, tuple[Path, float]]:
    result: dict[str, tuple[Path, float]] = {}
    for value in sorted(glob.glob(pattern, recursive=True)):
        path = Path(value)
        if not path.is_file() or path.suffix.casefold() != ".wav":
            continue
        prompt = prompt_id(path)
        if prompt in result:
            raise ValueError(f"duplicate {prompt} under pattern {pattern!r}")
        try:
            result[prompt] = (path.resolve(), audio_seconds(path, min_duration))
        except ValueError as error:
            if "duration" not in str(error):
                raise
    if not result:
        raise ValueError(f"no usable WAVs matched {pattern!r}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, metavar="SPEAKER=GLOB",
        help="repeat for each native training speaker; quote recursive globs",
    )
    parser.add_argument("--target-glob", required=True)
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-prompts", type=int, default=50)
    parser.add_argument("--sources-per-target", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=2.6)
    parser.add_argument("--min-train-pairs", type=int, default=600)
    parser.add_argument("--min-unique-target-minutes", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"[error] refusing to overwrite non-empty output: {args.out}")
    if args.sources_per_target < 1:
        raise SystemExit("[error] sources-per-target must be positive")

    source_indices = {}
    for value in args.source:
        if "=" not in value:
            raise SystemExit(f"[error] expected SPEAKER=GLOB, got {value!r}")
        speaker, pattern = value.split("=", 1)
        speaker = speaker.strip().casefold()
        if not speaker or speaker in source_indices:
            raise SystemExit(f"[error] invalid/duplicate source speaker: {speaker!r}")
        source_indices[speaker] = indexed(pattern, args.min_duration)
    if len(source_indices) < 2:
        raise SystemExit("[error] require at least two source speakers")
    if args.sources_per_target > len(source_indices):
        raise SystemExit("[error] sources-per-target exceeds available speakers")

    targets = indexed(args.target_glob, args.min_duration)
    available = sorted(
        prompt for prompt in targets
        if sum(prompt in index for index in source_indices.values())
        >= args.sources_per_target
    )
    if len(available) <= args.val_prompts:
        raise SystemExit("[error] validation split leaves no training prompts")
    rng = random.Random(args.seed)
    rng.shuffle(available)
    val_set = set(available[: args.val_prompts])
    split_prompts = {
        "train": sorted(set(available) - val_set),
        "val": sorted(val_set),
    }

    speaker_counts = collections.Counter()
    manifests = {"train": [], "val": []}
    for split, prompts in split_prompts.items():
        for prompt in prompts:
            candidates = [speaker for speaker, index in source_indices.items() if prompt in index]
            # Prefer the least-used speakers. Seeded noise only breaks exact ties.
            tie_break = {speaker: rng.random() for speaker in candidates}
            chosen = sorted(candidates, key=lambda speaker: (speaker_counts[speaker], tie_break[speaker]))[
                : args.sources_per_target
            ]
            target_path, target_duration = targets[prompt]
            for speaker in chosen:
                source_path, source_duration = source_indices[speaker][prompt]
                speaker_counts[speaker] += 1
                manifests[split].append({
                    "schema_version": 1,
                    "pair_id": f"{speaker}_{prompt}__to__{args.target_speaker}_{prompt}",
                    "prompt_id": prompt,
                    "source_utt": f"{speaker}_{prompt}",
                    "source_speaker": speaker,
                    "raw_source_wav_path": str(source_path),
                    "source_duration_seconds": round(source_duration, 6),
                    "target_utt": f"{args.target_speaker}_{prompt}_ft",
                    "target_speaker": args.target_speaker,
                    "raw_target_wav_path": str(target_path),
                    "target_duration_seconds": round(target_duration, 6),
                    "selection_policy": "pristine_prompt_match_balanced_source",
                })

    unique_train_prompts = {row["prompt_id"] for row in manifests["train"]}
    unique_target_minutes = sum(targets[prompt][1] for prompt in unique_train_prompts) / 60
    failures = []
    if len(manifests["train"]) < args.min_train_pairs:
        failures.append(f"train pairs {len(manifests['train'])} < {args.min_train_pairs}")
    if unique_target_minutes < args.min_unique_target_minutes:
        failures.append(
            f"unique target minutes {unique_target_minutes:.3f} < {args.min_unique_target_minutes:.3f}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "status": "fail" if failures else "pass",
        "target_speaker": args.target_speaker,
        "train_pairs": len(manifests["train"]),
        "val_pairs": len(manifests["val"]),
        "train_unique_target_recordings": len(unique_train_prompts),
        "train_unique_target_minutes": round(unique_target_minutes, 3),
        "source_speakers": dict(sorted(speaker_counts.items())),
        "sources_per_target": args.sources_per_target,
        "audio_policy": "pristine paths only; no copy, warp, resample, or synthesis",
        "failures": failures,
    }
    (args.out / "selection_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
