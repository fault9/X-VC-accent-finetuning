#!/usr/bin/env python3
"""Prepare exact target-speaker cross-pairs as corpora for phone-tier MFA.

This consumes the already filtered latent-crosspair manifests, but copies only
the pristine ``raw_source_wav_path`` and ``raw_target_wav_path`` recordings.
It never copies a warped waveform and never constructs an alignment map. The
result is two ordinary MFA corpora with matching ARCTIC transcripts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import wave
from pathlib import Path


PROMPT_RE = re.compile(r"(arctic_[ab]\d{4})")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    pattern = re.compile(r"(arctic_[ab]\d{4})\s+(.+)")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip().strip("() ")
            match = pattern.search(line)
            if not match:
                continue
            prompt, text = match.groups()
            text = text.strip().strip('"').strip()
            if text.endswith(")"):
                text = text[:-1].strip()
            prompts[prompt] = re.sub(r"\s+", " ", text)
    if not prompts:
        raise ValueError(f"no ARCTIC prompts parsed from {path}")
    return prompts


def prompt_id(row: dict) -> str:
    for key in ("source_utt", "target_utt", "raw_source_wav_path"):
        match = PROMPT_RE.search(str(row.get(key, "")))
        if match:
            return match.group(1)
    raise ValueError(f"cannot determine prompt ID for {row.get('source_utt')}")


def source_speaker(row: dict) -> str:
    return str(row["source_utt"]).split("_arctic_", 1)[0]


def target_speaker(row: dict) -> str:
    return str(row["target_utt"]).split("__", 1)[0].split("_", 1)[0]


def pristine_path(row: dict, side: str) -> Path:
    key = f"raw_{side}_wav_path"
    if not row.get(key):
        raise ValueError(f"manifest row lacks {key}: {row.get('source_utt')}")
    path = Path(row[key])
    if not path.is_file():
        raise FileNotFoundError(f"missing pristine {side} audio: {path}")
    return path


def wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"MFA input must be mono: {path}")
        sample_rate = handle.getframerate()
        duration = handle.getnframes() / sample_rate
    if sample_rate != 16000:
        raise ValueError(f"MFA input must be 16 kHz, got {sample_rate}: {path}")
    return sample_rate, duration


def copy_utterance(audio: Path, transcript: str, destination: Path) -> float:
    _, duration = wav_info(audio)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio, destination.with_suffix(".wav"))
    destination.with_suffix(".lab").write_text(
        transcript + "\n", encoding="utf-8"
    )
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-val", type=int)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"[error] refusing to overwrite non-empty output: {args.out}"
        )

    prompts = load_prompts(args.prompts_file)
    splits = {
        "train": [
            row for row in load_jsonl(args.train_manifest)
            if target_speaker(row) == args.target_speaker
        ],
        "val": [
            row for row in load_jsonl(args.val_manifest)
            if target_speaker(row) == args.target_speaker
        ],
    }
    expected = {"train": args.expected_train, "val": args.expected_val}
    for split, count in expected.items():
        if count is not None and len(splits[split]) != count:
            raise SystemExit(
                f"[error] selected {len(splits[split])} {split} rows; expected {count}"
            )

    train_prompts = {prompt_id(row) for row in splits["train"]}
    val_prompts = {prompt_id(row) for row in splits["val"]}
    overlap = train_prompts & val_prompts
    if overlap:
        raise SystemExit(
            "[error] prompt leakage between train and val: "
            + ", ".join(sorted(overlap)[:20])
        )

    missing_prompts = sorted(
        {prompt_id(row) for rows in splits.values() for row in rows}
        - prompts.keys()
    )
    if missing_prompts:
        raise ValueError(
            "PROMPTS is missing IDs: " + ", ".join(missing_prompts[:20])
        )

    source_root = args.out / "mfa_corpus" / "source"
    target_root = args.out / "mfa_corpus" / "target"
    selected_root = args.out / "selected_manifests"
    selected_root.mkdir(parents=True, exist_ok=True)
    durations = {"source": 0.0, "target": 0.0}

    for split, rows in splits.items():
        output_rows = []
        for row in rows:
            prompt = prompt_id(row)
            source_id = str(row["source_utt"])
            target_id = str(row["target_utt"])
            if target_id.endswith("_ft"):
                target_id = target_id[:-3]
            durations["source"] += copy_utterance(
                pristine_path(row, "source"),
                prompts[prompt],
                source_root / source_speaker(row) / source_id,
            )
            durations["target"] += copy_utterance(
                pristine_path(row, "target"),
                prompts[prompt],
                target_root / args.target_speaker / target_id,
            )
            enriched = dict(row)
            enriched.update({
                "mfa_source_id": source_id,
                "mfa_target_id": target_id,
                "prompt_text": prompts[prompt],
            })
            output_rows.append(enriched)
        with (selected_root / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "target_speaker": args.target_speaker,
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "train_prompts": len(train_prompts),
        "val_prompts": len(val_prompts),
        "prompt_overlap": 0,
        "source_minutes": round(durations["source"] / 60, 3),
        "target_minutes": round(durations["target"] / 60, 3),
        "audio_policy": "pristine raw source and target; no warp or resampling",
        "sample_rate": 16000,
    }
    (args.out / "mfa_prepare_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"\nMFA corpora written under {args.out / 'mfa_corpus'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
