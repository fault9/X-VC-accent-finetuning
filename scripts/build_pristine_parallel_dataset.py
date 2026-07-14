#!/usr/bin/env python3
"""Promote a validated phone-aware MFA corpus into a durable parallel dataset.

The output contains copied (never symlinked) pristine WAVs, transcripts, true
phone-tier TextGrids, prompt-disjoint manifests with relocatable paths, dataset
metadata, and SHA-256 checksums.  It does not construct a time map or claim the
recordings are frame-synchronous.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import wave
from pathlib import Path


PROMPT_RE = re.compile(r"(arctic_[ab]\d{4})", re.IGNORECASE)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt(row: dict) -> str:
    for value in (
        row.get("prompt_id"), row.get("source_utt"), row.get("target_utt"),
        row.get("mfa_source_id"), row.get("mfa_target_id"),
    ):
        match = PROMPT_RE.search(str(value or ""))
        if match:
            return match.group(1).lower()
    raise ValueError(f"cannot determine prompt for {row.get('source_utt')!r}")


def _source_speaker(row: dict) -> str:
    return str(row["mfa_source_id"]).split("_arctic_", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_info(path: Path, expected_rate: int, min_duration: float) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"expected mono WAV: {path}")
        if handle.getframerate() != expected_rate:
            raise ValueError(
                f"expected {expected_rate} Hz, got {handle.getframerate()}: {path}"
            )
        duration = handle.getnframes() / handle.getframerate()
    if duration < min_duration:
        raise ValueError(f"duration {duration:.3f}s < {min_duration:.3f}s: {path}")
    return duration


def _textgrids(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.TextGrid")):
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"duplicate TextGrid stem {path.stem!r} under {root}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r'name\s*=\s*"phones?"', text, re.IGNORECASE):
            raise ValueError(f"TextGrid lacks a genuine phone tier: {path}")
        index[key] = path
    if not index:
        raise ValueError(f"no TextGrids under {root}")
    return index


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"conflicting duplicate destination: {destination}")
        return
    shutil.copy2(source, destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--path-prefix",
        type=Path,
        required=True,
        help="relocatable manifest prefix, e.g. data/hindi_asi_pristine_parallel_221",
    )
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=2.6)
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"[error] refusing to overwrite non-empty output: {args.out}")

    prepared = args.prepared_root
    selected = prepared / "selected_manifests"
    source_corpus = prepared / "mfa_corpus" / "source"
    target_corpus = prepared / "mfa_corpus" / "target" / args.target_speaker
    source_grids = _textgrids(prepared / "mfa_align" / "source")
    target_grids = _textgrids(prepared / "mfa_align" / "target")
    splits = {
        split: _jsonl(selected / f"{split}.jsonl") for split in ("train", "val")
    }

    train_prompts = {_prompt(row) for row in splits["train"]}
    val_prompts = {_prompt(row) for row in splits["val"]}
    overlap = train_prompts & val_prompts
    if overlap:
        raise SystemExit(
            "[error] prompt overlap: " + ", ".join(sorted(overlap)[:20])
        )

    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_minutes = target_minutes = 0.0
    source_speakers = collections.Counter()
    manifest_rows: dict[str, list[dict]] = {"train": [], "val": []}

    def relative(destination: Path) -> str:
        return (args.path_prefix / destination.relative_to(args.out)).as_posix()

    for split, rows in splits.items():
        seen_pairs = set()
        for row in rows:
            source_id = str(row["mfa_source_id"])
            target_id = str(row["mfa_target_id"])
            prompt = _prompt(row)
            speaker = _source_speaker(row)
            pair_key = (source_id, target_id)
            if pair_key in seen_pairs:
                raise ValueError(f"duplicate pair in {split}: {pair_key}")
            seen_pairs.add(pair_key)

            source_wav = source_corpus / speaker / f"{source_id}.wav"
            target_wav = target_corpus / f"{target_id}.wav"
            source_lab = source_corpus / speaker / f"{source_id}.lab"
            target_lab = target_corpus / f"{target_id}.lab"
            for required in (source_wav, target_wav, source_lab, target_lab):
                if not required.is_file():
                    raise FileNotFoundError(required)

            source_minutes += _wav_info(
                source_wav, args.sample_rate, args.min_duration
            ) / 60.0
            target_minutes += _wav_info(
                target_wav, args.sample_rate, args.min_duration
            ) / 60.0
            source_speakers[speaker] += 1

            dst_source = args.out / "wavs" / "source" / speaker / source_wav.name
            dst_target = args.out / "wavs" / "target" / args.target_speaker / target_wav.name
            dst_source_text = args.out / "transcripts" / "source" / speaker / f"{source_id}.txt"
            dst_target_text = args.out / "transcripts" / "target" / args.target_speaker / f"{target_id}.txt"
            dst_source_grid = args.out / "alignments" / "source" / speaker / f"{source_id}.TextGrid"
            dst_target_grid = args.out / "alignments" / "target" / args.target_speaker / f"{target_id}.TextGrid"

            _copy_exact(source_wav, dst_source)
            _copy_exact(target_wav, dst_target)
            _copy_exact(source_lab, dst_source_text)
            _copy_exact(target_lab, dst_target_text)
            try:
                source_grid = source_grids[source_id.casefold()]
                target_grid = target_grids[target_id.casefold()]
            except KeyError as exc:
                raise FileNotFoundError(f"missing phone TextGrid for {exc.args[0]}") from exc
            _copy_exact(source_grid, dst_source_grid)
            _copy_exact(target_grid, dst_target_grid)

            manifest_rows[split].append(
                {
                    "schema_version": 1,
                    "pair_id": f"{source_id}__to__{target_id}",
                    "prompt_id": prompt,
                    "prompt_text": str(row.get("prompt_text", "")).strip(),
                    "source_utt": source_id,
                    "source_speaker": speaker,
                    "source_wav_path": relative(dst_source),
                    "raw_source_wav_path": relative(dst_source),
                    "source_text_path": relative(dst_source_text),
                    "source_textgrid_path": relative(dst_source_grid),
                    "target_utt": target_id + "_ft",
                    "target_speaker": args.target_speaker,
                    "target_wav_path": relative(dst_target),
                    "raw_target_wav_path": relative(dst_target),
                    "target_text_path": relative(dst_target_text),
                    "target_textgrid_path": relative(dst_target_grid),
                    "alignment_policy": "phone_intervals_for_supervision_only",
                }
            )

    for split, rows in manifest_rows.items():
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": 1,
        "name": args.out.name,
        "language": "English",
        "target_accent": "Hindi/Indian English",
        "target_speaker": args.target_speaker,
        "train_pairs": len(manifest_rows["train"]),
        "val_pairs": len(manifest_rows["val"]),
        "train_prompts": len(train_prompts),
        "val_prompts": len(val_prompts),
        "prompt_overlap": 0,
        "source_speakers": dict(sorted(source_speakers.items())),
        "source_minutes": round(source_minutes, 3),
        "target_minutes_with_pair_repetition": round(target_minutes, 3),
        "sample_rate": args.sample_rate,
        "audio_policy": "copied pristine recordings; mono 16 kHz; no warp, resampling, or symlinks",
        "alignment_policy": "MFA phones retained as interval annotations only",
        "intended_use": [
            "X-VC source-stream localization",
            "phone- or token-level pronunciation supervision",
            "clean reconstruction controls",
        ],
        "not_frame_synchronous": True,
        "warning": "Do not use as direct frame-wise waveform regression pairs without an alignment-aware objective.",
    }
    (args.out / "dataset_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    checksum_path = args.out / "checksums.sha256"
    checked_files = sorted(
        path for path in args.out.rglob("*") if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in checked_files:
            handle.write(f"{_sha256(path)}  {path.relative_to(args.out).as_posix()}\n")

    print(json.dumps(metadata, indent=2))
    print(f"\nCanonical pristine dataset written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
