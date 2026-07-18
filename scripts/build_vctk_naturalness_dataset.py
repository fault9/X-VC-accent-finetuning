#!/usr/bin/env python3
"""Build a portable target-only naturalness dataset for four VCTK voices.

The short reference WAVs are *conditioning stimuli*, not the training corpus.
Training uses other pristine utterances from the same speakers as self-pairs:

    source == target == real target-speaker recording

This is deliberately different from the accent experiments.  There is no
cross-speaker regression, DTW/MFA alignment, waveform warping, generated
teacher audio, or pronunciation loss.  Prompt IDs are split globally so the
same sentence cannot cross train/validation/evaluation boundaries.

The builder also creates a small, balanced unseen-source evaluation set with
transcripts and a strict X-VC evaluation plan.  Source FLAC is decoded and all
output is written as mono 16 kHz PCM-16 WAV so the resulting directory can be
validated, archived, and moved to the GPU container without host-specific paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


_UTT_RE = re.compile(r"^(?P<speaker>[^_]+)_(?P<prompt>\d+)_mic1\.flac$")


def _stable_score(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    try:
        import soxr

        return soxr.resample(audio, source_sr, target_sr, quality="VHQ")
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(source_sr, target_sr)
        return resample_poly(audio, target_sr // divisor, source_sr // divisor)


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        audio = audio.mean(axis=1, keepdims=True)
    audio = _resample(audio[:, 0], int(sample_rate), 16_000)
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite samples in {path}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak >= 1.0:
        audio = audio * (0.999 / peak)
    return np.asarray(audio, dtype=np.float32)


def _write_audio(source: Path, destination: Path) -> float:
    audio = _load_mono_16k(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, 16_000, subtype="PCM_16")
    return len(audio) / 16_000.0


def _read_reference_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"file", "source_speaker", "utterance_filenames_used"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid reference manifest: {path}")
    speakers = [row["source_speaker"] for row in rows]
    if len(speakers) != len(set(speakers)):
        raise ValueError("reference manifest contains duplicate target speakers")
    return rows


def _speaker_files(vctk_root: Path, speaker: str) -> dict[str, Path]:
    directory = vctk_root / "wav48_silence_trimmed" / speaker
    found: dict[str, Path] = {}
    for path in sorted(directory.glob(f"{speaker}_*_mic1.flac")):
        match = _UTT_RE.match(path.name)
        if match:
            found[match.group("prompt")] = path
    if not found:
        raise FileNotFoundError(f"no mic1 utterances for {speaker}: {directory}")
    return found


def _duration_seconds(path: Path) -> float:
    return float(sf.info(path).duration)


def choose_training_prompts(
    files: dict[str, Path],
    forbidden: set[str],
    *,
    speaker: str,
    seed: int,
    minimum_prompts: int,
    minimum_minutes: float,
) -> list[str]:
    """Select a deterministic per-speaker corpus that clears both data floors."""
    available = sorted(
        set(files) - forbidden,
        key=lambda prompt: _stable_score(f"{speaker}:{prompt}", seed),
    )
    selected: list[str] = []
    seconds = 0.0
    target_seconds = minimum_minutes * 60.0
    for prompt in available:
        selected.append(prompt)
        seconds += _duration_seconds(files[prompt])
        if len(selected) >= minimum_prompts and seconds >= target_seconds:
            break
    if len(selected) < minimum_prompts or seconds < target_seconds:
        raise ValueError(
            f"insufficient eligible training audio for {speaker}: "
            f"{len(selected)} prompts / {seconds / 60.0:.3f} minutes; need at "
            f"least {minimum_prompts} prompts / {minimum_minutes:.3f} minutes"
        )
    return sorted(selected)


def choose_prompt_splits(
    target_files: dict[str, dict[str, Path]],
    reference_prompts: set[str],
    *,
    train_prompts: int,
    val_prompts: int,
    eval_prompts: int,
    seed: int,
) -> dict[str, list[str]]:
    """Choose balanced, prompt-disjoint splits shared by every target voice."""
    common = set.intersection(*(set(items) for items in target_files.values()))
    available = sorted(common - reference_prompts, key=lambda p: _stable_score(p, seed))
    needed = train_prompts + val_prompts + eval_prompts
    if len(available) < needed:
        raise ValueError(
            f"need {needed} common non-reference prompts, found {len(available)}"
        )
    eval_ids = available[:eval_prompts]
    val_ids = available[eval_prompts : eval_prompts + val_prompts]
    train_ids = available[eval_prompts + val_prompts : needed]
    return {"train": sorted(train_ids), "val": sorted(val_ids), "eval": sorted(eval_ids)}


def _manifest_path(prefix: str, relative: Path) -> str:
    return (Path(prefix) / relative).as_posix()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> dict:
    reference_dir = Path(args.references_dir).resolve()
    vctk_root = Path(args.vctk_root).resolve()
    output = Path(args.out)
    prefix = args.path_prefix or output.as_posix()
    reference_rows = _read_reference_manifest(reference_dir / "manifest.csv")
    target_speakers = [row["source_speaker"] for row in reference_rows]
    target_files_all = {
        speaker: _speaker_files(vctk_root, speaker) for speaker in target_speakers
    }
    target_files = {
        speaker: {
            prompt: path for prompt, path in files.items()
            if _duration_seconds(path) >= args.min_utterance_duration
        }
        for speaker, files in target_files_all.items()
    }
    eval_speakers = [item.strip() for item in args.eval_speakers.split(",") if item.strip()]
    eval_files = {speaker: _speaker_files(vctk_root, speaker) for speaker in eval_speakers}

    reference_prompts: set[str] = set()
    reference_name_by_speaker: dict[str, str] = {}
    for row in reference_rows:
        speaker = row["source_speaker"]
        reference_name_by_speaker[speaker] = row["file"]
        for name in row["utterance_filenames_used"].split(";"):
            match = _UTT_RE.match(name.strip())
            if match:
                reference_prompts.add(match.group("prompt"))

    # Validation is balanced across all four target voices. Training prompts
    # are then sampled independently per voice from the remaining prompt pool;
    # there is no benefit in forcing a self-reconstruction corpus to contain
    # the exact same sentence inventory for every speaker.
    target_common = set.intersection(*(set(items) for items in target_files.values()))
    val_available = sorted(
        target_common - reference_prompts,
        key=lambda prompt: _stable_score(prompt, args.seed),
    )
    if len(val_available) < args.val_prompts:
        raise ValueError(
            f"need {args.val_prompts} shared target validation prompts, "
            f"found {len(val_available)}"
        )
    val_ids = sorted(val_available[: args.val_prompts])

    # Evaluation prompts are chosen separately from unseen source speakers and
    # remain disjoint from validation, references, and all eventual training.
    eval_files = {
        speaker: {
            prompt: path for prompt, path in files.items()
            if _duration_seconds(path) >= args.min_eval_duration
        }
        for speaker, files in eval_files.items()
    }
    eval_common = set.intersection(*(set(items) for items in eval_files.values()))
    eval_forbidden = reference_prompts | set(val_ids)
    eval_available = sorted(
        eval_common - eval_forbidden,
        key=lambda prompt: _stable_score(prompt, args.seed + 1),
    )
    if len(eval_available) < args.eval_prompts:
        raise ValueError(
            f"need {args.eval_prompts} common unseen-source evaluation prompts, "
            f"found {len(eval_available)}"
        )
    eval_ids = sorted(eval_available[: args.eval_prompts])

    train_ids: dict[str, list[str]] = {}
    train_forbidden = reference_prompts | set(val_ids) | set(eval_ids)
    for speaker, files in target_files.items():
        train_ids[speaker] = choose_training_prompts(
            files,
            train_forbidden,
            speaker=speaker,
            seed=args.seed,
            minimum_prompts=args.train_prompts,
            minimum_minutes=args.train_minutes,
        )
    splits = {"train": train_ids, "val": val_ids, "eval": eval_ids}

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # The exact four 10-second stimuli are copied/resampled once and never used
    # as waveform targets.  They remain the target references in every row.
    target_meta: dict[str, dict] = {}
    for row in reference_rows:
        speaker = row["source_speaker"]
        source = reference_dir / row["file"]
        target_id = Path(row["file"]).stem
        relative = Path("references") / f"{target_id}.wav"
        duration = _write_audio(source, output / relative)
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        target_meta[target_id] = {
            "group": "vctk_naturalness",
            "speaker": speaker,
            "sex": row.get("sex"),
            "f0_condition": row.get("f0_condition"),
            "duration_seconds": round(duration, 6),
            "sha256": digest,
        }

    durations: dict[str, float] = defaultdict(float)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    persona_durations: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    persona_rows: dict[str, dict[str, list[dict]]] = {
        Path(reference_name_by_speaker[speaker]).stem: {"train": [], "val": []}
        for speaker in target_speakers
    }
    for split in ("train", "val"):
        manifest_rows = []
        for speaker in target_speakers:
            reference_id = Path(reference_name_by_speaker[speaker]).stem
            reference_rel = Path("references") / f"{reference_id}.wav"
            prompt_ids = splits["train"][speaker] if split == "train" else splits[split]
            for prompt in prompt_ids:
                source = target_files[speaker][prompt]
                relative = (
                    Path("wavs") / split / speaker /
                    f"{speaker}_arctic_vctk{prompt}.wav"
                )
                duration = _write_audio(source, output / relative)
                durations[split] += duration
                persona_durations[reference_id][split] += duration
                counts[split][speaker] += 1
                wav_path = _manifest_path(prefix, relative)
                row = {
                        "source_utt": f"{speaker}_vctk_{prompt}",
                        "target_utt": f"{speaker}_vctk_{prompt}_ft",
                        "prompt_id": f"vctk_{prompt}",
                        "source_speaker": speaker,
                        "target_speaker": speaker,
                        "source_wav_path": wav_path,
                        "target_wav_path": wav_path,
                        "target_reference_wav_path": _manifest_path(prefix, reference_rel),
                        "raw_source_duration": round(duration, 6),
                        "raw_target_duration": round(duration, 6),
                        "training_mode": "target_only_self_reconstruction",
                    }
                manifest_rows.append(row)
                persona_rows[reference_id][split].append(row)
        _write_jsonl(output / "manifests" / f"{split}.jsonl", manifest_rows)

    for target_id, split_rows in persona_rows.items():
        for split, rows in split_rows.items():
            _write_jsonl(
                output / "manifests" / "by_persona" / target_id / f"{split}.jsonl",
                rows,
            )

    # Unseen-source evaluation: each source speaker is tested against each target
    # in a Latin-square assignment, avoiding a 4x Cartesian explosion while
    # balancing source, prompt, and target.
    usable_eval_prompts = [
        prompt for prompt in splits["eval"]
        if all(prompt in paths for paths in eval_files.values())
    ]
    if len(usable_eval_prompts) < args.eval_prompts:
        raise ValueError(
            f"only {len(usable_eval_prompts)}/{args.eval_prompts} reserved prompts "
            "exist for every unseen evaluation speaker"
        )

    assignments: dict[str, str] = {}
    target_ids = [Path(reference_name_by_speaker[s]).stem for s in target_speakers]
    eval_source_count = 0
    eval_scout_source_count = 0
    for speaker_index, speaker in enumerate(eval_speakers):
        transcript_dir = vctk_root / "txt" / speaker
        for prompt_index, prompt in enumerate(usable_eval_prompts):
            source = eval_files[speaker][prompt]
            stem = f"{speaker}_arctic_vctk{prompt}"
            destination = output / "eval_sources" / f"{stem}.wav"
            _write_audio(source, destination)
            transcript = transcript_dir / f"{speaker}_{prompt}.txt"
            if not transcript.is_file():
                raise FileNotFoundError(f"missing transcript: {transcript}")
            destination.with_suffix(".txt").write_text(
                transcript.read_text(encoding="utf-8").strip() + "\n",
                encoding="utf-8",
            )
            if prompt_index < 2:
                scout = output / "eval_sources_scout" / destination.name
                scout.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, scout)
                shutil.copy2(destination.with_suffix(".txt"), scout.with_suffix(".txt"))
                eval_scout_source_count += 1
            assignments[stem] = target_ids[(speaker_index + prompt_index) % len(target_ids)]
            eval_source_count += 1

    (output / "eval_targets").mkdir(parents=True, exist_ok=True)
    for target_id in target_ids:
        shutil.copy2(
            output / "references" / f"{target_id}.wav",
            output / "eval_targets" / f"{target_id}.wav",
        )
    (output / "eval_targets" / "targets_meta.json").write_text(
        json.dumps(target_meta, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_plan = {
        "source_group": "vctk_unseen_speakers",
        "target_group": "vctk_naturalness",
        "allowed_source_speakers": eval_speakers,
        "assignments": assignments,
        "note": "Balanced Latin-square assignment; prompts are absent from target-voice train/val.",
    }
    (output / "evaluation_plan.json").write_text(
        json.dumps(evaluation_plan, indent=2) + "\n", encoding="utf-8"
    )
    for target_id in target_ids:
        persona_plan = {
            "source_group": "vctk_unseen_speakers",
            "target_group": "vctk_naturalness",
            "allowed_source_speakers": eval_speakers,
            "assignments": {stem: target_id for stem in assignments},
            "persona": target_id,
            "note": "All unseen sources convert to one fixed persona reference.",
        }
        plan_path = output / "evaluation_plans" / f"{target_id}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(persona_plan, indent=2) + "\n", encoding="utf-8"
        )

    meta = {
        "schema_version": 1,
        "name": output.name,
        "purpose": "target-voice naturalness adaptation; no accent conversion",
        "target_speakers": target_speakers,
        "reference_policy": "exact 10-second stimuli; conditioning only; excluded from train/val targets",
        "training_policy": "real target-speaker self-pairs; pristine; no warp/alignment/generated audio",
        "sample_rate": 16_000,
        "minimum_training_utterance_duration": args.min_utterance_duration,
        "minimum_evaluation_utterance_duration": args.min_eval_duration,
        "training_window_seconds": 2.4,
        "short_training_utterance_policy": "zero-pad to the 2.4-second X-VC window",
        "minimum_training_minutes_per_persona": args.train_minutes,
        "minimum_training_prompts_per_persona": args.train_prompts,
        "prompt_disjoint": True,
        "reference_prompts_excluded": sorted(reference_prompts),
        "prompt_splits": splits,
        "counts": {split: dict(values) for split, values in counts.items()},
        "minutes": {split: round(seconds / 60.0, 3) for split, seconds in durations.items()},
        "eval_speakers": eval_speakers,
        "eval_source_count": eval_source_count,
        "eval_scout_source_count": eval_scout_source_count,
        "persona_manifests": {
            target_id: {
                split: len(rows) for split, rows in split_rows.items()
            }
            for target_id, split_rows in persona_rows.items()
        },
        "persona_minutes": {
            target_id: {
                split: round(seconds / 60.0, 3)
                for split, seconds in split_seconds.items()
            }
            for target_id, split_seconds in persona_durations.items()
        },
        "path_prefix": prefix,
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"\nPASS: portable naturalness dataset written to {output}")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-dir", required=True)
    parser.add_argument("--vctk-root", required=True,
                        help="VCTK-Corpus-0.92 root containing wav48_silence_trimmed/ and txt/")
    parser.add_argument("--out", default="data/vctk_naturalness_4voice")
    parser.add_argument("--path-prefix", default=None,
                        help="manifest path prefix on the training machine; default: --out")
    parser.add_argument(
        "--train-prompts", type=int, default=150,
        help="minimum number of training recordings per persona (default: 150)",
    )
    parser.add_argument(
        "--train-minutes", type=float, default=12.0,
        help="minimum accumulated training duration per persona (default: 12)",
    )
    parser.add_argument("--val-prompts", type=int, default=15)
    parser.add_argument("--eval-prompts", type=int, default=6)
    parser.add_argument(
        "--min-utterance-duration", type=float, default=1.8,
        help="minimum clean training speech duration; clips below X-VC's "
             "2.4-second window are zero-padded by the upstream dataloader",
    )
    parser.add_argument(
        "--min-eval-duration", type=float, default=2.4,
        help="minimum unseen-source evaluation duration (default: 2.4)",
    )
    parser.add_argument("--eval-speakers", default="p226,p228,p232,p237,p245")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
