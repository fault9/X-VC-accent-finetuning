#!/usr/bin/env python3
"""Hard preflight checks for native-to-accent X-VC cross-pair datasets."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


REQUIRED = ("source_utt", "source_wav_path", "target_utt", "target_wav_path")
EVAL_PROMPTS = {f"arctic_b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in REQUIRED if not row.get(key)]
            if missing:
                raise ValueError(f"{path}:{number}: missing {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def prompt_id(row: dict) -> str:
    stem = Path(row["source_wav_path"]).stem
    if "_arctic_" not in stem:
        raise ValueError(f"cannot derive ARCTIC prompt from {stem!r}")
    return "arctic_" + stem.split("_arctic_", 1)[1]


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected mono PCM16, got channels={wav.getnchannels()} "
                f"sample_width={wav.getsampwidth()}"
            )
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return audio, rate


def max_internal_zero_run(samples: np.ndarray) -> int:
    zero = samples == 0
    changes = np.diff(np.r_[False, zero, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    internal = [end - start for start, end in zip(starts, ends)
                if start > 0 and end < len(samples)]
    return max(internal, default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="dataset containing manifests/{train,val}.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-internal-zero-ms", type=float, default=100.0)
    parser.add_argument("--min-rms-dbfs", type=float, default=-45.0)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.0001)
    parser.add_argument("--max-abs-dc", type=float, default=0.01,
                        help="maximum absolute waveform mean on the [-1,1] scale")
    args = parser.parse_args()

    root = Path(args.data_root)
    manifests = root / "manifests"
    align_meta_path = root / "align_meta.json"
    align_meta = (
        json.loads(align_meta_path.read_text(encoding="utf-8"))
        if align_meta_path.is_file() else {}
    )
    qc_path = root / "alignment_qc.jsonl"
    qc_rows = (
        [json.loads(line) for line in qc_path.read_text(encoding="utf-8").splitlines()
         if line.strip()]
        if qc_path.is_file() else []
    )
    qc_by_source = {row["source_utt"]: row for row in qc_rows}
    splits = {
        name: load_manifest(manifests / f"{name}.jsonl")
        for name in ("train", "val")
    }

    prompts = {name: {prompt_id(row) for row in rows}
               for name, rows in splits.items()}
    overlap = prompts["train"] & prompts["val"]
    if overlap:
        raise SystemExit(
            "FAIL: train/val prompt leakage: " + ", ".join(sorted(overlap)[:20])
        )
    paired_prompts = prompts["train"] | prompts["val"] | EVAL_PROMPTS

    failures = []
    seen_targets = set()
    rms_values = []
    zero_runs_ms = []
    clipped_fractions = []
    dc_offsets = []
    durations = []
    reference_paths = set()
    for split, rows in splits.items():
        for index, row in enumerate(rows, 1):
            source = Path(row["source_wav_path"])
            target = Path(row["target_wav_path"])
            label = f"{split}:{index}"
            if not source.is_file() or not target.is_file():
                failures.append(f"{label}: missing path: {source} or {target}")
                continue
            if target in seen_targets:
                failures.append(f"{label}: duplicate target path: {target}")
            seen_targets.add(target)
            try:
                source_audio, source_rate = read_pcm16(source)
                target_audio, target_rate = read_pcm16(target)
            except Exception as exc:
                failures.append(f"{label}: {exc}")
                continue
            if source_rate != args.sample_rate or target_rate != args.sample_rate:
                failures.append(
                    f"{label}: sample rates source={source_rate}, target={target_rate}"
                )
            latent_mode = align_meta.get("warp_method") == "latent"
            if len(source_audio) != len(target_audio) and not latent_mode:
                failures.append(
                    f"{label}: duration mismatch source={len(source_audio)}, "
                    f"target={len(target_audio)} samples"
                )
            duration = len(target_audio if latent_mode else source_audio) / source_rate
            durations.append(duration)
            if duration < args.min_duration:
                failures.append(
                    f"{label}: duration {duration:.3f}s < {args.min_duration:.3f}s"
                )
            for field in ("raw_source_duration", "raw_target_duration"):
                if field in row and float(row[field]) < args.min_duration:
                    failures.append(
                        f"{label}: {field}={float(row[field]):.3f}s "
                        f"< {args.min_duration:.3f}s"
                    )
            rms = float(np.sqrt(np.mean(target_audio.astype(np.float64) ** 2)))
            dbfs = 20.0 * math.log10(max(rms, 1e-12) / 32768.0)
            rms_values.append(dbfs)
            zero_ms = 1000.0 * max_internal_zero_run(target_audio) / target_rate
            zero_runs_ms.append(zero_ms)
            clipped_fraction = float(np.mean(np.abs(target_audio.astype(np.int32)) >= 32760))
            clipped_fractions.append(clipped_fraction)
            dc_offset = abs(float(np.mean(target_audio.astype(np.float64))) / 32768.0)
            dc_offsets.append(dc_offset)
            if dbfs < args.min_rms_dbfs:
                failures.append(f"{label}: target RMS too low ({dbfs:.1f} dBFS): {target}")
            if zero_ms > args.max_internal_zero_ms:
                failures.append(
                    f"{label}: internal exact-zero gap {zero_ms:.1f} ms: {target}"
                )
            if clipped_fraction > args.max_clipped_fraction:
                failures.append(
                    f"{label}: clipped fraction {clipped_fraction:.6f}: {target}"
                )
            if dc_offset > args.max_abs_dc:
                failures.append(
                    f"{label}: absolute DC offset {dc_offset:.5f}: {target}"
                )

            reference_value = row.get("target_reference_wav_path")
            raw_source_value = row.get("raw_source_wav_path")
            raw_target_value = row.get("raw_target_wav_path")
            alignment_value = row.get("latent_alignment_path")
            if align_meta.get("warp_method") == "rubberband" and not raw_target_value:
                failures.append(f"{label}: missing raw_target_wav_path")
            if align_meta.get("warp_side") == "source" and not raw_source_value:
                failures.append(f"{label}: source-side warp missing raw_source_wav_path")
            if latent_mode and not raw_source_value:
                failures.append(f"{label}: latent alignment missing raw_source_wav_path")
            if latent_mode and not raw_target_value:
                failures.append(f"{label}: latent alignment missing raw_target_wav_path")
            if latent_mode and not alignment_value:
                failures.append(f"{label}: missing latent_alignment_path")
            if raw_source_value:
                raw_source = Path(raw_source_value)
                if not raw_source.is_file():
                    failures.append(f"{label}: missing raw source: {raw_source}")
                else:
                    try:
                        raw_source_audio, raw_source_rate = read_pcm16(raw_source)
                        if raw_source_rate != args.sample_rate:
                            failures.append(
                                f"{label}: raw source sample rate={raw_source_rate}"
                            )
                        if len(raw_source_audio) / raw_source_rate < args.min_duration:
                            failures.append(
                                f"{label}: raw source duration "
                                f"{len(raw_source_audio) / raw_source_rate:.3f}s "
                                f"< {args.min_duration:.3f}s"
                            )
                        if latent_mode and not np.array_equal(
                            source_audio, raw_source_audio
                        ):
                            failures.append(
                                f"{label}: latent-alignment source is not pristine"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad raw source: {exc}")
            if raw_target_value:
                raw_target = Path(raw_target_value)
                if not raw_target.is_file():
                    failures.append(f"{label}: missing raw target: {raw_target}")
                else:
                    try:
                        raw_audio, raw_rate = read_pcm16(raw_target)
                        raw_duration = len(raw_audio) / raw_rate
                        if raw_rate != args.sample_rate:
                            failures.append(f"{label}: raw target sample rate={raw_rate}")
                        if raw_duration < args.min_duration:
                            failures.append(
                                f"{label}: raw target duration {raw_duration:.3f}s "
                                f"< {args.min_duration:.3f}s"
                            )
                        if (
                            (
                                align_meta.get("warp_side") == "source"
                                or latent_mode
                            )
                            and not np.array_equal(target_audio, raw_audio)
                        ):
                            failures.append(
                                f"{label}: source-side supervision target is not "
                                f"byte-equivalent PCM to its pristine raw target"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad raw target: {exc}")
            if alignment_value:
                alignment_path = Path(alignment_value)
                if not alignment_path.is_file():
                    failures.append(
                        f"{label}: missing latent alignment: {alignment_path}"
                    )
                else:
                    try:
                        positions = np.load(alignment_path, allow_pickle=False)
                        expected = len(target_audio) // 320
                        if positions.ndim != 1 or len(positions) != expected:
                            failures.append(
                                f"{label}: latent map shape={positions.shape}, "
                                f"expected ({expected},)"
                            )
                        if not np.all(np.isfinite(positions)):
                            failures.append(f"{label}: latent map has non-finite values")
                        if len(positions) and (
                            float(np.min(positions)) < 0.0
                            or float(np.max(positions)) > 1.0
                            or np.any(np.diff(positions) < 0.0)
                        ):
                            failures.append(
                                f"{label}: latent map must be monotonic in [0,1]"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad latent alignment: {exc}")
            if align_meta.get("warp_method") == "rubberband" and not reference_value:
                failures.append(f"{label}: missing clean target_reference_wav_path")
            if reference_value:
                reference = Path(reference_value)
                reference_paths.add(reference)
                reference_parts = reference.stem.split("_", 1)
                reference_prompt = (
                    reference_parts[1] if len(reference_parts) == 2 else reference.stem
                )
                if reference_prompt in paired_prompts:
                    failures.append(
                        f"{label}: clean reference prompt leaks paired/eval text: "
                        f"{reference_prompt}"
                    )
                if not reference.is_file():
                    failures.append(f"{label}: missing clean reference: {reference}")
                else:
                    try:
                        reference_audio, reference_rate = read_pcm16(reference)
                        reference_duration = len(reference_audio) / reference_rate
                        if reference_rate != args.sample_rate:
                            failures.append(
                                f"{label}: reference sample rate={reference_rate}"
                            )
                        if reference_duration < args.min_duration:
                            failures.append(
                                f"{label}: reference duration {reference_duration:.3f}s "
                                f"< {args.min_duration:.3f}s"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad clean reference: {exc}")

    if align_meta.get("warp_method") in {"rubberband", "latent"}:
        version = str(align_meta.get("rubberband_version", ""))
        engine = align_meta.get("rubberband_engine")
        if align_meta.get("warp_method") == "rubberband":
            try:
                major = int(version.split(".", 1)[0].split()[-1])
            except (ValueError, IndexError):
                major = 0
            if major < 3 or engine != "r3":
                failures.append(
                    f"align_meta: final dataset requires Rubber Band >=3 R3; "
                    f"got version={version!r}, engine={engine!r}"
                )
        local = align_meta.get("local_stretch_ratio") or {}
        ratio_epsilon = 1e-3  # align_meta stores rounded summary statistics
        if local and (
            float(local.get("min", 0)) + ratio_epsilon
            < float(local.get("allowed_min", 0))
            or float(local.get("max", 1e9)) - ratio_epsilon
            > float(local.get("allowed_max", 1e9))
        ):
            failures.append(f"align_meta: local stretch guard violated: {local}")
        allowed_global = align_meta.get("allowed_global_stretch") or [0.0, 1e9]
        max_anchor_removal = float(
            align_meta.get("max_anchor_removal_fraction", 1.0)
        )
        manifest_sources = {
            row["source_utt"] for rows in splits.values() for row in rows
        }
        if set(qc_by_source) != manifest_sources:
            failures.append(
                "alignment_qc.jsonl source IDs do not exactly match manifests"
            )
        for source_utt, qc in qc_by_source.items():
            ratio = float(qc["global_stretch_ratio"])
            if not float(allowed_global[0]) <= ratio <= float(allowed_global[1]):
                failures.append(
                    f"{source_utt}: global stretch {ratio:.4f} outside {allowed_global}"
                )
            removal = float(qc["anchor_removal_fraction"])
            if removal > max_anchor_removal:
                failures.append(
                    f"{source_utt}: anchor removal {removal:.1%} "
                    f"> {max_anchor_removal:.1%}"
                )

    report = {
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "train_prompts": len(prompts["train"]),
        "val_prompts": len(prompts["val"]),
        "prompt_overlap": 0,
        "target_rms_dbfs_min": round(min(rms_values), 2) if rms_values else None,
        "target_rms_dbfs_median": round(float(np.median(rms_values)), 2) if rms_values else None,
        "internal_zero_run_ms_max": round(max(zero_runs_ms), 2) if zero_runs_ms else None,
        "duration_seconds_min": round(min(durations), 3) if durations else None,
        "clipped_fraction_max": round(max(clipped_fractions), 7) if clipped_fractions else None,
        "absolute_dc_max": round(max(dc_offsets), 6) if dc_offsets else None,
        "clean_reference_files": len(reference_paths),
        "rubberband_version": align_meta.get("rubberband_version"),
        "rubberband_engine": align_meta.get("rubberband_engine"),
        "alignment_qc_rows": len(qc_rows),
        "failures": len(failures),
    }
    print(json.dumps(report, indent=2))
    if failures:
        print("\nFirst failures:")
        for failure in failures[:30]:
            print(" -", failure)
        raise SystemExit(1)
    print("\nPASS: cross-pair dataset is structurally safe for a training smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
