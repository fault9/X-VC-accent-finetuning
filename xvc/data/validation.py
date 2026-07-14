"""Cross-pair dataset validation engine shared by old and new CLIs."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import (
    CROSSPAIR_SCHEMA_VERSION,
    QCGates,
    check_qc_row,
    missing_manifest_fields,
    required_manifest_fields_for_meta,
)


EVAL_PROMPTS = {f"arctic_b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}


@dataclass(frozen=True)
class Thresholds:
    sample_rate: int = 16000
    max_internal_zero_ms: float = 100.0
    min_rms_dbfs: float = -45.0
    min_duration: float = 3.0
    max_clipped_fraction: float = 0.0001
    max_abs_dc: float = 0.01


@dataclass
class ValidationResult:
    report: dict
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def write_report_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.report, indent=2) + "\n", encoding="utf-8"
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = missing_manifest_fields(row)
            if missing:
                raise ValueError(f"{path}:{number}: missing {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _prompt_id(row: dict) -> str:
    stem = Path(row["source_wav_path"]).stem
    if "_arctic_" not in stem:
        raise ValueError(f"cannot derive ARCTIC prompt from {stem!r}")
    return "arctic_" + stem.split("_arctic_", 1)[1]


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected mono PCM16, got channels={wav.getnchannels()} "
                f"sample_width={wav.getsampwidth()}"
            )
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return audio, rate


def _max_internal_zero_run(samples: np.ndarray) -> int:
    zero = samples == 0
    changes = np.diff(np.r_[False, zero, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    internal = [
        end - start
        for start, end in zip(starts, ends)
        if start > 0 and end < len(samples)
    ]
    return max(internal, default=0)


def _empty_report() -> dict:
    return {
        "schema_version": CROSSPAIR_SCHEMA_VERSION,
        "train_pairs": 0,
        "val_pairs": 0,
        "train_prompts": 0,
        "val_prompts": 0,
        "prompt_overlap": 0,
        "source_speakers": None,
        "target_speakers": None,
        "target_rms_dbfs_min": None,
        "target_rms_dbfs_median": None,
        "internal_zero_run_ms_max": None,
        "duration_seconds_min": None,
        "clipped_fraction_max": None,
        "absolute_dc_max": None,
        "clean_reference_files": 0,
        "rubberband_version": None,
        "rubberband_engine": None,
        "alignment_qc_rows": 0,
        "failures": 0,
    }


def validate_crosspair_dataset(
    data_root: str | Path, thresholds: Thresholds | None = None
) -> ValidationResult:
    thresholds = thresholds or Thresholds()
    root = Path(data_root)
    align_meta = _load_json(root / "align_meta.json")
    qc_path = root / "alignment_qc.jsonl"
    qc_rows = (
        [
            json.loads(line)
            for line in qc_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if qc_path.is_file()
        else []
    )
    failures: list[str] = []
    try:
        splits = {
            split: _load_jsonl(root / "manifests" / f"{split}.jsonl")
            for split in ("train", "val")
        }
    except Exception as exc:
        report = _empty_report()
        failures.append(str(exc))
        report["failures"] = len(failures)
        return ValidationResult(report, failures)

    prompts = {
        split: {_prompt_id(row) for row in rows} for split, rows in splits.items()
    }
    overlap = prompts["train"] & prompts["val"]
    if overlap:
        failures.append(
            "train/val prompt leakage: " + ", ".join(sorted(overlap)[:20])
        )
    paired_prompts = prompts["train"] | prompts["val"] | EVAL_PROMPTS
    latent_mode = align_meta.get("warp_method") == "latent"
    required_for_mode = required_manifest_fields_for_meta(align_meta)

    seen_targets: set[Path] = set()
    rms_values: list[float] = []
    zero_runs_ms: list[float] = []
    clipped_fractions: list[float] = []
    dc_offsets: list[float] = []
    durations: list[float] = []
    reference_paths: set[Path] = set()

    for split, rows in splits.items():
        for index, row in enumerate(rows, 1):
            label = f"{split}:{index}"
            for field in required_for_mode:
                if not row.get(field):
                    failures.append(f"{label}: missing {field}")
            source = Path(row["source_wav_path"])
            target = Path(row["target_wav_path"])
            if not source.is_file() or not target.is_file():
                failures.append(f"{label}: missing path: {source} or {target}")
                continue
            if target in seen_targets:
                failures.append(f"{label}: duplicate target path: {target}")
            seen_targets.add(target)
            try:
                source_audio, source_rate = _read_pcm16(source)
                target_audio, target_rate = _read_pcm16(target)
            except Exception as exc:
                failures.append(f"{label}: {exc}")
                continue

            if (
                source_rate != thresholds.sample_rate
                or target_rate != thresholds.sample_rate
            ):
                failures.append(
                    f"{label}: sample rates source={source_rate}, target={target_rate}"
                )
            if len(source_audio) != len(target_audio) and not latent_mode:
                failures.append(
                    f"{label}: duration mismatch source={len(source_audio)}, "
                    f"target={len(target_audio)} samples"
                )
            duration = len(target_audio if latent_mode else source_audio) / source_rate
            durations.append(duration)
            if duration < thresholds.min_duration:
                failures.append(
                    f"{label}: duration {duration:.3f}s < "
                    f"{thresholds.min_duration:.3f}s"
                )
            for field in ("raw_source_duration", "raw_target_duration"):
                if field in row and float(row[field]) < thresholds.min_duration:
                    failures.append(
                        f"{label}: {field}={float(row[field]):.3f}s < "
                        f"{thresholds.min_duration:.3f}s"
                    )

            rms = float(np.sqrt(np.mean(target_audio.astype(np.float64) ** 2)))
            dbfs = 20.0 * math.log10(max(rms, 1e-12) / 32768.0)
            rms_values.append(dbfs)
            zero_ms = 1000.0 * _max_internal_zero_run(target_audio) / target_rate
            zero_runs_ms.append(zero_ms)
            clipped = float(
                np.mean(np.abs(target_audio.astype(np.int32)) >= 32760)
            )
            clipped_fractions.append(clipped)
            dc_offset = abs(
                float(np.mean(target_audio.astype(np.float64))) / 32768.0
            )
            dc_offsets.append(dc_offset)
            if dbfs < thresholds.min_rms_dbfs:
                failures.append(f"{label}: target RMS too low ({dbfs:.1f} dBFS): {target}")
            if zero_ms > thresholds.max_internal_zero_ms:
                failures.append(f"{label}: internal exact-zero gap {zero_ms:.1f} ms: {target}")
            if clipped > thresholds.max_clipped_fraction:
                failures.append(f"{label}: clipped fraction {clipped:.6f}: {target}")
            if dc_offset > thresholds.max_abs_dc:
                failures.append(f"{label}: absolute DC offset {dc_offset:.5f}: {target}")

            raw_source_value = row.get("raw_source_wav_path")
            raw_target_value = row.get("raw_target_wav_path")
            alignment_value = row.get("latent_alignment_path")
            reference_value = row.get("target_reference_wav_path")

            if raw_source_value:
                raw_source = Path(raw_source_value)
                if not raw_source.is_file():
                    failures.append(f"{label}: missing raw source: {raw_source}")
                else:
                    try:
                        raw_audio, raw_rate = _read_pcm16(raw_source)
                        raw_duration = len(raw_audio) / raw_rate
                        if raw_rate != thresholds.sample_rate:
                            failures.append(f"{label}: raw source sample rate={raw_rate}")
                        if raw_duration < thresholds.min_duration:
                            failures.append(
                                f"{label}: raw source duration {raw_duration:.3f}s < "
                                f"{thresholds.min_duration:.3f}s"
                            )
                        if latent_mode and not np.array_equal(source_audio, raw_audio):
                            failures.append(f"{label}: latent-alignment source is not pristine")
                    except Exception as exc:
                        failures.append(f"{label}: bad raw source: {exc}")

            if raw_target_value:
                raw_target = Path(raw_target_value)
                if not raw_target.is_file():
                    failures.append(f"{label}: missing raw target: {raw_target}")
                else:
                    try:
                        raw_audio, raw_rate = _read_pcm16(raw_target)
                        raw_duration = len(raw_audio) / raw_rate
                        if raw_rate != thresholds.sample_rate:
                            failures.append(f"{label}: raw target sample rate={raw_rate}")
                        if raw_duration < thresholds.min_duration:
                            failures.append(
                                f"{label}: raw target duration {raw_duration:.3f}s < "
                                f"{thresholds.min_duration:.3f}s"
                            )
                        if (
                            (align_meta.get("warp_side") == "source" or latent_mode)
                            and not np.array_equal(target_audio, raw_audio)
                        ):
                            failures.append(
                                f"{label}: source-side supervision target is not "
                                "byte-equivalent PCM to its pristine raw target"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad raw target: {exc}")

            if alignment_value:
                alignment_path = Path(alignment_value)
                if not alignment_path.is_file():
                    failures.append(f"{label}: missing latent alignment: {alignment_path}")
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

            if reference_value:
                reference = Path(reference_value)
                reference_paths.add(reference)
                parts = reference.stem.split("_", 1)
                reference_prompt = parts[1] if len(parts) == 2 else reference.stem
                if reference_prompt in paired_prompts:
                    failures.append(
                        f"{label}: clean reference prompt leaks paired/eval text: "
                        f"{reference_prompt}"
                    )
                if not reference.is_file():
                    failures.append(f"{label}: missing clean reference: {reference}")
                else:
                    try:
                        ref_audio, ref_rate = _read_pcm16(reference)
                        ref_duration = len(ref_audio) / ref_rate
                        if ref_rate != thresholds.sample_rate:
                            failures.append(f"{label}: reference sample rate={ref_rate}")
                        if ref_duration < thresholds.min_duration:
                            failures.append(
                                f"{label}: reference duration {ref_duration:.3f}s < "
                                f"{thresholds.min_duration:.3f}s"
                            )
                    except Exception as exc:
                        failures.append(f"{label}: bad clean reference: {exc}")

    if align_meta.get("warp_method") in {"rubberband", "latent"}:
        if align_meta.get("warp_method") == "rubberband":
            version = str(align_meta.get("rubberband_version", ""))
            engine = align_meta.get("rubberband_engine")
            try:
                major = int(version.split(".", 1)[0].split()[-1])
            except (ValueError, IndexError):
                major = 0
            if major < 3 or engine != "r3":
                failures.append(
                    "align_meta: final dataset requires Rubber Band >=3 R3; "
                    f"got version={version!r}, engine={engine!r}"
                )
        local = align_meta.get("local_stretch_ratio") or {}
        epsilon = 1e-3
        if local and (
            float(local.get("min", 0)) + epsilon < float(local.get("allowed_min", 0))
            or float(local.get("max", 1e9)) - epsilon
            > float(local.get("allowed_max", 1e9))
        ):
            failures.append(f"align_meta: local stretch guard violated: {local}")

        qc_by_source = {row["source_utt"]: row for row in qc_rows}
        manifest_sources = {
            row["source_utt"] for rows in splits.values() for row in rows
        }
        if set(qc_by_source) != manifest_sources:
            failures.append("alignment_qc.jsonl source IDs do not exactly match manifests")
        gates = QCGates.from_align_meta(align_meta)
        for source_utt, row in qc_by_source.items():
            failures.extend(check_qc_row(source_utt, row, gates))

    report = {
        "schema_version": CROSSPAIR_SCHEMA_VERSION,
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "train_prompts": len(prompts["train"]),
        "val_prompts": len(prompts["val"]),
        "prompt_overlap": len(overlap),
        "source_speakers": align_meta.get("source_speakers"),
        "target_speakers": align_meta.get("target_speakers"),
        "target_rms_dbfs_min": round(min(rms_values), 2) if rms_values else None,
        "target_rms_dbfs_median": (
            round(float(np.median(rms_values)), 2) if rms_values else None
        ),
        "internal_zero_run_ms_max": (
            round(max(zero_runs_ms), 2) if zero_runs_ms else None
        ),
        "duration_seconds_min": round(min(durations), 3) if durations else None,
        "clipped_fraction_max": (
            round(max(clipped_fractions), 7) if clipped_fractions else None
        ),
        "absolute_dc_max": round(max(dc_offsets), 6) if dc_offsets else None,
        "clean_reference_files": len(reference_paths),
        "rubberband_version": align_meta.get("rubberband_version"),
        "rubberband_engine": align_meta.get("rubberband_engine"),
        "alignment_qc_rows": len(qc_rows),
        "failures": len(failures),
    }
    return ValidationResult(report, failures)
