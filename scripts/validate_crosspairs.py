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
    args = parser.parse_args()

    root = Path(args.data_root)
    manifests = root / "manifests"
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

    failures = []
    seen_targets = set()
    rms_values = []
    zero_runs_ms = []
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
            if len(source_audio) != len(target_audio):
                failures.append(
                    f"{label}: duration mismatch source={len(source_audio)}, "
                    f"target={len(target_audio)} samples"
                )
            rms = float(np.sqrt(np.mean(target_audio.astype(np.float64) ** 2)))
            dbfs = 20.0 * math.log10(max(rms, 1e-12) / 32768.0)
            rms_values.append(dbfs)
            zero_ms = 1000.0 * max_internal_zero_run(target_audio) / target_rate
            zero_runs_ms.append(zero_ms)
            if dbfs < args.min_rms_dbfs:
                failures.append(f"{label}: target RMS too low ({dbfs:.1f} dBFS): {target}")
            if zero_ms > args.max_internal_zero_ms:
                failures.append(
                    f"{label}: internal exact-zero gap {zero_ms:.1f} ms: {target}"
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
