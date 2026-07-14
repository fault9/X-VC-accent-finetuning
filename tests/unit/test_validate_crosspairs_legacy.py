"""Tests for scripts/validate_crosspairs.py (now a wrapper over xvc.data.validation).

Builds tiny synthetic cross-pair datasets (mono PCM16, 16 kHz, >3 s) and runs
the validator CLI exactly like run_guarded_train_eval.sh does.

History: Phase 1 pinned a KNOWN BUG here — a legacy ``alignment_qc.jsonl`` row
without ``anchor_removal_fraction`` crashed with a raw ``KeyError``. The
engine now (a) fails with an actionable reason when ``align_meta.json``
configures the gate, and (b) tolerates the missing field when no gate is
configured (the legacy defaults made the gate vacuous in that case anyway).
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

VALIDATOR = REPO_ROOT / "scripts" / "validate_crosspairs.py"
SAMPLE_RATE = 16000
HOP = 320  # latent map frames per sample, fixed by the validator


def write_wav(path: Path, seconds: float = 3.5, seed: int = 0) -> None:
    """Mono PCM16 noise that passes every quality gate (RMS, DC, zeros, clip)."""
    rng = np.random.default_rng(seed)
    samples = rng.integers(-3000, 3000, size=int(seconds * SAMPLE_RATE))
    samples = (samples - int(samples.mean())).astype(np.int16)
    samples[samples == 0] = 7  # no internal zero runs
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(samples.tobytes())


def build_dataset(root: Path, train_prompts, val_prompts, latent: bool = False):
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    for split, prompts in (("train", train_prompts), ("val", val_prompts)):
        rows = []
        for i, prompt in enumerate(prompts):
            source = root / "wav" / f"clb_arctic_{prompt}.wav"
            target = root / "wav" / f"ASI_{split}_arctic_{prompt}.wav"
            write_wav(source, seed=i)
            write_wav(target, seed=100 + i)
            row = {
                "source_utt": f"clb_arctic_{prompt}",
                "source_wav_path": str(source),
                "target_utt": f"ASI_arctic_{prompt}_ch",
                "target_wav_path": str(target),
            }
            if latent:
                # Latent mode: source/target stay pristine; a monotonic [0,1]
                # DTW position map covers the target timeline.
                with wave.open(str(target), "rb") as f:
                    n_target = f.getnframes()
                positions = np.linspace(0.0, 1.0, n_target // HOP).astype(np.float32)
                align_path = root / "align" / f"{prompt}.npy"
                align_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(align_path, positions)
                row.update(
                    raw_source_wav_path=str(source),
                    raw_target_wav_path=str(target),
                    latent_alignment_path=str(align_path),
                )
            rows.append(row)
        with (manifests / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def write_qc(root: Path, rows):
    with (root / "alignment_qc.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_validator(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--data-root", str(root)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_valid_dataset_passes_and_saves_report(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001", "a0003"], val_prompts=["a0002"])
    result = run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout[: result.stdout.rindex("}") + 1])
    assert report["train_pairs"] == 2
    assert report["val_pairs"] == 1
    assert report["failures"] == 0
    assert report["schema_version"] == 1
    saved = json.loads((tmp_path / "validation_report.json").read_text())
    assert saved == report


def test_train_val_prompt_leakage_fails(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0001"])
    result = run_validator(tmp_path)
    assert result.returncode != 0
    assert "prompt leakage" in (result.stdout + result.stderr)


def test_missing_required_manifest_field_fails_with_reason(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"])
    train = tmp_path / "manifests" / "train.jsonl"
    row = json.loads(train.read_text())
    del row["source_utt"]
    train.write_text(json.dumps(row) + "\n")
    result = run_validator(tmp_path)
    assert result.returncode != 0
    assert "source_utt" in (result.stdout + result.stderr)


def test_missing_gated_qc_field_is_actionable_failure_not_keyerror(tmp_path):
    """The Phase-1 pinned KeyError, now fixed: align_meta configures the
    anchor-removal gate, the QC rows predate the field -> a readable failure
    naming the field and the remedy, no traceback."""
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"], latent=True)
    (tmp_path / "align_meta.json").write_text(
        json.dumps({"warp_method": "latent", "max_anchor_removal_fraction": 0.4})
    )
    write_qc(tmp_path, [
        {"source_utt": "clb_arctic_a0001", "global_stretch_ratio": 1.0},
        {"source_utt": "clb_arctic_a0002", "global_stretch_ratio": 1.0},
    ])
    result = run_validator(tmp_path)
    assert result.returncode != 0
    assert "KeyError" not in result.stderr
    combined = result.stdout + result.stderr
    assert "anchor_removal_fraction" in combined
    assert "no safe default" in combined


def test_missing_qc_field_without_configured_gate_is_tolerated(tmp_path):
    """Old archives whose align_meta never configured the gates must still be
    inspectable: missing QC fields are not failures when no gate is active."""
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"], latent=True)
    (tmp_path / "align_meta.json").write_text(json.dumps({"warp_method": "latent"}))
    write_qc(tmp_path, [
        {"source_utt": "clb_arctic_a0001"},
        {"source_utt": "clb_arctic_a0002"},
    ])
    result = run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_configured_gate_still_enforced(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"], latent=True)
    (tmp_path / "align_meta.json").write_text(
        json.dumps({"warp_method": "latent", "max_anchor_removal_fraction": 0.4})
    )
    write_qc(tmp_path, [
        {"source_utt": "clb_arctic_a0001", "anchor_removal_fraction": 0.9},
        {"source_utt": "clb_arctic_a0002", "anchor_removal_fraction": 0.1},
    ])
    result = run_validator(tmp_path)
    assert result.returncode != 0
    assert "anchor removal 90.0% > 40.0%" in (result.stdout + result.stderr)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
