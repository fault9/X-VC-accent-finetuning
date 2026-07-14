"""Characterization tests for scripts/validate_crosspairs.py (legacy validator).

Builds a tiny synthetic cross-pair dataset (mono PCM16, 16 kHz, >3 s) and runs
the validator as a subprocess, exactly like run_guarded_train_eval.sh does.

Also pins the KNOWN BUG this refactor must fix: an ``alignment_qc.jsonl`` row
without ``anchor_removal_fraction`` crashes with a raw ``KeyError`` instead of
a validation failure. When xvc.data.validation replaces the internals, that
test flips to asserting an actionable error message.
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


def build_dataset(root: Path, train_prompts, val_prompts) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    for split, prompts in (("train", train_prompts), ("val", val_prompts)):
        rows = []
        for i, prompt in enumerate(prompts):
            source = root / "wav" / f"clb_arctic_{prompt}.wav"
            target = root / "wav" / f"ASI_{split}_arctic_{prompt}.wav"
            write_wav(source, seed=i)
            write_wav(target, seed=100 + i)
            rows.append(
                {
                    "source_utt": f"clb_arctic_{prompt}",
                    "source_wav_path": str(source),
                    "target_utt": f"ASI_arctic_{prompt}_ch",
                    "target_wav_path": str(target),
                }
            )
        with (manifests / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def run_validator(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--data-root", str(root)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_valid_dataset_passes(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001", "a0003"], val_prompts=["a0002"])
    result = run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout[: result.stdout.rindex("}") + 1])
    assert report["train_pairs"] == 2
    assert report["val_pairs"] == 1
    assert report["failures"] == 0


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


def test_missing_anchor_removal_fraction_is_a_raw_keyerror(tmp_path):
    """KNOWN BUG (pinned on purpose): legacy QC rows without
    ``anchor_removal_fraction`` crash the validator with a bare KeyError
    instead of a validation failure. xvc.data.validation must turn this into
    an actionable schema error; this test documents the pre-fix behavior of
    the legacy script and will be updated if/when the script is rewired.
    """
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"])
    (tmp_path / "align_meta.json").write_text(json.dumps({"warp_method": "latent"}))
    qc_rows = [
        {"source_utt": "clb_arctic_a0001", "global_stretch_ratio": 1.0},
        {"source_utt": "clb_arctic_a0002", "global_stretch_ratio": 1.0},
    ]
    with (tmp_path / "alignment_qc.jsonl").open("w") as f:
        for row in qc_rows:
            f.write(json.dumps(row) + "\n")
    result = run_validator(tmp_path)
    assert result.returncode != 0
    assert "KeyError: 'anchor_removal_fraction'" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
