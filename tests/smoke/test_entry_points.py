"""Smoke tests for the consolidated entry points (scripts/train.py,
scripts/infer.py, scripts/validate_dataset.py).

No GPU, no torchrun: scripts/train.py is exercised up to the exec boundary via
``dry_run=true`` (config resolution, validation, resolved-config write, run
metadata); scripts/infer.py only through its argument/failure paths (the model
load needs the released checkpoint).
"""

import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import REPO_ROOT
from tests.unit.test_validate_crosspairs_legacy import build_dataset


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ----------------------------------------------------------------- train.py
def test_train_rejects_unknown_keys():
    result = run("train.py", "experiment=finetune_hindi", "rnak=4")
    assert result.returncode != 0
    assert "rnak" in result.stdout + result.stderr


def test_train_unknown_experiment_lists_alternatives():
    result = run("train.py", "experiment=does_not_exist")
    assert result.returncode != 0
    assert "not found" in result.stderr + result.stdout


def test_train_missing_warmstart_checkpoint_is_actionable(tmp_path):
    # finetune_hindi's datasets don't exist locally either, but the checkpoint
    # check must fire only after config validation passes; give the config
    # existing dataset stubs.
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"target_utt":"x"}\n')
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"base_config: configs/xvc.yaml\n"
        f"datasets:\n  train: [{manifest}]\n  val: [{manifest}]\n"
    )
    result = run("train.py", f"config={overlay}",
                 f"log_dir={tmp_path / 'exp'}",
                 f"checkpoint={tmp_path / 'missing.pt'}")
    assert result.returncode != 0
    assert "pretrained checkpoint not found" in result.stdout + result.stderr


def test_train_dry_run_resolves_validates_and_records(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"target_utt":"x"}\n')
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"base_config: configs/xvc.yaml\n"
        f"datasets:\n  train: [{manifest}]\n  val: [{manifest}]\n"
    )
    stub_ckpt = tmp_path / "stub.pt"
    stub_ckpt.write_bytes(b"not a real checkpoint")
    log_dir = tmp_path / "exp"

    result = run(
        "train.py", f"config={overlay}", f"log_dir={log_dir}",
        f"checkpoint={stub_ckpt}", "lr=5e-5", "dry_run=true",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (log_dir / "composed_config.yaml").is_file()
    assert (log_dir / "run_meta.json").is_file()
    meta = json.loads((log_dir / "run_meta.json").read_text())
    assert "scripts/train.py" in meta.get("launcher_cli", "")

    # The lr override landed in the resolved config the runtime will consume.
    resolved = (log_dir / "composed_config.yaml").read_text()
    assert "5e-05" in resolved or "5.0e-05" in resolved
    # And the exec line targets the unchanged runtime.
    assert "-m bins.train" in result.stdout.replace("'", "")


def test_train_invalid_config_fails_before_anything_else(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("{}\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"base_config: configs/xvc.yaml\n"
        f"datasets:\n  train: [{manifest}]\n  val: [{manifest}]\n"
        f"model:\n  generator:\n    lora:\n      enabled: true\n      r: -3\n"
    )
    result = run("train.py", f"config={overlay}", f"log_dir={tmp_path / 'exp'}",
                 "dry_run=true")
    assert result.returncode != 0
    assert "rank must be a positive" in result.stdout + result.stderr


# ----------------------------------------------------------------- infer.py
def test_infer_requires_the_four_keys():
    result = run("infer.py", "checkpoint=ckpts/none.pt")
    assert result.returncode != 0
    assert "missing required key(s)" in result.stdout + result.stderr
    assert "source" in result.stdout + result.stderr


def test_infer_checks_files_before_heavy_imports(tmp_path):
    result = run(
        "infer.py", f"checkpoint={tmp_path / 'no.pt'}",
        "source=examples/source.wav", "target=examples/target.wav",
        f"output={tmp_path / 'out.wav'}",
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stdout + result.stderr


# ------------------------------------------------------- validate_dataset.py
def test_validate_dataset_positional_form(tmp_path):
    build_dataset(tmp_path, train_prompts=["a0001"], val_prompts=["a0002"])
    result = run("validate_dataset.py", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "validation_report.json").read_text())
    assert report["schema_version"] == 1
