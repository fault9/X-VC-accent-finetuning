#!/usr/bin/env python3
"""Primary X-VC training entry point.

Resolves and validates an experiment config, records reproducibility metadata,
runs the cross-pair dataset preflight, then launches the unchanged training
runtime (``bins/train.py``) under torchrun — exactly the sequence
``scripts/finetune.sh`` performs, minus the interactive HF-publish prompt.

Usage (repo root, container):

    python scripts/train.py experiment=finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16
    python scripts/train.py config=configs/finetune_hindi.yaml gpus=1 seed=1234
    python scripts/train.py experiment=<name> resume_step=1500       # resume
    python scripts/train.py experiment=<name> checkpoint=ckpts/xvc.pt lr=5e-5

``experiment=NAME`` looks for ``configs/experiment/NAME.yaml`` (compositional,
Phase 4) first and falls back to ``configs/NAME.yaml`` (legacy overlay), so
every historical experiment name keeps working.

Keys: experiment | config | checkpoint (default ckpts/xvc.pt for fresh runs) |
log_dir (default exp/<experiment>) | resume_step (default 0) | seed (1234) |
gpus (1) | port (10201) | num_workers (4) | lr | wandb (false) |
validate (true) | deepspeed_config (configs/ds_stage2.json) | engine (deepspeed)

This launcher is import-light (no torch); the heavy work happens inside the
torchrun children. Legacy entry points are unchanged: ``bins/train.py`` is
still the runtime, ``scripts/finetune.sh`` still works.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from xvc.utils.cli import as_bool, parse_key_value_args
from xvc.utils.config import (
    ConfigValidationError,
    load_config,
    validate_experiment_config,
)

KNOWN_KEYS = {
    "experiment", "config", "checkpoint", "log_dir", "resume_step", "seed",
    "gpus", "port", "num_workers", "lr", "wandb", "validate",
    "deepspeed_config", "engine", "project", "dry_run",
}


def _resolve_config_path(overrides: dict) -> tuple[Path, str]:
    """(config path, experiment name) from experiment=/config= overrides."""
    if "config" in overrides:
        path = Path(overrides["config"])
        if not path.is_file():
            raise SystemExit(f"config not found: {path}")
        return path, overrides.get("experiment", path.stem)
    if "experiment" in overrides:
        name = overrides["experiment"]
        for candidate in (
            REPO_ROOT / "configs" / "experiment" / f"{name}.yaml",
            REPO_ROOT / "configs" / f"{name}.yaml",
        ):
            if candidate.is_file():
                return candidate, name
        raise SystemExit(
            f"experiment '{name}' not found under configs/experiment/ or "
            f"configs/ — available experiments: "
            + ", ".join(sorted(
                p.stem for p in (REPO_ROOT / "configs").glob("finetune_*.yaml")
            )[:8]) + ", ..."
        )
    raise SystemExit(
        "specify experiment=<name> or config=<path> "
        "(e.g. `python scripts/train.py experiment=finetune_hindi`)"
    )


def _maybe_validate_dataset(cfg, min_duration_env: str) -> None:
    """Cross-pair preflight, mirroring finetune.sh: if the train manifest lives
    in a <root>/manifests/ directory with a val sibling, run the validator."""
    train_entries = cfg.get("datasets", {}).get("train", [])
    entries = [train_entries] if isinstance(train_entries, str) else list(train_entries)
    if not entries:
        return
    manifest = Path(str(entries[0]))
    if manifest.parent.name != "manifests":
        return
    root = manifest.parent.parent
    if not (root / "manifests" / "val.jsonl").is_file():
        return
    print(f"[train] validating cross-pair dataset at {root}")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_crosspairs.py"),
         "--data-root", str(root), "--min-duration", min_duration_env],
        check=True, cwd=REPO_ROOT,
    )


def main(argv=None) -> int:
    # Config-internal paths (base_config, datasets, checkpoints) are repo-root
    # relative throughout this repo; behave the same from any invocation dir.
    os.chdir(REPO_ROOT)
    overrides, passthrough = parse_key_value_args(
        sys.argv[1:] if argv is None else argv
    )
    unknown = set(overrides) - KNOWN_KEYS
    if unknown:
        raise SystemExit(
            f"unknown key(s): {sorted(unknown)}; known: {sorted(KNOWN_KEYS)}"
        )
    if passthrough:
        raise SystemExit(
            f"unexpected arguments {passthrough}; this entry point takes "
            f"key=value pairs (legacy flags belong to bins/train.py)"
        )

    config_path, experiment = _resolve_config_path(overrides)
    log_dir = Path(overrides.get("log_dir", REPO_ROOT / "exp" / experiment))
    resume_step = int(overrides.get("resume_step", 0))
    seed = int(overrides.get("seed", 1234))
    gpus = int(overrides.get("gpus", 1))
    port = int(overrides.get("port", 10201))
    num_workers = int(overrides.get("num_workers", 4))
    engine = overrides.get("engine", "deepspeed")
    deepspeed_config = overrides.get("deepspeed_config", "configs/ds_stage2.json")
    checkpoint = Path(overrides.get("checkpoint", "ckpts/xvc.pt"))
    enable_wandb = as_bool(overrides.get("wandb", "false"), "wandb")
    run_preflight = as_bool(overrides.get("validate", "true"), "validate")

    # 1) Resolve (recursively / compositionally) and validate before any GPU work.
    cfg = load_config(config_path)
    if "lr" in overrides:
        OmegaConf.update(cfg, "model.generator.optim_conf.lr", float(overrides["lr"]))
    try:
        validate_experiment_config(cfg, check_paths=True)
    except ConfigValidationError as exc:
        raise SystemExit(f"[train] {exc}") from exc

    # 2) Materialize the resolved config; the runtime consumes a plain file
    #    (same pattern as finetune.sh's patched copies — the legacy loader
    #    resolves base_config only one level, a resolved file has none).
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = log_dir / "composed_config.yaml"
    OmegaConf.save(cfg, resolved_path)

    # 3) Warm start vs resume, exactly like finetune.sh.
    checkpoint_args = []
    if resume_step == 0:
        if not checkpoint.is_file():
            raise SystemExit(
                f"pretrained checkpoint not found: {checkpoint}\n"
                f"Download the released X-VC checkpoint to ckpts/ "
                f"(see docs/finetuning.md), or pass checkpoint=<path>."
            )
        checkpoint_args = ["--checkpoint", str(checkpoint)]

    # 4) Dataset preflight before touching the GPU.
    if run_preflight and resume_step == 0:
        _maybe_validate_dataset(cfg, os.environ.get("XVC_VALIDATE_MIN_DURATION", "3.0"))

    # 5) Reproducibility record — before training, so crashed runs are traceable.
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    cli_record = "python scripts/train.py " + " ".join(
        shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv)
    )
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "write_run_meta.py"),
         "--log_dir", str(log_dir), "--config", str(resolved_path),
         *(["--checkpoint", str(checkpoint)] if checkpoint_args else []),
         "--seed", str(seed), "--cli", f"{cli_record} (tag {tag})"],
        check=True, cwd=REPO_ROOT,
    )

    # 6) Hand over to the unchanged runtime under torchrun.
    runtime = [
        "torchrun", f"--nnodes=1", f"--nproc_per_node={gpus}",
        f"--master_port={port}",
        "-m", "bins.train",
        "--config", str(resolved_path),
        "--log_dir", str(log_dir),
        "--train_engine", engine,
        *(["--deepspeed_config", deepspeed_config] if engine == "deepspeed" else []),
        "--resume_step", str(resume_step),
        "--seed", str(seed),
        "--num_workers", str(num_workers),
        "--project", overrides.get("project", "x-vc-finetune"),
        "--date", tag,
        *(["--enable_wandb"] if enable_wandb else []),
        *checkpoint_args,
    ]
    print("[train] exec:", " ".join(shlex.quote(a) for a in runtime))
    if as_bool(overrides.get("dry_run", "false"), "dry_run"):
        print("[train] dry_run=true: config resolved, validated, and saved; "
              "run metadata written; not launching torchrun.")
        return 0
    os.execvp(runtime[0], runtime)


if __name__ == "__main__":
    raise SystemExit(main())
