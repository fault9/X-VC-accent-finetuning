#!/usr/bin/env python3
"""
Smoke test for X-VC accent fine-tuning - validate before launching a full run.

Runs up to three independent stages:

  1. manifests  Validate the train/val JSONL: schema, that every referenced WAV
                exists and is readable, and total durations. STANDARD LIBRARY ONLY
                (no torch), so it runs on a plain interpreter / laptop.

  2. batch      Build the real dataloader and pull ONE batch; print tensor shapes.
                Needs torch + audio deps, but NO large model downloads.

  3. forward    Instantiate the full model, apply freezing, print trainable/frozen
                parameter counts, run ONE forward pass on GPU, and report peak GPU
                memory. This stage DOWNLOADS the pretrained dependencies
                (GLM-4-Voice tokenizer, ERes2Net) and needs a CUDA GPU.

Examples
--------
Local (no torch) - just validate the manifests:
    python bins/smoke_test.py --stage manifests \
        --manifests data/finetuning_audio/manifests/arabic/train.jsonl \
                    data/finetuning_audio/manifests/arabic/val.jsonl

On the GPU container - full check for one accent:
    python bins/smoke_test.py --stage all --config configs/finetune_arabic.yaml

Data-only check driven by the config (needs omegaconf; no model download):
    python bins/smoke_test.py --stage batch --config configs/finetune_arabic.yaml

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import wave
from pathlib import Path
from typing import List, Optional

# Make the repository importable whether run as `python bins/smoke_test.py`
# or `python -m bins.smoke_test`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FIELDS = ("source_wav_path", "target_wav_path", "target_utt")


# --------------------------------------------------------------------------- #
# Stage 1: manifest validation (standard library only)
# --------------------------------------------------------------------------- #
def _resolve(path_str: str) -> Path:
    """Resolve a manifest path (repo-relative or absolute) to a filesystem path."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _wav_seconds(path: Path) -> Optional[float]:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else None
    except Exception:
        return None


def validate_manifests(manifest_paths: List[str], limit: Optional[int] = None) -> bool:
    """Return True if all manifests are well-formed and every WAV is readable."""
    ok = True
    print("=" * 72)
    print("STAGE 1 - manifest validation")
    print("=" * 72)
    for mpath in manifest_paths:
        mfile = _resolve(mpath)
        print(f"\n[manifest] {mpath}")
        if not mfile.is_file():
            print(f"  ERROR: manifest file not found: {mfile}")
            ok = False
            continue

        n = 0
        missing_fields = 0
        missing_wavs = 0
        unreadable = 0
        total_sec = 0.0
        self_recon = 0
        with open(mfile, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if limit is not None and n >= limit:
                    break
                n += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ERROR: line {line_no} is not valid JSON: {e}")
                    ok = False
                    continue
                if not all(k in rec for k in REQUIRED_FIELDS):
                    missing_fields += 1
                    ok = False
                    continue
                if rec["source_wav_path"] == rec["target_wav_path"]:
                    self_recon += 1
                for key in ("source_wav_path", "target_wav_path"):
                    wp = _resolve(rec[key])
                    if not wp.is_file():
                        missing_wavs += 1
                        ok = False
                        if missing_wavs <= 3:
                            print(f"  MISSING: {rec[key]}")
                        continue
                    dur = _wav_seconds(wp)
                    if dur is None:
                        unreadable += 1
                        ok = False
                        if unreadable <= 3:
                            print(f"  UNREADABLE: {rec[key]}")
                    elif key == "target_wav_path":
                        total_sec += dur

        print(
            f"  lines={n}  self_reconstruction={self_recon}/{n}  "
            f"target_audio={total_sec/60.0:.2f} min"
        )
        if missing_fields:
            print(f"  ERROR: {missing_fields} line(s) missing required fields {REQUIRED_FIELDS}")
        if missing_wavs:
            print(f"  ERROR: {missing_wavs} referenced WAV(s) not found")
        if unreadable:
            print(f"  ERROR: {unreadable} referenced WAV(s) unreadable")
        if not (missing_fields or missing_wavs or unreadable):
            print("  OK")
    print("\n" + ("PASS" if ok else "FAIL") + " - stage 1 (manifests)")
    return ok


# --------------------------------------------------------------------------- #
# Config helpers (used by stages 2 & 3; import omegaconf lazily)
# --------------------------------------------------------------------------- #
def _load_config(config_path: str):
    from utils.file import load_config
    return load_config(config_path)


def _manifests_from_config(config) -> List[str]:
    manifests: List[str] = []
    for split in ("train", "val"):
        entry = config["datasets"][split]
        # entry may be a str, list, or OmegaConf ListConfig (which is neither
        # list nor tuple) — coerce every element to str.
        if isinstance(entry, str):
            manifests.append(entry)
        else:
            manifests.extend(str(m) for m in entry)
    return manifests


# --------------------------------------------------------------------------- #
# Stage 2: build the dataloader and pull one batch (torch; no downloads)
# --------------------------------------------------------------------------- #
def load_one_batch(config, batch_size: Optional[int] = None):
    import hydra

    if batch_size is not None:
        config["dataloader"]["static"]["batch_size"] = batch_size

    print("=" * 72)
    print("STAGE 2 - build dataloader, pull one batch")
    print("=" * 72)
    dataset_sampler = hydra.utils.instantiate(config["dataloader"], config)
    dataset = dataset_sampler.sample()
    bs = config["dataloader"]["static"]["batch_size"]
    print(f"  batch_size = {bs}")

    batch = next(iter(dataset))
    print("  one batch pulled. tensor shapes:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"    {k:22s} {tuple(v.shape)}  {v.dtype}")
        elif isinstance(v, list):
            print(f"    {k:22s} list[{len(v)}]  e.g. {v[0] if v else None!r}")
        else:
            print(f"    {k:22s} {type(v).__name__} = {v!r}")
    print("\nPASS - stage 2 (batch)")
    return batch


# --------------------------------------------------------------------------- #
# Stage 3: instantiate model, freeze, forward pass, report GPU memory
# --------------------------------------------------------------------------- #
def forward_pass(config, batch=None, batch_size: Optional[int] = None):
    import torch
    from types import SimpleNamespace
    from utils.train_utils import (
        init_models,
        freeze_model_parameters,
        params_statistic,
        verify_trainable_modules,
    )

    print("=" * 72)
    print("STAGE 3 - instantiate model, freeze, forward pass, GPU memory")
    print("=" * 72)
    if not torch.cuda.is_available():
        print("  ERROR: no CUDA device available; stage 3 requires a GPU.")
        return False

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    # init_models expects args with .resume_step and .checkpoint.
    args = SimpleNamespace(resume_step=0, checkpoint=None)
    print("  instantiating models (this downloads pretrained deps on first run)...")
    models, config = init_models(args, config)
    freeze_model_parameters(models, config)
    params_statistic(models)  # prints trainable / frozen counts
    verify_trainable_modules(models, config)  # hard-asserts the freeze is exact

    for m in models.values():
        m.to(device).train()
    models["generator"].semantic_encoder.eval()

    if batch is None:
        batch = load_one_batch(config, batch_size=batch_size)

    # Mirror the trainer's batch preparation (VCCodecTrainer.update_batch).
    def to_dev(x):
        return x.to(device) if hasattr(x, "to") else x

    model_batch = {k: to_dev(v) for k, v in batch.items()}
    model_batch["source_wav"] = model_batch["source_wav"].unsqueeze(1)
    model_batch["target_wav"] = model_batch["target_wav"].unsqueeze(1)
    model_batch["step"] = 0

    print("  running one forward pass...")
    with torch.no_grad():
        outputs = models["generator"](model_batch)
    print("  forward OK. output shapes:")
    for k in ("recons", "pred", "zq"):
        if k in outputs and hasattr(outputs[k], "shape"):
            print(f"    {k:12s} {tuple(outputs[k].shape)}")

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    print(f"\n  peak GPU memory: allocated {peak:.2f} GiB | reserved {reserved:.2f} GiB")
    print("\nPASS - stage 3 (forward)")
    return True


# --------------------------------------------------------------------------- #
# Stage 4: full-loop train smoke — N steps -> checkpoint saves -> the INFERENCE
# path loads that checkpoint -> converts one real clip -> audible output.
# This closes the save/load loop stages 1-3 don't cover: the most common failure
# of a fresh training program is discovering at step 5,000 that inference can't
# load what training saves.
# --------------------------------------------------------------------------- #
def train_roundtrip(config_path: str, steps: int = 20, log_dir: Optional[str] = None,
                    num_workers: int = 2):
    import json
    import subprocess
    import tempfile

    from omegaconf import OmegaConf

    print("=" * 72)
    print(f"STAGE 4 - train {steps} steps -> save -> reload via inference -> convert")
    print("=" * 72)

    log_dir = Path(log_dir) if log_dir else REPO_ROOT / "exp" / "smoke_train"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Patched copy of the config (NOT an overlay: load_config resolves base_config
    # one level deep only): tiny run, save/validate at half-way and at the end.
    cfg = OmegaConf.load(config_path)
    half = max(1, steps // 2)
    OmegaConf.update(cfg, "total_step", steps)
    OmegaConf.update(cfg, "save_interval", half)
    OmegaConf.update(cfg, "keep_interval", half)
    OmegaConf.update(cfg, "val_interval", half)
    OmegaConf.update(cfg, "log_interval", 1)
    OmegaConf.update(cfg, "dataloader.static.batch_size", 2)
    smoke_cfg = log_dir / "config_smoke.yaml"
    OmegaConf.save(cfg, smoke_cfg)

    ckpt = REPO_ROOT / "ckpts" / "xvc.pt"
    if not ckpt.is_file():
        print(f"  ERROR: released checkpoint not found at {ckpt}")
        return False

    cmd = [
        "torchrun", "--nnodes=1", "--nproc_per_node=1", "--master_port=10299",
        "-m", "bins.train",
        "--config", str(smoke_cfg),
        "--log_dir", str(log_dir),
        "--train_engine", "deepspeed",
        "--deepspeed_config", "configs/ds_stage2.json",
        "--resume_step", "0",
        "--checkpoint", str(ckpt),
        "--num_workers", str(num_workers),
        "--project", "smoke",
        "--date", "smoke",
    ]
    print("  launching:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"  ERROR: training subprocess exited {proc.returncode}")
        return False

    # The checkpoint the trainer saved:
    saved = sorted((log_dir / "ckpt").glob("*.pt"))
    saved = [p for p in saved if not p.is_symlink() and p.stem.isdigit()]
    if not saved:
        print(f"  ERROR: training saved no checkpoint under {log_dir}/ckpt "
              f"(save_interval={half}, total_step={steps})")
        return False
    last = saved[-1]
    print(f"  training saved: {[p.name for p in saved]}; reloading {last.name} "
          f"via the inference path...")

    # Reload EXACTLY like inference does, then convert one val clip.
    from bins.infer_utils import load_pair_as_tensors, load_xvc, run_offline, to_numpy_audio
    import soundfile as sf

    run_cfg = log_dir / "config.yaml"  # resolved config saved by the trainer
    cfg2, model, device = load_xvc(str(run_cfg), str(last), 0, False)

    # First val record = one real (source, target) pair.
    from utils.file import load_config as _lc
    val_entry = _lc(str(run_cfg))["datasets"]["val"]
    val_manifest = val_entry[0] if not isinstance(val_entry, str) else val_entry
    with open(REPO_ROOT / val_manifest, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    src_p = str(REPO_ROOT / rec["source_wav_path"])
    tgt_p = str(REPO_ROOT / rec["target_wav_path"])
    source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
        src_p, tgt_p, cfg2, device, 1280, False)
    recon = run_offline(model, source_wav, target_wav, target_wav_cond)
    out = to_numpy_audio(recon)
    out_path = log_dir / "smoke_conversion.wav"
    sf.write(str(out_path), out, int(cfg2["sample_rate"]))

    dur = len(out) / int(cfg2["sample_rate"])
    print(f"  conversion OK: {out_path} ({dur:.2f}s). LISTEN TO IT.")
    print("\nPASS - stage 4 (train roundtrip)")
    return True


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", choices=["manifests", "batch", "forward", "train", "all"], default="all",
        help="Which stage(s) to run (default: all). 'train' runs the full "
             "train->save->reload->convert roundtrip (GPU + ckpts/xvc.pt; NOT part "
             "of 'all' — run it explicitly before the first real launch).",
    )
    parser.add_argument(
        "--train-steps", type=int, default=20,
        help="Steps for the train roundtrip stage (default 20).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader workers for the train stage (default 2). Use 0 on "
             "containers with a small /dev/shm (workers pass batches through "
             "shared memory; 0 keeps loading in the main process).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Fine-tune YAML (e.g. configs/finetune_arabic.yaml). Required for batch/forward.",
    )
    parser.add_argument(
        "--manifests", nargs="*", default=None,
        help="Explicit JSONL paths for the manifests stage (bypasses --config; stdlib only).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override dataloader batch size for the smoke test (e.g. 2).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Check only the first N lines of each manifest (faster).",
    )
    args = parser.parse_args(argv)

    stages = ["manifests", "batch", "forward"] if args.stage == "all" else [args.stage]

    # Resolve manifest list for stage 1.
    config = None
    manifest_paths = args.manifests
    if manifest_paths is None and args.config is not None:
        config = _load_config(args.config)
        manifest_paths = _manifests_from_config(config)

    all_ok = True
    batch = None
    for stage in stages:
        if stage == "manifests":
            if not manifest_paths:
                print("[skip] manifests: provide --manifests or --config")
                all_ok = False
                continue
            all_ok &= validate_manifests(manifest_paths, limit=args.limit)
        elif stage == "batch":
            if config is None:
                if args.config is None:
                    print("[skip] batch: --config is required")
                    all_ok = False
                    continue
                config = _load_config(args.config)
            batch = load_one_batch(config, batch_size=args.batch_size)
        elif stage == "forward":
            if config is None:
                if args.config is None:
                    print("[skip] forward: --config is required")
                    all_ok = False
                    continue
                config = _load_config(args.config)
            all_ok &= bool(forward_pass(config, batch=batch, batch_size=args.batch_size))
        elif stage == "train":
            if args.config is None:
                print("[skip] train: --config is required")
                all_ok = False
                continue
            all_ok &= bool(train_roundtrip(args.config, steps=args.train_steps,
                                           num_workers=args.num_workers))
        print()

    print("=" * 72)
    print("SMOKE TEST: " + ("PASS" if all_ok else "FAIL"))
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
