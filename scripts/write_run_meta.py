#!/usr/bin/env python3
"""
Write run_meta.json into a training run directory — the reproducibility record.

Captures everything the per-run config.yaml does NOT carry:

    * git commit hash (+ dirty flag)
    * sha256 of the base checkpoint being warm-started from
    * sha256 of every train/val manifest referenced by the config
    * seed, the exact launcher CLI, timestamp
    * pip freeze (full environment snapshot)

Called by scripts/finetune.sh before torchrun; safe to run standalone:

    python scripts/write_run_meta.py --log_dir exp/finetune_arabic \
        --config configs/finetune_arabic.yaml --checkpoint ckpts/xvc.pt \
        --seed 1234 --cli "bash scripts/finetune.sh --accent arabic"

Standard library only (reads the YAML config via a minimal line scan if
omegaconf is unavailable, so it also runs on the plain local interpreter).

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _run(cmd: list) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
        ).stdout.strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def manifests_from_config(config_path: Path) -> list:
    """Train/val manifest paths from the config (omegaconf if present, else scan)."""
    try:
        from utils.file import load_config  # resolves base_config chains
        cfg = load_config(str(config_path))
        out = []
        for split in ("train", "val"):
            entry = cfg["datasets"][split]
            out.extend([str(e) for e in entry] if not isinstance(entry, str) else [entry])
        return out
    except Exception:
        # Fallback: naive scan for .jsonl mentions (keeps this script stdlib-only).
        out = []
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("- ").strip()
            if line.endswith(".jsonl"):
                out.append(line)
        return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="base checkpoint (warm start)")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--cli", default=None, help="the launcher command line, verbatim")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for m in manifests_from_config(Path(args.config)):
        p = Path(m) if os.path.isabs(m) else REPO_ROOT / m
        manifests[m] = sha256_file(p) if p.is_file() else "<missing>"

    meta = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "config": args.config,
        "base_checkpoint": args.checkpoint,
        "base_checkpoint_sha256": (
            sha256_file(Path(args.checkpoint)) if args.checkpoint and Path(args.checkpoint).is_file()
            else None
        ),
        "manifests_sha256": manifests,
        "seed": args.seed,
        "launcher_cli": args.cli,
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        "python": sys.version,
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }

    out_path = log_dir / "run_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"run meta written to {out_path}")
    if meta["git_dirty"]:
        print("[warn] working tree is DIRTY — the recorded commit does not fully "
              "describe the code that will train", file=sys.stderr)
    for m, digest in manifests.items():
        if digest == "<missing>":
            print(f"[warn] manifest not found: {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
