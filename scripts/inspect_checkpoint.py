#!/usr/bin/env python3
"""Inspect an X-VC checkpoint without building a model.

Prints (as JSON): layout (full-model / lora-only / bare state_dict), the
models inside, tensor/parameter counts, dtype histogram, LoRA tensor counts,
whether EMA weights are present, and whether the file is merged (servable by
the stock architecture with no adapter code).

Usage:
    python scripts/inspect_checkpoint.py exp/<run>/ckpt/000100.pt
    python scripts/inspect_checkpoint.py ckpts/xvc.pt merged.pt   # several at once
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xvc.training.checkpointing import describe_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.checkpoints:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 1
        print(describe_checkpoint(path).to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
