#!/usr/bin/env python
"""Export a merged, stock-architecture checkpoint from a LoRA accent fine-tune.

Folds every LoRA adapter (``scaling * B @ A``) into its base ``weight`` and writes
a checkpoint with the ``lora_A`` / ``lora_B`` tensors stripped. The result is
loadable by the RELEASED X-VC architecture with zero adapter code and zero
inference overhead -- hand it straight to Hear-Me-Out / PersonaPlex serving.

Serve the merged file with any NON-LoRA config of the same architecture, e.g.
``configs/finetune_crosspair_hindi_latent_400.yaml`` (LoRA does not change dims).

Usage:
  python scripts/merge_lora.py \
    --config exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/config.yaml \
    --ckpt   exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/ckpt/000300.pt \
    --out    exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/merged_step300.pt
"""
import argparse
import hashlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch

from models.codec.sac.model import XVC
from models.codec.sac.modules.lora import LoRALinear, export_merged_state_dict


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="training config.yaml (has the lora block)")
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint (base + lora_A/lora_B)")
    ap.add_argument("--out", required=True, help="output merged checkpoint path")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="multiply the adapter delta before folding it in "
                         "(>1 deepens the learned accent shift at some texture "
                         "cost -- gate scaled variants by ear before serving)")
    args = ap.parse_args()

    device = torch.device("cpu")
    # load_from_checkpoint injects the adapter topology from the config, then loads
    # base + lora_A/lora_B by exact key match.
    model = XVC.load_from_checkpoint(Path(args.config), Path(args.ckpt), device)
    if args.lora_scale != 1.0:
        n_scaled = 0
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.scaling *= args.lora_scale
                n_scaled += 1
        print(f"[merge_lora] adapter delta scaled x{args.lora_scale:g} on {n_scaled} layers")

    n_adapters = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    if n_adapters == 0:
        raise SystemExit(
            "[merge_lora] no LoRALinear layers found -- is lora.enabled set in the config, "
            "and does the checkpoint contain adapters?"
        )

    merged = export_merged_state_dict(model)  # merges in place, strips lora_* keys
    assert not any(".lora_A" in k or ".lora_B" in k for k in merged), "lora keys leaked"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Match the pipeline's on-disk layout: {model_name: state_dict}. Only the
    # generator is needed for inference/serving.
    torch.save({"generator": merged}, out)

    print(f"[merge_lora] merged {n_adapters} adapter(s)")
    print(f"[merge_lora] wrote {out}  sha256={_sha256(str(out))}")
    print(
        "[merge_lora] serve with a NON-LoRA config of the same architecture, e.g.\n"
        "  python scripts/eval_checkpoints.py run "
        "--config configs/finetune_crosspair_hindi_latent_400.yaml "
        f"--include-base {out} ..."
    )


if __name__ == "__main__":
    main()
