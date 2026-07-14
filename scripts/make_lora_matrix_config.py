#!/usr/bin/env python3
"""Generate one explicit X-VC LoRA matrix config from a reviewed template.

This keeps the experiment runner compact without hiding the actual settings:
every generated YAML is saved under ``exp/persona_matrix/configs`` and recorded
by the normal run metadata.  Only rank, alpha, LR, batch size, step count, and
LoRA host set may change here; datasets and the training objective remain those
of the selected template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HOSTS = {
    # Harsha's proposed surface, interpreted literally: LoRA every Linear under
    # acoustic_converter.  This includes attention, both converter MLPs,
    # input/output projections, and speaker-conditioned AdaLN linears.  None
    # means "no include filter" to the repository's LoRA injector.
    "acoustic": {
        "modules": ["acoustic_converter"],
        "include": None,
    },
    # Known stack-distill control used by L10/L15: converter realization plus
    # prenet, including converter modulation and prenet pointwise projections.
    "acoustic_prenet": {
        "modules": ["acoustic_converter", "prenet"],
        "include": [
            "attn.", "ff_x.ff", "ff_c.ff", "attn_norm_x.", "norm_out.", "pwconv",
        ],
    },
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", choices=sorted(HOSTS), required=True)
    ap.add_argument("--rank", type=int, choices=(1, 2, 4), required=True)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch-size", type=int, choices=(4, 8), required=True)
    ap.add_argument("--total-step", type=int, required=True)
    args = ap.parse_args(argv)

    if args.lr <= 0 or args.alpha <= 0 or args.total_step <= 0:
        raise SystemExit("[error] alpha, lr, and total-step must be positive")

    from omegaconf import OmegaConf

    template = Path(args.template)
    if not template.is_file():
        raise SystemExit(f"[error] template missing: {template}")
    cfg = OmegaConf.load(template)
    host = HOSTS[args.host]

    OmegaConf.update(cfg, "dataloader.static.batch_size", args.batch_size)
    OmegaConf.update(cfg, "total_step", args.total_step)
    OmegaConf.update(cfg, "model.generator.trainable_modules", host["modules"])
    OmegaConf.update(cfg, "model.generator.lora.enabled", True)
    OmegaConf.update(cfg, "model.generator.lora.r", args.rank)
    OmegaConf.update(cfg, "model.generator.lora.alpha", args.alpha)
    OmegaConf.update(cfg, "model.generator.lora.target_modules", host["modules"])
    OmegaConf.update(cfg, "model.generator.lora.include", host["include"])
    OmegaConf.update(cfg, "model.generator.optim_conf.lr", args.lr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out)

    metadata = {
        "template": str(template),
        "generated_config": str(out),
        "host": args.host,
        "trainable_modules": host["modules"],
        "include": host["include"],
        "rank": args.rank,
        "alpha": args.alpha,
        "scaling_alpha_over_rank": args.alpha / args.rank,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_step": args.total_step,
        "note": (
            "Fixed alpha/r is the standard parameterization. Lower rank is a "
            "capacity/regularization hypothesis, not a guarantee against overfit."
        ),
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
