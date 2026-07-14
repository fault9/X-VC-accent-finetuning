"""Trainable-parameter reporting.

``get_trainable_parameter_report`` builds a structured report (JSON-friendly)
of what is trainable after freezing/LoRA injection; ``params_statistic`` is
the historical per-submodule log line, moved from ``utils/train_utils.py``
(re-exported there) so it is importable without deepspeed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import torch.nn as nn

from xvc.adapters.lora import LoRALinear

log = logging.getLogger(__name__)


@dataclass
class TrainableParameterReport:
    """Structured summary of a model's trainable set (one nn.Module)."""

    total_parameters: int
    trainable_parameters: int
    percent_trainable: float
    # qualified parameter names that require grad, grouped by top-level submodule
    trainable_by_module: Dict[str, List[str]] = field(default_factory=dict)
    # qualified names of every LoRA-wrapped linear layer
    lora_wrapped_layers: List[str] = field(default_factory=list)

    def to_json(self, path: Path | str | None = None) -> str:
        payload = json.dumps(asdict(self), indent=2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload)
        return payload


def get_trainable_parameter_report(model: nn.Module) -> TrainableParameterReport:
    """Inspect `model` AFTER freezing/injection and report the trainable set.

    Raises ``RuntimeError`` if nothing is trainable — a configuration that
    expects training but froze everything must fail before the optimizer is
    built, not train a no-op.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_module: Dict[str, List[str]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".", 1)[0]
        by_module.setdefault(top, []).append(name)

    lora_layers = [
        name for name, m in model.named_modules() if isinstance(m, LoRALinear)
    ]

    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters: the freeze/LoRA configuration disabled "
            "gradients everywhere. Check `trainable_modules` / `lora.enabled` "
            "in the model config."
        )

    return TrainableParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        percent_trainable=100.0 * trainable / max(total, 1),
        trainable_by_module=by_module,
        lora_wrapped_layers=lora_layers,
    )


def params_statistic(models: Dict[str, nn.Module]) -> None:
    # Report total / trainable / frozen parameter counts, with a per-submodule
    # breakdown so the effect of freezing (which modules are actually trainable)
    # is visible at startup.
    total_num_params = 0
    total_num_trainable_params = 0
    lines = []

    for model_name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        total_num_params += total
        total_num_trainable_params += trainable

        lines.append(
            "{}: total {:.2f}M | trainable {:.2f}M | frozen {:.2f}M".format(
                model_name, total / 1e6, trainable / 1e6, frozen / 1e6
            )
        )
        # One level of submodules (e.g. acoustic_converter, prenet, ...).
        for sub_name, sub_module in model.named_children():
            sub_total = sum(p.numel() for p in sub_module.parameters())
            if sub_total == 0:
                continue
            sub_trainable = sum(p.numel() for p in sub_module.parameters() if p.requires_grad)
            if sub_trainable == sub_total:
                state = "trainable"
            elif sub_trainable == 0:
                state = "frozen"
            else:
                state = "partial"
            lines.append(
                "    - {:<20s} {:8.3f}M  [{}]".format(sub_name, sub_total / 1e6, state)
            )

    total_num_frozen_params = total_num_params - total_num_trainable_params
    pct = 100.0 * total_num_trainable_params / max(total_num_params, 1)
    lines.append(
        "TOTAL: {:.2f}M | trainable {:.2f}M ({:.1f}%) | frozen {:.2f}M".format(
            total_num_params / 1e6,
            total_num_trainable_params / 1e6,
            pct,
            total_num_frozen_params / 1e6,
        )
    )
    num_param_info = "\n".join(lines)

    if int(os.environ.get("RANK", 0)) == 0:
        log.info("Model parameters statistic:\n{}".format(num_param_info))
