"""Clean injection API around the custom LoRA engine (:mod:`xvc.adapters.lora`).

The engine (`LoRALinear`, substring matching, config-block injection) is the
original, checkpoint-compatible implementation. This module adds the
explicit-keyword API used by new code and tests, with:

* deterministic target matching (module traversal order, pinned by tests);
* an injection report listing every adapted layer;
* loud failure when nothing matches;
* explicit duplicate-wrap detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import torch.nn as nn

from xvc.adapters.lora import (
    LoRALinear,
    find_lora_targets,
    inject_lora as _engine_inject,
)


class DuplicateInjectionError(RuntimeError):
    """Raised when injection targets a layer that is already LoRA-wrapped."""


class NoTargetsMatchedError(RuntimeError):
    """Raised when the include/exclude filters match no ``nn.Linear`` layer."""


@dataclass
class InjectionReport:
    """What one ``inject_lora`` call adapted."""

    rank: int
    alpha: float
    dropout: float
    target_modules: List[str]
    include_patterns: Optional[List[str]]
    exclude_patterns: Optional[List[str]]
    adapted_layers: List[str] = field(default_factory=list)

    @property
    def num_adapted(self) -> int:
        return len(self.adapted_layers)

    def adapter_parameter_count(self, model: nn.Module) -> int:
        return sum(
            m.lora_A.numel() + m.lora_B.numel()
            for m in model.modules()
            if isinstance(m, LoRALinear)
        )


def _already_wrapped(
    model: nn.Module, top_modules: Sequence[str],
) -> List[str]:
    prefixes = tuple(f"{m}." for m in top_modules) if top_modules else ()
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        and (not prefixes or name.startswith(prefixes))
    ]


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    include_patterns: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    target_modules: Optional[Iterable[str]] = None,
) -> InjectionReport:
    """Replace matching ``nn.Linear`` layers under `module` with ``LoRALinear``.

    Args:
        module: root module (typically the XVC generator).
        rank / alpha / dropout: standard LoRA hyperparameters
            (scaling = alpha / rank; ``lora_B`` zero-init keeps the wrapped
            layer an exact no-op until training moves it).
        include_patterns / exclude_patterns: substring filters on qualified
            layer names (e.g. ``["attn.", "ff_x.ff"]``). Matching is plain
            substring containment, identical to the training configs.
        target_modules: top-level submodule names to scope the search
            (e.g. ``["acoustic_converter"]``). ``None`` searches everywhere.

    Raises:
        DuplicateInjectionError: a targeted layer is already wrapped.
        NoTargetsMatchedError: the filters matched nothing (a config that
            would otherwise "train" zero adapters).
        ValueError: rank <= 0.
    """
    if rank <= 0:
        raise ValueError(f"LoRA rank must be a positive integer, got {rank}")

    top = list(target_modules) if target_modules else [
        name for name, _ in module.named_children()
    ]
    include = list(include_patterns) if include_patterns else None
    exclude = list(exclude_patterns) if exclude_patterns else None

    wrapped = _already_wrapped(module, top)
    if wrapped:
        raise DuplicateInjectionError(
            f"{len(wrapped)} layer(s) under {top} are already LoRA-wrapped "
            f"(first: {wrapped[0]}). Injecting twice would stack adapters and "
            f"corrupt checkpoint keys; build a fresh model instead."
        )

    if not find_lora_targets(module, top, include, exclude):
        raise NoTargetsMatchedError(
            f"No nn.Linear layer matched target_modules={top}, "
            f"include={include}, exclude={exclude}. Check the patterns against "
            f"`model.named_modules()` -- acoustic_converter block linears are "
            f"'attn.to_q'/'attn.to_out'/'ff_x.ff'/'ff_c.ff'."
        )

    adapted = _engine_inject(
        module, top, r=rank, alpha=alpha, dropout=dropout,
        include=include, exclude=exclude,
    )
    return InjectionReport(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=top,
        include_patterns=include,
        exclude_patterns=exclude,
        adapted_layers=adapted,
    )
