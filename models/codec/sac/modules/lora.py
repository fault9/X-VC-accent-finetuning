"""Minimal, auditable LoRA for X-VC accent adapters.

Design goals (see CHANGES.md "LoRA accent adapters"):

  * **Checkpoint-compatible.** `LoRALinear` subclasses `nn.Linear`, so the base
    `weight`/`bias` keep their ORIGINAL parameter names. A stock X-VC checkpoint
    (no LoRA) loads straight into the base with no key remapping; only the new
    `lora_A`/`lora_B` tensors are added. A LoRA checkpoint loads into a LoRA-
    injected model by exact key match.
  * **No-op at init.** `lora_B` is zero-initialised, so a freshly-injected
    adapter is the identity map: warm-starting from stock weights reproduces
    stock behaviour until training moves `lora_B`.
  * **Merge for deployment.** `merge()` folds `scaling * B @ A` into `weight`,
    after which the forward pass is a plain `F.linear` (zero adapter overhead).
    `export_merged_state_dict()` returns a state_dict loadable by the STOCK
    architecture, so Hear-Me-Out / PersonaPlex serving needs no LoRA code.

The module is deliberately small and dependency-free (no `peft`): the training
stack already freezes a backbone and optimises only `requires_grad` params, so
LoRA needs nothing more than (1) swapping targeted `nn.Linear` layers and
(2) marking only the adapter tensors trainable.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """`nn.Linear` with an additive low-rank adapter: ``y = W x + b + s (B A) x``.

    ``W``/``b`` are the frozen base (original parameter names ``weight``/``bias``);
    ``A`` (``lora_A``, shape ``[r, in]``) and ``B`` (``lora_B``, shape ``[out, r]``)
    are the trainable adapter with scale ``s = alpha / r``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")
        self.r = int(r)
        self.lora_alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        # A ~ Kaiming (as in the LoRA paper), B = 0 => delta starts at exactly 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self._merged = False

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, r: int, alpha: float, dropout: float
    ) -> "LoRALinear":
        """Wrap an existing `nn.Linear`, REUSING its `weight`/`bias` tensors so the
        base parameters (and their state_dict keys) are preserved exactly."""
        obj = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        obj.weight = linear.weight
        if linear.bias is not None:
            obj.bias = linear.bias
        # Match the base tensor's device/dtype for the freshly-created adapter.
        obj.lora_A.data = obj.lora_A.data.to(linear.weight.device, linear.weight.dtype)
        obj.lora_B.data = obj.lora_B.data.to(linear.weight.device, linear.weight.dtype)
        return obj

    def _delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self._merged:
            return result
        lora = self.lora_dropout(x)
        lora = F.linear(lora, self.lora_A)  # (.., in) -> (.., r)
        lora = F.linear(lora, self.lora_B)  # (.., r)  -> (.., out)
        return result + self.scaling * lora

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into `weight` (idempotent); forward becomes plain linear."""
        if self._merged:
            return
        self.weight.add_(self._delta().to(self.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        self.weight.sub_(self._delta().to(self.weight.dtype))
        self._merged = False

    def extra_repr(self) -> str:  # shows up in print_model(), aids auditing
        return f"{super().extra_repr()}, r={self.r}, alpha={self.lora_alpha}, scaling={self.scaling:g}"


# --------------------------------------------------------------------------- #
# Targeting + injection
# --------------------------------------------------------------------------- #
def _match(name: str, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> bool:
    if exclude and any(p in name for p in exclude):
        return False
    if include:
        return any(p in name for p in include)
    return True


def _replace_submodule(root: nn.Module, qualified: str, new: nn.Module) -> None:
    """Swap the submodule at a dotted path. `setattr` routes `nn.Module` values
    into `_modules` by name, so numeric children (ModuleList/Sequential indices
    like `to_out.0`) are handled the same as named attributes."""
    parent_path, _, child = qualified.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child, new)


def find_lora_targets(
    root: nn.Module,
    top_modules: Iterable[str],
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """Qualified names of `nn.Linear` layers under `top_modules` matching the
    include/exclude substring filters (excludes any already-wrapped layer)."""
    prefixes = tuple(f"{m}." for m in top_modules)
    out: List[str] = []
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and name.startswith(prefixes) and _match(name, include, exclude):
            out.append(name)
    return out


def inject_lora(
    root: nn.Module,
    top_modules: Iterable[str],
    r: int,
    alpha: float,
    dropout: float,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """Replace targeted `nn.Linear` layers with `LoRALinear`. Returns adapted names."""
    targets = find_lora_targets(root, top_modules, include, exclude)
    for name in targets:
        base = root.get_submodule(name)
        _replace_submodule(root, name, LoRALinear.from_linear(base, r, alpha, dropout))
    return targets


def _as_list(v) -> Optional[List[str]]:
    if v is None:
        return None
    v = list(v)
    return v or None


def inject_lora_into_generator(generator: nn.Module, lora_cfg, trainable_modules=None) -> dict:
    """Inject LoRA into a generator (XVC) from a config block. Pure (no logging).

    `lora_cfg` keys: enabled, r, alpha, dropout, target_modules, include, exclude.
    `target_modules` defaults to `trainable_modules`, which defaults to
    `["acoustic_converter"]`. Returns an info dict describing what was adapted.
    """
    top = _as_list(lora_cfg.get("target_modules", None)) \
        or _as_list(trainable_modules) \
        or ["acoustic_converter"]
    r = int(lora_cfg.get("r", 8))
    alpha = float(lora_cfg.get("alpha", 2 * r))
    dropout = float(lora_cfg.get("dropout", 0.0))
    include = _as_list(lora_cfg.get("include", None))
    exclude = _as_list(lora_cfg.get("exclude", None))
    adapted = inject_lora(generator, top, r, alpha, dropout, include, exclude)
    if not adapted:
        # lora.enabled but the filters matched nothing -> a misconfiguration that
        # would otherwise "succeed" by training zero adapters. Fail loudly.
        raise RuntimeError(
            "LoRA is enabled but no nn.Linear layer matched the target filters "
            f"(target_modules={list(top)}, include={include}, exclude={exclude}). "
            "Check the names against the model -- e.g. acoustic_converter block "
            "linears are 'attn.to_q'/'attn.to_out'/'ff_x.ff'/'ff_c.ff'."
        )
    n_params = 0
    for m in generator.modules():
        if isinstance(m, LoRALinear):
            n_params += m.lora_A.numel() + m.lora_B.numel()
    return {
        "targets": list(top),
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "include": include,
        "exclude": exclude,
        "adapted": adapted,
        "n_adapters": len(adapted),
        "n_lora_params": n_params,
    }


# --------------------------------------------------------------------------- #
# Freeze + deploy helpers
# --------------------------------------------------------------------------- #
def is_lora_param_name(name: str) -> bool:
    return ".lora_A" in name or ".lora_B" in name


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> int:
    """Freeze everything except LoRA tensors. `bias`: 'none' | 'lora_only' | 'all'.
    Returns the trainable-parameter count."""
    for name, p in model.named_parameters():
        p.requires_grad = is_lora_param_name(name)
    if bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == "all":
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
    elif bias != "none":
        raise ValueError(f"unknown lora bias mode: {bias!r}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def merge_all_lora(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
            n += 1
    return n


@torch.no_grad()
def export_merged_state_dict(model: nn.Module) -> dict:
    """Merge every adapter and return a state_dict with the LoRA tensors stripped —
    loadable by the stock (non-LoRA) architecture for zero-overhead serving."""
    merge_all_lora(model)
    return {k: v for k, v in model.state_dict().items() if not is_lora_param_name(k)}
