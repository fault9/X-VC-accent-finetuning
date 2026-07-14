"""Checkpoint inspection and LoRA-aware helpers.

The on-disk contract (unchanged, see REFACTOR_PLAN.md §7): a training
checkpoint is ``{model_name: state_dict}`` — ``generator``, optionally
``discriminator`` and ``ema_generator`` (EMA keys carry an ``ema_model.``
prefix). LoRA runs add ``*.lora_A``/``*.lora_B`` tensors next to the unchanged
base keys; merged exports strip them.

This module gives that contract one importable home (previously spread over
``utils/checkpoint.py``, ``bins/infer_utils.py``, ``scripts/merge_lora.py``,
``scripts/publish_checkpoint.py``). It is import-light: torch only — no
``utils.log`` (which drags in wandb/matplotlib) and no deepspeed.

Additive: :func:`save_lora_only_checkpoint` writes adapter-only checkpoints
(format_version 1) that are ~1000x smaller than full dumps; they load through
:func:`xvc.adapters.load_lora_state_dict` onto a warm-started, LoRA-injected
model. Full-model checkpoints are untouched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from xvc.adapters.lora import is_lora_param_name

#: Version of the LoRA-only checkpoint container written by this module.
#: Historical full-model checkpoints predate versioning and are treated as
#: format 0 (bare {model_name: state_dict}).
LORA_CHECKPOINT_FORMAT_VERSION = 1

MODEL_KEYS = ("generator", "discriminator", "ema_generator")


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def load_checkpoint_file(path: Path | str) -> Dict[str, Any]:
    """``torch.load`` with the settings every consumer in this repo uses."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_generator_state_dict(
    checkpoint: Dict[str, Any] | Path | str,
    prefer_ema: bool = False,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Return ``(source_key, state_dict)`` for the generator weights.

    Handles the two historical layouts: ``{"generator": sd}`` and EMA dumps
    (``ema_generator`` with ``ema_model.``-prefixed keys). A bare state_dict
    (e.g. a merged export saved without the wrapper) is passed through.
    """
    if not isinstance(checkpoint, dict):
        checkpoint = load_checkpoint_file(checkpoint)
    if prefer_ema and "ema_generator" in checkpoint:
        return "ema_generator", _strip_prefix(checkpoint["ema_generator"], "ema_model.")
    if "generator" in checkpoint:
        return "generator", checkpoint["generator"]
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return "<bare state_dict>", checkpoint
    raise KeyError(
        f"no 'generator' entry in checkpoint (models present: "
        f"{sorted(k for k in checkpoint if isinstance(checkpoint[k], dict))})"
    )


def extract_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Only the ``lora_A``/``lora_B`` tensors of a generator state_dict."""
    return {k: v for k, v in state_dict.items() if is_lora_param_name(k)}


def strip_lora_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop adapter tensors (NOT a merge — use ``merge_lora_weights`` first if
    the adapter's effect should be kept)."""
    return {k: v for k, v in state_dict.items() if not is_lora_param_name(k)}


def save_lora_only_checkpoint(
    generator_state_dict: Dict[str, torch.Tensor],
    path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an adapter-only checkpoint: ``{format_version, lora_only,
    generator: {lora tensors}, metadata}``.

    Raises ``ValueError`` if the state_dict holds no adapter tensors.
    Returns the container that was written (without the tensors' data moved).
    """
    lora_sd = extract_lora_state_dict(generator_state_dict)
    if not lora_sd:
        raise ValueError(
            "state_dict contains no lora_A/lora_B tensors; nothing to save "
            "(a merged or stock checkpoint has no separable adapter)"
        )
    container = {
        "format_version": LORA_CHECKPOINT_FORMAT_VERSION,
        "lora_only": True,
        "generator": lora_sd,
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(container, path)
    return container


@dataclass
class ModelEntryDescription:
    tensor_count: int
    parameter_count: int
    lora_tensor_count: int
    dtypes: Dict[str, int] = field(default_factory=dict)
    sample_keys: List[str] = field(default_factory=list)


@dataclass
class CheckpointDescription:
    path: str
    format: str                      # "full-model" | "lora-only" | "bare state_dict"
    format_version: int              # 0 for historical unversioned checkpoints
    models: Dict[str, ModelEntryDescription] = field(default_factory=dict)
    has_ema: bool = False
    is_merged: bool = True           # True when no lora_* keys anywhere
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _describe_state_dict(sd: Dict[str, torch.Tensor]) -> ModelEntryDescription:
    dtypes: Dict[str, int] = {}
    n_params = 0
    n_lora = 0
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
        n_params += v.numel()
        if is_lora_param_name(k):
            n_lora += 1
    return ModelEntryDescription(
        tensor_count=len(sd),
        parameter_count=n_params,
        lora_tensor_count=n_lora,
        dtypes=dtypes,
        sample_keys=sorted(sd.keys())[:5],
    )


def describe_checkpoint(path: Path | str) -> CheckpointDescription:
    """Inspect a checkpoint file without a model: layout, models, LoRA tensors."""
    ckpt = load_checkpoint_file(path)

    if isinstance(ckpt, dict) and ckpt.get("lora_only", False):
        desc = CheckpointDescription(
            path=str(path),
            format="lora-only",
            format_version=int(ckpt.get("format_version", 1)),
            metadata=dict(ckpt.get("metadata", {})),
        )
        desc.models["generator"] = _describe_state_dict(ckpt["generator"])
        desc.is_merged = False
        return desc

    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        desc = CheckpointDescription(
            path=str(path), format="bare state_dict", format_version=0
        )
        desc.models["<root>"] = _describe_state_dict(ckpt)
        desc.is_merged = desc.models["<root>"].lora_tensor_count == 0
        return desc

    desc = CheckpointDescription(path=str(path), format="full-model", format_version=0)
    for name, sd in ckpt.items():
        if isinstance(sd, dict):
            desc.models[name] = _describe_state_dict(sd)
    desc.has_ema = "ema_generator" in desc.models
    desc.is_merged = all(m.lora_tensor_count == 0 for m in desc.models.values())
    return desc
