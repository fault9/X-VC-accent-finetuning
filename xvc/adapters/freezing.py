"""Single source of truth for parameter freezing and trainable-set verification.

Moved verbatim (behavior-preserving) from ``utils/train_utils.py`` so the
logic is importable without deepspeed. ``utils.train_utils`` re-exports these
names, so ``bins/train.py`` and every historical import keep working.

Freeze precedence per model config (unchanged):

1. ``no_grad: true``            -> whole model frozen;
2. ``lora.enabled: true``       -> freeze all, re-enable only lora_A/lora_B
                                   (+ biases per ``lora.train_bias``);
3. ``trainable_modules: [...]`` -> freeze all, unfreeze the listed top-level
                                   submodules (full-weight fine-tuning);
4. legacy per-submodule ``no_grad`` flags.

``verify_trainable_modules`` is the startup hard gate: the live trainable set
must equal the requested whitelist exactly, or training aborts.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch.nn as nn

from xvc.adapters.lora import mark_only_lora_as_trainable

log = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def freeze_all_parameters(model: nn.Module) -> None:
    """Disable gradients on every parameter of `model`."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_lora_parameters(model: nn.Module) -> int:
    """Enable gradients ONLY on ``lora_A``/``lora_B`` tensors (base stays as-is
    for already-frozen models). Returns the number of re-enabled tensors."""
    n = 0
    for name, param in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            param.requires_grad = True
            n += 1
    return n


def freeze_model_parameters(models: Dict[str, nn.Module], config) -> None:
    # Disable gradient computations for parameters of specific models as defined in the configuration.
    for model_name in config.model:
        model_cfg = config.model[model_name]

        # (1) Whole-model freeze.
        if model_cfg.get("no_grad", False):
            freeze_all_parameters(models[model_name])
            continue

        # (1b) LoRA fine-tuning. Freeze the entire model, then re-enable ONLY the
        # injected adapter tensors (lora_A / lora_B, plus biases if `train_bias`
        # requests it). The base backbone stays frozen -- this is the parameter-
        # efficient counterpart to the whitelist unfreeze in (2). `trainable_modules`
        # must still list the LoRA-host submodules so verify_trainable_modules and
        # the warm-start gate see the expected set.
        lora_cfg = model_cfg.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", False):
            n_trainable = mark_only_lora_as_trainable(
                models[model_name], bias=lora_cfg.get("train_bias", "none")
            )
            if _is_rank_zero():
                log.info(
                    "[lora] '%s': froze base; %.3fM trainable adapter params",
                    model_name, n_trainable / 1e6,
                )
            continue

        # (2) Whitelist-based freezing (fine-tuning). If `trainable_modules` is set,
        # freeze the entire model and then re-enable gradients only for the listed
        # submodules; everything not named stays frozen. This is how the accent
        # fine-tuning configs keep only `acoustic_converter` and `prenet` trainable.
        trainable_modules = model_cfg.get("trainable_modules", None)
        if trainable_modules is not None:
            freeze_all_parameters(models[model_name])
            for sub_name in trainable_modules:
                if not hasattr(models[model_name], sub_name):
                    log.warning(
                        "trainable_modules: '%s' is not a submodule of '%s'; skipping",
                        sub_name, model_name,
                    )
                    continue
                sub_model = getattr(models[model_name], sub_name)
                if not isinstance(sub_model, nn.Module):
                    log.warning(
                        "trainable_modules: '%s.%s' is not an nn.Module; skipping",
                        model_name, sub_name,
                    )
                    continue
                for param in sub_model.parameters():
                    param.requires_grad = True
            continue

        # (3) Legacy per-submodule freezing via `no_grad` on each submodule config.
        for sub_model_name in model_cfg:
            sub_model_config = model_cfg[sub_model_name]
            if hasattr(sub_model_config, "get") and sub_model_config.get("no_grad", False):
                sub_model = getattr(models[model_name], sub_model_name)
                for param in sub_model.parameters():
                    param.requires_grad = False


def verify_trainable_modules(models: Dict[str, nn.Module], config) -> int:
    """Hard gate: confirm the freeze actually produced the trainable set we asked for.

    For every model that declares a `trainable_modules` whitelist (the accent
    fine-tune configs list `[acoustic_converter, prenet]`), assert that:

      * the set of top-level submodules that own >=1 `requires_grad` parameter is
        EXACTLY that whitelist -- no more (a stray unfrozen module) and no less (a
        misspelled entry that silently froze everything);
      * no whitelisted submodule ended up fully frozen (typo / wrong name);
      * every trainable parameter in the model lives under a whitelisted submodule.

    For every model marked `no_grad: True`, assert it has zero trainable params.

    Raises AssertionError with a readable diff on any mismatch, so a broken freeze
    fails at startup instead of training "successfully" from the wrong parameters.
    Returns the total trainable-parameter count (also logged).
    """
    total_trainable = 0
    for model_name in config.model:
        model = models[model_name]
        model_cfg = config.model[model_name]

        # Submodules that actually carry trainable params after the freeze.
        live = set()
        for sub_name, sub in model.named_children():
            if any(p.requires_grad for p in sub.parameters()):
                live.add(sub_name)
        model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_trainable += model_trainable

        if model_cfg.get("no_grad", False):
            assert model_trainable == 0, (
                f"[freeze] model '{model_name}' is no_grad but has "
                f"{model_trainable} trainable params (live submodules: {sorted(live)})"
            )
            continue

        whitelist = model_cfg.get("trainable_modules", None)
        if whitelist is None:
            # Not a whitelist-frozen model (e.g. an unused discriminator); nothing
            # to assert here -- its trainability is governed elsewhere.
            continue

        expected = set(whitelist)
        # 1) named whitelist entries must exist as submodules
        missing_named = [m for m in expected if not hasattr(model, m)]
        assert not missing_named, (
            f"[freeze] trainable_modules for '{model_name}' name non-existent "
            f"submodules {missing_named}; available: {[n for n, _ in model.named_children()]}"
        )
        # 2) live trainable submodules must equal the whitelist exactly
        extra = sorted(live - expected)      # unfrozen but not requested -> leak
        absent = sorted(expected - live)     # requested but frozen -> typo/empty
        assert not extra and not absent, (
            f"[freeze] model '{model_name}': trainable submodules {sorted(live)} "
            f"!= requested {sorted(expected)}"
            + (f"; UNEXPECTEDLY TRAINABLE: {extra}" if extra else "")
            + (f"; REQUESTED BUT FROZEN: {absent}" if absent else "")
        )
        # 3) no trainable parameter may live outside a whitelisted submodule
        #    (catches params attached directly to the model, not via a child)
        prefixes = tuple(f"{m}." for m in expected)
        stray = [
            n for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(prefixes)
        ]
        assert not stray, (
            f"[freeze] model '{model_name}' has trainable params outside "
            f"{sorted(expected)}: {stray[:8]}{' ...' if len(stray) > 8 else ''}"
        )
        if _is_rank_zero():
            log.info(
                "[freeze] '%s' verified: trainable submodules == %s (%.3fM params)",
                model_name, sorted(expected), model_trainable / 1e6,
            )

    if _is_rank_zero():
        log.info("[freeze] total trainable parameters: %.3fM", total_trainable / 1e6)
    return total_trainable
