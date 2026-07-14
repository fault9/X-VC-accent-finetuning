"""Config loading and validation for X-VC experiments.

Two entry points:

* :func:`load_config` — like the legacy ``utils.file.load_config`` but resolves
  ``base_config`` chains **recursively** (the legacy loader stops after one
  level, which is why ``finetune.sh`` writes patched config copies instead of
  overlay-on-overlay). It also understands the Phase-4 compositional
  ``defaults:`` list. The legacy loader is untouched; old scripts keep their
  one-level behavior.

* :func:`validate_experiment_config` — explicit pre-construction checks with
  actionable messages. Used by the new ``scripts/train.py`` entry point;
  the legacy ``bins/train.py`` path stays permissive so no historical run is
  rejected retroactively.

Only depends on omegaconf.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


class ConfigValidationError(ValueError):
    """One or more config values are invalid; ``errors`` lists all of them."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        bullet = "\n  - ".join(self.errors)
        super().__init__(
            f"invalid experiment config ({len(self.errors)} problem(s)):\n  - {bullet}"
        )


def _resolve_relative(path_str: str, relative_to: Path) -> Path:
    """Resolve `path_str` the way the legacy loader effectively did: repo-root
    (CWD)-relative first, then relative to the referencing config file."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    candidate = relative_to.parent / p
    return candidate if candidate.exists() else p


def load_config(config_path: Path | str, _seen: Optional[set] = None) -> DictConfig:
    """Load a config, resolving ``base_config`` chains recursively.

    Merge order matches the legacy loader for a single level
    (base first, then the overlay), applied at every level; cycles raise.
    A ``defaults:`` list (Phase-4 compositional groups) is resolved by
    :func:`compose_config` before ``base_config`` handling.
    """
    path = Path(config_path)
    _seen = _seen or set()
    real = path.resolve()
    if real in _seen:
        chain = " -> ".join(str(p) for p in _seen)
        raise ConfigValidationError([f"base_config cycle: {chain} -> {real}"])
    _seen.add(real)

    config = OmegaConf.load(path)

    if config.get("defaults", None) is not None:
        return compose_config(path)

    base = config.get("base_config", None)
    if base is not None:
        base_path = _resolve_relative(str(base), path)
        if not base_path.exists():
            raise ConfigValidationError(
                [f"{path}: base_config '{base}' does not exist "
                 f"(looked at '{base_path}'); run from the repo root or use an "
                 f"absolute path"]
            )
        base_cfg = load_config(base_path, _seen)
        config = OmegaConf.merge(base_cfg, config)

    return config


def compose_config(experiment_path: Path | str) -> DictConfig:
    """Compose a Phase-4 experiment config from its ``defaults:`` list.

    The experiment file lists config-group entries relative to ``configs/``::

        defaults:
          - model/xvc_16k
          - adapter/lora_acoustic_r4
          - dataset/crosspair_hindi_latent_400_asi
          - training/finetune_lora_short
          - _self_

    Groups merge in list order; ``_self_`` marks where the experiment's own
    overrides apply (appended last if omitted — same as Hydra's convention).
    """
    path = Path(experiment_path)
    cfg = OmegaConf.load(path)
    defaults = cfg.get("defaults", None)
    if defaults is None:
        raise ConfigValidationError([f"{path}: compose_config needs a 'defaults' list"])

    # configs/ root: experiment files live in configs/experiment/.
    configs_root = path.resolve().parent.parent

    own = OmegaConf.masked_copy(
        cfg, [k for k in cfg.keys() if k != "defaults"]
    )
    merged = OmegaConf.create()
    self_applied = False
    for entry in defaults:
        if entry == "_self_":
            merged = OmegaConf.merge(merged, own)
            self_applied = True
            continue
        group_path = configs_root / f"{entry}.yaml"
        if not group_path.exists():
            raise ConfigValidationError(
                [f"{path}: defaults entry '{entry}' not found at {group_path}"]
            )
        merged = OmegaConf.merge(merged, load_config(group_path))
    if not self_applied:
        merged = OmegaConf.merge(merged, own)
    return merged


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check_lora_block(errors: List[str], gen_cfg) -> None:
    lora = gen_cfg.get("lora", None)
    if not lora or not lora.get("enabled", False):
        return
    r = lora.get("r", 8)
    if not isinstance(r, int) or r <= 0:
        errors.append(
            f"model.generator.lora.r = {r!r}: LoRA rank must be a positive "
            f"integer (the sweep used 1/2/4/8/16)"
        )
    alpha = lora.get("alpha", 2 * r if isinstance(r, int) else 16)
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        errors.append(
            f"model.generator.lora.alpha = {alpha!r}: must be a positive number "
            f"(scaling = alpha / r)"
        )
    dropout = lora.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        errors.append(
            f"model.generator.lora.dropout = {dropout!r}: must be in [0, 1)"
        )
    for key in ("target_modules", "include", "exclude"):
        val = lora.get(key, None)
        if val is not None and len(list(val)) == 0:
            errors.append(
                f"model.generator.lora.{key} = []: an empty list matches "
                f"nothing; omit the key or list at least one pattern"
            )


def _check_trainable_modules(errors: List[str], gen_cfg) -> None:
    trainable = gen_cfg.get("trainable_modules", None)
    if trainable is None:
        return
    # Recognized = submodules the generator config actually constructs.
    constructed = {
        k for k in gen_cfg.keys()
        if isinstance(gen_cfg.get(k), DictConfig) and gen_cfg[k].get("_target_", None)
    }
    unknown = [m for m in trainable if m not in constructed]
    if unknown:
        errors.append(
            f"model.generator.trainable_modules names unknown submodule(s) "
            f"{unknown}; constructed submodules are {sorted(constructed)}"
        )
    if len(list(trainable)) == 0:
        errors.append(
            "model.generator.trainable_modules is an empty list: nothing would "
            "train; remove the key (full training) or list the modules to tune"
        )


def _check_dataloader(errors: List[str], cfg) -> None:
    dl = cfg.get("dataloader", None)
    if dl is None:
        errors.append("missing 'dataloader' block")
        return
    recon = float(dl.get("reconstruction_ratio", dl.get("reconstruct_ratio", 0.0)) or 0.0)
    reversed_ratio = float(dl.get("reversed_ratio", dl.get("swap_ratio", 0.0)) or 0.0)
    for name, v in (("reconstruction_ratio", recon), ("reversed_ratio", reversed_ratio)):
        if not 0.0 <= v <= 1.0:
            errors.append(f"dataloader.{name} = {v}: must be in [0, 1]")
    if recon + reversed_ratio > 1.0:
        errors.append(
            f"dataloader.reconstruction_ratio + reversed_ratio = "
            f"{recon + reversed_ratio:g} > 1: the role-assignment draw needs "
            f"headroom for standard pairs"
        )
    if dl.get("latent_alignment", False) and reversed_ratio > 0:
        errors.append(
            "dataloader.latent_alignment=true with reversed_ratio > 0: the "
            "latent DTW map is native->L2 and cannot be reused for reversed "
            "(L2->native) pairs; set reversed_ratio: 0"
        )


def _check_datasets(errors: List[str], cfg, check_paths: bool) -> None:
    datasets = cfg.get("datasets", None)
    if datasets is None:
        errors.append("missing 'datasets' block (train/val manifest lists)")
        return
    for split in ("train", "val"):
        entries = datasets.get(split, None)
        entries = [entries] if isinstance(entries, str) else list(entries or [])
        if not entries:
            errors.append(f"datasets.{split} is empty: point it at >=1 JSONL manifest")
            continue
        for entry in entries:
            if "<path_to" in str(entry):
                errors.append(
                    f"datasets.{split} still contains the placeholder {entry!r}; "
                    f"set the manifest path"
                )
            elif check_paths and not Path(str(entry)).exists():
                errors.append(f"datasets.{split}: manifest not found: {entry}")


def _check_losses_and_rates(errors: List[str], cfg) -> None:
    gen_cfg = cfg.get("model", {}).get("generator", None)
    if gen_cfg is None:
        errors.append("missing 'model.generator' block")
        return
    loss_cfg = gen_cfg.get("loss_config", None)
    if loss_cfg is not None:
        for name, weight in (loss_cfg.get("loss_weights", None) or {}).items():
            if not isinstance(weight, (int, float)) or float(weight) < 0:
                errors.append(
                    f"model.generator.loss_config.loss_weights.{name} = "
                    f"{weight!r}: must be a non-negative number"
                )
        sr_loss = loss_cfg.get("sample_rate", None)
        sr_top = cfg.get("sample_rate", None)
        if sr_loss is not None and sr_top is not None and sr_loss != sr_top:
            errors.append(
                f"sample_rate mismatch: top-level {sr_top} vs "
                f"loss_config.sample_rate {sr_loss} — mel losses would be "
                f"computed on the wrong scale"
            )
    mel = gen_cfg.get("mel_extractor", None)
    if mel is not None and cfg.get("sample_rate", None) is not None:
        if mel.get("sample_rate", cfg.sample_rate) != cfg.sample_rate:
            errors.append(
                f"sample_rate mismatch: top-level {cfg.sample_rate} vs "
                f"mel_extractor.sample_rate {mel.sample_rate}"
            )


def validate_experiment_config(cfg: DictConfig, check_paths: bool = False) -> None:
    """Validate a fully-resolved training config before model construction.

    `check_paths=True` additionally requires dataset manifests to exist on
    this machine (off by default: configs are routinely resolved on machines
    that don't hold the data).

    Raises :class:`ConfigValidationError` listing every problem at once.
    """
    errors: List[str] = []

    gen_cfg = cfg.get("model", {}).get("generator", None)
    if gen_cfg is None:
        raise ConfigValidationError(["missing 'model.generator' block"])

    _check_lora_block(errors, gen_cfg)
    _check_trainable_modules(errors, gen_cfg)
    _check_dataloader(errors, cfg)
    _check_datasets(errors, cfg, check_paths)
    _check_losses_and_rates(errors, cfg)

    if errors:
        raise ConfigValidationError(errors)
