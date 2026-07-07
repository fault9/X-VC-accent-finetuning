import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
import utils.log as log
from omegaconf import OmegaConf
from utils.file import create_symbolic_link, resolve_symbolic_link


def resume_checkpoint(models: Dict[str, nn.Module], model_dir: Path, resume_step: int, skip_models: list = None):
    if resume_step == -1:
        ckpt_path = resolve_symbolic_link(f"{model_dir}/last.pt")
    else:
        resume_step = str(resume_step).zfill(6)
        ckpt_path = f"{model_dir}/{resume_step}.pt"

    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Not find checkpoint for {resume_step} step: {ckpt_path}")

    log.info("Resume: loading from checkpoint {}".format(ckpt_path))

    state_dict = {}
    state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for k in models.keys():
        if k in skip_models: continue
        missing_keys, unexpected_keys = models[k].load_state_dict(
            state_dict[k], strict=False
        )
        for key in missing_keys:
            log.info("missing tensor in {}: {}".format(k, key))
        for key in unexpected_keys:
            log.info("unexpected tensor in {}: {}".format(k, key))

    optim_conf = {}
    cfg_path = ckpt_path.replace(".pt", ".yaml")
    if os.path.isfile(cfg_path):
        config = OmegaConf.load(cfg_path)

    return models, config

def resume_ema_checkpoint(ema_model: nn.Module, model_dir: Path, resume_step: int, key_name: str = "ema_generator"):
    if resume_step == -1:
        ckpt_path = resolve_symbolic_link(f"{model_dir}/last.pt")
    else:
        resume_step = str(resume_step).zfill(6)
        ckpt_path = f"{model_dir}/{resume_step}.pt"

    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Not find EMA checkpoint for {resume_step} step: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get(key_name, state_dict)
    log.info("Resume: loading EMA from checkpoint {}".format(ckpt_path))

    missing_keys, unexpected_keys = ema_model.load_state_dict(
        state_dict, strict=False
    )
    for key in missing_keys:
        log.info("missing tensor in ema model: {}".format(key))
    for key in unexpected_keys:
        log.info("unexpected tensor in ema model: {}".format(key))

    return ema_model


def _report_and_gate_load(
    model_name: str,
    missing_keys: list,
    unexpected_keys: list,
    require_modules=None,
    strict_unexpected: bool = True,
):
    """Log a LOUD summary of a strict=False load and fail on a bad load.

    A `strict=False` load silently tolerates a state_dict that matches almost
    nothing -- the classic "trains successfully from garbage" failure. This gate
    makes the outcome visible and turns two specific cases into hard errors:

      * `unexpected_keys` (weights in the checkpoint with no home in the model) ->
        an architecture / config mismatch; fail unless `strict_unexpected=False`.
      * `missing_keys` that fall inside a `require_modules` prefix (the modules we
        actually fine-tune -- prenet / acoustic_converter) -> those weights MUST
        come from the warm-start checkpoint, so a miss there means training from
        random init exactly where it matters. Fail.

    Missing keys OUTSIDE the required modules (e.g. a `semantic_encoder` that is
    loaded separately from its own pretrained release) are reported but tolerated.
    """
    n_missing, n_unexpected = len(missing_keys), len(unexpected_keys)
    log.info(
        "Checkpoint load [%s]: %d missing, %d unexpected key(s)",
        model_name, n_missing, n_unexpected,
    )

    required_missing = []
    if require_modules:
        prefixes = tuple(f"{m}." for m in require_modules)
        # LoRA adapter tensors (lora_A / lora_B) are freshly initialised by design
        # (B is zero -> identity at init) and are ABSENT from a stock warm-start
        # checkpoint. Exclude them so a legitimate LoRA warm-start is not misread as
        # "training a fine-tuned module from random init".
        required_missing = [
            k for k in missing_keys
            if k.startswith(prefixes) and ".lora_A" not in k and ".lora_B" not in k
        ]
        lora_missing = [k for k in missing_keys if ".lora_A" in k or ".lora_B" in k]
        if lora_missing:
            log.info(
                "  [%s] %d LoRA adapter tensor(s) freshly initialised "
                "(absent from warm-start checkpoint, as expected)",
                model_name, len(lora_missing),
            )

    # Report (loudly if anything looks off).
    show = lambda keys: keys[:10] + (["...(+%d more)" % (len(keys) - 10)] if len(keys) > 10 else [])
    if n_unexpected:
        log.warning("  [%s] unexpected tensors: %s", model_name, show(unexpected_keys))
    if required_missing:
        log.error("  [%s] MISSING trainable-module tensors: %s", model_name, show(required_missing))
    elif n_missing:
        log.warning(
            "  [%s] %d missing tensor(s) OUTSIDE fine-tuned modules "
            "(tolerated -- e.g. separately-loaded frozen backbone): %s",
            model_name, n_missing, show(missing_keys),
        )

    problems = []
    if strict_unexpected and n_unexpected:
        problems.append(f"{n_unexpected} unexpected key(s) (checkpoint/model architecture mismatch)")
    if required_missing:
        problems.append(
            f"{len(required_missing)} missing key(s) inside fine-tuned modules "
            f"{list(require_modules)} -- warm-start would train these from random init"
        )
    if problems:
        raise RuntimeError(
            f"Checkpoint load for model '{model_name}' failed verification: "
            + "; ".join(problems)
            + ". Refusing to train from a partially-loaded checkpoint."
        )


def load_checkpoint(
        models: Union[Dict[str, nn.Module], nn.Module],
        ckpt_path: Path,
        require_modules_by_model: dict = None,
        allow_missing_models=("discriminator",),
    ):
    """
    Load models' states from a checkpoint file, verifying the load did not silently
    skip the weights we depend on.

    Args:
        models: A single model or a dictionary of models to load states into.
        ckpt_path: Path of the checkpoint file from which to load states.
        require_modules_by_model: Optional {model_name: [submodule, ...]} of top-level
            submodules that MUST be fully present in the checkpoint (the fine-tuned
            modules). A missing key inside these is a hard error.
        allow_missing_models: Model keys that may be entirely absent from the
            checkpoint (the released X-VC `xvc.pt` ships no `discriminator`); such a
            model keeps its freshly-initialized weights. Any OTHER absent model is a
            hard error rather than a silent skip.
    """
    log.info("Checkpoint: loading from checkpoint %s" % ckpt_path)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    require_modules_by_model = require_modules_by_model or {}

    if isinstance(models, dict):
        # If models is a dictionary, load state dictionaries individually
        for k in models:
            if k not in state_dict:
                # A wholly-absent model is tolerated only if explicitly allowed
                # (e.g. the discriminator, unused when adversarial loss is off).
                if k in allow_missing_models:
                    log.warning(
                        "checkpoint has no state for model '%s'; keeping its "
                        "initialized weights (allowed)", k,
                    )
                    continue
                raise RuntimeError(
                    f"Checkpoint '{ckpt_path}' has no state for required model '{k}' "
                    f"(present: {sorted(state_dict.keys())}). Refusing to train it "
                    f"from random init."
                )
            missing_keys, unexpected_keys = models[k].load_state_dict(
                state_dict[k], strict=False
            )
            _report_and_gate_load(
                k, missing_keys, unexpected_keys,
                require_modules=require_modules_by_model.get(k),
            )
    else:
        # If a single model is provided, load the state dictionary directly
        missing_keys, unexpected_keys = models.load_state_dict(
                state_dict, strict=False
            )
        _report_and_gate_load(
            "model", missing_keys, unexpected_keys,
            require_modules=require_modules_by_model.get("model"),
        )


def save_checkpoints(
        models: Union[Dict[str, nn.Module], nn.Module], 
        ckpt_path: Path
    ):
    """
    Saves the state dictionaries of provided models to a checkpoint file.

    Args:
        models (Union[Dict[str, nn.Module], nn.Module]): A model or a dictionary of models to save.
        ckpt_path (Path): The path to save the checkpoint file.
    """

    log.info("Checkpoint: save to checkpoint %s" % ckpt_path)

    if isinstance(models, dict):
        state_dict = {k: model.state_dict() for k, model in models.items()}
    else:
        state_dict = models.state_dict() 

    torch.save(state_dict, ckpt_path)
    

def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        log.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        log.warning("for information, the existing modules in model are:")
        log.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    log.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        log.info("Checkpoint: loading from checkpoint %s for CPU" % enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location="cpu")
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        log.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs


def clean_stale_checkpoints(directory: Path, latest_checkpoint: str, interval: int = 100000):
    """
    Removes old model files that do not meet the retention interval criteria, except the latest model file.
    
    Args:
        directory (Path): The directory containing model files.
        latest_checkpoint (str): The filename of the latest model to keep.
        interval (int): The interval at which to keep model files (based on the numeric value in their filenames).
    """
    for filename in os.listdir(directory):
        if os.path.islink(f'{directory}/{filename}'): continue
        if '.pt' not in filename and '.yaml' not in filename: continue
        basensme = os.path.splitext(filename)[0]
        if basensme != latest_checkpoint and int(basensme) % interval != 0:
            os.remove(os.path.join(directory, filename))


def strip_prefix(state_dict, prefix: str):
    """
    Remove a common string prefix from all keys in a state_dict, if present.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

