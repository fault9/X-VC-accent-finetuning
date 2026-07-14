"""Deprecated import path — the LoRA implementation lives in ``xvc.adapters.lora``.

This shim keeps every historical import working unchanged
(``from models.codec.sac.modules.lora import LoRALinear`` etc.), including
``isinstance`` checks across call sites: there is exactly one ``LoRALinear``
class object, defined in :mod:`xvc.adapters.lora`.

New code should import from :mod:`xvc.adapters` instead.
"""

from xvc.adapters.lora import (  # noqa: F401
    LoRALinear,
    export_merged_state_dict,
    find_lora_targets,
    inject_lora,
    inject_lora_into_generator,
    is_lora_param_name,
    mark_only_lora_as_trainable,
    merge_all_lora,
)

__all__ = [
    "LoRALinear",
    "export_merged_state_dict",
    "find_lora_targets",
    "inject_lora",
    "inject_lora_into_generator",
    "is_lora_param_name",
    "mark_only_lora_as_trainable",
    "merge_all_lora",
]
