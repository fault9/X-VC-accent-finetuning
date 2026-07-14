"""LoRA adapters, freezing, and trainable-parameter reporting for X-VC.

The engine is the original custom implementation (:mod:`xvc.adapters.lora`,
formerly ``models/codec/sac/modules/lora.py`` — that path re-exports from
here). No external LoRA framework is used.
"""

from xvc.adapters.freezing import (  # noqa: F401
    freeze_all_parameters,
    freeze_model_parameters,
    unfreeze_lora_parameters,
    verify_trainable_modules,
)
from xvc.adapters.injection import (  # noqa: F401
    DuplicateInjectionError,
    InjectionReport,
    NoTargetsMatchedError,
    inject_lora,
)
from xvc.adapters.lora import (  # noqa: F401
    LoRALinear,
    export_merged_state_dict,
    inject_lora_into_generator,
    is_lora_param_name,
    load_lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora_weights,
)
from xvc.adapters.reporting import (  # noqa: F401
    TrainableParameterReport,
    get_trainable_parameter_report,
    params_statistic,
)

__all__ = [
    "DuplicateInjectionError",
    "InjectionReport",
    "LoRALinear",
    "NoTargetsMatchedError",
    "TrainableParameterReport",
    "export_merged_state_dict",
    "freeze_all_parameters",
    "freeze_model_parameters",
    "get_trainable_parameter_report",
    "inject_lora",
    "inject_lora_into_generator",
    "is_lora_param_name",
    "load_lora_state_dict",
    "mark_only_lora_as_trainable",
    "merge_lora_weights",
    "params_statistic",
    "unfreeze_lora_parameters",
    "verify_trainable_modules",
]
