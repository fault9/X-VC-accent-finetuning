"""Tests for the new centralized adapter API (xvc.adapters).

The engine behavior is pinned by tests/unit/test_lora.py; these cover the
clean API layer: injection reports, duplicate detection, freezing helpers,
trainable-parameter reporting, and LoRA-only state-dict loading.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from xvc.adapters import (
    DuplicateInjectionError,
    NoTargetsMatchedError,
    freeze_all_parameters,
    freeze_model_parameters,
    get_trainable_parameter_report,
    inject_lora,
    load_lora_state_dict,
    merge_lora_weights,
    unfreeze_lora_parameters,
    verify_trainable_modules,
)
from tests.unit.test_lora import TinyGenerator

torch.manual_seed(0)


def make_adapted(include=("attn.",)):
    model = TinyGenerator()
    report = inject_lora(
        model, rank=2, alpha=4.0, dropout=0.0,
        include_patterns=list(include), target_modules=["acoustic_converter"],
    )
    return model, report


def test_injection_report_lists_every_adapted_layer():
    model, report = make_adapted()
    assert report.adapted_layers == [
        "acoustic_converter.attn.to_q",
        "acoustic_converter.attn.to_k",
        "acoustic_converter.attn.to_out.0",
    ]
    assert report.num_adapted == 3
    assert report.adapter_parameter_count(model) > 0


def test_invalid_rank_rejected():
    with pytest.raises(ValueError, match="rank"):
        inject_lora(TinyGenerator(), rank=0, alpha=4.0)


def test_no_match_raises_typed_error():
    with pytest.raises(NoTargetsMatchedError):
        inject_lora(TinyGenerator(), rank=2, alpha=4.0,
                    include_patterns=["nope"])


def test_duplicate_injection_raises_typed_error():
    model, _ = make_adapted()
    with pytest.raises(DuplicateInjectionError):
        inject_lora(model, rank=2, alpha=4.0,
                    include_patterns=["attn."],
                    target_modules=["acoustic_converter"])


def test_freeze_and_unfreeze_lora():
    model, _ = make_adapted()
    freeze_all_parameters(model)
    assert not any(p.requires_grad for p in model.parameters())
    n = unfreeze_lora_parameters(model)
    assert n == 6  # 3 layers x (lora_A + lora_B)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert all(".lora_" in k for k in trainable) and len(trainable) == 6


def test_report_groups_by_module_and_lists_lora_layers():
    model, _ = make_adapted()
    freeze_all_parameters(model)
    unfreeze_lora_parameters(model)
    report = get_trainable_parameter_report(model)
    assert set(report.trainable_by_module) == {"acoustic_converter"}
    assert len(report.lora_wrapped_layers) == 3
    assert 0 < report.percent_trainable < 100
    assert '"lora_wrapped_layers"' in report.to_json()


def test_report_fails_when_nothing_trainable():
    model = TinyGenerator()
    freeze_all_parameters(model)
    with pytest.raises(RuntimeError, match="No trainable parameters"):
        get_trainable_parameter_report(model)


def test_config_driven_freeze_matches_whitelist_gate():
    """freeze_model_parameters + verify_trainable_modules on a mock config —
    the exact functions bins/train.py runs (moved out of train_utils)."""
    models = {"generator": TinyGenerator(), "discriminator": nn.Linear(4, 4)}
    config = OmegaConf.create(
        {
            "model": {
                "generator": {"trainable_modules": ["acoustic_converter"]},
                "discriminator": {"no_grad": True},
            }
        }
    )
    freeze_model_parameters(models, config)
    total = verify_trainable_modules(models, config)
    assert total == sum(
        p.numel() for p in models["generator"].acoustic_converter.parameters()
    )

    # A whitelist naming a non-existent submodule must abort at startup.
    bad = OmegaConf.create(
        {"model": {"generator": {"trainable_modules": ["not_a_module"]}}}
    )
    freeze_model_parameters({"generator": TinyGenerator()}, bad)
    with pytest.raises(AssertionError):
        verify_trainable_modules({"generator": TinyGenerator()}, bad)


def test_config_driven_lora_freeze():
    model, _ = make_adapted()
    models = {"generator": model}
    config = OmegaConf.create(
        {
            "model": {
                "generator": {
                    "trainable_modules": ["acoustic_converter"],
                    "lora": {"enabled": True},
                }
            }
        }
    )
    freeze_model_parameters(models, config)
    total = verify_trainable_modules(models, config)
    assert total == 6 * 2 * 16  # 6 tensors, r=2, dim=16


def test_load_lora_state_dict_roundtrip():
    torch.manual_seed(7)
    source, _ = make_adapted()
    for m in source.modules():
        if hasattr(m, "lora_B"):
            nn.init.normal_(m.lora_B, std=0.1)

    # historical layout: {"generator": full_state_dict}. The fresh model shares
    # the same warm-start base (same seed); only the adapters are loaded.
    ckpt = {"generator": source.state_dict()}
    torch.manual_seed(7)
    fresh, _ = make_adapted()
    loaded = load_lora_state_dict(fresh, ckpt)
    assert len(loaded) == 6
    x = torch.randn(2, 16)
    assert torch.allclose(source(x), fresh(x), atol=1e-6)


def test_load_lora_state_dict_errors():
    model, _ = make_adapted()
    with pytest.raises(ValueError, match="no lora_A/lora_B"):
        load_lora_state_dict(model, {"weight": torch.zeros(1)})
    plain = TinyGenerator()  # no adapters injected -> topology mismatch
    with pytest.raises(KeyError, match="no matching LoRALinear"):
        load_lora_state_dict(plain, model.state_dict())


def test_merge_lora_weights_alias():
    model, _ = make_adapted()
    x = torch.randn(2, 16)
    y = model(x)
    assert merge_lora_weights(model) == 3
    assert torch.allclose(y, model(x), atol=1e-5)
