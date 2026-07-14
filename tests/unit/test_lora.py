"""Characterization tests for the custom LoRA implementation.

These pin the behavioral contract of ``models/codec/sac/modules/lora.py``
BEFORE any refactoring, so later changes can prove they preserved it:

* a freshly injected adapter is a numerical no-op (lora_B zero-init);
* base ``weight``/``bias`` keep their original state_dict keys;
* merge/unmerge round-trips exactly;
* target matching is deterministic and fails loudly on zero matches;
* re-injection cannot double-wrap a layer;
* ``mark_only_lora_as_trainable`` freezes exactly the right set;
* ``export_merged_state_dict`` yields a stock-architecture state dict.

No GPU, no pretrained checkpoint, torch CPU only.
"""

import pytest
import torch
import torch.nn as nn

from models.codec.sac.modules.lora import (
    LoRALinear,
    export_merged_state_dict,
    find_lora_targets,
    inject_lora,
    inject_lora_into_generator,
    mark_only_lora_as_trainable,
    merge_all_lora,
)

torch.manual_seed(0)


class TinyBlock(nn.Module):
    """Mimics the acoustic_converter naming (attn.to_q / ff_x.ff / to_out.0)."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.attn = nn.ModuleDict(
            {
                "to_q": nn.Linear(dim, dim),
                "to_k": nn.Linear(dim, dim),
                "to_out": nn.Sequential(nn.Linear(dim, dim)),
            }
        )
        self.ff_x = nn.ModuleDict({"ff": nn.Linear(dim, dim)})
        self.proj_other = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.attn["to_q"](x) + self.attn["to_k"](x)
        x = self.attn["to_out"](x)
        return self.ff_x["ff"](x) + self.proj_other(x)


class TinyGenerator(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.acoustic_converter = TinyBlock(dim)
        self.prenet = nn.ModuleDict({"pwconv": nn.Linear(dim, dim)})
        self.semantic_adapter = nn.Linear(dim, dim)

    def forward(self, x):
        return self.acoustic_converter(x) + self.prenet["pwconv"](x)


def test_fresh_adapter_is_identity():
    base = nn.Linear(8, 4)
    wrapped = LoRALinear.from_linear(base, r=4, alpha=16.0, dropout=0.0)
    x = torch.randn(3, 8)
    assert torch.allclose(base(x), wrapped(x), atol=0.0), (
        "zero-initialised lora_B must make the adapter an exact no-op"
    )


def test_base_parameter_names_preserved():
    base = nn.Linear(8, 4)
    wrapped = LoRALinear.from_linear(base, r=2, alpha=4.0, dropout=0.0)
    keys = set(wrapped.state_dict().keys())
    assert keys == {"weight", "bias", "lora_A", "lora_B"}
    # The base tensors are REUSED, not copied — the checkpoint contract.
    assert wrapped.weight is base.weight
    assert wrapped.bias is base.bias


def test_merge_unmerge_roundtrip():
    wrapped = LoRALinear(8, 4, r=2, alpha=4.0)
    # Move the adapter off its no-op init so merge actually changes weight.
    nn.init.normal_(wrapped.lora_B, std=0.1)
    x = torch.randn(5, 8)
    y_unmerged = wrapped(x)
    w_before = wrapped.weight.detach().clone()

    wrapped.merge()
    assert torch.allclose(y_unmerged, wrapped(x), atol=1e-5)
    wrapped.merge()  # idempotent
    assert torch.allclose(y_unmerged, wrapped(x), atol=1e-5)

    wrapped.unmerge()
    assert torch.allclose(wrapped.weight, w_before, atol=1e-6)
    assert torch.allclose(y_unmerged, wrapped(x), atol=1e-5)


def test_target_matching_is_deterministic_and_scoped():
    model = TinyGenerator()
    targets = find_lora_targets(
        model, ["acoustic_converter"], include=["attn.", "ff_x.ff"]
    )
    assert targets == [
        "acoustic_converter.attn.to_q",
        "acoustic_converter.attn.to_k",
        "acoustic_converter.attn.to_out.0",
        "acoustic_converter.ff_x.ff",
    ]
    # exclude filter
    assert "acoustic_converter.attn.to_k" not in find_lora_targets(
        model, ["acoustic_converter"], include=["attn."], exclude=["to_k"]
    )


def test_injection_replaces_only_matched_layers():
    model = TinyGenerator()
    adapted = inject_lora(
        model, ["acoustic_converter"], r=2, alpha=4.0, dropout=0.0,
        include=["attn.", "ff_x.ff"],
    )
    assert len(adapted) == 4
    assert isinstance(model.acoustic_converter.attn["to_q"], LoRALinear)
    assert not isinstance(model.acoustic_converter.proj_other, LoRALinear)
    assert not isinstance(model.prenet["pwconv"], LoRALinear)


def test_injection_preserves_model_output():
    torch.manual_seed(1)
    model = TinyGenerator()
    x = torch.randn(2, 16)
    y_before = model(x)
    inject_lora(model, ["acoustic_converter", "prenet"], r=4, alpha=16.0, dropout=0.0)
    assert torch.allclose(y_before, model(x), atol=1e-6)


def test_no_match_fails_loudly():
    model = TinyGenerator()
    with pytest.raises(RuntimeError, match="no nn.Linear layer matched"):
        inject_lora_into_generator(
            model, {"enabled": True, "r": 4, "include": ["does_not_exist"]}
        )


def test_double_injection_is_rejected():
    model = TinyGenerator()
    inject_lora_into_generator(model, {"enabled": True, "r": 2, "include": ["attn."]})
    # A second injection with the same filters finds nothing un-wrapped:
    # the fail-loud path doubles as duplicate-wrap detection.
    with pytest.raises(RuntimeError, match="no nn.Linear layer matched"):
        inject_lora_into_generator(
            model, {"enabled": True, "r": 2, "include": ["attn."]}
        )


def test_mark_only_lora_as_trainable():
    model = TinyGenerator()
    inject_lora(model, ["acoustic_converter"], r=2, alpha=4.0, dropout=0.0,
                include=["attn."])
    n = mark_only_lora_as_trainable(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(".lora_A" in t or ".lora_B" in t for t in trainable)
    assert n == sum(
        p.numel() for name, p in model.named_parameters() if name in trainable
    )
    assert not model.semantic_adapter.weight.requires_grad


def test_export_merged_state_dict_is_stock_shaped():
    torch.manual_seed(2)
    model = TinyGenerator()
    stock_keys = set(model.state_dict().keys())
    x = torch.randn(2, 16)
    inject_lora(model, ["acoustic_converter"], r=2, alpha=4.0, dropout=0.0)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)
    y_adapted = model(x)

    merged = export_merged_state_dict(model)
    assert set(merged.keys()) == stock_keys, (
        "merged export must load into the stock architecture with no key diff"
    )
    fresh = TinyGenerator()
    fresh.load_state_dict(merged, strict=True)
    assert torch.allclose(y_adapted, fresh(x), atol=1e-5), (
        "merged stock model must reproduce the adapted model's output"
    )


def test_merge_all_lora_counts():
    model = TinyGenerator()
    inject_lora(model, ["prenet"], r=2, alpha=4.0, dropout=0.0)
    assert merge_all_lora(model) == 1
