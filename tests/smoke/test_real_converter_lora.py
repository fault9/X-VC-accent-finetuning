"""Smoke tests injecting LoRA into the REAL AcousticConverter (reduced dims).

The unit tests use a mock; these verify the actual module the experiments
adapt: the include patterns from the training configs ("attn.", "ff_x.ff",
"ff_c.ff", AdaLN's "attn_norm_x."/"norm_out.") match the real layer names, a
forward pass is unchanged at injection time, merging preserves the output,
and one optimizer step moves ONLY the adapter tensors.

CPU-only; reduced dimensions (dim=64, depth=2) — no pretrained weights.
"""

import copy

import pytest
import torch
import torch.nn as nn

from xvc.adapters import (
    freeze_all_parameters,
    get_trainable_parameter_report,
    inject_lora,
    merge_lora_weights,
    unfreeze_lora_parameters,
)

converter_module = pytest.importorskip(
    "models.codec.sac.modules.acoustic_converter",
    reason="needs x-transformers/einops (pip install -r requirements.txt)",
)

# The include sets used by the experiment configs (see configs/adapter/).
CONVERTER_INCLUDE = ["attn.", "ff_x.ff", "ff_c.ff"]
ADALN_INCLUDE = ["attn_norm_x.", "norm_out."]


class Generator(nn.Module):
    """Container so config-style targeting (`acoustic_converter`) applies."""

    def __init__(self):
        super().__init__()
        self.acoustic_converter = converter_module.AcousticConverter(
            in_channels_x=32, in_channels_c=8, condition_dim=12,
            dim=64, depth=2, heads=2, dim_head=16, ff_mult=2,
            position_agnostic=False, convert="after_prenet",
        )

    def forward(self, x, c, spk):
        return self.acoustic_converter(x, c, spk)


def synthetic_batch(batch: int = 2, time: int = 20):
    torch.manual_seed(0)
    return (
        torch.randn(batch, 32, time),   # x:   B x D_x x T (post-prenet stream)
        torch.randn(batch, 8, time),    # c:   B x D_c x T (frame condition / mel)
        torch.randn(batch, 12),         # spk: B x condition_dim (speaker embedding)
    )


def wake_adaln_gates(model: nn.Module) -> None:
    """Give the AdaLN-Zero modulation projections non-zero weights.

    The converter uses DiT-style AdaLN-Zero: at RANDOM init the per-block
    gates are exactly 0, so attention/FFN outputs (and any LoRA delta inside
    them) are multiplied away. The real experiments warm-start from the
    released checkpoint where these projections are trained non-zero; this
    helper mimics that so functional tests exercise the adapted paths.
    """
    for name, module in model.named_modules():
        if ("attn_norm_x" in name or "norm_out" in name) and isinstance(
            module, nn.Linear
        ):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.normal_(module.bias, std=0.02)


def test_config_include_patterns_match_real_layers():
    model = Generator()
    report = inject_lora(
        model, rank=4, alpha=16.0,
        include_patterns=CONVERTER_INCLUDE,
        target_modules=["acoustic_converter"],
    )
    # Every non-final block adapts 12 linears (8 attention incl. the _c
    # conditioning stream + 2x2 FFN); the FINAL block does not update the
    # conditioning stream (no to_out_c / ff_c), so it contributes 9.
    depth = 2
    assert report.num_adapted == 12 * (depth - 1) + 9
    assert all(
        any(p in name for p in CONVERTER_INCLUDE) for name in report.adapted_layers
    )
    # The reference-conditioning stream (to_k_c / to_v_c) IS adapted via "attn.",
    # the documented behavior the target-conditioned sweep relies on.
    assert any(".attn.to_k_c" in n for n in report.adapted_layers)
    assert any(".ff_x.ff" in n for n in report.adapted_layers)
    assert any(".ff_c.ff" in n for n in report.adapted_layers)
    # AdaLN projections are NOT in this set (they're the separate arm below).
    assert not any("attn_norm_x" in n for n in report.adapted_layers)


def test_adaln_include_set_targets_modulation_projections():
    model = Generator()
    report = inject_lora(
        model, rank=8, alpha=16.0,
        include_patterns=ADALN_INCLUDE,
        target_modules=["acoustic_converter"],
    )
    assert report.num_adapted > 0
    assert all("attn_norm_x" in n or "norm_out" in n for n in report.adapted_layers)


def test_injection_is_noop_and_merge_preserves_output():
    torch.manual_seed(1)
    model = Generator().eval()
    wake_adaln_gates(model)
    x, c, spk = synthetic_batch()
    with torch.no_grad():
        y_stock = model(x, c, spk)

    inject_lora(model, rank=4, alpha=16.0,
                include_patterns=CONVERTER_INCLUDE,
                target_modules=["acoustic_converter"])
    with torch.no_grad():
        y_injected = model(x, c, spk)
    assert torch.allclose(y_stock, y_injected, atol=1e-6), (
        "zero-init LoRA must not change the real converter's output"
    )

    # Give the adapters a real (nonzero) delta, then merge.
    for m in model.modules():
        if hasattr(m, "lora_B"):
            nn.init.normal_(m.lora_B, std=0.05)
    with torch.no_grad():
        y_adapted = model(x, c, spk)
    assert not torch.allclose(y_stock, y_adapted, atol=1e-4)
    merge_lora_weights(model)
    with torch.no_grad():
        y_merged = model(x, c, spk)
    assert torch.allclose(y_adapted, y_merged, atol=1e-5)


def test_one_training_step_moves_only_adapters():
    torch.manual_seed(2)
    model = Generator()
    wake_adaln_gates(model)
    inject_lora(model, rank=4, alpha=16.0,
                include_patterns=CONVERTER_INCLUDE,
                target_modules=["acoustic_converter"])
    freeze_all_parameters(model)
    unfreeze_lora_parameters(model)
    report = get_trainable_parameter_report(model)
    assert set(report.trainable_by_module) == {"acoustic_converter"}

    base_before = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if ".lora_" not in k
    }
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    x, c, spk = synthetic_batch()
    loss = nn.functional.mse_loss(model(x, c, spk), torch.randn(2, 32, 20))
    loss.backward()
    optimizer.step()

    for k, v in model.state_dict().items():
        if ".lora_" in k:
            continue
        assert torch.equal(v, base_before[k]), f"frozen base drifted: {k}"
    moved = [
        k for k, v in model.state_dict().items()
        if k.endswith("lora_B") and v.abs().sum() > 0
    ]
    assert moved, "no adapter moved after an optimizer step"
