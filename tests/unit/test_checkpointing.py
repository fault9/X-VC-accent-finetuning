"""Tests for xvc.training.checkpointing: inspection and LoRA-only round trips."""

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from tests.conftest import REPO_ROOT
from tests.unit.test_lora import TinyGenerator
from xvc.adapters import inject_lora, load_lora_state_dict
from xvc.training.checkpointing import (
    describe_checkpoint,
    extract_lora_state_dict,
    load_generator_state_dict,
    save_lora_only_checkpoint,
    strip_lora_keys,
)


def make_lora_generator(seed: int = 3):
    torch.manual_seed(seed)
    model = TinyGenerator()
    inject_lora(model, rank=2, alpha=4.0, include_patterns=["attn."],
                target_modules=["acoustic_converter"])
    for m in model.modules():
        if hasattr(m, "lora_B"):
            nn.init.normal_(m.lora_B, std=0.1)
    return model


def test_load_generator_state_dict_layouts():
    model = make_lora_generator()
    sd = model.state_dict()

    key, out = load_generator_state_dict({"generator": sd})
    assert key == "generator" and out.keys() == sd.keys()

    ema = {"ema_generator": {f"ema_model.{k}": v for k, v in sd.items()},
           "generator": sd}
    key, out = load_generator_state_dict(ema, prefer_ema=True)
    assert key == "ema_generator" and out.keys() == sd.keys()

    key, out = load_generator_state_dict(sd)  # bare state_dict passthrough
    assert key == "<bare state_dict>"

    with pytest.raises(KeyError, match="generator"):
        load_generator_state_dict({"discriminator": sd})


def test_lora_only_checkpoint_roundtrip(tmp_path):
    source = make_lora_generator(seed=11)
    path = tmp_path / "adapter.pt"
    container = save_lora_only_checkpoint(
        source.state_dict(), path, metadata={"experiment": "unit-test", "r": 2}
    )
    assert container["format_version"] == 1
    assert len(container["generator"]) == 6

    # Load onto a fresh model with the same warm-start base.
    fresh = make_lora_generator(seed=11)
    for m in fresh.modules():  # perturb adapters so loading must fix them
        if hasattr(m, "lora_B"):
            nn.init.zeros_(m.lora_B)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    load_lora_state_dict(fresh, ckpt)
    x = torch.randn(2, 16)
    assert torch.allclose(source(x), fresh(x), atol=1e-6)


def test_save_lora_only_rejects_stock_state_dict(tmp_path):
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="no lora_A/lora_B"):
        save_lora_only_checkpoint(TinyGenerator().state_dict(), tmp_path / "x.pt")


def test_extract_and_strip_partition_the_state_dict():
    sd = make_lora_generator().state_dict()
    lora = extract_lora_state_dict(sd)
    base = strip_lora_keys(sd)
    assert set(lora) | set(base) == set(sd)
    assert not set(lora) & set(base)
    assert len(lora) == 6


def test_describe_checkpoint_full_model(tmp_path):
    model = make_lora_generator()
    path = tmp_path / "full.pt"
    torch.save({"generator": model.state_dict(),
                "ema_generator": {f"ema_model.{k}": v
                                  for k, v in model.state_dict().items()}}, path)
    desc = describe_checkpoint(path)
    assert desc.format == "full-model"
    assert desc.has_ema
    assert not desc.is_merged
    assert desc.models["generator"].lora_tensor_count == 6
    assert desc.models["generator"].parameter_count > 0


def test_describe_checkpoint_lora_only_and_cli(tmp_path):
    model = make_lora_generator()
    path = tmp_path / "adapter.pt"
    save_lora_only_checkpoint(model.state_dict(), path, metadata={"r": 2})
    desc = describe_checkpoint(path)
    assert desc.format == "lora-only"
    assert desc.metadata == {"r": 2}

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "inspect_checkpoint.py"),
         str(path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert '"format": "lora-only"' in result.stdout
