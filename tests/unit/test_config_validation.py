"""Tests for xvc.utils.config: recursive loading and explicit validation."""

import pytest
from omegaconf import OmegaConf

from tests.conftest import REPO_ROOT
from xvc.utils.config import (
    ConfigValidationError,
    load_config,
    validate_experiment_config,
)

CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def legacy(name: str):
    return load_config(CONFIG_DIR / name)


# ------------------------------------------------------------------ loading
def test_recursive_loader_matches_legacy_one_level():
    from utils.file import load_config as legacy_load
    name = "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml"
    new = OmegaConf.to_container(legacy(name), resolve=False)
    old = OmegaConf.to_container(legacy_load(CONFIG_DIR / name), resolve=False)
    assert new == old, "single-level overlays must resolve identically"


def test_recursive_loader_handles_overlay_of_overlay(tmp_path):
    child = CONFIG_DIR / "finetune_hindi.yaml"
    grandchild = tmp_path / "overlay.yaml"
    grandchild.write_text(
        f"base_config: {child}\nmodel:\n  generator:\n    optim_conf:\n      lr: 5e-5\n"
    )
    cfg = load_config(grandchild)
    # Unlike the legacy loader, the grandparent (configs/xvc.yaml) is present:
    assert cfg.sample_rate == 16000
    assert cfg.model.generator.optim_conf.lr == pytest.approx(5e-5)
    assert cfg.model.generator._target_ == "models.codec.sac.model.XVC"


def test_cycle_detection(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"base_config: {b}\n")
    b.write_text(f"base_config: {a}\n")
    with pytest.raises(ConfigValidationError, match="cycle"):
        load_config(a)


def test_missing_base_config_is_actionable(tmp_path):
    orphan = tmp_path / "orphan.yaml"
    orphan.write_text("base_config: configs/does_not_exist.yaml\n")
    with pytest.raises(ConfigValidationError, match="does_not_exist"):
        load_config(orphan)


# --------------------------------------------------------------- validation
def test_all_legacy_finetune_configs_validate():
    for path in sorted(CONFIG_DIR.glob("finetune_*.yaml")):
        validate_experiment_config(legacy(path.name))  # must not raise


def test_bad_lora_rank_rejected():
    cfg = legacy("finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml")
    cfg.model.generator.lora.r = 0
    with pytest.raises(ConfigValidationError, match="rank must be a positive"):
        validate_experiment_config(cfg)


def test_empty_include_list_rejected():
    cfg = legacy("finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml")
    cfg.model.generator.lora.include = []
    with pytest.raises(ConfigValidationError, match="matches nothing"):
        validate_experiment_config(cfg)


def test_unknown_trainable_module_rejected():
    cfg = legacy("finetune_hindi.yaml")
    cfg.model.generator.trainable_modules = ["acoustic_converter", "typo_module"]
    with pytest.raises(ConfigValidationError, match="typo_module"):
        validate_experiment_config(cfg)


def test_latent_alignment_with_reversed_ratio_rejected():
    cfg = legacy("finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml")
    cfg.dataloader.reversed_ratio = 0.2
    with pytest.raises(ConfigValidationError, match="latent"):
        validate_experiment_config(cfg)


def test_placeholder_dataset_rejected():
    cfg = load_config(CONFIG_DIR / "xvc.yaml")
    with pytest.raises(ConfigValidationError, match="placeholder"):
        validate_experiment_config(cfg)


def test_negative_loss_weight_rejected():
    cfg = legacy("finetune_hindi.yaml")
    cfg.model.generator.loss_config.loss_weights.mel_loss = -1.0
    with pytest.raises(ConfigValidationError, match="mel_loss"):
        validate_experiment_config(cfg)


def test_sample_rate_mismatch_rejected():
    cfg = legacy("finetune_hindi.yaml")
    cfg.model.generator.mel_extractor.sample_rate = 24000
    with pytest.raises(ConfigValidationError, match="sample_rate mismatch"):
        validate_experiment_config(cfg)


def test_errors_are_collected_not_first_only():
    cfg = legacy("finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml")
    cfg.model.generator.lora.r = -1
    cfg.dataloader.reconstruction_ratio = 2.0
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_experiment_config(cfg)
    assert len(excinfo.value.errors) >= 2
