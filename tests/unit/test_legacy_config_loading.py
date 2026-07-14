"""Characterization tests for legacy config loading.

Pins how ``utils.file.load_config`` resolves the 51 ``configs/finetune_*.yaml``
overlays TODAY, so the Phase-4 compositional configs can be checked for
behavioral equivalence against these resolved trees.

Documents (rather than fixes) the known limitation that ``base_config`` is
resolved only ONE level deep — scripts/finetune.sh works around it by writing
patched config copies instead of overlay-on-overlay.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.conftest import REPO_ROOT
from utils.file import load_config

CONFIG_DIR = REPO_ROOT / "configs"
LEGACY_CONFIGS = sorted(CONFIG_DIR.glob("finetune_*.yaml"))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    # base_config paths inside the yamls are repo-root relative.
    monkeypatch.chdir(REPO_ROOT)


@pytest.mark.parametrize("path", LEGACY_CONFIGS, ids=lambda p: p.stem)
def test_legacy_config_resolves(path: Path):
    cfg = load_config(path)
    # Every overlay must inherit the full model tree from configs/xvc.yaml.
    assert cfg.model.generator._target_ == "models.codec.sac.model.XVC"
    assert cfg.dataloader._target_ == "models.codec.sac.dataloader.VCSSLWAVDataset"
    assert cfg.sample_rate == 16000
    # And declare its own datasets (no overlay may train on the placeholder).
    train = list(cfg.datasets.train)
    assert train and "<path_to" not in str(train[0])


def test_lora_config_values_survive_merge():
    cfg = load_config(
        CONFIG_DIR / "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml"
    )
    lora = cfg.model.generator.lora
    assert lora.enabled is True
    assert lora.r == 4
    assert lora.alpha == 16
    assert list(lora.target_modules) == ["acoustic_converter"]
    assert list(cfg.model.generator.trainable_modules) == ["acoustic_converter"]
    # Overlay must not clobber base-model architecture fields.
    assert cfg.model.generator.acoustic_converter.depth == 6
    assert cfg.model.generator.optim_conf.lr == pytest.approx(1e-4)


def test_distill_config_targets_acoustic_and_prenet():
    cfg = load_config(CONFIG_DIR / "finetune_distill_hindi_asi_lora_r4_alpha16.yaml")
    assert list(cfg.model.generator.lora.target_modules) == [
        "acoustic_converter",
        "prenet",
    ]
    assert cfg.dataloader.latent_alignment is False


def test_base_config_merge_is_one_level_only(tmp_path):
    """Documented limitation: an overlay of an overlay drops the grandparent.

    finetune.sh relies on this by writing patched copies instead of stacking
    overlays. The new loader (xvc.utils.config) resolves recursively; the
    legacy loader must keep its current one-level behavior.
    """
    child = CONFIG_DIR / "finetune_hindi.yaml"
    grandchild = tmp_path / "overlay.yaml"
    grandchild.write_text(f"base_config: {child}\nseed_marker: 1\n")
    cfg = load_config(grandchild)
    assert cfg.seed_marker == 1
    assert cfg.get("base_config") is not None
    # The grandparent (configs/xvc.yaml) was NOT merged in:
    assert cfg.get("sample_rate") is None
