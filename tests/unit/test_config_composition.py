"""Behavioral-equivalence tests for the compositional config groups.

Each configs/experiment/*.yaml declares its legacy counterpart in a comment;
this test resolves BOTH and requires the trees to be IDENTICAL (not merely
similar) — the composed configs may not change a single training-relevant
value. This is the Phase-4 gate for marking legacy configs as mapped.
"""

import pytest
from omegaconf import OmegaConf

from tests.conftest import REPO_ROOT
from utils.file import load_config as legacy_load
from xvc.utils.config import load_config as new_load

CONFIG_DIR = REPO_ROOT / "configs"

# experiment name -> legacy overlay it must reproduce exactly
EQUIVALENCE = {
    "lora_hindi_asi_r1":
        "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16.yaml",
    "lora_hindi_asi_r2":
        "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16.yaml",
    "lora_hindi_asi_r4":
        "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml",
    "lora_hindi_asi_r4_lr5e-5":
        "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml",
    "lora_hindi_asi_acoustic_prenet_r4_lr5e-5":
        "finetune_crosspair_hindi_latent_400_asi_lora_acoustic_prenet_r4_alpha16_lr5e-5.yaml",
    "distill_hindi_asi_r4":
        "finetune_distill_hindi_asi_lora_r4_alpha16.yaml",
    "option_a_hindi":
        "finetune_hindi.yaml",
}


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def _diff(old: dict, new: dict, prefix: str = "") -> list:
    problems = []
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in old:
            problems.append(f"only in composed: {path} = {new[key]!r}")
        elif key not in new:
            problems.append(f"only in legacy:   {path} = {old[key]!r}")
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            problems.extend(_diff(old[key], new[key], path))
        elif old[key] != new[key]:
            problems.append(f"differs: {path}: legacy={old[key]!r} composed={new[key]!r}")
    return problems


@pytest.mark.parametrize("experiment, legacy", sorted(EQUIVALENCE.items()))
def test_composed_config_equals_legacy(experiment, legacy):
    legacy_cfg = OmegaConf.to_container(
        legacy_load(CONFIG_DIR / legacy), resolve=False
    )
    composed_cfg = OmegaConf.to_container(
        new_load(CONFIG_DIR / "experiment" / f"{experiment}.yaml"), resolve=False
    )
    problems = _diff(legacy_cfg, composed_cfg)
    assert not problems, (
        f"{experiment} is not behaviorally identical to {legacy}:\n  "
        + "\n  ".join(problems)
    )


def test_every_experiment_config_is_equivalence_tested():
    experiments = {p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")}
    untested = experiments - set(EQUIVALENCE)
    assert not untested, (
        f"experiment configs without a legacy-equivalence entry: {sorted(untested)} "
        f"— add them to EQUIVALENCE (or mark them as intentionally new)"
    )
