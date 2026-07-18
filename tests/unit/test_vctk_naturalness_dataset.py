from pathlib import Path

from scripts.build_vctk_naturalness_dataset import choose_prompt_splits


def test_prompt_splits_are_balanced_disjoint_and_deterministic():
    files = {
        speaker: {f"{index:03d}": Path(f"{speaker}_{index:03d}.flac") for index in range(1, 31)}
        for speaker in ("p225", "p240", "p273", "p274")
    }
    first = choose_prompt_splits(
        files, {"001", "002"}, train_prompts=12, val_prompts=4,
        eval_prompts=3, seed=7,
    )
    second = choose_prompt_splits(
        files, {"001", "002"}, train_prompts=12, val_prompts=4,
        eval_prompts=3, seed=7,
    )
    assert first == second
    assert len(first["train"]) == 12
    assert len(first["val"]) == 4
    assert len(first["eval"]) == 3
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["eval"]))
    assert not (set(first["val"]) & set(first["eval"]))
    assert "001" not in set().union(*map(set, first.values()))


def test_prompt_splits_use_only_common_prompts():
    files = {
        "a": {"001": Path("a1"), "002": Path("a2"), "003": Path("a3")},
        "b": {"002": Path("b2"), "003": Path("b3"), "004": Path("b4")},
    }
    result = choose_prompt_splits(
        files, set(), train_prompts=1, val_prompts=1, eval_prompts=0, seed=3,
    )
    assert set(result["train"] + result["val"]) == {"002", "003"}
