"""CPU tests for the persona codec token/sequence utilities."""

import numpy as np
import pytest

from xvc.persona_codec.tokens import (
    assert_prompt_disjoint,
    group_acoustic_codes,
    length_ratio_stats,
    levenshtein_alignment,
    trim_to_group_ratio,
    validate_token_ids,
)


class TestValidateTokenIds:
    def test_accepts_valid_range(self):
        validate_token_ids(np.array([0, 5, 15], dtype=np.int32), 16, "x")

    def test_rejects_out_of_vocab(self):
        with pytest.raises(ValueError, match="outside vocabulary"):
            validate_token_ids(np.array([0, 16], dtype=np.int32), 16, "x")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="outside vocabulary"):
            validate_token_ids(np.array([-1, 3], dtype=np.int32), 16, "x")

    def test_rejects_empty_and_float(self):
        with pytest.raises(ValueError, match="empty"):
            validate_token_ids(np.array([], dtype=np.int32), 16, "x")
        with pytest.raises(ValueError, match="integer"):
            validate_token_ids(np.array([1.0, 2.0]), 16, "x")


class TestGrouping:
    def test_exact_multiple_passes_through(self):
        sem, ac, dropped = trim_to_group_ratio(30, 120, 4)
        assert (sem, ac, dropped) == (30, 120, 0)

    def test_one_frame_edge_is_trimmed(self):
        # Encoder edge behaviour: acoustic one frame short of the last group.
        sem, ac, dropped = trim_to_group_ratio(30, 119, 4)
        assert (sem, ac) == (29, 116)
        assert dropped == 3

    def test_extra_acoustic_frames_dropped(self):
        sem, ac, dropped = trim_to_group_ratio(30, 121, 4)
        assert (sem, ac, dropped) == (30, 120, 1)

    def test_large_disagreement_fails(self):
        with pytest.raises(ValueError, match="disagree"):
            trim_to_group_ratio(30, 100, 4)

    def test_full_group_drop_fails(self):
        # One whole extra group (a full semantic step of unaccounted audio)
        # must fail closed, not be silently truncated from the shard.
        with pytest.raises(ValueError, match="full group"):
            trim_to_group_ratio(30, 124, 4)
        with pytest.raises(ValueError, match="full group"):
            trim_to_group_ratio(100, 407, 4)

    def test_group_reshape(self):
        grouped = group_acoustic_codes(np.arange(12, dtype=np.int64), 4)
        assert grouped.shape == (3, 4)
        assert grouped[1].tolist() == [4, 5, 6, 7]

    def test_group_reshape_rejects_ragged(self):
        with pytest.raises(ValueError, match="do not group"):
            group_acoustic_codes(np.arange(13, dtype=np.int64), 4)


class TestLevenshtein:
    def test_identical(self):
        result = levenshtein_alignment([1, 2, 3], [1, 2, 3])
        assert result["distance"] == 0
        assert result["matches"] == 3
        assert result["matched_pairs"] == [(0, 0), (1, 1), (2, 2)]

    def test_substitution_insertion_deletion_counts(self):
        # source 1 2 3 4 -> target 1 9 3 4 5: one substitution, one insertion.
        result = levenshtein_alignment([1, 2, 3, 4], [1, 9, 3, 4, 5])
        assert result["distance"] == 2
        assert result["substitutions"] == 1
        assert result["insertions"] == 1
        assert result["deletions"] == 0

    def test_pure_deletion(self):
        result = levenshtein_alignment([1, 2, 3], [1, 3])
        assert result["distance"] == 1
        assert result["deletions"] == 1

    def test_normalized_bounds(self):
        result = levenshtein_alignment([1, 2], [3, 4, 5, 6])
        assert 0.0 <= result["normalized"] <= 1.0


class TestPromptDisjoint:
    def test_disjoint_ok(self):
        assert_prompt_disjoint(
            [{"prompt_id": "arctic_a0001"}], [{"prompt_id": "arctic_a0002"}]
        )

    def test_overlap_fails(self):
        with pytest.raises(ValueError, match="prompt overlap"):
            assert_prompt_disjoint(
                [{"prompt_id": "arctic_a0001"}],
                [{"prompt_id": "ARCTIC_A0001"}],  # case-insensitive
            )

    def test_missing_prompt_fails(self):
        with pytest.raises(ValueError, match="without prompt_id"):
            assert_prompt_disjoint([{"pair_id": "x"}], [{"prompt_id": "y"}])


def test_length_ratio_stats_keys():
    stats = length_ratio_stats([1.0, 1.2, 0.9])
    assert stats["n"] == 3
    assert stats["min"] <= stats["p50"] <= stats["max"]
