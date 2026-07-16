"""StreamVoice-inspired target-persona codec-sequence generation for X-VC.

Token/sequence utilities (:mod:`xvc.persona_codec.tokens`) are numpy-only and
CPU-testable; checkpoint and renderer glue (:mod:`xvc.persona_codec.runtime`)
needs torch and, for rendering, a loaded X-VC model.
"""

from xvc.persona_codec.tokens import (
    EXPECTED_GROUP_SIZE,
    assert_prompt_disjoint,
    group_acoustic_codes,
    length_ratio_stats,
    levenshtein_alignment,
    prompt_ids,
    trim_to_group_ratio,
    validate_token_ids,
)

__all__ = [
    "EXPECTED_GROUP_SIZE",
    "assert_prompt_disjoint",
    "group_acoustic_codes",
    "length_ratio_stats",
    "levenshtein_alignment",
    "prompt_ids",
    "trim_to_group_ratio",
    "validate_token_ids",
]
