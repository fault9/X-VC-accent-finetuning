"""Tiny ``key=value`` CLI parsing shared by scripts/train.py and scripts/infer.py.

The new entry points use Hydra-style overrides (``experiment=name``,
``checkpoint=path``) instead of ``--flags`` so a command reads like the config
it produces. No dependency on argparse quirks; unknown keys are the caller's
responsibility to reject.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def parse_key_value_args(argv: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Split ``argv`` into ``{key: value}`` overrides and passthrough args.

    ``key=value`` tokens become overrides; everything else (e.g. legacy
    ``--flags``) is returned verbatim in order, so wrappers can forward them.
    A bare ``key=`` yields an empty string value.
    """
    overrides: Dict[str, str] = {}
    passthrough: List[str] = []
    for token in argv:
        if "=" in token and not token.startswith("-"):
            key, _, value = token.partition("=")
            overrides[key] = value
        else:
            passthrough.append(token)
    return overrides, passthrough


def as_bool(value: str, key: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key}={value!r}: expected a boolean (true/false)")
