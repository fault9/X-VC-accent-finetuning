#!/usr/bin/env python
"""Dependency-free helpers for genuine phone-tier AccentBridge supervision."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SILENCE = {"", "sil", "sp", "spn", "<eps>", "<unk>", "silence", "noise"}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"').strip()


def read_textgrid(path: Path) -> dict[str, list[Interval]]:
    """Parse MFA/Praat long TextGrid interval tiers without extra packages."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[Interval]] = {}
    for block in re.split(r"\n\s*item \[\d+\]:", text):
        if "IntervalTier" not in block:
            continue
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        intervals = []
        for match in re.finditer(
            r"xmin\s*=\s*([0-9.eE+-]+)\s*"
            r"xmax\s*=\s*([0-9.eE+-]+)\s*"
            r"text\s*=\s*(\"(?:[^\"]|\"\")*\"|[^\n\r]+)",
            block,
            flags=re.S,
        ):
            intervals.append(Interval(float(match.group(1)), float(match.group(2)),
                                      _unquote(match.group(3))))
        if intervals:
            tiers[name_match.group(1)] = intervals
    if not tiers:
        raise ValueError(f"could not parse interval tiers: {path}")
    return tiers


def phone_tier(path: Path, requested: str = "phones") -> tuple[str, list[Interval], float]:
    """Return a real phone tier. Deliberately never falls back to words."""
    tiers = read_textgrid(path)
    candidates = []
    for name, intervals in tiers.items():
        lname = name.casefold()
        if requested.casefold() == lname or "phone" in lname or "segment" in lname:
            candidates.append((name, intervals))
    if not candidates:
        raise ValueError(
            f"no genuine phone tier in {path}; found {sorted(tiers)}. "
            "Word-tier MFA is not accepted for this experiment."
        )
    name, intervals = max(candidates, key=lambda x: len(x[1]))
    xmax = max(x.end for x in intervals)
    phones = [x for x in intervals if x.end > x.start and normalize_phone(x.label)]
    if not phones:
        raise ValueError(f"phone tier {name!r} contains no usable phones: {path}")
    return name, phones, xmax


def normalize_phone(label: str) -> str:
    """Normalize for cross-speaker matching while retaining stress in metadata."""
    label = label.strip().casefold()
    if label in SILENCE:
        return ""
    # MFA dictionaries sometimes use ARPABET stress digits inconsistently.
    return re.sub(r"[012]$", "", re.sub(r"[^a-z0-9]+", "", label))


def align_phone_sequences(source: list[Interval], target: list[Interval]):
    """Levenshtein-align phone sequences and return exact-label interval pairs."""
    a = [normalize_phone(x.label) for x in source]
    b = [normalize_phone(x.label) for x in target]
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "del"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]), "diag"),
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])
    pairs = []
    i, j = n, m
    while i or j:
        op = back[i][j]
        if op == "diag":
            if a[i - 1] == b[j - 1] and a[i - 1]:
                pairs.append((source[i - 1], target[j - 1], a[i - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    match_rate = len(pairs) / max(n, m, 1)
    return pairs, match_rate


def index_textgrids(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.TextGrid")) + list(root.rglob("*.textgrid"))
    out = {p.stem: p for p in paths}
    if not out:
        raise ValueError(f"no TextGrid files under {root}")
    return out
