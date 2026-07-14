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
            intervals.append(
                Interval(
                    float(match.group(1)),
                    float(match.group(2)),
                    _unquote(match.group(3)),
                )
            )
        if intervals:
            tiers[name_match.group(1)] = intervals
    if not tiers:
        raise ValueError(f"could not parse interval tiers: {path}")
    return tiers


def phone_tier(
    path: Path, requested: str = "phones"
) -> tuple[str, list[Interval], float]:
    """Return a real phone tier. Deliberately never falls back to words."""
    tiers = read_textgrid(path)
    candidates = []
    for name, intervals in tiers.items():
        lower = name.casefold()
        if requested.casefold() == lower or "phone" in lower or "segment" in lower:
            candidates.append((name, intervals))
    if not candidates:
        raise ValueError(
            f"no genuine phone tier in {path}; found {sorted(tiers)}. "
            "Word-tier MFA is not accepted for this experiment."
        )
    name, intervals = max(candidates, key=lambda item: len(item[1]))
    xmax = max(interval.end for interval in intervals)
    phones = [
        interval
        for interval in intervals
        if interval.end > interval.start and normalize_phone(interval.label)
    ]
    if not phones:
        raise ValueError(f"phone tier {name!r} contains no usable phones: {path}")
    return name, phones, xmax


def normalize_phone(label: str) -> str:
    """Normalize for matching while retaining the original label in metadata."""
    label = label.strip().casefold()
    if label in SILENCE:
        return ""
    return re.sub(r"[012]$", "", re.sub(r"[^a-z0-9]+", "", label))


def align_phone_sequences(source: list[Interval], target: list[Interval]):
    """Levenshtein-align phone sequences and return exact-label interval pairs."""
    source_labels = [normalize_phone(interval.label) for interval in source]
    target_labels = [normalize_phone(interval.label) for interval in target]
    n, m = len(source_labels), len(target_labels)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        distance[i][0], back[i][0] = i, "del"
    for j in range(1, m + 1):
        distance[0][j], back[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (
                    distance[i - 1][j - 1]
                    + (source_labels[i - 1] != target_labels[j - 1]),
                    "diag",
                ),
                (distance[i - 1][j] + 1, "del"),
                (distance[i][j - 1] + 1, "ins"),
            ]
            distance[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    pairs = []
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation == "diag":
            if source_labels[i - 1] == target_labels[j - 1] and source_labels[i - 1]:
                pairs.append(
                    (source[i - 1], target[j - 1], source_labels[i - 1])
                )
            i -= 1
            j -= 1
        elif operation == "del":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, len(pairs) / max(n, m, 1)


def index_textgrids(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.TextGrid")) + list(root.rglob("*.textgrid"))
    indexed = {path.stem: path for path in paths}
    if not indexed:
        raise ValueError(f"no TextGrid files under {root}")
    return indexed
