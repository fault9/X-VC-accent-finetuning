"""Shared helpers for unseen-source evaluation runners and accent gate CLIs.

Single source of truth for the ``speaker_arctic_prompt.wav`` filename
conventions, the fail-closed unseen-source / training-prompt-overlap checks,
and the calibrated accent-gate body.  Both the (superseded) joint persona
mapper runners and the persona codec runners import from here so their gate
semantics cannot silently diverge.  CPU-only stdlib; no torch.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

SPEAKER_RE = re.compile(r"([^_]+)_arctic_", re.IGNORECASE)
PROMPT_RE = re.compile(r"(arctic_[ab]\d{4})", re.IGNORECASE)


def speaker_from_name(path: Path) -> str:
    """Corpus speaker prefix from ``speaker_arctic_prompt.wav`` naming."""
    match = SPEAKER_RE.match(path.stem)
    return match.group(1).casefold() if match else path.stem.split("_", 1)[0].casefold()


def prompt_from_name(path: Path) -> Optional[str]:
    """The ``arctic_[ab]NNNN`` prompt id embedded in a wav name, if any."""
    match = PROMPT_RE.search(path.stem)
    return match.group(1).casefold() if match else None


def check_unseen_sources(
    source_paths: Sequence[Path],
    seen_speakers: Iterable[str],
    *,
    require_unseen: bool,
    min_unseen_speakers: int,
    seen_in: str = "training",
) -> tuple[set[str], set[str]]:
    """Fail closed when evaluation sources overlap training source speakers.

    Returns ``(eval_speakers, overlap)`` for audit metadata.
    """
    seen = {str(value).casefold() for value in seen_speakers}
    eval_speakers = {speaker_from_name(path) for path in source_paths}
    overlap = seen & eval_speakers
    if require_unseen and overlap:
        raise SystemExit(
            f"[error] evaluation source speakers were seen in {seen_in}: "
            f"{sorted(overlap)}"
        )
    if require_unseen and len(eval_speakers) < min_unseen_speakers:
        raise SystemExit(
            f"[error] only {len(eval_speakers)} unseen evaluation speaker(s); "
            f"require {min_unseen_speakers}: {sorted(eval_speakers)}"
        )
    return eval_speakers, overlap


def check_training_prompt_overlap(
    source_paths: Sequence[Path], training_manifest: Optional[str]
) -> list[str]:
    """Fail closed when evaluation prompts appear in the training manifest."""
    if not training_manifest:
        return []
    train_rows = [
        json.loads(line)
        for line in Path(training_manifest).read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_prompts = {str(row.get("prompt_id", "")).casefold() for row in train_rows}
    eval_prompts = {
        prompt for prompt in (prompt_from_name(path) for path in source_paths) if prompt
    }
    prompt_overlap = sorted(train_prompts & eval_prompts)
    if prompt_overlap:
        raise SystemExit(
            f"[error] evaluation overlaps training prompts: {prompt_overlap[:20]}"
        )
    return prompt_overlap


def run_accent_gate(
    *,
    summary: str,
    out: str,
    stock_set: str,
    candidate_set: str,
    max_mos_drop: float,
    max_wer_increase: float,
    max_sim_drop: float,
    min_indian_prob_gain: float,
    calibration: Optional[str],
    calibration_target_set: str,
    calibration_native_set: str,
    min_accent_gap_closed: float,
    interpretation: str,
) -> int:
    """Gate a candidate condition against stock on accent/voice/quality/WER.

    Reads a ``score_xvc_accent_stream_audit.py`` condition summary, applies
    the shared thresholds, writes the gate JSON, and returns the process exit
    code (0 pass / 1 fail).
    """
    with Path(summary).open(encoding="utf-8") as handle:
        rows = {row["set"]: row for row in csv.DictReader(handle)}
    required = {stock_set, candidate_set}
    if not required <= rows.keys():
        raise SystemExit(f"[error] missing summary rows: {sorted(required - rows.keys())}")
    stock = rows[stock_set]
    candidate = rows[candidate_set]

    def number(row: dict[str, Any], key: str) -> float:
        value = row.get(key, "")
        if value in {"", "None", None}:
            raise ValueError(f"missing numeric metric {key!r} for {row.get('set')}")
        return float(value)

    deltas = {
        "mos": number(candidate, "mos_mean") - number(stock, "mos_mean"),
        "wer": number(candidate, "wer_mean") - number(stock, "wer_mean"),
        "sim": number(candidate, "sim_mean") - number(stock, "sim_mean"),
        "indian_frac": number(candidate, "indian_frac") - number(stock, "indian_frac"),
        "indian_prob": (
            number(candidate, "indian_prob_mean")
            - number(stock, "indian_prob_mean")
        ),
    }
    failures = []
    if deltas["mos"] < -max_mos_drop:
        failures.append(f"MOS delta {deltas['mos']:.4f} < {-max_mos_drop:.4f}")
    if deltas["wer"] > max_wer_increase:
        failures.append(f"WER delta {deltas['wer']:.4f} > {max_wer_increase:.4f}")
    if deltas["sim"] < -max_sim_drop:
        failures.append(
            f"target-speaker similarity delta {deltas['sim']:.4f} < {-max_sim_drop:.4f}"
        )
    calibration_result = None
    if calibration:
        with Path(calibration).open(encoding="utf-8") as handle:
            calibration_rows = {row["set"]: row for row in csv.DictReader(handle)}
        required_sets = {calibration_target_set, calibration_native_set}
        missing_sets = required_sets - calibration_rows.keys()
        if missing_sets:
            raise SystemExit(
                f"[error] calibration is missing sets: {sorted(missing_sets)}"
            )
        target_probability = number(
            calibration_rows[calibration_target_set], "indian_prob_mean"
        )
        native_probability = number(
            calibration_rows[calibration_native_set], "indian_prob_mean"
        )
        ceiling_gap = target_probability - native_probability
        if ceiling_gap <= 0:
            raise SystemExit(
                "[error] genuine-target Indian posterior must exceed native calibration"
            )
        gap_closed = deltas["indian_prob"] / ceiling_gap
        calibration_result = {
            "path": calibration,
            "target_set": calibration_target_set,
            "native_set": calibration_native_set,
            "target_indian_probability": target_probability,
            "native_indian_probability": native_probability,
            "human_gap": ceiling_gap,
            "candidate_gap_closed": gap_closed,
        }
        if gap_closed < min_accent_gap_closed:
            failures.append(
                f"calibrated Indian-posterior gap closed {gap_closed:.4f} "
                f"< {min_accent_gap_closed:.4f}"
            )
    elif deltas["indian_prob"] < min_indian_prob_gain:
        failures.append(
            f"Indian posterior gain {deltas['indian_prob']:.4f} "
            f"< {min_indian_prob_gain:.4f}"
        )
    result = {
        "status": "pass" if not failures else "fail",
        "summary": summary,
        "stock_set": stock_set,
        "candidate_set": candidate_set,
        "stock": stock,
        "candidate": candidate,
        "deltas": {key: round(value, 6) for key, value in deltas.items()},
        "thresholds": {
            "max_mos_drop": max_mos_drop,
            "max_wer_increase": max_wer_increase,
            "max_target_speaker_similarity_drop": max_sim_drop,
            "min_indian_probability_gain": min_indian_prob_gain,
            "min_calibrated_accent_gap_closed": min_accent_gap_closed,
        },
        "accent_calibration": calibration_result,
        "failures": failures,
        "interpretation": interpretation,
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1
