#!/usr/bin/env python3
"""Measure raw-vs-warped intelligibility and naturalness for cross pairs.

The raw accented target is the baseline: an accent-caused ASR error is not a
warp failure unless the warped version makes it worse. Results are written as
JSONL and consumed by ``filter_crosspairs.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_checkpoints import MOSPredictor, Whisper, norm_text, word_error_rate


PROMPT_RE = re.compile(r'\(\s*(arctic_[ab]\d+)\s+"(.*)"\s*\)')


def load_prompts(path: Path) -> dict[str, str]:
    prompts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PROMPT_RE.search(line)
        if match:
            prompts[match.group(1)] = match.group(2)
    if not prompts:
        raise ValueError(f"no ARCTIC prompts parsed from {path}")
    return prompts


def load_rows(root: Path) -> list[dict]:
    rows = []
    for split in ("train", "val"):
        path = root / "manifests" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["split"] = split
                rows.append(row)
    return rows


def prompt_id(row: dict) -> str:
    stem = Path(row["source_wav_path"]).stem
    return "arctic_" + stem.split("_arctic_", 1)[1]


def edit_count(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        previous, d[0] = d[0], i
        for j, hyp_word in enumerate(hyp, 1):
            current = min(
                d[j] + 1,
                d[j - 1] + 1,
                previous + (ref_word != hyp_word),
            )
            previous, d[j] = d[j], current
    return d[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prompts", required=True,
                        help="L2-ARCTIC PROMPTS file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--wer-model", default="small")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--skip-mos", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import soundfile as sf
    import torch

    root = Path(args.data_root)
    output = Path(args.out)
    prompts = load_prompts(Path(args.prompts))
    rows = load_rows(root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    whisper = Whisper(args.wer_model, device)
    mos = None if args.skip_mos else MOSPredictor(device)

    completed = set()
    if args.resume and output.is_file():
        completed = {
            json.loads(line)["source_utt"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    raw_cache: dict[str, dict] = {}

    with output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["source_utt"] in completed:
                continue
            prompt = prompt_id(row)
            reference_words = norm_text(prompts[prompt])
            raw_path = row.get("raw_target_wav_path")
            if not raw_path:
                raise ValueError(f"{row['source_utt']}: missing raw_target_wav_path")

            if raw_path not in raw_cache:
                raw_hypothesis = whisper.transcribe(raw_path)
                raw_words = norm_text(raw_hypothesis)
                raw_audio, raw_sr = sf.read(raw_path, dtype="float32")
                raw_cache[raw_path] = {
                    "transcript": raw_hypothesis.strip(),
                    "wer": word_error_rate(reference_words, raw_words),
                    "errors": edit_count(reference_words, raw_words),
                    "utmos": mos.score(raw_audio, raw_sr) if mos else None,
                }
            raw = raw_cache[raw_path]

            warped_path = row["target_wav_path"]
            warped_hypothesis = whisper.transcribe(warped_path)
            warped_words = norm_text(warped_hypothesis)
            warped_audio, warped_sr = sf.read(warped_path, dtype="float32")
            warped_wer = word_error_rate(reference_words, warped_words)
            warped_errors = edit_count(reference_words, warped_words)
            warped_utmos = mos.score(warped_audio, warped_sr) if mos else None

            result = {
                "split": row["split"],
                "source_utt": row["source_utt"],
                "target_utt": row["target_utt"],
                "prompt_id": prompt,
                "reference_text": prompts[prompt],
                "raw_transcript": raw["transcript"],
                "warped_transcript": warped_hypothesis.strip(),
                "raw_wer": round(float(raw["wer"]), 6),
                "warped_wer": round(float(warped_wer), 6),
                "wer_delta": round(float(warped_wer - raw["wer"]), 6),
                "raw_errors": int(raw["errors"]),
                "warped_errors": int(warped_errors),
                "added_errors": int(warped_errors - raw["errors"]),
                "raw_utmos": (
                    round(float(raw["utmos"]), 6)
                    if raw["utmos"] is not None else None
                ),
                "warped_utmos": (
                    round(float(warped_utmos), 6)
                    if warped_utmos is not None else None
                ),
                "utmos_drop": (
                    round(float(raw["utmos"] - warped_utmos), 6)
                    if raw["utmos"] is not None and warped_utmos is not None
                    else None
                ),
            }
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(rows)}] {row['source_utt']} "
                f"WER {raw['wer']:.3f}->{warped_wer:.3f} "
                f"MOS {raw['utmos']}->{warped_utmos}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
