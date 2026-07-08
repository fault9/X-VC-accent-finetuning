#!/usr/bin/env python3
"""Prepare same-prompt L1/L2 cross-pairs as Montreal Forced Aligner corpora.

This is the first half of the MFA-guided latent-alignment experiment.  It
selects/filters cross-pair rows using the same guards as ``align_crosspairs.py``
and writes two MFA corpora:

    <out>/mfa_corpus/source/<speaker>/<utt>.wav + .lab
    <out>/mfa_corpus/target/<speaker>/<utt>.wav + .lab

The paired rows are also written to:

    <out>/selected_manifests/{train,val}.jsonl

Run MFA on both corpora, then feed the TextGrid outputs to
``build_mfa_latent_crosspairs.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from align_crosspairs import (
    filter_input_rows,
    load_manifest,
    prompt_id,
    raw_target_path,
    source_speaker,
    stratified_limit,
    target_speaker,
)


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load L2/CMU-ARCTIC PROMPTS into ``arctic_a0001 -> text``."""
    prompts: dict[str, str] = {}
    pattern = re.compile(r"(arctic_[ab]\d{4})\s+(.+)")
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            # Common formats include:
            #   arctic_a0001 Author of ...
            #   ( arctic_a0001 "Author of ..." )
            line = line.strip("() ")
            match = pattern.search(line)
            if not match:
                continue
            pid, text = match.groups()
            text = text.strip().strip('"').strip()
            if text.endswith(")"):
                text = text[:-1].strip()
            text = re.sub(r"\s+", " ", text)
            prompts[pid] = text
    if not prompts:
        raise ValueError(f"no ARCTIC prompts parsed from {path}")
    return prompts


def resolve_source_path(row: dict, source_root: Path) -> Path:
    path = Path(row["source_wav_path"])
    return path if path.is_absolute() else source_root / path


def safe_copy_pair(audio: Path, lab_text: str, dst_stem: Path) -> None:
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audio, dst_stem.with_suffix(".wav"))
    dst_stem.with_suffix(".lab").write_text(lab_text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--resplit-val-prompts", type=int, default=None)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--l2-root", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=2.6)
    parser.add_argument("--min-global-stretch", type=float, default=0.85)
    parser.add_argument("--max-global-stretch", type=float, default=1.20)
    args = parser.parse_args()

    out = Path(args.out)
    source_root = Path(args.source_root)
    l2_root = Path(args.l2_root)
    prompts = load_prompts(args.prompts_file)

    splits = {
        "train": load_manifest(args.train_manifest),
        "val": load_manifest(args.val_manifest),
    }
    skipped_short = 0
    skipped_global = 0
    for name in splits:
        splits[name], stats = filter_input_rows(
            splits[name],
            l2_root=l2_root,
            source_root=source_root,
            minimum=args.min_duration,
            min_global_stretch=args.min_global_stretch,
            max_global_stretch=args.max_global_stretch,
        )
        skipped_short += stats["short"]
        skipped_global += stats["global_stretch"]

    train_prompts = {prompt_id(row) for row in splits["train"]}
    val_prompts = {prompt_id(row) for row in splits["val"]}
    overlap = train_prompts & val_prompts
    if overlap:
        if args.resplit_val_prompts is None:
            raise SystemExit(
                "[error] input manifests leak prompt IDs across train/val; pass "
                "--resplit-val-prompts N to repair legacy manifests"
            )
        combined = splits["train"] + splits["val"]
        prompt_ids = sorted({prompt_id(row) for row in combined})
        random.Random(1234).shuffle(prompt_ids)
        held_out = set(prompt_ids[: args.resplit_val_prompts])
        splits = {
            "train": [row for row in combined if prompt_id(row) not in held_out],
            "val": [row for row in combined if prompt_id(row) in held_out],
        }
        print(
            f"[repair] re-split legacy manifests by prompt: "
            f"train={len(splits['train'])}, val={len(splits['val'])}"
        )

    for index, name in enumerate(("train", "val")):
        limit = args.train_limit if name == "train" else args.val_limit
        if limit is not None:
            splits[name] = stratified_limit(splits[name], limit, 1234 + index)

    missing_prompts = sorted({prompt_id(row) for rows in splits.values() for row in rows} - set(prompts))
    if missing_prompts:
        raise ValueError(
            "PROMPTS file is missing required IDs: " + ", ".join(missing_prompts[:20])
        )

    source_corpus = out / "mfa_corpus" / "source"
    target_corpus = out / "mfa_corpus" / "target"
    manifest_dir = out / "selected_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows_out: dict[str, list[dict]] = {"train": [], "val": []}
    for split, rows in splits.items():
        for row in rows:
            pid = prompt_id(row)
            text = prompts[pid]
            src_path = resolve_source_path(row, source_root)
            tgt_path = raw_target_path(row, l2_root)
            mfa_source_id = row["source_utt"]
            mfa_target_id = row["target_utt"][:-3] if row["target_utt"].endswith("_ft") else row["target_utt"]

            safe_copy_pair(
                src_path,
                text,
                source_corpus / source_speaker(row) / mfa_source_id,
            )
            safe_copy_pair(
                tgt_path,
                text,
                target_corpus / target_speaker(row) / mfa_target_id,
            )
            enriched = dict(row)
            enriched["mfa_source_id"] = mfa_source_id
            enriched["mfa_target_id"] = mfa_target_id
            enriched["prompt_text"] = text
            rows_out[split].append(enriched)

    for split, rows in rows_out.items():
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "train_pairs": len(rows_out["train"]),
        "val_pairs": len(rows_out["val"]),
        "train_prompts": len({prompt_id(row) for row in rows_out["train"]}),
        "val_prompts": len({prompt_id(row) for row in rows_out["val"]}),
        "prompt_overlap": len(
            {prompt_id(row) for row in rows_out["train"]}
            & {prompt_id(row) for row in rows_out["val"]}
        ),
        "minimum_raw_duration_seconds": args.min_duration,
        "allowed_global_stretch": [args.min_global_stretch, args.max_global_stretch],
        "skipped_short_pairs": skipped_short,
        "skipped_global_stretch_pairs": skipped_global,
        "source_corpus": str(source_corpus).replace("\\", "/"),
        "target_corpus": str(target_corpus).replace("\\", "/"),
    }
    (out / "mfa_prepare_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"\nMFA corpora written under {out / 'mfa_corpus'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
