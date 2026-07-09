#!/usr/bin/env python
"""Preflight gate: refuse to run an eval whose source clips overlap training.

Why this exists (see CHANGES.md "Eval contamination"): the pipeline reserves
ARCTIC prompts b0002-b0012 as held-out eval content (`build_crosspairs.py`
subtracts EVAL_PROMPTS from every cross-pair dataset), and `data/eval_sources`
was originally pinned to exactly those clips. At some point the eval source
directory drifted to a-prompt clips whose text content IS in the training
pairs -- both as same-speaker sources and as target-side renditions -- which
silently turned the "held-out" eval into a partial training-fit check. This
gate makes that drift a hard preflight failure instead of a silent bias.

Checks each `<speaker>_arctic_<prompt>.wav` in --source-dir against the train
manifest:

  * SOURCE-SEEN : (speaker, prompt) appears as a training source utterance --
                  the model trained on this exact clip's content+voice.
  * TARGET-SEEN : the prompt appears on the training target side -- the model
                  trained to produce the L2 speaker's rendition of this text.

Any hit fails the check (exit 1) unless --warn-only is given. Pure stdlib.

Usage:
    python scripts/check_eval_overlap.py \
        --source-dir data/eval_sources \
        --train-manifest data/crosspair_hindi_latent_400/manifests/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROMPT = re.compile(r"arctic_([ab]\d{4})")


def prompt_of(name: str):
    m = _PROMPT.search(name)
    return m.group(1) if m else None


def speaker_of(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_arctic_")[0] if "_arctic_" in stem else Path(path).parent.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, help="eval source wav dir")
    ap.add_argument("--train-manifest", required=True, action="append",
                    help="train.jsonl of the dataset being trained on; repeatable")
    ap.add_argument("--warn-only", action="store_true",
                    help="report overlap but exit 0 (for auditing old runs)")
    args = ap.parse_args(argv)

    train_sources = set()   # (speaker, prompt)
    train_tgt_prompts = set()
    for mpath in args.train_manifest:
        for line in open(mpath, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sp = prompt_of(row["source_wav_path"])
            if sp:
                train_sources.add((speaker_of(row["source_wav_path"]), sp))
            tp = prompt_of(row["target_wav_path"])
            if tp:
                train_tgt_prompts.add(tp)

    sources = sorted(Path(args.source_dir).glob("*.wav"))
    if not sources:
        print(f"[error] no wavs in {args.source_dir}", file=sys.stderr)
        return 1

    hits = []
    unparsed = 0
    for wav in sources:
        p = prompt_of(wav.stem)
        if p is None:
            unparsed += 1
            continue
        spk = speaker_of(str(wav))
        kinds = []
        if (spk, p) in train_sources:
            kinds.append("SOURCE-SEEN")
        if p in train_tgt_prompts:
            kinds.append("TARGET-SEEN")
        if kinds:
            hits.append((wav.stem, "+".join(kinds)))

    n = len(sources)
    print(f"[eval-overlap] {n} eval source clip(s) vs "
          f"{len(train_sources)} train source utts / "
          f"{len(train_tgt_prompts)} train target prompts "
          f"({unparsed} unparsable stems ignored)")
    for stem, kind in hits:
        print(f"  CONTAMINATED {stem}: {kind}")

    if hits:
        msg = (f"{len(hits)}/{n} eval source clip(s) overlap the training data. "
               "Evaluating on them measures training fit, not generalization. "
               "Use the reserved held-out prompts (arctic_b0002-b0012 clips) or "
               "rebuild eval_sources from prompts absent from the train manifest.")
        if args.warn_only:
            print(f"[eval-overlap] WARNING: {msg}")
            return 0
        print(f"[eval-overlap] FAIL: {msg}", file=sys.stderr)
        return 1
    print("[eval-overlap] PASS: no eval source clip overlaps the train manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
