#!/usr/bin/env python
"""Retroactively split existing per-clip eval results by train contamination.

Repairs the interpretation of every eval table already collected (see
CHANGES.md "Eval contamination"): the eval sources that were actually used are
a-prompt clips, some of whose text content sits inside the training pairs. This
script re-reads the per-clip CSVs that `eval_checkpoints.py run` wrote, flags
each row against the train manifest, and re-aggregates per (run, step,
contamination class) -- zero GPU, no re-synthesis.

Classes per eval row (from the SOURCE clip's speaker+prompt):
  * clean        : prompt absent from the train manifest on both axes
  * contaminated : SOURCE-SEEN (same speaker+prompt trained as source) and/or
                   TARGET-SEEN (prompt trained on the target side)

The decisive readout is the accent canary split: if indian flips concentrate in
contaminated rows, the observed "accent signal" was partly memorized renditions
and generalization is weaker than the aggregate tables suggested; if clean rows
flip at a similar rate, the aggregate conclusions stand.

Usage (container, repo root):
    python scripts/split_eval_by_contamination.py \
        --train-manifest data/crosspair_hindi_latent_400/manifests/train.jsonl \
        --per-clip "exp/finetune_crosspair_hindi_latent_400_lora_*/eval_compare/*.csv" \
        --out exp/eval_contamination_split

Pure stdlib. Writes flags.csv (every row + flags), summary.csv (aggregates),
and prints the per-step clean-vs-contaminated comparison per run.
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROMPT = re.compile(r"arctic_([ab]\d{4})")


def prompt_of(name: str):
    m = _PROMPT.search(name)
    return m.group(1) if m else None


def speaker_of(stem: str) -> str:
    return stem.split("_arctic_")[0] if "_arctic_" in stem else stem


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-manifest", required=True, action="append",
                    help="train.jsonl the evaluated checkpoints were trained on; "
                         "repeatable (union)")
    ap.add_argument("--per-clip", required=True, action="append",
                    help="glob(s) of per-clip eval CSVs written by eval_checkpoints.py")
    ap.add_argument("--out", default="exp/eval_contamination_split")
    args = ap.parse_args(argv)

    train_sources = set()
    train_tgt_prompts = set()
    for mpath in args.train_manifest:
        for line in open(mpath, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sp = prompt_of(row["source_wav_path"])
            if sp:
                stem = Path(row["source_wav_path"]).stem
                train_sources.add((speaker_of(stem), sp))
            tp = prompt_of(row["target_wav_path"])
            if tp:
                train_tgt_prompts.add(tp)

    csvs = sorted({p for g in args.per_clip for p in globlib.glob(g, recursive=True)})
    if not csvs:
        raise SystemExit(f"[error] no CSV matched {args.per_clip}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    flagged = []
    for path in csvs:
        run = Path(path).parent.parent.name  # exp/<run>/eval_compare/x.csv -> <run>
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row.get("source", "")
                p = prompt_of(src)
                spk = speaker_of(src)
                source_seen = (spk, p) in train_sources if p else False
                target_seen = p in train_tgt_prompts if p else False
                flagged.append({
                    "run": run, "csv": path, "step": row.get("step"),
                    "source": src, "prompt": p,
                    "source_seen": source_seen, "target_seen": target_seen,
                    "class": "contaminated" if (source_seen or target_seen) else "clean",
                    "sim_cosine": _f(row.get("sim_cosine")),
                    "wer": _f(row.get("wer")),
                    "mos_pred": _f(row.get("mos_pred")),
                    "accent_label": row.get("accent_label"),
                })

    with open(out_root / "flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
        w.writeheader()
        w.writerows(flagged)

    # Aggregate per (run, step, class).
    groups = defaultdict(list)
    for r in flagged:
        groups[(r["run"], r["step"], r["class"])].append(r)

    def step_key(s):
        return (0, 0) if s == "base" else (1, int(s))

    summaries = []
    for (run, step, cls) in sorted(groups, key=lambda k: (k[0], step_key(k[1]), k[2])):
        sub = groups[(run, step, cls)]
        indian = sum(1 for r in sub if r["accent_label"] == "indian")
        summaries.append({
            "run": run, "step": step, "class": cls, "n": len(sub),
            "sim_mean": _mean([r["sim_cosine"] for r in sub]),
            "wer_mean": _mean([r["wer"] for r in sub]),
            "mos_mean": _mean([r["mos_pred"] for r in sub]),
            "indian": indian,
            "indian_frac": round(indian / len(sub), 4),
        })
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    # Human-readable: clean vs contaminated side by side, per run/step.
    n_contam_clips = len({r["source"] for r in flagged if r["class"] == "contaminated"})
    n_clips = len({r["source"] for r in flagged})
    print(f"[contamination] {n_contam_clips}/{n_clips} distinct eval source clips "
          f"overlap training (vs {len(train_sources)} train source utts)")
    for run in sorted({s["run"] for s in summaries}):
        print(f"\n== {run} ==")
        print(f"  {'step':>6} | {'clean: n indian mos':^26} | contaminated: n indian mos")
        steps = sorted({s["step"] for s in summaries if s["run"] == run}, key=step_key)
        for step in steps:
            cells = {}
            for cls in ("clean", "contaminated"):
                m = next((s for s in summaries
                          if s["run"] == run and s["step"] == step and s["class"] == cls),
                         None)
                cells[cls] = (f"{m['n']:>2} {m['indian']:>2}/{m['n']:<2} {m['mos_mean']}"
                              if m else "--")
            print(f"  {step:>6} | {cells['clean']:^26} | {cells['contaminated']}")

    print(f"\n[contamination] wrote {out_root}/flags.csv, summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
