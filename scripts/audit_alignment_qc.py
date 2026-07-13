#!/usr/bin/env python
"""Rank crosspair DTW alignments by quality and emit an exclusion list.

align_crosspairs.py already writes per-pair QC (<dataset>/alignment_qc.jsonl):
global stretch ratio, DTW anchor repairs, local stretch extremes, and the
fitted mel-distance of the DTW warp vs uniform resampling. This script turns
that into an actionable filter: pairs are flagged for hard reasons (DTW no
better than uniform resampling, heavy anchor repair, local stretch pinned at
the safety bounds, extreme global stretch) plus the worst tail by absolute
DTW mel distance, and written in --exclude-file format (pair IDs, one per
line, '#' comments) for a filtered rebuild:

    python scripts/audit_alignment_qc.py \
        --qc data/crosspair_hindi_latent_400/alignment_qc.jsonl \
        --manifest data/crosspair_hindi_latent_400_asi/manifests/train.jsonl \
        --out exclude_bad_alignments.txt
    python scripts/align_crosspairs.py ... --exclude-file exclude_bad_alignments.txt

Every alignment error is a training signal teaching the model a wrong frame
correspondence -- under the latent objective these are prime suspects for the
naturalness collapse. Distill datasets need no such audit (same-timeline by
construction); their analogue is the teacher-render MOS gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_target_key(row: dict) -> str:
    """Derive the QC rows' 'SPEAKER:arctic_prompt' key from a manifest row."""
    utt = row["target_utt"][:-3] if row["target_utt"].endswith("_ft") else row["target_utt"]
    speaker, rest = utt.split("__", 1)
    return f"{speaker}:arctic_{rest.split('_arctic_', 1)[1]}"


def pair_key(source_utt: str, target_key: str) -> str:
    """source_utt alone is ambiguous once one source pairs with several target
    speakers (ASI + RRBI) -- exclusions are keyed per (source, target) pair."""
    return f"{source_utt} {target_key}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc", required=True, help="alignment_qc.jsonl from align_crosspairs.py")
    ap.add_argument("--manifest", default=None,
                    help="optional manifest(s, comma-separated); restrict the audit "
                         "to pairs present there (e.g. the ASI persona subset)")
    ap.add_argument("--out", default=None, help="write exclusions here (default: stdout summary only)")
    ap.add_argument("--worst-frac", type=float, default=0.10,
                    help="also exclude this worst fraction by DTW mel distance")
    ap.add_argument("--max-anchor-removal", type=float, default=0.10,
                    help="flag pairs above this anchor-repair fraction")
    ap.add_argument("--min-improvement", type=float, default=0.0,
                    help="flag pairs whose DTW mel distance is not better than "
                         "uniform resampling by more than this")
    ap.add_argument("--stretch-bounds", default="0.87,1.15",
                    help="flag global source/target stretch outside lo,hi")
    args = ap.parse_args(argv)

    rows = load_jsonl(args.qc)
    if args.manifest:
        keep = set()
        for m in args.manifest.split(","):
            keep |= {pair_key(r["source_utt"], manifest_target_key(r))
                     for r in load_jsonl(m.strip())}
        rows = [r for r in rows if pair_key(r["source_utt"], r["target_key"]) in keep]
        missing = keep - {pair_key(r["source_utt"], r["target_key"]) for r in rows}
        if missing:
            print(f"[warn] {len(missing)} manifest pairs have no QC row "
                  f"(older build?): e.g. {sorted(missing)[:3]}")
    if not rows:
        raise SystemExit("[error] no QC rows to audit")

    lo, hi = (float(x) for x in args.stretch_bounds.split(","))
    flagged: dict[str, list[str]] = {}

    def flag(r, reason):
        flagged.setdefault(pair_key(r["source_utt"], r["target_key"]), []).append(reason)

    for r in rows:
        if r.get("diagnostic_mel_improvement") is not None \
                and r["diagnostic_mel_improvement"] <= args.min_improvement:
            flag(r, f"dtw_no_better_than_resample({r['diagnostic_mel_improvement']:+.2f})")
        if r.get("anchor_removal_fraction", 0) > args.max_anchor_removal:
            flag(r, f"anchor_repairs({r['anchor_removal_fraction']:.0%})")
        ls_min, ls_max = r.get("local_stretch_min"), r.get("local_stretch_max")
        if ls_min is not None and ls_min <= 0.34:
            flag(r, f"local_stretch_min({ls_min:.2f})")
        if ls_max is not None and ls_max >= 2.9:
            flag(r, f"local_stretch_max({ls_max:.2f})")
        gs = r.get("global_stretch_ratio")
        if gs is not None and not (lo <= gs <= hi):
            flag(r, f"global_stretch({gs:.2f})")

    by_dist = sorted(rows, key=lambda r: -r.get("diagnostic_mel_dist_dtw", 0.0))
    n_worst = max(1, int(len(rows) * args.worst_frac))
    for r in by_dist[:n_worst]:
        flag(r, f"worst_{args.worst_frac:.0%}_mel_dist({r['diagnostic_mel_dist_dtw']:.2f})")

    dists = sorted(r.get("diagnostic_mel_dist_dtw", 0.0) for r in rows)
    print(f"[audit] {len(rows)} pairs; mel_dist_dtw median "
          f"{dists[len(dists) // 2]:.2f}, p90 {dists[int(len(dists) * 0.9)]:.2f}, "
          f"max {dists[-1]:.2f}")
    print(f"[audit] flagged {len(flagged)} pairs "
          f"({len(flagged) / len(rows):.0%}) for exclusion")
    reasons: dict[str, int] = {}
    for rs in flagged.values():
        for reason in rs:
            reasons[reason.split("(")[0]] = reasons.get(reason.split("(")[0], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {reason}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# audit_alignment_qc.py over {args.qc}\n")
            f.write("# columns: source_utt target_key (pair-unique; source_utt "
                    "alone is ambiguous when a source pairs with several "
                    "target speakers)\n")
            for key in sorted(flagged):
                f.write(f"{key}  # {'; '.join(flagged[key])}\n")
        print(f"[out] {args.out} ({len(flagged)} pair exclusions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
