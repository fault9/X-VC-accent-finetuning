#!/usr/bin/env python
"""Go/no-go analysis for the AccentBridge: is the native-vs-L2 difference even
VISIBLE in X-VC's content representations?

Reads the shards written by scripts/extract_accentbridge_pairs.py and reports,
per representation level:

  * same-prompt distance  : aligned native-warped vs L2 features of the SAME
    prompt (cosine + normalized L2 per frame, then per-pair means). This is the
    signal a bridge would have to learn.
  * shuffled-pair control : native-warped of pair i vs L2 of pair j (same
    speaker-pair, different prompt, cropped to common length) — the content-
    difference ceiling. If same-prompt distance approaches this, alignment or
    representation is broken.
  * token mismatch        : frame-level disagreement of warped-native vs L2
    WhisperVQ ids, plus normalized Levenshtein on the unwarped sequences.

Verdict logic (printed + saved):
  * same-prompt cosine > ~0.98 and token mismatch < ~5%  -> representation is
    effectively ACCENT-INVARIANT at this level: a tiny editor has nothing to
    learn here; try another level or abandon Path A/B at this insertion point.
  * same-prompt distance clearly above the invariance zone AND clearly below
    the shuffled ceiling -> structured accent signal exists: GO for the bridge.
  * same-prompt ~ shuffled -> alignment/representation problem; do not train.

Usage:
    python scripts/analyze_accentbridge_pairs.py \
        --pairs-dir data/accentbridge_pairs/val --out exp/accentbridge_analysis

Pure torch + stdlib; CPU is fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", required=True,
                    help="output dir of extract_accentbridge_pairs.py (per split)")
    ap.add_argument("--out", default="exp/accentbridge_analysis")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    pairs_dir = Path(args.pairs_dir)
    shards = sorted(pairs_dir.glob("shard_*.pt"))
    if not shards:
        raise SystemExit(f"[error] no shards in {pairs_dir}")
    items = []
    for s in shards:
        items.extend(torch.load(s, map_location="cpu"))
        if args.limit and len(items) >= args.limit:
            items = items[: args.limit]
            break
    print(f"[analyze] {len(items)} pair(s) from {len(shards)} shard(s)")

    def frame_metrics(a, b):
        """a, b: (C, T) fp16 -> mean frame cosine + normalized L2."""
        a = a.float()
        b = b.float()
        cos = F.cosine_similarity(a, b, dim=0).mean().item()
        l2 = ((a - b).norm(dim=0) / (b.norm(dim=0) + 1e-8)).mean().item()
        return cos, l2

    reprs = ["sem_adapted"] + (["zq"] if "zq_tgt" in items[0] else [])
    rows = []
    for it in items:
        m = it["meta"]
        row = {"pair": f"{m['source_utt']}->{m['target_utt']}",
               "speaker_pair": f"{m['source_speaker']}->{m['target_speaker']}",
               "prompt": m["prompt"], "dataset": m["dataset"],
               "T": int(it["sem_adapted_tgt"].shape[-1])}
        for r in reprs:
            src = it[f"{r}_src_warped"] if r == "sem_adapted" else it["zq_src_warped"]
            tgt = it[f"{r}_tgt"]
            cos, l2 = frame_metrics(src, tgt)
            row[f"{r}_cos"] = round(cos, 4)
            row[f"{r}_l2n"] = round(l2, 4)
        tw, tt = it["tokens_src_warped"], it["tokens_tgt"]
        T = min(tw.shape[-1], tt.shape[-1])
        row["token_frame_mismatch"] = round(float((tw[:T] != tt[:T]).float().mean()), 4)
        ts, tu = it["tokens_src"].tolist(), it["tokens_tgt"].tolist()
        row["token_edit_dist_norm"] = round(
            levenshtein(ts, tu) / max(len(ts), len(tu), 1), 4)
        rows.append(row)

    # Shuffled control: pair i's warped source vs pair (i+1)'s target within the
    # same speaker-pair (deterministic rotation, cropped to common length).
    by_sp = defaultdict(list)
    for i, it in enumerate(items):
        by_sp[rows[i]["speaker_pair"]].append(i)
    shuf = defaultdict(list)
    for sp, idxs in by_sp.items():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            j = idxs[(k + 1) % len(idxs)]
            a = items[i]["sem_adapted_src_warped"]
            b = items[j]["sem_adapted_tgt"]
            T = min(a.shape[-1], b.shape[-1])
            cos, l2 = frame_metrics(a[:, :T], b[:, :T])
            shuf["cos"].append(cos)
            shuf["l2n"].append(l2)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "per_pair.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    groups = {"ALL": rows}
    for sp in sorted({r["speaker_pair"] for r in rows}):
        groups[sp] = [r for r in rows if r["speaker_pair"] == sp]
    print(f"\n{'group':<14} {'n':>3} {'sem_cos':>8} {'sem_l2n':>8} "
          f"{'tok_mis':>8} {'tok_edit':>8}")
    for g, sub in groups.items():
        summary[g] = {
            "n": len(sub),
            "sem_cos": _mean([r["sem_adapted_cos"] for r in sub]),
            "sem_l2n": _mean([r["sem_adapted_l2n"] for r in sub]),
            "token_frame_mismatch": _mean([r["token_frame_mismatch"] for r in sub]),
            "token_edit_dist_norm": _mean([r["token_edit_dist_norm"] for r in sub]),
        }
        if "zq_cos" in sub[0]:
            summary[g]["zq_cos"] = _mean([r["zq_cos"] for r in sub])
            summary[g]["zq_l2n"] = _mean([r["zq_l2n"] for r in sub])
        s = summary[g]
        print(f"{g:<14} {s['n']:>3} {s['sem_cos']:>8} {s['sem_l2n']:>8} "
              f"{s['token_frame_mismatch']:>8} {s['token_edit_dist_norm']:>8}")
    ctrl = {"cos": _mean(shuf["cos"]), "l2n": _mean(shuf["l2n"]), "n": len(shuf["cos"])}
    print(f"{'SHUFFLED-CTRL':<14} {ctrl['n']:>3} {ctrl['cos']:>8} {ctrl['l2n']:>8}")

    # Verdict.
    a = summary["ALL"]
    verdict = "INCONCLUSIVE"
    reasons = []
    if a["sem_cos"] is not None and ctrl["cos"] is not None:
        gap = 1.0 - a["sem_cos"]
        ceiling = 1.0 - ctrl["cos"]
        if gap < 0.02 and a["token_frame_mismatch"] < 0.05:
            verdict = "ACCENT-INVARIANT"
            reasons.append("same-prompt features nearly identical and tokens "
                           "barely disagree — a tiny editor has no signal here")
        elif ceiling > 0 and gap > 0.6 * ceiling:
            verdict = "ALIGNMENT-OR-REPR-BROKEN"
            reasons.append("same-prompt distance approaches the shuffled ceiling "
                           "— aligned pairs are barely closer than random ones")
        else:
            verdict = "GO"
            reasons.append(f"structured signal: same-prompt gap {gap:.4f} vs "
                           f"shuffled ceiling {ceiling:.4f}, token mismatch "
                           f"{a['token_frame_mismatch']:.1%}")
    print(f"\nVERDICT: {verdict} -- {'; '.join(reasons)}")

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"groups": summary, "shuffled_control": ctrl,
                   "verdict": verdict, "reasons": reasons}, f, indent=2)
    print(f"[analyze] wrote {out_root}/per_pair.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
