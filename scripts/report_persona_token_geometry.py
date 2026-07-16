#!/usr/bin/env python
"""Phase 0 report: answer the representation-audit questions from token shards.

Reads shards produced by `scripts/extract_persona_codec_tokens.py` and reports:
semantic/acoustic vocabulary sizes, the semantic/acoustic frame-rate relation,
the target/source length-ratio distribution, semantic token edit distance,
acoustic code agreement, how often insertions/deletions are required, and
whether target acoustic groups can be reliably associated with target semantic
steps.  CPU-only; no audio or model is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="shard dir from extract_persona_codec_tokens")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from xvc.persona_codec.tokens import (
        group_acoustic_codes,
        length_ratio_stats,
        levenshtein_alignment,
    )

    pairs_root = Path(args.pairs)
    meta = json.loads((pairs_root / "meta.json").read_text(encoding="utf-8"))
    group = int(meta["acoustic_group_size"])

    ratios = []
    edit_rows = []
    grouping_exact = 0
    grouping_reliable = 0
    grouping_total = 0
    dropped_hist: dict[int, int] = {}
    acoustic_agree_matched = []
    acoustic_agree_positional = []
    semantic_identity = []
    max_sem_id = -1
    max_ac_id = -1

    for split in ("train", "val"):
        items = torch.load(pairs_root / f"{split}.pt", map_location="cpu", weights_only=False)
        for item in items:
            src_sem = item["src_semantic"].numpy()
            tgt_sem = item["tgt_semantic"].numpy()
            src_ac = item["src_acoustic"].numpy()
            tgt_ac = item["tgt_acoustic"].numpy()
            max_sem_id = max(max_sem_id, int(src_sem.max()), int(tgt_sem.max()))
            max_ac_id = max(max_ac_id, int(src_ac.max()), int(tgt_ac.max()))

            ratios.append(len(tgt_sem) / len(src_sem))
            for side in ("source", "target"):
                grouping_total += 1
                trims = item["trims"][side]
                dropped = trims["dropped_acoustic_frames"]
                dropped_hist[dropped] = dropped_hist.get(dropped, 0) + 1
                if dropped == 0 and trims["semantic_raw"] == trims["semantic"]:
                    grouping_exact += 1
                # Reliable association only requires that no semantic step was
                # lost; a sub-group encoder-edge trim (< group codes) is the
                # expected +-1-frame behaviour, not a geometry failure.
                if trims["semantic_raw"] == trims["semantic"] and dropped < group:
                    grouping_reliable += 1

            alignment = levenshtein_alignment(src_sem, tgt_sem)
            matched_pairs = alignment.pop("matched_pairs")
            alignment.update({"split": split, "pair_id": item["pair_id"]})
            edit_rows.append(alignment)
            semantic_identity.append(alignment["matches"] / max(len(src_sem), len(tgt_sem)))

            # Acoustic agreement between the streams: (a) on semantically
            # matched steps via the token alignment, (b) positionally under
            # min-length truncation as a control.
            src_groups = group_acoustic_codes(src_ac, group)
            tgt_groups = group_acoustic_codes(tgt_ac, group)
            if matched_pairs:
                agree = np.mean(
                    [
                        float((src_groups[i] == tgt_groups[j]).mean())
                        for i, j in matched_pairs
                        if i < len(src_groups) and j < len(tgt_groups)
                    ]
                )
                acoustic_agree_matched.append(float(agree))
            shared = min(len(src_groups), len(tgt_groups))
            acoustic_agree_positional.append(
                float((src_groups[:shared] == tgt_groups[:shared]).mean())
            )

    n = len(edit_rows)
    normalized = [row["normalized"] for row in edit_rows]
    insert_rate = [row["insertions"] / row["target_len"] for row in edit_rows]
    delete_rate = [row["deletions"] / row["source_len"] for row in edit_rows]
    needs_indel = sum(
        1 for row in edit_rows if row["insertions"] > 0 or row["deletions"] > 0
    )

    report = {
        "pairs_root": str(pairs_root).replace("\\", "/"),
        "n_pairs": n,
        "semantic_vocab_size": meta["semantic_vocab_size"],
        "acoustic_vocab_size": meta["acoustic_vocab_size"],
        "observed_max_semantic_id": max_sem_id,
        "observed_max_acoustic_id": max_ac_id,
        "semantic_token_hz": meta["semantic_token_hz"],
        "acoustic_code_hz": meta["acoustic_code_hz"],
        "acoustic_group_size": group,
        "frame_rate_relation": (
            f"{meta['acoustic_code_hz']:.2f} Hz acoustic / "
            f"{meta['semantic_token_hz']:.2f} Hz semantic = {group} codes per semantic step"
        ),
        "grouping": {
            "streams_checked": grouping_total,
            "exact_group_multiple": grouping_exact,
            "exact_fraction": grouping_exact / grouping_total if grouping_total else None,
            "reliable_association": grouping_reliable,
            "reliable_fraction": (
                grouping_reliable / grouping_total if grouping_total else None
            ),
            "dropped_acoustic_frame_histogram": {
                str(k): v for k, v in sorted(dropped_hist.items())
            },
            "verdict": (
                "acoustic groups associate reliably with semantic steps"
                if grouping_total and grouping_reliable / grouping_total >= 0.95
                else "grouping unreliable - investigate before training"
            ),
        },
        "target_over_source_semantic_length_ratio": length_ratio_stats(ratios),
        "semantic_edit_distance_normalized": length_ratio_stats(normalized),
        "semantic_identity_fraction": length_ratio_stats(semantic_identity),
        "insertion_rate_per_target_token": length_ratio_stats(insert_rate),
        "deletion_rate_per_source_token": length_ratio_stats(delete_rate),
        "pairs_requiring_insertions_or_deletions": {
            "count": needs_indel,
            "fraction": needs_indel / n if n else None,
        },
        "acoustic_group_agreement_on_matched_semantic_steps": (
            length_ratio_stats(acoustic_agree_matched) if acoustic_agree_matched else None
        ),
        "acoustic_group_agreement_positional_control": length_ratio_stats(
            acoustic_agree_positional
        ),
        "reading_guide": (
            "High normalized semantic edit distance plus low acoustic agreement "
            "means both streams must be generated, not copied or locally edited; "
            "a broad length-ratio distribution means frame-wise resampling is the "
            "wrong output model and variable-length generation with EOS is required."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "per_pair_edit.jsonl").open("w", encoding="utf-8") as handle:
        for row in edit_rows:
            handle.write(json.dumps(row) + "\n")
    for key in (
        "frame_rate_relation",
        "grouping",
        "target_over_source_semantic_length_ratio",
        "semantic_edit_distance_normalized",
        "pairs_requiring_insertions_or_deletions",
    ):
        print(f"{key}: {json.dumps(report[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
