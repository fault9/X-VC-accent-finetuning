#!/usr/bin/env python
"""Extract pristine native->target-persona dual-stream supervision from X-VC.

Both recordings stay on their original timelines. MFA TextGrids provide only
matched phone spans; no waveform or feature resampling is performed. The
output contains no source-speaker conditioning, allowing one target-persona
mapper to accept arbitrary source voices at inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load_rows(dataset_root: Path, split: str):
    manifest = dataset_root / "manifests" / f"{split}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/xvc.yaml")
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--max-pairs", type=int, default=0, help="per split; 0 means all")
    parser.add_argument("--shard-size", type=int, default=24)
    parser.add_argument("--min-label-match", type=float, default=0.90)
    parser.add_argument("--min-phone-segments", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    from bins.infer_utils import load_xvc
    from models.codec.sac.utils import process_audio
    from xvc.data.stream_swap import phone_segments_from_textgrids

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.out)
    if output_root.exists() and any(output_root.rglob("shard_*.pt")):
        raise SystemExit(
            f"[error] refusing to mix with existing extracted shards: {output_root}"
        )
    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    hop = int(cfg["latent_hop_length"])

    @torch.inference_mode()
    def encode(path: str):
        audio = process_audio(path, cfg, hop)
        waveform = torch.from_numpy(audio).view(1, 1, -1).float().to(device)
        token_ids = model.semantic_encoder.extract_and_encode(
            waveform.squeeze(1)
        )["speech_tokens"]
        semantic = model.semantic_adapter(
            model.semantic_encoder.embed_ids(token_ids).transpose(1, 2)
        )
        acoustic_outputs = model.acoustic_quantizer(model.acoustic_encoder(waveform))
        acoustic_zq, acoustic_indices = acoustic_outputs[0], acoustic_outputs[1]
        frames = min(semantic.shape[-1], acoustic_zq.shape[-1], acoustic_indices.shape[-1])
        return (
            semantic[0, :, :frames].half().cpu(),
            acoustic_zq[0, :, :frames].half().cpu(),
            acoustic_indices[0, :frames].cpu(),
        )

    # ASI records each prompt once, while multiple native speakers can point to
    # that same target realization. Cache by prompt to avoid redundant X-VC
    # encoding without collapsing the distinct source->target training pairs.
    target_cache = {}

    splits = ("train", "val") if args.split == "all" else (args.split,)
    overall = {"seen": 0, "kept": 0, "failed": 0, "splits": {}}
    all_failures = []
    for split in splits:
        rows = _load_rows(dataset_root, split)
        rows = [row for row in rows if args.target_speaker.casefold() in str(row.get("target_speaker", "")).casefold()]
        if args.max_pairs:
            rows = rows[: args.max_pairs]
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        manifest_path = split_root / "pairs_manifest.jsonl"
        items = []
        shard_index = 0
        kept = 0
        failures = []

        def flush():
            nonlocal items, shard_index
            if not items:
                return
            torch.save(items, split_root / f"shard_{shard_index:04d}.pt")
            items = []
            shard_index += 1

        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in rows:
                overall["seen"] += 1
                try:
                    source_semantic, source_zq, source_codes = encode(row["source_wav_path"])
                    prompt_key = row.get("prompt_id") or row["target_wav_path"]
                    if prompt_key not in target_cache:
                        target_cache[prompt_key] = encode(row["target_wav_path"])
                    target_semantic, target_zq, target_codes = target_cache[prompt_key]
                    segments, phone_meta = phone_segments_from_textgrids(
                        row["source_textgrid_path"],
                        row["target_textgrid_path"],
                        source_semantic.shape[-1],
                        target_semantic.shape[-1],
                        min_label_match=args.min_label_match,
                        min_matched_phones=args.min_phone_segments,
                    )
                    item = {
                        "source_semantic": source_semantic,
                        "source_zq": source_zq,
                        "source_codes": source_codes,
                        "target_semantic": target_semantic,
                        "target_zq": target_zq,
                        "target_codes": target_codes,
                        "phone_segments": segments,
                        "phone_meta": phone_meta,
                        "meta": {
                            "dataset": dataset_root.name,
                            "split": split,
                            "pair_id": row.get("pair_id"),
                            "prompt_id": row.get("prompt_id"),
                            "source_utt": row["source_utt"],
                            "source_speaker": row["source_speaker"],
                            "source_wav_path": row["source_wav_path"],
                            "target_utt": row["target_utt"],
                            "target_speaker": row["target_speaker"],
                            "target_wav_path": row["target_wav_path"],
                        },
                    }
                    manifest.write(
                        json.dumps(
                            {
                                "shard": f"shard_{shard_index:04d}.pt",
                                "index": len(items),
                                "source_frames": source_semantic.shape[-1],
                                "target_frames": target_semantic.shape[-1],
                                "matched_phones": len(segments),
                                **item["meta"],
                            }
                        )
                        + "\n"
                    )
                    items.append(item)
                    kept += 1
                    overall["kept"] += 1
                    if len(items) >= args.shard_size:
                        flush()
                    print(f"  [{split}] {kept}/{len(rows)} {row['source_utt']} phones={len(segments)}")
                except Exception as exc:  # retain the usable corpus, record every rejection
                    failure = {
                        "split": split,
                        "source_utt": row.get("source_utt"),
                        "target_utt": row.get("target_utt"),
                        "error": str(exc),
                    }
                    failures.append(failure)
                    all_failures.append(failure)
                    overall["failed"] += 1
                    print(f"  [skip] {row.get('source_utt')}: {exc}", file=sys.stderr)
            flush()
        overall["splits"][split] = {
            "seen": len(rows),
            "kept": kept,
            "coverage": round(kept / max(len(rows), 1), 6),
            "shards": shard_index,
        }

    coverage_pass = bool(overall["kept"]) and all(
        info["coverage"] >= 0.75 for info in overall["splits"].values()
    )
    summary = {
        "schema_version": 1,
        "status": "pass" if coverage_pass else "fail",
        "purpose": "source-agnostic target-persona dual-stream supervision",
        "dataset_root": str(dataset_root),
        "target_speaker": args.target_speaker,
        "config": args.config,
        "checkpoint": args.ckpt,
        "alignment_policy": "genuine MFA phone spans only; no audio/feature warping",
        "source_identity_used_by_mapper": False,
        **overall,
        "failures": all_failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))
    if not coverage_pass:
        print("[error] phone-pair coverage below 75%; refusing a biased training set", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
