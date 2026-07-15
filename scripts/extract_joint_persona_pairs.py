#!/usr/bin/env python
"""Extract pristine native->target-persona dual-stream supervision from X-VC.

Both recordings stay on their original timelines. MFA TextGrids provide only
matched phone spans; no waveform or feature resampling is performed. The
output contains no source-speaker conditioning, allowing one target-persona
mapper to accept arbitrary source voices at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load_rows(dataset_root: Path, split: str):
    manifest = dataset_root / "manifests" / f"{split}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def _stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _crop_audio(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """Exact fixed waveform crop with zero padding only at recording edges."""
    end = start + length
    source_start = max(0, start)
    source_end = min(len(audio), end)
    cropped = audio[source_start:source_end]
    return np.pad(
        cropped,
        (source_start - start, end - source_end),
        mode="constant",
    ).astype(np.float32, copy=False)


def _window_phone_segments(
    matches,
    source_start_seconds: float,
    target_start_seconds: float,
    window_seconds: float,
    source_frames: int,
    target_frames: int,
    *,
    min_phone_frames: int = 2,
):
    """Map only fully contained matched phones into two independent windows."""

    def frame_range(interval, window_start, frames):
        lo = int(round((interval[0] - window_start) / window_seconds * frames))
        hi = int(round((interval[1] - window_start) / window_seconds * frames))
        lo = max(0, min(frames, lo))
        hi = max(lo + 1, min(frames, hi))
        length = hi - lo
        if length < min_phone_frames:
            return None
        margin = min(1, int(round(length * 0.15)))
        if length - 2 * margin < min_phone_frames:
            margin = 0
        return [lo + margin, hi - margin]

    source_end = source_start_seconds + window_seconds
    target_end = target_start_seconds + window_seconds
    segments = []
    for match in matches:
        source_interval = match["src_seconds"]
        target_interval = match["tgt_seconds"]
        # Cut boundary phones are deliberately excluded: convolutional edge
        # context may differ, and partial-phone targets are ambiguous.
        if not (
            source_start_seconds <= source_interval[0]
            and source_interval[1] <= source_end
            and target_start_seconds <= target_interval[0]
            and target_interval[1] <= target_end
        ):
            continue
        source_range = frame_range(source_interval, source_start_seconds, source_frames)
        target_range = frame_range(target_interval, target_start_seconds, target_frames)
        if source_range is None or target_range is None:
            continue
        segments.append(
            {
                "phone": match["phone"],
                "src": source_range,
                "tgt": target_range,
                "duration_ratio": match["duration_ratio"],
                "confidence": match["confidence"],
            }
        )
    return segments


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
    parser.add_argument("--window-seconds", type=float, default=2.4)
    parser.add_argument("--windows-per-pair", type=int, default=2)
    parser.add_argument("--val-windows-per-pair", type=int, default=1)
    parser.add_argument("--mapper-history-frames", type=int, default=62)
    parser.add_argument("--max-lookahead-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    from bins.infer_utils import load_xvc
    from models.codec.sac.utils import process_audio
    from xvc.data.stream_swap import matched_phone_intervals_from_textgrids

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.out)
    if output_root.exists() and any(output_root.rglob("shard_*.pt")):
        raise SystemExit(
            f"[error] refusing to mix with existing extracted shards: {output_root}"
        )
    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    hop = int(cfg["latent_hop_length"])
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(round(args.window_seconds * sample_rate))
    expected_stream_frames = int(round(args.window_seconds * 50.0))
    if window_samples <= 0 or window_samples % hop:
        raise SystemExit("[error] waveform window must be positive and divisible by latent_hop_length")
    if args.windows_per_pair < 1 or args.val_windows_per_pair < 1:
        raise SystemExit("[error] windows-per-pair values must be positive")
    if args.mapper_history_frames < 0 or args.max_lookahead_frames < 0:
        raise SystemExit("[error] mapper context values must be non-negative")

    @torch.inference_mode()
    def encode_waveform(audio: np.ndarray):
        if len(audio) != window_samples:
            raise ValueError(f"expected {window_samples} waveform samples, got {len(audio)}")
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
        if frames != expected_stream_frames:
            raise ValueError(
                f"2.4 s X-VC window should encode to {expected_stream_frames} frames, "
                f"got semantic={semantic.shape[-1]} acoustic={acoustic_zq.shape[-1]} "
                f"codes={acoustic_indices.shape[-1]}"
            )
        return (
            semantic[0, :, :frames].half().cpu(),
            acoustic_zq[0, :, :frames].half().cpu(),
            acoustic_indices[0, :frames].cpu(),
        )

    audio_cache = {}
    target_window_cache = {}

    def processed_audio(path: str):
        if path not in audio_cache:
            audio_cache[path] = process_audio(path, cfg, hop)
        return audio_cache[path]

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
        rows_kept = 0
        shortfalls = 0
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
                    matches, phone_meta = matched_phone_intervals_from_textgrids(
                        row["source_textgrid_path"],
                        row["target_textgrid_path"],
                        min_label_match=args.min_label_match,
                        min_matched_phones=args.min_phone_segments,
                    )
                    source_audio = processed_audio(row["source_wav_path"])
                    target_audio = processed_audio(row["target_wav_path"])
                    source_duration = len(source_audio) / sample_rate
                    target_duration = len(target_audio) / sample_rate
                    windows_requested = (
                        args.val_windows_per_pair if split == "val" else args.windows_per_pair
                    )
                    rng = _stable_rng(
                        args.seed,
                        str(row.get("pair_id") or row.get("source_utt")),
                    )
                    candidates = list(matches)
                    rng.shuffle(candidates)
                    created = 0
                    used_starts = set()
                    for match in candidates:
                        if created >= windows_requested:
                            break
                        source_phone = match["src_seconds"]
                        target_phone = match["tgt_seconds"]
                        source_phone_duration = source_phone[1] - source_phone[0]
                        target_phone_duration = target_phone[1] - target_phone[0]
                        source_position_min = (
                            args.mapper_history_frames / 50.0
                            + source_phone_duration / 2.0
                        )
                        source_position_max = (
                            args.window_seconds
                            - args.max_lookahead_frames / 50.0
                            - source_phone_duration / 2.0
                        )
                        if source_position_max <= source_position_min:
                            continue
                        source_position = rng.uniform(
                            source_position_min, source_position_max
                        )
                        source_center = sum(source_phone) / 2.0
                        target_center = sum(target_phone) / 2.0
                        source_start_seconds = float(
                            np.clip(
                                source_center - source_position,
                                0.0,
                                max(0.0, source_duration - args.window_seconds),
                            )
                        )
                        target_position = float(
                            np.clip(
                                args.window_seconds / 2.0,
                                target_phone_duration / 2.0,
                                args.window_seconds - target_phone_duration / 2.0,
                            )
                        )
                        target_start_seconds = float(
                            np.clip(
                                target_center - target_position,
                                0.0,
                                max(0.0, target_duration - args.window_seconds),
                            )
                        )
                        source_start_sample = int(round(source_start_seconds * sample_rate))
                        target_start_sample = int(round(target_start_seconds * sample_rate))
                        start_key = (source_start_sample, target_start_sample)
                        if start_key in used_starts:
                            continue
                        source_window = _crop_audio(
                            source_audio, source_start_sample, window_samples
                        )
                        source_semantic, source_zq, source_codes = encode_waveform(
                            source_window
                        )
                        target_key = (row["target_wav_path"], target_start_sample)
                        if target_key not in target_window_cache:
                            target_window = _crop_audio(
                                target_audio, target_start_sample, window_samples
                            )
                            target_window_cache[target_key] = encode_waveform(target_window)
                        target_semantic, target_zq, target_codes = target_window_cache[target_key]
                        segments = _window_phone_segments(
                            matches,
                            source_start_sample / sample_rate,
                            target_start_sample / sample_rate,
                            args.window_seconds,
                            source_semantic.shape[-1],
                            target_semantic.shape[-1],
                        )
                        valid_end = source_semantic.shape[-1] - args.max_lookahead_frames
                        usable = []
                        for segment in segments:
                            source_start, source_end = segment["src"]
                            target_start, target_end = segment["tgt"]
                            overlap_start = max(
                                source_start, args.mapper_history_frames
                            )
                            overlap_end = min(source_end, valid_end)
                            if overlap_end - overlap_start < 2:
                                continue
                            source_length = source_end - source_start
                            target_length = target_end - target_start
                            projected_target_length = round(
                                (overlap_end - overlap_start)
                                / max(source_length, 1)
                                * target_length
                            )
                            if projected_target_length >= 2:
                                usable.append(segment)
                        if len(usable) < args.min_phone_segments:
                            continue
                        window_id = f"{row.get('pair_id') or row['source_utt']}__w{created:02d}"
                        meta = {
                            "dataset": dataset_root.name,
                            "split": split,
                            "pair_id": row.get("pair_id"),
                            "window_id": window_id,
                            "prompt_id": row.get("prompt_id"),
                            "source_utt": row["source_utt"],
                            "source_speaker": row["source_speaker"],
                            "source_wav_path": row["source_wav_path"],
                            "source_window_start_seconds": round(
                                source_start_sample / sample_rate, 6
                            ),
                            "source_encoded_from_window": True,
                            "target_utt": row["target_utt"],
                            "target_speaker": row["target_speaker"],
                            "target_wav_path": row["target_wav_path"],
                            "target_window_start_seconds": round(
                                target_start_sample / sample_rate, 6
                            ),
                            "target_encoded_from_window": True,
                            "window_seconds": args.window_seconds,
                        }
                        item = {
                            "source_semantic": source_semantic,
                            "source_zq": source_zq,
                            "source_codes": source_codes,
                            "target_semantic": target_semantic,
                            "target_zq": target_zq,
                            "target_codes": target_codes,
                            "phone_segments": segments,
                            "phone_meta": {
                                **phone_meta,
                                "context_valid_phones": len(usable),
                            },
                            "meta": meta,
                        }
                        manifest.write(
                            json.dumps(
                                {
                                    "shard": f"shard_{shard_index:04d}.pt",
                                    "index": len(items),
                                    "source_frames": source_semantic.shape[-1],
                                    "target_frames": target_semantic.shape[-1],
                                    "matched_phones": len(segments),
                                    "context_valid_phones": len(usable),
                                    **meta,
                                }
                            )
                            + "\n"
                        )
                        items.append(item)
                        used_starts.add(start_key)
                        created += 1
                        kept += 1
                        overall["kept"] += 1
                        if len(items) >= args.shard_size:
                            flush()
                    if created == 0:
                        raise ValueError("could not construct a context-valid 2.4 s window")
                    rows_kept += 1
                    if created < windows_requested:
                        shortfalls += 1
                    print(
                        f"  [{split}] rows={rows_kept}/{len(rows)} windows={kept} "
                        f"{row['source_utt']} created={created}/{windows_requested}"
                    )
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
            "rows_kept": rows_kept,
            "windows_kept": kept,
            "coverage": round(rows_kept / max(len(rows), 1), 6),
            "window_shortfalls": shortfalls,
            "shards": shard_index,
        }

    coverage_pass = bool(overall["kept"]) and all(
        info["coverage"] >= 0.75 for info in overall["splits"].values()
    )
    summary = {
        "schema_version": 2,
        "status": "pass" if coverage_pass else "fail",
        "purpose": "source-agnostic target-persona dual-stream supervision",
        "dataset_root": str(dataset_root),
        "target_speaker": args.target_speaker,
        "config": args.config,
        "checkpoint": args.ckpt,
        "alignment_policy": "genuine MFA phone spans select supervision only; no audio/feature warping",
        "window_policy": {
            "waveform_crop_before_frozen_encoder": True,
            "window_seconds": args.window_seconds,
            "window_frames": expected_stream_frames,
            "train_windows_per_pair": args.windows_per_pair,
            "val_windows_per_pair": args.val_windows_per_pair,
            "mapper_history_frames": args.mapper_history_frames,
            "max_lookahead_frames": args.max_lookahead_frames,
            "target_timeline_independent": True,
        },
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
