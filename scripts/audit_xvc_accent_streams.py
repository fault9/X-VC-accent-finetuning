#!/usr/bin/env python
"""Causally localize accent information in X-VC's source-side streams.

This is a diagnostic renderer, not a training script.  For matched native/ASI
prompts it independently swaps the source semantic stream and quantized
acoustic stream before the existing prenet/converter/decoder:

  native_sem__native_zq       native control
  asi_sem__native_zq          semantic-stream intervention
  native_sem__asi_zq          acoustic-code intervention
  asi_sem__asi_zq_mapped      both ASI streams mapped to native time
  asi_sem__asi_zq_original    ASI identity round-trip (no time mapping)

MFA phone intervals are used only to map target activations onto the native
timeline for this audit.  No waveform or feature map is saved into training
data, and no model parameter is updated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


CONDITIONS = {
    "native_sem__native_zq": "Native semantic stream + native acoustic codes.",
    "asi_sem__native_zq": "Phone-mapped ASI semantic stream + native acoustic codes.",
    "native_sem__asi_zq": "Native semantic stream + phone-mapped ASI acoustic codes.",
    "asi_sem__asi_zq_mapped": "Both ASI streams phone-mapped to native time.",
    "asi_sem__asi_zq_original": "Original-timeline ASI identity round-trip; no mapping.",
}


def _target_matches(item: dict[str, Any], speaker: str) -> bool:
    wanted = speaker.casefold()
    fields = [
        item.get("target_speaker"),
        item.get("target_utt"),
        item.get("target_wav_path"),
        item.get("meta", {}).get("target_speaker") if isinstance(item.get("meta"), dict) else None,
    ]
    return any(wanted in str(value).casefold() for value in fields if value)


def _load_items(root: Path, split: str, speaker: str, limit: int | None):
    import torch

    split_dir = root / split
    shards = sorted(split_dir.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"no .pt shards under {split_dir}")
    items: list[dict[str, Any]] = []
    for shard in shards:
        payload = torch.load(shard, map_location="cpu")
        if isinstance(payload, dict):
            payload = payload.get("items", [payload])
        for item in payload:
            if not _target_matches(item, speaker):
                continue
            if not item.get("phone_segments"):
                continue
            items.append(item)
            if limit and len(items) >= limit:
                return items
    return items


def _path(item: dict[str, Any], name: str) -> str:
    value = item.get(name)
    if not value and isinstance(item.get("meta"), dict):
        value = item["meta"].get(name)
    if not value:
        raise KeyError(f"pair item has no {name}")
    return str(value)


def _utterance(item: dict[str, Any], fallback_path: str) -> str:
    for key in ("source_utt", "source_key", "prompt_id"):
        if item.get(key):
            return str(item[key]).replace(":", "__")
        if isinstance(item.get("meta"), dict) and item["meta"].get(key):
            return str(item["meta"][key]).replace(":", "__")
    return Path(fallback_path).stem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs-root", required=True,
                        help="phone-annotated AccentBridge pair root")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--target-speaker", default="ASI")
    parser.add_argument("--reference", required=True,
                        help="fixed ASI reference used for every condition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--min-phone-segments", type=int, default=5)
    parser.add_argument(
        "--audio-search-root",
        action="append",
        default=None,
        help="fallback root for stale pair-shard audio paths; repeatable (default: data)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch

    from bins.infer_utils import (
        load_pair_as_tensors,
        load_xvc,
        precompute_conditions,
        to_numpy_audio,
    )
    from models.codec.sac.utils import process_audio
    from xvc.data.stream_swap import build_phone_frame_map, resolve_audio_path

    out = Path(args.out)
    if out.exists() and any(out.rglob("*.wav")) and not args.overwrite:
        raise SystemExit(f"[error] output already contains wavs: {out}; pass --overwrite")
    for condition in CONDITIONS:
        (out / condition / "wavs").mkdir(parents=True, exist_ok=True)
        (out / condition / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)

    items = _load_items(
        Path(args.pairs_root), args.split, args.target_speaker, args.max_pairs
    )
    items = [item for item in items
             if len(item.get("phone_segments", [])) >= args.min_phone_segments]
    if not items:
        raise SystemExit("[error] no qualifying phone-annotated target-speaker pairs")
    print(f"[audit] selected {len(items)} {args.target_speaker} pair(s) from {args.split}")
    audio_search_roots = args.audio_search_root or ["data"]

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    _, reference_wav, reference_cond = load_pair_as_tensors(
        args.reference,
        args.reference,
        cfg,
        device,
        int(cfg["latent_hop_length"]),
        True,
    )
    speaker_condition, frame_condition = precompute_conditions(
        model, reference_wav, reference_cond
    )
    sample_rate = int(cfg["sample_rate"])

    @torch.inference_mode()
    def encode(path: str):
        wav_np = process_audio(path, cfg, int(cfg["latent_hop_length"]))
        wav = torch.from_numpy(wav_np).view(1, 1, -1).float().to(device)
        tokens = model.semantic_encoder.extract_and_encode(wav.squeeze(1))["speech_tokens"]
        sem = model.semantic_encoder.embed_ids(tokens)
        sem = model.semantic_adapter(sem.transpose(1, 2))  # B,C,T
        zq = model.acoustic_quantizer(model.acoustic_encoder(wav))[0]  # B,C,T
        common = min(sem.shape[-1], zq.shape[-1])
        return sem[..., :common], zq[..., :common]

    def sample_linear(stream: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        pos = torch.as_tensor(positions, device=stream.device, dtype=torch.float32)
        left = pos.floor().long().clamp(0, stream.shape[-1] - 1)
        right = (left + 1).clamp(0, stream.shape[-1] - 1)
        weight = (pos - left.float()).view(1, 1, -1)
        return stream.index_select(-1, left) * (1.0 - weight) + stream.index_select(-1, right) * weight

    def sample_nearest(stream: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        index = torch.as_tensor(positions, device=stream.device).round().long()
        index = index.clamp(0, stream.shape[-1] - 1)
        return stream.index_select(-1, index)

    @torch.inference_mode()
    def decode(sem: torch.Tensor, zq: torch.Tensor) -> np.ndarray:
        if sem.shape[-1] != zq.shape[-1]:
            raise ValueError(f"stream length mismatch: sem={sem.shape[-1]} zq={zq.shape[-1]}")
        combined = torch.cat([sem.transpose(1, 2), zq.transpose(1, 2)], dim=2)
        hidden = model.prenet(combined.transpose(1, 2), speaker_condition)
        hidden = model.acoustic_converter(hidden, frame_condition, speaker_condition)
        return np.asarray(to_numpy_audio(model.acoustic_decoder(hidden)), dtype=np.float32)

    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
    pair_metadata: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        recorded_source_path = _path(item, "source_wav_path")
        recorded_target_path = _path(item, "target_wav_path")
        try:
            source_resolved, source_remapped = resolve_audio_path(
                recorded_source_path, audio_search_roots
            )
            target_resolved, target_remapped = resolve_audio_path(
                recorded_target_path, audio_search_roots
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(
                f"[error] cannot recover pristine audio for pair {index}: {exc}\n"
                "Restore/upload the phone-aware MFA corpus and rerun with "
                "--audio-search-root data/mfa_hindi_phoneaware_asi/mfa_corpus"
            ) from exc
        source_path = str(source_resolved)
        target_path = str(target_resolved)
        if source_remapped or target_remapped:
            print(
                f"  [path repair] {Path(recorded_source_path).name} -> {source_path}; "
                f"{Path(recorded_target_path).name} -> {target_path}"
            )
        name = _utterance(item, recorded_source_path)
        filename = f"{name}__streamswap.wav"

        native_sem, native_zq = encode(source_path)
        target_sem, target_zq = encode(target_path)
        native_frames = int(native_sem.shape[-1])
        target_frames = int(target_sem.shape[-1])

        stored_source_frames = int(item.get("sem_adapted_src", native_sem).shape[-1])
        stored_target_frames = int(item.get("sem_adapted_tgt", target_sem).shape[-1])
        positions, map_info = build_phone_frame_map(
            native_frames,
            target_frames,
            item["phone_segments"],
            source_annotation_frames=stored_source_frames,
            target_annotation_frames=stored_target_frames,
        )
        np.save(out / "maps" / f"{name}.npy", positions)

        target_sem_mapped = sample_linear(target_sem, positions)
        target_zq_mapped = sample_nearest(target_zq, positions)
        streams = {
            "native_sem__native_zq": (native_sem, native_zq),
            "asi_sem__native_zq": (target_sem_mapped, native_zq),
            "native_sem__asi_zq": (native_sem, target_zq_mapped),
            "asi_sem__asi_zq_mapped": (target_sem_mapped, target_zq_mapped),
            "asi_sem__asi_zq_original": (target_sem, target_zq),
        }

        for condition, (sem, zq) in streams.items():
            wav_path = out / condition / "wavs" / filename
            sf.write(str(wav_path), decode(sem, zq), sample_rate)
            manifests[condition].append(
                {
                    "source_utt": name,
                    "source_wav_path": source_path.replace("\\", "/"),
                    "target_utt": f"{name}__streamswap_ft",
                    "target_wav_path": str(wav_path).replace("\\", "/"),
                    "target_reference_wav_path": str(args.reference).replace("\\", "/"),
                    "audit_condition": condition,
                    "audit_original_target_wav_path": target_path.replace("\\", "/"),
                }
            )

        pair_metadata.append(
            {
                "source_utt": name,
                "source_wav_path": source_path,
                "target_wav_path": target_path,
                "recorded_source_wav_path": recorded_source_path,
                "recorded_target_wav_path": recorded_target_path,
                "audio_path_remapped": bool(source_remapped or target_remapped),
                "native_frames": native_frames,
                "target_frames": target_frames,
                "phone_segments": len(item["phone_segments"]),
                "map": map_info,
            }
        )
        print(f"  {index}/{len(items)} {name}: native={native_frames} target={target_frames} frames")

    for condition, rows in manifests.items():
        manifest_dir = out / condition / "manifests"
        selected_split = args.split
        for split in ("train", "val"):
            with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                if split == selected_split:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

    meta = {
        "schema_version": 1,
        "purpose": "causal source-stream localization; never training data",
        "pairs_root": args.pairs_root,
        "split": args.split,
        "target_speaker": args.target_speaker,
        "reference": args.reference,
        "config": args.config,
        "checkpoint": args.ckpt,
        "n": len(items),
        "mapping": {
            "semantic": "linear sampling of target stream at MFA-phone-derived positions",
            "acoustic_codes": "nearest-frame sampling at the same positions",
            "training_use": "prohibited; diagnostic only",
        },
        "audio_search_roots": audio_search_roots,
        "conditions": CONDITIONS,
        "pairs": pair_metadata,
    }
    with (out / "audit_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"[audit] rendered {len(items)} pair(s) x {len(CONDITIONS)} conditions -> {out}")
    print("[audit] compare asi_sem__asi_zq_mapped with asi_sem__asi_zq_original to measure mapping cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
