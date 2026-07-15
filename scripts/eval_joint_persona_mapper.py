#!/usr/bin/env python
"""Render stock X-VC and a joint mapper on source speakers unseen in training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


CONDITIONS = {
    "stock_xvc": "Frozen X-VC with the target-persona reference; no mapper.",
    "joint_persona_mapper": "Same frozen X-VC/reference with the joint mapper.",
}


def speaker_from_name(path: Path) -> str:
    match = re.match(r"([^_]+)_arctic_", path.stem)
    return match.group(1).casefold() if match else path.stem.split("_", 1)[0].casefold()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--mapper-ckpt", required=True)
    parser.add_argument("--config", default="configs/xvc.yaml")
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--require-unseen-source", action="store_true")
    parser.add_argument("--min-unseen-speakers", type=int, default=2)
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="canonical train manifest; evaluation prompt overlap fails closed",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=2400)
    parser.add_argument("--current-ms", type=int, default=120)
    parser.add_argument("--smooth-ms", type=int, default=20)
    parser.add_argument("--future-ms", type=int, default=100)
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch

    from bins.infer_utils import (
        load_pair_as_tensors,
        load_xvc,
        precompute_conditions,
        run_stream_chunk_forward,
        run_streaming,
        to_numpy_audio,
    )
    from xvc.runtime import load_persona_stream_editor

    source_paths = sorted(Path(args.source_dir).glob("*.wav"))
    if args.max_sources:
        source_paths = source_paths[: args.max_sources]
    if not source_paths:
        raise SystemExit(f"[error] no wavs under {args.source_dir}")
    mapper_payload = torch.load(args.mapper_ckpt, map_location="cpu")
    if mapper_payload.get("model") not in {"JointAccentMapper", "PostPrenetAccentMapper"}:
        raise SystemExit("[error] checkpoint is not a supported persona stream editor")
    seen = {str(value).casefold() for value in mapper_payload.get("source_speakers_seen", [])}
    eval_speakers = {speaker_from_name(path) for path in source_paths}
    overlap = seen & eval_speakers
    if args.require_unseen_source and overlap:
        raise SystemExit(
            f"[error] evaluation source speakers were seen in mapper training: {sorted(overlap)}"
        )
    if args.require_unseen_source and len(eval_speakers) < args.min_unseen_speakers:
        raise SystemExit(
            f"[error] only {len(eval_speakers)} unseen evaluation speaker(s); "
            f"require {args.min_unseen_speakers}: {sorted(eval_speakers)}"
        )
    prompt_overlap = []
    if args.training_manifest:
        train_rows = [
            json.loads(line)
            for line in Path(args.training_manifest).read_text(encoding="utf-8").splitlines()
            if line
        ]
        train_prompts = {str(row.get("prompt_id", "")).casefold() for row in train_rows}
        eval_prompts = set()
        for path in source_paths:
            match = re.search(r"(arctic_[ab]\d{4})", path.stem, flags=re.I)
            if match:
                eval_prompts.add(match.group(1).casefold())
        prompt_overlap = sorted(train_prompts & eval_prompts)
        if prompt_overlap:
            raise SystemExit(
                f"[error] unseen-source evaluation overlaps training prompts: {prompt_overlap[:20]}"
            )

    output = Path(args.out)
    if output.exists() and any(output.rglob("*.wav")) and not args.overwrite:
        raise SystemExit(f"[error] output already contains wavs: {output}; pass --overwrite")
    for condition in CONDITIONS:
        (output / condition / "wavs").mkdir(parents=True, exist_ok=True)
        (output / condition / "manifests").mkdir(parents=True, exist_ok=True)

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    editor = load_persona_stream_editor(
        args.mapper_ckpt,
        model,
        device,
        expected_target_persona=mapper_payload.get("target_persona"),
    )
    history_ms = args.chunk_ms - args.current_ms - args.smooth_ms - args.future_ms
    editor.validate_stream_window(history_ms, args.smooth_ms, args.future_ms)
    codebook = model.acoustic_quantizer.codebook.weight.detach().to(device)
    codebook_sha256 = hashlib.sha256(
        codebook.detach().float().cpu().numpy().tobytes()
    ).hexdigest()
    expected_codebook_sha256 = mapper_payload.get("quantizer_codebook_sha256")
    if expected_codebook_sha256 and codebook_sha256 != expected_codebook_sha256:
        raise SystemExit(
            "[error] mapper and renderer use different X-VC acoustic codebooks"
        )
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
    manifests = {condition: [] for condition in CONDITIONS}
    latency_ms = {condition: [] for condition in CONDITIONS}

    @torch.inference_mode()
    def render(path: Path):
        source_wav, _, _ = load_pair_as_tensors(
            str(path),
            args.reference,
            cfg,
            device,
            int(cfg["latent_hop_length"]),
            True,
        )
        if args.streaming:
            kwargs = {
                "sample_rate": int(cfg["sample_rate"]),
                "chunk_ms": args.chunk_ms,
                "current_ms": args.current_ms,
                "smooth_ms": args.smooth_ms,
                "future_ms": args.future_ms,
            }
            stock, stock_latency = run_streaming(
                model, source_wav, speaker_condition, frame_condition,
                stream_editor=None, **kwargs,
            )
            edited, edited_latency = run_streaming(
                model, source_wav, speaker_condition, frame_condition,
                stream_editor=editor, **kwargs,
            )
        else:
            stock_latency = []
            edited_latency = []
            stock = run_stream_chunk_forward(
                model, source_wav, speaker_condition, frame_condition
            )
            edited = run_stream_chunk_forward(
                model, source_wav, speaker_condition, frame_condition,
                stream_editor=editor,
                stream_window={
                    "history_ms": history_ms,
                    "smooth_ms": args.smooth_ms,
                    "future_ms": args.future_ms,
                },
            )
        return (
            np.asarray(to_numpy_audio(stock), dtype=np.float32),
            np.asarray(to_numpy_audio(edited), dtype=np.float32),
            int(cfg["sample_rate"]),
            stock_latency,
            edited_latency,
        )

    for index, source_path in enumerate(source_paths, start=1):
        stock, edited, sample_rate, stock_latency, edited_latency = render(source_path)
        latency_ms["stock_xvc"].extend(stock_latency)
        latency_ms["joint_persona_mapper"].extend(edited_latency)
        filename = f"{source_path.stem}__persona.wav"
        for condition, audio in (
            ("stock_xvc", stock),
            ("joint_persona_mapper", edited),
        ):
            target_path = output / condition / "wavs" / filename
            sf.write(str(target_path), audio, sample_rate)
            manifests[condition].append(
                {
                    "source_utt": source_path.stem,
                    "source_speaker": speaker_from_name(source_path),
                    "source_wav_path": str(source_path).replace("\\", "/"),
                    "target_utt": target_path.stem,
                    "target_wav_path": str(target_path).replace("\\", "/"),
                    "target_reference_wav_path": str(args.reference).replace("\\", "/"),
                    "target_persona": mapper_payload.get("target_persona"),
                    "condition": condition,
                }
            )
        print(f"  {index}/{len(source_paths)} {source_path.name}")

    for condition, rows in manifests.items():
        manifest_root = output / condition / "manifests"
        (manifest_root / "train.jsonl").write_text("", encoding="utf-8")
        with (manifest_root / "val.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    meta = {
        "schema_version": 1,
        "purpose": "unseen-source target-voice-plus-accent evaluation",
        "conditions": CONDITIONS,
        "manifest_condition": "stock_xvc",
        "source_dir": args.source_dir,
        "source_speakers": sorted(eval_speakers),
        "source_speakers_seen_in_training": sorted(seen),
        "unseen_source_overlap": sorted(overlap),
        "training_prompt_overlap": prompt_overlap,
        "target_persona": mapper_payload.get("target_persona"),
        "reference": args.reference,
        "mapper_checkpoint": args.mapper_ckpt,
        "mapper_model": mapper_payload.get("model"),
        "streaming": args.streaming,
        "stream_window_ms": {
            "chunk": args.chunk_ms,
            "history": history_ms,
            "current": args.current_ms,
            "smooth": args.smooth_ms,
            "future": args.future_ms,
        },
        "xvc_config": args.config,
        "xvc_checkpoint": args.ckpt,
        "quantizer_codebook_sha256": codebook_sha256,
        "voice_policy": "identical X-VC target reference and frozen voice stack in both conditions",
        "n": len(source_paths),
        "latency_ms": {
            condition: {
                "n": len(values),
                "mean": round(float(np.mean(values)), 4) if values else None,
                "p50": round(float(np.percentile(values, 50)), 4) if values else None,
                "p95": round(float(np.percentile(values, 95)), 4) if values else None,
            }
            for condition, values in latency_ms.items()
        },
    }
    if args.streaming:
        meta["latency_ms"]["p95_mapper_overhead"] = round(
            meta["latency_ms"]["joint_persona_mapper"]["p95"]
            - meta["latency_ms"]["stock_xvc"]["p95"],
            4,
        )
    (output / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[render] {len(source_paths)} unseen-source clips x 2 conditions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
