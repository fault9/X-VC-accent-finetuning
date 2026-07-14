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
    from models.joint_accent_mapper import JointAccentMapper

    source_paths = sorted(Path(args.source_dir).glob("*.wav"))
    if args.max_sources:
        source_paths = source_paths[: args.max_sources]
    if not source_paths:
        raise SystemExit(f"[error] no wavs under {args.source_dir}")
    mapper_payload = torch.load(args.mapper_ckpt, map_location="cpu")
    if mapper_payload.get("model") != "JointAccentMapper":
        raise SystemExit("[error] checkpoint is not a JointAccentMapper")
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
    mapper = JointAccentMapper(**mapper_payload["config"])
    mapper.load_state_dict(mapper_payload["state_dict"])
    mapper.to(device).eval()
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
        tokens = model.semantic_encoder.extract_and_encode(
            source_wav.squeeze(1)
        )["speech_tokens"]
        semantic = model.semantic_adapter(
            model.semantic_encoder.embed_ids(tokens).transpose(1, 2)
        )
        acoustic_outputs = model.acoustic_quantizer(
            model.acoustic_encoder(source_wav)
        )
        source_zq, source_codes = acoustic_outputs[0], acoustic_outputs[1]
        frames = min(semantic.shape[-1], source_zq.shape[-1], source_codes.shape[-1])
        semantic = semantic[:, :, :frames]
        source_zq = source_zq[:, :, :frames]
        source_codes = source_codes[:, :frames]
        source_code = model.acoustic_quantizer.embed_code(source_codes).transpose(1, 2)
        edited_semantic, edited_code = mapper(semantic, source_zq, source_code)
        edited_indices = mapper.nearest_codes(edited_code, codebook)
        edited_zq = model.acoustic_quantizer.vq2emb(edited_indices, out_proj=True)

        def decode(sem, zq):
            combined = torch.cat([sem.transpose(1, 2), zq.transpose(1, 2)], dim=2)
            hidden = model.prenet(combined.transpose(1, 2), speaker_condition)
            hidden = model.acoustic_converter(
                hidden, frame_condition, speaker_condition
            )
            return np.asarray(
                to_numpy_audio(model.acoustic_decoder(hidden)), dtype=np.float32
            )

        return decode(semantic, source_zq), decode(edited_semantic, edited_zq), int(cfg["sample_rate"])

    for index, source_path in enumerate(source_paths, start=1):
        stock, edited, sample_rate = render(source_path)
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
        "xvc_config": args.config,
        "xvc_checkpoint": args.ckpt,
        "quantizer_codebook_sha256": codebook_sha256,
        "voice_policy": "identical X-VC target reference and frozen voice stack in both conditions",
        "n": len(source_paths),
    }
    (output / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[render] {len(source_paths)} unseen-source clips x 2 conditions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
