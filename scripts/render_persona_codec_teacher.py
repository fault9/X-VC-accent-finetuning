#!/usr/bin/env python
"""Render stock X-VC vs the offline persona codec teacher on unseen sources.

Both conditions use the identical frozen X-VC stack and the identical target
persona reference (voice conditioning).  The teacher only replaces the
*content* of the two discrete streams: source semantic tokens are decoded
into predicted ASI-style semantic tokens + grouped acoustic codes, which are
re-embedded through the frozen codebooks and rendered by the stock
prenet/acoustic_converter/decoder.

Output layout matches `scripts/score_xvc_accent_stream_audit.py`, so scoring
and gating reuse the existing evaluation stack unchanged:

    python scripts/score_xvc_accent_stream_audit.py --audit-root <out> ...
    python scripts/gate_persona_codec_teacher.py --summary <scored>/condition_summary.csv ...

Per-stage synchronized timings (token extraction, teacher decode, X-VC
render) are recorded per clip; these are *offline* numbers for the latency
contract report, not the streaming gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from xvc.evaluation import (
    check_training_prompt_overlap,
    check_unseen_sources,
    speaker_from_name,
)

CONDITIONS = {
    "stock_xvc": "Frozen X-VC with the target-persona reference; no generator.",
    "persona_codec_teacher": (
        "Same frozen X-VC/reference; offline teacher replaces both discrete streams."
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--target-persona", required=True,
        help="the persona this evaluation claims to test; the checkpoint's "
             "target_persona must match or loading fails closed",
    )
    parser.add_argument("--config", default="configs/xvc.yaml")
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--min-sources", type=int, default=20,
                        help="fail closed below this many rendered prompts (0 disables)")
    parser.add_argument("--require-unseen-source", action="store_true")
    parser.add_argument("--min-unseen-speakers", type=int, default=2)
    parser.add_argument("--training-manifest", default=None,
                        help="canonical train manifest; prompt overlap fails closed")
    parser.add_argument("--max-len-ratio", type=float, default=1.6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch

    from bins.infer_utils import (
        load_pair_as_tensors,
        load_xvc,
        precompute_conditions,
        run_stream_chunk_forward,
        to_numpy_audio,
    )
    from xvc.persona_codec.runtime import (
        extract_source_token_streams,
        file_sha256,
        load_persona_codec_generator,
        render_token_streams,
    )

    source_paths = sorted(Path(args.source_dir).glob("*.wav"))
    if args.max_sources:
        source_paths = source_paths[: args.max_sources]
    if not source_paths:
        raise SystemExit(f"[error] no wavs under {args.source_dir}")
    if args.min_sources and len(source_paths) < args.min_sources:
        raise SystemExit(
            f"[error] only {len(source_paths)} evaluation sources; the teacher gate "
            f"requires >= {args.min_sources} (pass --min-sources 0 only for smoke runs)"
        )

    payload_preview = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    if payload_preview.get("model") != "PersonaCodecTeacher":
        raise SystemExit("[error] checkpoint is not a PersonaCodecTeacher")
    seen = {str(v).casefold() for v in payload_preview.get("source_speakers_seen", [])}
    eval_speakers, overlap = check_unseen_sources(
        source_paths, seen,
        require_unseen=args.require_unseen_source,
        min_unseen_speakers=args.min_unseen_speakers,
        seen_in="teacher training",
    )
    prompt_overlap = check_training_prompt_overlap(source_paths, args.training_manifest)

    output = Path(args.out)
    if output.exists() and any(output.rglob("*.wav")) and not args.overwrite:
        raise SystemExit(f"[error] output already contains wavs: {output}; pass --overwrite")
    for condition in CONDITIONS:
        (output / condition / "wavs").mkdir(parents=True, exist_ok=True)
        (output / condition / "manifests").mkdir(parents=True, exist_ok=True)

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    # The operator asserts the intended persona; reading the expectation from
    # the checkpoint itself would make the mismatch check vacuous.
    teacher = load_persona_codec_generator(
        args.teacher_ckpt, model, device,
        expected_target_persona=args.target_persona,
    )

    # Fail closed up front on clips longer than the teacher's data-derived
    # position bound, instead of crashing mid-run after wavs were written.
    semantic_hz = float(cfg["sample_rate"]) / float(cfg["latent_hop_length"])
    max_positions = int(teacher.max_source_positions)
    too_long = []
    for path in source_paths:
        info = sf.info(str(path))
        estimated_steps = math.ceil(info.frames / info.samplerate * semantic_hz) + 2
        if estimated_steps > max_positions:
            too_long.append(f"{path.name} (~{estimated_steps} semantic steps)")
    if too_long:
        raise SystemExit(
            f"[error] {len(too_long)} source clip(s) exceed the teacher's "
            f"max_source_positions of {max_positions} semantic steps "
            f"(~{max_positions / semantic_hz:.1f} s): {too_long[:10]}; remove "
            "them or retrain the teacher on longer material"
        )

    _, reference_wav, reference_cond = load_pair_as_tensors(
        args.reference, args.reference, cfg, device, int(cfg["latent_hop_length"]), True
    )
    speaker_condition, frame_condition = precompute_conditions(
        model, reference_wav, reference_cond
    )

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result, (time.perf_counter() - start) * 1000.0

    manifests = {condition: [] for condition in CONDITIONS}
    stage_ms = {"token_extraction": [], "teacher_decode": [], "xvc_render": [], "stock_total": []}
    decode_stats = []

    @torch.inference_mode()
    def render(path: Path):
        source_wav, _, _ = load_pair_as_tensors(
            str(path), args.reference, cfg, device, int(cfg["latent_hop_length"]), True
        )
        stock, stock_ms = timed(
            lambda: run_stream_chunk_forward(model, source_wav, speaker_condition, frame_condition)
        )
        streams, extract_ms = timed(lambda: extract_source_token_streams(model, source_wav))
        source_steps = int(streams["semantic_tokens"].shape[1])
        if source_steps > max_positions:
            raise SystemExit(
                f"[error] {path.name}: {source_steps} semantic steps exceed the "
                f"teacher's max_source_positions {max_positions}"
            )
        decoded, decode_ms = timed(
            lambda: teacher.greedy_decode(
                streams["semantic_tokens"].long(), max_len_ratio=args.max_len_ratio
            )
        )
        sem = decoded["semantic_tokens"]
        if sem.shape[1] < 1:
            raise SystemExit(f"[error] teacher produced an empty sequence for {path.name}")
        groups = decoded["acoustic_groups"]
        ac = groups.reshape(1, -1)
        candidate, render_ms = timed(
            lambda: render_token_streams(model, sem, ac, speaker_condition, frame_condition)
        )
        stage_ms["token_extraction"].append(extract_ms)
        stage_ms["teacher_decode"].append(decode_ms)
        stage_ms["xvc_render"].append(render_ms)
        stage_ms["stock_total"].append(stock_ms)
        decode_stats.append(
            {
                "source": path.name,
                "source_steps": int(streams["semantic_tokens"].shape[1]),
                "decoded_steps": int(sem.shape[1]),
                "stopped_by_eos": bool(decoded["stopped_by_eos"]),
            }
        )
        return (
            np.asarray(to_numpy_audio(stock), dtype=np.float32),
            np.asarray(to_numpy_audio(candidate), dtype=np.float32),
            int(cfg["sample_rate"]),
        )

    for index, source_path in enumerate(source_paths, start=1):
        stock, candidate, sample_rate = render(source_path)
        filename = f"{source_path.stem}__persona.wav"
        for condition, audio in (
            ("stock_xvc", stock),
            ("persona_codec_teacher", candidate),
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
                    "target_persona": teacher.payload.get("target_persona"),
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

    def summarize(values):
        if not values:
            return None
        return {
            "n": len(values),
            "mean": round(float(np.mean(values)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
        }

    meta = {
        "schema_version": 1,
        "purpose": "offline persona-codec-teacher falsification gate (Phase 1)",
        "conditions": CONDITIONS,
        "manifest_condition": "stock_xvc",
        "source_dir": args.source_dir,
        "source_speakers": sorted(eval_speakers),
        "source_speakers_seen_in_training": sorted(seen),
        "unseen_source_overlap": sorted(overlap),
        "training_prompt_overlap": prompt_overlap,
        "target_persona": teacher.payload.get("target_persona"),
        "reference": args.reference,
        "teacher_checkpoint": args.teacher_ckpt,
        "teacher_checkpoint_sha256": file_sha256(args.teacher_ckpt),
        "teacher_step": teacher.payload.get("step"),
        "max_len_ratio": args.max_len_ratio,
        "streaming": False,
        "xvc_config": args.config,
        "xvc_checkpoint": args.ckpt,
        "quantizer_codebook_sha256": teacher.payload.get("quantizer_codebook_sha256"),
        "semantic_codebook_sha256": teacher.payload.get("semantic_codebook_sha256"),
        "voice_policy": (
            "identical X-VC target reference and frozen voice stack in both conditions"
        ),
        "n": len(source_paths),
        "decode_stats": decode_stats,
        "offline_stage_ms": {name: summarize(values) for name, values in stage_ms.items()},
        "latency_note": (
            "offline whole-utterance timings; the streaming latency gate applies to "
            "the Phase 3 student, not this teacher"
        ),
    }
    (output / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[render] {len(source_paths)} clips x {len(CONDITIONS)} conditions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
