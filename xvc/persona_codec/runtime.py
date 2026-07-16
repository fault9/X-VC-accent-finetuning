"""Fail-closed load/save and frozen-renderer glue for persona codec generators.

The persona codec generator replaces the *content* of X-VC's two discrete
streams (WhisperVQ semantic token IDs and acoustic-code IDs) with predicted
target-persona sequences; everything that renders those streams — semantic
codebook, semantic adapter, acoustic codebook/vq2emb, prenet, acoustic
converter, decoder, and the target-reference conditioning — stays the frozen
stock X-VC stack.  With no generator loaded the stream content is passed
through untouched, so stock behaviour is preserved byte-for-byte.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch

SUPPORTED_GENERATORS = {"PersonaCodecTeacher", "PersonaCodecStudent"}


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().float().cpu().numpy().tobytes()
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_codebook(xvc_model) -> torch.Tensor:
    """The frozen WhisperVQ codebook that defines semantic token identity."""
    codebook = getattr(xvc_model.semantic_encoder, "codebook", None)
    if codebook is None:
        raise ValueError("X-VC model has no semantic codebook")
    return codebook.weight


def acoustic_codebook(xvc_model) -> torch.Tensor:
    quantizer = getattr(xvc_model, "acoustic_quantizer", None)
    if quantizer is None:
        raise ValueError("X-VC model has no acoustic_quantizer")
    return quantizer.codebook.weight


def adapter_upsample_ratio(xvc_model, probe_steps: int = 8) -> int:
    """Measure the semantic adapter's actual upsample ratio (fail-closed).

    The acoustic group size the generator predicts per semantic step must
    equal this ratio, or the two rendered streams drift apart on a shared
    timeline.  Probing the live adapter instead of trusting a config value
    keeps the check honest across config or checkpoint changes.
    """
    weight = semantic_codebook(xvc_model)
    probe = weight.new_zeros(1, weight.shape[1], probe_steps)
    with torch.no_grad():
        frames = int(xvc_model.semantic_adapter(probe).shape[-1])
    if frames < probe_steps or frames % probe_steps != 0:
        raise ValueError(
            f"semantic adapter maps {probe_steps} steps to {frames} frames — "
            "not an integer upsample; persona codec grouping is unsound"
        )
    return frames // probe_steps


def live_token_geometry(xvc_model) -> dict[str, int]:
    """Read the served model's token geometry instead of assuming it."""
    sem_codebook = semantic_codebook(xvc_model)
    ac_codebook = acoustic_codebook(xvc_model)
    return {
        "semantic_vocab_size": int(sem_codebook.shape[0]),
        "acoustic_vocab_size": int(ac_codebook.shape[0]),
    }


def geometry_from_xvc(xvc_model) -> dict[str, Any]:
    """Vocabulary sizes and codebook hashes read from a live X-VC model."""
    return {
        **live_token_geometry(xvc_model),
        "semantic_codebook_sha256": tensor_sha256(semantic_codebook(xvc_model)),
        "quantizer_codebook_sha256": tensor_sha256(acoustic_codebook(xvc_model)),
    }


def geometry_from_extraction_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """The geometry the token shards were actually extracted with.

    Pinning the checkpoint to the *extraction-time* codebooks is the correct
    provenance: the trained generator speaks that codebook's token language,
    and the fail-closed loader compares these hashes against whatever X-VC
    model tries to render it later.
    """
    geometry = {
        "semantic_vocab_size": int(meta["semantic_vocab_size"]),
        "acoustic_vocab_size": int(meta["acoustic_vocab_size"]),
        "semantic_codebook_sha256": meta["semantic_codebook_sha256"],
        "quantizer_codebook_sha256": meta["quantizer_codebook_sha256"],
    }
    if not geometry["semantic_codebook_sha256"] or not geometry["quantizer_codebook_sha256"]:
        raise ValueError("extraction meta is missing codebook hashes")
    return geometry


def build_generator_payload(
    *,
    model_name: str,
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    target_persona: str,
    geometry: dict[str, Any],
    group_size: int,
    step: int,
    training_meta: Optional[dict[str, Any]] = None,
    validation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Checkpoint payload with every field the fail-closed loader verifies.

    ``geometry`` comes from :func:`geometry_from_extraction_meta` (preferred:
    pins the codebooks that produced the training tokens) or
    :func:`geometry_from_xvc`.
    """
    if model_name not in SUPPORTED_GENERATORS:
        raise ValueError(f"unsupported generator model: {model_name!r}")
    for key in (
        "semantic_vocab_size",
        "acoustic_vocab_size",
        "semantic_codebook_sha256",
        "quantizer_codebook_sha256",
    ):
        if not geometry.get(key):
            raise ValueError(f"geometry is missing {key}")
    return {
        "format_version": 1,
        "model": model_name,
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "config": dict(config),
        "target_persona": str(target_persona),
        "source_speaker_conditioning": False,
        "xvc_voice_conditioning_required": True,
        "unseen_source_gate_required": True,
        "semantic_vocab_size": int(geometry["semantic_vocab_size"]),
        "acoustic_vocab_size": int(geometry["acoustic_vocab_size"]),
        "acoustic_group_size": int(group_size),
        "semantic_codebook_sha256": geometry["semantic_codebook_sha256"],
        "quantizer_codebook_sha256": geometry["quantizer_codebook_sha256"],
        "waveform_or_feature_warping": False,
        "step": int(step),
        "training_meta": training_meta or {},
        "validation": validation or {},
    }


def load_persona_codec_generator(
    checkpoint: str | Path,
    xvc_model,
    device: torch.device | str,
    *,
    expected_target_persona: Optional[str] = None,
):
    """Load a persona codec generator, refusing every unsafe mismatch.

    Refuses: unknown model class, source-speaker conditioning, checkpoints
    that drop X-VC voice conditioning, token-vocabulary or group-size
    mismatches against the live X-VC model, and semantic/acoustic codebook
    hashes that differ from the codebooks the generator was trained against.
    """
    from models.persona_codec_generator import (
        PersonaCodecStudent,
        PersonaCodecTeacher,
    )

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    payload["_checkpoint_sha256"] = file_sha256(checkpoint)

    model_name = payload.get("model")
    if model_name not in SUPPORTED_GENERATORS:
        raise ValueError(f"unsupported persona codec generator: {model_name!r}")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("persona codec checkpoint is missing config/state_dict")
    if payload.get("source_speaker_conditioning") is not False:
        raise ValueError(
            "runtime refuses a source-speaker-conditioned persona generator"
        )
    if payload.get("xvc_voice_conditioning_required") is not True:
        raise ValueError("checkpoint does not preserve X-VC target-voice conditioning")

    target = payload.get("target_persona")
    if expected_target_persona and (
        str(target).casefold() != expected_target_persona.casefold()
    ):
        raise ValueError(
            f"expected target persona {expected_target_persona!r}, "
            f"checkpoint has {target!r}"
        )

    geometry = live_token_geometry(xvc_model)
    for key in ("semantic_vocab_size", "acoustic_vocab_size"):
        if int(payload.get(key, -1)) != geometry[key]:
            raise ValueError(
                f"{key} mismatch: checkpoint {payload.get(key)!r} "
                f"vs live X-VC {geometry[key]}"
            )
    for key, live in (
        ("semantic_codebook_sha256", tensor_sha256(semantic_codebook(xvc_model))),
        ("quantizer_codebook_sha256", tensor_sha256(acoustic_codebook(xvc_model))),
    ):
        expected = payload.get(key)
        if not expected:
            raise ValueError(f"checkpoint is missing {key}")
        if expected != live:
            raise ValueError(
                f"persona generator and X-VC use different codebooks ({key})"
            )

    cls = PersonaCodecTeacher if model_name == "PersonaCodecTeacher" else PersonaCodecStudent
    generator = cls(**config)
    if int(payload.get("acoustic_group_size", -1)) != int(generator.group_size):
        raise ValueError("checkpoint acoustic_group_size disagrees with its config")
    live_group = adapter_upsample_ratio(xvc_model)
    if int(generator.group_size) != live_group:
        raise ValueError(
            f"checkpoint acoustic_group_size {generator.group_size} disagrees with "
            f"the live semantic adapter upsample ratio {live_group}"
        )
    generator.load_state_dict(state_dict, strict=True)
    generator.to(device).eval()
    generator.payload = payload
    return generator


@torch.inference_mode()
def extract_source_token_streams(xvc_model, source_wav: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encode one waveform into raw discrete token IDs of both streams.

    Unlike ``bins.infer_utils.extract_source_streams`` this keeps the semantic
    stream as 12.5 Hz *token IDs* (pre-adapter) because the generator operates
    on discrete sequences, not on the upsampled continuous embedding.
    """
    feat = xvc_model.semantic_encoder.extract_and_encode(source_wav.squeeze(1))
    semantic_tokens = feat["speech_tokens"]
    acoustic_latent = xvc_model.acoustic_encoder(source_wav)
    quantized = xvc_model.acoustic_quantizer(acoustic_latent)
    acoustic_indices = quantized[1]
    return {
        "semantic_tokens": semantic_tokens,       # [B, T_sem] @ 12.5 Hz
        "acoustic_indices": acoustic_indices,     # [B, T_ac]  @ 50 Hz
    }


@torch.inference_mode()
def embed_token_streams(
    xvc_model,
    semantic_tokens: torch.Tensor,
    acoustic_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Embed discrete token IDs into the exact streams the prenet consumes.

    This is the *only* way predicted sequences enter the renderer: through the
    frozen semantic codebook + adapter and the frozen acoustic codebook
    (``vq2emb`` with its native out-projection).  It is identical to the stock
    embedding path, so passing the unmodified source IDs reproduces the stock
    streams (the identity/bypass guarantee).
    """
    if semantic_tokens.dtype != torch.long:
        semantic_tokens = semantic_tokens.long()
    if acoustic_indices.dtype != torch.long:
        acoustic_indices = acoustic_indices.long()
    semantic = xvc_model.semantic_encoder.embed_ids(semantic_tokens)
    semantic = xvc_model.semantic_adapter(
        semantic.transpose(1, 2)
    ).transpose(1, 2)                                   # [B, 4*T_sem, 1024]
    acoustic_zq = xvc_model.acoustic_quantizer.vq2emb(
        acoustic_indices, out_proj=True
    )                                                   # [B, 1024, T_ac]
    # The two streams must share one timeline: tolerate only the sub-group
    # encoder-edge mismatch (< one semantic step); anything larger means the
    # token geometry is wrong and rendering it would produce silently
    # misaligned audio.
    ratio = round(semantic.shape[1] / semantic_tokens.shape[1])
    if abs(semantic.shape[1] - acoustic_indices.shape[-1]) >= ratio:
        raise ValueError(
            f"semantic ({semantic.shape[1]} frames from "
            f"{semantic_tokens.shape[1]} tokens) and acoustic "
            f"({acoustic_indices.shape[-1]} frames) streams disagree by a full "
            f"semantic step at upsample ratio {ratio}"
        )
    frames = min(semantic.shape[1], acoustic_zq.shape[-1], acoustic_indices.shape[-1])
    return {
        "semantic": semantic[:, :frames],
        "acoustic_zq": acoustic_zq[:, :, :frames],
        "acoustic_indices": acoustic_indices[:, :frames],
    }


@torch.inference_mode()
def render_token_streams(
    xvc_model,
    semantic_tokens: torch.Tensor,
    acoustic_indices: torch.Tensor,
    speaker_condition: torch.Tensor,
    frame_condition: torch.Tensor,
) -> torch.Tensor:
    """Render discrete token streams through the frozen X-VC renderer."""
    from bins.infer_utils import render_source_streams

    streams = embed_token_streams(xvc_model, semantic_tokens, acoustic_indices)
    return render_source_streams(
        xvc_model, streams, speaker_condition, frame_condition
    )
