"""Load a frozen, source-agnostic accent+voice stream editor safely."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from models.joint_accent_mapper import JointAccentMapper, PostPrenetAccentMapper


def _codebook_sha256(codebook: torch.Tensor) -> str:
    return hashlib.sha256(
        codebook.detach().float().cpu().numpy().tobytes()
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PersonaStreamEditor(nn.Module):
    """Stage-aware adapter consumed by the shared X-VC inference functions."""

    def __init__(self, mapper: nn.Module, stage: str, payload: dict[str, Any]):
        super().__init__()
        if stage not in {"pre_prenet", "post_prenet"}:
            raise ValueError(f"unsupported persona-editor stage: {stage}")
        self.mapper = mapper
        self.stage = stage
        self.payload = payload

    @property
    def target_persona(self) -> Optional[str]:
        value = self.payload.get("target_persona")
        return None if value is None else str(value)

    @property
    def lookahead_ms(self) -> int:
        return int(self.mapper.lookahead_ms)

    def validate_stream_window(
        self, history_ms: int, smooth_ms: int, future_ms: int
    ) -> None:
        self.mapper.validate_stream_window(history_ms, smooth_ms, future_ms)

    def edit_pre_prenet(
        self,
        semantic: torch.Tensor,
        acoustic_zq: torch.Tensor,
        acoustic_indices: torch.Tensor,
        acoustic_quantizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stage != "pre_prenet":
            return semantic, acoustic_zq
        code = acoustic_quantizer.embed_code(acoustic_indices).transpose(1, 2)
        edited_semantic, edited_code = self.mapper(
            semantic.transpose(1, 2), acoustic_zq, code
        )
        # Use the quantizer's native nearest-neighbour and projection path so
        # training/evaluation/runtime cannot silently disagree on code geometry.
        _, edited_indices, _ = acoustic_quantizer.decode_latents(
            edited_code
        )
        edited_zq = acoustic_quantizer.vq2emb(edited_indices, out_proj=True)
        return edited_semantic.transpose(1, 2), edited_zq

    def edit_post_prenet(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mapper(hidden) if self.stage == "post_prenet" else hidden


def load_persona_stream_editor(
    checkpoint: str | Path,
    xvc_model,
    device: torch.device | str,
    *,
    expected_target_persona: Optional[str] = None,
) -> PersonaStreamEditor:
    """Load an editor and fail closed on model/persona/codebook mismatch."""
    payload = torch.load(str(checkpoint), map_location="cpu")
    payload["_checkpoint_sha256"] = _file_sha256(checkpoint)
    model_name = payload.get("model")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("persona-editor checkpoint is missing config/state_dict")

    if model_name == "JointAccentMapper":
        mapper = JointAccentMapper(**config)
        stage = "pre_prenet"
        quantizer = getattr(xvc_model, "acoustic_quantizer", None)
        if quantizer is None:
            raise ValueError("pre-prenet editor requires X-VC acoustic_quantizer")
        expected_sha = payload.get("quantizer_codebook_sha256")
        actual_sha = _codebook_sha256(quantizer.codebook.weight)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError("persona editor and X-VC use different codebooks")
    elif model_name == "PostPrenetAccentMapper":
        mapper = PostPrenetAccentMapper(**config)
        stage = "post_prenet"
    else:
        raise ValueError(f"unsupported persona-editor model: {model_name!r}")

    target = payload.get("target_persona")
    if expected_target_persona and str(target).casefold() != expected_target_persona.casefold():
        raise ValueError(
            f"expected target persona {expected_target_persona!r}, checkpoint has {target!r}"
        )
    if payload.get("source_speaker_conditioning") not in {False, None}:
        raise ValueError("runtime refuses a source-speaker-conditioned persona mapper")
    if payload.get("xvc_voice_conditioning_required") is False:
        raise ValueError("checkpoint does not preserve X-VC target-voice conditioning")

    mapper.load_state_dict(state_dict, strict=True)
    mapper.to(device).eval()
    editor = PersonaStreamEditor(mapper, stage, payload)
    editor.to(device).eval()
    return editor
