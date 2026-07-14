#!/usr/bin/env python3
"""Primary X-VC inference entry point (single source -> target conversion).

Usage (repo root):

    python scripts/infer.py \
        checkpoint=exp/<run>/ckpt/000100.pt \
        source=examples/source.wav \
        target=examples/target.wav \
        output=outputs/converted.wav

Keys:
    checkpoint  required. Training checkpoint or merged export.
    source      required. Source (content) waveform.
    target      required. Target-speaker reference waveform.
    output      required. Output wav path (written exactly there).
    config      optional. Model config; default: the run's own resolved
                config next to the checkpoint (exp/<run>/config.yaml),
                falling back to configs/xvc.yaml for stock checkpoints.
                LoRA checkpoints NEED their training config so the adapter
                topology is re-created before loading.
    device      optional GPU index (default 0; CPU if CUDA unavailable).
    ema         optional bool: prefer EMA weights when present (default false).
    mask_target_condition  optional bool (default false).
    current/chunk/future/smooth  optional streaming knobs in ms; current=0
                (default) is offline conversion, current>0 streams.

Wraps the same core as ``bins/infer_single.py`` (which remains available);
this entry point differs only in taking an explicit output path.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvc.utils.cli import as_bool, parse_key_value_args

KNOWN_KEYS = {
    "checkpoint", "source", "target", "output", "config", "device", "ema",
    "mask_target_condition", "current", "chunk", "future", "smooth",
}


def _default_config_for(checkpoint: Path) -> Path:
    """The run's resolved config (exp/<run>/ckpt/x.pt -> exp/<run>/config.yaml),
    else the stock config."""
    run_config = checkpoint.parent.parent / "config.yaml"
    if run_config.is_file():
        return run_config
    return REPO_ROOT / "configs" / "xvc.yaml"


def main(argv=None) -> int:
    overrides, passthrough = parse_key_value_args(
        sys.argv[1:] if argv is None else argv
    )
    unknown = set(overrides) - KNOWN_KEYS
    if unknown or passthrough:
        raise SystemExit(
            f"unknown arguments: {sorted(unknown) + passthrough}; "
            f"known keys: {sorted(KNOWN_KEYS)}"
        )
    missing = [k for k in ("checkpoint", "source", "target", "output")
               if k not in overrides]
    if missing:
        raise SystemExit(
            f"missing required key(s): {missing}\n"
            f"example: python scripts/infer.py checkpoint=ckpts/xvc.pt "
            f"source=examples/source.wav target=examples/target.wav "
            f"output=outputs/converted.wav"
        )

    checkpoint = Path(overrides["checkpoint"])
    source = Path(overrides["source"])
    target = Path(overrides["target"])
    output = Path(overrides["output"])
    for name, path in (("checkpoint", checkpoint), ("source", source),
                       ("target", target)):
        if not path.is_file():
            raise SystemExit(f"{name} not found: {path}")
    config = Path(overrides.get("config", _default_config_for(checkpoint)))
    if not config.is_file():
        raise SystemExit(
            f"config not found: {config} — pass config=<path> (a LoRA "
            f"checkpoint needs the training config that declares its adapters)"
        )

    # Heavy imports only after argument errors are ruled out.
    import soundfile as sf

    from bins.infer_utils import (
        load_pair_as_tensors,
        load_xvc,
        precompute_conditions,
        run_offline,
        run_streaming,
        to_numpy_audio,
    )

    device_id = int(overrides.get("device", 0))
    ema_load = as_bool(overrides.get("ema", "false"), "ema")
    mask_target_condition = as_bool(
        overrides.get("mask_target_condition", "false"), "mask_target_condition"
    )
    current_ms = int(overrides.get("current", 0))

    print(f"[infer] config: {config}")
    cfg, model, device = load_xvc(str(config), str(checkpoint), device_id, ema_load)
    source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
        source_wav_path=str(source),
        target_wav_path=str(target),
        cfg=cfg,
        device=device,
        latent_hop_length=int(cfg.get("latent_hop_length", 1280)),
        mask_target_condition=mask_target_condition,
    )

    if current_ms == 0:
        recon = run_offline(model, source_wav, target_wav, target_wav_cond)
    else:
        speaker_condition, frame_condition = precompute_conditions(
            model, target_wav, target_wav_cond
        )
        recon, latency_ms = run_streaming(
            model=model,
            source_wav=source_wav,
            speaker_condition=speaker_condition,
            frame_condition=frame_condition,
            sample_rate=int(cfg["sample_rate"]),
            chunk_ms=int(overrides.get("chunk", 2400)),
            current_ms=current_ms,
            future_ms=int(overrides.get("future", 0)),
            smooth_ms=int(overrides.get("smooth", 0)),
        )
        print(f"[infer] avg chunk latency: {sum(latency_ms) / len(latency_ms):.1f} ms")

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), to_numpy_audio(recon), samplerate=int(cfg["sample_rate"]))
    print(f"[infer] source: {source}")
    print(f"[infer] target: {target}")
    print(f"[infer] saved:  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
