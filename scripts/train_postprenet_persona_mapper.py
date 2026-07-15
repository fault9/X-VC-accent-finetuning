#!/usr/bin/env python
"""Train a causal accent editor on X-VC's unified post-prenet stream.

X-VC and its target-voice conditioning remain frozen. MFA intervals select
phone-local comparison regions only; neither waveform nor representation is
resampled. The mapper never receives a source-speaker identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.train_joint_persona_mapper import (  # noqa: E402
    balanced_batches,
    collate,
    context_valid_items,
    load_items,
    masked_mean,
)


def _post_features(prenet, batch):
    import torch

    with torch.no_grad():
        source = prenet(
            torch.cat([batch["source_sem"], batch["source_zq"]], dim=1), None
        )
        target = prenet(
            torch.cat([batch["target_sem"], batch["target_zq"]], dim=1), None
        )
    return source, target


def evaluate(mapper, prenet, items, codebook, device, gamma):
    import torch

    from xvc.training.monotonic import phonewise_single_stream_loss

    rows = []
    mapper.eval()
    with torch.inference_mode():
        for item in items:
            batch = collate([item], codebook, device)
            source, target = _post_features(prenet, batch)
            predicted, delta = mapper(source, return_delta=True)
            post_loss, phones = phonewise_single_stream_loss(
                predicted, target, batch["segments"], gamma=gamma
            )
            identity = masked_mean(
                (mapper(target) - target).pow(2), batch["target_mask"]
            )
            rows.append(
                {
                    "speaker": item["meta"]["source_speaker"],
                    "post": float(post_loss),
                    "identity": float(identity),
                    "delta": float(
                        masked_mean(delta.pow(2), batch["source_mask"]).sqrt()
                    ),
                    "phones": phones,
                }
            )
    mapper.train()
    grouped = {"all": rows}
    for speaker in sorted({row["speaker"] for row in rows}):
        grouped[speaker] = [row for row in rows if row["speaker"] == speaker]
    return {
        name: {
            **{
                key: round(sum(row[key] for row in group) / len(group), 6)
                for key in ("post", "identity", "delta")
            },
            "n": len(group),
        }
        for name, group in grouped.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--config", default="configs/xvc.yaml")
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--lookahead-frames", type=int, default=4)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-delta", type=float, default=0.01)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-ms", type=int, default=2160)
    parser.add_argument("--smooth-ms", type=int, default=20)
    parser.add_argument("--future-ms", type=int, default=100)
    parser.add_argument("--expected-window-frames", type=int, default=120)
    parser.add_argument("--min-context-phones", type=int, default=1)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from bins.infer_utils import load_xvc
    from models.joint_accent_mapper import PostPrenetAccentMapper
    from xvc.training.monotonic import phonewise_single_stream_loss

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    train_items = load_items(Path(args.train_dir))
    val_items = load_items(Path(args.val_dir))
    source_speakers = sorted(
        {item["meta"]["source_speaker"] for item in train_items}
    )
    if len(source_speakers) < 2:
        raise SystemExit("[error] require at least two native training speakers")

    device_index = device.index if device.type == "cuda" and device.index is not None else 0
    _, xvc_model, _ = load_xvc(args.config, args.ckpt, device_index, False)
    xvc_model.eval()
    for parameter in xvc_model.parameters():
        parameter.requires_grad_(False)
    prenet = xvc_model.prenet
    codebook = xvc_model.acoustic_quantizer.codebook.weight.detach().float().to(device)
    # Keep only the frozen stage needed to construct post-prenet targets. The
    # encoders/converter/decoder would otherwise waste several GB throughout
    # this small-mapper training run.
    xvc_model.prenet = None
    del xvc_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mapper_config = {
        "input_dim": int(train_items[0]["source_semantic"].shape[0]),
        "hidden": args.hidden,
        "layers": args.layers,
        "lookahead_frames": args.lookahead_frames,
        "input_dropout": args.input_dropout,
    }
    mapper = PostPrenetAccentMapper(**mapper_config).to(device)
    mapper.validate_stream_window(args.history_ms, args.smooth_ms, args.future_ms)
    train_items = context_valid_items(
        train_items,
        mapper.receptive_history_frames,
        mapper.lookahead_frames,
        args.expected_window_frames,
        args.min_context_phones,
    )
    val_items = context_valid_items(
        val_items,
        mapper.receptive_history_frames,
        mapper.lookahead_frames,
        args.expected_window_frames,
        args.min_context_phones,
    )
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    batches = balanced_batches(train_items, args.batch, rng)
    history = []
    train_rows = []
    best_score = float("inf")
    print(
        f"[mapper] PostPrenetAccentMapper params={mapper.n_params()/1e6:.3f}M "
        f"history={mapper.receptive_history_ms}ms lookahead={mapper.lookahead_ms}ms",
        flush=True,
    )
    print(
        f"[data] train={len(train_items)} val={len(val_items)} speakers={source_speakers}",
        flush=True,
    )
    print(
        f"[window] raw={args.expected_window_frames} frames "
        f"valid=[{mapper.receptive_history_frames}, "
        f"{args.expected_window_frames - mapper.lookahead_frames})",
        flush=True,
    )
    print("[voice] all X-VC weights and reference conditioning remain frozen", flush=True)

    for step in range(1, args.steps + 1):
        batch_items = next(batches)
        batch = collate(batch_items, codebook, device)
        source, target = _post_features(prenet, batch)
        if args.feature_noise:
            source = source + args.feature_noise * torch.randn_like(source) * batch["source_mask"]
        predicted, delta = mapper(source, return_delta=True)
        post_loss, phones = phonewise_single_stream_loss(
            predicted, target, batch["segments"], gamma=args.gamma
        )
        identity_loss = masked_mean(
            (mapper(target) - target).pow(2), batch["target_mask"]
        )
        smooth_loss = masked_mean(
            (delta[:, :, 1:] - delta[:, :, :-1]).pow(2),
            batch["source_mask"][:, :, 1:],
        )
        delta_loss = masked_mean(delta.pow(2), batch["source_mask"])
        loss = (
            post_loss
            + args.lambda_identity * identity_loss
            + args.lambda_smooth * smooth_loss
            + args.lambda_delta * delta_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "post": float(post_loss.detach()),
            "identity": float(identity_loss.detach()),
            "smooth": float(smooth_loss.detach()),
            "delta": float(delta_loss.detach()),
            "gradient_norm": float(gradient_norm),
            "phones": phones,
        }
        train_rows.append(row)
        for key, value in row.items():
            if key not in {"step", "phones"}:
                writer.add_scalar(f"train/{key}", value, step)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:04d} loss={row['loss']:.4f} post={row['post']:.4f} "
                f"id={row['identity']:.4f} phones={phones}",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            summary = evaluate(mapper, prenet, val_items, codebook, device, args.gamma)
            all_metrics = summary["all"]
            worst = max(
                metrics["post"]
                for name, metrics in summary.items()
                if name != "all"
            )
            score = all_metrics["post"] + 0.5 * all_metrics["identity"] + 0.25 * worst
            history.append({"step": step, "selection_score": score, "speakers": summary})
            for key, value in all_metrics.items():
                writer.add_scalar(f"val/{key}", value, step)
            payload = {
                "format_version": 1,
                "model": "PostPrenetAccentMapper",
                "state_dict": {
                    key: value.detach().cpu() for key, value in mapper.state_dict().items()
                },
                "config": mapper_config,
                "training": vars(args),
                "target_persona": train_items[0]["meta"]["target_speaker"],
                "source_speakers_seen": source_speakers,
                "source_speaker_conditioning": False,
                "xvc_voice_conditioning_required": True,
                "unseen_source_gate_required": True,
                "window_policy": {
                    "waveform_crop_before_frozen_encoder": True,
                    "window_frames": args.expected_window_frames,
                    "context_valid_start": mapper.receptive_history_frames,
                    "context_valid_end": args.expected_window_frames
                    - mapper.lookahead_frames,
                    "feature_warping": False,
                },
                "step": step,
                "validation": summary,
            }
            torch.save(payload, output / "last.pt")
            if score < best_score:
                best_score = score
                torch.save(payload, output / "best.pt")
            print(
                f"[val] step={step} score={score:.4f} post={all_metrics['post']:.4f} "
                f"id={all_metrics['identity']:.4f}",
                flush=True,
            )
            (output / "validation_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

    writer.close()
    with (output / "train_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=train_rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(train_rows)
    print(f"[done] best={output / 'best.pt'} score={best_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
