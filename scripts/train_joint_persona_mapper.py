#!/usr/bin/env python
"""Train a source-agnostic joint X-VC target-persona mapper.

X-VC stays frozen. The mapper receives no source-speaker id and edits semantic
and acoustic streams through one shared causal trunk. Real target-persona
streams provide supervision through phone-local differentiable monotonic
alignment; neither side is warped or resampled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def load_items(root: Path):
    import torch

    items = []
    for shard in sorted(root.glob("shard_*.pt")):
        items.extend(torch.load(shard, map_location="cpu"))
    if not items:
        raise ValueError(f"no extracted pair shards under {root}")
    return items


def balanced_batches(items, batch_size: int, rng: random.Random):
    """Yield source-speaker-balanced batches without exposing ids to the model."""
    by_speaker = defaultdict(list)
    for item in items:
        by_speaker[item["meta"]["source_speaker"]].append(item)
    speakers = sorted(by_speaker)
    if not speakers:
        raise ValueError("no source speakers in training items")
    while True:
        batch = []
        offset = rng.randrange(len(speakers))
        for index in range(batch_size):
            speaker = speakers[(offset + index) % len(speakers)]
            batch.append(rng.choice(by_speaker[speaker]))
        yield batch


def collate(items, codebook, device):
    import torch
    import torch.nn.functional as F

    batch = len(items)
    source_t = max(item["source_semantic"].shape[-1] for item in items)
    target_t = max(item["target_semantic"].shape[-1] for item in items)
    sem_dim = items[0]["source_semantic"].shape[0]
    zq_dim = items[0]["source_zq"].shape[0]
    source_sem = torch.zeros(batch, sem_dim, source_t)
    source_zq = torch.zeros(batch, zq_dim, source_t)
    source_codes = torch.zeros(batch, source_t, dtype=torch.long)
    source_mask = torch.zeros(batch, 1, source_t)
    target_sem = torch.zeros(batch, sem_dim, target_t)
    target_zq = torch.zeros(batch, zq_dim, target_t)
    target_codes = torch.zeros(batch, target_t, dtype=torch.long)
    target_mask = torch.zeros(batch, 1, target_t)
    segments = []
    for index, item in enumerate(items):
        source_frames = item["source_semantic"].shape[-1]
        target_frames = item["target_semantic"].shape[-1]
        source_sem[index, :, :source_frames] = item["source_semantic"].float()
        source_zq[index, :, :source_frames] = item["source_zq"].float()
        source_codes[index, :source_frames] = item["source_codes"].long()
        source_mask[index, :, :source_frames] = 1
        target_sem[index, :, :target_frames] = item["target_semantic"].float()
        target_zq[index, :, :target_frames] = item["target_zq"].float()
        target_codes[index, :target_frames] = item["target_codes"].long()
        target_mask[index, :, :target_frames] = 1
        segments.append(item["phone_segments"])
    source_codes = source_codes.to(device)
    target_codes = target_codes.to(device)
    return {
        "source_sem": source_sem.to(device),
        "source_zq": source_zq.to(device),
        "source_code": F.embedding(source_codes, codebook).transpose(1, 2),
        "source_codes": source_codes,
        "source_mask": source_mask.to(device),
        "target_sem": target_sem.to(device),
        "target_zq": target_zq.to(device),
        "target_code": F.embedding(target_codes, codebook).transpose(1, 2),
        "target_codes": target_codes,
        "target_mask": target_mask.to(device),
        "segments": segments,
    }


def masked_mean(value, mask):
    return (value * mask).sum() / (mask.sum() * value.shape[1] + 1e-8)


def evaluate(mapper, items, codebook, device, gamma, code_temperature):
    import torch
    import torch.nn.functional as F

    from xvc.training.monotonic import (
        phonewise_aligned_code_agreement,
        phonewise_discrete_code_loss,
        phonewise_dual_stream_loss,
    )

    rows = []
    mapper.eval()
    with torch.inference_mode():
        for item in items:
            batch = collate([item], codebook, device)
            predicted_sem, predicted_code, sem_delta, code_delta = mapper(
                batch["source_sem"],
                batch["source_zq"],
                batch["source_code"],
                return_deltas=True,
            )
            sem_loss, code_loss, phones = phonewise_dual_stream_loss(
                predicted_sem,
                predicted_code,
                batch["target_sem"],
                batch["target_code"],
                batch["segments"],
                gamma=gamma,
            )
            code_logits = mapper.code_logits(
                predicted_code, codebook, temperature=code_temperature
            )
            discrete_code_loss, discrete_phones = phonewise_discrete_code_loss(
                code_logits,
                batch["target_codes"],
                batch["segments"],
                gamma=gamma,
            )
            if discrete_phones != phones:
                raise RuntimeError("continuous and discrete phone counts differ")
            identity_sem, identity_code = mapper(
                batch["target_sem"], batch["target_zq"], batch["target_code"]
            )
            identity = masked_mean(
                (identity_sem - batch["target_sem"]).pow(2), batch["target_mask"]
            ) + masked_mean(
                1.0
                - F.cosine_similarity(
                    identity_code, batch["target_code"], dim=1
                ).unsqueeze(1),
                batch["target_mask"],
            )
            nearest = mapper.nearest_codes(predicted_code, codebook)
            source_indices = batch["source_codes"]
            changed = ((nearest != source_indices).float() * batch["source_mask"][:, 0]).sum()
            valid = batch["source_mask"].sum()
            agreement = phonewise_aligned_code_agreement(
                nearest,
                source_indices,
                batch["target_codes"],
                batch["source_code"],
                batch["target_code"],
                batch["segments"],
            )
            rows.append(
                {
                    "speaker": item["meta"]["source_speaker"],
                    "semantic": float(sem_loss),
                    "acoustic": float(code_loss),
                    "discrete_code": float(discrete_code_loss),
                    "identity": float(identity),
                    "semantic_delta": float(
                        masked_mean(sem_delta.pow(2), batch["source_mask"]).sqrt()
                    ),
                    "code_delta": float(
                        masked_mean(code_delta.pow(2), batch["source_mask"]).sqrt()
                    ),
                    "code_change_fraction": float(changed / valid.clamp_min(1)),
                    "aligned_source_code_agreement": agreement["source"],
                    "aligned_predicted_code_agreement": agreement["predicted"],
                    "aligned_target_code_gain": agreement["gain"],
                    "phones": phones,
                }
            )
    mapper.train()
    grouped = {"all": rows}
    for speaker in sorted({row["speaker"] for row in rows}):
        grouped[speaker] = [row for row in rows if row["speaker"] == speaker]
    summary = {}
    for name, group in grouped.items():
        summary[name] = {
            key: round(sum(row[key] for row in group) / len(group), 6)
            for key in (
                "semantic",
                "acoustic",
                "discrete_code",
                "identity",
                "semantic_delta",
                "code_delta",
                "code_change_fraction",
                "aligned_source_code_agreement",
                "aligned_predicted_code_agreement",
                "aligned_target_code_gain",
            )
        }
        summary[name]["n"] = len(group)
    return summary


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
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lookahead-frames", type=int, default=4)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda-semantic", type=float, default=1.0)
    parser.add_argument("--lambda-acoustic", type=float, default=1.0)
    parser.add_argument(
        "--lambda-discrete-code",
        type=float,
        default=0.0,
        help="weight for phone-local ASI code-id NLL; zero preserves the original objective",
    )
    parser.add_argument(
        "--code-temperature",
        type=float,
        default=0.1,
        help="temperature for frozen-codebook cosine logits",
    )
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-delta", type=float, default=0.01)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    from bins.infer_utils import load_xvc
    from models.joint_accent_mapper import JointAccentMapper
    from xvc.training.monotonic import (
        phonewise_discrete_code_loss,
        phonewise_dual_stream_loss,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.lambda_discrete_code < 0:
        raise SystemExit("[error] --lambda-discrete-code must be non-negative")
    if args.code_temperature <= 0:
        raise SystemExit("[error] --code-temperature must be positive")
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    train_items = load_items(Path(args.train_dir))
    val_items = load_items(Path(args.val_dir))
    source_speakers = sorted({item["meta"]["source_speaker"] for item in train_items})
    if len(source_speakers) < 2:
        raise SystemExit("[error] require at least two native training speakers")

    xvc_device = device.index if device.type == "cuda" and device.index is not None else 0
    _, xvc_model, _ = load_xvc(args.config, args.ckpt, xvc_device, False)
    quantizer = xvc_model.acoustic_quantizer
    codebook = quantizer.codebook.weight.detach().float().to(device)
    code_dim = int(codebook.shape[1])
    codebook_sha256 = hashlib.sha256(
        codebook.detach().cpu().numpy().tobytes()
    ).hexdigest()
    del xvc_model
    del quantizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mapper_config = {
        "semantic_dim": train_items[0]["source_semantic"].shape[0],
        "acoustic_dim": train_items[0]["source_zq"].shape[0],
        "code_dim": code_dim,
        "hidden": args.hidden,
        "layers": args.layers,
        "lookahead_frames": args.lookahead_frames,
        "input_dropout": args.input_dropout,
    }
    mapper = JointAccentMapper(**mapper_config).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    batch_iterator = balanced_batches(train_items, args.batch, rng)
    history = []
    train_rows = []
    best_score = float("inf")
    print(f"[mapper] {mapper.extra_repr()}", flush=True)
    print(
        f"[data] train={len(train_items)} val={len(val_items)} speakers={source_speakers}",
        flush=True,
    )
    print("[voice] X-VC frozen; target voice remains reference-conditioned", flush=True)
    print(
        f"[objective] continuous_acoustic=1.0 discrete_code={args.lambda_discrete_code} "
        f"temperature={args.code_temperature}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        batch = collate(next(batch_iterator), codebook, device)
        source_sem = batch["source_sem"]
        source_zq = batch["source_zq"]
        if args.feature_noise:
            source_sem = source_sem + args.feature_noise * torch.randn_like(source_sem) * batch["source_mask"]
            source_zq = source_zq + args.feature_noise * torch.randn_like(source_zq) * batch["source_mask"]
        predicted_sem, predicted_code, sem_delta, code_delta = mapper(
            source_sem,
            source_zq,
            batch["source_code"],
            return_deltas=True,
        )
        semantic_loss, acoustic_loss, phones = phonewise_dual_stream_loss(
            predicted_sem,
            predicted_code,
            batch["target_sem"],
            batch["target_code"],
            batch["segments"],
            gamma=args.gamma,
        )
        if args.lambda_discrete_code:
            code_logits = mapper.code_logits(
                predicted_code, codebook, temperature=args.code_temperature
            )
            discrete_code_loss, discrete_phones = phonewise_discrete_code_loss(
                code_logits,
                batch["target_codes"],
                batch["segments"],
                gamma=args.gamma,
            )
            if discrete_phones != phones:
                raise RuntimeError("continuous and discrete phone counts differ")
        else:
            discrete_code_loss = predicted_code.new_zeros(())
        identity_sem, identity_code = mapper(
            batch["target_sem"], batch["target_zq"], batch["target_code"]
        )
        identity_loss = masked_mean(
            (identity_sem - batch["target_sem"]).pow(2), batch["target_mask"]
        ) + masked_mean(
            1.0
            - F.cosine_similarity(
                identity_code, batch["target_code"], dim=1
            ).unsqueeze(1),
            batch["target_mask"],
        )
        smooth_loss = masked_mean(
            (sem_delta[:, :, 1:] - sem_delta[:, :, :-1]).pow(2),
            batch["source_mask"][:, :, 1:],
        ) + masked_mean(
            (code_delta[:, :, 1:] - code_delta[:, :, :-1]).pow(2),
            batch["source_mask"][:, :, 1:],
        )
        delta_loss = masked_mean(
            sem_delta.pow(2), batch["source_mask"]
        ) + masked_mean(code_delta.pow(2), batch["source_mask"])
        loss = (
            args.lambda_semantic * semantic_loss
            + args.lambda_acoustic * acoustic_loss
            + args.lambda_discrete_code * discrete_code_loss
            + args.lambda_identity * identity_loss
            + args.lambda_smooth * smooth_loss
            + args.lambda_delta * delta_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        optimizer.step()
        train_row = {
            "step": step,
            "loss": float(loss.detach()),
            "semantic": float(semantic_loss.detach()),
            "acoustic": float(acoustic_loss.detach()),
            "discrete_code": float(discrete_code_loss.detach()),
            "identity": float(identity_loss.detach()),
            "smooth": float(smooth_loss.detach()),
            "delta": float(delta_loss.detach()),
            "gradient_norm": float(gradient_norm),
            "phones": phones,
        }
        train_rows.append(train_row)
        for key, value in train_row.items():
            if key not in {"step", "phones"}:
                writer.add_scalar(f"train/{key}", value, step)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={train_row['loss']:.4f} "
                f"sem={train_row['semantic']:.4f} acu={train_row['acoustic']:.4f} "
                f"disc={train_row['discrete_code']:.4f} "
                f"id={train_row['identity']:.4f} phones={phones}",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            summary = evaluate(
                mapper,
                val_items,
                codebook,
                device,
                args.gamma,
                args.code_temperature,
            )
            all_metrics = summary["all"]
            worst_speaker = max(
                metrics["semantic"]
                + metrics["acoustic"]
                + args.lambda_discrete_code * metrics["discrete_code"]
                for name, metrics in summary.items()
                if name != "all"
            )
            score = (
                all_metrics["semantic"]
                + all_metrics["acoustic"]
                + args.lambda_discrete_code * all_metrics["discrete_code"]
                + 0.5 * all_metrics["identity"]
                + 0.25 * worst_speaker
            )
            row = {"step": step, "selection_score": score, "speakers": summary}
            history.append(row)
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"val/{key}", value, step)
            payload = {
                "format_version": 1,
                "model": "JointAccentMapper",
                "state_dict": {key: value.detach().cpu() for key, value in mapper.state_dict().items()},
                "config": mapper_config,
                "training": vars(args),
                "target_persona": train_items[0]["meta"]["target_speaker"],
                "source_speakers_seen": source_speakers,
                "source_speaker_conditioning": False,
                "xvc_voice_conditioning_required": True,
                "quantizer_codebook_sha256": codebook_sha256,
                "unseen_source_gate_required": True,
                "step": step,
                "validation": summary,
            }
            torch.save(payload, output / "last.pt")
            if score < best_score:
                best_score = score
                torch.save(payload, output / "best.pt")
            print(
                f"[val] step={step} score={score:.4f} sem={all_metrics['semantic']:.4f} "
                f"acu={all_metrics['acoustic']:.4f} disc={all_metrics['discrete_code']:.4f} "
                f"id={all_metrics['identity']:.4f} "
                f"code_change={all_metrics['code_change_fraction']:.3f} "
                f"target_gain={all_metrics['aligned_target_code_gain']:.4f}",
                flush=True,
            )
            (output / "validation_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
    writer.close()
    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(train_row))
        writer_csv.writeheader()
        writer_csv.writerows(train_rows)
    (output / "run_meta.json").write_text(
        json.dumps(
            {
                "target_persona": train_items[0]["meta"]["target_speaker"],
                "source_speakers_seen": source_speakers,
                "source_speaker_conditioning": False,
                "voice_policy": "frozen X-VC target reference supplies voice; mapper edits content streams only",
                "acoustic_objective": {
                    "continuous_phone_cosine": args.lambda_acoustic,
                    "discrete_target_code_nll": args.lambda_discrete_code,
                    "code_temperature": args.code_temperature,
                    "inference_decision_matched": bool(args.lambda_discrete_code),
                },
                "acceptance_policy": "must pass unseen-source MOS/WER/target-speaker-similarity listening gate",
                "best_selection_score": best_score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] best={output / 'best.pt'} last={output / 'last.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
