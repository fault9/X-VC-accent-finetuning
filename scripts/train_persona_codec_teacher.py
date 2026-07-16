#!/usr/bin/env python
"""Phase 1: train the offline upper-bound persona codec teacher.

Maps native WhisperVQ semantic token sequences to the complete real-ASI
target semantic + grouped acoustic-code sequences with teacher-forced
cross-entropy over discrete token IDs and EOS-terminated variable-length
output.  No continuous L2 objective, no frame-wise resampling, no source
speaker conditioning of any kind.

Checkpoints are written every validation interval.  Validation loss is a
*plumbing* signal only: final checkpoint selection must come from rendered
canaries (`scripts/render_persona_codec_teacher.py` + the scoring/gate
scripts), never from latent loss alone.
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


def load_split(pairs_root: Path, split: str):
    import torch

    path = pairs_root / f"{split}.pt"
    if not path.is_file():
        raise SystemExit(f"[error] missing token shard: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def collate(items, group_size: int, device):
    """Pad a batch of token pairs; True in padding masks marks padding."""
    import torch

    src_lens = [len(item["src_semantic"]) for item in items]
    tgt_lens = [len(item["tgt_semantic"]) for item in items]
    max_src, max_tgt = max(src_lens), max(tgt_lens)
    batch = len(items)
    src = torch.zeros(batch, max_src, dtype=torch.long)
    src_pad = torch.ones(batch, max_src, dtype=torch.bool)
    tgt_sem = torch.zeros(batch, max_tgt, dtype=torch.long)
    tgt_ac = torch.zeros(batch, max_tgt, group_size, dtype=torch.long)
    for row, item in enumerate(items):
        source = item["src_semantic"].long()
        target = item["tgt_semantic"].long()
        groups = item["tgt_acoustic"].long().view(-1, group_size)
        if len(groups) != len(target):
            raise SystemExit(
                f"[error] {item['pair_id']}: {len(groups)} acoustic groups for "
                f"{len(target)} semantic steps"
            )
        src[row, : len(source)] = source
        src_pad[row, : len(source)] = False
        tgt_sem[row, : len(target)] = target
        tgt_ac[row, : len(target)] = groups
    return {
        "src_tokens": src.to(device),
        "src_padding_mask": src_pad.to(device),
        "tgt_sem": tgt_sem.to(device),
        "tgt_ac": tgt_ac.to(device),
        "tgt_lengths": torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="shard dir from extract_persona_codec_tokens")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--acoustic-loss-weight", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--decode-val-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len-ratio", type=float, default=1.6)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from models.persona_codec_generator import PersonaCodecTeacher
    from xvc.persona_codec.runtime import (
        build_generator_payload,
        geometry_from_extraction_meta,
    )
    from xvc.persona_codec.tokens import levenshtein_alignment

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs_root = Path(args.pairs)
    meta = json.loads((pairs_root / "meta.json").read_text(encoding="utf-8"))
    if meta.get("waveform_or_feature_warping") is not False:
        raise SystemExit("[error] token shards do not certify warp-free extraction")
    group_size = int(meta["acoustic_group_size"])

    train_items = load_split(pairs_root, "train")
    val_items = load_split(pairs_root, "val")
    # Fail closed at startup instead of dying at the first validation step.
    for name, items in (("train", train_items), ("val", val_items)):
        if not items:
            raise SystemExit(
                f"[error] {name} shard has zero pairs; validation loss, greedy "
                "decode checks, and checkpoint selection all need a non-empty "
                "prompt-disjoint val split"
            )
    if args.decode_val_items < 1:
        raise SystemExit("[error] --decode-val-items must be >= 1")
    train_prompts = {str(p).casefold() for p in meta["train_prompt_ids"]}
    val_prompts = {str(p).casefold() for p in meta["val_prompt_ids"]}
    overlap = sorted(train_prompts & val_prompts)
    if overlap:
        raise SystemExit(f"[error] train/val prompt overlap: {overlap[:10]}")
    for name, items, prompts in (
        ("train", train_items, train_prompts),
        ("val", val_items, val_prompts),
    ):
        stray = {i["prompt_id"].casefold() for i in items} - prompts
        if stray:
            raise SystemExit(f"[error] {name} shard prompts not in meta: {sorted(stray)[:10]}")

    # The teacher receives no source-speaker identity; speakers are recorded in
    # the payload only so unseen-source evaluation can fail closed.  Val
    # speakers count as seen because the val split drives checkpoint
    # selection (best_val_loss.pt) and model sizing.
    source_speakers = sorted(
        {item["source_speaker"].casefold() for item in train_items + val_items}
    )
    if len(source_speakers) < 2:
        raise SystemExit("[error] need >= 2 native source speakers in training data")

    device = torch.device(args.device)
    # Checkpoints are pinned to the codebooks that produced the training
    # tokens (recorded at extraction time); the fail-closed loader compares
    # these hashes against the live X-VC model at render time.
    geometry = geometry_from_extraction_meta(meta)

    max_src = max(len(i["src_semantic"]) for i in train_items + val_items)
    max_tgt = max(len(i["tgt_semantic"]) for i in train_items + val_items)
    model_config = {
        "semantic_vocab_size": int(meta["semantic_vocab_size"]),
        "acoustic_vocab_size": int(meta["acoustic_vocab_size"]),
        "group_size": group_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_source_positions": max(128, int(max_src * args.max_len_ratio) + 16),
        "max_target_positions": max(128, int(max_tgt * args.max_len_ratio) + 16),
    }
    teacher = PersonaCodecTeacher(**model_config).to(device)
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher] {n_params / 1e6:.2f}M parameters, config: {model_config}")

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr, betas=(0.9, 0.98))

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return max(0.05, 1.0 - (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tb"))
    rng = random.Random(args.seed)

    def batches():
        while True:
            order = list(range(len(train_items)))
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                chunk = [train_items[i] for i in order[start : start + args.batch_size]]
                yield collate(chunk, group_size, device)

    @torch.inference_mode()
    def validate(step: int) -> dict:
        teacher.eval()
        totals = {"loss": 0.0, "sem_loss": 0.0, "ac_loss": 0.0, "sem_accuracy": 0.0, "ac_accuracy": 0.0}
        n_batches = 0
        for start in range(0, len(val_items), args.batch_size):
            batch = collate(val_items[start : start + args.batch_size], group_size, device)
            losses = teacher.compute_loss(
                batch["src_tokens"], batch["src_padding_mask"],
                batch["tgt_sem"], batch["tgt_ac"], batch["tgt_lengths"],
                acoustic_loss_weight=args.acoustic_loss_weight,
            )
            for key in totals:
                totals[key] += float(losses[key])
            n_batches += 1
        metrics = {key: value / n_batches for key, value in totals.items()}

        # Greedy-decode a fixed subset: sequence-level sanity (edit distance to
        # the real ASI target, length behaviour, EOS use) - still not a
        # substitute for rendered canaries.
        edits, ratios, eos_count = [], [], 0
        subset = val_items[: args.decode_val_items]
        for item in subset:
            src = item["src_semantic"].long().view(1, -1).to(device)
            decoded = teacher.greedy_decode(src, max_len_ratio=args.max_len_ratio)
            hyp = decoded["semantic_tokens"][0].tolist()
            ref = item["tgt_semantic"].tolist()
            edits.append(levenshtein_alignment(hyp, ref)["normalized"] if hyp else 1.0)
            ratios.append(len(hyp) / max(len(ref), 1))
            eos_count += int(decoded["stopped_by_eos"])
        metrics.update(
            {
                "decode_edit_norm": sum(edits) / len(edits),
                "decode_len_ratio": sum(ratios) / len(ratios),
                "decode_eos_fraction": eos_count / len(subset),
            }
        )
        for key, value in metrics.items():
            writer.add_scalar(f"val/{key}", value, step)
        teacher.train()
        return metrics

    train_rows, val_rows = [], []
    best_val = float("inf")
    teacher.train()
    iterator = batches()
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        losses = teacher.compute_loss(
            batch["src_tokens"], batch["src_padding_mask"],
            batch["tgt_sem"], batch["tgt_ac"], batch["tgt_lengths"],
            acoustic_loss_weight=args.acoustic_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        row = {
            "step": step,
            "loss": float(losses["loss"]),
            "sem_loss": float(losses["sem_loss"]),
            "ac_loss": float(losses["ac_loss"]),
            "sem_accuracy": float(losses["sem_accuracy"]),
            "ac_accuracy": float(losses["ac_accuracy"]),
            "grad_norm": float(grad_norm),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        train_rows.append(row)
        for key, value in row.items():
            if key != "step":
                writer.add_scalar(f"train/{key}", value, step)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:05d} loss={row['loss']:.4f} sem={row['sem_loss']:.4f} "
                f"ac={row['ac_loss']:.4f} sem_acc={row['sem_accuracy']:.3f} "
                f"ac_acc={row['ac_accuracy']:.3f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(step)
            val_rows.append({"step": step, **metrics})
            print(
                f"[val] step={step} loss={metrics['loss']:.4f} "
                f"decode_edit={metrics['decode_edit_norm']:.4f} "
                f"len_ratio={metrics['decode_len_ratio']:.3f} "
                f"eos={metrics['decode_eos_fraction']:.2f}",
                flush=True,
            )
            payload = build_generator_payload(
                model_name="PersonaCodecTeacher",
                state_dict=teacher.state_dict(),
                config=model_config,
                target_persona=meta["target_persona"],
                geometry=geometry,
                group_size=group_size,
                step=step,
                training_meta={
                    "pairs_root": str(pairs_root).replace("\\", "/"),
                    "pairs_meta": meta,
                    "args": vars(args),
                    "source_speakers_seen": source_speakers,
                    "selection_policy": "rendered canaries decide; val loss is plumbing only",
                },
                validation={"step": step, **metrics},
            )
            # Extra loader-checked provenance for unseen-source evaluation.
            payload["source_speakers_seen"] = source_speakers
            torch.save(payload, output / "last.pt")
            torch.save(payload, output / f"step_{step:06d}.pt")
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                torch.save(payload, output / "best_val_loss.pt")

    writer.close()
    for name, rows in (("train_curve.csv", train_rows), ("val_curve.csv", val_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            table = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            table.writeheader()
            table.writerows(rows)
    print(
        f"[done] checkpoints under {output} - select via rendered canaries "
        f"(render_persona_codec_teacher.py + score + gate), not val loss."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
