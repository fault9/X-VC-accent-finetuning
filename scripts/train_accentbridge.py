#!/usr/bin/env python
"""Train the small streaming AccentBridge on extracted representation pairs.

The original objective uses DTW-aligned frame MSE. With ``--phone-aware`` the
bridge instead receives pristine native-timeline features, exactly as it will
at inference. MFA supplies interval metadata only. Each matched phone is pooled
into onset/middle/offset means plus a global standard deviation; no waveform or
feature stream is warped or resampled.

Phone-aware objective::

    edited = bridge(native_source)
    L = lambda_phone * (
            mse(phase_delta(edited), phase_delta(target))
          + phone_std_weight * mse(std_delta(edited), std_delta(target)))
      + lambda_id * mse(bridge(target), target)
      + lambda_smooth * temporal_variation(delta)
      + lambda_delta * mean(delta ** 2)

Run ``annotate_accentbridge_phone_supervision.py`` first. It accepts genuine
phone-tier MFA only and never falls back to words.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def load_pairs(pairs_dir: Path, limit=None, target_speaker=None, exclude_sources=None):
    import torch

    items = []
    for shard in sorted(pairs_dir.glob("shard_*.pt")):
        for item in torch.load(shard, map_location="cpu"):
            meta = item["meta"]
            if target_speaker and target_speaker not in meta["target_speaker"]:
                continue
            if exclude_sources and meta["source_speaker"] in exclude_sources:
                continue
            items.append(item)
            if limit and len(items) >= limit:
                return items
    return items


def collate(batch_items, device, phone_aware=False):
    """Pad source and target independently; phone mode preserves native time."""
    import torch

    channels = batch_items[0]["sem_adapted_tgt"].shape[0]
    source_key = "sem_adapted_src" if phone_aware else "sem_adapted_src_warped"
    source_t = max(item[source_key].shape[-1] for item in batch_items)
    target_t = max(item["sem_adapted_tgt"].shape[-1] for item in batch_items)
    batch = len(batch_items)
    source = torch.zeros(batch, channels, source_t)
    target = torch.zeros(batch, channels, target_t)
    source_mask = torch.zeros(batch, 1, source_t)
    target_mask = torch.zeros(batch, 1, target_t)
    segments = []
    for i, item in enumerate(batch_items):
        src = item[source_key].float()
        tgt = item["sem_adapted_tgt"].float()
        source[i, :, :src.shape[-1]] = src
        target[i, :, :tgt.shape[-1]] = tgt
        source_mask[i, :, :src.shape[-1]] = 1.0
        target_mask[i, :, :tgt.shape[-1]] = 1.0
        segments.append(item.get("phone_segments", []))
    return (source.to(device), target.to(device), source_mask.to(device),
            target_mask.to(device), segments)


def masked_mse(a, b, mask):
    return (((a - b) ** 2) * mask).sum() / (mask.sum() * a.shape[1] + 1e-8)


def phase_stats(segment):
    """Return three duration-normalized regional means and whole-phone std."""
    import torch.nn.functional as F

    if segment.shape[-1] >= 3:
        phase = F.adaptive_avg_pool1d(segment.unsqueeze(0), 3)[0]
    else:
        phase = segment.mean(dim=-1, keepdim=True).expand(-1, 3)
    return phase, segment.std(dim=-1, unbiased=False)


def phone_realization_loss(edited, source, target, phone_segments, std_weight):
    """Compare target-relative phone shifts without aligning feature frames."""
    phase_sum = edited.new_zeros(())
    std_sum = edited.new_zeros(())
    weight_sum = 0.0
    n_phones = 0
    for i, segments in enumerate(phone_segments):
        for segment in segments:
            s0, s1 = segment["src"]
            t0, t1 = segment["tgt"]
            if s1 <= s0 or t1 <= t0:
                continue
            src_phase, src_std = phase_stats(source[i, :, s0:s1])
            edit_phase, edit_std = phase_stats(edited[i, :, s0:s1])
            tgt_phase, tgt_std = phase_stats(target[i, :, t0:t1])
            # Express the objective as an edit delta. Algebraically the source
            # cancels for means, but this form makes the intended mechanism
            # explicit and supports separate phase/std diagnostics.
            predicted_phase_delta = edit_phase - src_phase
            target_phase_delta = tgt_phase - src_phase
            predicted_std_delta = edit_std - src_std
            target_std_delta = tgt_std - src_std
            weight = float(segment.get("weight", 1.0))
            phase_sum = phase_sum + weight * (
                predicted_phase_delta - target_phase_delta).pow(2).mean()
            std_sum = std_sum + weight * (
                predicted_std_delta - target_std_delta).pow(2).mean()
            weight_sum += weight
            n_phones += 1
    if n_phones == 0:
        raise RuntimeError("batch contains no usable phone segments")
    phase_loss = phase_sum / max(weight_sum, 1e-8)
    std_loss = std_sum / max(weight_sum, 1e-8)
    return phase_loss + std_weight * std_loss, phase_loss, std_loss, n_phones


def _phone_val_metrics(bridge, items, device, std_weight):
    import torch
    import torch.nn.functional as F

    pre_sq = post_sq = weight_sum = 0.0
    pre_cos, post_cos, id_drift = [], [], []
    n_phones = 0
    with torch.inference_mode():
        for item in items:
            source = item["sem_adapted_src"].float().to(device)
            target = item["sem_adapted_tgt"].float().to(device)
            edited = bridge(source.unsqueeze(0))[0]
            for segment in item["phone_segments"]:
                s0, s1 = segment["src"]
                t0, t1 = segment["tgt"]
                src_phase, src_std = phase_stats(source[:, s0:s1])
                edit_phase, edit_std = phase_stats(edited[:, s0:s1])
                tgt_phase, tgt_std = phase_stats(target[:, t0:t1])
                weight = float(segment.get("weight", 1.0))
                pre = ((src_phase - tgt_phase).pow(2).mean()
                       + std_weight * (src_std - tgt_std).pow(2).mean())
                post = ((edit_phase - tgt_phase).pow(2).mean()
                        + std_weight * (edit_std - tgt_std).pow(2).mean())
                pre_sq += weight * pre.item()
                post_sq += weight * post.item()
                weight_sum += weight
                n_phones += 1
                pre_cos.append(F.cosine_similarity(
                    src_phase.flatten(), tgt_phase.flatten(), dim=0).item())
                post_cos.append(F.cosine_similarity(
                    edit_phase.flatten(), tgt_phase.flatten(), dim=0).item())
            identity = bridge(target.unsqueeze(0))[0]
            id_drift.append((identity - target).norm(dim=0).mean().item()
                            / (target.norm(dim=0).mean().item() + 1e-8))
    pre_mse = pre_sq / max(weight_sum, 1e-8)
    post_mse = post_sq / max(weight_sum, 1e-8)
    return {
        "pre_cos": round(sum(pre_cos) / max(len(pre_cos), 1), 4),
        "post_cos": round(sum(post_cos) / max(len(post_cos), 1), 4),
        "pre_l2": round(math.sqrt(pre_mse), 4),
        "post_l2": round(math.sqrt(post_mse), 4),
        "gap_closed_l2": round(1 - math.sqrt(post_mse) / max(math.sqrt(pre_mse), 1e-8), 4),
        "phone_pre_l2": round(math.sqrt(pre_mse), 4),
        "phone_post_l2": round(math.sqrt(post_mse), 4),
        "phone_gap_closed_l2": round(
            1 - math.sqrt(post_mse) / max(math.sqrt(pre_mse), 1e-8), 4),
        "identity_drift": round(sum(id_drift) / max(len(id_drift), 1), 5),
        "phones_evaluated": n_phones,
    }


def val_metrics(bridge, items, device, phone_aware=False, phone_std_weight=0.25):
    if phone_aware:
        return _phone_val_metrics(bridge, items, device, phone_std_weight)

    import torch
    import torch.nn.functional as F

    pre_cos, post_cos, pre_l2, post_l2, id_drift = [], [], [], [], []
    with torch.inference_mode():
        for item in items:
            source = item["sem_adapted_src_warped"].float().unsqueeze(0).to(device)
            target = item["sem_adapted_tgt"].float().unsqueeze(0).to(device)
            edited = bridge(source)
            pre_cos.append(F.cosine_similarity(source, target, dim=1).mean().item())
            post_cos.append(F.cosine_similarity(edited, target, dim=1).mean().item())
            pre_l2.append((source - target).norm(dim=1).mean().item())
            post_l2.append((edited - target).norm(dim=1).mean().item())
            id_drift.append((bridge(target) - target).norm(dim=1).mean().item()
                            / (target.norm(dim=1).mean().item() + 1e-8))
    n = len(items)
    return {
        "pre_cos": round(sum(pre_cos) / n, 4),
        "post_cos": round(sum(post_cos) / n, 4),
        "pre_l2": round(sum(pre_l2) / n, 4),
        "post_l2": round(sum(post_l2) / n, 4),
        "gap_closed_l2": round(1 - sum(post_l2) / max(sum(pre_l2), 1e-8), 4),
        "identity_drift": round(sum(id_drift) / n, 5),
    }


def _gradient_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            norm = parameter.grad.detach().float().norm(2).item()
            total += norm * norm
    return math.sqrt(total)


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(out, curves, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        steps = [row["step"] for row in curves]
        for key in ("loss", "l_phone", "l_phone_phase", "l_phone_std",
                    "l_id", "l_smooth", "l_delta"):
            if curves and key in curves[0]:
                axes[0].plot(steps, [row[key] for row in curves], label=key)
        axes[0].set(title="Training losses", xlabel="step", ylabel="loss")
        axes[0].legend(fontsize=8)
        if history:
            val_steps = [row["step"] for row in history]
            for key in ("phone_post_l2", "phone_gap_closed_l2", "identity_drift"):
                if key in history[0]:
                    axes[1].plot(val_steps, [row[key] for row in history], marker="o", label=key)
        axes[1].set(title="Validation phone metrics", xlabel="step")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # plotting must not invalidate a completed run
        print(f"[warning] could not write curves.png: {exc}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookahead-frames", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-id", type=float, default=0.5)
    ap.add_argument("--lambda-smooth", type=float, default=0.1)
    ap.add_argument("--lambda-delta", type=float, default=0.01)
    ap.add_argument("--gap-weight", type=float, default=0.0,
                    help="legacy DTW frame-gap weight; incompatible with --phone-aware")
    ap.add_argument("--phone-aware", action="store_true",
                    help="train on pristine native features with phase-pooled phone loss")
    ap.add_argument("--lambda-phone", type=float, default=0.25)
    ap.add_argument("--phone-std-weight", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target-speaker", default=None)
    ap.add_argument("--exclude-sources", default=None)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import random
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from models.accentbridge import AccentBridge

    if args.lambda_phone < 0 or args.phone_std_weight < 0 or args.lambda_delta < 0:
        raise SystemExit("[error] loss weights must be non-negative")
    if args.phone_aware and args.gap_weight > 0:
        raise SystemExit("[error] --phone-aware and --gap-weight are separate ablations")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    exclude = set(args.exclude_sources.split(",")) if args.exclude_sources else None
    train_items = load_pairs(Path(args.train_dir), args.limit, args.target_speaker, exclude)
    val_items = load_pairs(Path(args.val_dir), None, args.target_speaker)
    if not train_items or not val_items:
        raise SystemExit("[error] empty train or val pair set")
    if args.phone_aware:
        for split, items in (("train", train_items), ("val", val_items)):
            missing = [i for i, item in enumerate(items)
                       if "phone_segments" not in item or not item["phone_segments"]]
            if missing:
                raise SystemExit(f"[error] {split} item {missing[0]} lacks phone_segments; "
                                 "run annotate_accentbridge_phone_supervision.py")

    dim = train_items[0]["sem_adapted_tgt"].shape[0]
    bridge = AccentBridge(dim=dim, hidden=args.hidden, n_layers=args.layers,
                          kernel_size=args.kernel,
                          lookahead_frames=args.lookahead_frames).to(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out / "tensorboard"))
    print(f"[bridge] {bridge.extra_repr()}")
    print(f"[data] train={len(train_items)} val={len(val_items)} dim={dim}")
    print(f"[objective] phone_aware={args.phone_aware} lambda_phone={args.lambda_phone} "
          f"std_weight={args.phone_std_weight} lambda_delta={args.lambda_delta}")
    baseline = val_metrics(bridge, val_items, device, args.phone_aware,
                           args.phone_std_weight)
    print(f"[baseline] {baseline} (zero-init bridge == identity)")

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
    best = None
    history, curves = [], []
    zero = torch.zeros((), device=device)

    for step in range(1, args.steps + 1):
        batch_items = random.sample(train_items, min(args.batch, len(train_items)))
        source, target, source_mask, target_mask, segments = collate(
            batch_items, device, args.phone_aware)
        edited, delta = bridge(source, return_delta=True)

        if args.phone_aware:
            l_tgt = zero
            l_phone, l_phase, l_std, n_phones = phone_realization_loss(
                edited, source, target, segments, args.phone_std_weight)
        else:
            if args.gap_weight > 0:
                with torch.no_grad():
                    gap = (target - source).norm(dim=1, keepdim=True) * target_mask
                    gap /= (gap.sum() / target_mask.sum().clamp(min=1.0))
                weight = (1.0 + args.gap_weight * gap) * target_mask
                l_tgt = (((edited - target) ** 2) * weight).sum() \
                    / (weight.sum() * source.shape[1] + 1e-8)
            else:
                l_tgt = masked_mse(edited, target, target_mask)
            l_phone = l_phase = l_std = zero
            n_phones = 0

        identity = bridge(target)
        l_id = masked_mse(identity, target, target_mask)
        l_smooth = ((delta[:, :, 1:] - delta[:, :, :-1]).abs()
                    * source_mask[:, :, 1:]).sum() \
            / (source_mask[:, :, 1:].sum() * source.shape[1] + 1e-8)
        l_delta = (delta.pow(2) * source_mask).sum() \
            / (source_mask.sum() * source.shape[1] + 1e-8)
        loss = (l_tgt + args.lambda_phone * l_phone + args.lambda_id * l_id
                + args.lambda_smooth * l_smooth + args.lambda_delta * l_delta)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = _gradient_norm(bridge.parameters())
        optimizer.step()

        scalar = {
            "step": step, "loss": loss.item(), "l_tgt": l_tgt.item(),
            "l_phone": l_phone.item(), "l_phone_phase": l_phase.item(),
            "l_phone_std": l_std.item(), "l_id": l_id.item(),
            "l_smooth": l_smooth.item(), "l_delta": l_delta.item(),
            "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
            "phones": n_phones,
        }
        for key, value in scalar.items():
            if key not in {"step", "phones"}:
                writer.add_scalar(f"train/{key}", value, step)
        writer.add_scalar("train/phones_per_batch", n_phones, step)
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            curves.append(scalar)

        if step % args.val_every == 0 or step == args.steps:
            metrics = val_metrics(bridge, val_items, device, args.phone_aware,
                                  args.phone_std_weight)
            metrics.update({"step": step, "loss": round(loss.item(), 6),
                            "l_phone": round(l_phone.item(), 6),
                            "l_phone_phase": round(l_phase.item(), 6),
                            "l_phone_std": round(l_std.item(), 6),
                            "l_id": round(l_id.item(), 6),
                            "l_smooth": round(l_smooth.item(), 6),
                            "l_delta": round(l_delta.item(), 6)})
            history.append(metrics)
            for key, value in metrics.items():
                if key != "step" and isinstance(value, (int, float)):
                    writer.add_scalar(f"val/{key}", value, step)
            print(f"[{step:>5}] loss={metrics['loss']} post_l2={metrics['post_l2']} "
                  f"gap_closed={metrics['gap_closed_l2']} "
                  f"id_drift={metrics['identity_drift']}")
            selection_key = "phone_post_l2" if args.phone_aware else "post_l2"
            if best is None or metrics[selection_key] < best[selection_key]:
                best = metrics
                torch.save({
                    "state_dict": bridge.state_dict(),
                    "config": {"dim": dim, "hidden": args.hidden,
                               "n_layers": args.layers,
                               "kernel_size": args.kernel,
                               "lookahead_frames": args.lookahead_frames},
                    "training": {"phone_aware": args.phone_aware,
                                 "lambda_phone": args.lambda_phone,
                                 "phone_std_weight": args.phone_std_weight,
                                 "lambda_id": args.lambda_id,
                                 "lambda_smooth": args.lambda_smooth,
                                 "lambda_delta": args.lambda_delta},
                    "metrics": metrics,
                }, out / "bridge.pt")

    writer.flush()
    writer.close()
    _write_csv(out / "training_curves.csv", curves)
    _write_csv(out / "validation_metrics.csv", history)
    _plot_curves(out, curves, history)
    with (out / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "baseline": baseline, "best": best,
                   "history": history}, handle, indent=2, default=str)
    print(f"[train] best: {best}")
    print(f"[train] wrote {out}/bridge.pt, train_metrics.json, "
          "training_curves.csv, validation_metrics.csv, curves.png, tensorboard/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
