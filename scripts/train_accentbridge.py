#!/usr/bin/env python
"""Train the tiny AccentBridge on extracted representation pairs — feature-level
only, minutes not hours, plain torch (no DeepSpeed), CPU-capable.

Objective (residual editing of the post-semantic_adapter stream):
    edited = bridge(sem_src_warped)          # native content, L2 timeline
    L = mse(edited, sem_tgt)                                 [target loss]
      + lambda_id * mse(bridge(sem_tgt), sem_tgt)            [identity anchor:
            an L2 input must pass through unchanged — protects against the
            bridge "accenting" already-accented speech at runtime]
      + lambda_smooth * mean |delta[t] - delta[t-1]|         [temporal smoothness]

Run scripts/analyze_accentbridge_pairs.py FIRST: if the verdict is
ACCENT-INVARIANT there is nothing for this model to learn.

Lookahead is explicit and fixed at train time (--lookahead-frames, 20 ms per
frame at this insertion point); train one model per latency budget:
    --lookahead-frames 0   -> +0 ms
    --lookahead-frames 4   -> +80 ms
    --lookahead-frames 8   -> +160 ms

Smoke run:
    python scripts/train_accentbridge.py \
        --train-dir data/accentbridge_pairs/train \
        --val-dir data/accentbridge_pairs/val \
        --lookahead-frames 0 --steps 200 --limit 64 \
        --out exp/accentbridge_l0_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def load_pairs(pairs_dir: Path, limit=None, target_speaker=None, exclude_sources=None):
    """Load shard items; optionally keep one persona's pairs and/or hold out
    native source speakers (for unseen-source generalization checks)."""
    import torch
    items = []
    for s in sorted(pairs_dir.glob("shard_*.pt")):
        for it in torch.load(s, map_location="cpu"):
            m = it["meta"]
            if target_speaker and target_speaker not in m["target_speaker"]:
                continue
            if exclude_sources and m["source_speaker"] in exclude_sources:
                continue
            items.append(it)
            if limit and len(items) >= limit:
                return items
    return items


def collate(batch_items, device):
    """Pad (C, T) pairs to the batch max T; returns src, tgt, mask (B, 1, T)."""
    import torch
    C = batch_items[0]["sem_adapted_tgt"].shape[0]
    T = max(it["sem_adapted_tgt"].shape[-1] for it in batch_items)
    B = len(batch_items)
    src = torch.zeros(B, C, T)
    tgt = torch.zeros(B, C, T)
    mask = torch.zeros(B, 1, T)
    for i, it in enumerate(batch_items):
        t = it["sem_adapted_tgt"].shape[-1]
        src[i, :, :t] = it["sem_adapted_src_warped"].float()
        tgt[i, :, :t] = it["sem_adapted_tgt"].float()
        mask[i, :, :t] = 1.0
    return src.to(device), tgt.to(device), mask.to(device)


def masked_mse(a, b, mask):
    return (((a - b) ** 2) * mask).sum() / (mask.sum() * a.shape[1] + 1e-8)


def val_metrics(bridge, items, device):
    import torch
    import torch.nn.functional as F
    pre_cos, post_cos, pre_l2, post_l2, id_drift = [], [], [], [], []
    with torch.inference_mode():
        for it in items:
            s = it["sem_adapted_src_warped"].float().unsqueeze(0).to(device)
            t = it["sem_adapted_tgt"].float().unsqueeze(0).to(device)
            e = bridge(s)
            pre_cos.append(F.cosine_similarity(s, t, dim=1).mean().item())
            post_cos.append(F.cosine_similarity(e, t, dim=1).mean().item())
            pre_l2.append((s - t).norm(dim=1).mean().item())
            post_l2.append((e - t).norm(dim=1).mean().item())
            id_drift.append((bridge(t) - t).norm(dim=1).mean().item()
                            / (t.norm(dim=1).mean().item() + 1e-8))
    n = len(items)
    return {
        "pre_cos": round(sum(pre_cos) / n, 4), "post_cos": round(sum(post_cos) / n, 4),
        "pre_l2": round(sum(pre_l2) / n, 4), "post_l2": round(sum(post_l2) / n, 4),
        "gap_closed_l2": round(1 - (sum(post_l2) / max(sum(pre_l2), 1e-8)), 4),
        "identity_drift": round(sum(id_drift) / n, 5),
    }


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
    ap.add_argument("--gap-weight", type=float, default=0.0,
                    help="weight target-loss frames by their src->tgt feature "
                         "gap (0 = uniform): concentrates capacity on "
                         "accent-bearing frames instead of the global offset")
    ap.add_argument("--limit", type=int, default=None, help="cap train pairs (smoke)")
    ap.add_argument("--target-speaker", default=None,
                    help="train a per-persona bridge: keep only pairs whose "
                         "target speaker matches (e.g. TNI or ASI)")
    ap.add_argument("--exclude-sources", default=None,
                    help="comma list of native source speakers to HOLD OUT of "
                         "training (e.g. 'clb' -> validate unseen-source "
                         "generalization on clb pairs)")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import random
    import torch

    from models.accentbridge import AccentBridge

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    excl = set(args.exclude_sources.split(",")) if args.exclude_sources else None
    train_items = load_pairs(Path(args.train_dir), args.limit,
                             args.target_speaker, excl)
    val_items = load_pairs(Path(args.val_dir), None, args.target_speaker)
    if not train_items or not val_items:
        raise SystemExit("[error] empty train or val pair set")
    dim = train_items[0]["sem_adapted_tgt"].shape[0]

    bridge = AccentBridge(dim=dim, hidden=args.hidden, n_layers=args.layers,
                          kernel_size=args.kernel,
                          lookahead_frames=args.lookahead_frames).to(device)
    print(f"[bridge] {bridge.extra_repr()}")
    print(f"[data] train={len(train_items)} val={len(val_items)} dim={dim}")
    print(f"[baseline] {val_metrics(bridge, val_items, device)}  "
          "(zero-init bridge == identity: post must equal pre here)")

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
    best = None
    history = []
    for step in range(1, args.steps + 1):
        batch_items = random.sample(train_items, min(args.batch, len(train_items)))
        src, tgt, mask = collate(batch_items, device)
        edited, delta = bridge(src, return_delta=True)
        if args.gap_weight > 0:
            with torch.no_grad():
                g = (tgt - src).norm(dim=1, keepdim=True) * mask   # (B,1,T)
                g = g / (g.sum() / mask.sum().clamp(min=1.0))      # mean 1 on valid
            w = (1.0 + args.gap_weight * g) * mask
            l_tgt = (((edited - tgt) ** 2) * w).sum() / (w.sum() * src.shape[1] + 1e-8)
        else:
            l_tgt = masked_mse(edited, tgt, mask)
        id_out = bridge(tgt)
        l_id = masked_mse(id_out, tgt, mask)
        l_sm = ((delta[:, :, 1:] - delta[:, :, :-1]).abs() * mask[:, :, 1:]).sum() \
            / (mask[:, :, 1:].sum() * src.shape[1] + 1e-8)
        loss = l_tgt + args.lambda_id * l_id + args.lambda_smooth * l_sm
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.val_every == 0 or step == args.steps:
            m = val_metrics(bridge, val_items, device)
            m.update({"step": step, "loss": round(loss.item(), 5),
                      "l_tgt": round(l_tgt.item(), 5), "l_id": round(l_id.item(), 5)})
            history.append(m)
            print(f"[{step:>5}] loss={m['loss']} post_cos={m['post_cos']} "
                  f"(pre {m['pre_cos']}) gap_closed={m['gap_closed_l2']} "
                  f"id_drift={m['identity_drift']}")
            if best is None or m["post_l2"] < best["post_l2"]:
                best = m
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": bridge.state_dict(),
                            "config": {"dim": dim, "hidden": args.hidden,
                                       "n_layers": args.layers,
                                       "kernel_size": args.kernel,
                                       "lookahead_frames": args.lookahead_frames},
                            "metrics": m}, out / "bridge.pt")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "best": best, "history": history},
                  f, indent=2, default=str)
    print(f"[train] best: {best}")
    print(f"[train] wrote {out}/bridge.pt, train_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
