#!/usr/bin/env python
"""LoRA hygiene verification — closes the "implementation bug" worry with five
offline checks, no training involved. Exit code 0 = all checks passed.

Checks
------
  A. adapted-layer list   -- every LoRALinear in the loaded model, grouped by
                             host module (acoustic_converter / prenet), with
                             per-host layer counts and adapter parameter totals.
  B. frozen-drift audit   -- state-dict comparison: every NON-LoRA tensor in the
                             trained checkpoint must be BITWISE identical to the
                             stock warm-start (after dtype cast if training ran
                             reduced precision). Any drift = the freeze leaked.
                             Also reports how many lora_B tensors moved off zero
                             (proof the adapters, and only the adapters, trained).
  C. freeze + optimizer   -- runs the REAL trainer functions
     audit                   (freeze_model_parameters, verify_trainable_modules)
                             and builds the optimizer EXACTLY like
                             init_optimizer_and_scheduler does, then asserts:
                             requires_grad set == intended LoRA set (per the
                             config's train_bias policy) == optimizer param set.
  D. merge equivalence    -- per LoRALinear: y_unmerged(x) == y_merged(x) within
                             fp tolerance on random inputs, and unmerge() restores
                             the original weight and output. Since merge() mutates
                             ONLY LoRALinear internals and the network is a fixed
                             composition of modules, per-layer functional equality
                             implies whole-network equality — this is exact, not a
                             sampling heuristic over the composed model.
  E. merged-export audit  -- export_merged_state_dict(): no lora_* key survives,
                             and the exported key set covers the stock checkpoint's
                             generator keys exactly (so scripts/merge_lora.py output
                             loads into the stock architecture with nothing missing).

Usage (container, conda xvc, repo root):
    # against a trained LoRA run:
    python scripts/verify_lora_hygiene.py \
        --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_prenet_r8.yaml \
        --ckpt exp/finetune_crosspair_hindi_latent_400_lora_acoustic_prenet_r8/ckpt/001000.pt \
        --stock ckpts/xvc.pt

    # against the stock checkpoint (checks A/C/D/E; drift audit is then a
    # self-comparison and lora_B being all-zero is reported as NOTE, not FAIL):
    python scripts/verify_lora_hygiene.py \
        --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml \
        --ckpt ckpts/xvc.pt --stock ckpts/xvc.pt

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # repo root -> bins, models, utils

FAILURES = []


def _result(name: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _load_generator_state(path: str):
    import torch
    sd = torch.load(path, map_location="cpu")
    return sd["generator"] if "generator" in sd else sd


# --------------------------------------------------------------------------- #
# A. adapted-layer list
# --------------------------------------------------------------------------- #
def check_adapted_layers(model) -> dict:
    from models.codec.sac.modules.lora import LoRALinear
    by_host = defaultdict(list)
    n_params = 0
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            host = name.split(".", 1)[0]
            by_host[host].append(name)
            n_params += m.lora_A.numel() + m.lora_B.numel()
    print("\n== A. adapted layers ==")
    for host in sorted(by_host):
        print(f"  {host}: {len(by_host[host])} LoRALinear layer(s)")
        for n in by_host[host]:
            m = model.get_submodule(n)
            print(f"    {n}  [{m.in_features}->{m.out_features}, r={m.r}, "
                  f"alpha={m.lora_alpha:g}]")
    total = sum(len(v) for v in by_host.values())
    _result("A adapted-layers", total > 0,
            f"{total} layers across {sorted(by_host)}; {n_params/1e6:.3f}M adapter params")
    return by_host


# --------------------------------------------------------------------------- #
# B. frozen-drift audit (pure state-dict level)
# --------------------------------------------------------------------------- #
def check_frozen_drift(trained_path: str, stock_path: str, tol: float):
    import torch
    from models.codec.sac.modules.lora import is_lora_param_name
    trained = _load_generator_state(trained_path)
    stock = _load_generator_state(stock_path)

    print("\n== B. frozen-drift audit ==")
    lora_keys = [k for k in trained if is_lora_param_name(k)]
    base_keys = [k for k in trained if not is_lora_param_name(k)]

    missing = [k for k in stock if k not in trained]
    extra_base = [k for k in base_keys if k not in stock]
    drifted = []
    max_diff = 0.0
    for k in base_keys:
        if k not in stock:
            continue
        t, s = trained[k], stock[k]
        if t.shape != s.shape:
            drifted.append((k, "SHAPE MISMATCH"))
            continue
        # Frozen params must be bitwise equal to the dtype-cast stock tensor:
        # reduced-precision training stores bf16(stock), which still compares
        # exactly equal after casting. Real drift never survives this test.
        s_cast = s.to(t.dtype)
        diff = (t.float() - s_cast.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > tol:
            drifted.append((k, f"max|d|={diff:.3e}"))

    for k, why in drifted[:10]:
        print(f"  DRIFT {k}: {why}")
    if len(drifted) > 10:
        print(f"  ... (+{len(drifted) - 10} more)")
    _result("B frozen-base unchanged", not drifted and not missing,
            f"{len(base_keys)} base tensors, max|diff|={max_diff:.3e}, "
            f"{len(missing)} missing, {len(extra_base)} extra-base")

    if lora_keys:
        b_keys = [k for k in lora_keys if ".lora_B" in k]
        moved = sum(1 for k in b_keys if trained[k].abs().max().item() > 0)
        norms = sorted(float(trained[k].float().norm()) for k in b_keys)
        detail = (f"{moved}/{len(b_keys)} lora_B tensors nonzero "
                  f"(|B| min={norms[0]:.3e} max={norms[-1]:.3e})")
        if moved == 0:
            print(f"  NOTE: all lora_B are zero — fresh/untrained adapters ({detail})")
        else:
            _result("B adapters trained", True, detail)
    else:
        print("  NOTE: checkpoint contains no lora_* keys (stock checkpoint?)")


# --------------------------------------------------------------------------- #
# C. freeze + optimizer audit (real trainer code paths)
# --------------------------------------------------------------------------- #
def check_freeze_and_optimizer(model, full_cfg):
    import torch.optim as optim
    from omegaconf import OmegaConf
    from models.codec.sac.modules.lora import LoRALinear, is_lora_param_name
    from utils.train_utils import freeze_model_parameters, verify_trainable_modules

    print("\n== C. freeze + optimizer audit ==")
    gen_cfg = full_cfg.model.generator
    audit_cfg = OmegaConf.create({"model": {"generator": gen_cfg}})
    models = {"generator": model}

    freeze_model_parameters(models, audit_cfg)          # the trainer's freeze
    n_trainable = verify_trainable_modules(models, audit_cfg)  # the trainer's gate

    bias_mode = gen_cfg.lora.get("train_bias", "none") if gen_cfg.get("lora") else "none"
    lora_bias_names = set()
    if bias_mode == "lora_only":
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear) and m.bias is not None:
                lora_bias_names.add(f"{name}.bias")

    def intended(name: str) -> bool:
        if is_lora_param_name(name):
            return True
        if bias_mode == "all" and name.endswith(".bias"):
            return True
        return name in lora_bias_names

    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {n for n, _ in model.named_parameters() if intended(n)}
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    for n in unexpected[:10]:
        print(f"  UNEXPECTED trainable: {n}")
    for n in missing[:10]:
        print(f"  MISSING trainable: {n}")
    _result("C requires_grad == intended LoRA set", not unexpected and not missing,
            f"{len(actual)} trainable tensors, {n_trainable/1e6:.3f}M params, "
            f"train_bias={bias_mode!r}")

    # Optimizer built EXACTLY like utils.train_utils.init_optimizer_and_scheduler.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, **gen_cfg["optim_conf"])
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    grad_ids = {id(p) for p in model.parameters() if p.requires_grad}
    frozen_in_opt = sum(1 for p in model.parameters()
                        if not p.requires_grad and id(p) in opt_ids)
    _result("C optimizer sees exactly the trainable set",
            opt_ids == grad_ids and frozen_in_opt == 0,
            f"{len(opt_ids)} tensors in optimizer, lr={gen_cfg['optim_conf']['lr']}")


# --------------------------------------------------------------------------- #
# D. merge equivalence (per layer; composition => whole-network equivalence)
# --------------------------------------------------------------------------- #
def check_merge_equivalence(model, rel_tol: float = 1e-4):
    import torch
    from models.codec.sac.modules.lora import LoRALinear

    print("\n== D. merge equivalence ==")
    torch.manual_seed(1234)
    model.eval()
    worst_merge = worst_restore = 0.0
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        n += 1
        x = torch.randn(4, m.in_features, dtype=m.weight.dtype,
                        device=m.weight.device)
        with torch.inference_mode():
            y0 = m(x)
            w_before = m.weight.detach().clone()
            m.merge()
            y1 = m(x)
            m.unmerge()
            y2 = m(x)
        scale = y0.abs().max().item() + 1e-12
        rel_merge = (y0 - y1).abs().max().item() / scale
        rel_restore = max((y0 - y2).abs().max().item() / scale,
                          (m.weight - w_before).abs().max().item()
                          / (w_before.abs().max().item() + 1e-12))
        worst_merge = max(worst_merge, rel_merge)
        worst_restore = max(worst_restore, rel_restore)
        if rel_merge > rel_tol or rel_restore > rel_tol:
            print(f"  MISMATCH {name}: merge rel={rel_merge:.3e} "
                  f"restore rel={rel_restore:.3e}")
    _result("D merged == unmerged (per layer)", worst_merge <= rel_tol,
            f"{n} layers, worst rel diff {worst_merge:.3e} (tol {rel_tol:g})")
    _result("D unmerge restores original", worst_restore <= rel_tol,
            f"worst rel diff {worst_restore:.3e}")


# --------------------------------------------------------------------------- #
# E. merged-export audit
# --------------------------------------------------------------------------- #
def check_merged_export(model, stock_path: str):
    from models.codec.sac.modules.lora import export_merged_state_dict, is_lora_param_name

    print("\n== E. merged-export audit ==")
    exported = export_merged_state_dict(model)  # merges in place; model is DONE after this
    leftover = [k for k in exported if is_lora_param_name(k)]
    _result("E no lora_* keys in export", not leftover,
            f"{len(exported)} tensors exported")

    stock = _load_generator_state(stock_path)
    missing = [k for k in stock if k not in exported]
    extra = [k for k in exported if k not in stock]
    for k in missing[:10]:
        print(f"  MISSING vs stock: {k}")
    if extra:
        print(f"  NOTE: {len(extra)} exported key(s) absent from stock ckpt "
              f"(model-only buffers are expected): {extra[:5]}")
    _result("E export covers stock architecture keys", not missing,
            f"{len(stock)} stock keys covered, {len(extra)} extra")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a LoRA training config")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint to verify (a trained LoRA run's .pt, or stock)")
    ap.add_argument("--stock", default="ckpts/xvc.pt",
                    help="stock warm-start checkpoint (drift + export reference)")
    ap.add_argument("--drift-tol", type=float, default=0.0,
                    help="max allowed |diff| on frozen tensors after dtype cast "
                         "(default 0.0 = bitwise)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    from bins.infer_utils import load_xvc
    from utils.file import load_config

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    full_cfg = load_config(args.config)

    check_adapted_layers(model)
    check_frozen_drift(args.ckpt, args.stock, args.drift_tol)
    check_freeze_and_optimizer(model, full_cfg)
    check_merge_equivalence(model)
    check_merged_export(model, args.stock)   # mutates the model; keep last

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
