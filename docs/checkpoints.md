# Checkpoints

## On-disk contract (unchanged, load-bearing)

A training checkpoint `exp/<run>/ckpt/NNNNNN.pt` is a dict keyed by model
name:

```python
{
  "generator":     {<state_dict>},          # always
  "discriminator": {<state_dict>},          # base training only
  "ema_generator": {"ema_model.<key>": …},  # when ema_update: true
}
```

* Every checkpoint has a sibling `NNNNNN.yaml` (the resolved config at save
  time) and a `last.pt` / `last.yaml` symlink pair.
* LoRA runs add `*.lora_A` / `*.lora_B` keys next to the **unchanged** base
  keys (`LoRALinear` subclasses `nn.Linear`, so nothing is renamed).
* Merged exports (`scripts/merge_lora.py`) fold the adapters into the base
  weights and strip `lora_*`: loadable by the released architecture.
* DeepSpeed ZeRO shards are consolidated to fp32 at save time
  (`utils/train_utils.py::save_models`); the file above is what lands on disk.
* The released checkpoint `ckpts/xvc.pt` has `generator` only — an absent
  `discriminator` is tolerated on load; any other absent model is an error.

## Loading paths

| Situation | Mechanism |
|---|---|
| warm start (`--checkpoint` / `checkpoint=`) | `utils/checkpoint.py::load_checkpoint` — strict=False plus a gate: unexpected keys or missing keys inside `trainable_modules` abort (LoRA `lora_*` misses are expected and tolerated) |
| resume (`resume_step=N`) | `resume_checkpoint` from `exp/<run>/ckpt/` (+ optimizer/scheduler/EMA state) |
| inference | `XVC.load_from_checkpoint(config, ckpt, device)` — re-creates the LoRA topology from the config before loading; `ema_load=True` prefers `ema_generator` |
| adapter-only | `xvc.adapters.load_lora_state_dict(model, state_dict)` — loads only `lora_*`, accepts the `{"generator": …}` and `ema_model.`-prefixed layouts |

## Inspection

```bash
python scripts/inspect_checkpoint.py exp/<run>/ckpt/000100.pt
```

Reports layout (`full-model` / `lora-only` / `bare state_dict`), models,
tensor + parameter counts, dtype histogram, LoRA tensor counts, EMA presence,
and whether the file is merged. Backed by
`xvc.training.checkpointing.describe_checkpoint`.

## LoRA-only checkpoints (new, additive format)

```python
from xvc.training.checkpointing import save_lora_only_checkpoint
save_lora_only_checkpoint(model.state_dict(), "adapter.pt",
                          metadata={"experiment": "...", "r": 4})
```

Container: `{"format_version": 1, "lora_only": True, "generator": {lora
tensors}, "metadata": {...}}`. Historical full-model checkpoints are treated
as format 0 and are **not** affected.

## Provenance

Every run directory contains `run_meta.json` (git commit + dirty flag,
checkpoint & manifest sha256s, seed, launcher CLI, pip freeze) written before
training starts, plus the resolved `config.yaml`. Publishing to HF goes
through `scripts/publish_checkpoint.py` (never git).
