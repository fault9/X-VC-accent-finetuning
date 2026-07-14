# Migration guide: old vs new commands

The 2026-07 refactor added consolidated entry points, a library package
(`xvc/`), compositional configs, and tests — **without changing training or
inference behavior**. Every historical command still works; this table shows
the preferred new form.

## Commands

| Task | Old | New |
|---|---|---|
| fine-tune / LoRA train | `bash scripts/finetune.sh --accent <name>` or `--config configs/finetune_<...>.yaml` | `python scripts/train.py experiment=<name>` |
| base X-VC training | `bash scripts/train.sh` | unchanged (still the multi-GPU launcher) |
| resume | `finetune.sh --accent <name> --resume_step 1500` | `python scripts/train.py experiment=<name> resume_step=1500` |
| LR variant | `finetune.sh --lr 5e-5` (writes a patched config copy) | `python scripts/train.py experiment=<name> lr=5e-5` |
| single-pair inference | `bash scripts/infer_single.sh` / `python -m bins.infer_single --config ... --ckpt ... --source_wav_path ... --target_wav_path ... --save_dir ...` | `python scripts/infer.py checkpoint=... source=... target=... output=...` |
| dataset validation | `python scripts/validate_crosspairs.py --data-root <root>` | `python scripts/validate_dataset.py <root>` (same engine; old CLI kept) |
| checkpoint inspection | ad-hoc `torch.load` in a REPL | `python scripts/inspect_checkpoint.py <file.pt>` |
| merge LoRA for serving | `python scripts/merge_lora.py ...` | unchanged |
| LoRA hygiene audit | `python scripts/verify_lora_hygiene.py ...` | unchanged |
| smoke test | `python bins/smoke_test.py --stage all --config ...` | unchanged (plus `pytest tests/` for the CPU suite) |

`scripts/train.py experiment=` accepts **legacy config names** too
(`experiment=finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16`
uses `configs/<name>.yaml` when no `configs/experiment/<name>.yaml` exists).

## Configs

New experiments compose groups (`configs/{model,adapter,dataset,training}/`);
see `configs/LEGACY_MAPPING.md` for the full old→new mapping (7 experiments
mapped exactly with a resolved-tree-equality test; family recipes for the
rest). Legacy overlays are untouched and still load through every historical
path, including saved `exp/<run>/config.yaml` files.

## Imports

| Old | New | Status |
|---|---|---|
| `models.codec.sac.modules.lora` | `xvc.adapters.lora` | old path is a re-export shim (same class objects; `isinstance` safe) |
| `utils.train_utils.freeze_model_parameters` / `verify_trainable_modules` / `params_statistic` | `xvc.adapters` | old names re-exported from `utils.train_utils` |
| — | `xvc.data.schemas`, `xvc.data.validation` | new (validator engine; `scripts/validate_crosspairs.py` wraps it) |
| — | `xvc.utils.config` | new (recursive loader + validation; legacy `utils.file.load_config` unchanged, still one-level) |
| — | `xvc.training.checkpointing` | new (inspection + LoRA-only checkpoints) |

`pip install -e .` installs the `xvc` package only; `models/`, `utils/`,
`bins/` intentionally stay repo-root-relative (saved configs carry
`models.codec.sac.*` `_target_` strings, and generic names like `utils` must
not be installed site-wide). Run entry points from the repo root, as before.

## Behavioral changes (complete list)

1. **`validate_crosspairs.py` QC gating** — a legacy `alignment_qc.jsonl` row
   missing a gated field (e.g. `anchor_removal_fraction`) is now a readable
   validation failure when `align_meta.json` configures the gate, and is
   tolerated when it does not. Previously both cases crashed with a raw
   `KeyError`. Legacy default gates were vacuous, so pass/fail outcomes are
   unchanged for datasets that ever validated successfully. The validator
   also saves `validation_report.json` (with `schema_version`) next to the
   dataset.
2. **`models/base/fsq/residual_fsq.py`** — added a missing
   `from math import ceil` (ruff F821). The module is never imported at
   runtime (the bug proves it); no behavior change.
3. Everything else is additive: new files, re-export shims, and wrappers.

## Not migrated (intentionally)

* The one-off sweep/probe runners (`scripts/run_*.sh`, `scripts/probe_*.sh`)
  are archived experiment drivers; they still run but new sweeps should be
  small experiment configs + `scripts/train.py`.
* Candidate dead code listed in `REFACTOR_PLAN.md` §18 is still in place —
  removal is Phase 6, only after another round of reference checks.
* The dataloader's fixed 2.4 s collation lengths (see `docs/datasets.md`).
