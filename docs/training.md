# Training

All training runs on a Linux GPU container (DeepSpeed); nothing here runs on
the curation machine. Commands assume the repo root and the `xvc` conda env.

## The entry point

```bash
python scripts/train.py experiment=lora_hindi_asi_r4
```

`experiment=NAME` resolves `configs/experiment/NAME.yaml` (compositional)
first, then `configs/NAME.yaml` (legacy overlay) — every historical
experiment name works. What it does, in order:

1. resolve the config (recursive `base_config` / `defaults:` composition);
2. **validate** it (`xvc.utils.config.validate_experiment_config`) — bad LoRA
   ranks, empty include lists, placeholder dataset paths, ratio conflicts and
   sample-rate mismatches abort here with all problems listed;
3. write the resolved config to `exp/<name>/composed_config.yaml`;
4. run the cross-pair dataset preflight (`validate=false` to skip);
5. write `exp/<name>/run_meta.json` (git commit, checkpoint + manifest
   sha256s, seed, CLI, pip freeze);
6. exec `torchrun -m bins.train` — the unchanged runtime — with the same
   flags `scripts/finetune.sh` always used.

Useful keys: `checkpoint=` (warm start, default `ckpts/xvc.pt`), `log_dir=`,
`resume_step=N` (resume from `exp/<name>/ckpt/`, no warm start), `seed=`,
`gpus=`, `port=`, `lr=` (override `optim_conf.lr`), `wandb=true`,
`dry_run=true` (stop before torchrun; used by the smoke tests).

## What you should see at startup

The runtime logs, before the first step: the resolved config path, LoRA
injection report (every adapted layer), the freeze verification
(`[freeze] 'generator' verified: trainable submodules == [...]`), and the
parameter statistic (total / trainable / frozen per submodule). If the
trainable set is not exactly what the config requested, training aborts.

## Legacy commands (still supported)

```bash
bash scripts/finetune.sh --accent joint            # accent-group fine-tune
bash scripts/finetune.sh --config configs/finetune_hindi.yaml --lr 3e-5
bash scripts/train.sh                               # upstream base training
bash scripts/run_guarded_train_eval.sh --accent ... --config ...   # guarded runs
```

## Resuming

```bash
python scripts/train.py experiment=<name> resume_step=1500   # or -1 for last
```

Resume loads model/optimizer state from `exp/<name>/ckpt/001500.pt` and the
sibling yaml; no `--checkpoint` warm start is applied.

## Adding an experiment

Compose it (see `configs/LEGACY_MAPPING.md` for the groups):

```yaml
# configs/experiment/my_arm.yaml
defaults:
  - model/xvc_16k
  - adapter/lora_acoustic
  - dataset/crosspair_hindi_latent_400_asi
  - training/lora_sweep_short
  - _self_

model:
  generator:
    lora:
      r: 2
    optim_conf:
      lr: 0.00005
```

Then add an entry to `EQUIVALENCE` in
`tests/unit/test_config_composition.py` if it mirrors a legacy config (the
test enforces that every experiment file is accounted for).

## Reproducing a published experiment

Every run directory contains `config.yaml` (resolved), `run_meta.json`
(provenance) and checkpoints. To re-run:

```bash
python scripts/train.py config=exp/<run>/config.yaml log_dir=exp/<run>_repro \
    checkpoint=ckpts/xvc.pt seed=<seed from run_meta.json>
```
