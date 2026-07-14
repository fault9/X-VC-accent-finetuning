# Legacy config → compositional config mapping

The compositional system composes an experiment from four groups:

```yaml
defaults:
  - model/xvc_16k        # full X-VC tree (references legacy configs/xvc.yaml)
  - adapter/<placement>  # what is trainable: LoRA placement or full-weight whitelist
  - dataset/<data>       # manifests + pairing behavior (ratios, latent alignment)
  - training/<recipe>    # steps, intervals, batch size, LR, EMA, warmup
  - _self_               # the experiment's own (minimal) overrides
```

Resolution is done by `xvc.utils.config.load_config` (used by
`scripts/train.py`); groups merge in list order, `_self_` last.

**Guarantee:** an experiment file listed under “Exactly mapped” resolves to a
config tree **identical** to its legacy overlay — enforced by
`tests/unit/test_config_composition.py`, which fails if a single key differs
and if any experiment file lacks an equivalence entry.

**Legacy configs are not deleted.** They remain loadable by every historical
command (`bins/train.py -c configs/finetune_...`, `finetune.sh --accent ...`,
saved `exp/<run>/config.yaml` files). New experiments should be written
compositionally.

## Config groups

| Group | File | Content |
|---|---|---|
| model | `model/xvc_16k.yaml` | the entire legacy `configs/xvc.yaml` (referenced, not copied) |
| adapter | `adapter/lora_acoustic.yaml` | LoRA r8/α16 on `acoustic_converter` (attn + both FFNs) |
| adapter | `adapter/lora_acoustic_prenet.yaml` | + prenet `pwconv` |
| adapter | `adapter/lora_distill_stack.yaml` | + AdaLN projections (`attn_norm_x.`, `norm_out.`) + prenet `pwconv` |
| adapter | `adapter/full_acoustic_prenet.yaml` | full-weight `[acoustic_converter, prenet]` whitelist (Option A) |
| dataset | `dataset/crosspair_hindi_latent_400_asi.yaml` | ASI latent-aligned cross-pairs, recon 0.2 |
| dataset | `dataset/distill_hindi_asi.yaml` | teacher-rendered ASI pairs, recon 0.1 |
| dataset | `dataset/selfrecon_hindi.yaml` | L2-ARCTIC Hindi self-reconstruction |
| training | `training/lora_sweep_short.yaml` | batch 4, 1000 steps, ckpt every 100, lr 1e-4 |
| training | `training/distill_short.yaml` | batch 8, 2000 steps, ckpt every 100, lr 1e-4 |
| training | `training/option_a_selfrecon.yaml` | batch 8, 3000 steps, ckpt every 250, lr 1e-5 |

## Exactly mapped (equivalence-tested)

| Legacy config | Experiment | Run |
|---|---|---|
| `finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16.yaml` | `experiment/lora_hindi_asi_r1.yaml` | `python scripts/train.py experiment=lora_hindi_asi_r1` |
| `finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16.yaml` | `experiment/lora_hindi_asi_r2.yaml` | `... experiment=lora_hindi_asi_r2` |
| `finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml` | `experiment/lora_hindi_asi_r4.yaml` | `... experiment=lora_hindi_asi_r4` |
| `finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml` | `experiment/lora_hindi_asi_r4_lr5e-5.yaml` | `... experiment=lora_hindi_asi_r4_lr5e-5` |
| `finetune_crosspair_hindi_latent_400_asi_lora_acoustic_prenet_r4_alpha16_lr5e-5.yaml` | `experiment/lora_hindi_asi_acoustic_prenet_r4_lr5e-5.yaml` | `... experiment=lora_hindi_asi_acoustic_prenet_r4_lr5e-5` |
| `finetune_distill_hindi_asi_lora_r4_alpha16.yaml` | `experiment/distill_hindi_asi_r4.yaml` | `... experiment=distill_hindi_asi_r4` |
| `finetune_hindi.yaml` | `experiment/option_a_hindi.yaml` | `... experiment=option_a_hindi` |

Legacy names also still work directly:
`python scripts/train.py experiment=finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16`
falls back to `configs/<name>.yaml` when no experiment file exists.

## Family recipes (legacy configs not yet given experiment files)

To map another legacy config: copy the nearest experiment file, adjust the
overrides below, and add an entry to `EQUIVALENCE` in
`tests/unit/test_config_composition.py`. The test tells you exactly which keys
still differ.

| Legacy family | Composition | Overrides encoded in the legacy filename |
|---|---|---|
| `finetune_{arabic,spanish,chinese,native,joint}.yaml` | model + `adapter/full_acoustic_prenet` + a `dataset/selfrecon_<group>` clone + `training/option_a_selfrecon` | dataset manifests dir (`data/finetuning_audio/manifests/<group>/`) |
| `finetune_crosspair_hindi_latent_{200,400,full,wide*}[...].yaml` (full-weight `recon*_lr*` / `semadapter` arms) | model + `adapter/full_acoustic_prenet` (+ `semantic_adapter` in `trainable_modules` for `semadapter`) + a `dataset/crosspair_*` clone + training overrides | dataset size/filter → dataset group; `reconNN` → `dataloader.reconstruction_ratio: 0.NN`; `lrX` → `model.generator.optim_conf.lr` |
| `finetune_crosspair_hindi_latent_400_lora_acoustic_r{8,16}.yaml` | like `lora_hindi_asi_r4` but `dataset/crosspair_hindi_latent_400` (two-persona, batch 8) | `lora.r`; `static.batch_size: 8` |
| `finetune_crosspair_hindi_latent_400_lora_acoustic_prenet_r8*.yaml` | + `adapter/lora_acoustic_prenet` | `lora.r`, lr, recon weight |
| `finetune_crosspair_hindi_latent_400{,_asi}_lora*_adaln_r8.yaml` | adapter clone with AdaLN-only / +AdaLN include sets | include list |
| `finetune_distill_hindi_asi_lora_r{1,2,8}.yaml`, `_x15_lora_r8` | like `distill_hindi_asi_r4` | `lora.r`; teacher variant → dataset group clone |
| `finetune_selfdistill_*`, `finetune_stackdistill_*` (l10/l12/l125/l15/union/v4) | like `distill_hindi_asi_r4` with a dataset group per teacher render dir | dataset path; `lora.r` |
| `finetune_asi_recononly_wide_lora_r4_alpha16.yaml` | `adapter/lora_acoustic` + wide recon-only dataset group | dataset path, ratios |
| `finetune_crosspair_hindi{,_dtw,_rubberband*,_sourcewarp_ab,_latent_ab,_latent_plumbing,_noadapter}.yaml` | superseded experiment arms (kept as history; see CHANGES.md) | — treat as legacy-frozen; map only if re-run |

Filename-knob legend: `rN` → `lora.r`; `alphaN` → `lora.alpha`;
`lrX` → `optim_conf.lr`; `reconNN` → `dataloader.reconstruction_ratio: 0.NN`
(the share of clean identity-anchor samples, not a loss weight);
`200/400/full/wide` → dataset size variant; `asi/asionly/hindim` → persona
filter; `latent/dtw/rubberband/sourcewarp` → alignment mode of the dataset
build.

## Verifying a new mapping

```bash
.venv/bin/python -m pytest tests/unit/test_config_composition.py -q
```
