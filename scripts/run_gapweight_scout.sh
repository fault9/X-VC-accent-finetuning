#!/bin/bash
# Gap-weight bridge scout: can frame-weighted bridge training transform the
# phones the uniform-loss bridge leaves native (American r, certain vowels)?
# Two arms, one knob each vs the shipped l0plus bridge:
#   g2     : --gap-weight 2.0 (concentrate loss on accent-bearing frames)
#   g2sm02 : gap-weight 2.0 + lambda-smooth 0.1 -> 0.02 (allow fast localized
#            deltas -- taps and burst substitutions are transients)
# Each arm: train bridge (minutes) -> 40-render probe through the SAME latent
# LoRA as l10 -> gate against the l10 baseline (32/40 shifted, 3 indian,
# MOS 3.018 on the shared 40 sources).
# Kill rule: an arm must beat that AND audibly move the r/vowels, else dead.
#   nohup bash scripts/run_gapweight_scout.sh > exp/run_logs/gapweight_scout.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

python scripts/train_accentbridge.py \
  --train-dir data/accentbridge_pairs/train \
  --val-dir data/accentbridge_pairs/val \
  --out exp/accentbridge_asi_g2 \
  --lookahead-frames 4 --hidden 192 --layers 4 --kernel 3 \
  --steps 2000 --batch 8 --lr 0.0003 \
  --lambda-id 0.5 --lambda-smooth 0.1 --gap-weight 2.0 \
  --target-speaker ASI --val-every 200 --seed 1234 --device cuda

python scripts/train_accentbridge.py \
  --train-dir data/accentbridge_pairs/train \
  --val-dir data/accentbridge_pairs/val \
  --out exp/accentbridge_asi_g2sm02 \
  --lookahead-frames 4 --hidden 192 --layers 4 --kernel 3 \
  --steps 2000 --batch 8 --lr 0.0003 \
  --lambda-id 0.5 --lambda-smooth 0.02 --gap-weight 2.0 \
  --target-speaker ASI --val-every 200 --seed 1234 --device cuda

python scripts/make_distill_dataset.py \
  --source-glob 'data/distill_sources_asi/*.wav' \
  --bridge-ckpt exp/accentbridge_asi_g2/bridge.pt --delta-scale 1.0 \
  --config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml \
  --ckpt exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt \
  --reference data/eval_targets/ASI.wav \
  --out data/sd_probe_bg2_l10 --limit 40 --device 0

python scripts/make_distill_dataset.py \
  --source-glob 'data/distill_sources_asi/*.wav' \
  --bridge-ckpt exp/accentbridge_asi_g2sm02/bridge.pt --delta-scale 1.0 \
  --config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml \
  --ckpt exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt \
  --reference data/eval_targets/ASI.wav \
  --out data/sd_probe_bg2sm02_l10 --limit 40 --device 0

python scripts/gate_teacher_renders.py \
  data/stackdistill_hindi_asi_l10/wavs \
  data/sd_probe_bg2_l10/wavs \
  data/sd_probe_bg2sm02_l10/wavs
