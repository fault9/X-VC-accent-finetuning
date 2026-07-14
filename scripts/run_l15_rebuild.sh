#!/bin/bash
# l15 rebuild: full-depth teacher renders + fresh student (see the config
# header of finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16.yaml).
# Stages: render 221 carriers through the l15 teacher stack (bridge x1.0
# through the latent LoRA at x1.5) -> gate vs the l10 renders -> guarded
# student train + reserved eval.
#   nohup bash scripts/run_l15_rebuild.sh > exp/run_logs/l15_rebuild.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

python scripts/make_distill_dataset.py \
  --source-glob 'data/distill_sources_asi/*.wav' \
  --bridge-ckpt exp/accentbridge_asi_l0plus/bridge.pt --delta-scale 1.0 \
  --config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml \
  --ckpt exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt \
  --lora-scale 1.5 \
  --reference data/eval_targets/ASI.wav \
  --out data/stackdistill_hindi_asi_l15 --device 0

python scripts/gate_teacher_renders.py \
  data/stackdistill_hindi_asi_l10/wavs \
  data/stackdistill_hindi_asi_l15/wavs

mkdir -p exp/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16 exp/run_logs

bash scripts/run_guarded_train_eval.sh \
  --accent stackdistill_hindi_asi_l15 \
  --config configs/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16.yaml \
  --log_dir exp/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16 \
  --source_dir data/eval_sources_reserved \
  --evaluation_plan configs/eval_hindi_native_to_asi.json \
  --validate_min_duration 2.6 \
  --steps 100,200,400,600,800,1000,1500,2000
