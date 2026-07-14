#!/bin/bash
# l15 rebuild: full-depth teacher renders + fresh student (see the config
# header of finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16.yaml).
# Stages: render all carriers through the l15 teacher stack (bridge x1.0
# through the latent LoRA at x1.5) -> score/filter + FAIL-CLOSED gate against
# the l10 renders -> guarded student train on accepted renders + reserved eval.
#   nohup bash scripts/run_l15_rebuild.sh > exp/run_logs/l15_rebuild.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

mkdir -p exp/run_logs exp/teacher_gates/l15_rebuild

echo "=== L15 REBUILD START $(date) ==="
echo "The student MOS/WER outcome is a hypothesis; only the teacher gate is pre-registered."

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
  data/stackdistill_hindi_asi_l15/wavs \
  --candidate-root data/stackdistill_hindi_asi_l15 \
  --filtered-root data/stackdistill_hindi_asi_l15_filtered \
  --baseline-dir data/stackdistill_hindi_asi_l10/wavs \
  --candidate-dir data/stackdistill_hindi_asi_l15/wavs \
  --reference data/eval_targets/ASI.wav \
  --config configs/xvc.yaml \
  --ckpt ckpts/xvc.pt \
  --out exp/teacher_gates/l15_rebuild \
  --limit 0 \
  --clip-mos-min 2.8 \
  --clip-wer-max 0.10 \
  --clip-sim-min 0.65 \
  --aggregate-mos-min 2.8 \
  --aggregate-wer-max 0.06 \
  --aggregate-sim-min 0.65 \
  --min-accent-depth-gain 0.05 \
  --min-retained-count 100 \
  --min-retained-fraction 0.50 \
  --min-retained-val 5

mkdir -p exp/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16 exp/run_logs

bash scripts/run_guarded_train_eval.sh \
  --accent stackdistill_hindi_asi_l15_filtered \
  --config configs/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16.yaml \
  --log_dir exp/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16 \
  --source_dir data/eval_sources_reserved \
  --evaluation_plan configs/eval_hindi_native_to_asi.json \
  --validate_min_duration 2.6 \
  --steps 100,200,400,600,800,1000,1500,2000

echo "=== L15 REBUILD DONE $(date) ==="
