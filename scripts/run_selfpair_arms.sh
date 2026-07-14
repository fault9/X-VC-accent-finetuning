#!/bin/bash
# Sequential runner for the two self-pair arms (see the config headers):
#   1. recon-only baseline  -- the "no pairs at all" falsification arm
#   2. v4 real-anchor student -- l10 distill pairs + real-ASI self-pairs
# Launch from the repo root after git pull:
#   nohup bash scripts/run_selfpair_arms.sh > exp/run_logs/arms_run.out 2>&1 &
set -e
cd "$(dirname "$0")/.."

# Self-contained env activation: launching from (base) must not matter.
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

python scripts/make_selfpair_manifest.py \
  --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/train.jsonl \
  --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/val.jsonl \
  --reference data/eval_targets/ASI.wav \
  --out data/asi_selfpairs_wide

python scripts/make_selfpair_manifest.py \
  --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/train.jsonl \
  --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/val.jsonl \
  --reference data/eval_targets/ASI.wav \
  --limit 70 \
  --out data/asi_selfpairs_wide70

mkdir -p exp/finetune_asi_recononly_wide_lora_r4_alpha16 \
         exp/finetune_stackdistill_hindi_asi_v4_realanchor_lora_r4_alpha16 \
         exp/run_logs

bash scripts/run_guarded_train_eval.sh \
  --accent asi_selfpairs_wide \
  --config configs/finetune_asi_recononly_wide_lora_r4_alpha16.yaml \
  --log_dir exp/finetune_asi_recononly_wide_lora_r4_alpha16 \
  --source_dir data/eval_sources_reserved \
  --evaluation_plan configs/eval_hindi_native_to_asi.json \
  --validate_min_duration 2.0 \
  --steps 100,200,400,600,800,1000

bash scripts/run_guarded_train_eval.sh \
  --accent stackdistill_hindi_asi_l10 \
  --config configs/finetune_stackdistill_hindi_asi_v4_realanchor_lora_r4_alpha16.yaml \
  --log_dir exp/finetune_stackdistill_hindi_asi_v4_realanchor_lora_r4_alpha16 \
  --source_dir data/eval_sources_reserved \
  --evaluation_plan configs/eval_hindi_native_to_asi.json \
  --validate_min_duration 2.6 \
  --steps 100,200,400,600,800,1000,1500,2000
