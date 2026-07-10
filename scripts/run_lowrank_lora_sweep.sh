#!/usr/bin/env bash
# Sequential low-rank LoRA sweep: acoustic_converter-only, ranks 1/2/4, alpha 16,
# batch 4, lr 1e-4, on crosspair_hindi_latent_400. See CHANGES.md
# "Target-conditioned low-rank LoRA sweep (2026-07-10)".
#
# Each arm goes through scripts/run_guarded_train_eval.sh, which does the
# CUDA/NVML preflight, crosspair + eval-contamination validation, training, and
# the checkpoint eval sweep. Arms run strictly sequentially (shared GPU); the
# first failing arm aborts the sweep so a broken setup does not burn GPU time
# on identical siblings.
#
# Usage (container, from repo root):
#   bash scripts/run_lowrank_lora_sweep.sh
#   RANKS="2 4" bash scripts/run_lowrank_lora_sweep.sh    # subset / resume
#   STEPS="100,200,400" bash scripts/run_lowrank_lora_sweep.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

ranks="${RANKS:-1 2 4}"
steps="${STEPS:-100,200,400,600,800,1000}"
accent="crosspair_hindi_latent_400"
# Reserved held-out prompts (arctic b0002-b0012), NOT the runner default
# data/eval_sources: that dir is the pre-2026-07-09 eval set whose prompts
# overlap latent_400 training -- the contamination gate rightly rejects it.
source_dir="${SOURCE_DIR:-data/eval_sources_reserved}"

# Fail fast on sample-rate drift in the eval dirs (dataset-side rates are
# already enforced by validate_crosspairs inside the guarded runner; consumers
# would silently resample -- we want drift loud, not silent).
python scripts/check_sample_rates.py --dirs "$source_dir" data/eval_targets --expect 16000

for r in $ranks; do
  name="finetune_crosspair_hindi_latent_400_lora_acoustic_r${r}_alpha16"
  # The guarded preflight `test -d $log_dir` runs BEFORE training creates the
  # dir, so a fresh run dies silently without this (the x1.5-arm trap).
  mkdir -p "exp/${name}"
  echo ""
  echo "############################################################"
  echo "### low-rank sweep arm r=${r}  ->  exp/${name}"
  echo "############################################################"
  bash scripts/run_guarded_train_eval.sh \
    --accent "$accent" \
    --config "configs/${name}.yaml" \
    --log_dir "exp/${name}" \
    --source_dir "$source_dir" \
    --validate_min_duration 3.0 \
    --steps "$steps"
done

echo ""
echo "### low-rank sweep done (ranks: ${ranks}); summaries:"
for r in $ranks; do
  echo "  exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r${r}_alpha16/eval_compare/summary.csv"
done
