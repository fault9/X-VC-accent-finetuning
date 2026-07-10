#!/usr/bin/env bash
# Sequential PERSONA-MODE low-rank DISTILL sweep: ranks 1/2/4 on the x1.2
# l0plus teacher data (data/distill_hindi_asi), converter+prenet+AdaLN target
# set -- one knob (rank) vs the known finetune_distill_hindi_asi_lora_r8 run.
# See CHANGES.md "Target-conditioned low-rank LoRA sweep (2026-07-10)".
#
# Each arm goes through scripts/run_guarded_train_eval.sh (CUDA preflight,
# eval-contamination gate, training, checkpoint eval). The crosspair
# validation stage is skipped automatically (distill datasets are not
# crosspair_*). Arms run strictly sequentially; first failure aborts.
#
# Usage (container, from repo root):
#   bash scripts/run_lowrank_distill_sweep.sh
#   RANKS="2 4" bash scripts/run_lowrank_distill_sweep.sh    # subset / resume

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

ranks="${RANKS:-1 2 4}"
# total_step is 2000 for distill arms (accent acquisition is slower there);
# the grid matches the distill-r8 eval rows for direct comparison.
steps="${STEPS:-100,200,400,600,800,1000,1500,2000}"
accent="distill_hindi_asi"
plan="configs/eval_hindi_native_to_asi.json"
source_dir="${SOURCE_DIR:-data/eval_sources_reserved}"

python scripts/check_sample_rates.py --dirs "$source_dir" data/eval_targets --expect 16000

for r in $ranks; do
  name="finetune_distill_hindi_asi_lora_r${r}_alpha16"
  # Guarded preflight `test -d $log_dir` runs BEFORE training creates the dir.
  mkdir -p "exp/${name}"
  echo ""
  echo "############################################################"
  echo "### low-rank distill arm r=${r}  ->  exp/${name}"
  echo "############################################################"
  bash scripts/run_guarded_train_eval.sh \
    --accent "$accent" \
    --config "configs/${name}.yaml" \
    --log_dir "exp/${name}" \
    --source_dir "$source_dir" \
    --evaluation_plan "$plan" \
    --validate_min_duration 2.6 \
    --steps "$steps"
done

echo ""
echo "### low-rank distill sweep done (ranks: ${ranks}); compare with:"
echo "  python scripts/compare_sweep_evals.py exp/finetune_distill_hindi_asi_lora_r1_alpha16 exp/finetune_distill_hindi_asi_lora_r2_alpha16 exp/finetune_distill_hindi_asi_lora_r4_alpha16 exp/finetune_distill_hindi_asi_lora_r8"
