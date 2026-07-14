#!/usr/bin/env bash
# One-command, sequential ASI persona experiment matrix.
#
# This is intentionally NOT parallel: every arm needs most of the single RTX
# 3090.  The script queues arms, evaluates after each, records failures, skips
# completed runs on restart, and produces a side-by-side table at the end.
#
# Matrix (alpha=16 throughout):
#   A. target-only / no-pair control, real ASI self-pairs (~19.2 min)
#      every Linear in acoustic_converter, batch 4:
#        ranks 1/2/4 x LRs 5e-5/1e-4
#   B. L15 same-timeline teacher distillation (only if teacher gate passes)
#      same six all-converter arms, plus the known filtered
#      converter+prenet r4,
#      batch-8 control.
#
# Rationale: standard alpha/r LoRA has approximately rank-independent early LR
# behavior, but the linked study reports some longer-run rank dependence and a
# somewhat lower optimum at rank 1.  The small 3x2 factorial therefore avoids
# confounding rank with LR.  Batch 4 is Harsha's small-batch proposal; the
# existing recipe's batch-8 +prenet arm is the control.
#
# Container usage:
#   nohup bash scripts/run_persona_experiment_matrix.sh \
#     > exp/run_logs/persona_matrix.out 2>&1 &
#
# Resume/reuse:
#   REUSE_L15_RENDER=1 nohup bash scripts/run_persona_experiment_matrix.sh ...
#   RUN_TARGET_ONLY=0 ...   # L15 matrix only
#   RUN_L15_MATRIX=0 ...    # target-only matrix only
#   RUN_DIVERSE_EVAL=0 ...  # skip final-checkpoint unseen-source evals
#   DRY_RUN=1 bash scripts/run_persona_experiment_matrix.sh  # configs only

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

run_target_only="${RUN_TARGET_ONLY:-1}"
run_l15_matrix="${RUN_L15_MATRIX:-1}"
run_diverse_eval="${RUN_DIVERSE_EVAL:-1}"
continue_on_failure="${CONTINUE_ON_ARM_FAILURE:-1}"
reuse_l15_render="${REUSE_L15_RENDER:-0}"
dry_run="${DRY_RUN:-0}"

matrix_root="exp/persona_matrix"
config_root="$matrix_root/configs"
mkdir -p "$config_root" exp/run_logs
status_file="$matrix_root/status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf "timestamp\tarm\tstage\tstatus\n" > "$status_file"
fi

declare -a completed_run_dirs=()
active_child_pid=""
active_child_pgid=""

cleanup_active_child() {
  local pgid="${active_child_pgid:-}"
  [[ -z "$pgid" ]] && return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    echo "[matrix] terminating active child process group $pgid" >&2
    kill -TERM -- "-$pgid" 2>/dev/null || true
    sleep 5
  fi
  if kill -0 -- "-$pgid" 2>/dev/null; then
    echo "[matrix] force-killing active child process group $pgid" >&2
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

on_matrix_exit() {
  local status=$?
  cleanup_active_child
  echo "[matrix] wrapper exit status=$status at $(date)" >&2
}

trap on_matrix_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_managed() {
  local status
  setsid "$@" &
  active_child_pid="$!"
  active_child_pgid="$active_child_pid"
  wait "$active_child_pid"
  status=$?
  active_child_pid=""
  active_child_pgid=""
  return "$status"
}

record_status() {
  local arm="$1" stage="$2" status="$3"
  printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$arm" "$stage" "$status" \
    >> "$status_file"
}

safe_lr_tag() {
  local lr="$1"
  lr="${lr/./p}"
  lr="${lr/-/m}"
  echo "$lr"
}

run_arm() {
  local objective="$1" template="$2" host="$3" rank="$4" lr="$5"
  local batch="$6" total_step="$7" steps="$8" min_duration="$9"
  local lr_tag name config_path run_dir status final_step final_ckpt

  lr_tag="$(safe_lr_tag "$lr")"
  name="matrix_${objective}_${host}_r${rank}_a16_lr${lr_tag}_b${batch}"
  config_path="$config_root/${name}.yaml"
  run_dir="exp/${name}"
  final_step="$total_step"
  printf -v final_ckpt "%s/ckpt/%06d.pt" "$run_dir" "$total_step"

  echo ""
  echo "################################################################"
  echo "### MATRIX ARM $name"
  echo "### objective=$objective host=$host rank=$rank alpha=16 lr=$lr batch=$batch"
  echo "################################################################"

  python scripts/make_lora_matrix_config.py \
    --template "$template" \
    --out "$config_path" \
    --host "$host" \
    --rank "$rank" \
    --alpha 16 \
    --lr "$lr" \
    --batch-size "$batch" \
    --total-step "$total_step"

  if [[ "$dry_run" == "1" ]]; then
    echo "[matrix] DRY_RUN: generated $config_path; no GPU work"
    record_status "$name" config_generation dry_run
    return 0
  fi

  mkdir -p "$run_dir"
  if [[ -f "$run_dir/eval_compare/summary.csv" ]]; then
    echo "[matrix] completed eval exists; skipping training: $run_dir"
    record_status "$name" primary_eval skipped_complete
    completed_run_dirs+=("$run_dir")
  elif [[ -f "$final_ckpt" ]]; then
    echo "[matrix] final checkpoint exists; resuming at primary eval: $final_ckpt"
    set +e
    run_managed bash scripts/run_guarded_train_eval.sh \
      --eval_only \
      --accent "$objective" \
      --log_dir "$run_dir" \
      --source_dir data/eval_sources_reserved \
      --evaluation_plan configs/eval_hindi_native_to_asi.json \
      --steps "$steps"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      record_status "$name" primary_eval_resume "failed_$status"
      echo "[matrix] resumed eval failed: $name (status $status)" >&2
      if [[ "$continue_on_failure" != "1" ]]; then
        return "$status"
      fi
      return 0
    fi
    record_status "$name" primary_eval_resume passed
    completed_run_dirs+=("$run_dir")
  elif compgen -G "$run_dir/ckpt/*.pt" > /dev/null; then
    echo "[matrix] PARTIAL checkpoints exist without a completed eval: $run_dir" >&2
    echo "[matrix] refusing to overwrite; resume or remove that arm explicitly" >&2
    record_status "$name" train partial_requires_attention
    return 0
  else
    set +e
    run_managed bash scripts/run_guarded_train_eval.sh \
      --accent "$objective" \
      --config "$config_path" \
      --log_dir "$run_dir" \
      --source_dir data/eval_sources_reserved \
      --evaluation_plan configs/eval_hindi_native_to_asi.json \
      --validate_min_duration "$min_duration" \
      --steps "$steps"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      record_status "$name" train_and_primary_eval "failed_$status"
      echo "[matrix] arm failed: $name (status $status)" >&2
      if [[ "$continue_on_failure" != "1" ]]; then
        return "$status"
      fi
      return 0
    fi
    record_status "$name" train_and_primary_eval passed
    completed_run_dirs+=("$run_dir")
  fi

  if [[ "$run_diverse_eval" == "1" ]] && \
     [[ ! -f "$run_dir/eval_compare_diverse/summary.csv" ]]; then
    set +e
    run_managed bash scripts/run_guarded_train_eval.sh \
      --eval_only \
      --accent "$objective" \
      --log_dir "$run_dir" \
      --source_dir data/eval_sources_diverse \
      --evaluation_plan configs/eval_diverse_to_asi.json \
      --eval_out eval_compare_diverse \
      --steps "$final_step"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      record_status "$name" diverse_eval passed
    else
      record_status "$name" diverse_eval "failed_$status"
      [[ "$continue_on_failure" == "1" ]] || return "$status"
    fi
  elif [[ "$run_diverse_eval" == "1" ]]; then
    echo "[matrix] completed diverse eval exists; skipping: $run_dir"
  fi
}

echo "=== ASI PERSONA MATRIX START $(date) ==="
echo "single-GPU sequential queue; no arms run concurrently"
echo "status file: $status_file"

if [[ "$dry_run" != "1" ]]; then
  eval_dirs=(data/eval_sources_reserved data/eval_targets)
  if [[ "$run_diverse_eval" == "1" ]]; then
    test -d data/eval_sources_diverse
    test -f configs/eval_diverse_to_asi.json
    eval_dirs+=(data/eval_sources_diverse)
  fi
  python scripts/check_sample_rates.py --dirs "${eval_dirs[@]}" --expect 16000
fi

l15_ready=0
if [[ "$run_l15_matrix" == "1" ]]; then
  if [[ "$dry_run" == "1" ]]; then
    l15_ready=1
  else
    echo ""
    echo "=== PREPARE + FAIL-CLOSED GATE L15 TEACHER ==="
    set +e
    run_managed env PREPARE_ONLY=1 REUSE_L15_RENDER="$reuse_l15_render" \
      bash scripts/run_l15_rebuild.sh
    l15_status=$?
    set -e
    if [[ "$l15_status" -eq 0 ]]; then
      l15_ready=1
      record_status l15_teacher prepare_and_gate passed
    else
      record_status l15_teacher prepare_and_gate "failed_$l15_status"
      echo "[matrix] L15 gate failed; L15 student arms will be skipped" >&2
      [[ "$continue_on_failure" == "1" ]] || exit "$l15_status"
    fi
  fi
fi

if [[ "$run_target_only" == "1" ]]; then
  echo ""
  echo "=== PREPARE REAL-ASI TARGET-ONLY SELF-PAIRS ==="
  target_ready=1
  if [[ "$dry_run" != "1" ]]; then
    if [[ ! -f data/asi_selfpairs_wide/manifests/train.jsonl ]] || \
       [[ ! -f data/asi_selfpairs_wide/manifests/val.jsonl ]]; then
      set +e
      python scripts/make_selfpair_manifest.py \
        --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/train.jsonl \
        --from-manifest data/crosspair_hindi_latent_wide_asionly/manifests/val.jsonl \
        --reference data/eval_targets/ASI.wav \
        --out data/asi_selfpairs_wide
      target_status=$?
      set -e
      if [[ "$target_status" -ne 0 ]]; then
        target_ready=0
        record_status target_only_data prepare "failed_$target_status"
        echo "[matrix] target-only data preparation failed; those arms will be skipped" >&2
        [[ "$continue_on_failure" == "1" ]] || exit "$target_status"
      fi
    fi
  fi

  if [[ "$target_ready" == "1" ]]; then
    target_template="configs/finetune_asi_recononly_wide_lora_r4_alpha16.yaml"
    for rank in 1 2 4; do
      for lr in 0.00005 0.0001; do
        run_arm asi_selfpairs_wide "$target_template" acoustic "$rank" "$lr" 4 \
          1000 "100,200,400,600,800,1000" 2.0
      done
    done
  fi
fi

if [[ "$l15_ready" == "1" ]]; then
  echo ""
  echo "=== L15 FILTERED SAME-TIMELINE DISTILL MATRIX ==="
  l15_template="configs/finetune_stackdistill_hindi_asi_l15_lora_r4_alpha16.yaml"
  for rank in 1 2 4; do
    for lr in 0.00005 0.0001; do
      run_arm stackdistill_hindi_asi_l15_filtered "$l15_template" acoustic "$rank" \
        "$lr" 4 2000 "100,200,400,600,800,1000,1500,2000" 2.6
    done
  done

  # Known-host control: same filtered L15 data, but retain prenet/AdaLN and the
  # established batch 8 recipe.  This tells us whether Harsha's converter-only
  # restriction helps, rather than assuming it.
  run_arm stackdistill_hindi_asi_l15_filtered "$l15_template" acoustic_prenet 4 \
    0.0001 8 2000 "100,200,400,600,800,1000,1500,2000" 2.6
fi

echo ""
echo "=== MATRIX PRIMARY-EVAL COMPARISON ==="
if [[ "${#completed_run_dirs[@]}" -gt 0 ]]; then
  python scripts/compare_sweep_evals.py "${completed_run_dirs[@]}" \
    | tee "$matrix_root/comparison.txt"
else
  echo "[matrix] no completed run summaries to compare" | tee "$matrix_root/comparison.txt"
fi

echo ""
echo "=== ASI PERSONA MATRIX DONE $(date) ==="
echo "status:     $status_file"
echo "comparison: $matrix_root/comparison.txt"
echo "Human listening is still required; the script does not collapse MOS/WER/"
echo "similarity/accent into an unvalidated single 'winner' score."
