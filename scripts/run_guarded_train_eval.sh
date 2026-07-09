#!/usr/bin/env bash
# Guarded X-VC train/eval runner.
#
# This wrapper supervises a training run and its optional evaluation from outside
# DeepSpeed/torchrun. It is intentionally separate from finetune.sh so the normal
# training entrypoint stays simple while long shared-GPU runs get:
#   - CUDA/NVML preflight checks before touching DeepSpeed
#   - timestamped full logs under exp/run_logs/
#   - train/eval subprocesses started in isolated process groups
#   - trap-based cleanup of only the child process group started by this wrapper
#   - post-run process/GPU checks for auditability
#
# Example:
#   bash scripts/run_guarded_train_eval.sh \
#     --accent crosspair_hindi_latent_800_strict \
#     --config configs/finetune_crosspair_hindi_latent_800_strict_recon20_semadapter_lr1e-5.yaml \
#     --log_dir exp/finetune_crosspair_hindi_latent_800_strict_recon20_semadapter_lr1e-5 \
#     --validate_min_duration 2.6 \
#     --steps 100,200,300,400,600,800,1000
#
# Eval-only after a completed training run:
#   bash scripts/run_guarded_train_eval.sh \
#     --eval_only \
#     --log_dir exp/finetune_crosspair_hindi_latent_800_strict_recon20_semadapter_lr1e-5 \
#     --steps 100,200,300,400,600,800,1000

set -euo pipefail

accent=""
config=""
log_dir=""
checkpoint="ckpts/xvc.pt"
steps="100,200,300,400,600,800,1000"
source_dir="data/eval_sources"
targets_dir="data/eval_targets"
evaluation_plan="configs/eval_hindi_native_to_accent.json"
validate_min_duration="3.0"
num_workers=0
gpus=1
resume_step=0
port=10201
seed=1234
run_train=1
run_eval=1
mos=1
accent_clf=1

usage() {
  sed -n '2,32p' "$0" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accent)                accent="$2"; shift 2 ;;
    --config)                config="$2"; shift 2 ;;
    --log_dir)               log_dir="$2"; shift 2 ;;
    --checkpoint)            checkpoint="$2"; shift 2 ;;
    --steps)                 steps="$2"; shift 2 ;;
    --source_dir)            source_dir="$2"; shift 2 ;;
    --targets_dir)           targets_dir="$2"; shift 2 ;;
    --evaluation_plan)       evaluation_plan="$2"; shift 2 ;;
    --validate_min_duration) validate_min_duration="$2"; shift 2 ;;
    --num_workers)           num_workers="$2"; shift 2 ;;
    --gpus)                  gpus="$2"; shift 2 ;;
    --resume_step)           resume_step="$2"; shift 2 ;;
    --port)                  port="$2"; shift 2 ;;
    --seed)                  seed="$2"; shift 2 ;;
    --eval_only)             run_train=0; run_eval=1; shift 1 ;;
    --no_eval)               run_eval=0; shift 1 ;;
    --no_mos)                mos=0; shift 1 ;;
    --no_accent_clf)         accent_clf=0; shift 1 ;;
    -h|--help)               usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$log_dir" ]]; then
  echo "missing required --log_dir" >&2
  usage
  exit 2
fi
if [[ "$run_train" -eq 1 ]] && [[ -z "$accent" || -z "$config" ]]; then
  echo "training requires --accent and --config" >&2
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"

mkdir -p exp/run_logs
safe_name="$(basename "$log_dir")"
stamp="$(date +%Y%m%d_%H%M%S)"
master_log="exp/run_logs/${safe_name}_${stamp}_guarded.log"

# Keep traps in this shell rather than inside a pipeline subshell.
exec > >(tee -a "$master_log") 2>&1

active_pid=""
active_pgid=""

cleanup_group() {
  local pgid="${1:-}"
  [[ -z "$pgid" ]] && return 0

  if kill -0 -- "-$pgid" 2>/dev/null; then
    echo "=== cleanup: TERM process group $pgid ==="
    kill -TERM -- "-$pgid" 2>/dev/null || true
    sleep 5
  fi

  if kill -0 -- "-$pgid" 2>/dev/null; then
    echo "=== cleanup: KILL process group $pgid ==="
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

on_exit() {
  local status=$?
  echo ""
  echo "=== GUARDED WRAPPER EXIT $(date) status=${status} ==="

  cleanup_group "${active_pgid:-}"

  echo "=== leftover train/eval-like processes visible in this container ==="
  pgrep -af 'torchrun|deepspeed|bins.train|eval_checkpoints|SpeechMOS|whisper' || echo "none"

  echo "=== GPU after guarded wrapper ==="
  nvidia-smi || true

  echo "=== guarded log ==="
  echo "$master_log"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_isolated() {
  local label="$1"
  shift

  echo ""
  echo "============================================================"
  echo "=== START ${label}: $(date)"
  echo "============================================================"

  setsid "$@" &
  active_pid="$!"
  active_pgid="$active_pid"

  set +e
  wait "$active_pid"
  local status=$?
  set -e

  cleanup_group "$active_pgid"
  active_pid=""
  active_pgid=""

  echo ""
  echo "============================================================"
  echo "=== DONE ${label}: $(date) status=${status}"
  echo "============================================================"

  return "$status"
}

echo "=== GUARDED X-VC RUN START $(date) ==="
echo "root_dir              : $root_dir"
echo "accent                : $accent"
echo "config                : $config"
echo "log_dir               : $log_dir"
echo "checkpoint            : $checkpoint"
echo "steps                 : $steps"
echo "validate_min_duration : $validate_min_duration"
echo "run_train             : $run_train"
echo "run_eval              : $run_eval"
echo "master_log            : $master_log"
hostname
pwd

echo ""
echo "=== preflight files ==="
test -f "$checkpoint"
if [[ "$run_train" -eq 1 ]]; then
  test -f "$config"
fi
if [[ "$run_train" -eq 1 && "$accent" == crosspair_* ]]; then
  test -f "data/${accent}/manifests/train.jsonl"
  test -f "data/${accent}/manifests/val.jsonl"
fi
if [[ "$run_eval" -eq 1 ]]; then
  test -d "$log_dir"
  test -d "$source_dir"
  test -d "$targets_dir"
  test -f "$evaluation_plan"
  source_count="$(find "$source_dir" -maxdepth 1 -type f -name '*.wav' | wc -l)"
  target_count="$(find "$targets_dir" -maxdepth 1 -type f -name '*.wav' | wc -l)"
  echo "eval source wavs      : $source_count"
  echo "eval target wavs      : $target_count"
  if [[ "$source_count" -eq 0 || "$target_count" -eq 0 ]]; then
    echo "[error] eval source/target directories must contain wav files" >&2
    exit 1
  fi
fi

echo ""
echo "=== preflight GPU ==="
nvidia-smi

echo ""
echo "=== preflight CUDA ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing to start training")
print("device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== preflight no existing train/eval-like process ==="
if pgrep -af 'torchrun|deepspeed|bins.train|eval_checkpoints' ; then
  echo "[error] train/eval-like process already running; refusing to start" >&2
  exit 1
fi

if [[ "$run_train" -eq 1 && "$accent" == crosspair_* ]]; then
  echo ""
  echo "=== preflight crosspair validation ==="
  python scripts/validate_crosspairs.py \
    --data-root "data/${accent}" \
    --min-duration "$validate_min_duration"
fi

# Eval sources must be held-out content: fail if any eval clip's prompt was
# trained on (as source or target side). See scripts/check_eval_overlap.py.
if [[ "$run_eval" -eq 1 && -n "$accent" && -f "data/${accent}/manifests/train.jsonl" ]]; then
  echo ""
  echo "=== preflight eval-source contamination check ==="
  python scripts/check_eval_overlap.py \
    --source-dir "$source_dir" \
    --train-manifest "data/${accent}/manifests/train.jsonl"
fi

if [[ "$run_train" -eq 1 ]]; then
  train_cmd=(
    env "XVC_VALIDATE_MIN_DURATION=${validate_min_duration}"
    bash scripts/finetune.sh
    --accent "$accent"
    --config "$config"
    --log_dir "$log_dir"
    --checkpoint "$checkpoint"
    --gpus "$gpus"
    --resume_step "$resume_step"
    --port "$port"
    --seed "$seed"
    --num_workers "$num_workers"
  )

  run_isolated "train ${safe_name}" "${train_cmd[@]}"
fi

if [[ "$run_eval" -eq 1 ]]; then
  eval_cmd=(
    python scripts/eval_checkpoints.py run
    --run-dir "$log_dir"
    --source-dir "$source_dir"
    --targets-dir "$targets_dir"
    --evaluation-plan "$evaluation_plan"
    --steps "$steps"
    --include-base "$checkpoint"
    --out "$log_dir/eval_compare"
  )
  [[ "$mos" -eq 1 ]] && eval_cmd+=(--mos)
  [[ "$accent_clf" -eq 1 ]] && eval_cmd+=(--accent-clf)

  run_isolated "eval ${safe_name}" "${eval_cmd[@]}"
fi

echo ""
echo "=== GUARDED X-VC RUN DONE $(date) ==="
