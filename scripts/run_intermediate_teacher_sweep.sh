#!/usr/bin/env bash
# Bracket the L10 -> L15 teacher-depth trade-off at L12 and L12.5.
#
# Each candidate is rendered on all carriers, scored against the same L10
# baseline, and filtered with the established quality/content/identity floors.
# A candidate that fails its gate is recorded and skipped. A passing candidate
# gets exactly one controlled r4/alpha16 student run, so teacher depth remains
# the only experimental treatment.
#
# Usage:
#   nohup bash scripts/run_intermediate_teacher_sweep.sh \
#     > exp/run_logs/intermediate_teacher_sweep.out 2>&1 &
#
# Optional environment variables:
#   REUSE_INTERMEDIATE_RENDER=1  reuse complete L12/L12.5 render manifests
#   TRAIN_PASSING=0              render + gate only, do not train students
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

mkdir -p exp/run_logs exp/teacher_gates
status_file="exp/teacher_gates/intermediate_teacher_status.tsv"
printf 'candidate\tscale\tgate\tstudent\n' > "$status_file"

reuse_render="${REUSE_INTERMEDIATE_RENDER:-0}"
train_passing="${TRAIN_PASSING:-1}"

required_files=(
  ckpts/xvc.pt
  exp/accentbridge_asi_l0plus/bridge.pt
  exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt
  configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml
  configs/finetune_stackdistill_hindi_asi_l12_lora_r4_alpha16.yaml
  configs/finetune_stackdistill_hindi_asi_l125_lora_r4_alpha16.yaml
  configs/eval_hindi_native_to_asi.json
  data/eval_targets/ASI.wav
  data/stackdistill_hindi_asi_l10/manifests/train.jsonl
  data/stackdistill_hindi_asi_l10/manifests/val.jsonl
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "[error] missing required file: $path" >&2; exit 1; }
done
[[ -d data/eval_sources_reserved ]] || {
  echo "[error] missing data/eval_sources_reserved" >&2
  exit 1
}

if pgrep -af 'torchrun|deepspeed|bins.train|eval_checkpoints' >/dev/null; then
  echo "[error] another X-VC train/eval process is already visible" >&2
  pgrep -af 'torchrun|deepspeed|bins.train|eval_checkpoints' >&2 || true
  exit 1
fi

gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ ! "$gpu_used_mib" =~ ^[0-9]+$ ]]; then
  echo "[error] could not read GPU memory use; refusing to start" >&2
  exit 1
fi
if (( gpu_used_mib > 1024 )); then
  echo "[error] GPU already uses ${gpu_used_mib} MiB (>1024 MiB); stop Hear-Me-Out/other GPU jobs first" >&2
  nvidia-smi >&2 || true
  exit 1
fi

render_and_gate() {
  local label="$1"
  local scale="$2"
  local data_root="data/stackdistill_hindi_asi_${label}"
  local filtered_root="${data_root}_filtered"
  local gate_out="exp/teacher_gates/${label}_rebuild"

  mkdir -p "$gate_out"
  echo ""
  echo "=== ${label^^} TEACHER RENDER + GATE START $(date) ==="

  local reuse_ok=0
  if [[ "$reuse_render" == "1" ]] && \
     [[ -f "$data_root/manifests/train.jsonl" ]] && \
     [[ -f "$data_root/manifests/val.jsonl" ]] && \
     [[ -f "$data_root/distill_meta.json" ]]; then
    if python - "$data_root/distill_meta.json" "$scale" <<'PY'
import json
import math
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
actual = float(meta.get("renderer_lora_scale", float("nan")))
expected = float(sys.argv[2])
raise SystemExit(0 if math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9) else 1)
PY
    then
      reuse_ok=1
      echo "=== reusing verified ${label^^} render dataset ==="
    else
      echo "[warn] existing ${label^^} metadata does not match scale $scale; rerendering"
    fi
  fi

  if [[ "$reuse_ok" != "1" ]]; then
    python scripts/make_distill_dataset.py \
      --source-glob 'data/distill_sources_asi/*.wav' \
      --bridge-ckpt exp/accentbridge_asi_l0plus/bridge.pt --delta-scale 1.0 \
      --config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml \
      --ckpt exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt \
      --lora-scale "$scale" \
      --reference data/eval_targets/ASI.wav \
      --out "$data_root" --device 0
  fi

  python scripts/gate_teacher_renders.py \
    data/stackdistill_hindi_asi_l10/wavs \
    "$data_root/wavs" \
    --candidate-root "$data_root" \
    --filtered-root "$filtered_root" \
    --baseline-dir data/stackdistill_hindi_asi_l10/wavs \
    --candidate-dir "$data_root/wavs" \
    --reference data/eval_targets/ASI.wav \
    --config configs/xvc.yaml \
    --ckpt ckpts/xvc.pt \
    --out "$gate_out" \
    --limit 0 \
    --clip-mos-min 2.8 \
    --clip-wer-max 0.10 \
    --clip-sim-min 0.65 \
    --aggregate-mos-min 2.8 \
    --aggregate-wer-max 0.06 \
    --aggregate-sim-min 0.65 \
    --min-accent-depth-gain 0.02 \
    --min-retained-count 100 \
    --min-retained-fraction 0.50 \
    --min-retained-val 5
}

train_student() {
  local label="$1"
  local config="configs/finetune_stackdistill_hindi_asi_${label}_lora_r4_alpha16.yaml"
  local run="finetune_stackdistill_hindi_asi_${label}_lora_r4_alpha16"

  bash scripts/run_guarded_train_eval.sh \
    --accent "stackdistill_hindi_asi_${label}_filtered" \
    --config "$config" \
    --log_dir "exp/$run" \
    --source_dir data/eval_sources_reserved \
    --evaluation_plan configs/eval_hindi_native_to_asi.json \
    --validate_min_duration 2.6 \
    --steps 100,200,400,600,800,1000,1500,2000
}

run_candidate() {
  local label="$1"
  local scale="$2"
  local gate_status student_status

  set +e
  render_and_gate "$label" "$scale"
  gate_status=$?
  set -e

  if [[ "$gate_status" -ne 0 ]]; then
    printf '%s\t%s\tfailed_%s\tskipped\n' "$label" "$scale" "$gate_status" >> "$status_file"
    echo "[sweep] ${label^^} failed its teacher gate; student skipped" >&2
    return 0
  fi

  if [[ "$train_passing" != "1" ]]; then
    printf '%s\t%s\tpassed\tdisabled\n' "$label" "$scale" >> "$status_file"
    return 0
  fi

  set +e
  train_student "$label"
  student_status=$?
  set -e
  if [[ "$student_status" -eq 0 ]]; then
    printf '%s\t%s\tpassed\tpassed\n' "$label" "$scale" >> "$status_file"
  else
    printf '%s\t%s\tpassed\tfailed_%s\n' "$label" "$scale" "$student_status" >> "$status_file"
  fi
}

echo "=== INTERMEDIATE TEACHER SWEEP START $(date) ==="
echo "quality thresholds match the L15 gate; paired depth-gain floor is 0.02"
echo "status: $status_file"

run_candidate l12 1.2
run_candidate l125 1.25

echo ""
echo "=== INTERMEDIATE TEACHER SWEEP DONE $(date) ==="
cat "$status_file"
