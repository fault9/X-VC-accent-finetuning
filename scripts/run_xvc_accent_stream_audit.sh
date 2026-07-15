#!/usr/bin/env bash
# Render the Hindi/ASI semantic-vs-acoustic stream localization audit.
# No training is performed and no YAML/config is generated.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_ROOT="${DATASET_ROOT:-data/hindi_asi_pristine_parallel_221}"
LEGACY_PAIRS_ROOT="${PAIRS_ROOT:-}"
REFERENCE="${REFERENCE:-data/eval_targets/ASI.wav}"
CONFIG="${CONFIG:-configs/xvc.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/xvc.pt}"
OUT="${OUT:-exp/xvc_accent_stream_audit_asi}"
MAX_PAIRS="${MAX_PAIRS:-40}"
DEVICE="${DEVICE:-0}"
SCORE="${SCORE:-1}"
METRICS_OUT="${METRICS_OUT:-${OUT}_metrics}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-12000}"

for required in "$REFERENCE" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required file: $required" >&2
    exit 1
  fi
done
if [[ -n "$LEGACY_PAIRS_ROOT" ]]; then
  if [[ ! -d "$LEGACY_PAIRS_ROOT/val" ]]; then
    echo "[error] missing legacy phone-pair shards: $LEGACY_PAIRS_ROOT/val" >&2
    exit 1
  fi
  input_args=(--pairs-root "$LEGACY_PAIRS_ROOT")
  input_label="legacy pairs root: $LEGACY_PAIRS_ROOT"
else
  if [[ ! -f "$DATASET_ROOT/manifests/val.jsonl" ]]; then
    echo "[error] missing pristine validation manifest: $DATASET_ROOT/manifests/val.jsonl" >&2
    exit 1
  fi
  input_args=(--dataset-root "$DATASET_ROOT")
  input_label="dataset root: $DATASET_ROOT"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi is unavailable; this audit requires a GPU container" >&2
  exit 70
fi
if ! gpu_memory="$(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "[error] NVML is unavailable; restart the GPU container before rendering" >&2
  exit 70
fi
free_gpu_mib="$(printf '%s\n' "$gpu_memory" | head -n1 | tr -d '[:space:]')"
if [[ ! "$free_gpu_mib" =~ ^[0-9]+$ ]]; then
  echo "[error] could not parse free GPU memory from nvidia-smi" >&2
  exit 70
fi
if (( free_gpu_mib < MIN_FREE_GPU_MIB )); then
  echo "[error] only ${free_gpu_mib} MiB GPU memory free; require ${MIN_FREE_GPU_MIB} MiB" >&2
  echo "Stop Hear-Me-Out/other GPU jobs, then rerun. Override only if intentional:" >&2
  echo "  MIN_FREE_GPU_MIB=<value> bash scripts/run_xvc_accent_stream_audit.sh" >&2
  exit 1
fi

echo "=== X-VC ACCENT STREAM AUDIT ==="
echo "$input_label"
echo "reference  : $REFERENCE"
echo "checkpoint : $CHECKPOINT"
echo "output     : $OUT"
echo "max pairs  : $MAX_PAIRS"
echo "min GPU MiB: $MIN_FREE_GPU_MIB"

python scripts/audit_xvc_accent_streams.py \
  "${input_args[@]}" \
  --split val \
  --target-speaker ASI \
  --reference "$REFERENCE" \
  --config "$CONFIG" \
  --ckpt "$CHECKPOINT" \
  --out "$OUT" \
  --max-pairs "$MAX_PAIRS" \
  --device "$DEVICE" \
  "$@"

if [[ "$SCORE" == "1" ]]; then
  echo
  echo "=== SCORE ALL FIVE CONDITIONS WITH ONE METRIC-MODEL LOAD ==="
  python scripts/score_xvc_accent_stream_audit.py \
    --audit-root "$OUT" \
    --reference "$REFERENCE" \
    --config "$CONFIG" \
    --ckpt "$CHECKPOINT" \
    --out "$METRICS_OUT" \
    --xvc-device "$DEVICE"
fi

cat <<EOF

Rendered conditions:
  $OUT/native_sem__native_zq/wavs
  $OUT/asi_sem__native_zq/wavs
  $OUT/native_sem__asi_zq/wavs
  $OUT/asi_sem__asi_zq_mapped/wavs
  $OUT/asi_sem__asi_zq_original/wavs

Listen by matching the same filename across these five directories.
Metrics table (when SCORE=1): $METRICS_OUT/condition_summary.csv
EOF
