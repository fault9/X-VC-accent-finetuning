#!/usr/bin/env bash
# Two post-prenet-only arms: the proven residual mapper and the experimental
# recurrent pronunciation editor. Frozen X-VC and one ASI reference are shared.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

cleanup() {
  status=$?
  trap - EXIT INT TERM
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  echo "=== POST-PRENET EXPERIMENTS EXIT status=$status $(date) ==="
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  exit "$status"
}
trap cleanup EXIT INT TERM

DATASET_ROOT="${DATASET_ROOT:-data/hindi_asi_pristine_parallel_scaleup}"
PAIRS_ROOT="${PAIRS_ROOT:-data/postprenet_windows_asi_scaleup_2400ms}"
SOURCE_DIR="${SOURCE_DIR:-data/eval_sources_joint_persona_unseen}"
REFERENCE="${REFERENCE:-data/eval_targets/ASI.wav}"
CONFIG="${CONFIG:-configs/xvc.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/xvc.pt}"
EXP_ROOT="${EXP_ROOT:-exp/postprenet4_asi_scaleup}"
ACCENT_CALIBRATION="${ACCENT_CALIBRATION:-exp/accent_posterior_calibration/summary.csv}"
ARMS="${ARMS:-residual pronunciation}"
BATCH="${BATCH:-4}"
LR="${LR:-0.0002}"
EPOCHS="${EPOCHS:-20}"
STEPS="${STEPS:-0}"
DEVICE="${DEVICE:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-12000}"
MIN_TRAIN_PAIRS="${MIN_TRAIN_PAIRS:-600}"
MIN_ACCEPTED_TRAIN_PAIRS="${MIN_ACCEPTED_TRAIN_PAIRS:-500}"
MIN_SOURCE_SPEAKERS="${MIN_SOURCE_SPEAKERS:-4}"
MIN_UNIQUE_TARGET_MINUTES="${MIN_UNIQUE_TARGET_MINUTES:-45}"
MIN_EVAL_SPEAKERS="${MIN_EVAL_SPEAKERS:-3}"

for path in \
  "$DATASET_ROOT/manifests/train.jsonl" "$DATASET_ROOT/manifests/val.jsonl" \
  "$REFERENCE" "$CONFIG" "$CHECKPOINT" "$ACCENT_CALIBRATION"; do
  [[ -f "$path" ]] || { echo "[error] missing required file: $path" >&2; exit 1; }
done
[[ -d "$SOURCE_DIR" ]] || { echo "[error] missing source dir: $SOURCE_DIR" >&2; exit 1; }

if ! free_gpu_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')"; then
  echo "[error] NVML unavailable; restart the GPU container" >&2
  exit 70
fi
[[ "$free_gpu_mib" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU reading" >&2; exit 70; }
(( free_gpu_mib >= MIN_FREE_GPU_MIB )) || {
  echo "[error] only ${free_gpu_mib} MiB free; need ${MIN_FREE_GPU_MIB}" >&2; exit 1;
}

mkdir -p "$EXP_ROOT" exp/run_logs
python scripts/check_persona_dataset_scale.py \
  --dataset-root "$DATASET_ROOT" --eval-source-dir "$SOURCE_DIR" \
  --target-speaker ASI --min-train-pairs "$MIN_TRAIN_PAIRS" \
  --min-source-speakers "$MIN_SOURCE_SPEAKERS" \
  --min-unique-target-minutes "$MIN_UNIQUE_TARGET_MINUTES" \
  --min-eval-speakers "$MIN_EVAL_SPEAKERS" \
  --out "$EXP_ROOT/dataset_gate.json"

if [[ ! -f "$PAIRS_ROOT/extraction_summary.json" ]]; then
  python scripts/extract_joint_persona_pairs.py \
    --dataset-root "$DATASET_ROOT" --config "$CONFIG" --ckpt "$CHECKPOINT" \
    --target-speaker ASI --out "$PAIRS_ROOT" --split all \
    --windows-per-pair 2 --val-windows-per-pair 1 --device "$DEVICE" \
    2>&1 | tee "$EXP_ROOT/extract.log"
fi

read -r accepted windows <<<"$(python - "$PAIRS_ROOT/extraction_summary.json" "$MIN_ACCEPTED_TRAIN_PAIRS" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
train = s.get("splits", {}).get("train", {})
if s.get("status") != "pass" or s.get("schema_version", 0) < 2:
    raise SystemExit(f"[error] invalid extracted cache: {s}")
accepted = int(train.get("rows_kept", 0))
windows = int(train.get("windows_kept", 0))
if accepted < int(sys.argv[2]):
    raise SystemExit(f"[error] accepted train pairs {accepted} < {sys.argv[2]}")
print(accepted, windows)
PY
)"

if (( STEPS <= 0 )); then
  STEPS=$(( (windows + BATCH - 1) / BATCH * EPOCHS ))
fi
printf 'arm\tstatus\taudio_gate\tlatency_gate\tsteps\taccepted_pairs\twindows\n' > "$EXP_ROOT/status.tsv"
echo "=== POST-PRENET 4-FRAME EXPERIMENTS $(date) ==="
echo "dataset       : $DATASET_ROOT"
echo "accepted rows : $accepted; encoded windows: $windows"
echo "steps         : $STEPS (${EPOCHS} exposure epochs when STEPS=0)"
echo "arms          : $ARMS"

passed=0
for arm_name in $ARMS; do
  arm="$EXP_ROOT/$arm_name"
  [[ ! -e "$arm/best.pt" ]] || { echo "[error] refusing to overwrite $arm" >&2; exit 1; }
  if [[ "$arm_name" == residual ]]; then
    python scripts/train_postprenet_persona_mapper.py \
      --train-dir "$PAIRS_ROOT/train" --val-dir "$PAIRS_ROOT/val" \
      --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm" \
      --steps "$STEPS" --batch "$BATCH" --lr "$LR" --layers 5 \
      --lookahead-frames 4 --device "cuda:$DEVICE" 2>&1 | tee "$arm.train.log"
  elif [[ "$arm_name" == pronunciation ]]; then
    python scripts/train_pronunciation_editor.py \
      --train-dir "$PAIRS_ROOT/train" --val-dir "$PAIRS_ROOT/val" \
      --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm" \
      --steps "$STEPS" --batch "$BATCH" --lr "$LR" --layers 2 \
      --lookahead-frames 4 --device "cuda:$DEVICE" 2>&1 | tee "$arm.train.log"
  else
    echo "[error] unknown arm: $arm_name" >&2; exit 2
  fi

  python scripts/eval_joint_persona_mapper.py \
    --source-dir "$SOURCE_DIR" --reference "$REFERENCE" \
    --mapper-ckpt "$arm/best.pt" --config "$CONFIG" --ckpt "$CHECKPOINT" \
    --out "$arm/eval_compare" --require-unseen-source \
    --min-unseen-speakers "$MIN_EVAL_SPEAKERS" \
    --training-manifest "$DATASET_ROOT/manifests/train.jsonl" \
    --streaming --chunk-ms 2400 --current-ms 120 --smooth-ms 20 --future-ms 100 \
    --device "$DEVICE" 2>&1 | tee "$arm.eval.log"
  python scripts/score_xvc_accent_stream_audit.py \
    --audit-root "$arm/eval_compare" --reference "$REFERENCE" \
    --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm/eval_metrics" \
    --xvc-device "$DEVICE" 2>&1 | tee "$arm.score.log"

  if python scripts/gate_joint_persona_mapper.py \
      --summary "$arm/eval_metrics/condition_summary.csv" \
      --calibration "$ACCENT_CALIBRATION" --out "$arm/gate.json"; then
    audio_gate=pass
  else
    audio_gate=fail
  fi
  if python scripts/gate_persona_latency.py \
      --meta "$arm/eval_compare/audit_meta.json" --out "$arm/latency_gate.json"; then
    latency_gate=pass
  else
    latency_gate=fail
  fi
  [[ "$audio_gate" == pass && "$latency_gate" == pass ]] && passed=$((passed + 1))
  printf '%s\tcomplete\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm_name" "$audio_gate" "$latency_gate" "$STEPS" "$accepted" "$windows" \
    >> "$EXP_ROOT/status.tsv"
done

cat "$EXP_ROOT/status.tsv"
echo "Listen to matched files under each {residual,pronunciation}/eval_compare/*/wavs"
(( passed > 0 )) || { echo "[result] neither post-prenet arm passed" >&2; exit 1; }
