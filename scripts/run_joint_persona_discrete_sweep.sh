#!/usr/bin/env bash
# Controlled latency-safe sweep for the inference-matched ASI code objective.
# X-VC remains frozen; every arm uses the same extracted pairs, ASI reference,
# unseen-source set, scoring stack, and acceptance gate.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

cleanup() {
  status=$?
  trap - EXIT INT TERM
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  echo "=== DISCRETE JOINT PERSONA WRAPPER EXIT status=$status $(date) ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

DATASET_ROOT="${DATASET_ROOT:-data/hindi_asi_pristine_parallel_221}"
PAIRS_ROOT="${PAIRS_ROOT:-data/joint_persona_windows_asi_2400ms}"
REFERENCE="${REFERENCE:-data/eval_targets/ASI.wav}"
SOURCE_DIR="${SOURCE_DIR:-data/eval_sources_joint_persona_clean}"
CONFIG="${CONFIG:-configs/xvc.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/xvc.pt}"
EXP_ROOT="${EXP_ROOT:-exp/joint_persona_discrete_asi}"
ACCENT_CALIBRATION="${ACCENT_CALIBRATION:-exp/accent_posterior_calibration/summary.csv}"
LOOKAHEADS="${LOOKAHEADS:-0 4}"
DISCRETE_WEIGHTS="${DISCRETE_WEIGHTS:-0.25 0.5}"
CODE_TEMPERATURE="${CODE_TEMPERATURE:-0.1}"
CODE_MARGIN_WEIGHT="${CODE_MARGIN_WEIGHT:-0.25}"
CODE_MARGIN="${CODE_MARGIN:-1.0}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-4}"
LR="${LR:-0.0002}"
DEVICE="${DEVICE:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-12000}"
MIN_UNSEEN_SPEAKERS="${MIN_UNSEEN_SPEAKERS:-2}"

for required in \
  "$DATASET_ROOT/manifests/train.jsonl" \
  "$DATASET_ROOT/manifests/val.jsonl" \
  "$REFERENCE" "$CONFIG" "$CHECKPOINT" "$ACCENT_CALIBRATION"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required file: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[error] missing unseen-source evaluation directory: $SOURCE_DIR" >&2
  exit 1
fi
if [[ "$(find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.wav' | wc -l)" -lt 10 ]]; then
  echo "[error] require at least 10 unseen-source wavs under $SOURCE_DIR" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi is unavailable; this run requires a GPU container" >&2
  exit 70
fi
if ! gpu_memory="$(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "[error] NVML is unavailable; restart the GPU container before training" >&2
  exit 70
fi
free_gpu_mib="$(printf '%s\n' "$gpu_memory" | head -n1 | tr -d '[:space:]')"
if [[ ! "$free_gpu_mib" =~ ^[0-9]+$ ]]; then
  echo "[error] could not parse free GPU memory from nvidia-smi" >&2
  exit 70
fi
if (( free_gpu_mib < MIN_FREE_GPU_MIB )); then
  echo "[error] only ${free_gpu_mib} MiB GPU memory free; require ${MIN_FREE_GPU_MIB} MiB" >&2
  exit 1
fi

mkdir -p "$EXP_ROOT" exp/run_logs
echo -e "lookahead_frames\tdiscrete_weight\tstatus\taudio_gate\tlatency_gate\tcode_change\ttarget_code_gain" > "$EXP_ROOT/status.tsv"
echo "=== DISCRETE JOINT TARGET-PERSONA SWEEP $(date) ==="
echo "target persona    : ASI voice + Hindi/Indian English accent"
echo "source policy     : no source-speaker conditioning; eval speakers unseen"
echo "voice policy      : frozen X-VC + identical ASI reference"
echo "lookaheads        : $LOOKAHEADS"
echo "discrete weights  : $DISCRETE_WEIGHTS"
echo "code temperature  : $CODE_TEMPERATURE"
echo "target code margin: weight=$CODE_MARGIN_WEIGHT margin=$CODE_MARGIN"

if [[ ! -f "$PAIRS_ROOT/extraction_summary.json" ]]; then
  python scripts/extract_joint_persona_pairs.py \
    --dataset-root "$DATASET_ROOT" \
    --config "$CONFIG" \
    --ckpt "$CHECKPOINT" \
    --target-speaker ASI \
    --out "$PAIRS_ROOT" \
    --split all \
    --device "$DEVICE" \
    2>&1 | tee "$EXP_ROOT/extract.log"
fi
python - "$PAIRS_ROOT/extraction_summary.json" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "pass":
    raise SystemExit(f"[error] extracted pair gate is not pass: {summary.get('status')}")
policy = summary.get("window_policy", {})
if summary.get("schema_version", 0) < 2 or not policy.get("waveform_crop_before_frozen_encoder"):
    raise SystemExit("[error] extracted pairs are not raw-window-encoded schema v2")
if float(policy.get("window_seconds", 0)) != 2.4 or int(policy.get("window_frames", 0)) != 120:
    raise SystemExit(f"[error] expected 2.4 s / 120-frame encoded windows, got {policy}")
for split in ("train", "val"):
    if not (Path(sys.argv[1]).parent / split).is_dir():
        raise SystemExit(f"[error] missing extracted split: {split}")
print("[pairs] extraction gate PASS", summary["splits"])
PY

passed=0
completed=0
for lookahead in $LOOKAHEADS; do
  for discrete_weight in $DISCRETE_WEIGHTS; do
    weight_tag="${discrete_weight//./p}"
    arm="$EXP_ROOT/lookahead_${lookahead}f_dw_${weight_tag}"
    if [[ -e "$arm/best.pt" ]]; then
      echo "[error] refusing to overwrite completed arm: $arm" >&2
      exit 1
    fi
    echo "=== TRAIN lookahead=${lookahead}f discrete=${discrete_weight} $(date) ==="
    python scripts/train_joint_persona_mapper.py \
      --train-dir "$PAIRS_ROOT/train" \
      --val-dir "$PAIRS_ROOT/val" \
      --config "$CONFIG" \
      --ckpt "$CHECKPOINT" \
      --out "$arm" \
      --steps "$STEPS" \
      --batch "$BATCH" \
      --lr "$LR" \
      --lookahead-frames "$lookahead" \
      --layers 5 \
      --lambda-discrete-code "$discrete_weight" \
      --lambda-code-margin "$CODE_MARGIN_WEIGHT" \
      --code-margin "$CODE_MARGIN" \
      --code-temperature "$CODE_TEMPERATURE" \
      --device "cuda:$DEVICE" \
      2>&1 | tee "$arm.train.log"

    mechanism="$(python - "$arm/best.pt" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu")
metrics = payload["validation"]["all"]
print(f'{metrics["code_change_fraction"]}\t{metrics["aligned_target_code_gain"]}')
PY
)"
    code_change="${mechanism%%$'\t'*}"
    target_gain="${mechanism#*$'\t'}"
    echo "[mechanism] code_change=$code_change target_aligned_gain=$target_gain"

    python scripts/eval_joint_persona_mapper.py \
      --source-dir "$SOURCE_DIR" \
      --reference "$REFERENCE" \
      --mapper-ckpt "$arm/best.pt" \
      --config "$CONFIG" \
      --ckpt "$CHECKPOINT" \
      --out "$arm/eval_compare" \
      --require-unseen-source \
      --min-unseen-speakers "$MIN_UNSEEN_SPEAKERS" \
      --training-manifest "$DATASET_ROOT/manifests/train.jsonl" \
      --streaming \
      --chunk-ms 2400 --current-ms 120 --smooth-ms 20 --future-ms 100 \
      --device "$DEVICE" \
      2>&1 | tee "$arm.eval.log"

    python scripts/score_xvc_accent_stream_audit.py \
      --audit-root "$arm/eval_compare" \
      --reference "$REFERENCE" \
      --config "$CONFIG" \
      --ckpt "$CHECKPOINT" \
      --out "$arm/eval_metrics" \
      --xvc-device "$DEVICE" \
      2>&1 | tee "$arm.score.log"

    completed=$((completed + 1))
    if python scripts/gate_joint_persona_mapper.py \
        --summary "$arm/eval_metrics/condition_summary.csv" \
        --calibration "$ACCENT_CALIBRATION" \
        --out "$arm/gate.json"; then
      audio_gate=pass
    else
      audio_gate=fail
    fi
    if python scripts/gate_persona_latency.py \
        --meta "$arm/eval_compare/audit_meta.json" \
        --out "$arm/latency_gate.json"; then
      latency_gate=pass
    else
      latency_gate=fail
    fi
    if [[ "$audio_gate" == pass && "$latency_gate" == pass ]]; then
      passed=$((passed + 1))
    fi
    echo -e "${lookahead}\t${discrete_weight}\tcomplete\t${audio_gate}\t${latency_gate}\t${code_change}\t${target_gain}" >> "$EXP_ROOT/status.tsv"
  done
done

echo "=== DISCRETE SWEEP COMPLETE $(date): completed=$completed passed=$passed ==="
cat "$EXP_ROOT/status.tsv"
echo "Listen under each lookahead_*_dw_*/eval_compare/{stock_xvc,joint_persona_mapper}/wavs"
if (( passed == 0 )); then
  echo "[result] no discrete arm passed the accent+voice gate" >&2
  exit 1
fi
