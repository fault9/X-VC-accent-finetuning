#!/usr/bin/env bash
# Compact, deployment-shaped comparison of the two justified mapper stages.
# Four arms: pre-prenet discrete vs post-prenet unified, each at 0/4 lookahead.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

cleanup() {
  status=$?
  trap - EXIT INT TERM
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  echo "=== PERSONA COMPARISON EXIT status=$status $(date) ==="
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  exit "$status"
}
trap cleanup EXIT INT TERM

DATASET_ROOT="${DATASET_ROOT:-data/hindi_asi_pristine_parallel_221}"
PAIRS_ROOT="${PAIRS_ROOT:-data/joint_persona_windows_asi_2400ms}"
REFERENCE="${REFERENCE:-data/eval_targets/ASI.wav}"
SOURCE_DIR="${SOURCE_DIR:-data/eval_sources_joint_persona_clean}"
CONFIG="${CONFIG:-configs/xvc.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/xvc.pt}"
EXP_ROOT="${EXP_ROOT:-exp/persona_mapper_comparison_asi}"
LOOKAHEADS="${LOOKAHEADS:-0 4}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-4}"
LR="${LR:-0.0002}"
DISCRETE_WEIGHT="${DISCRETE_WEIGHT:-0.5}"
CODE_MARGIN_WEIGHT="${CODE_MARGIN_WEIGHT:-0.25}"
DEVICE="${DEVICE:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-12000}"

for path in \
  "$DATASET_ROOT/manifests/train.jsonl" \
  "$DATASET_ROOT/manifests/val.jsonl" \
  "$REFERENCE" "$CONFIG" "$CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "[error] missing required file: $path" >&2; exit 1; }
done
[[ -d "$SOURCE_DIR" ]] || { echo "[error] missing source dir: $SOURCE_DIR" >&2; exit 1; }
source_count="$(find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.wav' | wc -l)"
(( source_count >= 10 )) || { echo "[error] need >=10 unseen-source wavs" >&2; exit 1; }

if ! free_gpu_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')"; then
  echo "[error] NVML unavailable; restart the GPU container" >&2
  exit 70
fi
[[ "$free_gpu_mib" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU reading" >&2; exit 70; }
(( free_gpu_mib >= MIN_FREE_GPU_MIB )) || {
  echo "[error] only ${free_gpu_mib} MiB free; need ${MIN_FREE_GPU_MIB}" >&2
  exit 1
}

mkdir -p "$EXP_ROOT" exp/run_logs
printf 'stage\tlookahead_frames\tstatus\taudio_gate\tlatency_gate\tcode_change\ttarget_code_gain\n' > "$EXP_ROOT/status.tsv"
echo "=== PERSONA MAPPER COMPARISON $(date) ==="
echo "stages      : pre-prenet discrete; post-prenet unified"
echo "lookaheads  : $LOOKAHEADS (20 ms/frame; no extra HMO window)"
echo "source eval : $SOURCE_DIR ($source_count unseen-source clips)"

if [[ ! -f "$PAIRS_ROOT/extraction_summary.json" ]]; then
  python scripts/extract_joint_persona_pairs.py \
    --dataset-root "$DATASET_ROOT" --config "$CONFIG" --ckpt "$CHECKPOINT" \
    --target-speaker ASI --out "$PAIRS_ROOT" --split all --device "$DEVICE" \
    2>&1 | tee "$EXP_ROOT/extract.log"
fi
python - "$PAIRS_ROOT/extraction_summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text(encoding="utf-8"))
if s.get("status") != "pass":
    raise SystemExit(f"[error] pair extraction failed: {s}")
policy = s.get("window_policy", {})
if s.get("schema_version", 0) < 2 or not policy.get("waveform_crop_before_frozen_encoder"):
    raise SystemExit("[error] extracted pairs are not raw-window-encoded schema v2")
if float(policy.get("window_seconds", 0)) != 2.4:
    raise SystemExit(f"[error] expected 2.4 s encoded windows, got {policy}")
if int(policy.get("window_frames", 0)) != 120:
    raise SystemExit(f"[error] expected 120-frame encoded windows, got {policy}")
for split in ("train", "val"):
    if not (p.parent / split).is_dir():
        raise SystemExit(f"[error] missing extracted split: {split}")
print("[pairs] PASS", s["splits"])
PY

passed=0
completed=0
for stage in preprenet_discrete postprenet_unified; do
  for lookahead in $LOOKAHEADS; do
    arm="$EXP_ROOT/${stage}_lookahead_${lookahead}f"
    [[ ! -e "$arm/best.pt" ]] || {
      echo "[error] refusing to overwrite $arm" >&2; exit 1;
    }
    echo "=== TRAIN stage=$stage lookahead=${lookahead}f $(date) ==="
    if [[ "$stage" == preprenet_discrete ]]; then
      python scripts/train_joint_persona_mapper.py \
        --train-dir "$PAIRS_ROOT/train" --val-dir "$PAIRS_ROOT/val" \
        --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm" \
        --steps "$STEPS" --batch "$BATCH" --lr "$LR" --layers 5 \
        --lookahead-frames "$lookahead" \
        --lambda-discrete-code "$DISCRETE_WEIGHT" \
        --lambda-code-margin "$CODE_MARGIN_WEIGHT" \
        --device "cuda:$DEVICE" 2>&1 | tee "$arm.train.log"
      mechanism="$(python - "$arm/best.pt" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu")["validation"]["all"]
print(f'{m["code_change_fraction"]}\t{m["aligned_target_code_gain"]}')
PY
)"
      code_change="${mechanism%%$'\t'*}"
      target_gain="${mechanism#*$'\t'}"
    else
      python scripts/train_postprenet_persona_mapper.py \
        --train-dir "$PAIRS_ROOT/train" --val-dir "$PAIRS_ROOT/val" \
        --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm" \
        --steps "$STEPS" --batch "$BATCH" --lr "$LR" --layers 5 \
        --lookahead-frames "$lookahead" --device "cuda:$DEVICE" \
        2>&1 | tee "$arm.train.log"
      code_change="na"
      target_gain="na"
    fi

    python scripts/eval_joint_persona_mapper.py \
      --source-dir "$SOURCE_DIR" --reference "$REFERENCE" \
      --mapper-ckpt "$arm/best.pt" --config "$CONFIG" --ckpt "$CHECKPOINT" \
      --out "$arm/eval_compare" --require-unseen-source --min-unseen-speakers 2 \
      --training-manifest "$DATASET_ROOT/manifests/train.jsonl" \
      --streaming --chunk-ms 2400 --current-ms 120 --smooth-ms 20 --future-ms 100 \
      --device "$DEVICE" 2>&1 | tee "$arm.eval.log"
    python scripts/score_xvc_accent_stream_audit.py \
      --audit-root "$arm/eval_compare" --reference "$REFERENCE" \
      --config "$CONFIG" --ckpt "$CHECKPOINT" --out "$arm/eval_metrics" \
      --xvc-device "$DEVICE" 2>&1 | tee "$arm.score.log"

    completed=$((completed + 1))
    if python scripts/gate_joint_persona_mapper.py \
        --summary "$arm/eval_metrics/condition_summary.csv" --out "$arm/gate.json"; then
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
    if [[ "$audio_gate" == pass && "$latency_gate" == pass ]]; then
      passed=$((passed + 1))
    fi
    printf '%s\t%s\tcomplete\t%s\t%s\t%s\t%s\n' \
      "$stage" "$lookahead" "$audio_gate" "$latency_gate" \
      "$code_change" "$target_gain" >> "$EXP_ROOT/status.tsv"
  done
done

echo "=== COMPARISON COMPLETE $(date): completed=$completed passed=$passed ==="
cat "$EXP_ROOT/status.tsv"
echo "Listen to matched files under each */eval_compare/{stock_xvc,joint_persona_mapper}/wavs"
(( passed > 0 )) || { echo "[result] no arm passed the accent+voice gate" >&2; exit 1; }
