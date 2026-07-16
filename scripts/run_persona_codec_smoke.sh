#!/usr/bin/env bash
# Guarded two-pair GPU smoke for the persona codec teacher path:
#   extract 2 pairs -> geometry report -> train 2 steps -> greedy decode
#   (inside validation) -> render 2 eval sources through the frozen renderer.
# Plumbing only: 2 steps cannot produce accent; success means every stage ran
# and every fail-closed check held. X-VC stays frozen throughout.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

cleanup() {
  status=$?
  trap - EXIT INT TERM
  echo "=== PERSONA CODEC SMOKE EXIT status=$status $(date) ==="
  exit "$status"
}
trap cleanup EXIT INT TERM

DATASET_ROOT="${DATASET_ROOT:-data/hindi_asi_pristine_parallel_221}"
REFERENCE="${REFERENCE:-data/eval_targets/ASI.wav}"
TARGET_PERSONA="${TARGET_PERSONA:-ASI}"
SOURCE_DIR="${SOURCE_DIR:-data/eval_sources_joint_persona_clean}"
CONFIG="${CONFIG:-configs/xvc.yaml}"
CHECKPOINT="${CHECKPOINT:-ckpts/xvc.pt}"
EXP_ROOT="${EXP_ROOT:-exp/persona_codec_smoke}"
MAX_PAIRS="${MAX_PAIRS:-2}"
STEPS="${STEPS:-2}"
DEVICE="${DEVICE:-0}"

for required in \
  "$DATASET_ROOT/manifests/train.jsonl" \
  "$DATASET_ROOT/manifests/val.jsonl" \
  "$REFERENCE" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required file: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[error] missing eval source dir: $SOURCE_DIR" >&2
  exit 1
fi

PAIRS_DIR="$EXP_ROOT/pairs"
TEACHER_DIR="$EXP_ROOT/teacher"
RENDER_DIR="$EXP_ROOT/render"

echo "=== [1/4] extract $MAX_PAIRS pairs ==="
python scripts/extract_persona_codec_tokens.py \
  --data-root "$DATASET_ROOT" \
  --config "$CONFIG" --ckpt "$CHECKPOINT" \
  --out "$PAIRS_DIR" \
  --device "$DEVICE" \
  --max-pairs "$MAX_PAIRS" \
  --overwrite

echo "=== [2/4] geometry report ==="
python scripts/report_persona_token_geometry.py \
  --pairs "$PAIRS_DIR" \
  --out "$EXP_ROOT/geometry"

echo "=== [3/4] train $STEPS steps ==="
python scripts/train_persona_codec_teacher.py \
  --pairs "$PAIRS_DIR" \
  --out "$TEACHER_DIR" \
  --steps "$STEPS" \
  --batch-size 2 \
  --val-every "$STEPS" \
  --warmup-steps 1 \
  --decode-val-items 1

echo "=== [4/4] render 2 sources (stock vs teacher) ==="
python scripts/render_persona_codec_teacher.py \
  --teacher-ckpt "$TEACHER_DIR/last.pt" \
  --source-dir "$SOURCE_DIR" \
  --reference "$REFERENCE" \
  --target-persona "$TARGET_PERSONA" \
  --config "$CONFIG" --ckpt "$CHECKPOINT" \
  --out "$RENDER_DIR" \
  --device "$DEVICE" \
  --max-sources 2 \
  --min-sources 0 \
  --overwrite

echo "[smoke] OK - artifacts under $EXP_ROOT (plumbing only, not an accent result)"
