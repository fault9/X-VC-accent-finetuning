#!/usr/bin/env bash
# Prepare genuine phone-span metadata without warping, then run a controlled
# pronunciation-loss sweep. Existing shards/checkpoints are never overwritten.
set -euo pipefail

PAIRS_ROOT="${PAIRS_ROOT:-data/accentbridge_pairs}"
SOURCE_ALIGN_DIR="${SOURCE_ALIGN_DIR:-data/mfa_hindi_400/mfa_align/source}"
TARGET_ALIGN_DIR="${TARGET_ALIGN_DIR:-data/mfa_hindi_400/mfa_align/target}"
PHONE_PAIRS_ROOT="${PHONE_PAIRS_ROOT:-data/accentbridge_pairs_phone_unwarped}"
EXP_ROOT="${EXP_ROOT:-exp/accentbridge_phoneaware_unwarped}"
TARGET_SPEAKER="${TARGET_SPEAKER:-ASI}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-cuda}"

for path in "$PAIRS_ROOT/train" "$PAIRS_ROOT/val" \
            "$SOURCE_ALIGN_DIR" "$TARGET_ALIGN_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "[error] missing required directory: $path" >&2
    exit 1
  fi
done

mkdir -p "$EXP_ROOT"

python scripts/annotate_accentbridge_phone_supervision.py \
  --pairs-root "$PAIRS_ROOT" \
  --source-align-dir "$SOURCE_ALIGN_DIR" \
  --target-align-dir "$TARGET_ALIGN_DIR" \
  --out "$PHONE_PAIRS_ROOT" \
  --target-speaker "$TARGET_SPEAKER" \
  --tier phones \
  --min-label-match 0.90 \
  --min-pair-coverage 0.80 \
  --min-duration-ratio 0.5 \
  --max-duration-ratio 2.0 \
  2>&1 | tee "$EXP_ROOT/prepare_phone_supervision.log"

for lambda in 0.1 0.25 0.5; do
  tag="${lambda/./p}"
  out="$EXP_ROOT/lambda_phone_${tag}"
  if [[ -e "$out/bridge.pt" ]]; then
    echo "[error] refusing to overwrite completed arm: $out" >&2
    exit 1
  fi
  echo "=== PHONE-AWARE ARM lambda_phone=$lambda $(date) ==="
  python scripts/train_accentbridge.py \
    --train-dir "$PHONE_PAIRS_ROOT/train" \
    --val-dir "$PHONE_PAIRS_ROOT/val" \
    --target-speaker "$TARGET_SPEAKER" \
    --phone-aware \
    --lambda-phone "$lambda" \
    --lookahead-frames 0 \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr 3e-4 \
    --lambda-id 0.5 \
    --lambda-smooth 0.1 \
    --lambda-delta 0.01 \
    --phone-std-weight 0.25 \
    --device "$DEVICE" \
    --out "$out" \
    2>&1 | tee "$EXP_ROOT/lambda_phone_${tag}.log"
done

python - "$EXP_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n=== PHONE-AWARE SWEEP SUMMARY ===")
for path in sorted(root.glob("lambda_phone_*/train_metrics.json")):
    obj = json.loads(path.read_text())
    best = obj.get("best") or {}
    print(path.parent.name,
          "post_l2=", best.get("post_l2"),
          "phone_post_l2=", best.get("phone_post_l2"),
          "phone_gap_closed=", best.get("phone_gap_closed_l2"),
          "identity_drift=", best.get("identity_drift"))
PY
