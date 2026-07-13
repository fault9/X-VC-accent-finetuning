#!/usr/bin/env bash
# Bridge-teacher probes: does the AccentBridge (semantic-level delta) reach
# more indian per unit texture damage than lora-scaling the converter LoRA?
# Renders the SAME first-40 distill_sources_asi clips as the LoRA probes at
# delta-scale 1.2 (v1 operating point) / 1.5 / 2.0, gates them, and also gates
# the existing v1 bridge dataset renders for reference. Same decision rule as
# the LoRA probes: max indian labels at MOS mean >= ~2.7, ears approve.
#
# Usage (container, conda xvc, repo root):  bash scripts/probe_bridge_teachers.sh
# Override the bridge checkpoint with:  BRIDGE=exp/<run>/bridge.pt bash scripts/...
set -euo pipefail
cd "$(dirname "$0")/.."

BRIDGE=${BRIDGE:-exp/accentbridge_asi_l0plus/bridge.pt}
if [ ! -f "$BRIDGE" ]; then
  echo "[error] bridge checkpoint not found: $BRIDGE -- candidates:"
  ls -d exp/accentbridge* 2>/dev/null || true
  exit 1
fi
SRC='data/distill_sources_asi/*.wav'
REF=data/eval_targets/ASI.wav

render() { # <delta-scale> <out>
  python scripts/make_distill_dataset.py \
    --source-glob "$SRC" --bridge-ckpt "$BRIDGE" --delta-scale "$1" \
    --reference "$REF" --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
    --out "$2" --limit 40 --device 0
}

render 1.2 data/sd_probe_bridge_x12
render 1.5 data/sd_probe_bridge_x15
render 2.0 data/sd_probe_bridge_x20

python scripts/gate_teacher_renders.py \
  data/distill_hindi_asi/wavs \
  data/sd_probe_bridge_x12/wavs \
  data/sd_probe_bridge_x15/wavs \
  data/sd_probe_bridge_x20/wavs
