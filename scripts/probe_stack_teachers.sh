#!/usr/bin/env bash
# STACK probes: bridge delta (segmental accent, ear-verified most-Indian but
# choppy/noisy alone at x1.5) rendered THROUGH the accented LoRA renderer
# (r4@100 lr5e-5: clean but mild alone). Hypothesis: two moderate pushes at two
# different levels beat one extreme push at either -- the bridge supplies the
# phone-level shift the classifier undercounts, the LoRA supplies the
# target-conditioned rendering, and neither is driven to its breaking point.
# Reference points from earlier rounds, all on the same first-40 clips:
#   bridge x1.5 alone:  MOS 2.10 (ears: most Indian, "very choppy and noisy")
#   LoRA  x3.0 alone:   indian 12/40, MOS 2.60
#   LoRA  x1.0 alone:   england 28/40, MOS 3.37 (v1 teacher)
#
# Usage (container, conda xvc, repo root):  bash scripts/probe_stack_teachers.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BRIDGE=${BRIDGE:-exp/accentbridge_asi_l0plus/bridge.pt}
LORA_CKPT=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt
LORA_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml
SRC='data/distill_sources_asi/*.wav'
REF=data/eval_targets/ASI.wav

render() { # <delta-scale> <lora-scale> <out>
  python scripts/make_distill_dataset.py \
    --source-glob "$SRC" --bridge-ckpt "$BRIDGE" --delta-scale "$1" \
    --config "$LORA_CFG" --ckpt "$LORA_CKPT" --lora-scale "$2" \
    --reference "$REF" --out "$3" --limit 40 --device 0
}

render 1.0 1.0 data/sd_probe_b10_l10
render 1.2 1.0 data/sd_probe_b12_l10
render 1.2 1.5 data/sd_probe_b12_l15
render 1.5 1.0 data/sd_probe_b15_l10
render 1.2 2.0 data/sd_probe_b12_l20

python scripts/gate_teacher_renders.py \
  data/sd_probe_b10_l10/wavs \
  data/sd_probe_b12_l10/wavs \
  data/sd_probe_b12_l15/wavs \
  data/sd_probe_b15_l10/wavs \
  data/sd_probe_b12_l20/wavs \
  data/sd_probe_bridge_x15/wavs \
  data/sd_probe_r4s100_x300/wavs
