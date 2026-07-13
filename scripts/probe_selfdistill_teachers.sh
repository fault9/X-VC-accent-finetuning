#!/usr/bin/env bash
# Render 40-clip probes of candidate self-distill TEACHERS along the
# accent-strength vs texture-damage axis, then gate them all side by side.
#
# The far anchor of that axis is r1@1000 lr1e-4 (indian 7/20 but MOS 1.62,
# ear-verdict "very robotic") -- too wrecked to distill from directly. These
# probes sample the middle: later r4 steps, and the LoRA delta of the EARLY
# clean checkpoints amplified at render time (--lora-scale). The r1 x2/x3
# probes are the cleanest test of "accent is rank-1-expressible": one learned
# direction, extrapolated, without the 900 extra steps of objective damage.
#
# Usage (container, conda xvc, repo root):  bash scripts/probe_selfdistill_teachers.sh
set -euo pipefail
cd "$(dirname "$0")/.."

R4_RUN=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5
R4_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml
R1_RUN=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16_lr5e-5
# topology-only (r=1) -- lr in the config is a training-time knob, irrelevant here:
R1_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16.yaml
SRC='data/distill_sources_asi/*.wav'
REF=data/eval_targets/ASI.wav

render() { # <run> <cfg> <step> <scale> <out>
  python scripts/make_selfdistill_dataset.py \
    --run-dir "$1" --config "$2" --step "$3" --lora-scale "$4" --out "$5" \
    --source-glob "$SRC" --reference "$REF" --limit 40 --device 0
}

render "$R4_RUN" "$R4_CFG" 100 1.5 data/sd_probe_r4s100_x15
render "$R4_RUN" "$R4_CFG" 100 2.0 data/sd_probe_r4s100_x20
render "$R4_RUN" "$R4_CFG" 300 1.0 data/sd_probe_r4s300
render "$R4_RUN" "$R4_CFG" 600 1.0 data/sd_probe_r4s600
render "$R1_RUN" "$R1_CFG" 100 2.0 data/sd_probe_r1s100_x20
render "$R1_RUN" "$R1_CFG" 100 3.0 data/sd_probe_r1s100_x30

# baseline first (the v1 teacher: r4@100 x1.0), then the candidates
python scripts/gate_teacher_renders.py \
  data/selfdistill_hindi_asi/wavs \
  data/sd_probe_r4s100_x15/wavs \
  data/sd_probe_r4s100_x20/wavs \
  data/sd_probe_r4s300/wavs \
  data/sd_probe_r4s600/wavs \
  data/sd_probe_r1s100_x20/wavs \
  data/sd_probe_r1s100_x30/wavs
