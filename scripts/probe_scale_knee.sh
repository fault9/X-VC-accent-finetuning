#!/usr/bin/env bash
# Round 2 teacher probes: locate the KNEE of the lora-scale axis, where indian
# labels ramp up before texture collapses. Round 1 bracketed it -- r1@100 went
# from indian 1/40 at x2.0 to 17/40 at x3.0 (MOS 2.92 -> 2.34); r4@100 was at
# indian 1/40, MOS 3.02 at x2.0 with a shallower MOS-per-scale slope. Later
# steps (r4@300/600) are eliminated: labels regressed while MOS collapsed.
# Target: max indian with MOS >= ~2.7 (student smoothing adds ~+0.1).
#
# Usage (container, conda xvc, repo root):  bash scripts/probe_scale_knee.sh
set -euo pipefail
cd "$(dirname "$0")/.."

R4_RUN=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5
R4_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml
R1_RUN=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16_lr5e-5
R1_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16.yaml
SRC='data/distill_sources_asi/*.wav'
REF=data/eval_targets/ASI.wav

render() { # <run> <cfg> <scale> <out>
  python scripts/make_selfdistill_dataset.py \
    --run-dir "$1" --config "$2" --step 100 --lora-scale "$3" --out "$4" \
    --source-glob "$SRC" --reference "$REF" --limit 40 --device 0
}

render "$R1_RUN" "$R1_CFG" 2.25 data/sd_probe_r1s100_x225
render "$R1_RUN" "$R1_CFG" 2.5  data/sd_probe_r1s100_x250
render "$R1_RUN" "$R1_CFG" 2.75 data/sd_probe_r1s100_x275
render "$R4_RUN" "$R4_CFG" 2.5  data/sd_probe_r4s100_x250
render "$R4_RUN" "$R4_CFG" 3.0  data/sd_probe_r4s100_x300

python scripts/gate_teacher_renders.py \
  data/sd_probe_r1s100_x20/wavs \
  data/sd_probe_r1s100_x225/wavs \
  data/sd_probe_r1s100_x250/wavs \
  data/sd_probe_r1s100_x275/wavs \
  data/sd_probe_r1s100_x30/wavs \
  data/sd_probe_r4s100_x250/wavs \
  data/sd_probe_r4s100_x300/wavs
