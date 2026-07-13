#!/usr/bin/env bash
# Round-3 stack probes around the EAR-VALIDATED winner: bridge x1.0 through an
# accented renderer (b10_l10: "more indian than v1", 32/40 shifted incl 3
# indian, MOS 3.02). Bridge stays at x1.0 -- the x1.2 rows paid MOS without
# adding indian. Two depth levers from here:
#   * raise the renderer's LoRA scale (alone it cost only -0.18/-0.35 MOS at
#     x1.5/x2.0),
#   * swap the renderer for the V1 STUDENT (cleanest accent renderer we own,
#     3.43 vs the latent teacher's 3.36), optionally scaled -- its delta was
#     learned from clean targets under the benign objective, so its ray should
#     price accent cheaper than the latent teacher's.
#
# Usage (container, conda xvc, repo root):  bash scripts/probe_stack_round3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BRIDGE=${BRIDGE:-exp/accentbridge_asi_l0plus/bridge.pt}
T_CKPT=exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5/ckpt/000100.pt
T_CFG=configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16_lr5e-5.yaml
S_CKPT=exp/finetune_selfdistill_hindi_asi_lora_r8/ckpt/002000.pt
S_CFG=configs/finetune_selfdistill_hindi_asi_lora_r8.yaml
SRC='data/distill_sources_asi/*.wav'
REF=data/eval_targets/ASI.wav

render() { # <cfg> <ckpt> <lora-scale> <out>
  python scripts/make_distill_dataset.py \
    --source-glob "$SRC" --bridge-ckpt "$BRIDGE" --delta-scale 1.0 \
    --config "$1" --ckpt "$2" --lora-scale "$3" \
    --reference "$REF" --out "$4" --limit 40 --device 0
}

render "$T_CFG" "$T_CKPT" 1.5 data/sd_probe_b10_l15
render "$T_CFG" "$T_CKPT" 2.0 data/sd_probe_b10_l20
render "$S_CFG" "$S_CKPT" 1.0 data/sd_probe_b10_s10
render "$S_CFG" "$S_CKPT" 1.5 data/sd_probe_b10_s15
render "$S_CFG" "$S_CKPT" 2.0 data/sd_probe_b10_s20

python scripts/gate_teacher_renders.py \
  data/sd_probe_b10_l10/wavs \
  data/sd_probe_b10_l15/wavs \
  data/sd_probe_b10_l20/wavs \
  data/sd_probe_b10_s10/wavs \
  data/sd_probe_b10_s15/wavs \
  data/sd_probe_b10_s20/wavs
