#!/bin/bash
# Merge-time accent dial on THE v2 persona (stackdistill l10, r4, step 2000):
# fold the LoRA delta at x1.0 / x1.25 / x1.5 into three stock-architecture
# checkpoints, then eval each on the reserved plan for numbers + A/B wavs.
# The accent-vs-texture trade is measured linear: expect deeper coloring and
# ~0.1-0.2 MOS cost per +0.25 of scale. Ears pick the serving point.
#   nohup bash scripts/run_v2_scale_ladder.sh > exp/run_logs/v2_scale_ladder.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate xvc

V2_CFG=configs/finetune_stackdistill_hindi_asi_l10_lora_r4_alpha16.yaml
V2_CKPT=exp/finetune_stackdistill_hindi_asi_l10_lora_r4_alpha16/ckpt/002000.pt
SERVE_CFG=configs/finetune_crosspair_hindi_latent_400.yaml   # non-LoRA, same arch

[ -f ckpts/xvc_persona_hindi_asi_v2.pt ] || \
  python scripts/merge_lora.py --config "$V2_CFG" --ckpt "$V2_CKPT" \
    --out ckpts/xvc_persona_hindi_asi_v2.pt

python scripts/merge_lora.py --config "$V2_CFG" --ckpt "$V2_CKPT" \
  --lora-scale 1.25 --out ckpts/xvc_persona_hindi_asi_v2_x125.pt

python scripts/merge_lora.py --config "$V2_CFG" --ckpt "$V2_CKPT" \
  --lora-scale 1.5 --out ckpts/xvc_persona_hindi_asi_v2_x150.pt

for scale in x100 x125 x150; do
  case "$scale" in
    x100) ckpt=ckpts/xvc_persona_hindi_asi_v2.pt ;;
    x125) ckpt=ckpts/xvc_persona_hindi_asi_v2_x125.pt ;;
    x150) ckpt=ckpts/xvc_persona_hindi_asi_v2_x150.pt ;;
  esac
  python scripts/eval_checkpoints.py run \
    --run-dir exp/persona_v2_scale_eval \
    --steps 999999 \
    --config "$SERVE_CFG" \
    --include-base "$ckpt" \
    --source-dir data/eval_sources_reserved \
    --targets-dir data/eval_targets \
    --evaluation-plan configs/eval_hindi_native_to_asi.json \
    --mos --accent-clf \
    --out "exp/persona_v2_scale_eval/$scale"
done

echo "ladder done; per-scale metrics + samples under exp/persona_v2_scale_eval/{x100,x125,x150}"
