#!/bin/bash
# Fine-tune X-VC on one speaker group (arabic / spanish / chinese / hindi / native / joint),
# "Option A": warm-start the released checkpoint, freeze everything except
# acoustic_converter + prenet, self-reconstruction objective.
#
# Warm-starts from the released X-VC checkpoint with a FRESH optimizer
# (--checkpoint loads weights only; --resume_step 0 means no optimizer/step restore).
# Single-GPU DeepSpeed by default (the training loop relies on DeepSpeed's engine).
#
# Usage:
#   bash scripts/finetune.sh --accent arabic
#   bash scripts/finetune.sh --accent joint                    # single joint checkpoint
#   bash scripts/finetune.sh --accent all                      # arabic,spanish,chinese,hindi,native sequentially
#   bash scripts/finetune.sh --accent arabic --lr 3e-5         # LR variant (thin-slice A/B)
#   bash scripts/finetune.sh --accent chinese --resume_step 1500   # resume a run
#
# Every run writes exp/<name>/run_meta.json (git commit, checkpoint+manifest sha256,
# seed, CLI, pip freeze) before training starts — reproducibility is not optional.
#
# Run from the repository root. Outputs go to exp/finetune_<accent>[_lr<lr>]/.

set -euo pipefail

accent="arabic"
checkpoint="ckpts/xvc.pt"
gpus=1
resume_step=0
port=10201
config=""
log_dir=""
seed=1234
lr=""
wandb=0
# DataLoader workers: 0 avoids /dev/shm entirely — required on containers with
# the 64MB Docker default (workers pass batches through shared memory).
num_workers=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accent)       accent="$2"; shift 2 ;;
    --checkpoint)   checkpoint="$2"; shift 2 ;;
    --gpus)         gpus="$2"; shift 2 ;;
    --resume_step)  resume_step="$2"; shift 2 ;;
    --port)         port="$2"; shift 2 ;;
    --config)       config="$2"; shift 2 ;;
    --log_dir)      log_dir="$2"; shift 2 ;;
    --seed)         seed="$2"; shift 2 ;;
    --lr)           lr="$2"; shift 2 ;;
    --num_workers)  num_workers="$2"; shift 2 ;;
    --wandb)        wandb=1; shift 1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"

# --- HF publishing link (one-time, optional) ---------------------------------
# So the post-training publish step just works: link your repo and your key once,
# saved to .hf_publish.env (gitignored, chmod 600) and auto-read by
# scripts/publish_checkpoint.py. Env vars win; enter to skip; never blocks
# unattended runs (60s timeout / no tty -> skip).
hf_env_file="$root_dir/.hf_publish.env"
# shellcheck disable=SC1090
[[ -f "$hf_env_file" ]] && source "$hf_env_file"
if { [[ -z "${XVC_PUBLISH_REPO:-}" ]] || [[ -z "${HF_TOKEN:-}" ]]; } && [[ -e /dev/tty ]]; then
  echo ""
  echo "HF publishing link (for publish_checkpoint.py after training; enter to skip)"
  _repo=""; _token=""
  if [[ -z "${XVC_PUBLISH_REPO:-}" ]]; then
    read -r -t 60 -p "  Link your repo (<user>/<repo>, e.g. fault9/xvc-accent-finetunes): " _repo < /dev/tty 2>/dev/tty || _repo=""
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    read -r -s -t 60 -p "  Link your key  (HF write token hf_..., input hidden): " _token < /dev/tty 2>/dev/tty || _token=""
    echo ""
  fi
  if [[ -n "$_repo" || -n "$_token" ]]; then
    touch "$hf_env_file" && chmod 600 "$hf_env_file"
    [[ -n "$_repo"  ]] && { echo "XVC_PUBLISH_REPO=$_repo" >> "$hf_env_file"; export XVC_PUBLISH_REPO="$_repo"; }
    [[ -n "$_token" ]] && { echo "HF_TOKEN=$_token" >> "$hf_env_file"; export HF_TOKEN="$_token"; }
    echo "  saved to $hf_env_file (gitignored — never commit this file)"
  fi
fi

# --accent all: the five per-group runs, sequentially, sharing all other flags.
if [[ "$accent" == "all" ]]; then
  for a in arabic spanish chinese hindi native; do
    extra=()
    [[ -n "$lr" ]] && extra+=(--lr "$lr")
    [[ "$wandb" -eq 1 ]] && extra+=(--wandb)
    bash "$script_dir/finetune.sh" --accent "$a" --checkpoint "$checkpoint" \
      --gpus "$gpus" --port "$port" --seed "$seed" --num_workers "$num_workers" "${extra[@]}"
  done
  exit 0
fi

[[ -z "$config" ]]  && config="configs/finetune_${accent}.yaml"
[[ -z "$log_dir" ]] && log_dir="exp/finetune_${accent}"

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config (accent='$accent')" >&2
  exit 1
fi

# LR variant: write a patched copy of the config (NOT an overlay — load_config
# resolves base_config only one level deep, so overlay-on-overlay would silently
# drop everything inherited from configs/xvc.yaml).
if [[ -n "$lr" ]]; then
  log_dir="${log_dir}_lr${lr}"
  mkdir -p "$log_dir"
  variant="$log_dir/config_lr.yaml"
  python - "$config" "$variant" "$lr" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
OmegaConf.update(cfg, "model.generator.optim_conf.lr", float(sys.argv[3]))
OmegaConf.save(cfg, sys.argv[2])
print(f"lr variant config: {sys.argv[2]} (lr={sys.argv[3]})")
PY
  config="$variant"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$log_dir"
tag="$(date +'%Y%m%d_%H%M%S')"

# On a fresh run (resume_step 0) warm-start from the released checkpoint.
# On a resume (resume_step > 0) load from log_dir/ckpt instead — no --checkpoint.
checkpoint_arg=""
if [[ "$resume_step" -eq 0 ]]; then
  if [[ ! -f "$checkpoint" ]]; then
    echo "Pretrained checkpoint not found: $checkpoint" >&2
    echo "Download the released X-VC checkpoint to ckpts/ (see docs/finetuning.md)." >&2
    exit 1
  fi
  checkpoint_arg="--checkpoint $checkpoint"
fi

wandb_arg=""
[[ "$wandb" -eq 1 ]] && wandb_arg="--enable_wandb"

echo "accent      : $accent"
echo "config      : $config"
echo "log_dir     : $log_dir"
echo "gpus        : $gpus"
echo "seed        : $seed"
echo "lr override : ${lr:-<config default>}"
echo "resume_step : $resume_step"
echo "checkpoint  : ${checkpoint_arg:-<resume from $log_dir/ckpt>}"

# Reproducibility record — BEFORE training starts, so even a crashed run is traceable.
python scripts/write_run_meta.py \
    --log_dir "$log_dir" \
    --config "$config" \
    ${checkpoint_arg:+--checkpoint "$checkpoint"} \
    --seed "$seed" \
    --cli "finetune.sh --accent $accent --seed $seed${lr:+ --lr $lr} --resume_step $resume_step (tag $tag)"

torchrun --nnodes=1 --nproc_per_node="${gpus}" --master_port="${port}" \
    -m bins.train \
    --config "${config}" \
    --log_dir "${log_dir}" \
    --train_engine deepspeed \
    --deepspeed_config configs/ds_stage2.json \
    --resume_step "${resume_step}" \
    --seed "${seed}" \
    --num_workers "${num_workers}" \
    --project "x-vc-finetune" \
    --date "${tag}" \
    ${wandb_arg} \
    ${checkpoint_arg}
