#!/usr/bin/env bash
# Persona-specific naturalness-only LoRA sweep for four fixed VCTK references.
#
# Each run sees exactly one target speaker and one fixed reference. The same
# frozen stock X-VC accepts unseen source speakers at evaluation time. Adapters
# are selected/loaded once per persona session; they are not run as extra models.
#
# Container usage:
#   nohup bash scripts/run_vctk_persona_naturalness_sweep.sh \
#     > exp/run_logs/vctk_persona_naturalness_sweep.out 2>&1 &

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"

data_root="${DATA_ROOT:-data/vctk_naturalness_4voice}"
template="${TEMPLATE:-configs/finetune_vctk_persona_naturalness.yaml}"
checkpoint="${CHECKPOINT:-ckpts/xvc.pt}"
exp_root="${EXP_ROOT:-exp/vctk_persona_naturalness_sweep}"
total_steps="${TOTAL_STEPS:-300}"
steps="${EVAL_STEPS:-50,100,150,200,250,300}"
max_sources="${MAX_SOURCES:-0}"
run_offline="${RUN_OFFLINE:-0}"
smoke="${SMOKE:-0}"
full_eval="${FULL_EVAL:-0}"
if [[ "$full_eval" == "1" ]]; then
  source_dir="$data_root/eval_sources"
else
  source_dir="$data_root/eval_sources_scout"
fi
target_dir="$data_root/eval_targets"
mkdir -p "$exp_root/configs" exp/run_logs

personas=(
  "female_high_p240_10s"
  "female_low_p225_10s"
  "male_high_p273_10s"
  "male_low_p274_10s"
)

persona_filter="${PERSONA_FILTER:-}"
if [[ -n "$persona_filter" ]]; then
  case " ${personas[*]} " in
    *" $persona_filter "*) personas=("$persona_filter") ;;
    *) echo "[error] unknown PERSONA_FILTER=$persona_filter" >&2; exit 2 ;;
  esac
fi

# Low-data rank advice: compare r=1/2/4 at fixed alpha=16. The second r2 arm
# tests the next conservative LR; the final arm is a bounded wider-converter
# control to determine whether conditioning-only capacity is too restrictive.
arms=(
  "cond_r1_a16_lr5e5|1|16|0.00005|conditioning"
  "cond_r2_a16_lr5e5|2|16|0.00005|conditioning"
  "cond_r4_a16_lr5e5|4|16|0.00005|conditioning"
  "cond_r2_a16_lr1e4|2|16|0.00010|conditioning"
  "wide_r1_a16_lr5e5|1|16|0.00005|wide"
)

if [[ "$smoke" == "1" ]]; then
  personas=("${personas[0]}")
  arms=("${arms[0]}")
  total_steps=2
  steps=2
  max_sources=2
  echo "[smoke] one persona, one arm, two steps, two evaluation sources"
fi

active_pgid=""
cleanup() {
  local status=$?
  if [[ -n "${active_pgid:-}" ]] && kill -0 -- "-$active_pgid" 2>/dev/null; then
    echo "[cleanup] TERM process group $active_pgid" >&2
    kill -TERM -- "-$active_pgid" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$active_pgid" 2>/dev/null || true
  fi
  echo "=== PERSONA NATURALNESS SWEEP EXIT status=$status $(date) ==="
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_managed() {
  local status
  setsid "$@" &
  active_pgid="$!"
  if wait "$active_pgid"; then status=0; else status=$?; fi
  active_pgid=""
  return "$status"
}

echo "=== VCTK PERSONA NATURALNESS SWEEP START $(date) ==="
echo "dataset       : $data_root"
echo "checkpoint    : $checkpoint"
echo "output        : $exp_root"
echo "steps         : $total_steps"
echo "source set     : $source_dir"
echo "source cap     : $max_sources (0 = all in source set)"
echo "objective     : one target persona per adapter; self-reconstruction only"

test -f "$template"
test -f "$checkpoint"
test -d "$source_dir"
test -d "$target_dir"
for persona in "${personas[@]}"; do
  test -f "$data_root/manifests/by_persona/$persona/train.jsonl"
  test -f "$data_root/manifests/by_persona/$persona/val.jsonl"
  test -f "$data_root/evaluation_plans/$persona.json"
  test -f "$target_dir/$persona.wav"
done

if pgrep -af 'torchrun|deepspeed|bins.train|eval_checkpoints'; then
  echo "[error] an X-VC train/eval process is already running" >&2
  exit 1
fi
nvidia-smi
python scripts/validate_crosspairs.py --data-root "$data_root" --min-duration 2.4

status_file="$exp_root/status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf 'timestamp\tpersona\tarm\tstage\tstatus\n' > "$status_file"
fi
printf -v final_step '%06d' "$total_steps"

for persona in "${personas[@]}"; do
  train_manifest="$data_root/manifests/by_persona/$persona/train.jsonl"
  val_manifest="$data_root/manifests/by_persona/$persona/val.jsonl"
  eval_plan="$data_root/evaluation_plans/$persona.json"

  for spec in "${arms[@]}"; do
    IFS='|' read -r name rank alpha lr scope <<< "$spec"
    config="$exp_root/configs/${persona}__${name}.yaml"
    run_dir="$exp_root/$persona/$name"
    mkdir -p "$run_dir"

    python - "$template" "$config" "$rank" "$alpha" "$lr" "$scope" \
      "$train_manifest" "$val_manifest" "$total_steps" "$smoke" <<'PY'
import sys
from omegaconf import OmegaConf

(source, destination, rank, alpha, lr, scope,
 train_manifest, val_manifest, total_steps, smoke) = sys.argv[1:]
cfg = OmegaConf.load(source)
cfg.datasets.train = [train_manifest]
cfg.datasets.val = [val_manifest]
cfg.total_step = int(total_steps)
if smoke == "1":
    cfg.log_interval = 1
    cfg.val_interval = int(total_steps)
    cfg.save_interval = int(total_steps)
    cfg.keep_interval = int(total_steps)
cfg.model.generator.lora.r = int(rank)
cfg.model.generator.lora.alpha = int(alpha)
cfg.model.generator.optim_conf.lr = float(lr)
if scope == "wide":
    cfg.model.generator.lora.include = [
        "attn.", "ff_x.ff", "ff_c.ff", "attn_norm_x.linear", "norm_out.linear"
    ]
with open(destination, "w", encoding="utf-8") as handle:
    handle.write(OmegaConf.to_yaml(cfg))
print(
    f"[config] {destination}: r={rank} alpha={alpha} lr={lr} "
    f"scope={scope} train={train_manifest}"
)
PY

    final="$run_dir/ckpt/${final_step}.pt"
    if [[ ! -f "$final" ]]; then
      echo "=== TRAIN persona=$persona arm=$name $(date) ==="
      run_managed torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        -m bins.train \
        --config "$config" \
        --log_dir "$run_dir" \
        --checkpoint "$checkpoint" \
        --train_engine torch_ddp \
        --num_workers 0 \
        --timeout 900 \
        --seed 20260718
      printf '%s\t%s\t%s\ttrain\tpassed\n' \
        "$(date -Iseconds)" "$persona" "$name" >> "$status_file"
    else
      echo "[resume] final checkpoint exists: $final"
      printf '%s\t%s\t%s\ttrain\tskipped_complete\n' \
        "$(date -Iseconds)" "$persona" "$name" >> "$status_file"
    fi

    modes=(streaming)
    if [[ "$run_offline" == "1" ]]; then modes=(offline streaming); fi
    for mode in "${modes[@]}"; do
      eval_dir="$run_dir/eval_$mode"
      if [[ -f "$eval_dir/metrics.csv" ]]; then
        echo "[resume] completed $mode metrics exist: $eval_dir"
        continue
      fi
      echo "=== EVAL persona=$persona arm=$name mode=$mode $(date) ==="
      command=(
        python scripts/eval_checkpoints.py run
        --run-dir "$run_dir"
        --source-dir "$source_dir"
        --targets-dir "$target_dir"
        --evaluation-plan "$eval_plan"
        --out "$eval_dir"
        --steps "$steps"
        --include-base "$checkpoint"
        --mos
        --mask-target-condition
        --device 0
      )
      if [[ "$max_sources" != "0" ]]; then command+=(--max-sources "$max_sources"); fi
      if [[ "$mode" == "streaming" ]]; then
        command+=(--streaming --chunk 2400 --current 120 --smooth 20 --future 100)
      fi
      run_managed "${command[@]}"
      python scripts/summarize_naturalness_eval.py \
        "$eval_dir/metrics.csv" --out "$eval_dir/naturalness_by_target.csv"
      printf '%s\t%s\t%s\teval_%s\tpassed\n' \
        "$(date -Iseconds)" "$persona" "$name" "$mode" >> "$status_file"
    done
  done
done

python - "$exp_root" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/eval_streaming/naturalness_by_target.csv")):
    persona, arm = path.parts[-4], path.parts[-3]
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["target"] != "__overall__" or row["step"] == "base":
                continue
            dmos = float(row["mos_delta_vs_base"])
            dwer = float(row["wer_delta_vs_base"])
            dsim = float(row["sim_delta_vs_base"])
            rows.append({
                "persona": persona,
                "arm": arm,
                "step": row["step"],
                "n": row["n"],
                "mos_mean": row["mos_mean"],
                "mos_delta_vs_base": row["mos_delta_vs_base"],
                "wer_delta_vs_base": row["wer_delta_vs_base"],
                "sim_delta_vs_base": row["sim_delta_vs_base"],
                "automatic_gate": "pass" if dmos > 0 and dwer <= 0.02 and dsim >= -0.02 else "fail",
            })
rows.sort(key=lambda row: (row["persona"], row["automatic_gate"] != "pass", -float(row["mos_delta_vs_base"])))
output = root / "leaderboard.csv"
if rows:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f"[leaderboard] {output} ({len(rows)} candidate checkpoints)")
PY

echo "=== SWEEP COMPLETE $(date) ==="
cat "$status_file"
echo "Leaderboard: $exp_root/leaderboard.csv"
echo "Automatic gates are screening only; select each persona by blinded listening."
