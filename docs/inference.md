# Inference

## The entry point

```bash
python scripts/infer.py \
    checkpoint=exp/<run>/ckpt/000100.pt \
    source=examples/source.wav \
    target=examples/target.wav \
    output=outputs/converted.wav
```

`source` carries the content; `target` is the reference clip whose voice (and
accent, after fine-tuning) the output takes. The config defaults to the run's
own `exp/<run>/config.yaml`; pass `config=` explicitly for checkpoints
outside a run directory (e.g. `config=configs/xvc.yaml` for the released
`ckpts/xvc.pt`).

Options: `device=0`, `ema=true` (prefer EMA weights when present),
`mask_target_condition=true`, and streaming knobs `current=` (>0 enables
streaming, ms of new audio per chunk), `chunk=`, `future=`, `smooth=`.

## Loading a LoRA checkpoint

A LoRA training checkpoint contains the frozen base **plus** `lora_A/lora_B`
tensors. It must be loaded with a config whose `lora:` block matches the run
(the adapter topology is re-created before the state dict loads — this is why
the run's own `config.yaml` is the default). A stock checkpoint loaded under
a LoRA config is also fine: the freshly-initialized adapters are a no-op, so
base-vs-adapter comparisons can share one config.

For serving without any LoRA code, merge first:

```bash
python scripts/merge_lora.py \
    --config exp/<run>/config.yaml \
    --ckpt   exp/<run>/ckpt/000100.pt \
    --out    exp/<run>/merged_step100.pt
# optionally --lora-scale 1.25 (gate scaled variants by ear before serving)
```

The merged file loads into the released architecture with any non-LoRA config
of the same dimensions.

Not sure what a file is? `python scripts/inspect_checkpoint.py <file.pt>`.

## Legacy commands (still supported)

```bash
bash scripts/infer_single.sh                       # examples/ demo pair
python -m bins.infer_single --config ... --ckpt ... \
    --source_wav_path ... --target_wav_path ... --save_dir outputs/
bash scripts/batch_infer_seedtts_offline.sh        # SeedTTS-eval batch + RTF
bash scripts/batch_infer_seedtts_stream.sh         # streaming batch + latency
```

## Checkpoint evaluation sweeps

`scripts/eval_checkpoints.py run --run-dir exp/<run> ...` converts the pinned
eval sources with every saved step, measures speaker similarity / accent
classification / DNSMOS, and ranks checkpoints (see `docs/finetuning.md`).
