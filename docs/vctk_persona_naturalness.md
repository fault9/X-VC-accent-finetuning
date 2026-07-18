# Persona-specific VCTK naturalness adaptation

## Goal

Train one small naturalness adapter for each of the four fixed 10-second VCTK
persona references while preserving X-VC content, target identity, streaming
latency, and the ability to accept an unknown source speaker. This is **not** an
accent conversion run, and the four personas never share an adapter.

## Why this differs from the accent experiments

The accent experiments asked the renderer to turn native pronunciation into a
different realization. Cross-pair regression, alignment, and stronger adapters
therefore moved accent but produced metallic/robotic artifacts. Naturalness does
not require that contradiction. The safe supervision is real target speech
reconstructing itself, with the exact deployment reference supplied only as the
conditioning signal.

The implementation follows X-VC's useful invariants:

- real clean target waveforms;
- 2.4-second random training segments;
- frozen semantic/acoustic encoders, quantizers, decoders, and speaker encoder;
- target mel condition masked over the generated segment;
- the released checkpoint as a fixed anchor;
- low-rank changes in the dual-conditioning converter;
- unseen source speakers and held-out prompts for checkpoint selection.

It intentionally excludes MFA, DTW, waveform warping, pronunciation loss,
accent labels, generated teacher targets, and adversarial training from a random
discriminator.

## Data

The four short WAVs remain untouched reference stimuli. Other pristine VCTK
utterances from `p225`, `p240`, `p273`, and `p274` provide persona-specific
self-reconstruction targets. Each target has 15 validation recordings and at
least 150 clean training recordings of at least 1.8 seconds. Each persona
receives at least 12 training minutes; the exact count varies because selection
is duration-gated. Utterances shorter than X-VC's 2.4-second training window are
zero-padded by the upstream dataloader; evaluation sources remain at least 2.4
seconds. Reference, train, validation, and evaluation
prompts are disjoint. The combined archive is only a storage convenience; the
runner uses `manifests/by_persona/<persona>/` and never mixes target voices.

Build on the machine that holds VCTK:

```bash
python scripts/build_vctk_naturalness_dataset.py \
  --references-dir ../vctk_target_references_10s_natural_f0_authority \
  --vctk-root /path/to/VCTK-Corpus-0.92 \
  --out data/vctk_naturalness_4voice \
  --path-prefix data/vctk_naturalness_4voice \
  --train-minutes 12 \
  --overwrite
```

Archive `data/vctk_naturalness_4voice`, move it to the GPU container, and run:

```bash
SMOKE=1 bash scripts/run_vctk_persona_naturalness_sweep.sh

nohup bash scripts/run_vctk_persona_naturalness_sweep.sh \
  > exp/run_logs/vctk_persona_naturalness_sweep.out 2>&1 &
```

The smoke command runs one persona, one arm, two steps, and two evaluation
sources. It writes to `exp/vctk_persona_naturalness_smoke`, separate from the
resumable full queue. Run the full command only after it completes successfully.

The default full command queues all four personas (five arms each). To run or
resume just one persona, set for example:

```bash
PERSONA_FILTER=female_high_p240_10s \
  bash scripts/run_vctk_persona_naturalness_sweep.sh
```

## Matrix and decision rule

For each persona, the sequential matrix tests conditioning-side LoRA at ranks 1,
2, and 4 with alpha 16, two small learning rates, and one rank-1 wider-converter
control. This matches the low-data rank advice and directly reuses the useful
part of the earlier self-pair matrix. It saves/evaluates steps 50 through 300.
Every checkpoint is compared with stock X-VC on a balanced scout set: two
prompt-held-out recordings from each of five unseen source speakers, using the
HMO streaming settings. Set `RUN_OFFLINE=1` for an additional offline pass. For
final confirmation after choosing candidates, use `FULL_EVAL=1` and a new
`EXP_ROOT` to evaluate the full 30-source set.

The output hierarchy is deliberately persona-specific:

```text
exp/vctk_persona_naturalness_sweep/<persona>/<arm>/
```

`leaderboard.csv` reports the automatic MOS/WER/similarity gates. It does not
replace blinded listening.

Do not select on training or validation loss. A candidate must:

1. improve streaming UTMOS overall and not materially regress any of the four
   targets;
2. keep WER increase at or below 0.02;
3. keep target-speaker cosine within 0.02 of stock (or improve it);
4. add no material streaming latency after LoRA merge;
5. win blinded listening on metallicness, buzz, boundary continuity, and voice
   similarity.

If no arm clears stock for a persona, keep stock X-VC for that persona. A failed
fine-tune is evidence that the remaining naturalness ceiling is in the frozen
codec/decoder or reference conditioning, not a reason to train longer.

At deployment, select the persona adapter once when the session starts. The
adapter does not add another network pass per chunk; merged LoRA has no material
runtime overhead. Dynamic persona switching requires loading the corresponding
adapter/checkpoint between sessions, not retraining and not MFA at runtime.
