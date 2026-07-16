# StreamVoice-style target-persona codec generator

Branch `streamvoice-persona-codec`. This replaces the MFA/feature-regression
persona mappers with a StreamVoice-inspired **target-persona codec-sequence
generator**: a model that maps native WhisperVQ semantic token sequences to
coherent ASI-style *semantic + acoustic* token sequences, rendered through the
completely frozen X-VC stack with the normal ASI reference conditioning.

We borrow StreamVoice/StreamVoice+'s semantic-to-acoustic generation, grouped
per-step acoustic prediction, two-stage (offline teacher -> causal student)
training, quality-gated self-refinement, and causal-delay principles.
StreamVoice itself does **not** perform accent conversion; the accent+voice
target-persona extension is ours.

## Why the previous approach is superseded

The post-prenet pronunciation editor (`models/pronunciation_editor.py`,
`scripts/train_pronunciation_editor.py`, `scripts/run_postprenet4_experiments.sh`)
and the joint accent mapper (`models/joint_accent_mapper.py`, docs in
`docs/joint_persona_mapper.md`) are **superseded but retained** for history.
Their best candidate improved target-speaker similarity (+0.045) but left the
calibrated Indian posterior essentially unchanged (0.0613 -> 0.0631) while
losing 0.24 MOS and tripling WER. Structural causes (not checkpoint luck):
MFA label-matching cannot represent accent-specific substitutions/insertions/
deletions; high label-match filtering removes exactly those events; phone-local
Soft-DTW + continuous regression average codes into off-manifold latents;
post-prenet editing happens after the streams are fused; and length-preserving
runtime cannot realize duration phenomena.

The stock stream-swap audit (`docs/xvc_accent_stream_audit.md`) showed X-VC
*can* render an Indian accent when given coherent genuine ASI streams
(9/18 Indian on the original timeline vs 0-1/18 for single-stream swaps), so
the failure is in how edited streams are produced — hence: generate both
discrete streams jointly instead of regressing continuous features.

## Verified token geometry (audit result)

Verified against `configs/xvc.yaml`, `models/codec/sac/modules/semantic_encoder.py`,
`models/codec/base/quantizer/factorized_vector_quantize.py`, and
`bins/infer_utils.py`; re-verified empirically per pair by
`scripts/extract_persona_codec_tokens.py` (nothing below is hard-coded without
a runtime check):

| stream | source | rate | vocabulary |
|---|---|---|---|
| semantic tokens | WhisperVQ (`zai-org/glm-4-voice-tokenizer`) | 12.5 Hz (1 token / 1280 samples / 80 ms) | `quantize_vocab_size` from the HF config (16384 expected; recorded at extraction) |
| acoustic codes | `FactorizedVectorQuantize` on the 320x acoustic encoder | 50 Hz (20 ms) | `codebook_size: 16384`, codebook dim 8 |

- `process_audio` pads waveforms to `latent_hop_length: 1280` samples, so
  acoustic length = **exactly 4x** semantic length up to a <=1-frame encoder
  edge (`ssl_per_sem_ratio: 4`); the extractor verifies and trims per pair and
  fails on larger disagreement.
- Fusion point: `semantic_encoder.embed_ids -> semantic_adapter`
  (`Decoder_with_upsample`, `sample_ratios [2,2]` = exact 4x upsample,
  1280-d -> 1024-d @ 50 Hz) concatenated with
  `acoustic_quantizer` zq (1024-d @ 50 Hz) -> `prenet` (2048 -> 1024) ->
  `acoustic_converter(x, mel_condition, speaker_condition)` -> `acoustic_decoder`.
- Re-embedding entry points for *predicted* IDs: `semantic_encoder.embed_ids`
  + frozen adapter for semantic; `acoustic_quantizer.vq2emb(ids, out_proj=True)`
  for acoustic (numerically identical to the stock quantized stream — unit
  tested).
- Insertion points: offline = the `streams` dict consumed by
  `bins.infer_utils.render_source_streams`; streaming = the same call inside
  `run_stream_chunk_forward` per 2.4 s window.
- Discrepancy flagged: the X-VC paper (arXiv 2604.12456) describes a 62.5 Hz
  SAC configuration and never states codebook sizes; this repository's shipped
  config is 50 Hz acoustic / 12.5 Hz semantic (120 frames per 2.4 s window is
  asserted in `scripts/extract_joint_persona_pairs.py`). The repository is
  authoritative here.

One causal step per semantic token is therefore ~80 ms and carries 4 acoustic
codes — matching the plan's assumption, now verified.

## Architecture

```
source waveform
  -> frozen WhisperVQ tokenizer            (12.5 Hz semantic token IDs)
  -> persona codec generator               (teacher now, student later)
       emits: target semantic token ID + 4 acoustic code IDs per step, EOS
  -> frozen embed_ids + semantic_adapter   (predicted semantic -> 50 Hz)
  -> frozen vq2emb                         (predicted codes -> 50 Hz zq)
  -> frozen prenet -> acoustic_converter -> decoder
  -> normal ASI reference conditioning     (speaker embedding + mel condition)
```

`models/persona_codec_generator.py`:

- **`PersonaCodecTeacher`** (Phase 1) — non-causal transformer encoder over
  source semantic tokens; causal decoder emits one target semantic step at a
  time, with a grouped acoustic head predicting the step's 4 codes in
  parallel (StreamVoice's per-step grouped acoustic prediction). Variable
  length via EOS; teacher-forced cross-entropy over discrete IDs only (no
  continuous L2); greedy (beam-1) decoding with a hard
  `ceil(src_len * max_len_ratio) + slack` bound.
- **`PersonaCodecStudent`** (Phase 3 scaffold) — causal decoder-only stack
  with explicit per-layer KV cache, 0/1-semantic-step delay (1 step = 80 ms,
  inside HMO's 120 ms right-context budget), StreamVoice-style semantic
  masking (`corrupt_source`), and a blank/emit/emit-repeat action head with a
  hard global expansion budget. Mechanically complete and unit-tested
  (cached-step == full forward; no future leakage); **not to be trained until
  the teacher passes its gate.**

Neither model has any source-speaker input; the target persona is fixed per
checkpoint. `xvc/persona_codec/runtime.py` loads checkpoints fail-closed:
unknown model class, any source-speaker conditioning, missing voice
conditioning, vocabulary/group-size mismatch against the live X-VC model, or
semantic/acoustic codebook SHA-256 drift all refuse to load. With no
generator supplied nothing changes: the stock inference paths are untouched
and the re-embedding path reproduces stock streams exactly (identity tests in
`tests/unit/test_persona_codec_runtime.py`).

## What trains offline vs what runs live

| stage | component | where |
|---|---|---|
| offline (container, GPU) | Phase 0 extraction, geometry report, teacher training, teacher rendering + scoring + gate | `scripts/extract_persona_codec_tokens.py`, `report_persona_token_geometry.py`, `train_persona_codec_teacher.py`, `render_persona_codec_teacher.py`, `score_xvc_accent_stream_audit.py`, `gate_persona_codec_teacher.py` |
| offline (after teacher passes) | Phase 2 pseudo-parallel generation (teacher labels arbitrary native audio, quality-gated; real ASI + stock/self-reconstruction anchors retained), then student distillation | to be added only after the gate |
| live (eventually) | the distilled `PersonaCodecStudent` only: frozen tokenizer -> student -> frozen renderer. No MFA, no transcripts, no ASR, no teacher at runtime | Phase 3, X-VC-only evaluation before any HMO wiring |

The teacher is *never* a runtime component; it is the falsification gate and
(later) the pseudo-parallel labeller.

## Container commands (Linux GPU container, `conda activate xvc`)

```bash
cd ~/X-VC
conda activate xvc

# 0) plumbing smoke: 2 pairs, extract -> train 2 steps -> decode -> render
nohup bash scripts/run_persona_codec_smoke.sh \
  > exp/run_logs/persona_codec_smoke.out 2>&1 &

# 1) Phase 0: full extraction + geometry report (answers the audit questions)
python scripts/extract_persona_codec_tokens.py \
  --data-root data/hindi_asi_pristine_parallel_221 \
  --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
  --out data/persona_codec_tokens_asi --device 0
python scripts/report_persona_token_geometry.py \
  --pairs data/persona_codec_tokens_asi \
  --out exp/persona_codec_geometry

# 2) Phase 1: offline teacher (200 train / 21 val, prompt-disjoint enforced)
nohup python scripts/train_persona_codec_teacher.py \
  --pairs data/persona_codec_tokens_asi \
  --out exp/persona_codec_teacher_asi \
  --steps 6000 --batch-size 16 --device cuda \
  > exp/run_logs/persona_codec_teacher.out 2>&1 &
tensorboard --logdir exp/persona_codec_teacher_asi/tb --bind_all

# 3) render >= 20 unseen-source prompts (CLB/SLT etc.), stock vs teacher,
#    identical ASI reference in both conditions
python scripts/render_persona_codec_teacher.py \
  --teacher-ckpt exp/persona_codec_teacher_asi/last.pt \
  --source-dir data/eval_sources_joint_persona_clean \
  --reference data/eval_targets/ASI.wav \
  --target-persona ASI \
  --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
  --out exp/persona_codec_teacher_eval \
  --require-unseen-source --min-unseen-speakers 2 \
  --training-manifest data/hindi_asi_pristine_parallel_221/manifests/train.jsonl \
  --device 0

# 4) score with the existing stack (MOS/WER/similarity/Indian posterior)
python scripts/score_xvc_accent_stream_audit.py \
  --audit-root exp/persona_codec_teacher_eval \
  --reference data/eval_targets/ASI.wav \
  --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
  --out exp/persona_codec_teacher_eval_scored

# 5) gate (calibrated genuine-ASI/native gap; see calibrate_eval_floor.py)
python scripts/gate_persona_codec_teacher.py \
  --summary exp/persona_codec_teacher_eval_scored/condition_summary.csv \
  --calibration exp/accent_posterior_calibration/summary.csv \
  --out exp/persona_codec_teacher_eval_scored/gate.json
```

Checkpoint selection: render/score/gate several `step_*.pt` checkpoints and
listen to matched filenames across `stock_xvc/wavs` and
`persona_codec_teacher/wavs`; validation loss is plumbing only. Also run the
GPU smoke as a pytest:
`pytest tests/smoke/test_persona_codec_smoke.py -m "gpu and slow" -s`.

## Teacher gate (hard stop conditions)

Against stock X-VC on the same >= 20 unseen-source prompts, same ASI
reference (defaults in `scripts/gate_persona_codec_teacher.py`):

- MOS drop <= 0.25;
- WER increase <= 0.05;
- target-speaker similarity drop <= 0.03;
- calibrated Indian-posterior gain >= 25% of the genuine-ASI/native gap;
- audible accent increase under matched blinded listening (ears outrank the
  classifier — CommonAccent undercounts segmental shifts).

**If the offline, full-context teacher fails this gate, stop.** Do not build
the streaming student, do not add Phase 2 data, and do not assume more data
fixes it — the 221-pair pilot exists precisely to falsify the architecture
cheaply. Only prepare the >= 45-minute / 4-source-speaker scale-up after the
teacher demonstrates the architecture works. Do not reuse L10 synthetic data
as teacher material; its accent/quality frontier was already inadequate.

## Latency contract (for the eventual student, not the teacher)

HMO geometry: 2.4 s window, 120 ms current region, 80 ms (4-frame) lookahead
precedent; candidate p95 overhead <= 10 ms; total p95 within the 120 ms
budget (`scripts/gate_persona_latency.py`). Design consequence already baked
in: autoregression runs at the verified 12.5 Hz semantic rate (~80 ms/step,
so ~1-2 steps per 120 ms current region) with the 4-code group predicted in
parallel per step — not a 50 Hz autoregressive decoder. The render runner
records synchronized per-stage p50/p95 (token extraction / teacher decode /
X-VC render / stock total) as an early cost signal; the real gate runs on the
student in streaming mode. No HMO integration until an X-VC-only evaluation
passes.

## Known limitations / open risks

- **Timeline drift vs streaming crops**: variable-length output changes local
  durations, but HMO's window cropping assumes a shared timeline. The student
  bounds local expansion (blank/emit/emit-repeat, hard budget) and must hold
  the long-term real-time rate; whether bounded drift survives the 2.4 s
  re-encode-per-chunk streaming path is an open Phase 3 question, deferred by
  design until the offline teacher proves the representation.
- **Conditional independence of the 4 codes per step** (given the decoder
  state) may cost acoustic detail; StreamVoice used a small sequential
  intra-step predictor — a fallback if rendered quality is poor.
- **221 pairs is small** for a 16384-vocabulary sequence model; the teacher
  may overfit prompts before learning pronunciation structure. The
  prompt-disjoint validation decode metrics and rendered canaries are the
  honest check, not training loss.
- **Persona conditioning is per-checkpoint** (documented target-specific
  checkpoint, no persona embedding); a multi-persona generator would need
  explicit conditioning added later.
- **The teacher input is semantic-only**: source prosody/duration information
  beyond what WhisperVQ tokens carry is not available to it. If rendered
  prosody is flat, the student design (which can consume more source context)
  does not inherit this limit.
- `tests/unit/test_phone_supervision.py` fails collection on the base branch
  (imports a module removed in the repo cleanup); unrelated to this work but
  it means `pytest tests/unit` needs `--ignore=tests/unit/test_phone_supervision.py`
  until fixed.

## History

- `docs/xvc_accent_stream_audit.md` — causal evidence both streams must change.
- `docs/joint_persona_mapper.md` — superseded pre/post-prenet mappers (kept).
- `docs/mfa_phone_alignment.md` — MFA remains analysis-only; it never defines
  the sequence target and never runs at inference.
