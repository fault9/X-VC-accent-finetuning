# Joint target-persona mapper

## Intended behavior

For every source speaker, preserve the words while producing the selected
target persona's **voice and accent**. For the current Hindi experiment, ASI is
the target persona. The mapper is not conditioned on BDL, RMS, CLB, or any
other source identity.

X-VC remains responsible for voice conversion:

1. Frozen X-VC source encoders extract semantic and acoustic-code streams.
2. A small source-agnostic mapper edits either both pre-prenet streams or the
   unified post-prenet stream.
3. Frozen X-VC prenet/converter/decoder render them using the normal ASI
   reference, speaker embedding, and frame condition.

The target reference is identical in stock and mapper evaluation. A mapper is
rejected if target-speaker similarity, intelligibility, or quality deteriorate
materially, even when the accent classifier improves.

## Why both streams

The causal stream-swap audit selected the joint branch:

| condition | Indian classification |
|---|---:|
| native semantic + ASI acoustic | 0/18 |
| ASI semantic + native acoustic | 1/18 |
| mapped ASI semantic + ASI acoustic | 6/18 |
| original-timeline ASI + ASI | 9/18 |

Independent heads could recreate the incompatible hybrid conditions. The
mapper therefore uses one shared trunk and jointly predicts a semantic
residual and an acoustic-code query. The latter is snapped back to X-VC's
frozen 16,384-entry codebook before rendering.

## Alignment and generalization

Training uses the canonical pristine native-to-ASI pairs. Neither waveform nor
feature stream is warped. Genuine MFA phone spans restrict a differentiable
monotonic loss to corresponding regions of the untouched source and target
sequences.

Generalization safeguards are structural rather than aspirational:

- no source-speaker id or embedding enters the mapper;
- training batches are balanced across native speakers;
- feature noise and channel dropout reduce source memorization;
- prompts do not overlap between train and validation;
- final evaluation must use at least two source speakers absent from mapper
  training, with no prompt overlap;
- stock X-VC and the mapper use the exact same target-persona reference.

The first implementation is length-preserving. It tests whether a learned
joint edit can reproduce the mapped-both causal result without copying ASI
streams at inference. A later monotonic duration/repeat head is warranted only
if this stage passes the voice/quality gate but remains below the
original-timeline ASI accent ceiling.

## Container run

```bash
cd ~/X-VC
conda activate xvc

mkdir -p exp/run_logs
nohup bash scripts/run_persona_mapper_comparison.sh \
  > exp/run_logs/persona_mapper_comparison_asi.out 2>&1 &

tail -f exp/run_logs/persona_mapper_comparison_asi.out
```

The default controlled comparison trains four arms: pre-prenet discrete and
post-prenet unified mappers, each with 0 or 80 ms lookahead. The old 160 ms arm
is excluded because HMO's existing right context is only 120 ms; including it
would silently add latency. It evaluates every arm through the exact HMO
`2400/120/20/100` ms overlapping-window path and writes
`exp/persona_mapper_comparison_asi/status.tsv`. Passing an automatic gate is
necessary but not sufficient: matched stock/mapper files must also be blindly
auditioned.

`data/eval_sources_joint_persona_clean` must therefore contain at least two
speakers not used for
training (for example CLB and SLT). Set `MIN_UNSEEN_SPEAKERS=1` only for a
short plumbing smoke test; such a result cannot support an any-source claim.

Training curves are written in TensorBoard format for every arm:

```bash
tensorboard --logdir exp/persona_mapper_comparison_asi --bind_all
```

The checkpoint selector includes the worst validation source-speaker loss, so
an arm cannot win solely by fitting one of the native training voices.

## Discrete codebook-aware follow-up

The first continuous-loss sweep preserved voice and naturalness but failed the
accent gate for all 0/4/8-frame lookaheads.  Its strongest automatic result,
8 frames, changed MOS by +0.0398 and target-speaker similarity by +0.026 while
keeping WER at 0.0147, but classified 0/17 unseen-source renders as Indian and
produced no audible accent difference in informal matched listening.
Validation explained the failure: only about 0--0.2% of acoustic code ids
changed.  The continuous code query could move closer to ASI embeddings while
remaining inside the original native code's nearest-neighbour region; the
hard codebook snap used by inference then discarded the edit.

`phonewise_discrete_code_loss` closes that train/inference mismatch.  It
computes cosine logits against X-VC's frozen 16,384-entry codebook and uses the
real ASI target code id as a phone-local monotonic NLL target.  Source and
target sequences remain pristine and unwarped.  The loss is normalized by
`log(codebook_size)`, so a uniform prediction is approximately 1.0 and weights
0.25/0.5/1.0 remain interpretable.

Code churn alone is not evidence of accent learning.  Validation therefore
holds a hard-DTW path fixed from untouched native/ASI code embeddings and
reports:

- `code_change_fraction`: how often edited ids differ from native ids;
- `aligned_source_code_agreement`: native agreement with ASI on the fixed path;
- `aligned_predicted_code_agreement`: edited agreement on that same path;
- `aligned_target_code_gain`: the target-directed improvement between them.

The guarded matrix keeps the same joint mapper, frozen X-VC renderer, ASI
reference, pristine pairs, and unseen CLB/SLT evaluation:

```bash
cd ~/X-VC
conda activate xvc

mkdir -p exp/run_logs
nohup bash scripts/run_joint_persona_discrete_sweep.sh \
  > exp/run_logs/joint_persona_discrete_asi.out 2>&1 &

tail -f exp/run_logs/joint_persona_discrete_asi.out
```

Defaults are a latency-safe 2x2 matrix: lookahead 0/4 frames and discrete
weights 0.25/0.5. A target-vs-native code-logit margin is also applied on a
fixed phone-local path, rewarding ASI-directed substitutions instead of code
churn. This is intentionally a training-objective
test, not a new architecture or a StreamVoice+ reimplementation.  An arm must
show positive target-aligned code gain and pass the existing unseen-source
MOS/WER/similarity/accent gate; matched listening remains decisive.

## Deployment-shaped training and latency contract

The live HMO window contains 120 X-VC frames: 108 history, 6 current, 1 smooth,
and 5 future frames at 20 ms/frame. This is also the original X-VC training
duration (`segment_duration: 2.4`). The pair extractor now crops pristine
source and target **waveforms** independently to 2.4 s before either waveform
passes through the frozen X-VC encoders. It never encodes a full utterance and
crops cached features afterward.

The two windows are anchored around the same MFA-matched phone but keep their
own timing. MFA selects phone regions only; it does not resample a waveform or
feature stream. With five causal dilated blocks and four lookahead frames, the
context-valid source interval is `[62, 116)`: about 54 supervised frames per
window. Every position in that interval has the same receptive context it
would have in an overlapping live window. Restricting training to the six HMO
frames emitted on one particular hop would discard most of the clean
supervision without making the convolution more deployment-faithful.

Both new mappers use five causal dilated blocks. Their 62-frame/1240 ms history
fits HMO's existing 2160 ms history. Six blocks would need 2520 ms and are
rejected at training and service startup. Lookahead is checked against the
existing 120 ms right context.

The post-prenet arm learns a residual after semantic and acoustic streams have
already been fused, immediately before the frozen acoustic converter. The
target reference, speaker encoder, converter, and decoder remain unchanged.

No Hear-Me-Out service change is made during this experiment. First evaluate
candidate checkpoints through `scripts/eval_joint_persona_mapper.py`, which
uses the exact HMO window geometry. If and only if a checkpoint passes the
unseen-source, listening, quality, and latency gates, the later HMO integration
will load it through the optional X-VC runtime loader with metadata checks such
as:

```bash
export XVC_PERSONA_MAPPER_CKPT=~/X-VC/exp/persona_mapper_comparison_asi/<arm>/best.pt
export XVC_PERSONA_TARGET=ASI
```

That future service integration would load the mapper once, validate
checkpoint/codebook/persona metadata, and run it inside the existing window.
It would request no extra audio, and an unset variable would retain stock
X-VC. These environment variables are not wired into Hear-Me-Out yet.

Evaluation synchronizes CUDA around every streaming window and records stock
and mapper p50/p95 runtime. An arm fails if mapper p95 adds more than 10 ms or
if total p95 exceeds the existing 120 ms current-chunk budget.

## Remaining limits

- The length-preserving mapper cannot insert or delete frames. A duration head
  is justified only after an arm adds audible accent without harming voice.
- Training currently has two native source speakers. Unseen CLB/SLT evaluation
  is mandatory, but more training voices would strengthen an "any source" claim.
- Each training pair currently yields two deterministic 2.4 s source/target
  windows and each validation pair yields one. This increases window placement
  diversity without pretending that correlated crops are new speakers.
- MFA annotations select comparison regions only. They never warp audio.
- Automatic accent labels are noisy. The gate uses the paired change in the
  continuous CommonAccent `indian` posterior. On 20 clips, a hard-label gain
  of 0.05 is only one argmax flip and is too brittle; `indian_frac` remains a
  secondary diagnostic. The automatic canary requires closing at least 25% of
  the measured genuine-ASI versus native posterior gap; it does not demand an
  arbitrary absolute probability or a perfect classifier label. Final
  selection requires matched, blinded listening plus the MOS/WER/ASI-similarity
  gates.
