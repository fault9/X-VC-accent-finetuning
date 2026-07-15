# Joint target-persona mapper

## Intended behavior

For every source speaker, preserve the words while producing the selected
target persona's **voice and accent**. For the current Hindi experiment, ASI is
the target persona. The mapper is not conditioned on BDL, RMS, CLB, or any
other source identity.

X-VC remains responsible for voice conversion:

1. Frozen X-VC source encoders extract semantic and acoustic-code streams.
2. The small mapper edits both streams through one shared causal trunk.
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
nohup bash scripts/run_joint_persona_mapper_sweep.sh \
  > exp/run_logs/joint_persona_mapper_asi.out 2>&1 &

tail -f exp/run_logs/joint_persona_mapper_asi.out
```

The default controlled sweep trains 0, 80, and 160 ms lookahead arms. It
extracts paired streams once, evaluates every arm on
`data/eval_sources_joint_persona_clean`, and
writes `exp/joint_persona_mapper_asi/status.tsv`. Passing an automatic gate is
necessary but not sufficient: matched stock/mapper files must also be blindly
auditioned.

`data/eval_sources_joint_persona_clean` must therefore contain at least two
speakers not used for
training (for example CLB and SLT). Set `MIN_UNSEEN_SPEAKERS=1` only for a
short plumbing smoke test; such a result cannot support an any-source claim.

Training curves are written in TensorBoard format for every arm:

```bash
tensorboard --logdir exp/joint_persona_mapper_asi --bind_all
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

Defaults are the pre-registered 3x3 matrix: lookahead 0/4/8 frames and
discrete weights 0.25/0.5/1.0.  This is intentionally a training-objective
test, not a new architecture or a StreamVoice+ reimplementation.  An arm must
show positive target-aligned code gain and pass the existing unseen-source
MOS/WER/similarity/accent gate; matched listening remains decisive.
