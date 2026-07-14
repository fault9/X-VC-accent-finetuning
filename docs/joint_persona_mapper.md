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
extracts paired streams once, evaluates every arm on `data/eval_sources`, and
writes `exp/joint_persona_mapper_asi/status.tsv`. Passing an automatic gate is
necessary but not sufficient: matched stock/mapper files must also be blindly
auditioned.

`data/eval_sources` must therefore contain at least two speakers not used for
training (for example CLB and SLT). Set `MIN_UNSEEN_SPEAKERS=1` only for a
short plumbing smoke test; such a result cannot support an any-source claim.

Training curves are written in TensorBoard format for every arm:

```bash
tensorboard --logdir exp/joint_persona_mapper_asi --bind_all
```

The checkpoint selector includes the worst validation source-speaker loss, so
an arm cannot win solely by fitting one of the native training voices.
