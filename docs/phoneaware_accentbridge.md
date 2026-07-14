# Phone-aware target-realization supervision

## Why this is a different experiment

The earlier latent-DTW, word-tier MFA, frame-gap weighting, full fine-tuning,
and LoRA runs did not explicitly supervise *where within a phone* the target
speaker realizes the shared transcript differently. Word-tier MFA fell back
from the requested phone tier and performed worse than DTW. The frame-gap
scout emphasized individual high-error frames, which can mostly be alignment
noise at boundaries.

This path uses the same real native-to-ASI/TNI shared-prompt pairs, but:

1. requires genuine phone tiers on both recordings (no word fallback);
2. sequence-aligns equal phone labels from the shared transcript;
3. leaves native audio and features on their original timeline;
4. uses MFA boundaries only to identify phone spans;
5. adaptively excludes uncertain boundary regions;
6. pools onset/middle/offset means and whole-phone variance; and
7. matches the bridge's phone-level edit to the real native-to-target edit.

It remains length-preserving at inference, and training now matches inference:
the bridge receives ordinary native features in both cases. The trained bridge
is still the same causal residual Conv1D module; MFA and TextGrids are
training-only.

This is pronunciation-sensitive supervision, not a hand-authored list of
stereotyped accent rules. Phones receive weight only when the actual target
recordings differ from their paired native recordings.

## Inputs

- `data/accentbridge_pairs/{train,val}` from
  `extract_accentbridge_pairs.py`;
- source and target MFA TextGrids with a real `phones`/`phone` tier;
- the source/target TextGrid filenames must match the WAV stems recorded in
  each extracted pair.

Before using an old MFA directory, inspect one file. If it contains only a
`words` tier, stop: that is the already-falsified word-tier experiment.

```bash
find data/mfa_hindi_400/mfa_align -iname '*.TextGrid' | head
grep -n 'name = "phones"' path/to/example.TextGrid
```

## Queue validation and both MFA alignments

First prepare the exact target-speaker subset from the pristine paths retained
in the latent-crosspair manifests. This copies audio and writes transcripts;
it does not warp or resample anything:

```bash
python scripts/prepare_phoneaware_mfa_corpus.py \
  --train-manifest data/crosspair_hindi_latent_400/manifests/train.jsonl \
  --val-manifest data/crosspair_hindi_latent_400/manifests/val.jsonl \
  --prompts-file /mnt/d/l2arctic_release_v5.0/PROMPTS \
  --target-speaker ASI \
  --expected-train 200 \
  --expected-val 21 \
  --out data/mfa_hindi_phoneaware_asi
```

Once `mfa_corpus/source` and `mfa_corpus/target` contain matching WAV and LAB
files, activate the environment where `mfa` is installed and run:

```bash
MFA_ROOT=data/mfa_hindi_phoneaware_asi \
MFA_DICTIONARY=english_us_mfa \
MFA_ACOUSTIC_MODEL=english_mfa \
MFA_NUM_JOBS=4 \
nohup bash scripts/run_phoneaware_mfa_queue.sh \
  > mfa_phoneaware_queue.out 2>&1 &
```

The standard validator checks corpus structure, OOVs, audio readability,
feature generation, and trial alignment. The much slower per-speaker language
model/lattice diagnostic is optional because these transcripts are copied
directly from the pinned ARCTIC `PROMPTS`; enable it only with
`MFA_TEST_TRANSCRIPTIONS=1` when specifically auditing transcript accuracy.

The queue validates and aligns source and target independently, audits every
output TextGrid for a genuine phone tier, then creates a small TextGrid archive
plus SHA256 for upload. It never builds a latent time map or warps audio.

## Prepare and train the controlled sweep

```bash
cd ~/X-VC
conda activate xvc

PAIRS_ROOT=data/accentbridge_pairs \
SOURCE_ALIGN_DIR=data/mfa_hindi_400/mfa_align/source \
TARGET_ALIGN_DIR=data/mfa_hindi_400/mfa_align/target \
TARGET_SPEAKER=ASI \
bash scripts/run_phoneaware_accentbridge_sweep.sh
```

The preparation gate requires at least 90% phone-label agreement per retained
pair and 80% coverage within the requested target speaker. Mixed ASI/TNI
AccentBridge shards are filtered before coverage is calculated. It reports missing TextGrids, missing phone
tiers, label mismatches, and supervised-frame counts in:

- `data/accentbridge_pairs_phone_unwarped/phone_supervision_meta.json`
- `data/accentbridge_pairs_phone_unwarped/phone_supervision_qc.csv`

The sweep trains three otherwise identical models:

- `lambda_phone=0.1`: gentle phone realization supervision;
- `lambda_phone=0.25`: moderate phone realization supervision;
- `lambda_phone=0.5`: strong phone realization supervision.

L10 remains the external baseline; the new bridge is not initialized from L10.
Do not combine `--phone-aware` with the old `--gap-weight`; the trainer rejects
that confounded condition.

## Curves and monitoring

Each arm writes:

- `tensorboard/` event logs;
- `training_curves.csv` with total, phone-phase, phone-variance, identity,
  smoothness, residual-delta, gradient-norm, and LR curves;
- `validation_metrics.csv`;
- `curves.png`; and
- the existing `train_metrics.json` and `bridge.pt`.

To monitor all arms on a reachable TensorBoard port:

```bash
tensorboard --logdir exp/accentbridge_phoneaware_unwarped \
  --host 0.0.0.0 --port 6006
```

## Decision rule

The feature-level gate is necessary but not sufficient. A useful arm should:

- close more phase-pooled phone gap than the zero-init baseline;
- keep identity drift and residual-delta magnitude controlled;
- preserve unseen-source behavior; and
- after rendering, improve pronunciation/accent in blinded listening without
  moving back onto the same MOS-versus-depth line.

If phone-aware feature metrics improve but rendered speech remains metallic,
the bottleneck is downstream realization/objective. If the phone metrics do
not improve, this post-semantic-adapter representation is not an effective
place to impose the pronunciation edit and the next step is a discrete-token
or phonetic posterior editor rather than more LoRA scaling.
