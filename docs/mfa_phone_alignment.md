# MFA phone alignment (metadata only)

MFA is used to locate corresponding phone intervals in pristine native and
ASI recordings. It never warps or resamples the audio and it is not needed at
runtime in Hear-Me-Out.

Run MFA in its own environment (the project uses MFA 3.4.x). On WSL:

```bash
micromamba activate mfa

python scripts/prepare_phoneaware_mfa_corpus.py \
  --train-manifest data/<source-dataset>/manifests/train.jsonl \
  --val-manifest data/<source-dataset>/manifests/val.jsonl \
  --prompts-file /path/to/L2-ARCTIC/PROMPTS \
  --target-speaker ASI \
  --expected-train 200 \
  --expected-val 21 \
  --out data/mfa_hindi_phoneaware_asi

MFA_ROOT=data/mfa_hindi_phoneaware_asi \
MFA_DICTIONARY=english_us_mfa \
MFA_ACOUSTIC_MODEL=english_mfa \
MFA_NUM_JOBS=4 \
bash scripts/run_phoneaware_mfa_queue.sh
```

The queue validates and aligns source and target corpora independently, then
requires genuine phone tiers on both sides. `MFA_TEST_TRANSCRIPTIONS=0` is the
default because the ARCTIC transcripts are already pinned; enabling the much
slower transcription diagnostic is optional.

Important gates:

- use phone tiers only; never fall back to word tiers;
- exclude OOV, missing-tier, low-match, very short, and pathological pairs;
- trim uncertain boundary frames in the loss, not in the waveform; and
- never create a time-warped training waveform or latent target.

After the queue passes, build and validate the canonical dataset using
`scripts/build_pristine_parallel_dataset.py` as described in
[`datasets.md`](datasets.md). The final mapper checkpoint contains learned
weights only; TextGrids and MFA are training-time metadata.
