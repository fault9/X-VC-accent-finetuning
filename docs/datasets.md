# Maintained Hindi/ASI datasets

## Canonical pristine parallel dataset

The maintained training/audit dataset is
`data/hindi_asi_pristine_parallel_221`:

- 200 train and 21 validation prompt-matched native-to-ASI pairs;
- pristine mono 16 kHz recordings;
- prompt-disjoint train/validation splits;
- genuine source and target MFA phone TextGrids;
- copied files with SHA256 checksums, not symlinks; and
- natural source and target durations (the pairs are not frame-synchronous).

The source and target say the same sentence, but neither waveform nor latent
features are warped. MFA is retained only as interval metadata for phone-local
monotonic objectives.

Build it where the prepared MFA corpus and alignments coexist:

```bash
python scripts/build_pristine_parallel_dataset.py \
  --prepared-root data/mfa_hindi_phoneaware_asi \
  --out data/hindi_asi_pristine_parallel_221 \
  --path-prefix data/hindi_asi_pristine_parallel_221 \
  --target-speaker ASI \
  --min-duration 2.6

python scripts/validate_crosspairs.py \
  --data-root data/hindi_asi_pristine_parallel_221 \
  --min-duration 2.6
```

`dataset_meta.json` records `not_frame_synchronous: true`. The validator then
checks source and target durations independently while still enforcing audio
format, prompt isolation, checksums, clipping, DC offset, silence, and path
integrity.

Transfer the complete dataset to the GPU container as a local archive:

```bash
tar -czf data/hindi_asi_pristine_parallel_221.tgz \
  -C data hindi_asi_pristine_parallel_221
```

Dataset directories and `*.tgz` archives are ignored by git.

## Manifest contract

Each `manifests/{train,val}.jsonl` row identifies the source and ASI recording,
utterance ids, prompt id, transcripts, alignment TextGrids, and checksums. The
schema and validation implementation live in:

- `xvc/data/schemas.py`
- `xvc/data/validation.py`
- `scripts/validate_crosspairs.py`

Do not feed these natural-duration pairs into direct frame-wise waveform
regression. They are intended for stream localization and alignment-aware
phone/token objectives.

## Derived feature shards

`scripts/extract_joint_persona_pairs.py` encodes the canonical recordings with
the frozen stock checkpoint and writes small `.pt` shards under
`data/joint_persona_pairs_asi/{train,val}`. A shard is only a storage bundle of
several examples; it does not change the data or training method. Shards are
derived cache and can be regenerated.

## Evaluation assets

The maintained sweep expects:

- `data/eval_targets/ASI.wav`: the pinned target-persona reference;
- `data/eval_sources_joint_persona_clean/*.wav`: at least ten evaluation
  sources from at least two speakers absent from mapper training; and
- no prompt overlap with the training manifest.

The mapper receives no source-speaker id. The unseen-speaker evaluation is
what tests the claim that any source can be converted to the target persona.

## Scale-up dataset for the post-prenet experiment

The 221-pair dataset is a valid pilot set, not the final data-scale test: after
phone/context gates it produced 178 unique rows and only BDL/RMS native
sources. The scale-up runner fails closed unless its dataset has:

- at least 600 training pairs;
- at least four diverse native training speakers;
- at least 45 minutes of **unique** real ASI target recordings (pair repeats
  do not increase this number); and
- prompt-disjoint evaluation speakers absent from training.

Build the larger input from the wide manifest's `raw_source_wav_path` and
`raw_target_wav_path` fields. The wide dataset is used only as an index of
pristine shared-prompt recordings; its warped audio/features are never used.
Run `prepare_phoneaware_mfa_corpus.py`, the metadata-only MFA queue, and then
`build_pristine_parallel_dataset.py` exactly as above, but write the result to
`data/hindi_asi_pristine_parallel_scaleup`. Do not satisfy the gate by pairing
one ASI utterance with several native speakers: the unique-ASI-minutes check
is specifically intended to catch that.

For example, if the wide dataset still contains pristine-path fields:

```bash
python scripts/prepare_phoneaware_mfa_corpus.py \
  --train-manifest data/crosspair_hindi_latent_wide_v2/manifests/train.jsonl \
  --val-manifest data/crosspair_hindi_latent_wide_v2/manifests/val.jsonl \
  --prompts-file /path/to/L2-ARCTIC/PROMPTS \
  --target-speaker ASI \
  --out data/mfa_hindi_phoneaware_asi_scaleup

MFA_ROOT=data/mfa_hindi_phoneaware_asi_scaleup \
MFA_DICTIONARY=english_us_mfa \
MFA_ACOUSTIC_MODEL=english_mfa \
MFA_NUM_JOBS=4 \
bash scripts/run_phoneaware_mfa_queue.sh

python scripts/build_pristine_parallel_dataset.py \
  --prepared-root data/mfa_hindi_phoneaware_asi_scaleup \
  --out data/hindi_asi_pristine_parallel_scaleup \
  --path-prefix data/hindi_asi_pristine_parallel_scaleup \
  --target-speaker ASI \
  --min-duration 2.6
```

Inspect the preparation metadata before MFA. If it contains under 45 unique
minutes of ASI, return to the raw L2-ARCTIC corpus and select more shared
prompts; do not compensate with synthetic L10 audio or repeated pairs.

The recommended training sources are BDL, RMS, CLB, and SLT. Evaluation must
then move to different speakers, for example ABA, MBMPS, and SVBI, with no
training-prompt overlap. `scripts/check_persona_dataset_scale.py` verifies all
of these conditions before encoding or training starts.
