# Datasets

## Manifest schema (version 1)

Training data is JSONL, one pair per line. Schema source of truth:
`xvc/data/schemas.py` (`CROSSPAIR_SCHEMA_VERSION = 1`).

Required on every cross-pair row:

| Field | Meaning |
|---|---|
| `source_utt` | source utterance id (QC rows are keyed by it) |
| `source_wav_path` | source (content) waveform |
| `target_utt` | target utterance id |
| `target_wav_path` | target (supervision) waveform |

Optional fields (`MANIFEST_OPTIONAL_FIELDS` documents each): precomputed
token/SSL paths, `target_reference_wav_path` (clean different-prompt
reference), `latent_alignment_path` (DTW map), `raw_source_wav_path` /
`raw_target_wav_path` (pristine pre-warp audio), raw durations.

`align_meta.json` makes fields conditionally required — e.g.
`warp_method: latent` requires the pristine raws and the DTW map on every
row. `alignment_qc.jsonl` carries per-source QC (`global_stretch_ratio`,
`anchor_removal_fraction`); its gates apply only when `align_meta.json`
configures them (`allowed_global_stretch`, `max_anchor_removal_fraction`).
A configured gate with a missing QC field is a validation failure with a
remedy, never a `KeyError`; without the gate, legacy archives stay
inspectable.

## Building cross-pair data

```bash
python scripts/prepare_finetuning_data.py select      # curate raw corpora
python scripts/prepare_finetuning_data.py manifest    # write JSONL manifests
python scripts/build_crosspairs.py ...                # native->L2 same-prompt pairs
python scripts/align_crosspairs.py ...                # DTW/rubberband/latent alignment
python scripts/filter_crosspairs.py ...               # persona subsets (e.g. ASI)
```

Speaker rosters live in `configs/data_groups.yaml` — never hardcode speakers.
Distillation datasets are rendered by `scripts/make_distill_dataset.py` /
`make_selfdistill_dataset.py` and use the same manifest schema (no DTW).

## Validating

```bash
python scripts/validate_dataset.py data/crosspair_hindi_latent_400_asi
```

Checks: schema, PCM16 mono 16 kHz, durations, RMS floor, internal zero runs,
clipping, DC offset, train/val prompt leakage against the 10 reserved eval
prompts, duplicate targets, pristine-audio byte equality (latent/source-warp
modes), DTW map shape/monotonicity, alignment-QC gates. Every rejected pair
is reported as `split:index: reason`; aggregate statistics (including
`schema_version`) are printed and saved to
`<root>/validation_report.json`.

`scripts/validate_crosspairs.py --data-root <root>` is the same engine with
the historical CLI (the guarded runner calls it). The training entry point
runs the preflight automatically for `*/manifests/train.jsonl` datasets.

## Training-time loading

`models/codec/sac/dataloader.py::VCSSLWAVDataset` re-reads the manifests each
epoch: role assignment (`reconstruction_ratio` → identity anchors,
`reversed_ratio` → swapped pairs), 2.4 s segmenting, high-pass filtering,
latent-alignment placeholder handling, collation.

Known assumption (documented, not yet refactored): the collator pins segment
lengths to `segment_duration: 2.4` × 16 kHz (38400 samples / 30 tokens / 120
SSL frames). Changing `segment_duration` in a config without updating
`stack_tensors_with_aligned_T` would silently truncate or pad — validate any
such change end to end.

## Phone-aware AccentBridge metadata

Phone-aware supervision is a separate, training-only metadata path. The
operator scripts prepare pristine native/target WAV+LAB corpora, run genuine
phone-tier MFA independently on each side, and annotate existing AccentBridge
feature shards with corresponding phone spans. MFA boundaries index regions;
they never warp or resample the audio or feature timeline.

Reusable TextGrid parsing and phone-sequence alignment live in
`xvc.data.phone_supervision`. Entry points remain under `scripts/`:

```text
prepare_phoneaware_mfa_corpus.py
  -> run_phoneaware_mfa_queue.sh
  -> annotate_accentbridge_phone_supervision.py
  -> run_phoneaware_accentbridge_sweep.sh
```

The full ASI procedure and quality gates are in
[`phoneaware_accentbridge.md`](phoneaware_accentbridge.md).

## Canonical pristine Hindi/ASI parallel dataset

After MFA succeeds, promote its copied corpus into the durable dataset used by
audits and future pronunciation/token supervision:

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

The builder copies rather than symlinks audio, retains transcripts and true
phone-tier TextGrids, preserves the prompt-disjoint split, writes relocatable
paths, and checksums every file. It records explicitly that the natural
recordings are parallel by prompt but not frame-synchronous. Consequently it
is valid for stream localization and alignment-aware phone/token objectives,
but not direct frame-wise waveform regression without an alignment-aware
objective.

`validate_crosspairs.py` reads `dataset_meta.json`. For a dataset explicitly
marked `not_frame_synchronous`, it validates source and target durations
independently instead of incorrectly requiring equal sample counts. All other
audio, split, path, and QC checks remain active, and a present
`checksums.sha256` is verified in full.
