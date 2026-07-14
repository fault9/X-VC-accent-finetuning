# X-VC accent stream localization audit

X-VC has no accent label, accent classifier, or single set of "accent
weights." The source pronunciation may be distributed across its frozen
semantic and quantized acoustic streams and their interaction with the
converter. Before building a deployable sequence editor, this audit performs a
causal intervention at the exact 50 Hz concatenation point used by X-VC.

## What the audit renders

For each held-out native/ASI prompt, stock X-VC and one fixed ASI reference
render five conditions:

| condition | semantic stream | quantized acoustic stream | timeline |
|---|---|---|---|
| `native_sem__native_zq` | native | native | native |
| `asi_sem__native_zq` | real ASI | native | native |
| `native_sem__asi_zq` | native | real ASI | native |
| `asi_sem__asi_zq_mapped` | real ASI | real ASI | native |
| `asi_sem__asi_zq_original` | real ASI | real ASI | original ASI |

MFA phone intervals construct the diagnostic target-to-native frame map.
Continuous semantic activations use linear sampling; quantized acoustic codes
use nearest-frame selection. This mapping is never used to train a checkpoint.
The original-timeline ASI condition is the round-trip ceiling, and its
comparison with the mapped-both condition measures mapping artifacts directly.

## Run on the GPU container

The phone-annotated pair root must already exist from the phone-aware pipeline.

```bash
cd ~/X-VC
conda activate xvc

nohup bash scripts/run_xvc_accent_stream_audit.sh \
  > exp/run_logs/xvc_accent_stream_audit.out 2>&1 &

tail -f exp/run_logs/xvc_accent_stream_audit.out
```

By default this renders up to 40 held-out ASI pairs and runs MOS, WER,
speaker-similarity, and accent-classifier scoring. The final table is:

```text
exp/xvc_accent_stream_audit_asi_metrics/condition_summary.csv
```

Set `SCORE=0` to render/listen without loading the metric models. Set
`MAX_PAIRS=10` for a short smoke test. Outputs are experiment artifacts under
`exp/` and are intentionally ignored by git.

Pair-feature shards may retain paths to an older dataset directory. The audit
searches `data/` for an exact basename and accepts multiple matches only when
their bytes are identical. If the original cross-pair root was deleted, unpack
the pristine phone-aware MFA `mfa_corpus` under `data/`; no latent-aligned
training dataset needs to be restored. Use repeated `--audio-search-root`
arguments to restrict or extend the search.

## Decision rules

- `asi_sem__native_zq` becomes convincingly Indian while the native control
  does not: edit the semantic sequence.
- `native_sem__asi_zq` becomes Indian: accent realization is materially carried
  by the acoustic-code stream; a semantic-only editor cannot be sufficient.
- Only `asi_sem__asi_zq_mapped` becomes Indian: the two streams must be edited
  jointly or under a cross-stream consistency objective.
- The mapped-both condition loses substantial quality/accent relative to the
  original ASI round trip: the phone map is too destructive for causal
  localization; improve the diagnostic alignment before drawing conclusions.
- Neither single-stream nor mapped-both becomes Indian while the original ASI
  round trip does: the mapping/intervention method is invalid.
- The original ASI round trip is not recognizably Indian: stop; the reference,
  selected pairs, or evaluation stack is not a valid ceiling.

Classifier labels are only a noisy screen. The same filenames must also be
blindly auditioned across all five directories. This experiment identifies the
representation to edit; it is not itself a deployable accent converter.
