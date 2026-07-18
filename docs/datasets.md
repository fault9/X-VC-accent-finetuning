# VCTK persona-naturalness dataset

The maintained dataset is `data/vctk_naturalness_4voice`. It contains one
independent self-reconstruction corpus for each fixed target persona:

- `female_high_p240_10s`
- `female_low_p225_10s`
- `male_high_p273_10s`
- `male_low_p274_10s`

The combined directory is a portable storage bundle only. Training reads
`manifests/by_persona/<persona>/{train,val}.jsonl`, so an adapter never sees
another persona's recordings.

Each persona has 15 validation utterances and at least 150 training utterances
from the same VCTK speaker as its reference. Selection continues until it also
reaches at least 12 training minutes. Audio is pristine mono 16 kHz PCM16 WAV
and at least 1.8 seconds long. Shorter-than-window training clips are zero-padded
to X-VC's 2.4-second window; evaluation clips remain at least 2.4 seconds. The
exact 10-second reference clips are conditioning inputs,
not reconstruction targets. Reference, train, validation, and evaluation
prompts are disjoint.

Build the dataset on the machine holding VCTK-Corpus-0.92:

```bash
python scripts/build_vctk_naturalness_dataset.py \
  --references-dir ../vctk_target_references_10s_natural_f0_authority \
  --vctk-root /path/to/VCTK-Corpus-0.92 \
  --out data/vctk_naturalness_4voice \
  --path-prefix data/vctk_naturalness_4voice \
  --train-minutes 12 \
  --overwrite

python scripts/validate_crosspairs.py \
  --data-root data/vctk_naturalness_4voice \
  --min-duration 1.8
```

The builder also creates:

- `eval_sources_scout/`: two held-out clips from each of five unseen speakers;
- `eval_sources/`: the full 30-source confirmation set;
- `eval_targets/`: the four fixed references; and
- `evaluation_plans/<persona>.json`: one-target assignments for each adapter.

Archive transfer artifact:

```bash
tar -czf data/vctk_naturalness_4voice.tgz \
  -C data vctk_naturalness_4voice
```

Datasets, archives, checkpoints, and evaluation outputs are ignored by git.
