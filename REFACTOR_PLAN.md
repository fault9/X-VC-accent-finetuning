# X-VC accent-finetuning — refactor plan and repository audit

Audit date: 2026-07-14, branch `target-conditioned-lowrank-lora`.
This document maps the repository **before** refactoring. Nothing in it changes
behavior; it is the safety baseline required before any code moves.

> **Status (2026-07-14):** Phases 1–5 are implemented on this branch —
> characterization tests, `xvc/` package (adapters/data/config/checkpointing),
> consolidated entry points, compositional configs with verified legacy
> equivalence, packaging, docs, and `MIGRATION.md`. Phase 6 (dead-code
> removal, §18) is deliberately deferred pending another reference check.
> The two behavioral changes shipped are listed exhaustively in
> `MIGRATION.md`. 132 CPU tests pass; sections below describe the repository
> as audited, and remain the map of the legacy layers that are still in place.

Scope guardrails for the whole refactor (from the project brief):

* no change to the mathematical behavior of X-VC (forward path, losses,
  conditioning, adapter placement, checkpoint semantics) without documentation;
* the custom LoRA implementation stays (no PEFT);
* no config or file is deleted until its behavior is mapped to the new system;
* old checkpoints must keep loading; no silent key renames;
* every experiment currently runnable must stay runnable during migration.

---

## 1. Current directory structure

```text
.
├── CHANGES.md                  # 1700-line lab notebook: method, rationale, results
├── CLAUDE.md                   # project conventions (no AI commit trailers, etc.)
├── README.md                   # upstream X-VC README + accent-finetuning quickstart
├── bins/                       # python entry points (train / infer / smoke test)
│   ├── train.py                # THE training entry point (DDP / DeepSpeed)
│   ├── infer_single.py         # single-pair offline/streaming inference CLI
│   ├── infer_utils.py          # model loading + offline/streaming inference core
│   ├── batch_infer_seedtts_offline.py / _stream.py   # SeedTTS-eval batch inference
│   └── smoke_test.py           # staged preflight: manifests / batch / forward / train
├── configs/                    # 55 yaml + 1 DeepSpeed json + 3 eval jsons
│   ├── xvc.yaml                # base config: data, dataloader, trainer, full model tree
│   ├── ds_stage2.json          # DeepSpeed ZeRO-2 config
│   ├── data_groups.yaml        # speaker roster (single source of truth for speakers)
│   ├── eval_*.json             # eval assignment maps (source spk -> target persona)
│   └── finetune_*.yaml         # 51 experiment overlays on xvc.yaml (see §11)
├── docs/
│   ├── finetuning.md           # operator guide for the accent fine-tune pipeline
│   └── streaming_accentbridge_plan.md
├── examples/ figures/          # demo wavs, README figures
├── models/
│   ├── accentbridge.py         # causal residual feature editor (bridge experiments)
│   ├── base/                   # base trainer/dataloader + generic modules (partly dead)
│   │   ├── base_trainer.py     # BaseTranier [sic]: step loop plumbing, AMP, logging
│   │   ├── base_dataloader.py  # BaseDataset over jsonl manifests
│   │   ├── base_datapipes.py   # candidate dead code
│   │   ├── fsq/                # candidate dead code (finite scalar quantization)
│   │   ├── loss/               # reconstruct.py, ssim.py (used by base losses)
│   │   └── modules/            # dac.py, ecapa_tdnn.py, fsq_encoder.py (mostly unused),
│   │                           # dac_utils/ (layers used by acoustic_encoder)
│   └── codec/
│       ├── base/
│       │   ├── base_codec_trainer.py   # train/validate loop shared by codec models
│       │   └── quantizer/              # factorized_vector_quantize.py (LIVE),
│       │                               # quantizer_gumbel.py, distrib.py (candidates)
│       └── sac/
│           ├── model.py         # XVC nn.Module + WavDiscriminator (+ loading, losses)
│           ├── trainer.py       # VCCodecTrainer (batch_forward, G/D steps)
│           ├── dataloader.py    # VCSSLWAVDataset + collation
│           ├── utils.py
│           ├── modules/         # acoustic_converter, acoustic_encoder, decoder (prenet
│           │                    # / semantic_adapter), lora.py, mel_extractor,
│           │                    # semantic_encoder, speaker_encoder, speaker_predictor,
│           │                    # sampler, vocoder/*, utils/* (ERes2Net etc.)
│           └── third_party/hf_whisper/   # vendored WhisperVQ (GLM-4-Voice tokenizer)
├── scripts/                    # 27 python + 16 shell scripts (mixed maturity, see §12)
└── utils/                      # log, checkpoint, file, audio, scheduler, train_utils
```

There is **no packaging** (`pyproject.toml`/`setup.py` absent), **no tests/**
directory, and every entry point assumes CWD = repo root (or does its own
`sys.path.insert`).

## 2. Main training entry points

One real entry point, several wrappers:

| Entry | Role |
|---|---|
| `bins/train.py` | The only training `main()`. argparse (`--config`, `--checkpoint`, `--resume_step`, `--train_engine`, DeepSpeed args) → `utils/train_utils.py` for everything. |
| `scripts/train.sh` | Upstream base-training launcher: torchrun, 8 GPUs, `configs/xvc.yaml`, DeepSpeed. |
| `scripts/finetune.sh` | Accent fine-tune launcher: warm start `ckpts/xvc.pt`, fresh optimizer, single-GPU DeepSpeed, writes `run_meta.json`, maps `--accent X` → `configs/finetune_X.yaml`, `--lr` writes a patched config copy. Interactive HF-publish prompt embedded. |
| `scripts/run_guarded_train_eval.sh` | Supervision wrapper around finetune.sh: CUDA/NVML preflight, dataset validation, training, checkpoint eval sweep. |
| `scripts/run_*.sh`, `scripts/probe_*.sh` (13 files) | Experiment-family sweep drivers that loop over configs and call the guarded runner or render scripts. One-shot research artifacts. |
| `scripts/train_accentbridge.py` | Separate trainer for the AccentBridge module (own loop, not bins/train.py). |

Training flow in `bins/train.py`:
`seed_everything` → `load_config` (OmegaConf + one-level `base_config` merge) →
`log.init` → `init_distributed` → `check_update_and_save_config` (saves resolved
config to `<log_dir>/config.yaml`) → `init_dataset_and_dataloader` (hydra
instantiates the dataloader class) → `init_models` (hydra instantiate → LoRA
inject → warm-start/resume load with gates) → `freeze_model_parameters` →
`params_statistic` → `verify_trainable_modules` (hard gate) → `wrap_cuda_model`
→ `init_optimizer_and_scheduler` (trainable params only; deepspeed.initialize)
→ optional EMA → `hydra.utils.instantiate(config["trainer"])` → epoch loop
`trainer.train(...)`.

## 3. Main inference entry points

| Entry | Role |
|---|---|
| `bins/infer_single.py` (+ `scripts/infer_single.sh`) | Single pair, offline (`--current 0`) or streaming. |
| `bins/infer_utils.py` | The core: `load_xvc` (config → `XVC.load_from_checkpoint`, sha256-logged), `load_pair_as_tensors`, `precompute_conditions`, `run_offline`, `run_streaming`. All other inference-ish scripts import from here. |
| `bins/batch_infer_seedtts_offline.py` / `_stream.py` (+ .sh) | SeedTTS-eval batch inference, RTF/latency reporting. |
| `scripts/eval_checkpoints.py` | `make-targets` / `run` subcommands: convert eval sources with each checkpoint, measure (speaker sim, accent classifier, DNSMOS), rank. 698 lines. |
| `scripts/make_distill_dataset.py`, `make_selfdistill_dataset.py`, `decode_latent_targets.py`, `gate_teacher_renders.py` | Teacher rendering for distillation datasets (inference + optional bridge/LoRA-scale knobs). |
| `scripts/merge_lora.py` | Folds LoRA into base weights → stock-architecture checkpoint for serving. |

## 4. Dataset preparation and validation flow

1. **Curation**: `scripts/prepare_finetuning_data.py select` reads
   `configs/data_groups.yaml` (speaker roster; **contains Windows paths
   `D:/datasets/...`** for the curation machine) and copies/pins wavs;
   `... manifest` writes JSONL manifests.
2. **Cross-pair building** (native → accented, same prompt):
   `scripts/build_crosspairs.py` → optionally `align_crosspairs.py` (DTW /
   rubberband / latent alignment; writes `align_meta.json`,
   `alignment_qc.jsonl`, per-pair `latent_alignment_path` .npy) →
   `filter_crosspairs.py` (ASI-only subsets etc.).
3. **Validation**: `scripts/validate_crosspairs.py --data-root ...` — hard
   preflight: manifest schema, PCM16 mono checks, sample rate, durations, RMS,
   zero runs, clipping, DC offset, train/val prompt leakage, latent-map
   monotonicity, alignment-QC gates. Prints a JSON report, exit 1 on failure.
4. **Distillation datasets**: `make_distill_dataset.py` /
   `make_selfdistill_dataset.py` render teacher outputs into
   `data/distill_*/manifests/*.jsonl`; the training side is the *same*
   cross-pair trainer pointed at rendered data (distillation is data-side —
   there is no distillation loss term in the code).
5. **Training-time loading**: `models/codec/sac/dataloader.py`
   `VCSSLWAVDataset.fetch_data` re-reads the JSONL rows, loads audio, applies
   role assignment (reconstruction / reversed / standard), segments to 2.4 s,
   handles latent-alignment placeholders, and `padding()` collates.

Manifest row schema (JSONL, one object per pair):
required by the dataloader: `target_utt`, `source_wav_path`, `target_wav_path`;
optional: `source_utt`, `source_token_path`, `target_semantic_path`,
`target_reference_wav_path`, `latent_alignment_path`, `raw_source_wav_path`,
`raw_target_wav_path`, `raw_source_duration`, `raw_target_duration`.
Sidecar files: `align_meta.json` (dataset-level: `warp_method`, `warp_side`,
rubberband version/engine, stretch guards, `max_anchor_removal_fraction`,
speaker lists) and `alignment_qc.jsonl` (per-source-utt:
`global_stretch_ratio`, `anchor_removal_fraction`, ...). **None of these are
versioned.**

## 5. Model construction flow

* Training: `init_models` → `hydra.utils.instantiate(config.model[name])` with
  `_target_: models.codec.sac.model.XVC` — recursive instantiation builds every
  submodule from its own `_target_` block (mel_extractor, semantic_encoder
  wrapper, semantic_adapter, acoustic_encoder, acoustic_converter,
  acoustic_quantizer, speaker_predictor, speaker_encoder, prenet,
  semantic_decoder, acoustic_decoder, loss_config). `XVC.__init__` additionally
  downloads/loads the pretrained WhisperVQ semantic encoder
  (`init_semantic_encoder`) when `from_pretrained` is set.
* Inference: `XVC.load_from_checkpoint(config_path, ckpt_path, device)`
  duplicates the construction logic **manually** (11 near-identical
  `hydra.utils.instantiate(...) if gen_cfg.get(...) else False` lines), then
  injects LoRA if the config asks, then loads the state dict (EMA-aware,
  strict=False with logged missing/unexpected keys).

## 6. How Hydra/OmegaConf instantiate modules

* Configs are plain OmegaConf YAML (not Hydra composition). `utils/file.py::
  load_config` merges `base_config:` **one level deep only** — an overlay of an
  overlay silently drops the grandparent (documented workaround in
  `finetune.sh`, which writes patched config copies instead of overlays).
* `hydra.utils.instantiate` (hydra-core 1.3.2) is used purely for recursive
  `_target_` instantiation of: the model tree, the dataloader
  (`config["dataloader"]`), and the trainer (`config["trainer"]`).
* There is **no config schema validation** anywhere: unknown keys are ignored,
  missing keys fail deep inside construction.

## 7. Where checkpoints are loaded

| Site | Purpose |
|---|---|
| `utils/checkpoint.py::load_checkpoint` | Warm start (`--checkpoint ckpts/xvc.pt`). strict=False + `_report_and_gate_load`: hard-fails on unexpected keys or missing keys inside `trainable_modules` (LoRA `lora_A/lora_B` misses are expected and tolerated). `{model_name: state_dict}` layout; absent `discriminator` allowed. |
| `utils/checkpoint.py::resume_checkpoint` / `resume_ema_checkpoint` | `--resume_step N` / `-1` → `<log_dir>/ckpt/NNNNNN.pt` (+ `.yaml` sibling config). |
| `utils/train_utils.py::init_models` (per-model `checkpoint:` key) | Legacy per-model load path (`torch.load(...)[model_name]`, strict=False, prints raw missing keys). |
| `models/codec/sac/model.py::XVC.load_from_checkpoint` | Inference-side load: rebuild model from config, re-inject LoRA topology, EMA-aware key choice (`ema_generator` with `ema_model.` prefix strip), strict=False. |
| `utils/train_utils.py::save_models` | Saving: DeepSpeed zero→fp32 consolidation per model, `{tag}.pt` = `{generator: sd, discriminator: sd, ema_generator: sd}` + `{tag}.yaml` config sibling + `last.pt` symlink + stale-checkpoint GC. |
| `scripts/merge_lora.py`, `scripts/publish_checkpoint.py`, `scripts/eval_checkpoints.py`, `scripts/verify_lora_hygiene.py` | Downstream consumers of the same layouts. |

**Checkpoint format (must be preserved):** a `.pt` file is a dict keyed by
model name (`generator`, optionally `discriminator`, `ema_generator`); values
are plain state_dicts with original module names; LoRA runs add `*.lora_A` /
`*.lora_B` keys next to the base keys (base keys unchanged — this is the
compatibility contract of `LoRALinear(nn.Linear)`). Merged exports strip
`lora_*` and load into the stock architecture.

## 8. Where parameters are frozen / unfrozen

All in `utils/train_utils.py::freeze_model_parameters` (single site, three
modes, priority order):

1. model-level `no_grad: true` → freeze whole model (discriminator in
   fine-tunes);
2. `lora.enabled: true` → `mark_only_lora_as_trainable` (freeze all, re-enable
   `lora_A/lora_B`, optional bias policy);
3. `trainable_modules: [...]` whitelist → freeze all, unfreeze listed top-level
   submodules (the non-LoRA fine-tune mode);
4. legacy per-submodule `no_grad` on submodule configs (base training:
   `acoustic_encoder.no_grad`, `acoustic_quantizer.no_grad`,
   `speaker_encoder.freeze`).

`verify_trainable_modules` is the startup hard gate (live trainable set must
equal the whitelist exactly); `params_statistic` prints total/trainable/frozen
per submodule. **This part of the codebase is already centralized and gated** —
the refactor should extract it into an importable module (no deepspeed import
required) rather than redesign it.

## 9. Where LoRA adapters are injected

`models/codec/sac/modules/lora.py` — self-contained, dependency-free,
well-documented (`LoRALinear(nn.Linear)`, zero-init `lora_B`, merge/unmerge,
`export_merged_state_dict`, substring include/exclude matching, fail-loud on
zero matches). Injection call sites:

* training: `utils/train_utils.py::maybe_inject_lora` (before checkpoint load);
* inference: `XVC.load_from_checkpoint` (before state-dict load);
* both driven by the config block `model.generator.lora:
  {enabled, r, alpha, dropout, target_modules, include, exclude, train_bias}`.

`scripts/verify_lora_hygiene.py` already implements adapted-layer reporting,
frozen-drift audit, optimizer-set equality, merge equivalence, and export
audit — but as a script against real checkpoints, not as unit tests.

## 10. Trainable modules per experiment family

| Family (configs) | Trainable set |
|---|---|
| Base X-VC (`xvc.yaml`) | generator (except `acoustic_encoder`, `acoustic_quantizer`, `speaker_encoder`, semantic encoder internals) + discriminator |
| “Option A” accent fine-tunes (`finetune_{arabic,spanish,chinese,hindi,native,joint}.yaml`) | `acoustic_converter` + `prenet` (full weights, whitelist mode) |
| Cross-pair latent fine-tunes (`finetune_crosspair_hindi_latent_400_recon*_lr*.yaml`) | `acoustic_converter` + `prenet`; `*_semadapter_*` variants add `semantic_adapter` |
| LoRA acoustic-only (`*_lora_acoustic_r{1,2,4,8,16}*`) | LoRA A/B inside `acoustic_converter` (include `attn.`, `ff_x.ff`, `ff_c.ff`) |
| LoRA acoustic+prenet (`*_lora_acoustic_prenet_*`) | LoRA inside `acoustic_converter` + `prenet` (adds `pwconv`) |
| LoRA AdaLN arms (`*_adaln_r8*`) | LoRA on AdaLN/conditioning projections (include `attn_norm_x.`, `norm_out.`) |
| Distill / self-distill / stack-distill (`finetune_distill_*`, `finetune_selfdistill_*`, `finetune_stackdistill_*`) | LoRA on `acoustic_converter` + `prenet` (attention + FFN + AdaLN + pwconv); objective is ordinary reconstruction against teacher-rendered targets |
| AccentBridge (`scripts/train_accentbridge.py`) | the bridge module only (X-VC fully frozen) |
| Discriminator | `no_grad: true` in every fine-tune; adversarial path also disabled via `generator_warmup_steps: 100000 > total_step` |

## 11. Duplicated / near-duplicated configs

51 `finetune_*.yaml` overlays; within each family they differ by 1–3 scalars
(the diff between `..._r2_alpha16.yaml` and `..._r4_alpha16.yaml` is `r: 2→4`
plus comment text). Families and their encoded-in-filename knobs:

* `finetune_{accent}.yaml` (6): accent → dataset paths + log naming.
* `finetune_crosspair_hindi*.yaml` (23): dataset size (200/400/full/wide),
  alignment mode (dtw/rubberband/latent/sourcewarp/plumbing/ab), persona
  filter (asi/asionly/hindim), recon weight (`recon20/30/40`), lr
  (`lr5e-6/1e-5/2e-5/5e-5`), semadapter, LoRA rank/alpha/targets.
* `finetune_distill_hindi_asi_lora_r{1,2,4,8}` + `x15` (5), `selfdistill` (2),
  `stackdistill` l10/l12/l125/l15/union/v4 (7): teacher variant + rank.
* `finetune_asi_recononly_wide_lora_r4_alpha16.yaml`, `finetune_joint.yaml`.

Everything encoded in the filename should become an experiment config that
overrides adapter/dataset/training groups. `scripts/make_lora_matrix_config.py`
already generates config-matrix variants programmatically — evidence the
duplication is mechanical.

## 12. Duplicated training / inference scripts

* **Sweep runners** (`run_lowrank_lora_sweep.sh`, `run_lowrank_distill_sweep.sh`,
  `run_v2_scale_ladder.sh`, `run_selfpair_arms.sh`, `run_gapweight_scout.sh`,
  `run_intermediate_teacher_sweep.sh`, `run_persona_experiment_matrix.sh`,
  `run_l15_rebuild.sh`, `probe_*.sh` ×5, `repair_distill_sources.sh`): all are
  “loop over N configs/scales → guarded train/eval or render”. Differ only in
  the arm list and constants. Candidates for one parametrized sweep driver +
  per-experiment arm lists (data, not code).
* **Teacher renderers**: `make_distill_dataset.py` vs
  `make_selfdistill_dataset.py` share structure (glob sources → run model(s) →
  write wavs + manifest).
* **Manifest validators**: three partially-overlapping implementations —
  `smoke_test.py` stage 1, `validate_crosspairs.py`, `filter_crosspairs.py`
  (plus `load_manifest` variants in several scripts).
* **Model loading**: `bins/infer_utils.load_xvc` vs ad-hoc loading in
  `eval_accentbridge.py`, `train_accentbridge.py`, `calibrate_eval_floor.py`.

## 13. Files mixing several responsibilities

* `utils/train_utils.py` (788 lines): CLI arg groups, distributed init, config
  mutation+saving, dataloader construction, model instantiation, LoRA
  injection, checkpoint warm-start, freezing, verification, parameter stats,
  DDP/DeepSpeed wrapping, checkpoint saving (incl. `os.system("rm -rf ...")`),
  optimizer+scheduler factory. The single biggest split target.
* `models/codec/sac/model.py`: XVC forward **plus** pretrained downloading,
  checkpoint loading, LoRA injection, loss construction — `load_from_checkpoint`
  duplicates the hydra construction tree.
* `bins/smoke_test.py` (449): validation + dataloader test + forward test +
  micro-train loop in one file.
* `scripts/eval_checkpoints.py` (698), `scripts/align_crosspairs.py` (1058),
  `scripts/prepare_finetuning_data.py` (598): multi-concern giants (acceptable
  as scripts, but shared pieces — manifest IO, model loading, metrics — should
  come from the package).
* `scripts/finetune.sh`: training launcher + interactive HF-token prompt +
  config patching + run-meta writing.
* `utils/log.py`: logging init + tensorboard + wandb + spectrogram plotting
  (imports `wandb`, `matplotlib`, `torch.utils.tensorboard` at module import).

## 14. Circular / fragile imports

* No true circular imports found (`models.*` → `utils.*` one-way; `bins` →
  both).
* **Fragile**: every `scripts/*.py` does `sys.path.insert(0, ...)` (22 sites);
  everything else assumes CWD = repo root.
* **Heavyweight imports at module import time**: `utils/train_utils.py` imports
  `deepspeed` unconditionally (so nothing that touches it is importable on a
  non-CUDA box); `utils/log.py` imports `wandb`+`matplotlib`+tensorboard;
  `models/codec/sac/model.py` imports `audiotools`. This blocks lightweight
  unit testing today.
* `import utils.log as log` inside `models/` couples model code to the repo-root
  layout (works only with repo root on `sys.path`).

## 15. Hard-coded paths and experiment assumptions

* `configs/data_groups.yaml`: `D:/datasets/...` (curation ran on Windows).
* Commented `/hpc_stor03/...` upstream path in `configs/xvc.yaml`.
* Relative-to-CWD conventions everywhere: `ckpts/xvc.pt`, `data/...`,
  `exp/...`, `pretrained/speech_eres2net_sv_en_voxceleb_16k`.
* Sweep runners hard-code checkpoint step lists, arm names, scales, and
  specific `exp/...` run dirs (acceptable for archived experiments; must not
  leak into the library).
* `EVAL_PROMPTS` (the 10 reserved arctic_b prompts) hard-coded in
  `validate_crosspairs.py`.

## 16. Configuration values encoded in filenames

See §11. Additionally `scripts/finetune.sh` derives `log_dir` from the config
filename, and `eval_checkpoints.py` parses run directories by name — the
filename is currently load-bearing metadata. The new experiment configs must
carry an explicit `experiment_name` so naming stops being semantics.

## 17. Schema inconsistencies (metadata / dataset validation)

* `source_utt`: REQUIRED by `validate_crosspairs.py`, optional per README and
  `smoke_test.py` (`REQUIRED_FIELDS` differs between the two validators).
* `alignment_qc.jsonl` rows: `validate_crosspairs.py` does raw
  `qc["global_stretch_ratio"]` / `qc["anchor_removal_fraction"]` →
  **`KeyError: 'anchor_removal_fraction'`** on older QC files (the reported
  crash). No versioning, no distinction between required/optional, no
  per-row rejection reasons for these gates.
* `align_meta.json` keys are probed with `.get()` defaults in some places and
  assumed in others; `warp_method` values (`rubberband`/`latent`/absent) gate
  different required fields implicitly.
* `dataloader.py` hard-codes collation lengths (`38400` samples = 2.4 s ×
  16 kHz, `30` tokens, `120` ssl frames) that silently assume
  `segment_duration: 2.4`, `sample_rate: 16000`, `latent_hop_length: 1280`,
  `ssl_per_sem_ratio: 4` — changing `segment_duration` in config would
  desynchronize the collator without an error.
* `utt = elem["target_utt"][:-3]` in `fetch_data` assumes a 3-char suffix on
  every utterance id.
* Dataset kwargs have duplicated aliases (`reconstruct_ratio` vs
  `reconstruction_ratio`, `swap_ratio` vs `reversed_ratio`, `mask_cond` vs
  `mask_target_condition`) handled ad hoc in `__init__`.

## 18. Candidate dead code (verify before removal — Phase 6 only)

Confirmed unreferenced by any config `_target_`, import, script, or doc
(checked with `rg` across py/yaml/sh/md):

* `models/base/fsq/` (finite_scalar_quantization.py, residual_fsq.py) and
  `models/base/modules/fsq_encoder.py`, `perceiver_encoder.py`,
  `ecapa_tdnn.py`, `pooling_layers.py` (the `models/base/modules` copy; the
  `sac/modules/utils/pooling_layers.py` copy IS used by ERes2Net),
  `models/base/modules/dac.py` (only `dac_utils/layers.py` is imported, by
  `acoustic_encoder.py`).
* `models/codec/base/quantizer/quantizer_gumbel.py`, `distrib.py`
  (only `factorized_vector_quantize.py` is instantiated).
* `models/codec/sac/modules/vocoder/msstft_discriminator.py`,
  `vocos_decoder.py` (only `wave_generator.py`, `wave_discriminator.py`,
  `stft_utils.py`, and `modules/vocos.py` via `decoder.py` are live).
* `models/base/base_datapipes.py` (imported only by `base_dataloader.py` —
  check which symbols), `utils/data_processor.py`, parts of
  `utils/commons.py` (`test_successful` used only under `if __name__` blocks),
  `utils/checkpoint.py::load_trained_modules` (no callers).
* `configs/finetune_crosspair_hindi{,_dtw,_rubberband*,_sourcewarp_ab,_latent_ab,_latent_plumbing,_noadapter}.yaml` — superseded experiment arms per
  CHANGES.md, but they document the method's history: mark **legacy**, don't
  delete.

All of the above stay in place until Phase 6, and anything removed gets a
tombstone note in MIGRATION.md.

## 19. Backwards-compatibility risks

1. **Checkpoint layout** (`{model_name: state_dict}` + EMA key + LoRA keys +
   sibling `.yaml`): consumed by serving (Hear-Me-Out/PersonaPlex via
   `merge_lora.py` exports), `publish_checkpoint.py`, and every eval script.
   Nothing may change here.
2. **Resolved `config.yaml` inside existing `exp/` run dirs**: inference loads
   these verbatim (`XVC.load_from_checkpoint`). New config machinery must keep
   accepting old resolved configs (same keys, `base_config` overlay style).
3. **CLI flags of `bins/train.py` / `bins/infer_single.py`** are baked into
   `finetune.sh`, the guarded runner, all sweep runners, and CHANGES.md
   command transcripts. Wrappers must preserve them.
4. **`load_config` one-level base_config merge**: scripts rely on the
   *limitation* (finetune.sh writes patched copies). Making it recursive is a
   behavior change — do it only in the new loader, not the old one.
5. **Module paths in `_target_` strings** (`models.codec.sac....`) are stored
   inside every saved run config — the old import paths must keep working
   forever (compat shims when moving files).
6. Ad-hoc consumers parse run-dir names (`eval_checkpoints.py`), so renaming
   experiment output conventions breaks eval tooling.

---

## 20. Minimal execution paths (current commands, recorded before changes)

1. **Standard X-VC inference**
   `bash scripts/infer_single.sh` →
   `python -m bins.infer_single --config configs/xvc.yaml --ckpt ckpts/xvc.pt
   --source_wav_path examples/source.wav --target_wav_path examples/target.wav
   --save_dir outputs/xvc_single --current 0`
   Path: `infer_utils.load_xvc` → `XVC.load_from_checkpoint` → `run_offline`
   → `model.inference(...)` → wav.

2. **Base X-VC training**
   `bash scripts/train.sh` → torchrun × deepspeed →
   `bins/train.py --train_engine deepspeed --config configs/xvc.yaml
   --deepspeed_config configs/ds_stage2.json --log_dir exp/xvc_16khz ...`

3. **LoRA fine-tune of `acoustic_converter`**
   `bash scripts/finetune.sh --accent crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16`
   (or `--config configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml`)
   → `bins/train.py --checkpoint ckpts/xvc.pt --resume_step 0 ...`
   Key semantics: inject LoRA **before** warm-start load; freeze = LoRA-only;
   `trainable_modules: [acoustic_converter]` is the verification whitelist.

4. **LoRA fine-tune of `acoustic_converter + prenet`**
   Same, with `configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_prenet_r4_alpha16_lr5e-5.yaml`
   (`target_modules: [acoustic_converter, prenet]`, include adds `pwconv`).

5. **Distillation training**
   (a) render: `python scripts/make_distill_dataset.py --source-glob ...
   --bridge-ckpt ... / --lora ckpt+cfg --out data/distill_hindi_asi`;
   (b) validate; (c) train: `bash scripts/finetune.sh --config
   configs/finetune_distill_hindi_asi_lora_r4_alpha16.yaml` — the “distill
   loss” is the ordinary reconstruction stack against teacher renders.

6. **Cross-pair dataset validation**
   `python scripts/validate_crosspairs.py --data-root data/crosspair_hindi_latent_400_asi`
   (also invoked automatically by `run_guarded_train_eval.sh`).

7. **Checkpoint loading + inference after LoRA training**
   Either load the LoRA checkpoint with its LoRA-enabled config
   (`bins/infer_single.py --config exp/<run>/config.yaml --ckpt exp/<run>/ckpt/000100.pt`),
   or merge for stock serving:
   `python scripts/merge_lora.py --config exp/<run>/config.yaml --ckpt ... --out merged.pt`
   → serve `merged.pt` with any non-LoRA config of the same architecture.
   Hygiene: `python scripts/verify_lora_hygiene.py --config ... --ckpt ... --stock ckpts/xvc.pt`.

---

## 21. Target architecture and phased plan

Target layout (introduced **gradually**, old paths kept as shims):

```text
xvc/                       # new installable package (pip install -e .)
├── adapters/              # lora.py (re-export of the existing impl), injection,
│   │                      # freezing, reporting
├── data/                  # schemas.py (versioned manifest/align_meta/qc schemas),
│   │                      # validation.py, manifest io
├── models/                # loading.py (XVC.load_from_checkpoint extracted)
├── training/              # checkpointing.py, setup split out of train_utils
├── inference/             # convert.py (from bins/infer_utils)
└── utils/                 # config.py (validated loader), logging.py
configs/                   # + model/ adapter/ dataset/ training/ experiment/ groups
scripts/train.py, infer.py, validate_dataset.py, inspect_checkpoint.py
tests/unit, tests/smoke
```

### Phase 1 — audit + safety baseline (this document)
* REFACTOR_PLAN.md (done), record runnable commands (§20).
* Add `tests/` with characterization tests that run **without** the released
  checkpoint, GPU, or deepspeed: LoRA no-op/merge/injection/matching, freeze
  verification on a mock model, config overlay resolution of every legacy
  config, validator behavior on fixture datasets.
* Constraint discovered: `utils/train_utils.py` imports deepspeed at module
  scope → new modules must be import-light so tests can run on CPU-only boxes.

### Phase 2 — shared utilities (no behavior change)
* `xvc.adapters`: thin, documented API over `models/codec/sac/modules/lora.py`
  (`inject_lora`, `freeze_all_parameters`, `unfreeze_lora_parameters`,
  `get_trainable_parameter_report`, `merge_lora_weights`,
  `load_lora_state_dict`) — the existing implementation is kept as the engine;
  old import path re-exports.
* `xvc.data.schemas` + `xvc.data.validation`: versioned dataclass schemas for
  manifest rows, `align_meta.json`, `alignment_qc.jsonl`; required vs optional
  fields; per-row rejection reasons; JSON stats output; **fixes the
  `anchor_removal_fraction` KeyError** with an actionable error instead of a
  crash (no invented defaults for scientific gates).
* `xvc.utils.config`: new loader with recursive base_config resolution +
  explicit validation (rank > 0, include patterns non-empty, paths exist,
  mutually exclusive modes, weights valid); legacy loader untouched.
* `xvc.training.checkpointing`: load/save/inspect helpers unifying the four
  load sites; `scripts/inspect_checkpoint.py` CLI.

### Phase 3 — entry-point consolidation
* `scripts/train.py` and `scripts/infer.py` as the obvious entry points
  (config-group aware), delegating to the same internals as `bins/train.py`;
  `bins/*` become documented compatibility wrappers with identical CLIs.
* `scripts/validate_dataset.py` wrapping the new validation module;
  `validate_crosspairs.py` becomes a wrapper.

### Phase 4 — compositional configs
* `configs/{model,adapter,dataset,training,experiment}/` groups; experiment
  configs contain only meaningful overrides; `configs/LEGACY_MAPPING.md` maps
  every old file to its composition; resolved-config equality checked
  mechanically (OmegaConf diff) before any legacy config is marked deprecated.

### Phase 5 — packaging, docs
* `pyproject.toml` (editable install), remove `sys.path` hacks from scripts by
  importing the package; `docs/{architecture,training,inference,datasets,lora,
  checkpoints,migration}.md`; MIGRATION.md with old↔new command table.

### Phase 6 — cleanup (only after everything above is verified)
* Retire confirmed dead code (§18) with MIGRATION.md tombstones; archive
  superseded experiment configs under `configs/legacy/` (kept loadable);
  collapse redundant wrappers.

### Working rules
Small reviewable commits; formatting-only changes never mixed with behavior;
`python -m compileall`, `pytest`, and (where installed) `ruff` after each step;
any compatibility risk called out in the commit message and MIGRATION.md.
