# Accent fine-tuning — operator guide

Step-by-step commands to fine-tune X-VC on the L2-ARCTIC accent groups plus the
native CMU-ARCTIC reference group. For *what* the method does and *why*, read
[`../CHANGES.md`](../CHANGES.md). This project is MIT-licensed and builds on the
upstream X-VC repository (https://github.com/Jerrister/X-VC) — see [`../LICENSE`](../LICENSE).

Strategy in one line: warm-start the released checkpoint, freeze everything except
`acoustic_converter` and `prenet`, train in self-reconstruction mode ("Option A").

Speaker groups (roster lives in [`../configs/data_groups.yaml`](../configs/data_groups.yaml),
never in code — the roster is exactly the study's conversion-target voices,
1M + 1F per group):

| Group   | Speakers            | Default budget (per speaker) |
|---------|---------------------|------------------------------|
| arabic  | ABA (M) + SKA (F)   | 10 min train / 1 min val     |
| spanish | EBVS (M) + MBMPS (F)| 10 min train / 1 min val     |
| chinese | TXHC (M) + LXC (F)  | 10 min train / 1 min val     |
| hindi   | ASI (M) + TNI (F)   | 10 min train / 1 min val     |
| native  | bdl (M) + slt (F)   | 10 min train / 1 min val     |

The **native group receives the identical fine-tuning treatment** — otherwise the
native-vs-non-native contrast is confounded with base-vs-fine-tuned model. A
**joint** manifest (all 10 speakers → one checkpoint) is also generated; see §5.
L2-ARCTIC ships 2M+2F per L1, so each accent group can be widened later by
editing `data_groups.yaml` and re-running select/manifest/make-targets.

---

## 0. Prerequisites

1. **Raw corpora** (local mirror, 16 kHz mono): L2-ARCTIC and CMU ARCTIC — paths
   configured in `configs/data_groups.yaml` (`sources:`).
2. **Released X-VC checkpoint** at `ckpts/xvc.pt` (https://huggingface.co/chenxie95/X-VC).
3. **Pretrained dependencies** used by the frozen encoders:
   - GLM-4-Voice tokenizer — auto-downloaded, or set a local path in `configs/xvc.yaml`.
   - ERes2Net speaker encoder — set `model.generator.speaker_encoder.pretrained_dir`.
4. **Eval sources**: a fixed folder of YOUR OWN script recordings
   (`data/eval_sources/*.wav`, 16 kHz mono, with sidecar `.txt` transcripts for
   WER) — these are the unseen sources the checkpoint curves are measured on.

---

## 1. Data preparation

Two subcommands (`select` curates audio from the raw corpora, `manifest` builds
the JSONL manifests). Both enforce per-file asserts — 16 kHz, mono, non-empty,
≥ 3 s — and `manifest` hard-fails on any train/val utterance overlap.

```bash
# curate: deterministic per-speaker split, 10 min train + 1 min val each
python scripts/prepare_finetuning_data.py select --minutes-per-speaker 10

# manifests for every group + the joint set; writes manifest_meta.json
# (sha256 per manifest, per-speaker minutes, gender tally)
python scripts/prepare_finetuning_data.py manifest --joint

# pin one reference clip per speaker from the val split (stimulus definition)
python scripts/eval_checkpoints.py make-targets --out data/eval_targets
```

Useful options: `--minutes-per-speaker all`, `--resample` (fix wrong-format
files instead of failing), `--filler-dir <vctk_slice> --filler-ratio 0.25`
(anti-forgetting rehearsal data), `--pair-mode {self,cross,mixed}` (cross is
EXPERIMENTAL — pairs share prompt text but are not frame-aligned, and the config
must then set `reconstruction_ratio: 0.0`; the tool warns about both).

Re-running `select --overwrite` changes the split — do that only before counted runs.

**Path portability.** Manifests store repo-relative POSIX paths; run everything
from the repository root. For a fixed container mount use
`manifest --abs-prefix /workspace/X-VC`.

---

## 2. Upload to the GPU container

Git does **not** carry data or checkpoints. Upload: the repo (or clone your
branch), `data/finetuning_audio/` (audio + manifests + meta), `data/eval_sources/`,
`data/eval_targets/`, `ckpts/xvc.pt`, and any local pretrained dirs.

---

## 3. Linux environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install faster-whisper pyloudnorm   # eval harness (WER, loudness norm)
```

---

## 4. Smoke tests (before any counted run)

```bash
# stages 1-3: manifests -> one real batch -> freeze + forward pass.
# Startup now HARD-ASSERTS: trainable submodules == {acoustic_converter, prenet}
# exactly, and the warm-start load fails loudly if prenet/converter weights are
# missing from the checkpoint or unexpected keys appear.
python bins/smoke_test.py --stage all --config configs/finetune_arabic.yaml --batch-size 2

# stage 4 (run explicitly): 20 steps -> checkpoint saves -> the INFERENCE path
# reloads that checkpoint -> converts one clip -> exp/smoke_train/smoke_conversion.wav.
# LISTEN TO THE FILE.
python bins/smoke_test.py --stage train --config configs/finetune_arabic.yaml
```

Then the deliberate kill-and-resume test, once:

```bash
bash scripts/finetune.sh --accent arabic          # Ctrl-C after the first save (step 250)
bash scripts/finetune.sh --accent arabic --resume_step 250
```

---

## 5. Train

```bash
bash scripts/finetune.sh --accent arabic          # one group
bash scripts/finetune.sh --accent native          # the native reference group
bash scripts/finetune.sh --accent all             # arabic,spanish,chinese,native sequentially
bash scripts/finetune.sh --accent joint           # ONE checkpoint over all 10 speakers
bash scripts/finetune.sh --accent arabic --lr 3e-5   # LR variant (thin-slice A/B)
```

Options: `--seed 1234` (default, logged), `--wandb`, `--gpus`, `--port`,
`--checkpoint`, `--resume_step`.

Every run writes **`exp/<name>/run_meta.json` before training starts**: git
commit (+dirty flag), base-checkpoint sha256, manifest sha256s, seed, launcher
CLI, pip freeze. A checkpoint without its run_meta is not a result.

### Joint vs per-accent (decision record)

Per-accent checkpoints confound accent with checkpoint idiosyncrasy and force a
service restart per condition switch (which breaks delivery blinding — the
restarter can't be the speaker). The **joint checkpoint avoids both**: one set of
weights, the target reference clip selects voice+accent per request. Default to
joint unless a capacity problem is demonstrated; if per-accent stands, the eval
harness (§7) doubles as the **cross-checkpoint parity gate** — WER/similarity/
timing for the native group must agree across checkpoints within stated tolerance.

### Chosen defaults (per group)

| Setting | per-accent | joint | Reason |
|---|---|---|---|
| learning rate | `1e-5` | `1e-5` | 10x below pretraining |
| total steps | `3000` | `6000` | ~20-45 epochs over the group's data |
| batch size | `8` | `8` | frozen backbone still runs every step |
| val interval | `100` | `200` | frequent val-loss curve (sanity only, see §7) |
| save interval | `250` | `500` | the eval harness reads every save |
| adversarial loss | off | off | `generator_warmup_steps > total_step` |
| EMA | off | off | saved checkpoint == trained weights |

---

## 6. Outputs

```
exp/finetune_<group>/
  config.yaml                 # fully-resolved config for this run
  run_meta.json               # commit, sha256s, seed, CLI, pip freeze
  ckpt/000250.pt ...          # checkpoints (+ .yaml config snapshot each)
  eval/                       # written by scripts/eval_checkpoints.py (§7)
```

---

## 7. Pick the frozen checkpoint — eval harness, NOT val loss

Training is self-reconstruction on the training speakers; **deployment converts
unseen sources into the fine-tuned targets**. The deployment-relevant direction
is *unseen source → accent target*, so checkpoints are chosen from that curve:

```bash
python scripts/eval_checkpoints.py run \
    --run-dir exp/finetune_arabic \
    --source-dir data/eval_sources \
    --targets-dir data/eval_targets \
    --include-base ckpts/xvc.pt \
    --loudnorm
```

For every checkpoint (plus the stock model as baseline) this converts every
source into every pinned target and logs per clip: **ERes2Net cosine similarity**
to the target reference, **Whisper WER** vs the script text, **duration delta**
vs source, optional CommonAccent confidence (`--accent-clf`). Results land in
`eval/metrics.csv`, `eval/summary.csv`, audible samples in `eval/samples/<step>/`.

Pick the frozen checkpoint from `summary.csv` (similarity up, WER flat-or-down,
duration stable, curve not decaying) **and listen to its samples** — ears gate
metrics. Copy the winning run-id + step + sha256 into the paper's methods notes
the day it is chosen.

Streaming (deployment-path) check of the winner:

```bash
python scripts/eval_checkpoints.py run --run-dir exp/finetune_arabic \
    --source-dir data/eval_sources --targets-dir data/eval_targets \
    --steps <winner> --streaming --current 320 --out exp/finetune_arabic/eval_stream
```

This logs `avg_latency_ms` and lets you diff offline-vs-streaming metrics. Verify
one streaming conversion by ear before deploying.

Validation loss (logged every `val_interval`) remains a divergence alarm — a
decaying val curve invalidates a run — but it does not select checkpoints.

---

## 8. Back up / publish (never commit weights to git)

Linking: the first `finetune.sh` run prompts *"link your repo / link your key"*
and saves both to the gitignored `.hf_publish.env`, which
`publish_checkpoint.py` reads automatically. Alternatively set
`XVC_PUBLISH_REPO` / `HF_TOKEN` env vars (they win over the file) or pass
`--repo-id` per call. The repo is auto-created on first push (`--private`
recommended).

```bash
python scripts/publish_checkpoint.py push \
    --repo-id your-org/xvc-accent-finetunes --private \
    --checkpoint exp/finetune_arabic/ckpt/000750.pt \
    --name finetune_arabic.pt \
    --config exp/finetune_arabic/config.yaml \
    --run-dir exp/finetune_arabic          # auto-attaches run_meta.json + eval/summary.csv
```

`push` prints a sha256; `pull` re-prints it for verification. Principle: git
holds the *recipe*, the model store holds the *weights + provenance sidecars*,
the container holds a throwaway *working copy*.

### Deploy into Hear-Me-Out

The hearmeout X-VC service (`services/xvc/server.py`) loads weights at startup.
`infra/run_all.sh` (VC engine choice = X-VC) offers every `*.pt` under
`$XVC_DIR/ckpts/` as a menu — the original release plus any pulled fine-tunes —
and sets `XVC_EMA_LOAD=0` automatically for non-original checkpoints. To skip
the menu (e.g. scripted launches):

```bash
export XVC_CKPT=$XVC_DIR/ckpts/finetune_joint.pt
export XVC_CONFIG=$XVC_DIR/configs/xvc.yaml   # architecture unchanged
export XVC_EMA_LOAD=0                         # fine-tunes carry no EMA weights
```

To get fine-tunes onto the container: `python scripts/publish_checkpoint.py pull
--repo-id <org>/xvc-accent-finetunes --name finetune_joint.pt --out ckpts/finetune_joint.pt`
(run inside `$XVC_DIR`). Point Hear-Me-Out's `infra/setup.sh` at this fork with
`XVC_URL=<fork-url> XVC_REF=accent-finetuning` so the serving clone carries the
pipeline (sha256 logging, eval harness).

Server-side requirements when integrating (this repo's inference glue already
logs the checkpoint sha256 at load — `bins/infer_utils.load_xvc`):

1. **Log `sha256(XVC_CKPT)` at startup** so every session is attributable to
   exact weights.
2. **Joint checkpoint ⇒ per-session target reference**: expose the target
   reference clip as a session parameter instead of baking one voice into the
   process — condition switches then need no restart, which restores delivery
   blinding. Reference clips must come from `data/eval_targets/` (they are the
   pinned stimulus definition).
3. **PP delivery at 24 kHz**: upsample the 16 kHz model output with the same
   resampler as the live HMO path (config-flagged), e.g.
   `soxr.resample(wav, 16000, 24000)`; loudness-normalize stimuli (EBU R128,
   `pyloudnorm`) to match the gate spec — `eval_checkpoints.py --loudnorm
   --out-sr 24000` produces delivery-format samples.

Fallback to stock X-VC is instant: point `XVC_CKPT` back at `ckpts/xvc.pt`.

---

## 9. Resume a run

```bash
bash scripts/finetune.sh --accent arabic --resume_step 1500
```

Loads weights **and** step count from `exp/.../ckpt`; does not re-read the
released checkpoint.

---

## 10. Acceptance runbook (execution proofs — nothing counts until these pass)

1. **Thin slice**: one group (or joint), ~2k steps, eval harness curves produced,
   five converted clips listened to — intelligible, target-similar,
   duration-matched, curves not decaying.
2. **Streaming proof**: one thin-slice output converted in streaming mode,
   verified by ear, latency logged.
3. **Round trip**: `publish_checkpoint.py pull` onto a clean container → serve →
   convert one clip successfully.
4. Kill-and-resume exercised once (see §4).

Only after all four: scale to the full training plan, and send Harsha the
thin-slice result with the joint-vs-per-accent question attached.

---

## Troubleshooting

- **`Checkpoint load ... failed verification`** — the warm-start gate found
  unexpected keys or missing prenet/converter weights. Read the listed keys: a
  wrong `--config`/checkpoint pairing looks exactly like this. Do NOT weaken the
  gate to "make it train".
- **`checkpoint has no state for model 'discriminator'`** — expected/harmless;
  the released `xvc.pt` ships only the generator, and the discriminator is
  unused (adversarial loss off).
- **`[freeze] ... trainable submodules != requested`** — the whitelist in the
  config doesn't match the model's actual children; fix the config, not the assert.
- **CUDA OOM** — lower `--batch-size` (smoke) or `dataloader.static.batch_size`.
- **`FileNotFound` for a WAV** — run from the repo root, or regenerate manifests
  with `--abs-prefix`.
- **Cross/mixed manifests train as self-recon** — the dataloader rewrites pairs
  when `reconstruction_ratio > 0`; set it to `0.0` in the config (the manifest
  tool warns about this).
