# X-VC accent fine-tuning changes and procedure

This fork tracks the PersonaPlex/X-VC accent-conversion work on top of upstream
X-VC (MIT License; https://github.com/Jerrister/X-VC). This file is intentionally
procedural: it documents what was tried, why the approach changed, and how to
reproduce the current fine-tuning/evaluation path.

## 2026-07-07 active procedure: latent-aligned native -> Hindi conversion

### Research requirement

The PersonaPlex experiment needs live/free native-English participant speech to
come out with an assigned non-native accent. In X-VC terms, this is the hard
direction:

```text
native/free source speech + accented target reference -> converted speech with target accent
```

Stock X-VC and self-reconstruction fine-tunes preserve the source pronunciation:
native input remains native-sounding even when the target reference speaker is
accented. The working hypothesis is therefore that X-VC must see explicit
native-source -> accented-target pairs during training.

### Upstream setup followed

We follow the upstream X-VC training structure where possible:

- clone X-VC code;
- install the X-VC Python environment;
- prepare the released X-VC checkpoint as the warm start;
- prepare required pretrained modules:
  - GLM-4-Voice tokenizer from Hugging Face;
  - ERes2Net speaker encoder from ModelScope;
  - released X-VC checkpoint at `ckpts/xvc.pt`;
- point config files at JSONL train/val manifests;
- train via `scripts/finetune.sh` / X-VC DeepSpeed training.

The upstream README describes general training, but it does not specify a
small-data accent-conversion fine-tuning recipe. The freeze policy, cross-pair
manifests, latent alignment, and evaluation gates below are project-specific.

### Attempt 1: accent self-reconstruction

Initial runs used accented utterances as both source and target:

```text
source_wav_path == target_wav_path
```

Only the middle conversion modules were trainable:

- trainable: `acoustic_converter`, `prenet`;
- frozen: content/semantic encoder, acoustic codec encoder/quantizer, speaker
  encoder, decoders, vocoder path, and most released X-VC weights.

This sometimes improved rendering quality for already-accented inputs, but it did
not teach native input to acquire the target accent. Conclusion: self-reconstruction
is not enough for the live native-speaker use case.

### Attempt 2: waveform-aligned native -> accented cross-pairs

Next, we built cross-pairs by matching shared ARCTIC prompt IDs:

```text
native CMU ARCTIC source (bdl/clb/rms/slt)
  -> matching Hindi L2-ARCTIC target (ASI/TNI)
```

Naive frame losses are unsafe because the recordings have different timings. We
therefore tried DTW-based waveform alignment:

- sentence-level stretch;
- DTW + overlap-add/source-warp variants;
- Rubber Band / source-warp variants.

These runs produced an audible Hindi-accent signal, which was the first evidence
that cross-pair training could work. However, the converted audio often became
metallic/flangery/robotic, and intelligibility degraded. Listening suggested that
the model learned artifacts from the warped waveform targets. Conclusion:
cross-pair training is promising, but waveform warping is the wrong target audio.

### Current fix: latent alignment, not waveform warping

The current branch keeps all source/target waveforms natural and applies the DTW
mapping inside training only. In other words, the audio files on disk are not
time-warped.

Implementation idea:

- compute DTW anchors between same-prompt native/L2 recordings;
- keep the source and target WAVs pristine;
- during training, align frozen native content/acoustic features to the accented
  target timeline:
  - post-adapter WhisperVQ/semantic features are linearly sampled;
  - SAC/acoustic discrete features are nearest-neighbour sampled;
- inference remains normal X-VC inference.

This preserves the frame-compatible training objective without baking metallic
time-warp artifacts into the target waveform.

## Current Hindi datasets

All current active pilots use Hindi L2-ARCTIC:

- male target speaker: `ASI`;
- female target speaker: `TNI`;
- native source speakers: `bdl`, `clb`, `rms`, `slt`;
- train/val split is by ARCTIC prompt ID, with no train/val prompt overlap;
- only utterances at least 3 seconds are kept;
- global duration ratio filter: target/source must be between `0.85` and `1.20`;
- target reference clips are separate reference utterances, not the paired target
  utterance itself.

### `crosspair_hindi_latent_200`

- data root: `data/crosspair_hindi_latent_200`;
- config: `configs/finetune_crosspair_hindi_latent_200.yaml`;
- train pairs: 199;
- validation pairs: 40;
- train prompts: 162;
- validation prompts: 30;
- source speakers: `bdl=50`, `clb=49`, `rms=50`, `slt=50`;
- target speakers: `ASI=100`, `TNI=99`;
- validation prompt overlap: `0`.

Training config:

- warm start: `ckpts/xvc.pt`;
- trainable modules: `acoustic_converter`, `prenet`;
- learning rate: `3e-5`;
- batch size: `8`;
- reconstruction ratio: `0.0`;
- reversed ratio: `0.0`;
- latent alignment: enabled;
- EMA update: disabled;
- adversarial/discriminator warmup effectively disabled for this short pilot
  (`generator_warmup_steps: 100000`);
- total steps: `500`;
- save/validation interval: `50`.

### `crosspair_hindi_latent_400`

- data root: `data/crosspair_hindi_latent_400`;
- config: `configs/finetune_crosspair_hindi_latent_400.yaml`;
- train pairs: 398;
- validation pairs: 40;
- train prompts: 290;
- validation prompts: 30;
- source speakers: `bdl=100`, `clb=98`, `rms=100`, `slt=100`;
- target speakers: `ASI=200`, `TNI=198`;
- validation prompt overlap: `0`;
- validator result: `failures: 0`.

Training config is the same as `latent_200` except:

- total steps: `1000`;
- save/validation interval: `100`.

`400` means approximately 400 training pairs, not 400 steps. We train to 1000
only to observe the checkpoint curve; based on `latent_200`, the likely usable
region is still expected to be early.

### `crosspair_hindi_latent_400_recon20_lr2e-5`

This is the first explicit MOS/naturalness repair run.

Motivation from `latent_400`:

- native -> Hindi accent transfer works;
- accent strength increases with training;
- MOS/naturalness and WER drift still degrade as the model moves away from the
  released X-VC behavior.

Fix tested here:

- keep the same `crosspair_hindi_latent_400` data;
- keep trainable modules limited to `acoustic_converter` and `prenet`;
- lower the learning rate from `3e-5` to `2e-5`;
- set `reconstruction_ratio: 0.2`;
- keep `reversed_ratio: 0.0`;
- train to 1000 steps, save every 100.

In latent-alignment mode, reconstruction samples are handled specially: when the
dataloader chooses reconstruction, the source becomes the target waveform and the
latent map becomes an identity map. This matters because reusing the original
native->L2 DTW map for target->target reconstruction would distort the clean
anchor and weaken the MOS repair.

Hypothesis:

```text
80% native -> Hindi accent pressure
20% target -> target clean reconstruction anchor
```

should preserve more of stock X-VC's naturalness while retaining the accent signal.

### `crosspair_hindi_latent_400_recon20_semadapter_lr1e-5`

This is the next pathway test after the reconstruction-anchor run.

Motivation:

- `latent_400_recon20_lr2e-5` improved the useful checkpoint region, especially
  step 200, but MOS remained far below stock;
- the remaining hypothesis is that `acoustic_converter + prenet` are forcing
  accent/pronunciation changes through acoustic rendering modules;
- accent is partly a content/pronunciation edit, so the model may need a small
  amount of trainable capacity in `semantic_adapter`.

Fix tested here:

- same `crosspair_hindi_latent_400` data;
- same 20% reconstruction anchor;
- trainable modules: `acoustic_converter`, `prenet`, `semantic_adapter`;
- lower LR: `1e-5`;
- total steps: `1000`;
- save/validation interval: `100`.

Hypothesis:

```text
content-pathway capacity + clean reconstruction anchor
  -> similar accent signal with less acoustic roughness / better MOS
```

Risk:

- `semantic_adapter` is closer to content/pronunciation, so it may improve accent
  efficiency;
- it may also increase WER if it over-edits content, hence the conservative LR.

### `crosspair_hindi_latent_400_recon30_semadapter_lr1e-5`

This is the next naturalness/roboticness ablation after listening to
`recon20_semadapter_lr1e-5`.

Motivation:

- adding `semantic_adapter` improved the objective curve: useful accent appeared
  with better WER/MOS than `acoustic_converter + prenet` alone;
- listening still reported robotic/rough artifacts;
- training longer is unlikely to fix this, because later checkpoints mostly
  increase accent strength rather than naturalness.

Fix tested here:

- same `crosspair_hindi_latent_400` data;
- trainable modules: `acoustic_converter`, `prenet`, `semantic_adapter`;
- lower LR remains `1e-5`;
- reconstruction anchor increased from `0.2` to `0.3`;
- total steps: `1000`;
- save/validation interval: `100`.

Hypothesis:

```text
70% native -> Hindi accent pressure
30% target -> target clean reconstruction anchor
  -> less roboticness / better MOS, possibly slightly weaker accent
```

Main comparison:

- compare against `recon20_semadapter_lr1e-5`, especially steps `400`, `600`,
  and `800`;
- if `recon30` weakens accent too much, keep `recon20` as the better balance;
- if `recon30` preserves a 7-9/10 Indian canary with higher MOS/listening
  naturalness, it becomes the preferred checkpoint family.

## Local dataset build procedure

The full seed cross-pair manifests live in the older local clone:

```text
C:\Users\felix\Scripts\X-VC-accent-finetuning\data\crosspair_hindi
```

The active code lives in:

```text
C:\Users\felix\Scripts\kth-project\X-VC-latent-alignment
```

Build `latent_400` from WSL:

```bash
cd /mnt/c/Users/felix/Scripts/kth-project/X-VC-latent-alignment
source ~/xvc-align/bin/activate

seed=/mnt/c/Users/felix/Scripts/X-VC-accent-finetuning
repo=/mnt/c/Users/felix/Scripts/kth-project/X-VC-latent-alignment

python scripts/align_crosspairs.py \
  --train-manifest "$seed/data/crosspair_hindi/manifests/train.jsonl" \
  --val-manifest "$seed/data/crosspair_hindi/manifests/val.jsonl" \
  --resplit-val-prompts 40 \
  --warp-method latent \
  --anchor-hop-frames 20 \
  --min-duration 3 \
  --min-global-stretch 0.85 \
  --max-global-stretch 1.20 \
  --train-limit 400 \
  --val-limit 40 \
  --source-root "$seed" \
  --l2-root /mnt/d/datasets/arctic-audio/l2arctic \
  --prompts-file /mnt/d/l2arctic_release_v5.0/PROMPTS \
  --out "$repo/data/crosspair_hindi_latent_400"
```

Validate and pack:

```bash
cd /mnt/c/Users/felix/Scripts/kth-project/X-VC-latent-alignment
source ~/xvc-align/bin/activate

python scripts/validate_crosspairs.py \
  --data-root data/crosspair_hindi_latent_400

tar -czf data/crosspair_hindi_latent_400.tgz \
  -C data crosspair_hindi_latent_400

sha256sum data/crosspair_hindi_latent_400.tgz
ls -lh data/crosspair_hindi_latent_400.tgz
```

Upload the tarball to Jupyter at:

```text
~/X-VC/data/crosspair_hindi_latent_400.tgz
```

## Jupyter/container setup after reset

The GitHub clone is only code. A fresh container also needs model assets:

- `ckpts/xvc.pt`;
- `pretrained/speech_eres2net_sv_en_voxceleb_16k/`;
- cached `zai-org/glm-4-voice-tokenizer`.

Known dependency pin after ModelScope installation:

```bash
python -m pip install 'huggingface_hub>=0.23.2,<1.0' \
  'fsspec[http]<2025.0,>=2022.5.0' \
  'packaging<25.0,>=20.0'
```

Pull active branch/configs:

```bash
cd ~/X-VC
git remote add fork https://github.com/fault9/X-VC-accent-finetuning.git 2>/dev/null || true
git fetch fork latent-alignment
git switch latent-alignment 2>/dev/null || git switch -c latent-alignment --track fork/latent-alignment
git pull --ff-only fork latent-alignment
```

Extract uploaded dataset and rewrite absolute local paths to container paths:

```bash
cd ~/X-VC
conda activate xvc

tar -xzf data/crosspair_hindi_latent_400.tgz -C data

python - <<'PY'
import json
from pathlib import Path

root = Path("/home/jovyan/X-VC/data/crosspair_hindi_latent_400").resolve()
old_prefixes = [
    "/mnt/c/Users/felix/Scripts/kth-project/X-VC-latent-alignment/data/crosspair_hindi_latent_400",
    "/mnt/c/Users/felix/Scripts/X-VC-accent-finetuning/data/crosspair_hindi_latent_400",
]

for mf in [root/"manifests/train.jsonl", root/"manifests/val.jsonl"]:
    rows = []
    for line in mf.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for k, v in list(row.items()):
            if isinstance(v, str):
                for old in old_prefixes:
                    if v.startswith(old):
                        row[k] = str(root) + v[len(old):]
        rows.append(json.dumps(row, ensure_ascii=False))
    mf.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("rewrote", mf)
PY

python scripts/validate_crosspairs.py \
  --data-root data/crosspair_hindi_latent_400
```

## Training commands

`latent_200`:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_200 \
  --num_workers 0 2>&1 | tee latent_200_lr3e-5.log
```

`latent_400`:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --num_workers 0 2>&1 | tee latent_400_lr3e-5.log
```

`latent_400` with reconstruction anchor and lower LR:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_recon20_lr2e-5.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_recon20_lr2e-5 \
  --num_workers 0 2>&1 | tee latent_400_recon20_lr2e-5.log
```

`latent_400` with reconstruction anchor plus `semantic_adapter`:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr1e-5.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr1e-5 \
  --num_workers 0 2>&1 | tee latent_400_recon20_semadapter_lr1e-5.log
```

`latent_400` with stronger reconstruction anchor plus `semantic_adapter`:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_recon30_semadapter_lr1e-5.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_recon30_semadapter_lr1e-5 \
  --num_workers 0 2>&1 | tee latent_400_recon30_semadapter_lr1e-5.log
```

## Evaluation protocol

Evaluate only in the intended study direction:

```text
held-out native-English source -> assigned Hindi target reference
```

Do not evaluate other accent-source directions as the main gate, because study
participants are expected to be native English speakers. Real accented recordings
are used only as positive-reference controls, not as sources for the main comparison.

Current evaluation plan:

- config: `configs/eval_hindi_native_to_accent.json`;
- approved native eval sources: `clb`, `rms`;
- assignments:
  - `clb -> TNI`;
  - `rms -> ASI`;
- compare stock X-VC (`ckpts/xvc.pt`) against fine-tuned checkpoints;
- metrics:
  - speaker similarity;
  - Whisper ASR/WER proxy against source transcription;
  - UTMOS/SpeechMOS proxy;
  - CommonAccent-style accent classifier, especially `indian` detections;
- listening remains required, because MOS/accent classifiers are proxies.

For quick checkpoint sweeps, sources are short held-out native utterances. Target
references should be longer, clean accented references (approximately 15 seconds
where possible) because target-reference duration stabilizes speaker/accent
conditioning.

Evaluate `latent_400` early checkpoints:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400/eval_compare_early
```

Evaluate full `latent_400` curve:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400/eval_compare
```

Evaluate the reconstruction-anchor run:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_recon20_lr2e-5 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_recon20_lr2e-5/eval_compare
```

Evaluate the reconstruction-anchor + `semantic_adapter` run:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr1e-5 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr1e-5/eval_compare
```

Evaluate the stronger-anchor + `semantic_adapter` run:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_recon30_semadapter_lr1e-5 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_recon30_semadapter_lr1e-5/eval_compare
```

## Results so far

`latent_200` full intermediate sweep:

| Checkpoint | Similarity | WER proxy | MOS proxy | Indian-class count |
|---|---:|---:|---:|---:|
| stock/base | 0.6380 | 0.0000 | 3.6943 | 1/10 |
| 50 | 0.6511 | 0.0303 | 2.4882 | 3/10 |
| 100 | 0.6603 | 0.0220 | 2.4928 | 8/10 |
| 150 | 0.6659 | 0.0671 | 2.4325 | 6/10 |
| 200 | 0.6740 | 0.0771 | 2.3722 | 9/10 |
| 300 | 0.6526 | 0.0971 | 2.3168 | 10/10 |
| 350 | 0.6338 | 0.0771 | 2.2985 | 9/10 |
| 400 | 0.6312 | 0.0945 | 1.9196 | 9/10 |
| 450 | 0.6280 | 0.1230 | 2.0710 | 10/10 |

Interpretation:

- the fine-tune can make native input register as Hindi/Indian-accented;
- the accent signal appears early, around 100-300 steps;
- longer training increases accent strength but hurts MOS/intelligibility;
- checkpoint 100 is the metric-first candidate;
- checkpoint 200/300 are stronger-accent alternatives that require listening;
- training to thousands of steps is not justified on `latent_200`.

The purpose of `latent_400` is to test whether more paired data gives the same
accent signal with less MOS/WER degradation.

`latent_400` result summary:

- extra data improved stability somewhat compared with `latent_200`;
- step 200 was the best metric-balance candidate;
- step 600 gave stronger accent but higher intelligibility cost;
- the main remaining issue is MOS/naturalness, not whether accent transfer is
  possible.

Next run:

- `configs/finetune_crosspair_hindi_latent_400_recon20_lr2e-5.yaml`;
- goal: recover MOS/naturalness via a 20% reconstruction anchor and lower LR.

Follow-up run:

- `configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr1e-5.yaml`;
- goal: test whether limited content-pathway adaptation improves the accent/MOS
  tradeoff.

Roboticness ablation:

- `configs/finetune_crosspair_hindi_latent_400_recon30_semadapter_lr1e-5.yaml`;
- goal: test whether a stronger clean reconstruction anchor reduces roboticness
  while retaining enough accent signal.

---

## Earlier plan: self-reconstruction fine-tunes

The notes below describe the first fine-tuning plan. They are retained as project
history, but this is **not** the active approach for live native-speaker accenting.

# How we fine-tune X-VC for accents

A plain-language description of our accent fine-tuning method, why each choice was
made, and how to reproduce it. Built on the upstream X-VC project (MIT License;
https://github.com/Jerrister/X-VC). Detailed commands live in `docs/finetuning.md`;
this note is the "what and why".

---

## 1. Goal

Fine-tune X-VC on five speaker groups — four L2-ARCTIC accents plus a **native
CMU-ARCTIC reference group**. Each group contains **exactly the study's
conversion-target voices (1M + 1F)**: in X-VC the target voice is supplied as a
reference clip at inference, so training on additional never-targeted speakers
would spend gradient steps on voices the study never converts into.

| Group   | Speakers             | Audio (train / val)      |
|---------|----------------------|--------------------------|
| Arabic  | ABA (M) + SKA (F)    | ~20 min / ~2 min         |
| Spanish | EBVS (M) + MBMPS (F) | ~20 min / ~2 min         |
| Chinese | TXHC (M) + LXC (F)   | ~20 min / ~2 min         |
| Hindi   | ASI (M) + TNI (F)    | ~20 min / ~2 min         |
| Native  | bdl (M) + slt (F)    | ~20 min / ~2 min         |

L2-ARCTIC ships 2M+2F per L1, so any group can be widened later (more accent
data disentangles accent from speaker idiosyncrasy at the cost of halving the
target voices' share of a fixed step budget) by editing
`configs/data_groups.yaml` and re-running the data pipeline.

The native group exists because the native reference **must share the accent
voices' training treatment** — if native cells came from the stock model while
accent cells came from fine-tunes, native-vs-non-native would be confounded with
base-vs-fine-tuned model. CMU ARCTIC is used (not e.g. VCTK) because L2-ARCTIC
was built as its companion corpus: same prompts, matched recording style — so the
native cells differ from the accent cells only in the thing under study.

Two checkpoint architectures are supported; **joint is the default** (one
checkpoint over all 10 speakers, the target reference clip selects voice+accent
at inference) because per-accent checkpoints confound accent with checkpoint
idiosyncrasy and force a service restart per condition switch. If per-accent is
retained, the native group gets its own checkpoint and the eval harness serves
as a cross-checkpoint parity gate.

We start from the **released, fully-trained X-VC model** and gently adapt it — we do
**not** train from scratch. The speaker roster lives in `configs/data_groups.yaml`.

## 2. The core idea: adapt a small middle, freeze everything else ("Option A")

X-VC is a chain of modules. We keep almost all of them exactly as released and only
let **two** modules learn from the accent data:

- **Trainable:** `acoustic_converter` and `prenet` — the two "conversion" modules in
  the middle of the network.
- **Frozen (unchanged from the released model):** the semantic encoder (content), the
  acoustic encoder + quantizer (the codec), the speaker encoder (voice identity), the
  three decoders, and the mel extractor.

**Why freeze so much?**
- The frozen modules were trained on far more data than our ~20 minutes per accent.
  Twenty minutes cannot improve them — it can only overfit or damage them.
- Those modules define fixed "coordinate systems" (what is said, how it sounds, who is
  speaking). Keeping them fixed means the two learning modules adapt against a stable
  target instead of a moving one.
- The result: the model's general voice-conversion ability and speaker transfer are
  preserved, and only the *rendering of accented speech* is nudged.

**Why these two modules?** They sit between the encoders and decoders and are where the
conversion actually happens, so they are the smallest place we can adapt accent while
leaving the rest intact. They are also the natural place to later try LoRA (a
lower-capacity alternative) as a comparison.

## 3. How we present the data: self-reconstruction

Each training example is a single accented utterance used as **both the source and the
target** (`source_wav_path == target_wav_path`). The model is asked to reproduce the
accented audio from itself.

**Why:** this is the most stable, lowest-variance way to train on very little data, and
it is one of X-VC's own native training modes (the paper's "reconstruction" mode). We
set `reconstruction_ratio = 1.0` and `reversed_ratio = 0.0` so **every** example is
self-reconstruction.

*Caveat we track:* self-reconstruction adapts the model under "reproduce" conditions,
not under real conversion (source ≠ target). Deployment converts **unseen sources
into the fine-tuned targets**, so checkpoints are selected in exactly that
direction: a fixed folder of unseen-source clips is converted into every pinned
target at each checkpoint, logging ERes2Net similarity, Whisper WER, and
duration-vs-source (`scripts/eval_checkpoints.py`). Validation loss remains a
divergence alarm only — it selects nothing.

## 4. Warm-start, fresh optimizer

We load the released checkpoint's **weights** as the starting point but start the
**optimizer from zero** (no inherited momentum/learning-rate state). This is a genuine
fine-tune, not a resumed training run.

## 5. Conservative training settings (and why)

| Setting | Value | Why |
|---|---|---|
| Learning rate | `1e-5` | 10x below the original training rate; small so we adapt without forgetting. |
| Total steps | `3000` | ~90 passes over the small set — enough to adapt two modules, not so many as to overfit. |
| Batch size | `8` | Conservative; the frozen backbone still runs on every step and uses most of the memory. |
| Validation | every `100` steps | Divergence alarm (checkpoints are picked by the conversion eval, not val loss). |
| Checkpoints | every `250` steps, all kept | The eval harness reads every save; the frozen one is chosen from its curves. |
| Adversarial/GAN loss | off | Kept off for this short, stable reconstruction fine-tune. |

Exact trainable/frozen parameter counts are **printed at the start of every run** so the
freeze is verifiable, not assumed.

## 6. What we deliberately did NOT do

- **No gender split.** Each group trains on both of its speakers (1M+1F). In
  X-VC the speaker/voice is supplied as a *condition* at inference, not learned into
  the weights, so per-gender models are unnecessary and would only halve the data.
  Mixing genders also keeps the adaptation about *accent*, not *voice*.
- **No LoRA yet.** We establish this simple baseline first; LoRA on the same two modules
  is the planned next comparison.
- **No VCTK as the native reference.** VCTK differs in recording conditions and is
  predominantly British-accented — it would be a fourth accent condition, not a
  native reference for American-English L2 targets. VCTK's supported role is
  optional rehearsal filler (`manifest --filler-dir`).

## 7. How to reproduce (high level)

1. Curate + manifest the data: `python scripts/prepare_finetuning_data.py select`
   then `... manifest --joint`; pin eval targets with
   `python scripts/eval_checkpoints.py make-targets`.
2. Smoke-test (stages 1–3), then the train round-trip
   (`bins/smoke_test.py --stage train`) and one deliberate kill-and-resume.
3. Launch fine-tuning per group (`scripts/finetune.sh --accent all`) or jointly
   (`--accent joint`), warm-starting from the released checkpoint. Every run
   writes `run_meta.json` (commit, data/checkpoint sha256s, seed, pip freeze).
4. Choose the frozen checkpoint from the unseen-source conversion curves
   (`scripts/eval_checkpoints.py run`, including the stock model as baseline),
   then listen to its samples. Record run-id + step + sha256 in the methods notes.

Step-by-step commands, environment setup, and output locations are in
`docs/finetuning.md`.
