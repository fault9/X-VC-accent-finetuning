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

### Overnight LR sweep around `recon20_semadapter`

Two additional queueable configs test whether the semantic-adapter curve should be
slower/gentler or faster/stronger:

- `configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6.yaml`
  - same `recon20_semadapter` setup;
  - LR `5e-6`;
  - hypothesis: slower movement may reduce roboticness/WER drift, possibly with
    weaker accent at 1000 steps.
- `configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5.yaml`
  - same `recon20_semadapter` setup;
  - LR `2e-5`;
  - hypothesis: accent may arrive earlier/stronger, but MOS/WER risk is higher.

These runs should be compared against the existing `lr1e-5` run at steps `200`,
`400`, `600`, and `800`.

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

Gentler `recon20_semadapter` LR sweep:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6 \
  --num_workers 0 2>&1 | tee latent_400_recon20_semadapter_lr5e-6.log
```

More aggressive `recon20_semadapter` LR sweep:

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5 \
  --num_workers 0 2>&1 | tee latent_400_recon20_semadapter_lr2e-5.log
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

Evaluate the gentler `lr5e-6` semantic-adapter run:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6/eval_compare
```

Evaluate the stronger `lr2e-5` semantic-adapter run:

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos \
  --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5/eval_compare
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

Overnight LR sweep:

- `configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr5e-6.yaml`;
- `configs/finetune_crosspair_hindi_latent_400_recon20_semadapter_lr2e-5.yaml`;
- goal: bracket the `lr1e-5` semantic-adapter run to see whether roboticness is
  better solved by gentler adaptation or stronger/faster accent learning.

## LoRA accent adapters (parameter-efficient fallback)

This is a fallback path to compare against full/whitelist-module fine-tuning. It
lives entirely on branch `lora-accent-adapters` (do not develop it on
`latent-alignment`). Nothing here changes the behaviour of the existing non-LoRA
configs: a config with no `lora` block trains exactly as before.

### Recommendation: which LoRA target set

The question is *which modules host the adapters*. The five options considered:

| # | Target set | Verdict |
|---|---|---|
| 1 | `acoustic_converter` attention + FFN | **Recommended first.** Highest-leverage, lowest MOS risk. This is the pilot. |
| 2 | `acoustic_converter` + `prenet` | Recommended escalation if (1) under-moves the accent. |
| 3 | + `semantic_adapter` | Ablation only -- closest to content, most likely to hurt WER. |
| 4 | frozen base + one adapter per condition | Deployment shape -- start per-accent; split by gender only if eval shows cross-gender degradation. |
| 5 | LoRA + latent alignment + recon anchor | The **objective/data recipe**, orthogonal to 1-4 -- use it regardless. |

Why `acoustic_converter` first. Accent is a content/pronunciation transform. In
X-VC the source content path is `semantic_encoder (frozen) -> semantic_adapter ->
(concat acoustic) -> prenet -> acoustic_converter -> decoders`. The
`acoustic_converter` is the DiT where token-mixing (attention) and channel-mixing
(FFN) happen under the frame/speaker conditioning; it is where the released
full-FT recipe already does most of its accent work. LoRA on its attention + FFN
linears (**Option 1**) is therefore the smallest, most naturalness-preserving
place to inject accent, with the released weights kept as the anchor.

`prenet` (Option 2) fuses the semantic+acoustic embeddings into the converter
input -- the natural second target if converter-only capacity is insufficient.
`semantic_adapter` (Option 3) sits directly on the frozen content encoder; it is
the most "pronunciation-side" module but also the most likely to distort phonetic
content, so keep it as an ablation, not a default (it is the same module the
`recon20_semadapter` full-FT run probes). Do **not** adapt the frozen semantic
encoder, acoustic codec, speaker encoder, or decoders -- those define the fixed
coordinate systems the adapter learns against, and 20 min of data cannot improve
them.

Option 5 is not an alternative to 1-3; it is the training setup. Latent alignment
is what makes the cross-pair frame losses coherent, and the 20% reconstruction
anchor is the MOS repair. The **recommended configuration is Option 1's target set
trained under Option 5's objective** -- exactly what the pilot config does.

Option 4 is the serving structure: one frozen base plus small swappable adapters,
so we never maintain many full checkpoints. **Start with one adapter per accent**
(both target voices of an accent share it): in X-VC the voice -- and thus gender --
is supplied by the reference clip at inference, not learned into the weights (the
same reason the roster notes give for "no gender split"). **Split an accent into
per-gender adapters only if eval shows cross-gender degradation** -- e.g. the
shared adapter helps the male target but hurts the female. A merged adapter is
~a few MB; the base is shared.

### What the implementation adds

Minimal, auditable, `peft`-free (the training stack already freezes a backbone and
optimises only `requires_grad` params):

- `models/codec/sac/modules/lora.py` -- `LoRALinear` subclasses `nn.Linear`, so the
  base `weight`/`bias` keep their ORIGINAL names (a stock checkpoint loads with no
  key remap); `lora_B` is zero-initialised, so a fresh adapter is the identity map
  (warm-start reproduces stock output until training moves it). Plus inject /
  freeze / merge helpers. If `lora.enabled` but the filters match **zero** linears,
  injection raises `RuntimeError` rather than training an empty adapter set.
- `utils/train_utils.py` -- `maybe_inject_lora()` (called inside `init_models`,
  before any checkpoint load) swaps the targeted linears; `freeze_model_parameters`
  gains a LoRA branch that freezes the base and trains only adapter tensors. The
  exact adapted layer list and trainable-parameter count are printed at startup
  (`[lora] ...`), and `verify_trainable_modules` still asserts the trainable set.
- `utils/checkpoint.py` -- the strict warm-start gate excludes `lora_A`/`lora_B`
  from "missing trainable-module tensors" (they are fresh by design).
- `models/codec/sac/model.py` -- `load_from_checkpoint` re-injects the adapter
  topology before loading, so eval/inference load LoRA checkpoints by exact key
  match. A stock checkpoint under a LoRA config loads with zero-adapter (= stock),
  giving a clean base-vs-adapter comparison on one config.
- `configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml` -- the pilot;
  `..._lora_acoustic_r16.yaml` -- the first capacity escalation (r=16, alpha=32,
  same acoustic_converter-only target set).
- `scripts/merge_lora.py` -- folds adapters into the base and exports a
  stock-architecture checkpoint for zero-overhead serving.

Config flag shape (under `model.generator`):

```yaml
trainable_modules: [acoustic_converter]   # LoRA-host module(s); base frozen
lora:
  enabled: true
  r: 8
  alpha: 16            # scaling = alpha / r
  dropout: 0.0
  target_modules: [acoustic_converter]     # defaults to trainable_modules
  include: ["attn.", "ff_x.ff", "ff_c.ff"] # substring filter on Linear names
  # train_bias: none  # none | lora_only | all
```

For the pilot this adapts **69 Linear layers** (47 attention + 22 FFN across the
6 converter blocks) and skips the AdaLN speaker-condition modulation and the
input/output dim adapters. `trainable_modules` must list the LoRA-host submodules
so the freeze verifier and warm-start gate see the expected set.

Learning rate. LoRA runs at ~5x the full-FT LR (`2e-5`): far fewer parameters, and
zero-init `B` damps the earliest updates, so the low-rank delta needs a larger step
to move within 1000 steps. Start at `1e-4`; raise toward `2e-4` if no Indian-accent
signal appears by ~step 300.

### Train the LoRA pilot (container)

```bash
cd ~/X-VC
conda activate xvc

bash scripts/finetune.sh \
  --accent crosspair_hindi_latent_400 \
  --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml \
  --log_dir exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8 \
  --num_workers 0 2>&1 | tee latent_400_lora_acoustic_r8.log
```

The `crosspair_*`-style accent name still triggers the cross-pair preflight on
`data/crosspair_hindi_latent_400`, and `--resume_step 0` warm-starts from
`ckpts/xvc.pt`. Swap in `..._lora_acoustic_r16.yaml` (with a matching `--log_dir`)
for the r16 capacity point.

**Startup sanity checklist** -- before letting a run proceed, confirm in the log:

- `[lora]` reports an **adapted layer count > 0** (69 for the pilot); a zero count
  now hard-fails, but still confirm the number matches expectation;
- the trainable-parameter total is **far below the full fine-tune** -- LoRA r8 is
  <1M params vs tens of M for `acoustic_converter + prenet` full-FT;
- the freeze verifier logs **`trainable submodules == ['acoustic_converter']`**
  (`[freeze] 'generator' verified: ...`), not `[acoustic_converter, prenet]`.

### Evaluate against stock and the full fine-tune

Same harness and study direction as the latent runs (held-out native source ->
assigned Hindi reference). One command sweeps the LoRA checkpoints and the stock
baseline (stock loads with zero-adapter under the LoRA config, so it is exactly
the released model):

```bash
python scripts/eval_checkpoints.py run \
  --run-dir exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8 \
  --source-dir data/eval_sources \
  --targets-dir data/eval_targets \
  --evaluation-plan configs/eval_hindi_native_to_accent.json \
  --steps 100,200,300,400,600,800,1000 \
  --include-base ckpts/xvc.pt \
  --mos --accent-clf \
  --out exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/eval_compare
```

To compare against full/whitelist-module fine-tuning, run the existing
`latent_400` and `latent_400_recon20_lr2e-5` evals and place the `eval_compare`
CSVs side by side: **stock base vs full-FT recon20 vs LoRA r8 (vs r16)**.

Adapter/checkpoint selection -- pick the earliest step that:

- raises the CommonAccent **`indian`** count meaningfully (the accent canary);
- keeps the **WER proxy** low (intelligibility preserved);
- holds the **MOS proxy** (UTMOS) near stock -- the LoRA hypothesis is that the
  frozen base + small delta degrades MOS less than full-FT at equal accent;
- keeps **speaker similarity** (ERes2Net cosine) to the pinned reference;
- then **listen** -- MOS/accent classifiers are proxies (the DTW-warp path passed
  metrics while sounding metallic).

### Checkpoint x config compatibility (important)

A LoRA checkpoint is `base + lora_*` tensors; the config decides whether the
adapter topology is rebuilt at load. Three cases:

- **Unmerged LoRA checkpoint + LoRA config -> correct.** This is how you EVALUATE
  LoRA checkpoints: `exp/<run>/ckpt/*.pt` with the run's `config.yaml` (which has
  the `lora` block). The adapters are rebuilt and the `lora_*` keys load by name.
- **Merged checkpoint + non-LoRA config -> correct.** This is how you SERVE. After
  `merge_lora.py` folds the delta into `weight` and strips the `lora_*` keys, the
  file is stock-architecture and loads under any non-LoRA config of the same shape
  (e.g. `configs/finetune_crosspair_hindi_latent_400.yaml`).
- **Unmerged LoRA checkpoint + non-LoRA config -> INVALID.** No adapters are
  injected, so `load_state_dict(strict=False)` silently drops every `lora_*` key
  and you serve the frozen base (≈ stock, no accent). This raises no error -- it
  quietly ignores the fine-tune. Never serve an unmerged checkpoint with a
  non-LoRA config; either evaluate unmerged-with-LoRA-config, or merge first.

### Deploy (serving)

Fold the chosen adapter into the base and serve the merged, stock-architecture
checkpoint (no LoRA code at inference):

```bash
python scripts/merge_lora.py \
  --config exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/config.yaml \
  --ckpt   exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/ckpt/000300.pt \
  --out    exp/finetune_crosspair_hindi_latent_400_lora_acoustic_r8/merged_step300.pt
```

Serve `merged_step300.pt` with any non-LoRA config of the same architecture (see
the compatibility rules above). Keeping adapters *unmerged* (one frozen base +
per-accent `lora_*` tensors, swapped at load) is the multi-condition option;
merging is the single-condition, zero-overhead option.

### Risks / unknowns

- **Capacity is the open question.** r=8 on the converter is <1M params. Full-FT
  moved the accent but cost MOS; it is not yet known whether a low-rank delta has
  enough capacity to shift *pronunciation* (a structured, content-correlated
  change, unlike a speaker/style tweak). Mitigation: the shipped r8 -> r16 sweep,
  then escalate the target set (add `prenet`). If r=16 + prenet still under-moves
  the `indian` canary, LoRA may be insufficient here and full-FT-with-anchor
  remains the fallback.
- **Accent direction may not lie in the low-rank subspace** the adapter spans;
  this is exactly what the eval canary tests, not something to assume.
- **Possible upside:** with far fewer parameters, LoRA may overfit latent-alignment
  imperfections less than full-FT -- i.e. better MOS at equal accent. Unverified.
- **Merge is numerically exact** (delta folded into `weight`), so the serving path
  carries no additional risk beyond the adapter it was built from.

### Hygiene verification

`scripts/verify_lora_hygiene.py` closes the "is the LoRA implementation buggy"
question with five offline checks (exit 0 = clean): (A) full adapted-layer list
per host module; (B) frozen-drift audit -- every non-LoRA tensor in a trained
checkpoint must be bitwise identical to the dtype-cast stock warm-start, and
lora_B movement proves only the adapters trained; (C) freeze + optimizer audit
through the REAL trainer functions (`freeze_model_parameters`,
`verify_trainable_modules`, optimizer built exactly like
`init_optimizer_and_scheduler`) asserting requires_grad set == intended LoRA
set == optimizer param set; (D) per-layer merge equivalence (merged ==
unmerged within fp tolerance; unmerge restores) -- per-layer equality implies
whole-network equality since merge only mutates LoRALinear internals; (E)
merged-export audit (no lora_* keys survive; export covers the stock
architecture's keys, validating scripts/merge_lora.py output). Run against a
trained run's checkpoint before trusting any merged artifact.

**Result (run against the trained +prenet r8 checkpoint): all five checks
passed** -- 101 intended layers injected, frozen base bitwise unchanged,
optimizer holds only adapter tensors, adapters demonstrably trained, merged ==
unmerged. The LoRA results are valid; the accent-vs-artifact limitation is a
property of the objective/representation, not an implementation bug.

### Pilot results (r8, r16) and the +prenet ablation

`latent_400` LoRA runs, 20-clip eval (10 clb->TNI, 10 rms->ASI), same pinned
references as the full-FT runs (base rows identical across runs):

- **r8 (converter-only)**: WER held at base level for all 1000 steps
  (0.016 -> 0.022; full-FT degraded 4-8x over the same horizon) -- the low-rank
  constraint protects intelligibility as hypothesised. But the accent canary
  under-moves (indian 1/20 at step 200, peak 7/20 at 800), and MOS decays on
  the same trajectory as full-FT (3.70 -> 2.33). Notably MOS loses a full point
  by step 200 *before* any accent registers, so the MOS problem is not
  accent-confound alone and not adapter capacity -- suspicion moves to the
  data/objective side (target-corpus channel; regression-only drift).
- **r16 (converter-only, 2x rank)**: no improvement over r8 -- same MOS
  trajectory, no accent gain, and WER now degrades (0.016 -> 0.056 at 1000).
  Rank inside the converter subspace is NOT the bottleneck; do not climb the
  rank ladder further on this target set.
- **Next: widen the subspace, not the rank** --
  `configs/finetune_crosspair_hindi_latent_400_lora_acoustic_prenet_r8.yaml`
  (Option 2 from the recommendation table): r=8 adapters on the converter set
  PLUS prenet's ConvNeXt channel-mixing linears (`pwconv1`/`pwconv2`, 16 blocks
  -> 32 layers). 101 adapted layers, ~1.6M trainable -- the r16 budget spent on
  a new module. `semantic_adapter` stays frozen. Startup must report BOTH hosts
  and 101 adapted layers; if prenet contributes 0 the include filter regressed
  and the run silently repeats r8 -- abort.
- **+prenet r8 result**: current best LoRA candidate (pending accent-count
  verification from the per-clip CSV). Off-trend U-shape: MOS bottoms at step
  400 (2.22) then RECOVERS to 2.51 at 1000 with sim recovering too (0.595 ->
  0.607), while WER degrades exactly in the recovery window (0.02 -> 0.056) --
  reads as over-push then partial re-convergence, possibly trading articulation
  for smoothness. Listen to 400-vs-1000 of the same clip to confirm.
- **Gentle arm** --
  `configs/finetune_crosspair_hindi_latent_400_lora_acoustic_prenet_r8_lr5e-5_recon30.yaml`:
  same target set, lr halved to 5e-5 AND recon anchor raised to 30% (one
  deliberate "gentleness" package, not a factorial ablation), total_step 2000
  because both knobs slow accent acquisition and 1000 would likely be
  inconclusive. Eval grid: `--steps 100,200,300,400,600,800,1000,1200,1600,2000`.
- **Metric calibration** -- `scripts/calibrate_eval_floor.py` scores REAL
  recordings (raw TNI/ASI corpus clips, optionally a cleaner accented corpus and
  the native sources) with the exact eval metric stack, no conversion involved.
  Establishes the achievable ceiling per proxy: MOS floor of genuine L2 speech,
  the accent classifier's actual recall on real accented clips (the canary
  ceiling is NOT 20/20), the intra-speaker ERes2Net sim ceiling, and the ASR
  accent penalty. Run BEFORE spending further GPU on artifact-reduction arms:
  if the raw L2 corpus scores ~2.5-3.0 MOS, recent runs are near ceiling and
  the "metallic drift" is substantially the measuring stick.
- **Calibration result**: raw L2 corpus scores MOS ~3.8-3.9 (native sources
  4.40; worst genuine clip ~2.9), so the fine-tunes' 2.2-2.5 is a REAL
  training-induced artifact gap of ~1.3 MOS, not a metric penalty. Classifier
  ceiling on genuine L2 speech: indian_frac 0.94 (TNI) / 0.76 (ASI). Sim
  ceiling (intra-speaker vs pinned refs): 0.755 / 0.800. Also found: the ASI
  pinned reference itself scores MOS 3.06 (vs 3.80 corpus mean) -- re-pinning
  from MOS-screened clips is a cheap candidate improvement for ASI conversions
  (changes the pinned stimulus; old tables stop being comparable).
- **Prior MFA attempt (context, do not re-run as-is)**: a phone-level MFA
  alignment path was already tried (`crosspair_hindi_mfa_latent_200`). It
  silently fell back to WORD-tier alignment (final mfa_tier_counts contained
  words only), its mel-dist QC was worse than uniform resampling (~13.77 vs
  12.84), and its training eval was poor (MOS ~1.8-2.34, degraded WER, almost
  no Indian-accent signal). Word-tier MFA is falsified; "switch to MFA" is not
  a validated fallback.
- **Framing correction (important for future alignment work)**: earlier
  discussion described the latent-alignment risk as "warped latent TARGETS".
  That is WRONG for this codebase. The DTW map warps the SOURCE-side frozen
  streams -- the pristine native clip's semantic embedding and quantized
  acoustic latents, resampled onto the L2 timeline inside
  `XVC._latent_aligned_source` (linear interp for continuous streams, nearest
  frame for quantized) -- while the training target is the PRISTINE L2
  waveform, which calibration showed is clean (MOS ~3.8-3.9). So the open
  question is the warped-INPUT regime (can the stack produce clean audio from
  time-warped input streams; do train-time warped inputs mismatch the
  unwarped inputs seen at inference), NOT target corruption. Reason about
  warped inputs, not warped targets.
- **Supervision-quality diagnostic** -- `scripts/decode_latent_targets.py`
  decodes what training actually feeds the loss, through the REAL dataset +
  `XVC.forward` code path (nothing reimplemented; branch choice verified per
  run): `identity_recon` (unwarped L2 latents, encode/decode control) vs
  `warped_crosspair` (native latents DTW-warped to the L2 timeline -- the
  accent-pressure inputs) vs `target_segment` (the pristine supervision
  audio). Scores with the calibration stack + the dataset-QC `mel_dist`.
  Decision rule: identity high / warped low -> alignment is the artifact
  source, invest in genuinely better alignment (proper phone-tier MFA or
  another method -- word-tier already failed); both high -> move to
  objective/feature-matching fixes (e.g. adversarial term is off in all
  fine-tunes); identity itself low -> inspect the latent-training round trip.

## Target-conditioned low-rank LoRA sweep (2026-07-10, supervisor feedback)

Branch: `target-conditioned-lowrank-lora`. New arms responding to supervisor
review; nothing here changes existing configs or their behaviour.

### Deployment framing: source speakers are content carriers

At serving time (Hear-Me-Out / PersonaPlex) the SOURCE speaker is unknown --
any user can be assigned the persona. The L1-ARCTIC sources (clb/rms) in the
cross-pair data must therefore be treated as generic content carriers for
supervision, never as speakers to specialize to. The adaptation we want is
target/accent-conditioned: given the accented reference (frame stream) and the
speaker embedding (global condition), render ANY source's content with the
accent. Three practical consequences:

- Overfitting to clb/rms-/pair-specific detail is a real failure mode with
  ~20-40 min of paired data, and a plausible contributor to the r8
  metallicness (full FT on latent_800_strict strengthened accent but went
  robotic with WER damage; distill/self-refinement improved MOS but weakened
  measured accent). Rank is the one regularization knob not yet swept
  DOWNWARD: r1/r2/r4 test whether a heavily rank-limited delta keeps the broad
  accent transform while lacking capacity for source-specific artifact detail.
- PERSONA MODE (project decision, carried over from the distill arms): one
  LoRA per accent x gender persona, trained and evaluated one at a time. This
  sweep trains the ASI (Hindi male) adapter on the ASI-filtered pairs
  (`data/crosspair_hindi_latent_400_asi`, ~200 train / 20 val) and evals ALL
  20 reserved sources -> ASI (`configs/eval_hindi_native_to_asi.json`,
  cross-gender clb->ASI included -- a real serving condition). The persona is
  defined by accent AND gender, not gender alone. The TNI (Hindi female) twin
  needs its filtered dataset built first (same grep recipe as _asi with
  `"TNI__`).
- The PROOF of "no source specialization" is an unseen-source eval (e.g.
  bdl/slt clips) -- still pending, flagged again here. Held-out prompts from
  the same two source speakers cannot show it.

### Where target conditioning enters acoustic_converter (audit)

Two entry points (`models/codec/sac/modules/acoustic_converter.py`):

1. Frame-level reference stream `c`: embedded by `input_embed.linear_cond`,
   then per block the context-side attention projections `attn.to_q_c` /
   `attn.to_k_c` / `attn.to_v_c` (+ `attn.to_out_c` in non-final blocks) and
   the context FFN `ff_c.ff`.
2. Utterance-level `speaker_condition`: the AdaLN modulation linears --
   `attn_norm_x.` (AdaLayerNormZero per block, modulates the x stream FROM the
   speaker condition) and the final `norm_out.` (AdaLayerNormZeroFinal).

So a conditioning-side-ONLY include set IS cleanly expressible with the
existing substring filters, no code change:

    include: ["attn.to_q_c", "attn.to_k_c", "attn.to_v_c", "attn.to_out_c",
              "ff_c.ff", "attn_norm_x.", "norm_out.", "linear_cond"]

(Substring gotchas: "attn.to_k" would ALSO match `to_k_c`, so the `_c`-suffixed
patterns are what isolates the context side; "attn.to_out_c" does not match the
x-side `attn.to_out.0`.)

Why the sweep does NOT default to it: the x-stream path (`to_q/to_k/to_v`,
`to_out.0`, `ff_x`) is where segmental pronunciation-transform capacity lives,
and the PATH C AdaLN arm already showed conditioning-side modulation alone
cannot do phone substitution. Adapting the x stream is not source
specialization -- those weights are shared across all sources; specialization
only arises by overfit, which is exactly what low rank + the recon anchor
guard against and what the unseen-source eval must verify. The sweep keeps the
proven converter set `["attn.", "ff_x.ff", "ff_c.ff"]` (which already adapts
the c-stream projections and `ff_c` -- the target-conditioning path is in);
the conditioning-only subset above is the documented follow-up if even r1
still shows source-overfit artifacts.

### The arms

| config | r | scaling (alpha 16) | batch | lr |
|---|---|---|---|---|
| `configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r1_alpha16.yaml` | 1 | 16 | 4 | 1e-4 |
| `configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16.yaml` | 2 | 8 | 4 | 1e-4 |
| `configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml` | 4 | 4 | 4 | 1e-4 |

Same recipe as the r8 pilot otherwise: recon anchor 0.2,
`latent_alignment: true`, total_step 1000, acoustic_converter-only host --
but on the ASI persona subset. Note alpha is FIXED at 16, so effective
scaling alpha/r grows as rank drops (r8 was 2.0) -- if a low rank trains
unstably, lower lr before touching alpha. Three knobs differ vs the original
r8 run (rank, batch 8 -> 4, ASI-only data); for an in-family r8 reference at
matched batch/data, regenerate from the r4 arm:

    sed 's/^      r: 4$/      r: 8/' configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r4_alpha16.yaml > configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r8_alpha16.yaml

LR variants are NOT pre-created (no config explosion). For the winning rank
only (r2 shown -- substitute the winner), generate 5e-5 / 2e-4:

    sed 's/lr: 0.0001/lr: 0.00005/' configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16.yaml > configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16_lr5e-5.yaml

    sed 's/lr: 0.0001/lr: 0.0002/' configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16.yaml > configs/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r2_alpha16_lr2e-4.yaml

### How to run (container)

    bash scripts/run_lowrank_lora_sweep.sh

Sequential r1 -> r2 -> r4 through the guarded runner (CUDA preflight,
crosspair validation, contamination gate per arm); eval steps
100,200,400,600,800,1000; `--validate_min_duration 3.0`; persona-mode plan
`configs/eval_hindi_native_to_asi.json` (all 20 sources -> ASI). Eval sources
are the RESERVED held-out prompts (`data/eval_sources_reserved`, arctic
b0002-b0012) -- the old `data/eval_sources` dir is the pre-2026-07-09
contaminated set and the gate rejects it. Comparability: every eval includes
the base checkpoint as a same-plan reference row, and rows line up with the
ASI-plan re-evals of the distill arms; do NOT compare against the original
r8/r16 tables (contaminated sources, gender-matched plan) or the distill-era
mixed-plan tables (base 3.6596 there included clb->TNI rows). Outputs land
under `exp/finetune_crosspair_hindi_latent_400_asi_lora_acoustic_r{1,2,4}_alpha16/`.
`RANKS="2 4" bash scripts/run_lowrank_lora_sweep.sh` resumes a partial sweep.

### Sample-rate consistency

Everything that consumes audio resamples to the model rate on load
(`utils/audio.py:load_audio`, soxr VHQ): the training dataloader loads source,
target, AND reference through it, and `eval_checkpoints.py` loads eval sources
and pinned references the same way. `validate_crosspairs.py` additionally
hard-fails any dataset wav (source/target/raw/reference) that is not 16 kHz.
The one surface that was only counted, never rate-checked, was the eval
source/target dirs: `scripts/check_sample_rates.py` (new) asserts every wav in
the given dirs is mono at the expected rate, and the sweep runner calls it
before touching the GPU. A mismatch there was silently tolerated before
(resampled on load), not wrong -- the check just makes rate drift loud.

### Evaluation guidance for this sweep

Priority order: (1) MOS/naturalness, (2) WER/intelligibility, (3) human
listening for accent, (4) accent classifier ONLY as a noisy proxy (the
calibration section: genuine L2 speech reaches indian_frac just 0.94/0.76 and
the classifier reads much Hindi-English as 'england'). Read the per-clip
`metrics.csv`, not only `summary.csv`.

Two explicit questions this sweep must answer:

- **Metallicness**: at matched steps, do r1/r2/r4 hold closer to the base row
  (same run, same ASI plan) than r8 and full FT did? Listen to `samples/600`
  of the SAME clip across ranks -- the artifact is audible before it is
  measurable.
- **Accent survival**: does ANY accent movement survive at r1? Compare the
  accent-label distribution against the same-run base row (under the mixed
  plan base was {us:18, england:2}; movement toward england/indian is signal)
  AND listen.

Stop rule: if by step 600-1000 the accent labels sit at the base distribution
and listening confirms no accent, do not extend the run -- that rank is below
the accent-capacity floor; escalate rank (or try the conditioning-only subset)
rather than train longer. Conversely, if accent moves but MOS falls toward
full-FT levels (~2.4), the low-rank regularization hypothesis is falsified at
that rank.

### Interim result (2026-07-10) + low-rank DISTILL arms

r1 and r2 on the latent objective both collapsed: base 3.60 -> ~3.0 @100 ->
~2.2-2.5 @200 -> **~1.6-1.8 floor from step 400 on**, WER degrading in step.
Halving the scaling (r2 = 8 vs r1 = 16) barely changed the trajectory, so
this is not primarily over-drive. Combined with full FT and r8(+AdaLN) all
collapsing on the same data, the reading is: **adapter capacity is not the
artifact source -- the latent-alignment objective degrades naturalness at
every size tried** (r1 -> full FT, a huge parameter span). The only setup
that has held MOS is the distillation objective (r8 student: 3.1-3.3).

Follow-up arms therefore move the rank sweep onto the distill objective,
keeping prenet in the target set (supervisor's rank suggestion was about
capacity, not about dropping prenet -- and the distill set already hosts
converter+prenet+AdaLN):
`configs/finetune_distill_hindi_asi_lora_r{1,2,4}_alpha16.yaml` -- ONE knob
(rank) vs `finetune_distill_hindi_asi_lora_r8.yaml` (batch stays 8, recon 0.1,
total_step 2000, x1.2 teacher data). Question: does a rank-limited delta on
the WORKING objective close the remaining base-vs-student gap (3.66 -> 3.29)
while keeping the teacher's accent? Run with
`bash scripts/run_lowrank_distill_sweep.sh` (eval: ASI plan on reserved
sources). CAVEAT: the existing distill-r8 eval_compare was produced under the
MIXED gender-matched plan -- for like-for-like columns, re-eval it on the ASI
plan (`--eval_only` after `mv`-ing its old eval_compare aside).

`scripts/compare_sweep_evals.py` prints side-by-side per-step tables
(MOS/WER/sim + accent-label counts) across any set of run dirs:

    python scripts/compare_sweep_evals.py exp/run_a exp/run_b exp/run_c

### LR leg result (2026-07-10): the window exists -> self-distill stage

Halving the LR (r1/r4 at 5e-5) reproduced the same collapse trajectory at
half speed -- no frontier shift at the bottom -- BUT sampled the acquisition
phase properly for the first time. **r4 alpha16 lr5e-5 @ step 100 is the best
accent-at-quality point of any arm to date**: MOS 3.356 (base 3.601), sim
0.731 (ABOVE base 0.657), WER 0.028, accent labels england 10/20 vs base
2/20 ('england' is the classifier's known reading of mild Hindi-English;
'indian' labels still only appear below ~2.1 MOS). Ear check: audible accent
with some residual metallicness. r1@5e-5 step 100 is similar on MOS/sim
(3.33/0.734).

That checkpoint becomes the TEACHER for a self-distill stage --
`scripts/make_selfdistill_dataset.py` renders a fine-tuned checkpoint (LoRA
topology from its config, checkpoint from `<run>/ckpt/<step>.pt`) over the
native distill sources into same-timeline pairs, exactly the
make_distill_dataset.py format (reserved prompts excluded, same split seed).
Student config `configs/finetune_selfdistill_hindi_asi_lora_r8.yaml` is ONE
knob (the teacher data) vs `finetune_distill_hindi_asi_lora_r8.yaml`, so
student quality differences attribute to teacher quality. GATE the teacher
renders (classifier one-liner + ears on `data/selfdistill_hindi_asi/wavs`)
before spending the student run; the student cannot exceed its targets --
the bet is only that it smooths part of the residual metallicness.

## Stack-distill: the ASI persona pipeline (2026-07-13)

### Self-distill v1: the repair stage works, the teacher was the ceiling

The r8 student on the r4@100-lr5e-5 teacher renders (teacher gate: england
126/us 95 over 221 clips, MOS 3.288) came out **flat for all 2000 steps**:
MOS 3.40-3.45 (ABOVE its teacher), sim 0.72-0.73, WER back at base (0.017),
england 8-10/20 at every checkpoint. Distillation demonstrably preserves the
teacher's accent, repairs ~+0.1 MOS of texture, and does not drift -- but it
cannot deepen accent, and the v1 persona sounded "ok but not very indian"
(ears). Every fix is therefore teacher-side.

### Teacher probes: the frontier, and what broke it

`scripts/gate_teacher_renders.py` + `scripts/probe_*.sh` render 40-clip
probes (same first-40 sources everywhere, deterministic) and gate them with
classifier label counts + MOS mean/floor; ears remain the final gate. The
`--lora-scale` knob on both dataset renderers multiplies a loaded LoRA delta
at render time. Findings, in order:

- **Amplifying the early delta buys accent linearly with MOS** (r1@100:
  indian 1->2->5->10->17 of 40 as scale goes 2.0->3.0 and MOS 2.92->2.34;
  r4 the same but a gentler slope -- r4@100 x3.0 = indian 12/40 @ 2.60 was
  the converter-side frontier point). Rank-1-expressibility confirmed
  constructively: the accent is one direction; scaling walks along it.
- **More training adds damage, not depth**: r4@300/600 renders LOST shifted
  labels while MOS collapsed (2.60/2.07). The direction is in place by step
  100; everything after is objective damage.
- **A prenet-included latent teacher arm does not break the frontier**
  (`..._prenet_r4_alpha16_lr5e-5.yaml`, 600 steps, dense eval every 50):
  same curve as converter-only (3.32@100 vs 3.36), slightly better early sim
  (0.741), first usable-quality indian blip at step 150 (1/20 @ arm MOS
  3.17), same collapse. Under this objective, extra surface is spent the
  same way.
- **The AccentBridge alone is dominated everywhere**: through stock X-VC its
  renders are us-majority at x1.2 (us 23/40, MOS 2.69) and collapse at
  x1.5/2.0 (2.10/1.54) -- but by EAR the x1.5 renders were the most Indian
  of anything to date ("very choppy and noisy"). The classifier (acoustic
  ECAPA) systematically undercounts the bridge's SEGMENTAL shift; ears
  outrank it, per the standing eval guidance.
- **The stack breaks the frontier**: bridge x1.0 rendered THROUGH the
  accented LoRA checkpoint (`make_distill_dataset.py --ckpt <lora ckpt>
  --config <its finetune config>`) = 32/40 shifted incl 3 indian at MOS
  3.02 (ears: clearly more Indian than the v1 teacher). The accented
  renderer renders bridge-edited features BETTER than stock (+0.17 MOS at
  bridge x1.5). Segments from the bridge + rendering from the LoRA, neither
  pushed to its breaking point.
- Rendering the bridge through the v1 STUDENT is cleaner still (3.09-3.22
  at matched scales) but shallower by ear; the l10 stack won on accent.

### Stack-distill v2 (the ASI persona candidate)

Teacher: bridge x1.0 + r4@100-lr5e-5 renderer -> `data/stackdistill_hindi_asi_l10`.
Students: `finetune_stackdistill_hindi_asi_l10_lora_r8.yaml` and the
`_r4_alpha16` twin (ONE knob: rank). Result -- **r4 == r8 within noise on
every axis at every step**, both flat to 2000:

| step 2000, reserved sources, ASI plan | r8 | r4 |
|---|---|---|
| MOS | 3.169 | 3.139 |
| sim | 0.759 | 0.757 (both project records) |
| shifted labels | 16/20 | 16/20 |
| indian | clb_b0006 @ 3.47 | clb_b0006 @ 3.35 |

First indian labels ever above 3.3 MOS, on the SAME clip in both runs.
vs the v1 student: -0.26 MOS traded for ~2x the accent shift -- the
deliberate depth-for-quality trade, ratified by ear at the teacher stage.
Rank story for the writeup: the accent DIRECTION is rank-1-expressible; the
accent+repair RENDERING compresses to r4 with zero loss (r8 twin identical);
**the persona ships as r4 @ step 2000** (LoRA merges at serving either way).

### Recipe for the next persona (TNI) and open items

Pipeline per persona: (1) latent LoRA arm on the persona crosspairs, lr 5e-5,
early checkpoints -- take step ~100; (2) probe teachers with
`gate_teacher_renders.py` (stack bridge x1.0 through the LoRA ckpt first --
it won for ASI); (3) EARS gate; (4) full render + r4 distill student, 2000
steps; (5) eval + ears at step 2000. For TNI specifically: build a clean
reference FIRST (`scripts/make_clean_reference.py` -- the TNI recordings are
noisy and X-VC conditions on the first 3 s of the reference; serving
loudnorm amplifies any hiss into static).

Open: PersonaPlex live-path ear check of the v2 r4 checkpoint (offline eval
does not cover the 16k->24k + loudnorm serving path); publish via
`scripts/publish_checkpoint.py`; the student-scale ratchet (amplify the
student's own delta -> re-distill) if more depth is wanted; and the
direct-route open question -- recon_ratio > 0.2, 5x data (L2-ARCTIC has
~1100 utts/speaker vs the ~200 pairs used), and DTW-alignment auditing were
never varied, and any of them could make the latent objective stable enough
to obsolete the distill stage entirely.

### Diverse-input eval (2026-07-13): unseen speakers confirmed

New stimulus `data/eval_sources_diverse` (ships as eval_sources_diverse.tgz;
plan `configs/eval_diverse_to_asi.json`, source_group `diverse_unseen` --
eval_checkpoints.py now accepts any declared source_group label): bdl/slt
(native, seen speakers) + UNSEEN accented sources ABA (Arabic M), MBMPS
(Spanish F), SVBI (Hindi F), 10 reserved prompts each. v2 r4 student @2000
vs same-set base: pooled sim 0.716 -> 0.7625 (unseen speakers, near the
0.80 ceiling), shifted labels 41/50, MOS 3.29 -> 2.99 (this input set is
harder for the base model too), WER 0.040 -> 0.072 (the real cost, on
accented inputs). Per speaker: ABA 9/10 shifted at sim 0.78 -- unseen-
speaker generalization CONFIRMED; SVBI (Hindi input) 10/10 shifted incl 5
indian -- accented input compounds rather than conflicts; MBMPS only 5/10 --
Spanish segmental habits partially push through (input-accent interference,
the case the diverse-carrier distill idea targets). Note: prompt b0002
flips indian across four different sources -- accent expression is partly
prompt-dependent. NOT comparable to reserved-set tables (different
stimulus); the reserved eval_compare is preserved as eval_compare_reserved.

## Direct-route falsification (2026-07-13): the collapse is intrinsic

The open question above was answered the same day with three arms, all on
the winning recipe (r4 alpha16 lr5e-5, converter-only, dense eval every 50
of 600 steps), each changing exactly one data-side variable:

1. **recon 0.4** (`..._r4_alpha16_lr5e-5_recon40.yaml`): doubled clean
   anchor. Identical curve through the usable window (3.365/3.071 at
   100/200 vs 3.356/3.048); only cushions the deep-collapse region.
   Anchoring dilutes exposure to the warped supervision, it does not
   neutralize it.
2. **alignment-filtered** (`..._asi_filtered_...`): `audit_alignment_qc.py`
   found ~32% of the training pairs alignment-suspect (global-stretch
   outliers, heavy DTW anchor repairs, local stretch at bounds, worst-decile
   mel distance); dropped them, 136 clean pairs. Step 100 identical to three
   decimals (3.354/0.729/england 10 vs 3.356/0.731/england 10); same
   collapse. The measurably-dirty third was not the poison.
3. **wide ASI-only** (`..._wide_asionly_...`): full-pool rebuild at a 2.0 s
   floor (references held >= 3.0 s), audit-filtered, 349 clean pairs =
   19.2 min ASI audio (75% more than ever before; ships in
   crosspair_hindi_latent_wide.tgz). Step 100: 3.369/0.734/england 10 --
   again identical -- and NO acquisition-window improvement either, so the
   teacher was never data-starved.

Complete falsification chain across the project: adapter capacity (r1 ->
full FT), LR (1e-4/5e-5), hosts (+-prenet), recon anchor (0.2/0.4),
alignment hygiene, data volume. Every arm lands on the same curve. Reading:
cross-speaker frame correspondence is INTRINSICALLY imperfect -- the audit
gates remove gross violations (epenthesis, pauses, production differences),
but the fine-grained mismatch present in every "clean" pair is the training
signal itself. The latent objective cannot be stabilized by data work.

**Consequences.** (a) The two-stage pipeline (brief latent teacher ->
freeze the transform into same-timeline renders -> distill student) is the
design conclusion for this architecture, not a workaround; the v2
stack-distill r4 stands as the ASI persona. (b) The audit/rebuild tooling
remains as permanent dataset hygiene (align_crosspairs.py now writes
pair-unique alignment files -- source_utt alone collides once one source
pairs with several targets -- and audit_alignment_qc.py emits pair-keyed
exclusions; subset dirs need align_meta.json AND a matching
alignment_qc.jsonl or validate_crosspairs.py rejects them). (c) A biased
note for the record: the alignment gates preferentially discard the most
strongly-accented renditions (heavy epenthesis is both "strong accent" and
"unalignable"), and the frame-synchronous architecture cannot reproduce
durational/rhythmic accent at all (dur_delta always 0) -- two structural
reasons the direct route also UNDER-DELIVERS accent depth, and two more
points for the distill stage, which needs no correspondence and inherits
whatever the teacher renders.

## Target-persona rank/LR/objective matrix (2026-07-14)

Harsha's recommendation is now represented as a controlled, one-command
matrix rather than a sequence of ad-hoc follow-ups.  One correction matters:
the linked Thinking Machines Lab article ("LoRA Without Regret",
https://thinkingmachines.ai/blog/lora/) does **not** establish that higher rank
always overfits or that the lowest rank is always best.  It finds that low
rank can become capacity-limited, that standard `alpha/r` parameterization
makes the early optimal LR approximately rank-independent, that smaller
batches can help, and that MLP coverage matters.  Its experiments are on LLMs,
so these are hypotheses to test in X-VC, not speech-model laws.  Our converter
target set already includes attention **and both MLPs**; it is not the
attention-only setup the article warns about.

The deployment objective is also separated cleanly:

- **Target-only/no-pair control:** real ASI recordings as self-pairs
  (`asi_selfpairs_wide`, ~19.2 min after QC), `acoustic_converter` only.  This
  tests whether adapting solely on the destination persona is enough to make
  arbitrary native inputs acquire that persona.  It is a valid control, but
  it never observes native-content -> accented-content conversion, so success
  is not assumed.
- **Source-independent conversion treatment:** filtered L15 same-timeline
  teacher renders.  The sources are generic content carriers, not identities
  being learned; the target is always ASI.  Unlike raw cross-speaker DTW pairs,
  these pseudo-pairs explicitly teach unknown-source content -> ASI-style
  output without frame-correspondence artifacts.

For each objective, the matrix runs ranks 1/2/4 at alpha 16 and batch 4,
crossed with LRs 5e-5 and 1e-4.  Standard `alpha/r` makes early LR behavior
approximately rank-independent, but the linked study still observed some
longer-run rank dependence (especially at rank 1); the small 3x2 factorial
therefore avoids confounding rank with LR.  In these new `acoustic` arms, LoRA covers every
`nn.Linear` under `acoustic_converter`: attention, both MLPs, input/output
projections, and speaker-conditioned AdaLN.  This is the literal
converter-only test and follows the article's all-weight-matrices warning more
closely than the older 69-layer core filter.  The L15 matrix also retains the
known filtered converter+prenet r4 / batch-8 recipe as a historical control,
so the supervisor's converter-only restriction is measured rather than
assumed.  Target-only arms run 1000 steps; L15 students run 2000.  Every arm
gets reserved-source eval and a final-checkpoint diverse/unseen-source eval.
Because that second set is the actual test of the deployment claim, it is a
fail-closed preflight requirement by default; `RUN_DIVERSE_EVAL=0` is the
explicit opt-out.

`scripts/run_persona_experiment_matrix.sh` prepares and gates L15, builds the
real-ASI self-pair dataset if needed, generates explicit configs under
`exp/persona_matrix/configs`, queues all arms **sequentially** on the single
GPU, skips completed arms on restart, refuses to overwrite partial runs,
records per-arm status, and prints a cross-run comparison.  A failed arm is
recorded and later arms continue by default; an L15 teacher-gate failure skips
all L15 students.  The master queue owns each long-running child process group;
INT/TERM/exit cleans up the active render/train/eval instead of orphaning GPU
work.  It deliberately does not invent a scalar winner from MOS,
WER, similarity, and a noisy accent classifier--the Pareto candidates still
require matched-clip listening.

The wide-data result is not discarded but scoped correctly.  Increasing the
direct ASI latent set to 349 clean pairs / ~19.2 min did not change its curve,
so another wide-DTW arm is redundant.  Wider **carrier** coverage may still
improve robustness to live, unseen inputs; it is assessed by the diverse eval
and can be expanded later.  ASI and TNI are not combined merely to claim
40 minutes: they are different target personas/voices and must remain separate
if the goal is one adapter per accent x gender target.

## L15 rebuild safety correction (2026-07-14)

The first L15 rebuild runner treated a 40-file, print-only MOS/accent report as
a "gate" and then always continued to training.  It did not measure content
agreement or speaker similarity, did not filter individual teacher errors, and
described a student MOS near 3.0 plus improved WER as an expected outcome.
That claim was stronger than the evidence: the merge-scale ladder already
showed that the deeper x1.5 delta increases WER, and ordinary distillation can
copy a teacher's recognition errors rather than repair them.

`scripts/gate_teacher_renders.py` now has a backwards-compatible quick-probe
mode and a production, fail-closed mode used by `run_l15_rebuild.sh`.  The
production mode scores **all** L10/L15 renders on matched carriers with the
same metric stack used elsewhere (UTMOS, CommonAccent, Whisper, ERes2Net).  WER
uses a real sidecar transcript when available and otherwise ASR(source) as an
explicitly labelled content-agreement proxy.  It writes per-clip/summary/
rejection CSVs, a machine-readable gate report, and filtered train/val
manifests.  Candidate renders are rejected below MOS 2.8, above utterance WER
0.10, or below similarity 0.65.  Before any GPU training starts, the retained
set must have at least 100 items / 50% of the render pool / 5 validation items,
mean MOS >=2.8, mean WER <=0.06, mean similarity >=0.65, and a paired ordinal
accent-depth gain >=0.05 over L10.  Failure exits non-zero and the rebuild
stops before `run_guarded_train_eval.sh`.

The ordinal accent gate records its assumption (`indian=2`, `england=1`, other
=0) because CommonAccent is not ground truth.  Passing it permits the student
experiment; it does not replace blinded listening.  The claimed 2.95--3.05
student MOS is now documented as a **hypothesis**.  The L10 student recipe is
otherwise held fixed so this remains a controlled deeper-teacher test.  Its
0.1 reconstruction branch is accurately described as teacher-render
reconstruction, not clean-human-audio anchoring; the previously tested real-ASI
anchor did not move the L10 plateau and is not silently added as a second knob.

## Eval contamination (found 2026-07-09)

The evals collected so far did NOT use the designed held-out sources. The
pipeline reserves ARCTIC prompts `b0002-b0012` as eval content
(`build_crosspairs.py` subtracts `EVAL_PROMPTS` from every cross-pair dataset;
verified: zero such rows in the built manifests), and `data/eval_sources` was
originally pinned to exactly those clb/rms clips. The eval runs actually used
**a-prompt clips**. Verified against the dtw-era pair pool: **all 20
actually-used eval sources overlap training on both axes** -- same
speaker+prompt as a training source, and the prompt's L2 rendition as a
training target. `crosspair_hindi_latent_400` (400-pair subset of that pool)
is expected to be PARTIALLY contaminated (~2/10 source-side, ~4/10 target-side
per eval speaker by chance); exact list needs one command against the
container manifests. Separately: all four CMU voices (clb/rms/bdl/slt) are
training source speakers, so even the reserved-prompt eval is seen-speaker --
the evaluation plan itself says "not final unseen-speaker evaluation".

Impact: the bias is inflationary (contamination makes accent/sim look BETTER),
so the negative conclusions -- accent-vs-artifact frontier, MOS gap -- hold a
fortiori, and cross-run rankings shared the same stimulus and remain valid.
What IS suspect: absolute accent counts (e.g. "7/20 indian"), and there is a
concrete risk that the indian flips concentrate on in-train prompts (memorized
renditions), which would mean fine-tuning generalized even less than reported.

Tools added:

- `scripts/check_eval_overlap.py` -- hard preflight gate, now wired into
  `run_guarded_train_eval.sh`: any eval source clip whose prompt was trained on
  (source- or target-side) fails the run before GPU time is spent.
- `scripts/split_eval_by_contamination.py` -- retroactive repair: re-reads the
  per-clip CSVs of every collected run, flags rows clean/contaminated against a
  train manifest, and re-aggregates (n / indian / MOS / sim / WER per step and
  class). Zero GPU. Run it over all `exp/*/eval_compare/*.csv` with the
  latent_400 train manifest to re-read every table in this file.

Going forward: restore `data/eval_sources` to the reserved `b0002-b0012` clips
(or rebuild from prompts the gate accepts). Tables produced after the switch
are a NEW stimulus definition and must not be compared against earlier ones.
The final PersonaPlex gate additionally needs unseen-SPEAKER sources.

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

## Intermediate teacher-depth bracket (2026-07-14)

L10 remains the quality-preserving stack-distillation baseline, but informal
listening found its Hindi accent too weak. L15 was not accepted as the remedy:
although its all-render accent-depth proxy increased from 0.7647 to 0.8824,
only 54/221 renders passed the fixed clip-quality filters, and the retained set
did not increase the classifier's Indian fraction over its paired L10 clips.
This suggests that simply amplifying the teacher can trade texture for a broad
US-to-England shift rather than reliably strengthening Hindi pronunciation.

The exact historical L10 renderer checkpoint (SHA256 `d053e9cd...`) was no
longer present when this bracket was launched. It is not scientifically valid
to relabel a similarly sized checkpoint as that teacher. The sweep therefore
uses the surviving alignment-filtered step-100 renderer (SHA256
`90b2f27e...`), whose recorded step-100 metrics matched the original teacher
to three decimals, and renders a new scale-1.0 baseline from it before doing
anything else. The checkpoint SHA is asserted at runtime. This makes the new
L10/L12/L12.5 bracket internally controlled, while keeping the historical L10
result separate.

`scripts/run_intermediate_teacher_sweep.sh` brackets that matched baseline at
renderer LoRA scales 1.2 (L12) and 1.25 (L12.5). Each scale is rendered over
the complete carrier set and passed through the same fail-closed MOS, WER,
speaker-similarity, retained-count, retained-fraction, and validation-count
floors used for L15. The paired ordinal accent-depth floor is pre-registered at
0.02 rather than L15's 0.05 because these are deliberately smaller treatments;
the CommonAccent labels remain a screening proxy and do not replace listening.
No clip-quality threshold is relaxed.

Only candidates that pass receive one controlled student run using r4,
alpha=16, LR=1e-4, batch 8, acoustic-converter+prenet targets, and 2000 steps.
The two new student configs differ from the established L10 r4 recipe only in
their gate-filtered teacher dataset. This is a teacher-depth comparison, not a
rank/LR sweep. A gate failure skips that student and continues to the other
candidate. `TRAIN_PASSING=0` provides a render-and-gate-only mode, and
`REUSE_INTERMEDIATE_RENDER=1` safely reuses complete render manifests.

## Genuine phone-aware target realization (2026-07-14)

Added a separate, opt-in AccentBridge experiment for the remaining causal
hypothesis: pronunciation edits may need supervision at phone interiors rather
than larger LoRA/teacher scale. This is not the previous MFA run. That run
requested phones but fell back to word tiers; word-MFA had worse alignment QC
than DTW and poor conversion results.

`scripts/annotate_accentbridge_phone_supervision.py` now requires genuine
source and target phone tiers and sequence-aligns corresponding phones. MFA is
metadata only: native audio and features are never warped or resampled. The
annotator adaptively trims uncertain boundaries, rejects extreme phone-duration
ratios, and derives stable phone-level realization weights from
onset/middle/offset and variance differences. It fails closed on word-only
TextGrids, low phone-label agreement, missing grids, insufficient usable phones,
or less than 80% retained coverage in either train or validation.

`scripts/train_accentbridge.py --phone-aware` consumes pristine native-timeline
features and matches the bridge's target-relative edit over three temporal
regions of each phone plus whole-phone variance. It retains identity,
smoothness, and residual-delta anchors. Normal training behavior is unchanged
without the flag; combining the old frame-gap weighting with phone-aware
weighting is rejected as a confounded experiment. Phone-aware checkpoints
retain the existing AccentBridge architecture/config schema, so rendering and
streaming inference need no MFA and remain compatible with
`make_distill_dataset.py`.

`scripts/run_phoneaware_accentbridge_sweep.sh` pre-registers three conservative
arms (`lambda_phone` 0.1, 0.25, 0.5), with L10 as the external baseline. The
trainer now writes TensorBoard events, raw CSV curves, validation CSV, and a
static PNG so loss components, identity drift, gradient norm, and validation
phone-gap closure can be audited together. See
`docs/phoneaware_accentbridge.md` for inputs and decision rules.

`scripts/run_phoneaware_mfa_queue.sh` now queues source/target validation and
alignment, then calls `audit_mfa_phone_tiers.py` and archives only the MFA
outputs needed on the training machine. The queue fails on missing corpora,
WAV/transcript count mismatch, missing MFA executable, pre-existing ambiguous
outputs, or any TextGrid without a genuine phone tier. It does not construct a
time map and never launches training automatically from the MFA environment.

`scripts/prepare_phoneaware_mfa_corpus.py` builds the exact target-speaker MFA
corpus from pristine raw paths in the latent-crosspair manifests. It refuses
to overwrite existing output and hard-checks the selected counts, prompt split,
audio existence, mono channel layout, and 16 kHz sample rate before MFA runs.
The MFA queue defaults to the MFA 3.x compatible `english_us_mfa` dictionary
and `english_mfa` acoustic model rather than treating an ARPA dictionary name
as an acoustic-model identifier.
The expensive per-speaker lattice-based transcription test is now opt-in via
`MFA_TEST_TRANSCRIPTIONS=1`; standard validation remains mandatory, and the
queue exposes a conservative `MFA_NUM_JOBS` setting (default 4).
Phone-supervision annotation now applies the requested target-speaker filter
before computing coverage, so mixed ASI/TNI feature shards can be supervised
against an intentionally ASI-only MFA corpus without counting TNI as missing.

### Repository-structure integration

The phone-aware branch now incorporates the package/config/entry-point/test
layout from `target-conditioned-lowrank-lora`. Reusable phone supervision
lives under `xvc.data`, operator scripts remain stable compatibility entry
points, and phone tests follow `tests/unit`. During integration we found that
the refactor's unanchored `data/` ignore rule had silently excluded the
documented `xvc/data` schema and validator modules; the ignore is root-anchored
and the missing package is restored here before adopting the new validator
wrappers.

### Causal semantic/acoustic stream localization audit

Added `audit_xvc_accent_streams.py` and
`run_xvc_accent_stream_audit.sh` to test where source pronunciation is carried
before building another accent model. The audit independently substitutes real
ASI semantic activations and real ASI quantized-acoustic activations at X-VC's
50 Hz concatenation point, with native/native and original-timeline ASI
round-trip controls. MFA phones provide only a diagnostic frame map; no mapped
feature becomes training data and no weight is updated. A one-pass metric run
produces MOS, WER, speaker-similarity, and accent summaries for all five
conditions. Pure mapping tests cover identity, duration scaling, monotonicity,
and fail-closed behavior. See `docs/xvc_accent_stream_audit.md`.

### Canonical pristine Hindi/ASI dataset

Added `build_pristine_parallel_dataset.py` to promote the already validated MFA
corpus into a relocatable long-lived dataset. It copies pristine mono/16 kHz
native and ASI recordings, transcripts, and genuine phone-tier TextGrids;
preserves prompt-disjoint train/validation manifests; and writes complete
SHA-256 checksums. It never uses symlinks, DTW, warping, or resampling, and its
metadata warns that prompt-parallel recordings are not frame-synchronous.
The shared validator now honors that explicit declaration by checking source
and target durations independently instead of demanding equal sample counts;
all remaining audio/split/path checks stay active, and canonical SHA-256
manifests are verified in full.

## Joint target-persona mapper after the causal stream audit (2026-07-15)

The ASI stream-swap audit found that neither ASI semantic features alone
(1/18 Indian) nor ASI acoustic codes alone (0/18) were sufficient, while the
matched pair was materially stronger (6/18 mapped; 9/18 on the original ASI
timeline). This closes the single-stream AccentBridge path.

The new joint mapper edits both source-side streams through one causal trunk,
then returns acoustic predictions to X-VC's frozen codebook. It is explicitly a
target-persona adapter: no source-speaker id enters the model, while the normal
frozen X-VC ASI reference path continues to provide target voice. Pristine
native/ASI sequences remain unwarped; MFA phones bound a differentiable
monotonic training loss. The controlled runner balances source speakers and
requires prompt-disjoint evaluation on at least two unseen source speakers,
with automatic MOS, WER, target-speaker-similarity, and Indian-accent gates.

See `docs/joint_persona_mapper.md` and
`scripts/run_joint_persona_mapper_sweep.sh`.
