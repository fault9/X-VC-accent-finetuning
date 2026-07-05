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
