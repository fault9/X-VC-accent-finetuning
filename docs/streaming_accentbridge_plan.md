# Streaming AccentBridge: feasibility report and staged plan

Scope: live/streaming zero-shot accent conversion for Hear-Me-Out (native
English live speech -> assigned accented English, Hindi pilot), re-scoped
around the deployment constraint that a permanent large seq2seq module in the
live path is NOT assumed acceptable.

Prior findings this plan builds on (all in CHANGES.md, with diagnostics on the
`lora-accent-adapters` branch): X-VC's content streams remain accent-neutral
for native input (decoded warped cross-pair inputs -> 0% Indian labels despite
Hindi timing + Hindi reference); fine-tuning the renderer forces accent through
a content-preserving pathway and buys accent only at the price of
metallic artifacts (invariant frontier across rank, target set, LR, anchor);
LoRA implementation verified correct; L2 target audio calibrated clean
(MOS ~3.8-3.9); best current checkpoint: LoRA acoustic_converter+prenet r8 on
latent_400 @ ~1000 steps.

---

## 1. Where the smallest possible bridge inserts

X-VC's runtime content path (verified in code, streaming and offline are the
same graph):

```
source wav (16 kHz)
  ├─ semantic_encoder.extract_and_encode -> speech_tokens   12.5 Hz discrete
  │    └─ embed_ids -> (B, T, 1280)                          12.5 Hz continuous
  │         └─ semantic_adapter -> (B, 1024, T')             50 Hz  continuous   ◄ INSERTION POINT
  └─ acoustic_encoder -> quantizer -> zq (B, 1024, T')       50 Hz  quantized
       └─ concat(sem, zq) -> prenet -> acoustic_converter -> acoustic_decoder -> wav
```

**Chosen insertion point: the post-`semantic_adapter` stream** — (B, 1024, T)
at **50 Hz (20 ms/frame)**. Exact code locations (one line each):

| path | file | where |
|---|---|---|
| streaming runtime | `bins/infer_utils.py` `run_stream_chunk_forward`, line 121 | after `semantic_adapter(...)`, before `torch.cat` at line 128 |
| offline inference | `models/codec/sac/model.py` `XVC.inference`, ~line 443 | after the `semantic_adapter` call |
| training / distillation | `models/codec/sac/model.py` `_latent_aligned_source`, ~line 234 | after the adapter inside the per-item loop |

Why this level and not the alternatives:

- **Discrete token level (12.5 Hz)**: editing WhisperVQ ids is the "big seq2seq"
  shape we're avoiding; 80 ms granularity; insert/delete pressure. Kept only as
  a fallback if the continuous level proves accent-invariant.
- **Acoustic latents `zq`**: carries source spectral detail entangled with
  speaker identity; editing it risks the zero-shot voice-cloning property.
  Extraction saves it anyway so the analysis can check where accent is visible.
- **Prenet input (concat)**: same as editing both streams at once; strictly
  more risk than semantic-only for no extra insight in v1.
- **Post-adapter semantic (chosen)**: pronunciation lives here by construction
  (it is the only content stream whose job is "what was said, how"); 20 ms
  frames give the finest lookahead granularity; continuous residual editing is
  length-stable and streaming-trivial; the adapter output is already 1024-d,
  matching a tiny 1x1-conv bridge.

## 2. Can it be causal / limited-lookahead? Latency accounting

Yes. The bridge (`models/accentbridge.py`) is a stack of dilated **causal**
Conv1d blocks with a single explicit `lookahead_frames` budget (right-pad by L,
crop L — every output frame sees input up to t+L exactly). Zero-init output
projection = exact identity at init (same philosophy as LoRA's B=0).

Latency model, on top of the existing streaming loop (`run_streaming`: emits
`current_ms` per hop from a history+current+future window; deployment default
current=320 ms, future=0):

| bridge config | algorithmic latency added | compute added (per 320 ms hop) |
|---|---|---|
| lookahead 0 (pure causal) | **+0 ms** | ~0.6M params, sub-ms GPU |
| lookahead 4 frames | **+80 ms** (set `future_ms=80`) | same |
| lookahead 8 frames | **+160 ms** (set `future_ms=160`) | same |

Key mechanical fact: bridge lookahead maps **1:1 onto the existing `future_ms`
window knob** — the streaming loop already supports buffering future input; a
lookahead-L bridge just requires `future_ms = L*20 ms` so emitted frames have
their right context. No new streaming machinery. The bridge is stateless
(convolutional), so X-VC's re-encode-per-chunk design needs no KV cache; left
context comes from the chunk's history region (receptive field ~2^layers
frames << the 1.7 s default history window).

Compute: at 50 Hz, a hop advances 16 frames; a 0.6M-param conv over 16 frames
is microseconds-to-sub-ms on the GPU already running the DiT + vocoder per hop.
Compute latency is a rounding error; **the only real cost is lookahead, and it
is a config integer.**

## 3. Can it be distilled into LoRA? (PATH A mechanics)

Yes, and this is the deployment-preferred endgame. The critical insight from
the failed fine-tuning campaign: regression toward **unreachable** targets
(real L2 audio through native content streams) produces averaging = metallic.
Distillation replaces the target with something **reachable by construction**:

```
teacher:  y_T = XVC_stock( bridge(sem(x)), zq(x), cond )   # on-manifold output
student:  y_S = XVC_LoRA (        sem(x),  zq(x), cond )   # raw features
loss:     || y_S - y_T ||  (waveform/mel)  and/or  feature-space:
          || prenet_LoRA(concat(sem,zq)) - prenet_stock(concat(bridge(sem),zq)) ||
```

The teacher's outputs are actual outputs of the stock renderer — clean,
on-manifold, artifact-free by construction — so the student is never asked to
contradict its inputs toward an unreachable point. The hypothesis to test:
**absorbing a realizable content edit into LoRA does not reproduce the metallic
failure**, because the failure came from unreachability, not from LoRA.

Constraints and expectations:

- Distill from a **lookahead-0 teacher** for exactness: a causal student can
  only absorb a causal teacher. (A lookahead-8 teacher can still be distilled
  approximately; expect softened accent.)
- Student target set: the proven LoRA infra (acoustic_converter+prenet r8,
  configs and hygiene checks already on `lora-accent-adapters`) — but likely
  the more natural host is **semantic_adapter LoRA** since the edit lives
  upstream of the concat; both are one-config experiments.
- **Success** = student ≈ teacher on the standard eval (accent within ~1-2
  clips of teacher, MOS within ~0.1, WER flat) with merged-LoRA runtime =
  stock architecture = **zero added latency, zero new runtime modules**.
- **Failure** = MOS collapses again during distillation. That would be a
  genuinely new fact (even realizable edits can't be absorbed by low-rank
  deltas) and would promote PATH B from fallback to the plan.
- Data for distillation is free: teacher outputs are generated offline from
  any native speech (no paired L2 data needed at this stage — the pairing is
  already baked into the bridge).

## 4. What changes for the Hear-Me-Out runtime

- **PATH A (distilled)**: nothing. Runtime stays `X-VC + merged LoRA` — the
  merge tooling and checkpoint-compatibility matrix from `lora-accent-adapters`
  apply unchanged. Zero new modules, zero added latency.
- **PATH B (tiny runtime bridge)**: `services/xvc/server.py` is **not in this
  repo** (no `services/` dir; it lives in Hear-Me-Out) — the in-repo mirror of
  the serving path is `bins/infer_utils.py::run_stream_chunk_forward`. Required
  changes, wherever that logic is replicated:
  1. load the bridge ckpt next to the model (one `AccentBridge(**cfg)` +
     `load_state_dict`, ~5 lines in the server's model-load path);
  2. one line in the chunk forward: `sem = bridge(sem)` after the adapter;
  3. if lookahead > 0: set the window's `future_ms` to `lookahead*20` (already
     a supported parameter of the streaming loop).
- Per-accent switching in either path: PATH B swaps a ~0.6M-param bridge ckpt
  (trivially hot-swappable); PATH A swaps merged LoRA checkpoints or keeps
  per-accent unmerged adapters on a shared frozen base.

## 5. What to test first (ordered by information per GPU-hour)

1. **Extraction + analysis** (`extract_accentbridge_pairs.py`,
   `analyze_accentbridge_pairs.py`) — minutes, decides everything. If the
   post-adapter representation is accent-invariant (verdict in the analysis
   output), PATH A/B at this level are dead and we know it before training
   anything. If tokens mismatch heavily while sem features don't, the signal
   is discrete-level and v2 targets token embeddings instead.
2. **Bridge L0 smoke → L0/L4/L8 sweep** (`train_accentbridge.py`) — CPU-to-
   minutes-GPU. Feature-level gap-closed + identity-drift metrics first; the
   lookahead sweep quantifies how much accent needs future context (a real
   phonetics question answered with three tiny runs).
3. **Synthesis spot-check** (`eval_accentbridge.py --synthesize`) — a few wavs
   through the real stack with the one-line insertion, scored by the standard
   eval/calibration stack. This is the first audio evidence and uses ~zero GPU.
4. **PATH A distillation** — only after 3 shows the bridged teacher moves
   accent at acceptable MOS.
5. **PATH C in parallel, only if idle GPU**: the AdaLN config
   (`configs/finetune_crosspair_hindi_latent_400_lora_acoustic_adaln_r8.yaml`)
   is one command with the existing runner; reference re-pin for ASI is a
   calibration-stack screen + `make-targets` re-run; a feature-matching loss
   is sketched below but NOT implemented (real trainer work).

Explicit go/no-go gates: analysis verdict must be GO before any bridge
training; bridge must close >~30% of the feature gap at identity drift <~2%
before synthesis; bridged synthesis must beat the best LoRA checkpoint's
accent-at-MOS point before distillation.

## 6. PATH C notes (kept deliberately small)

- **AdaLN LoRA**: config shipped (above). Expected bounded upside (global
  coloration only); run once to close the freeze-set question.
- **Feature-matching / perceptual loss (sketch only)**: add an L1 on
  mel-extractor features and/or intermediate `acoustic_decoder` activations
  between student output and target, replacing part of the plain regression
  loss, to let the model commit rather than average. Touches
  `models/codec/base/base_codec_trainer.py` loss assembly — a real trainer
  change; do not build unless PATHs A and B both fail.
- **Reference re-pin**: the ASI pinned reference scores MOS 3.06 vs its corpus
  3.80 (calibration finding). Screen ASI val clips with
  `calibrate_eval_floor.py`, re-run `eval_checkpoints.py make-targets`
  preferring high-MOS clips, re-baseline. Changes the pinned stimulus — tables
  before/after are not comparable.

## 7. Known caveats / blockers

- **Timeline mismatch (training vs runtime)**: bridge training pairs live on
  the L2 timeline (native features warped by the dataset's DTW maps — the same
  proven machinery training uses, `XVC._sample_bct_at_normalized_positions`).
  At runtime the bridge sees the native timeline. v1 accepts this (per-frame
  feature mapping transfers; timing distribution differs); if analysis shows
  high sensitivity, v2 inverts the maps to build native-timeline targets.
- **Eval contamination**: any synthesis eval must use the reserved held-out
  sources (`arctic_b0002-b0012`; see CHANGES.md "Eval contamination" and the
  preflight gate). Bridge feature metrics use val-split pairs, which are clean.
- **Rhythm ceiling stands**: a length-stable bridge cannot change durations —
  segmental accent only, same ceiling as everything else in this architecture.
  Documented, not solved, by this plan.
- **`services/xvc/server.py` absent from this repo**: PATH B integration
  specifics for Hear-Me-Out are mirrored against `run_stream_chunk_forward`;
  final wiring happens in the HMO repo.
- **WhisperVQ streamability**: the tokenizer is block-causal by design
  (CausalConv1d + block-causal attention masks in
  `models/codec/sac/modules/semantic_encoder.py`), consistent with the
  existing chunked streaming; the bridge adds no new assumption.

## 8. Smoke command set (container, conda xvc, repo root)

```bash
# 1. extract pairs (val split, 40 pairs, stock encoders)
python scripts/extract_accentbridge_pairs.py \
    --config configs/finetune_crosspair_hindi_latent_400_lora_acoustic_r8.yaml \
    --ckpt ckpts/xvc.pt \
    --data-root data/crosspair_hindi_latent_400 \
    --split val --limit 40 --device 0 --out data/accentbridge_pairs
# (repeat with --split train --limit 400 for the training set)

# 2. analyze -> go/no-go verdict
python scripts/analyze_accentbridge_pairs.py \
    --pairs-dir data/accentbridge_pairs/val --out exp/accentbridge_analysis

# 3. tiny bridge smoke (lookahead 0, 200 steps, ~minutes)
python scripts/train_accentbridge.py \
    --train-dir data/accentbridge_pairs/train \
    --val-dir data/accentbridge_pairs/val \
    --lookahead-frames 0 --steps 200 --limit 64 \
    --out exp/accentbridge_l0_smoke

# 4. feature-level shift report
python scripts/eval_accentbridge.py \
    --val-dir data/accentbridge_pairs/val \
    --bridge-ckpt exp/accentbridge_l0_smoke/bridge.pt \
    --out exp/accentbridge_l0_smoke/eval

# 5. optional: 6 wav pairs (bridged vs plain) through the real stack
python scripts/eval_accentbridge.py \
    --val-dir data/accentbridge_pairs/val \
    --bridge-ckpt exp/accentbridge_l0_smoke/bridge.pt \
    --out exp/accentbridge_l0_smoke/eval \
    --synthesize 6 --config configs/xvc.yaml --ckpt ckpts/xvc.pt --device 0
```

## 9. Expected outcome tree

- Analysis says GO, bridge closes the gap, synthesis moves accent at clean MOS
  -> distill (PATH A), runtime unchanged, **zero added latency**.
- Bridge works but only with lookahead -> PATH B at an explicit +80/160 ms,
  tiny module, `future_ms` knob; distillation optional/approximate.
- Analysis says ACCENT-INVARIANT at the semantic level -> try token-embedding
  level (80 ms granularity); if that too is invariant, X-VC's content
  representation cannot express the accent delta and the honest conclusion is
  a framework change (accent-capable content representation), documented with
  this measurement as the evidence.
```
