# Fresh container setup

This is the authoritative setup checklist for the maintained Hindi/ASI
target-persona pipeline. Run commands from the repository root.

## 1. GPU runtime first

Do not install or start training until both checks pass:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
PY
```

`Failed to initialize NVML` is a container/runtime problem. Reinstalling the
Conda environment does not repair it; restart the GPU-enabled container first.

## 2. Python environment

The upstream X-VC training/inference stack is declared in `requirements.txt`.
The maintained mapper additionally uses TensorBoard during training and the
same evaluation stack as `scripts/eval_checkpoints.py`:

- `faster-whisper` for WER;
- `speechbrain` for CommonAccent classification;
- `pyloudnorm` for optional loudness normalization;
- `soxr` and `soundfile` for audio I/O/resampling; and
- the SpeechMOS UTMOS model loaded through `torch.hub`.

```bash
conda create -n xvc python=3.10 -y
conda activate xvc
pip install -r requirements.txt
pip install -e ".[dev]"
```

MFA is deliberately not installed in this environment. Run it in the separate
MFA environment documented in [`mfa_phone_alignment.md`](mfa_phone_alignment.md).

## 3. External model assets

The repository does not track large assets. Restore these paths:

```text
ckpts/xvc.pt
pretrained/speech_eres2net_sv_en_voxceleb_16k/
data/hindi_asi_pristine_parallel_221/
data/eval_targets/ASI.wav
data/eval_sources_joint_persona_clean/*.wav
```

`configs/xvc.yaml` also resolves `zai-org/glm-4-voice-tokenizer` through the
Hugging Face cache. Internet is needed on the first load unless the snapshot is
already restored.

## 4. Verify before a long run

```bash
python scripts/validate_crosspairs.py \
  --data-root data/hindi_asi_pristine_parallel_221 \
  --min-duration 2.6

python -m pytest \
  tests/unit/test_build_pristine_parallel_dataset.py \
  tests/unit/test_prepare_phoneaware_mfa_corpus.py \
  tests/unit/test_stream_swap.py \
  tests/unit/test_joint_persona_mapper.py
```

Run a two-step plumbing smoke before the full matrix:

```bash
LOOKAHEADS="4" DISCRETE_WEIGHTS="0.25" STEPS=2 \
EXP_ROOT=exp/joint_persona_discrete_asi_smoke \
bash scripts/run_joint_persona_discrete_sweep.sh
```

This smoke verifies the full train/render/score plumbing. With only two update
steps, the final scientific accent gate may correctly return non-zero.

Then use the registered sweep:

```bash
mkdir -p exp/run_logs
nohup bash scripts/run_joint_persona_discrete_sweep.sh \
  > exp/run_logs/joint_persona_discrete_asi.out 2>&1 &
```

Code dependencies are organized as follows:

- `models/joint_accent_mapper.py`: mapper architecture;
- `xvc/training/monotonic.py`: phone-local monotonic and discrete-code losses;
- `xvc/data/`: dataset schemas, validation, TextGrid and stream alignment;
- `scripts/train_joint_persona_mapper.py`: training loop and TensorBoard;
- `scripts/eval_joint_persona_mapper.py`: matched stock/candidate rendering; and
- `scripts/score_xvc_accent_stream_audit.py`: MOS/WER/similarity/accent scoring.

All generated data, checkpoints, logs, caches, and archives stay under ignored
`data/`, `pretrained/`, `ckpts/`, and `exp/` paths.
