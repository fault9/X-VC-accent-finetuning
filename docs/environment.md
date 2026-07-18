# Fresh container setup

Run commands from the repository root.

## 1. Verify the GPU runtime

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

Restart the GPU-enabled container if NVML cannot initialize. Reinstalling the
Conda environment does not repair a missing GPU runtime.

## 2. Install X-VC dependencies

```bash
conda create -n xvc python=3.10 -y
conda activate xvc
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

The evaluation stack uses faster-whisper for WER, SpeechMOS/UTMOS through
`torch.hub`, ERes2Net for target-speaker similarity, and soundfile/soxr for
audio I/O. No MFA, accent classifier, or pronunciation model is required.

## 3. Restore external assets

```text
ckpts/xvc.pt
pretrained/speech_eres2net_sv_en_voxceleb_16k/
data/vctk_naturalness_4voice/
```

`configs/xvc.yaml` resolves `zai-org/glm-4-voice-tokenizer` through the Hugging
Face cache. The first load needs internet unless the snapshot is already
cached.

## 4. Validate and smoke-test

```bash
python scripts/validate_crosspairs.py \
  --data-root data/vctk_naturalness_4voice \
  --min-duration 1.8

SMOKE=1 bash scripts/run_vctk_persona_naturalness_sweep.sh
```

The smoke runs one persona, one LoRA arm, two training steps, and two evaluation
sources. After it succeeds, launch the registered queue documented in
[`vctk_persona_naturalness.md`](vctk_persona_naturalness.md).
