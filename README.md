# X-VC

[![arXiv](https://img.shields.io/badge/arXiv-2604.12456-b31b1b.svg)](https://arxiv.org/abs/2604.12456)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/chenxie95/X-VC)
[![Demo Page](https://img.shields.io/badge/Demo-Project%20Page-blue)](https://x-vc.github.io)
[![Online Demo](https://img.shields.io/badge/Demo-Online%20Demo-green)](https://x-vc.sjtuxlance.com)

Official code release for **X-VC: Zero-shot Streaming Voice Conversion in Codec Space**.

![X-VC overview](figures/overview.png)

## Online Demo

You may visit our real-time voice conversion system on the [online demo page](https://x-vc.sjtuxlance.com). First, choose a preset **target voice** or upload/record your own reference audio, then use the preview player to check the target voice. Click **Start Stream** to start real-time conversion. After speaking, click **Stop** to end the session; the saved input and converted output audio will appear in the **Saved Audio** section for playback. Please use headphones for a better experience, and report any issues you encounter.

![Online Demo](figures/demo.png)

## Environment Setup

### 1. Clone

```bash
git clone https://github.com/Jerrister/X-VC.git
cd X-VC
```

### 2. Create conda environment and install dependencies

```bash
conda create -n xvc python=3.10 -y
conda activate xvc
pip install -U pip
pip install -r requirements.txt
```

### 3. Prepare pretrained models

Prepare:
- [GLM-4-Voice-Tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer) (for semantic tokenization)
- [ERes2Net speaker encoder](https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k) (for speaker feature extraction)

Then set paths in [`configs/xvc.yaml`](configs/xvc.yaml), especially:
- `model.generator.semantic_encoder.encoder.from_pretrained`
- `model.generator.semantic_encoder.cfg`
- `model.generator.speaker_encoder.pretrained_dir`

### 4. Prepare checkpoints

Put [X-VC checkpoints](https://huggingface.co/chenxie95/X-VC) under `ckpts/`, for example:

```text
ckpts/
  xvc.pt
```

## Inference

### Single-pair Inference

Use [`scripts/infer_single.sh`](scripts/infer_single.sh).

```bash
bash scripts/infer_single.sh
```

Key arguments in this script:
- `current=0` for offline inference.
- `current>0` for streaming inference.
- `chunk/current/future/smooth` control streaming behavior.

Outputs are saved under `save_dir` (default: `outputs/xvc_single`).

### Batch Offline Inference (SeedTTS-eval as example)

Use [`scripts/batch_infer_seedtts_offline.sh`](scripts/batch_infer_seedtts_offline.sh).

```bash
bash scripts/batch_infer_seedtts_offline.sh
```

This script reports:
- `saved_dir`
- `total_rtf`

### Batch Streaming Inference (SeedTTS-eval as example)

Use [`scripts/batch_infer_seedtts_stream.sh`](scripts/batch_infer_seedtts_stream.sh).

```bash
bash scripts/batch_infer_seedtts_stream.sh
```

This script reports:
- `saved_dir`
- `avg_latency_ms`

## Training

### Step 1: Prepare pretrained dependencies

Before training, prepare the required pretrained dependencies:
- [SAC pretrained checkpoint(s)](https://huggingface.co/Soul-AILab/SAC-16k-62_5Hz) (for model initialization)

Then set corresponding paths in [`configs/xvc.yaml`](configs/xvc.yaml), especially:
- `model.generator.checkpoint`
- `model.discriminator.checkpoint`

### Step 2: Prepare training data

Organize your training/validation data in JSONL format and set:
- `datasets.train`
- `datasets.val`

in [`configs/xvc.yaml`](configs/xvc.yaml).

### Step 3: Modify training configs

You can adjust training behavior in:
- [`configs/xvc.yaml`](configs/xvc.yaml) (main training config)
- [`configs/ds_stage2.json`](configs/ds_stage2.json) (DeepSpeed config)

### Step 4: Start training

Use [`scripts/train.sh`](scripts/train.sh).

```bash
bash scripts/train.sh
```

Notes:
- Default training engine is DeepSpeed (`configs/ds_stage2.json`).
- Main experiment config is `configs/xvc.yaml`.
- Set your `WANDB_API_KEY` in `scripts/train.sh` before running if you use wandb logging.

## Data Format

Training config points to JSONL files in `configs/xvc.yaml`:
- `datasets.train`
- `datasets.val`

Each JSONL line should be a JSON object.

Required fields:
- `target_utt`
- `source_wav_path`
- `target_wav_path`

Optional field:
- `source_utt`

Minimal example:

```json
{"source_utt":"utt_0001","source_wav_path":"<path_to_source>","target_utt":"utt_0002","target_wav_path":"<path_to_target>"}
```

## Accent Fine-tuning (L2-ARCTIC)

This fork adds a reproducible pipeline to fine-tune X-VC on L2-ARCTIC accent groups
(Arabic, Spanish, Chinese, Hindi — 2 speakers each, 1M+1F, the study's exact
conversion-target voices) plus a native CMU-ARCTIC reference group, starting from
the released X-VC checkpoint.

- **Method and rationale (for reviewers):** [`CHANGES.md`](CHANGES.md)
- **Step-by-step operator guide:** [`docs/finetuning.md`](docs/finetuning.md)
- **Phone-aware AccentBridge experiment:**
  [`docs/phoneaware_accentbridge.md`](docs/phoneaware_accentbridge.md)
- **Hindi/ASI semantic-vs-acoustic stream audit:**
  [`docs/xvc_accent_stream_audit.md`](docs/xvc_accent_stream_audit.md)
- **Code architecture / refactor map:** [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md),
  [`docs/architecture.md`](docs/architecture.md)

### Consolidated entry points

```bash
pip install -e .                                   # installs the xvc package
pytest tests/                                      # CPU test suite (no GPU, no checkpoints)

python scripts/train.py experiment=lora_hindi_asi_r4          # train (see docs/training.md)
python scripts/infer.py checkpoint=exp/<run>/ckpt/000100.pt \
    source=src.wav target=ref.wav output=out.wav              # convert (docs/inference.md)
python scripts/validate_dataset.py data/<crosspair_root>      # dataset preflight
python scripts/inspect_checkpoint.py <file.pt>                # what is in this checkpoint?
```

Experiments compose config groups under
`configs/{model,adapter,dataset,training,experiment}/`
([`configs/LEGACY_MAPPING.md`](configs/LEGACY_MAPPING.md)); LoRA / freezing /
dataset-schema / checkpoint utilities live in the `xvc/` package
([`docs/lora.md`](docs/lora.md), [`docs/datasets.md`](docs/datasets.md),
[`docs/checkpoints.md`](docs/checkpoints.md)). All historical commands keep
working — see [`MIGRATION.md`](MIGRATION.md).

Reusable phone-tier parsing and sequence matching live in
`xvc.data.phone_supervision`; MFA corpus preparation, alignment, annotation,
and controlled sweep commands remain explicit operator scripts under
`scripts/`.

Quick start:

```bash
python scripts/prepare_finetuning_data.py select          # curate from raw corpora
python scripts/prepare_finetuning_data.py manifest --joint
python scripts/eval_checkpoints.py make-targets           # pin eval reference clips
python bins/smoke_test.py --stage all --config configs/finetune_arabic.yaml
python bins/smoke_test.py --stage train --config configs/finetune_arabic.yaml  # full loop
bash scripts/finetune.sh --accent joint                   # or: --accent all
python scripts/eval_checkpoints.py run --run-dir exp/finetune_joint \
    --source-dir data/eval_sources --targets-dir data/eval_targets \
    --include-base ckpts/xvc.pt                           # checkpoint selection curves
```

"Option A" baseline: warm-start the released checkpoint, freeze everything except
`acoustic_converter` and `prenet`, and train in self-reconstruction mode. See
[`configs/finetune_arabic.yaml`](configs/finetune_arabic.yaml) and the two documents above.

## Acknowledgements

This codebase builds upon open-source components from [SAC](https://github.com/Soul-AILab/SAC) and the broader audio generation ecosystem.

The accent fine-tuning pipeline builds on the original **X-VC** release
(https://github.com/Jerrister/X-VC); the project remains under the MIT License (see [`LICENSE`](LICENSE)).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{zheng2026xvczeroshotstreamingvoice,
      title={X-VC: Zero-shot Streaming Voice Conversion in Codec Space}, 
      author={Qixi Zheng and Yuxiang Zhao and Tianrui Wang and Wenxi Chen and Kele Xu and Yikang Li and Qinyuan Chen and Xipeng Qiu and Kai Yu and Xie Chen},
      year={2026},
      eprint={2604.12456},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2604.12456},
}
```
## License

This project is licensed under the MIT License.
