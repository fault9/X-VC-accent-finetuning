"""Two-pair GPU smoke: extract -> train two steps -> decode -> render.

Wraps `scripts/run_persona_codec_smoke.sh`; needs the real X-VC checkpoint,
dataset, ASI reference, and a CUDA GPU, so it is excluded from the default
CPU run (see pytest.ini).  Run on the training container with:

    pytest tests/smoke/test_persona_codec_smoke.py -m "gpu and slow" -s
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    REPO_ROOT / "ckpts" / "xvc.pt",
    REPO_ROOT / "data" / "hindi_asi_pristine_parallel_221" / "manifests" / "train.jsonl",
    REPO_ROOT / "data" / "eval_targets" / "ASI.wav",
    REPO_ROOT / "data" / "eval_sources_joint_persona_clean",
]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.slow
def test_two_pair_smoke(tmp_path):
    missing = [str(path) for path in REQUIRED if not path.exists()]
    if missing:
        pytest.skip(f"missing runtime assets: {missing}")
    if not _cuda_available():
        pytest.skip("needs a CUDA GPU")

    env = dict(os.environ)
    env.update({"EXP_ROOT": str(tmp_path / "persona_codec_smoke"), "MAX_PAIRS": "2", "STEPS": "2"})
    result = subprocess.run(
        ["bash", "scripts/run_persona_codec_smoke.sh"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, f"smoke failed:\n{result.stdout}\n{result.stderr}"

    exp = tmp_path / "persona_codec_smoke"
    assert (exp / "pairs" / "meta.json").is_file()
    assert (exp / "geometry" / "geometry_report.json").is_file()
    assert (exp / "teacher" / "last.pt").is_file()
    render = exp / "render"
    assert (render / "audit_meta.json").is_file()
    stock = sorted((render / "stock_xvc" / "wavs").glob("*.wav"))
    candidate = sorted((render / "persona_codec_teacher" / "wavs").glob("*.wav"))
    assert len(stock) == len(candidate) == 2
    assert [p.name for p in stock] == [p.name for p in candidate]
