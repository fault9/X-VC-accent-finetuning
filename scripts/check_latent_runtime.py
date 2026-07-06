#!/usr/bin/env python3
"""Fast container-side checks for X-VC latent alignment tensor sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.codec.sac.model import XVC


def main() -> int:
    features = torch.tensor([[[0.0, 10.0, 20.0]]])
    positions = torch.tensor([[0.0, 0.25, 0.5, 1.0]])

    linear = XVC._sample_bct_at_normalized_positions(features, positions)
    expected_linear = torch.tensor([[[0.0, 5.0, 10.0, 20.0]]])
    if not torch.allclose(linear, expected_linear):
        raise SystemExit(f"linear latent sampler failed: {linear}")

    nearest = XVC._sample_bct_nearest(features, positions)
    expected_nearest = torch.tensor([[[0.0, 0.0, 10.0, 20.0]]])
    if not torch.equal(nearest, expected_nearest):
        raise SystemExit(f"nearest latent sampler failed: {nearest}")

    print("PASS: latent linear and quantized-nearest samplers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
