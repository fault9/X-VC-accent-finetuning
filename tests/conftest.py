"""Shared test setup.

Puts the repository root on sys.path so `models.*` / `utils.*` / `xvc.*`
import the same way they do for `bins/train.py` run from the repo root.
Once the package is pip-installed (`pip install -e .`), this becomes a no-op.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
