"""Pytest path setup: tests import both `src.*` and `orchestrator.src.*`."""

import sys
from pathlib import Path

_ORCHESTRATOR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ORCHESTRATOR_DIR.parent

for p in (str(_ORCHESTRATOR_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
