"""DISENT multi-person voice assistant package."""

from __future__ import annotations

import sys
from pathlib import Path

# The model code lives in src/ and imports `config`, `models`, etc. as
# top-level modules. Make that work regardless of how uvicorn is launched.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
