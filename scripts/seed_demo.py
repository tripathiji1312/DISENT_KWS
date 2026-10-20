#!/usr/bin/env python3
"""Seed the demo registry from demo_assets/manifest.json.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/seed_demo.py --data-dir data
    PYTHONPATH=src .venv/bin/python scripts/seed_demo.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.demo import seed_demo  # noqa: E402
from app.inference import Engine  # noqa: E402
from app.registry import Registry  # noqa: E402
from app.settings import DEMO_ASSETS_DIR, MODEL_PATH  # noqa: E402
from app.audio import read_wav_bytes, to_16k_mono  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--assets", type=Path, default=DEMO_ASSETS_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--force", action="store_true", help="re-enroll existing demo names")
    args = parser.parse_args()

    if (Path.home() / ".cache" / "huggingface" / "speechbrain" / "spkrec-ecapa-voxceleb").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    registry = Registry(args.data_dir)
    engine = Engine(model_path=args.model)
    result = seed_demo(engine, registry, assets_dir=args.assets, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
