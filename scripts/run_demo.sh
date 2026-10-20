#!/usr/bin/env bash
# Start the single-worker demo server.
#
# On the host:
#   ./scripts/run_demo.sh
# On a laptop (secure-context localhost for the microphone):
#   ssh -N -L 8000:127.0.0.1:8000 tripathiji@<host>
#   open http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:$PWD"
if [ -d "$HOME/.cache/huggingface/speechbrain/spkrec-ecapa-voxceleb" ]; then
  export HF_HUB_OFFLINE=1
else
  echo "WARNING: ECAPA cache missing; the first run will download it from Hugging Face." >&2
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
echo "DISENT Voice dashboard: http://${HOST}:${PORT}"
echo "Use SSH port forwarding for microphone access from another device."
exec .venv/bin/python -m uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --log-level info
