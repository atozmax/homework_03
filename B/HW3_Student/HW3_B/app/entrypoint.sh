#!/usr/bin/env bash
# entrypoint.sh — runtime contract for the HW3_B container.
# Sets the env vars that test_env_contract.py requires, then execs uvicorn.

set -euo pipefail

# 1. Threading (pinned to container CPU limit; uvicorn worker will inherit)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# 2. Tokenizers fork-parallelism OFF (prevents deadlocks in uvicorn workers)
export TOKENIZERS_PARALLELISM=false

# 3. Device (default cpu; can be overridden to "cuda" in HW3_C Deployment)
export BUNDLE_DEVICE="${BUNDLE_DEVICE:-cpu}"

# 4. Hashing determinism
export PYTHONHASHSEED=0

# 5. Quick log
echo "[entrypoint] BUNDLE_DIR=${BUNDLE_DIR:-/app/bundle}"
echo "[entrypoint] BUNDLE_DEVICE=${BUNDLE_DEVICE}"
echo "[entrypoint] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[entrypoint] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[entrypoint] TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"

# 6. SIGTERM handler — uvicorn already handles this, but log it
trap 'echo "[entrypoint] SIGTERM received, draining…"; kill -TERM "$PID" 2>/dev/null || true; wait "$PID" || true' TERM INT

# 7. Start uvicorn (exec so PID 1 is the python process; signals work)
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${UVICORN_WORKERS:-1}" \
  --log-level "${LOG_LEVEL:-info}"
