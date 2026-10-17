#!/usr/bin/env bash
# Start the FastAPI backend of the Global-MMLU evaluation tool.
#
# Shell: bash (macOS, Linux, WSL2).
# Prerequisites: Conda with the `global-mmlu` environment (see README), and a
#   root `.env` with the OCI settings.
# Inputs (environment variables, all optional):
#   CONDA_ENV  Conda environment name (default: global-mmlu)
#   HOST       bind address (default: 127.0.0.1)
#   PORT       port (default: 8000)
# Extra arguments are passed to uvicorn, for example `--reload`.
# Side effects: runs a local HTTP server in the foreground; stop it with Ctrl+C.
# Usage: backend/start.sh [--reload]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-global-mmlu}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# CONDA_EXE is set by `conda init`; fall back to conda on the PATH.
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "Error: conda not found. Install Conda or run 'conda init'." >&2
  exit 1
fi
if ! "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import uvicorn" >/dev/null 2>&1; then
  echo "Error: Conda env '${CONDA_ENV}' is missing or lacks the dependencies." >&2
  echo "Create it with: conda create -n ${CONDA_ENV} python=3.12 && conda run -n ${CONDA_ENV} pip install -r requirements-dev.txt" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/../.env" ]]; then
  echo "Warning: no .env in the project root; defaults apply (copy .env.example)." >&2
fi

echo "Backend on http://${HOST}:${PORT} (Conda env ${CONDA_ENV}). Stop with Ctrl+C."
exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  uvicorn app.main:app --app-dir "${SCRIPT_DIR}" --host "${HOST}" --port "${PORT}" "$@"
