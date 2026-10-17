#!/usr/bin/env bash
# Start the Next.js UI of the Global-MMLU evaluation tool in development mode.
#
# Shell: bash (macOS, Linux, WSL2).
# Prerequisites: Node.js 20+ and npm; dependencies installed with `npm install`
#   in this folder; the backend running (see backend/start.sh).
# Inputs (environment variables, all optional):
#   PORT  port of the UI (default: 3000). If you change it, add the new origin
#         to CORS_ORIGINS in the root .env.
# Side effects: runs a local HTTP server in the foreground; stop it with Ctrl+C.
# Usage: frontend/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-3000}"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm not found. Install Node.js 20 or later." >&2
  exit 1
fi
if [[ ! -d "${SCRIPT_DIR}/node_modules" ]]; then
  echo "Error: dependencies missing. Run: npm --prefix ${SCRIPT_DIR} install" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/.env.local" ]]; then
  echo "Warning: no frontend/.env.local; the UI uses http://localhost:8000 as backend." >&2
fi

cd "${SCRIPT_DIR}"
echo "UI on http://localhost:${PORT}. Stop with Ctrl+C."
exec npm run dev -- --port "${PORT}"
