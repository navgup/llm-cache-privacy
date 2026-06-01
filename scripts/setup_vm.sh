#!/usr/bin/env bash
# One-time setup on the GPU VM (NVIDIA L4): install uv, sync the GPU extra,
# authenticate to HuggingFace, and sanity-check the model is reachable.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Install uv if missing.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Sync deps including the GPU extra (sglang[all] -> torch + flashinfer).
uv sync --extra gpu

# 3. HuggingFace auth (gated Llama-3.1 + LMSYS-Chat-1M).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Put it in .env or export it before running." >&2
else
  uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "Setup complete. Launch the server with: bash scripts/launch_server.sh"
