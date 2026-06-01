#!/usr/bin/env bash
# Launch the SGLang server in Docker on the L4 VM.
# Usage: bash scripts/launch_server.sh [configs/baseline.yaml]
#
# The full `docker run ...` argv is built by kvleak.server.build_docker_command
# (single source of truth for flags). HF_TOKEN is passed through from the env
# (loaded from .env) so it never appears in the command line.
set -euo pipefail

CONFIG="${1:-configs/baseline.yaml}"
cd "$(dirname "$0")/.."

# uv lives in ~/.local/bin, which a non-login shell (e.g. tmux) may not have on PATH.
export PATH="$HOME/.local/bin:$PATH"

# Load HF_TOKEN (gated Llama-3.1 weights + tokenizer).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
export HF_TOKEN="${HF_TOKEN:-}"

# Use rootless docker if the user is in the docker group; else sudo -E (preserves
# HF_TOKEN). `docker info` succeeds without sudo only when group membership is active.
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo -E docker)
fi

# Ensure the HF cache dir exists on the host so the bind mount works.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Build the command (kvleak prints the argv starting at "docker"; we swap in the
# detected docker invocation, sudo or not).
mapfile -t CMD < <(uv run python - "$CONFIG" <<'PY'
import sys
from kvleak.config import load_config
from kvleak.server import build_docker_command
cfg = load_config(sys.argv[1])
for tok in build_docker_command(cfg.server):
    print(tok)
PY
)

# CMD[0] is "docker"; replace it with the detected invocation.
if [[ ${#CMD[@]} -lt 2 || "${CMD[0]}" != "docker" ]]; then
  echo "ERROR: failed to build docker command (is uv on PATH? does 'uv run' work?)." >&2
  exit 1
fi
echo "Launching: ${DOCKER[*]} ${CMD[*]:1}"
exec "${DOCKER[@]}" "${CMD[@]:1}"
