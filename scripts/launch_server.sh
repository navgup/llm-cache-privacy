#!/usr/bin/env bash
# Launch the SGLang server on the L4 VM using the args derived from a config.
# Usage: bash scripts/launch_server.sh [configs/baseline.yaml]
set -euo pipefail

CONFIG="${1:-configs/baseline.yaml}"
cd "$(dirname "$0")/.."

# Load HF_TOKEN from .env if present (gated model weights / tokenizer).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

# Print the exact launch command the manager would build, then exec it.
# (kvleak.server.build_launch_args is the single source of truth for flags.)
exec uv run --extra gpu python - "$CONFIG" <<'PY'
import os
import sys

from kvleak.config import load_config
from kvleak.server import build_launch_args

cfg = load_config(sys.argv[1])
args = build_launch_args(cfg.server)
print("Launching:", " ".join(args), flush=True)
os.execvp(args[0], args)
PY
