#!/usr/bin/env bash
# End-to-end: Phase 4 (background traffic) + Experiment 2 (eviction/cache-size sweep).
# Designed to run inside a tmux session on the VM; poll /tmp/run.log for progress.
# The sweep checkpoints per config, so a preemption keeps completed configs.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && source .env; set +a
CONFIG="${1:-configs/baseline.yaml}"

echo "=== [1/6] data prep (LMSYS background + length prefixes) ==="
[ -s data/processed/lmsys_background.jsonl ] || \
  uv run python -m kvleak.data.lmsys --config "$CONFIG"
[ -s data/processed/length_prefixes.jsonl ] || \
  uv run python -m kvleak.data.long_prefixes --config "$CONFIG"

echo "=== [2/6] launch default server for Phase 4 ==="
docker rm -f sglang-kvleak 2>/dev/null || true
bash scripts/launch_server.sh "$CONFIG" > /tmp/server.log 2>&1 &
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "server healthy after ~$((i*4))s"; break; }
  sleep 4
done

echo "=== [3/6] Phase 4: background baseline (0.5x/1x/2x) ==="
uv run python -m kvleak.experiments.background_baseline --config "$CONFIG"

echo "=== [4/6] stop default server (the sweep manages its own containers) ==="
docker stop sglang-kvleak 2>/dev/null || true
sleep 3

echo "=== [5/6] Experiment 2: eviction-policy + cache-size sweep ==="
uv run python -m kvleak.experiments.eviction_sweep --config "$CONFIG"

echo "=== [6/6] analyses ==="
uv run python -m kvleak.analysis.background_robustness --config "$CONFIG" || true
uv run python -m kvleak.analysis.eviction --config "$CONFIG" || true
echo "=== ALL DONE ==="
