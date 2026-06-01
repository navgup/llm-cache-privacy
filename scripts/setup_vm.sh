#!/usr/bin/env bash
# One-time setup on the GPU VM (NVIDIA L4): install Docker + the NVIDIA container
# runtime, pull the pinned SGLang image, and create a slim uv env for the attack
# client (base deps only — SGLang itself runs in Docker, not in this env).
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="lmsysorg/sglang:v0.5.3-cu129"

# 1. Docker engine.
if ! command -v docker >/dev/null 2>&1; then
  echo "=== installing Docker ==="
  curl -fsSL https://get.docker.com | sudo sh
fi

# 2. NVIDIA container runtime (the toolkit binary is already on the VM image).
echo "=== configuring NVIDIA container runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker || echo "nvidia-ctk configure skipped"
sudo systemctl restart docker

# 3. Let this user run docker without sudo (takes effect on next login).
sudo usermod -aG docker "$USER" || true

# 4. Pull the pinned image (sudo works regardless of group membership timing).
echo "=== pulling $IMAGE (large; ~minutes) ==="
sudo docker pull "$IMAGE"

# 5. GPU smoke test through the container.
echo "=== GPU smoke test ==="
sudo docker run --rm --gpus all "$IMAGE" \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 6. uv + slim client env (no GPU/torch — just httpx/datasets/transformers/sklearn).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

echo
echo "Setup complete."
echo "NOTE: log out and back in (or run 'newgrp docker') so 'docker' works without sudo."
echo "Then launch the server with: bash scripts/launch_server.sh"
