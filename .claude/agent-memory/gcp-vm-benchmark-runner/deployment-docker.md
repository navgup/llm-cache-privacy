---
name: deployment-docker
description: SGLang must run via Docker on the L4, never pip/uv — and why
metadata:
  type: feedback
---

Run SGLang via the official Docker image `lmsysorg/sglang:v0.5.3-cu129`, launched by `scripts/launch_server.sh`. Never `pip install sglang` / `uv sync --extra gpu` (that extra was removed from pyproject on 2026-05-31).

**Why:** A from-source/pip install spirals on this VM — the L4 is Ada/sm89, but recent SGLang/flashinfer prebuilt kernels target sm90/sm100, forcing a slow source build that also created `~/.local` (pip --user) conflicts. The Docker image ships kernels that work on sm89, sidestepping both. The host driver (580) supports the CUDA 12.9 image.

**How to apply:** On a fresh VM run `scripts/setup_vm.sh` (installs Docker + nvidia runtime, pulls the image, `uv sync` for the slim client env). The `kvleak` client/data/analysis is pure Python (no torch) and talks to the container over HTTP at 127.0.0.1:30000. Bind-mount the host HF cache (`~/.cache/huggingface`) so the ~16 GB Llama-3.1-8B weights persist across container restarts. Disk is tight (~44 GB free after cleanup) — image + weights ≈ 36 GB. See [[gcp-vm-instance]].
