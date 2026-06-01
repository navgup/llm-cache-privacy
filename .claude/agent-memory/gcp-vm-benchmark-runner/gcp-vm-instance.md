---
name: gcp-vm-instance
description: The current GCP VM instance, how to reach it, and its specs
metadata:
  type: project
---

All experiments run on GCP instance `instance-clone-v2` (zone `us-central1-a`, project `cs349d-496005`). Connect: `gcloud compute ssh instance-clone-v2 --zone=us-central1-a --command '<cmd>'`.

**Why:** The user provisioned this on 2026-05-31 to replace a deprecated older instance. Do NOT use any old instance (`instance-20260511-*` or the original `instance-clone`).

**How to apply:** Specs — single NVIDIA L4 (~23 GB usable), driver 580.126.20, Ubuntu/Python 3.10.12. ~97 GB disk, ~44 GB free after cleaning the failed pip-build artifacts. Repo synced to `~/cs281-final`. No git remote → sync by `gcloud compute scp --recurse`. Verify whether it is a SPOT/preemptible instance and use tmux for long runs accordingly. Deployment is Docker — see [[deployment-docker]].
