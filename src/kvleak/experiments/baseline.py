"""Experiment 1: baseline attack AUC.

For each MedQA probe we collect a matched cache-miss / cache-hit TTFT pair:

  1. flush_cache()           -> clean radix tree
  2. ttft(prefix)            -> MISS sample (cold prefill)
  3. ttft(prefix)            -> HIT  sample (step 2 cached it; the attacker's hit)
  4. flush_cache()           -> prevent contamination of the next probe

This yields both TTFT distributions for AUC while honoring the report's
single-informative-probe model (the attacker's real-world hit is exactly the
step-3 measurement). All probing is strictly sequential.

Run on the VM after launching the server::

    uv run python -m kvleak.experiments.baseline --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from ..client import ProbeClient
from ..config import ExperimentConfig, load_config
from ..server import SGLangServer


def load_probes(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_baseline(
    cfg: ExperimentConfig,
    *,
    manage_server: bool = False,
    repeats: int = 1,
) -> Path:
    """Run the baseline protocol and write raw measurements.

    Args:
        cfg: experiment config.
        manage_server: if True, launch/stop SGLang here; otherwise assume an
            already-running server at ``cfg.server.base_url``.
        repeats: independent miss/hit pairs per probe (>1 for variance estimates).

    Returns the path to the raw measurements JSONL.
    """
    probes = load_probes(cfg.medqa_probes_file)
    if not probes:
        raise RuntimeError(
            f"no probes found in {cfg.medqa_probes_file}; run kvleak.data.medqa first"
        )

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "baseline_raw.jsonl"

    server = SGLangServer(cfg.server) if manage_server else None
    if server is not None:
        server.start()

    client = ProbeClient(cfg.server.base_url)
    # The flush_cache control lives on the server manager; build a lightweight
    # one purely for the control endpoints when we don't own the process.
    ctrl = server or SGLangServer(cfg.server)

    records: list[dict] = []
    try:
        client.warmup(n=2)
        for probe in tqdm(probes, desc="probes"):
            text = probe["text"]
            for trial in range(repeats):
                ctrl.flush_cache()
                miss_ms = client.ttft(text)  # cold
                hit_ms = client.ttft(text)  # warm (cached by the cold probe)
                for label, ttft_ms in (("miss", miss_ms), ("hit", hit_ms)):
                    records.append(
                        {
                            "prefix_id": probe["id"],
                            "bucket": probe["bucket"],
                            "n_tokens": probe["n_tokens"],
                            "label": label,  # "hit" = positive class
                            "ttft_ms": ttft_ms,
                            "trial": trial,
                            "ts": time.time(),
                        }
                    )
            ctrl.flush_cache()
    finally:
        client.close()
        if server is not None:
            server.stop()

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} measurements -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline attack (Experiment 1)")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument(
        "--manage-server",
        action="store_true",
        help="launch and stop SGLang from this process (default: use a running server)",
    )
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_baseline(cfg, manage_server=args.manage_server, repeats=args.repeats)


if __name__ == "__main__":
    main()
