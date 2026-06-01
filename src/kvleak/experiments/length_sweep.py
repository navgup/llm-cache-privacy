"""Prefix-length crossover sweep.

For each long-context prefix truncated to each ladder length, measure a matched
cache-miss / cache-hit TTFT pair (flush -> cold -> warm -> flush), exactly as in
the baseline. Aggregating ΔTTFT(length) locates where the cache-hit timing
signal flips sign (hit-slower -> hit-faster) as the cold prefill cost grows.

Run on the VM after launching the server::

    uv run python -m kvleak.experiments.length_sweep --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import time

from tqdm import tqdm

from ..client import ProbeClient
from ..config import ExperimentConfig, load_config
from ..server import SGLangServer


def load_prefixes(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_length_sweep(cfg: ExperimentConfig, *, manage_server: bool = False):
    prefixes = load_prefixes(cfg.length_prefixes_file)
    if not prefixes:
        raise RuntimeError(
            f"no length-prefixes in {cfg.length_prefixes_file}; "
            "run kvleak.data.long_prefixes first"
        )

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "length_sweep_raw.jsonl"

    server = SGLangServer(cfg.server) if manage_server else None
    if server is not None:
        server.start()
    client = ProbeClient(cfg.server.base_url)
    ctrl = server or SGLangServer(cfg.server)

    records: list[dict] = []
    try:
        client.warmup(n=2)
        for row in tqdm(prefixes, desc="length-sweep"):
            text = row["text"]
            for trial in range(cfg.length_sweep.repeats):
                ctrl.flush_cache()
                miss_ms = client.ttft(text)  # cold prefill of `length` tokens
                hit_ms = client.ttft(text)  # warm (1-token extend)
                for label, ttft_ms in (("miss", miss_ms), ("hit", hit_ms)):
                    records.append(
                        {
                            "base_id": row["base_id"],
                            "length": row["length"],
                            "n_tokens": row["n_tokens"],
                            "label": label,
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
    ap = argparse.ArgumentParser(description="Run prefix-length crossover sweep")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--manage-server", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_length_sweep(cfg, manage_server=args.manage_server)


if __name__ == "__main__":
    main()
