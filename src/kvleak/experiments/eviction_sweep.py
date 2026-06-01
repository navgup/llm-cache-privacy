"""Experiment 2: eviction-policy + cache-size attack-surface sweep.

At a fixed prefix length, measure the **attack window as background volume** — how
many distinct LMSYS background prompts must be injected after a victim prefix is
cached before it gets evicted (so an attacker probe no longer sees a hit). Cache
size (``--max-total-tokens``) and eviction policy (``--radix-eviction-policy``)
are exactly the knobs that control this, so we sweep both.

Per config we restart the SGLang container (these are launch-time flags),
calibrate a per-config hit/miss TTFT threshold, then for each background volume N
and each victim: flush -> cache victim -> inject N background prompts -> probe.
A probe classified as a hit means the victim survived N prompts of pressure.

Records are appended per-config so a preemption keeps completed configs.

Run on the VM (LMSYS background + length-prefixes prepared; docker available)::

    uv run python -m kvleak.experiments.eviction_sweep --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from ..background import PromptSource, inject, load_background
from ..client import ProbeClient
from ..config import ExperimentConfig, ServerConfig, load_config
from ..server import SGLangServer


def _victim_prefixes(cfg: ExperimentConfig) -> list[dict]:
    """Fixed-length victim prefixes from the length-prefix ladder."""
    rows = [json.loads(l) for l in open(cfg.length_prefixes_file) if l.strip()]
    victims = [r for r in rows if r["length"] == cfg.sweep.fixed_length]
    if not victims:
        raise RuntimeError(
            f"no length-{cfg.sweep.fixed_length} prefixes in "
            f"{cfg.length_prefixes_file}; run kvleak.data.long_prefixes "
            f"(and include {cfg.sweep.fixed_length} in length_sweep.lengths)"
        )
    return victims[: cfg.sweep.n_victims]


def _build_configs(cfg: ExperimentConfig) -> list[dict]:
    """Server configs to sweep, deduped. Each tagged with the sweep it serves."""
    s = cfg.sweep
    configs: dict[tuple, dict] = {}
    for tok in s.cache_sizes:
        key = (s.size_policy, tok)
        configs.setdefault(key, {"policy": s.size_policy, "max_total_tokens": tok,
                                 "sweeps": []})["sweeps"].append("cache_size")
    for pol in s.eviction_policies:
        key = (pol, s.policy_cache_tokens)
        configs.setdefault(key, {"policy": pol, "max_total_tokens": s.policy_cache_tokens,
                                 "sweeps": []})["sweeps"].append("eviction_policy")
    return list(configs.values())


def _calibrate(client: ProbeClient, ctrl: SGLangServer, victims: list[dict]) -> dict:
    """Per-config hit/miss TTFT references + threshold (length>crossover: hit faster)."""
    warm, cold = [], []
    for v in victims[: min(4, len(victims))]:
        ctrl.flush_cache()
        cold.append(client.ttft(v["text"]))  # cold (just flushed)
        warm.append(client.ttft(v["text"]))  # warm (cached by the line above)
    ctrl.flush_cache()
    warm_med, cold_med = float(np.median(warm)), float(np.median(cold))
    return {"warm_ms": warm_med, "cold_ms": cold_med,
            "threshold_ms": (warm_med + cold_med) / 2.0}


def run_sweep(cfg: ExperimentConfig) -> str:
    victims = _victim_prefixes(cfg)
    background = load_background(cfg.lmsys_background_file)
    if not background:
        raise RuntimeError("no LMSYS background; run kvleak.data.lmsys first")
    server_configs = _build_configs(cfg)

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "eviction_sweep_raw.jsonl"
    open(out_path, "w").close()  # truncate; we append per config
    calib_path = cfg.results_path / "eviction_sweep_calib.json"
    calib_all: list[dict] = []

    for sc in server_configs:
        server_cfg: ServerConfig = replace(
            cfg.server,
            max_total_tokens=sc["max_total_tokens"],
            radix_eviction_policy=sc["policy"],
            disable_radix_cache=False,
        )
        # Clear any leftover container with our name before launching.
        subprocess.run(["docker", "rm", "-f", server_cfg.container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tag = f"{sc['policy']}@{sc['max_total_tokens']}tok"
        print(f"\n=== config {tag} (sweeps={sc['sweeps']}) — starting server ===",
              flush=True)
        server = SGLangServer(server_cfg)
        server.start()
        client = ProbeClient(server_cfg.base_url)
        try:
            info = server.server_info()
            max_tokens = info.get("max_total_num_tokens")
            client.warmup(n=2)
            calib = _calibrate(client, server, victims)
            calib.update({"config": tag, "policy": sc["policy"],
                          "max_total_tokens": sc["max_total_tokens"],
                          "server_max_total_num_tokens": max_tokens,
                          "sweeps": sc["sweeps"]})
            calib_all.append(calib)
            print(f"  calib: warm={calib['warm_ms']:.1f}ms cold={calib['cold_ms']:.1f}ms "
                  f"thr={calib['threshold_ms']:.1f}ms  server_max_tokens={max_tokens}",
                  flush=True)

            source = PromptSource(background, seed=cfg.seed)
            batch: list[dict] = []
            for N in tqdm(cfg.sweep.volume_ladder, desc=tag):
                for v in victims:
                    server.flush_cache()
                    client.ttft(v["text"])           # victim request (caches it)
                    inject(client, source, N)         # background eviction pressure
                    probe_ms = client.ttft(v["text"])  # attacker probe
                    cached = probe_ms <= calib["threshold_ms"]
                    batch.append(
                        {
                            "config": tag,
                            "policy": sc["policy"],
                            "max_total_tokens": sc["max_total_tokens"],
                            "server_max_total_num_tokens": max_tokens,
                            "sweeps": sc["sweeps"],
                            "volume": N,
                            "victim": v["base_id"],
                            "n_tokens": v["n_tokens"],
                            "probe_ttft_ms": probe_ms,
                            "cached": bool(cached),
                        }
                    )
                server.flush_cache()
            with open(out_path, "a") as f:  # checkpoint this config
                for rec in batch:
                    f.write(json.dumps(rec) + "\n")
            print(f"  config {tag} done ({len(batch)} measurements appended)",
                  flush=True)
        finally:
            client.close()
            server.stop()

    with open(calib_path, "w") as f:
        json.dump(calib_all, f, indent=2)
    print(f"\nWrote sweep results -> {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2: eviction/cache-size sweep")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
