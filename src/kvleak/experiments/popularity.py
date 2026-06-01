"""Experiment 2b: how cache popularity structure interacts with eviction policy.

The main sweep used all-distinct background (every entry frequency 1), which makes
LFU degenerate to LRU. Here the background has a popularity skew, so the policies
can diverge. Two measurements, per policy (lru, lfu), at a fixed cache size:

  (a) Skewed-eviction window — a RARE victim (frequency 1) under skewed background
      (a small hot pool recurs, building frequency). Prediction: under LFU the hot
      set is sticky and crowds out the rare victim, so its attack window is SHORTER
      than under LRU.

  (b) Popularity oracle — probe CANDIDATE prefixes of known frequency once each,
      after a recency-EQUALIZED access schedule (so all candidates share the same
      last-access time; only their frequency differs) plus cold filler for pressure.
      Prediction: under LFU, P(cached) increases with frequency (a cross-user
      popularity oracle); under LRU it does not (recency is equalized).

Probing stays single-shot (one probe per victim/candidate) — that's the correct
detection model; the divergence comes from the background, not the probing.

    uv run python -m kvleak.experiments.popularity --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import replace

from tqdm import tqdm

from ..background import (
    PromptSource,
    inject_skewed,
    inject_texts,
    load_background,
)
from ..client import ProbeClient
from ..config import ExperimentConfig, ServerConfig, load_config
from ..server import SGLangServer
from .eviction_sweep import _calibrate


def _length_prefixes(cfg: ExperimentConfig) -> list[dict]:
    rows = [json.loads(l) for l in open(cfg.length_prefixes_file) if l.strip()]
    return [r for r in rows if r["length"] == cfg.popularity.fixed_length]


def _oracle_schedule(candidates: list[dict], rng: random.Random) -> list[str]:
    """Recency-equalized access schedule: build frequency in an early phase, then
    one final round touching every candidate (equal last-access)."""
    early: list[str] = []
    for c in candidates:
        early += [c["text"]] * (c["freq"] - 1)  # f-1 extra accesses
    rng.shuffle(early)
    final_round = [c["text"] for c in candidates]  # one each, most-recent, equal
    return early + final_round


def run_popularity(cfg: ExperimentConfig) -> str:
    pc = cfg.popularity
    prefixes = _length_prefixes(cfg)
    n_cand = len(pc.candidate_freqs) * pc.candidates_per_freq
    if len(prefixes) < n_cand + pc.n_victims:
        raise RuntimeError(
            f"need {n_cand + pc.n_victims} length-{pc.fixed_length} prefixes, "
            f"have {len(prefixes)}; raise n_base_texts in length_sweep or lower counts"
        )
    candidates = prefixes[:n_cand]
    freqs = [f for f in pc.candidate_freqs for _ in range(pc.candidates_per_freq)]
    for c, f in zip(candidates, freqs):
        c["freq"] = f
    victims = prefixes[n_cand : n_cand + pc.n_victims]

    background = load_background(cfg.lmsys_background_file)
    if not background:
        raise RuntimeError("no background corpus; run kvleak.data.lmsys first")
    hot_pool = [b["text"] for b in background[: pc.hot_pool_size]]
    cold_all = background[pc.hot_pool_size :]

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "popularity_raw.jsonl"
    open(out_path, "w").close()

    for policy in pc.policies:
        server_cfg: ServerConfig = replace(
            cfg.server, max_total_tokens=pc.cache_tokens,
            radix_eviction_policy=policy, disable_radix_cache=False,
        )
        subprocess.run(["docker", "rm", "-f", server_cfg.container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n=== policy={policy} @ {pc.cache_tokens}tok — starting server ===",
              flush=True)
        server = SGLangServer(server_cfg)
        server.start()
        client = ProbeClient(server_cfg.base_url)
        rng = random.Random(cfg.seed)
        n = 0
        out_f = open(out_path, "a")  # checkpoint each record immediately

        def emit(rec: dict) -> None:
            nonlocal n
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n += 1

        try:
            client.warmup(n=2)
            calib = _calibrate(client, server, victims)
            thr = calib["threshold_ms"]
            print(f"  calib: warm={calib['warm_ms']:.1f} cold={calib['cold_ms']:.1f} "
                  f"thr={thr:.1f}ms", flush=True)

            # --- (a) skewed-eviction window for a rare victim ---
            cold_a = PromptSource(cold_all, seed=cfg.seed)
            for N in tqdm(pc.volume_ladder, desc=f"{policy} evict"):
                for v in victims:
                    server.flush_cache()
                    client.ttft(v["text"])  # victim cached (frequency 1)
                    inject_skewed(client, cold_a, hot_pool, N, pc.p_hot, rng)
                    probe = client.ttft(v["text"])
                    emit({"part": "evict", "policy": policy, "volume": N,
                          "victim": v["base_id"], "probe_ttft_ms": probe,
                          "cached": bool(probe <= thr)})
                server.flush_cache()

            # --- (b) popularity oracle ---
            # Rebuild the full popularity state fresh for EACH target and probe
            # only that one. Probing all candidates in a single cache state would
            # cascade: a miss re-caches the probed prefix and evicts the next
            # candidate, so every probe reads cold.
            cold_b = PromptSource(cold_all, seed=cfg.seed + 1)
            for target in tqdm(candidates, desc=f"{policy} oracle"):
                server.flush_cache()
                inject_texts(client, _oracle_schedule(candidates, rng))  # build freq
                inject_texts(client, cold_b.next_batch(pc.oracle_filler))  # pressure
                probe = client.ttft(target["text"])  # probe exactly one target
                emit({"part": "oracle", "policy": policy,
                      "candidate": target["base_id"], "freq": target["freq"],
                      "probe_ttft_ms": probe, "cached": bool(probe <= thr)})
                server.flush_cache()
            print(f"  policy={policy} done ({n} measurements)", flush=True)
        finally:
            out_f.close()
            client.close()
            server.stop()

    print(f"\nWrote popularity results -> {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2b: popularity vs eviction policy")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    run_popularity(load_config(args.config))


if __name__ == "__main__":
    main()
