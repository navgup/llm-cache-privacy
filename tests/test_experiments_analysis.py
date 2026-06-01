"""Tests for Phase-4 background robustness and Exp-2 eviction analysis (no GPU)."""

from __future__ import annotations

import numpy as np

from kvleak.config import ExperimentConfig, load_config
from kvleak.analysis import background_robustness as bg
from kvleak.analysis import eviction as ev


def test_config_loads_background_and_sweep_sections():
    cfg = load_config("configs/baseline.yaml")
    assert cfg.background.rates == [0.5, 1.0, 2.0]
    assert cfg.sweep.fixed_length == 512
    assert "lru" in cfg.sweep.eviction_policies
    assert cfg.sweep.volume_ladder[0] == 0


def _bg_records(rate: float, sep: float, n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        hit = float(rng.normal(120, 5))          # hit slower regime (short prefix)
        miss = float(rng.normal(120 - sep, 5))   # sep>0 -> miss faster
        for label, ms in (("hit", hit), ("miss", miss)):
            recs.append({"prefix_id": f"p{i}", "bucket": "short", "n_tokens": 100,
                         "label": label, "ttft_ms": ms, "rate": rate})
    return recs


def test_background_aggregate_auc_and_percentiles():
    recs = _bg_records(0.5, sep=40) + _bg_records(2.0, sep=40, seed=1)
    rows = bg.aggregate(recs)
    assert [r["rate"] for r in rows] == [0.5, 2.0]
    for r in rows:
        assert r["auc"] > 0.95           # separable regardless of polarity
        assert r["direction"] == "hit_slower"
        assert r["probe_p99_ms"] >= r["probe_p50_ms"]


def test_attack_window_interpolation():
    curve = [
        {"volume": 0, "hit_rate": 1.0, "n": 6},
        {"volume": 4, "hit_rate": 1.0, "n": 6},
        {"volume": 8, "hit_rate": 0.8, "n": 6},
        {"volume": 16, "hit_rate": 0.4, "n": 6},
        {"volume": 32, "hit_rate": 0.0, "n": 6},
    ]
    assert abs(ev.attack_window(curve) - 14.0) < 1e-6


def test_attack_window_never_evicted_and_immediate():
    never = [{"volume": v, "hit_rate": 1.0, "n": 6} for v in (0, 4, 8)]
    assert ev.attack_window(never) == 8.0  # last volume in range
    immediate = [{"volume": 0, "hit_rate": 0.2, "n": 6}]
    assert ev.attack_window(immediate) == 0.0


def _ev_records(config, policy, mtt, sweeps, evict_at):
    """cached=True while volume < evict_at, else False."""
    recs = []
    for N in (0, 4, 8, 16, 32, 64):
        for v in range(6):
            recs.append({"config": config, "policy": policy, "max_total_tokens": mtt,
                         "server_max_total_num_tokens": mtt, "sweeps": sweeps,
                         "volume": N, "victim": f"ctx-{v}", "n_tokens": 512,
                         "probe_ttft_ms": 124.0 if N < evict_at else 187.0,
                         "cached": N < evict_at})
    return recs


def test_eviction_summarize_smaller_cache_shorter_window():
    recs = (
        _ev_records("lru@2048tok", "lru", 2048, ["cache_size"], evict_at=8)
        + _ev_records("lru@8192tok", "lru", 8192, ["cache_size"], evict_at=64)
    )
    summary = ev.summarize(recs)
    w_small = summary["lru@2048tok"]["attack_window_prompts"]
    w_big = summary["lru@8192tok"]["attack_window_prompts"]
    assert w_small < w_big  # smaller cache -> shorter attack window
