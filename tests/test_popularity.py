"""Tests for Experiment 2b (popularity) analysis + skewed injection (no GPU)."""

from __future__ import annotations

import random

from kvleak.config import load_config
from kvleak.analysis import popularity as pop
from kvleak.background import PromptSource, inject_skewed
from kvleak.experiments.popularity import _oracle_schedule


def test_config_loads_popularity_section():
    cfg = load_config("configs/baseline.yaml")
    assert cfg.popularity.policies == ["lru", "lfu"]
    assert cfg.popularity.candidate_freqs[0] == 1
    assert cfg.popularity.fixed_length == 512


def test_oracle_schedule_frequency_and_recency():
    cands = [{"text": "A", "freq": 1}, {"text": "B", "freq": 4}]
    sched = _oracle_schedule(cands, random.Random(0))
    # total accesses == sum of freqs; A once, B four times
    assert sched.count("A") == 1 and sched.count("B") == 4
    # recency equalized: the final round (last len(cands) entries) touches each once
    assert sorted(sched[-2:]) == ["A", "B"]


def test_inject_skewed_repeats_hot_pool():
    sent = []

    class FakeClient:
        def ttft(self, text):
            sent.append(text)
            return 1.0

    cold = PromptSource([{"text": f"cold{i}"} for i in range(100)], seed=0)
    hot = ["HOT0", "HOT1"]
    inject_skewed(FakeClient(), cold, hot, n=200, p_hot=0.9, rng=random.Random(0))
    assert len(sent) == 200
    hot_hits = sum(1 for t in sent if t.startswith("HOT"))
    assert hot_hits > 120  # ~90% from a 2-element pool -> heavy repetition
    # cold draws are distinct (no reuse within the run)
    cold_sent = [t for t in sent if t.startswith("cold")]
    assert len(set(cold_sent)) == len(cold_sent)


def test_summarize_oracle_and_evict():
    recs = []
    # evict: lfu shorter window than lru
    for pol, evict_at in (("lru", 64), ("lfu", 16)):
        for N in (0, 8, 16, 32, 64):
            for v in range(4):
                recs.append({"part": "evict", "policy": pol, "volume": N,
                             "victim": f"v{v}", "probe_ttft_ms": 1.0,
                             "cached": N < evict_at})
    # oracle: lfu increases with freq, lru flat at 0
    for fr in (1, 2, 4, 8, 16, 32):
        for v in range(2):
            recs.append({"part": "oracle", "policy": "lfu", "candidate": f"c{fr}_{v}",
                         "freq": fr, "probe_ttft_ms": 1.0, "cached": fr >= 8})
            recs.append({"part": "oracle", "policy": "lru", "candidate": f"d{fr}_{v}",
                         "freq": fr, "probe_ttft_ms": 1.0, "cached": False})
    s = pop.summarize(recs)
    assert s["evict"]["lfu"]["attack_window_prompts"] < s["evict"]["lru"]["attack_window_prompts"]
    lfu_pts = {p["freq"]: p["p_cached"] for p in s["oracle"]["lfu"]}
    assert lfu_pts[1] == 0.0 and lfu_pts[32] == 1.0   # increasing oracle
    lru_pts = {p["freq"]: p["p_cached"] for p in s["oracle"]["lru"]}
    assert all(v == 0.0 for v in lru_pts.values())     # flat (recency equalized)
