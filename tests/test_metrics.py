"""Metric sanity checks on synthetic data (no GPU/server needed).

Covers BOTH signal polarities: a hit may be faster (lower TTFT) on large/slow
setups, or slower (higher TTFT) for short prefixes on a fast GPU. The analysis
must report a high AUC either way (separability is sign-agnostic).
"""

from __future__ import annotations

import numpy as np

from kvleak.config import ExperimentConfig
from kvleak.analysis.metrics import (
    auc,
    bits_leaked,
    compute_metrics,
    detect_direction,
    oriented_score,
    precision_at_recall,
    separability_auc,
    split_by_prefix,
    youden_threshold,
)


def _oriented(sep: float, n: int = 500, seed: int = 0):
    """(score, y) where positives (y=1) have HIGHER score by ~sep."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(50.0 + sep, 5.0, n)
    neg = rng.normal(50.0, 5.0, n)
    score = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    return score, y


# --- low-level metrics operate on an oriented score (higher = more hit-like) ---

def test_auc_well_separated():
    score, y = _oriented(sep=40.0)
    assert auc(score, y) > 0.95


def test_auc_overlapping():
    score, y = _oriented(sep=0.0)
    assert abs(auc(score, y) - 0.5) < 0.1


def test_youden_threshold_between_means():
    score, y = _oriented(sep=40.0)
    thr = youden_threshold(score, y)
    assert 50.0 < thr < 90.0


def test_precision_at_recall_high_when_separable():
    score, y = _oriented(sep=40.0)
    prec, thr = precision_at_recall(score, y, target_recall=0.8)
    assert prec > 0.9


def test_bits_leaked_bounds():
    score, y = _oriented(sep=40.0)
    thr = youden_threshold(score, y)
    bits = bits_leaked(score, y, thr)
    assert 0.0 <= bits <= 1.0 + 1e-9  # MI of a binary channel <= 1 bit
    assert bits > 0.7


def test_bits_leaked_near_zero_when_random():
    score, y = _oriented(sep=0.0)
    thr = youden_threshold(score, y)
    assert bits_leaked(score, y, thr) < 0.1


def test_split_by_prefix_disjoint_and_grouped():
    prefix_ids = [f"p{i}" for i in range(100) for _ in range(2)]
    calib, evalset = split_by_prefix(prefix_ids, calib_frac=0.2, seed=1)
    assert calib.isdisjoint(evalset)
    assert len(calib) == 20
    assert len(calib) + len(evalset) == 100


# --- direction detection + separability are sign-agnostic ---

def test_detect_direction():
    rng = np.random.default_rng(0)
    hit = rng.normal(50, 5, 200)
    miss = rng.normal(90, 5, 200)
    ttft = np.concatenate([hit, miss])
    y = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    assert detect_direction(ttft, y) == -1  # hit faster (lower ttft)
    assert detect_direction(ttft, 1 - y) == 1  # flip labels -> hit slower


def test_separability_invariant_to_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 5, 200)
    b = rng.normal(90, 5, 200)
    ttft = np.concatenate([a, b])
    y = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    assert separability_auc(ttft, y) > 0.95
    assert separability_auc(ttft, 1 - y) > 0.95  # same separability either polarity


# --- end-to-end compute_metrics handles BOTH polarities ---

def _records(hit_mean: float, miss_mean: float, n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    recs = []
    buckets = ["short", "medium", "long"]
    for i in range(n):
        b = buckets[i % 3]
        hit = float(rng.normal(hit_mean, 5.0))
        miss = float(rng.normal(miss_mean, 5.0))
        for label, ms in (("hit", hit), ("miss", miss)):
            recs.append(
                {"prefix_id": f"p{i}", "bucket": b, "n_tokens": 100,
                 "label": label, "ttft_ms": ms, "trial": 0}
            )
    return recs


def test_compute_metrics_hit_faster():
    cfg = ExperimentConfig(seed=0)
    report = compute_metrics(_records(hit_mean=50, miss_mean=90), cfg)
    assert report["direction"] == "hit_faster"
    assert report["overall"]["auc"] > 0.9
    assert report["overall"]["delta_median_ms"] > 0  # miss - hit > 0


def test_compute_metrics_hit_slower_inverted():
    # The L4 short-prefix regime: cache hit is SLOWER than miss.
    cfg = ExperimentConfig(seed=0)
    report = compute_metrics(_records(hit_mean=124, miss_mean=88), cfg)
    assert report["direction"] == "hit_slower"
    assert report["overall"]["auc"] > 0.9  # still detectable, just inverted
    assert report["overall"]["delta_median_ms"] < 0  # miss - hit < 0
    for b in report["by_bucket"].values():
        assert b["separability_auc"] > 0.9
