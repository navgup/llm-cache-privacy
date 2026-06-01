"""Metric sanity checks on synthetic TTFT distributions (no GPU/server needed)."""

from __future__ import annotations

import numpy as np

from kvleak.analysis.metrics import (
    auc,
    bits_leaked,
    precision_at_recall,
    split_by_prefix,
    youden_threshold,
)


def _synthetic(sep: float, n: int = 500, seed: int = 0):
    """Build (ttft, y) where hits (y=1) have lower TTFT by ``sep`` ms."""
    rng = np.random.default_rng(seed)
    hit = rng.normal(50.0, 5.0, n)
    miss = rng.normal(50.0 + sep, 5.0, n)
    ttft = np.concatenate([hit, miss])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    return ttft, y


def test_auc_well_separated():
    ttft, y = _synthetic(sep=40.0)
    assert auc(ttft, y) > 0.95


def test_auc_overlapping():
    ttft, y = _synthetic(sep=0.0)
    assert abs(auc(ttft, y) - 0.5) < 0.1


def test_youden_threshold_between_means():
    ttft, y = _synthetic(sep=40.0)
    thr = youden_threshold(ttft, y)
    # hit mean ~50, miss mean ~90 -> threshold should land between.
    assert 50.0 < thr < 90.0


def test_precision_at_recall_high_when_separable():
    ttft, y = _synthetic(sep=40.0)
    prec, thr = precision_at_recall(ttft, y, target_recall=0.8)
    assert prec > 0.9
    assert thr > 0


def test_bits_leaked_bounds():
    ttft, y = _synthetic(sep=40.0)
    thr = youden_threshold(ttft, y)
    bits = bits_leaked(ttft, y, thr)
    assert 0.0 <= bits <= 1.0 + 1e-9  # MI of a binary channel <= 1 bit (float slack)
    # Near-perfect separation of a balanced binary label -> close to 1 bit.
    assert bits > 0.7


def test_bits_leaked_near_zero_when_random():
    ttft, y = _synthetic(sep=0.0)
    thr = youden_threshold(ttft, y)
    assert bits_leaked(ttft, y, thr) < 0.1


def test_split_by_prefix_disjoint_and_grouped():
    prefix_ids = [f"p{i}" for i in range(100) for _ in range(2)]  # hit+miss each
    calib, evalset = split_by_prefix(prefix_ids, calib_frac=0.2, seed=1)
    assert calib.isdisjoint(evalset)
    assert len(calib) == 20
    assert len(calib) + len(evalset) == 100
