"""Threshold calibration and attack metrics.

Convention: the **positive class is a cache hit**. We do NOT assume which way the
timing signal points. On large/slow-prefill setups a hit is *faster* (lower
TTFT); but for short prefixes on a fast GPU (e.g. Llama-3.1-8B on an L4) a hit
can be *slower* — the cached path runs a batch-size-1 decode/extend through a
CUDA graph whose fixed overhead exceeds a cheap cold prefill. The attack only
needs the hit/miss TTFT distributions to be *separable*, regardless of sign.

So we learn the polarity from the calibration split and build an oriented score
(higher score = more hit-like): ``score = direction * ttft`` where
``direction = +1`` if hits are slower and ``-1`` if hits are faster. All
low-level metrics operate on that oriented score with the rule
``predict hit iff score >= threshold``.

Reported metrics (Experiment 1 deliverable):
  - AUC                          (threshold-free separability, on held-out eval)
  - precision @ target recall    (attacker precision at e.g. 80% recall)
  - bits leaked per probe        (mutual information of the calibrated decision)
  - direction                    (hit_faster | hit_slower) + signed median ΔTTFT

Run after the experiment::

    uv run python -m kvleak.analysis.metrics --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mutual_info_score, roc_auc_score

from ..config import ExperimentConfig, load_config


def labels_to_int(labels: list[str]) -> np.ndarray:
    """Map 'hit' -> 1 (positive), 'miss' -> 0."""
    return np.array([1 if x == "hit" else 0 for x in labels], dtype=int)


# Backwards-compatible alias (older callers imported the private name).
_labels_to_int = labels_to_int


def detect_direction(ttft_ms: np.ndarray, y: np.ndarray) -> int:
    """Learn the signal polarity: +1 if hits are SLOWER (higher TTFT), else -1.

    Decided by comparing class median TTFTs. Should be called on the calibration
    split only so the eval split stays blind.
    """
    hit = ttft_ms[y == 1]
    miss = ttft_ms[y == 0]
    if len(hit) == 0 or len(miss) == 0:
        return -1
    return 1 if np.median(hit) >= np.median(miss) else -1


def oriented_score(ttft_ms: np.ndarray, direction: int) -> np.ndarray:
    """Oriented detector score where higher = more hit-like."""
    return direction * ttft_ms


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC for an oriented score (higher = more hit-like)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def separability_auc(ttft_ms: np.ndarray, y: np.ndarray) -> float:
    """Direction-free separability: AUC under whichever polarity is better.

    Used for per-bucket reporting so a bucket whose sign differs from the global
    calibrated direction still reports its true detectability (>= 0.5).
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    a = float(roc_auc_score(y, ttft_ms))
    return max(a, 1.0 - a)


def youden_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Oriented-score threshold maximizing Youden's J for rule score >= thr."""
    candidates = np.unique(score)
    best_thr, best_j = float(candidates[0]), -np.inf
    pos = y == 1
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int((~pos).sum()), 1)
    for thr in candidates:
        pred = score >= thr
        tpr = np.sum(pred & pos) / n_pos
        fpr = np.sum(pred & ~pos) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def precision_at_recall(
    score: np.ndarray, y: np.ndarray, target_recall: float
) -> tuple[float, float]:
    """Max precision among thresholds with recall >= target_recall.

    Operates on the oriented score (rule: predict hit iff score >= threshold).
    Returns ``(precision, threshold)``.
    """
    pos = y == 1
    n_pos = max(int(pos.sum()), 1)
    best_prec, best_thr = 0.0, float(np.min(score))
    for thr in np.unique(score):
        pred = score >= thr
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        recall = tp / n_pos
        if recall >= target_recall and (tp + fp) > 0:
            prec = tp / (tp + fp)
            if prec > best_prec:
                best_prec, best_thr = prec, float(thr)
    return best_prec, best_thr


def bits_leaked(score: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Mutual information (bits) between true label and the thresholded decision."""
    pred = (score >= threshold).astype(int)
    mi_nats = mutual_info_score(y, pred)
    return float(mi_nats / math.log(2))


def split_by_prefix(
    prefix_ids: list[str], calib_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    """Split unique prefixes into calibration / eval sets (hit & miss stay together)."""
    uniq = sorted(set(prefix_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_calib = max(1, int(round(len(uniq) * calib_frac)))
    calib = set(uniq[:n_calib])
    return calib, set(uniq[n_calib:])


@dataclass
class MetricBlock:
    n_hit: int
    n_miss: int
    auc: float
    precision_at_recall: float
    target_recall: float
    bits_leaked: float
    threshold_ms: float  # threshold expressed back in TTFT ms
    direction: str  # "hit_slower" | "hit_faster"
    ttft_hit_median_ms: float
    ttft_miss_median_ms: float
    delta_median_ms: float  # miss - hit (POSITIVE = hit faster; NEGATIVE = hit slower)


def _block(
    ttft: np.ndarray,
    y: np.ndarray,
    direction: int,
    threshold_score: float,
    target_recall: float,
) -> MetricBlock:
    score = oriented_score(ttft, direction)
    prec, _ = precision_at_recall(score, y, target_recall)
    hit_ms = ttft[y == 1]
    miss_ms = ttft[y == 0]
    hit_med = float(np.median(hit_ms)) if len(hit_ms) else float("nan")
    miss_med = float(np.median(miss_ms)) if len(miss_ms) else float("nan")
    return MetricBlock(
        n_hit=int((y == 1).sum()),
        n_miss=int((y == 0).sum()),
        auc=auc(score, y),
        precision_at_recall=prec,
        target_recall=target_recall,
        bits_leaked=bits_leaked(score, y, threshold_score),
        threshold_ms=direction * threshold_score,  # back to ms
        direction="hit_slower" if direction > 0 else "hit_faster",
        ttft_hit_median_ms=hit_med,
        ttft_miss_median_ms=miss_med,
        delta_median_ms=miss_med - hit_med,
    )


def compute_metrics(records: list[dict], cfg: ExperimentConfig) -> dict:
    """Full Experiment-1 metric report from raw measurements."""
    prefix_ids = [r["prefix_id"] for r in records]
    ttft = np.array([r["ttft_ms"] for r in records], dtype=float)
    y = labels_to_int([r["label"] for r in records])
    buckets = np.array([r["bucket"] for r in records])

    calib_ids, _ = split_by_prefix(prefix_ids, cfg.analysis.calib_frac, cfg.seed)
    in_calib = np.array([pid in calib_ids for pid in prefix_ids])
    eval_mask = ~in_calib
    target = cfg.analysis.target_recall

    # Learn polarity + threshold on calibration only.
    direction = detect_direction(ttft[in_calib], y[in_calib])
    threshold_score = youden_threshold(
        oriented_score(ttft[in_calib], direction), y[in_calib]
    )

    report: dict = {
        "direction": "hit_slower" if direction > 0 else "hit_faster",
        "calibration_threshold_ms": direction * threshold_score,
        "n_calib_measurements": int(in_calib.sum()),
        "n_eval_measurements": int(eval_mask.sum()),
        "overall": asdict(
            _block(ttft[eval_mask], y[eval_mask], direction, threshold_score, target)
        ),
        "by_bucket": {},
    }
    for b in sorted(set(buckets.tolist())):
        bm = eval_mask & (buckets == b)
        if bm.sum() == 0:
            continue
        block = asdict(
            _block(ttft[bm], y[bm], direction, threshold_score, target)
        )
        # Per-bucket separability is direction-free so a bucket whose sign flips
        # vs. the global direction still shows its true detectability.
        block["separability_auc"] = separability_auc(ttft[bm], y[bm])
        report["by_bucket"][b] = block
    return report


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute baseline attack metrics")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None, help="override raw measurements path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_path = Path(args.raw) if args.raw else cfg.results_path / "baseline_raw.jsonl"
    records = load_records(raw_path)
    report = compute_metrics(records, cfg)

    o = report["overall"]
    print(f"Signal direction: {report['direction']}  (hit median vs miss median)")
    print(f"Calibrated threshold: {report['calibration_threshold_ms']:.3f} ms")
    print(
        f"EVAL  AUC={o['auc']:.4f}  "
        f"precision@{int(o['target_recall']*100)}%recall={o['precision_at_recall']:.4f}  "
        f"bits/probe={o['bits_leaked']:.4f}"
    )
    print(
        f"      median TTFT  hit={o['ttft_hit_median_ms']:.2f}ms  "
        f"miss={o['ttft_miss_median_ms']:.2f}ms  "
        f"Δ(miss-hit)={o['delta_median_ms']:.2f}ms"
    )
    for b, m in report["by_bucket"].items():
        print(
            f"  [{b:>6}] AUC={m['auc']:.4f}  sep={m['separability_auc']:.4f}  "
            f"Δ(miss-hit)={m['delta_median_ms']:.2f}ms"
        )

    out = cfg.results_path / "baseline_metrics.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
