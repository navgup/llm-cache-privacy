"""Threshold calibration and attack metrics.

Convention: the **positive class is a cache hit**. Hits have *lower* TTFT, so
when a higher-is-positive score is needed we use ``-ttft``. The detector rule is
``predict hit iff ttft <= threshold``.

Reported metrics (Experiment 1 deliverable):
  - AUC                          (threshold-free separability)
  - precision @ target recall    (attacker precision at e.g. 80% recall)
  - bits leaked per probe        (mutual information of the calibrated decision)

Threshold is fit on a calibration split via Youden's J and evaluated, blind, on
the held-out split.

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


def _labels_to_int(labels: list[str]) -> np.ndarray:
    """Map 'hit' -> 1 (positive), 'miss' -> 0."""
    return np.array([1 if x == "hit" else 0 for x in labels], dtype=int)


def auc(ttft_ms: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC for the hit-vs-miss timing signal."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, -ttft_ms))  # lower ttft => more hit-like


def youden_threshold(ttft_ms: np.ndarray, y: np.ndarray) -> float:
    """TTFT threshold maximizing Youden's J (TPR - FPR) for rule ttft <= thr."""
    candidates = np.unique(ttft_ms)
    best_thr, best_j = float(candidates[0]), -np.inf
    pos = y == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    for thr in candidates:
        pred = ttft_ms <= thr
        tpr = np.sum(pred & pos) / n_pos
        fpr = np.sum(pred & neg) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def precision_at_recall(
    ttft_ms: np.ndarray, y: np.ndarray, target_recall: float
) -> tuple[float, float]:
    """Max precision among thresholds with recall >= target_recall.

    Returns ``(precision, threshold)``. Threshold is in TTFT ms (predict hit iff
    ttft <= threshold).
    """
    pos = y == 1
    n_pos = max(int(pos.sum()), 1)
    best_prec, best_thr = 0.0, float(np.max(ttft_ms))
    for thr in np.unique(ttft_ms):
        pred = ttft_ms <= thr
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        recall = tp / n_pos
        if recall >= target_recall and (tp + fp) > 0:
            prec = tp / (tp + fp)
            if prec > best_prec:
                best_prec, best_thr = prec, float(thr)
    return best_prec, best_thr


def bits_leaked(ttft_ms: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Mutual information (bits) between true label and the thresholded decision."""
    pred = (ttft_ms <= threshold).astype(int)
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
    threshold_ms: float
    ttft_hit_median_ms: float
    ttft_miss_median_ms: float
    delta_median_ms: float


def _block(
    ttft: np.ndarray, y: np.ndarray, threshold: float, target_recall: float
) -> MetricBlock:
    hit_ms = ttft[y == 1]
    miss_ms = ttft[y == 0]
    prec, _ = precision_at_recall(ttft, y, target_recall)
    hit_med = float(np.median(hit_ms)) if len(hit_ms) else float("nan")
    miss_med = float(np.median(miss_ms)) if len(miss_ms) else float("nan")
    return MetricBlock(
        n_hit=int((y == 1).sum()),
        n_miss=int((y == 0).sum()),
        auc=auc(ttft, y),
        precision_at_recall=prec,
        target_recall=target_recall,
        bits_leaked=bits_leaked(ttft, y, threshold),
        threshold_ms=threshold,
        ttft_hit_median_ms=hit_med,
        ttft_miss_median_ms=miss_med,
        delta_median_ms=miss_med - hit_med,
    )


def compute_metrics(records: list[dict], cfg: ExperimentConfig) -> dict:
    """Full Experiment-1 metric report from raw measurements."""
    prefix_ids = [r["prefix_id"] for r in records]
    ttft = np.array([r["ttft_ms"] for r in records], dtype=float)
    y = _labels_to_int([r["label"] for r in records])
    buckets = np.array([r["bucket"] for r in records])

    calib_ids, _ = split_by_prefix(
        prefix_ids, cfg.analysis.calib_frac, cfg.seed
    )
    in_calib = np.array([pid in calib_ids for pid in prefix_ids])

    # Calibrate threshold on calibration split only.
    threshold = youden_threshold(ttft[in_calib], y[in_calib])

    eval_mask = ~in_calib
    target = cfg.analysis.target_recall

    report: dict = {
        "calibration_threshold_ms": threshold,
        "n_calib_measurements": int(in_calib.sum()),
        "n_eval_measurements": int(eval_mask.sum()),
        "overall": asdict(
            _block(ttft[eval_mask], y[eval_mask], threshold, target)
        ),
        "by_bucket": {},
    }
    for b in sorted(set(buckets.tolist())):
        bm = eval_mask & (buckets == b)
        if bm.sum() == 0:
            continue
        report["by_bucket"][b] = asdict(_block(ttft[bm], y[bm], threshold, target))
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

    out = cfg.results_path / "baseline_metrics.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    o = report["overall"]
    print(f"Calibrated threshold: {report['calibration_threshold_ms']:.3f} ms")
    print(
        f"EVAL  AUC={o['auc']:.4f}  "
        f"precision@{int(o['target_recall']*100)}%recall={o['precision_at_recall']:.4f}  "
        f"bits/probe={o['bits_leaked']:.4f}"
    )
    print(
        f"      median TTFT  hit={o['ttft_hit_median_ms']:.2f}ms  "
        f"miss={o['ttft_miss_median_ms']:.2f}ms  Δ={o['delta_median_ms']:.2f}ms"
    )
    for b, m in report["by_bucket"].items():
        print(f"  [{b:>6}] AUC={m['auc']:.4f}  Δmedian={m['delta_median_ms']:.2f}ms")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
