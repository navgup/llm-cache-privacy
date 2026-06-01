"""Plots for the baseline attack: TTFT distributions and the ROC curve.

Run after the experiment::

    uv run python -m kvleak.analysis.plots --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (VM has no display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ..config import load_config  # noqa: E402
from .metrics import _labels_to_int, load_records  # noqa: E402


def plot_ttft_hist(ttft: np.ndarray, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(float(ttft.min()), float(np.percentile(ttft, 99)), 60)
    ax.hist(ttft[y == 0], bins=bins, alpha=0.6, label="miss (cold)", color="tab:red")
    ax.hist(ttft[y == 1], bins=bins, alpha=0.6, label="hit (cached)", color="tab:blue")
    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("count")
    ax.set_title("TTFT distribution: cache hit vs miss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(ttft: np.ndarray, y: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y, -ttft)  # lower ttft => more hit-like
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="tab:blue", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: cache-hit detection via TTFT")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot baseline attack figures")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_path = Path(args.raw) if args.raw else cfg.results_path / "baseline_raw.jsonl"
    records = load_records(raw_path)
    ttft = np.array([r["ttft_ms"] for r in records], dtype=float)
    y = _labels_to_int([r["label"] for r in records])

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    hist_out = cfg.results_path / "ttft_hist.png"
    roc_out = cfg.results_path / "roc.png"
    plot_ttft_hist(ttft, y, hist_out)
    plot_roc(ttft, y, roc_out)
    print(f"Wrote {hist_out}\nWrote {roc_out}")


if __name__ == "__main__":
    main()
