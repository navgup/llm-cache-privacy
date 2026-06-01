"""Phase 4 analysis: does background load mask the timing signal?

Per background rate, reports the direction-agnostic AUC and the probe-TTFT
percentiles (the load indicator). Plots AUC vs rate and p50/p99 vs rate.

    uv run python -m kvleak.analysis.background_robustness --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import load_config  # noqa: E402
from .metrics import auc, detect_direction, labels_to_int, oriented_score  # noqa: E402


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(records: list[dict]) -> list[dict]:
    rates = sorted({r["rate"] for r in records})
    rows = []
    for rate in rates:
        recs = [r for r in records if r["rate"] == rate]
        ttft = np.array([r["ttft_ms"] for r in recs], float)
        y = labels_to_int([r["label"] for r in recs])
        direction = detect_direction(ttft, y)
        a = auc(oriented_score(ttft, direction), y)
        hit = ttft[y == 1]
        miss = ttft[y == 0]
        rows.append(
            {
                "rate": rate,
                "auc": a,
                "direction": "hit_slower" if direction > 0 else "hit_faster",
                "hit_median_ms": float(np.median(hit)),
                "miss_median_ms": float(np.median(miss)),
                "delta_median_ms": float(np.median(miss) - np.median(hit)),
                "probe_p50_ms": float(np.percentile(ttft, 50)),
                "probe_p99_ms": float(np.percentile(ttft, 99)),
            }
        )
    return rows


def plot(rows: list[dict], out_dir: Path) -> list[Path]:
    rates = [r["rate"] for r in rows]
    auc_v = [r["auc"] for r in rows]
    p50 = [r["probe_p50_ms"] for r in rows]
    p99 = [r["probe_p99_ms"] for r in rows]

    f1 = out_dir / "background_auc.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, auc_v, "-o", color="tab:green")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("background rate (×)")
    ax.set_ylabel("attack AUC")
    ax.set_title("Attack AUC vs background load")
    ax.legend()
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)

    f2 = out_dir / "background_latency.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, p50, "-o", color="tab:blue", label="probe p50")
    ax.plot(rates, p99, "-o", color="tab:red", label="probe p99")
    ax.set_xlabel("background rate (×)")
    ax.set_ylabel("probe TTFT (ms)")
    ax.set_title("Probe TTFT percentiles vs background load")
    ax.legend()
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)
    return [f1, f2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 background robustness analysis")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else cfg.results_path / "background_raw.jsonl"
    rows = aggregate(load_records(raw))

    print(f"{'rate':>6} {'AUC':>7} {'direction':>11} {'Δmed_ms':>9} "
          f"{'p50_ms':>8} {'p99_ms':>8}")
    for r in rows:
        print(f"{r['rate']:>6} {r['auc']:>7.4f} {r['direction']:>11} "
              f"{r['delta_median_ms']:>9.2f} {r['probe_p50_ms']:>8.1f} "
              f"{r['probe_p99_ms']:>8.1f}")

    out = cfg.results_path
    with open(out / "background_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    paths = plot(rows, out)
    print("Wrote " + ", ".join(str(p) for p in [out / "background_metrics.json", *paths]))


if __name__ == "__main__":
    main()
