"""Analyze the prefix-length crossover sweep.

For each length, pairs the cold (miss) and warm (hit) TTFT per (base_text, trial)
and reports the median ΔTTFT = miss - hit with a bootstrap CI. A NEGATIVE Δ means
the cache hit is SLOWER (the short-prefix L4 regime); the crossover length is
where Δ passes through zero into positive (hit faster, the classic side channel).

Run after the sweep::

    uv run python -m kvleak.analysis.crossover --config configs/baseline.yaml
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


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _paired_deltas(recs: list[dict]) -> np.ndarray:
    """ΔTTFT = miss - hit per (base_id, trial) within one length."""
    miss = {(r["base_id"], r["trial"]): r["ttft_ms"] for r in recs if r["label"] == "miss"}
    hit = {(r["base_id"], r["trial"]): r["ttft_ms"] for r in recs if r["label"] == "hit"}
    keys = sorted(set(miss) & set(hit))
    return np.array([miss[k] - hit[k] for k in keys], dtype=float)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def aggregate(records: list[dict]) -> list[dict]:
    lengths = sorted({r["length"] for r in records})
    rows = []
    for L in lengths:
        recs = [r for r in records if r["length"] == L]
        cold = np.array([r["ttft_ms"] for r in recs if r["label"] == "miss"])
        warm = np.array([r["ttft_ms"] for r in recs if r["label"] == "hit"])
        deltas = _paired_deltas(recs)
        lo, hi = _bootstrap_ci(deltas)
        rows.append(
            {
                "length": L,
                "n_pairs": int(len(deltas)),
                "cold_median_ms": float(np.median(cold)) if len(cold) else float("nan"),
                "warm_median_ms": float(np.median(warm)) if len(warm) else float("nan"),
                "delta_median_ms": float(np.median(deltas)) if len(deltas) else float("nan"),
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
            }
        )
    return rows


def find_crossover(rows: list[dict]) -> float | None:
    """Linearly interpolate the length where median Δ crosses zero (neg->pos)."""
    pts = [(r["length"], r["delta_median_ms"]) for r in rows]
    for (l0, d0), (l1, d1) in zip(pts, pts[1:]):
        if d0 == 0:
            return float(l0)
        if (d0 < 0) != (d1 < 0):  # sign change
            return float(l0 + (l1 - l0) * (0 - d0) / (d1 - d0))
    return None


def plot_curves(rows: list[dict], crossover: float | None, out_dir: Path) -> list[Path]:
    lengths = np.array([r["length"] for r in rows])
    cold = np.array([r["cold_median_ms"] for r in rows])
    warm = np.array([r["warm_median_ms"] for r in rows])
    delta = np.array([r["delta_median_ms"] for r in rows])
    lo = np.array([r["delta_ci_lo"] for r in rows])
    hi = np.array([r["delta_ci_hi"] for r in rows])

    # Figure 1: cold vs warm median TTFT (the mechanism — cold rises, warm ~flat).
    f1 = out_dir / "crossover_ttft.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lengths, cold, "-o", color="tab:red", label="miss (cold prefill)")
    ax.plot(lengths, warm, "-o", color="tab:blue", label="hit (warm, 1-tok extend)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("median TTFT (ms)")
    ax.set_title("TTFT vs prefix length: cache hit vs miss")
    ax.legend()
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)

    # Figure 2: ΔTTFT(miss-hit) vs length with zero line + crossover.
    f2 = out_dir / "crossover_delta.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(lengths, lo, hi, color="tab:purple", alpha=0.2, label="95% CI")
    ax.plot(lengths, delta, "-o", color="tab:purple", label="Δ = miss − hit")
    ax.axhline(0, color="gray", ls="--", lw=1)
    if crossover is not None:
        ax.axvline(crossover, color="black", ls=":", lw=1,
                   label=f"crossover ≈ {crossover:.0f} tok")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("ΔTTFT  (ms)   [>0: hit faster]")
    ax.set_title("Cache-hit timing signal vs prefix length")
    ax.legend()
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)
    return [f1, f2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze length crossover sweep")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else cfg.results_path / "length_sweep_raw.jsonl"
    records = load_records(raw)
    rows = aggregate(records)
    crossover = find_crossover(rows)

    print(f"{'length':>8} {'n':>4} {'cold_ms':>9} {'warm_ms':>9} "
          f"{'delta_ms':>9} {'95% CI':>20}")
    for r in rows:
        print(f"{r['length']:>8} {r['n_pairs']:>4} {r['cold_median_ms']:>9.2f} "
              f"{r['warm_median_ms']:>9.2f} {r['delta_median_ms']:>9.2f} "
              f"[{r['delta_ci_lo']:>7.2f}, {r['delta_ci_hi']:>7.2f}]")
    if crossover is not None:
        print(f"\nΔTTFT crosses zero at ~{crossover:.0f} tokens "
              f"(below: hit slower; above: hit faster).")
    else:
        sign = "hit slower (Δ<0)" if rows and rows[-1]["delta_median_ms"] < 0 else "hit faster (Δ>0)"
        print(f"\nNo sign crossing in the swept range — all {sign}.")

    out = cfg.results_path
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "crossover.json", "w") as f:
        json.dump({"crossover_tokens": crossover, "by_length": rows}, f, indent=2)
    paths = plot_curves(rows, crossover, out)
    print("Wrote " + ", ".join(str(p) for p in [out / "crossover.json", *paths]))


if __name__ == "__main__":
    main()
