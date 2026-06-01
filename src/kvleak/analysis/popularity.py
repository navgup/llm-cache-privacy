"""Experiment 2b analysis: popularity structure vs eviction policy.

(a) Skewed-eviction survival curves (lru vs lfu) for a rare victim + attack window.
(b) Popularity oracle: P(cached) vs candidate frequency, per policy.

    uv run python -m kvleak.analysis.popularity --config configs/baseline.yaml
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
from .eviction import attack_window, hit_rate_curve  # noqa: E402


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    evict = [r for r in records if r["part"] == "evict"]
    oracle = [r for r in records if r["part"] == "oracle"]
    policies = sorted({r["policy"] for r in records})

    evict_summary = {}
    for pol in policies:
        recs = [r for r in evict if r["policy"] == pol]
        if recs:
            curve = hit_rate_curve(recs)
            evict_summary[pol] = {"curve": curve,
                                  "attack_window_prompts": attack_window(curve)}

    oracle_summary = {}
    for pol in policies:
        recs = [r for r in oracle if r["policy"] == pol]
        freqs = sorted({r["freq"] for r in recs})
        pts = []
        for fr in freqs:
            sub = [r for r in recs if r["freq"] == fr]
            pts.append({"freq": fr,
                        "p_cached": float(np.mean([1.0 if r["cached"] else 0.0 for r in sub])),
                        "n": len(sub)})
        oracle_summary[pol] = pts
    return {"evict": evict_summary, "oracle": oracle_summary}


def plot(summary: dict, out: Path) -> list[Path]:
    paths = []
    # (a) skewed-eviction survival curves.
    if summary["evict"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for pol, s in sorted(summary["evict"].items()):
            vols = [c["volume"] for c in s["curve"]]
            hr = [c["hit_rate"] for c in s["curve"]]
            ax.plot(vols, hr, "-o",
                    label=f"{pol} (window≈{s['attack_window_prompts']:.0f})")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("skewed background volume injected (# prompts)")
        ax.set_ylabel("rare-victim cache-hit survival rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Rare-victim attack window: LRU vs LFU (skewed background)")
        ax.legend()
        f = out / "popularity_evict.png"
        fig.tight_layout(); fig.savefig(f, dpi=150); plt.close(fig); paths.append(f)

    # (b) popularity oracle.
    if summary["oracle"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for pol, pts in sorted(summary["oracle"].items()):
            fr = [p["freq"] for p in pts]
            pc = [p["p_cached"] for p in pts]
            ax.plot(fr, pc, "-o", label=pol)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate access frequency (popularity)")
        ax.set_ylabel("P(cached) — probe says 'hit'")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Popularity oracle: P(cached) vs frequency (recency equalized)")
        ax.legend()
        f = out / "popularity_oracle.png"
        fig.tight_layout(); fig.savefig(f, dpi=150); plt.close(fig); paths.append(f)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2b analysis")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else cfg.results_path / "popularity_raw.jsonl"
    summary = summarize(load_records(raw))

    print("=== (a) rare-victim attack window (skewed background) ===")
    for pol, s in summary["evict"].items():
        print(f"  {pol}: window ≈ {s['attack_window_prompts']:.1f} prompts")
    print("=== (b) popularity oracle  P(cached) by frequency ===")
    for pol, pts in summary["oracle"].items():
        cells = "  ".join(f"f={p['freq']}:{p['p_cached']:.2f}" for p in pts)
        print(f"  {pol}:  {cells}")

    out = cfg.results_path
    with open(out / "popularity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    paths = plot(summary, out)
    print("Wrote " + ", ".join(str(p) for p in [out / "popularity_summary.json", *paths]))


if __name__ == "__main__":
    main()
