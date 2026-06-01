"""Experiment 2 analysis: eviction window vs cache size and eviction policy.

For each server config, computes the cache-hit survival rate as a function of
injected background volume, and the **attack window** = the largest volume at
which the victim is still cached for a majority of trials (hit-rate >= 0.5),
interpolated. Produces one figure per sweep (cache size, eviction policy).

    uv run python -m kvleak.analysis.eviction --config configs/baseline.yaml
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


def hit_rate_curve(recs: list[dict]) -> list[dict]:
    vols = sorted({r["volume"] for r in recs})
    curve = []
    for v in vols:
        sub = [r for r in recs if r["volume"] == v]
        hr = float(np.mean([1.0 if r["cached"] else 0.0 for r in sub]))
        curve.append({"volume": v, "hit_rate": hr, "n": len(sub)})
    return curve


def attack_window(curve: list[dict]) -> float:
    """Interpolated background volume where hit-rate crosses 0.5 (the window)."""
    pts = [(c["volume"], c["hit_rate"]) for c in curve]
    if pts and pts[0][1] < 0.5:
        return 0.0
    for (v0, h0), (v1, h1) in zip(pts, pts[1:]):
        if h1 < 0.5 <= h0:
            return float(v0 + (v1 - v0) * (h0 - 0.5) / (h0 - h1))
    return float(pts[-1][0]) if pts else float("nan")  # never evicted in range


def summarize(records: list[dict]) -> dict:
    configs = sorted({r["config"] for r in records})
    out = {}
    for cfg_tag in configs:
        recs = [r for r in records if r["config"] == cfg_tag]
        curve = hit_rate_curve(recs)
        out[cfg_tag] = {
            "policy": recs[0]["policy"],
            "max_total_tokens": recs[0]["max_total_tokens"],
            "server_max_total_num_tokens": recs[0].get("server_max_total_num_tokens"),
            "sweeps": recs[0]["sweeps"],
            "curve": curve,
            "attack_window_prompts": attack_window(curve),
        }
    return out


def _plot_group(summary: dict, sweep: str, label_key: str, out: Path) -> Path | None:
    items = [(t, s) for t, s in summary.items() if sweep in s["sweeps"]]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1][label_key])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tag, s in items:
        vols = [c["volume"] for c in s["curve"]]
        hr = [c["hit_rate"] for c in s["curve"]]
        ax.plot(vols, hr, "-o", label=f"{label_key}={s[label_key]} "
                f"(window≈{s['attack_window_prompts']:.0f})")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("background volume injected (# prompts)")
    ax.set_ylabel("cache-hit survival rate")
    ax.set_ylim(-0.02, 1.02)
    title = "Attack window vs cache size" if sweep == "cache_size" \
        else "Attack window vs eviction policy"
    ax.set_title(title)
    ax.legend(fontsize=8)
    f = out / (f"eviction_{sweep}.png")
    fig.tight_layout(); fig.savefig(f, dpi=150); plt.close(fig)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2 eviction analysis")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else cfg.results_path / "eviction_sweep_raw.jsonl"
    records = load_records(raw)
    summary = summarize(records)

    print(f"{'config':>16} {'sweeps':>26} {'server_tokens':>13} {'window(prompts)':>16}")
    for tag, s in summary.items():
        print(f"{tag:>16} {','.join(s['sweeps']):>26} "
              f"{str(s['server_max_total_num_tokens']):>13} "
              f"{s['attack_window_prompts']:>16.1f}")

    out = cfg.results_path
    with open(out / "eviction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    paths = [p for p in (
        _plot_group(summary, "cache_size", "max_total_tokens", out),
        _plot_group(summary, "eviction_policy", "policy", out),
    ) if p]
    print("Wrote " + ", ".join(str(p) for p in [out / "eviction_summary.json", *paths]))


if __name__ == "__main__":
    main()
