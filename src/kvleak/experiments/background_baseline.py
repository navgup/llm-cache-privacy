"""Phase 4: baseline attack under concurrent background load.

Reruns the matched cache-miss / cache-hit probe protocol over a MedQA subset
while a background replayer fires requests at 0.5x / 1x / 2x the base rate.
Because SGLang batches concurrent requests, the timed probe inherits queueing
jitter from the background load — this measures whether that load masks the
cache-hit timing signal.

Important: we CANNOT flush the radix cache between probes here, because under
concurrent load the server is never idle and ``/flush_cache`` returns 400. Instead
we flush once per rate *before* starting the replayer (server idle), then rely on
"first send of a fresh prefix is a cache miss": each MedQA probe is sent twice in
a row — the first send is the cold miss (it was not cached; the background corpus
is different text), the second is the warm hit. Distinct probes per rate, so no
probe is pre-cached. Records are written incrementally per rate.

Run on the VM (server already up, background corpus prepared)::

    uv run python -m kvleak.experiments.background_baseline --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import time

from tqdm import tqdm

from ..background import (
    BackgroundConfig,
    BackgroundReplayer,
    PromptSource,
    load_background,
)
from ..client import ProbeClient
from ..config import ExperimentConfig, load_config
from ..server import SGLangServer
from .baseline import load_probes


def run_background_baseline(cfg: ExperimentConfig) -> str:
    probes = load_probes(cfg.medqa_probes_file)[: cfg.background.n_probes]
    if not probes:
        raise RuntimeError("no MedQA probes; run kvleak.data.medqa first")
    background = load_background(cfg.lmsys_background_file)
    if not background:
        raise RuntimeError("no LMSYS background; run kvleak.data.lmsys first")

    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "background_raw.jsonl"
    open(out_path, "w").close()  # truncate; we append per rate

    client = ProbeClient(cfg.server.base_url)
    ctrl = SGLangServer(cfg.server)  # HTTP control only (flush_cache)
    rate_stats: list[dict] = []
    total = 0
    try:
        client.warmup(n=2)
        for rate in cfg.background.rates:
            # Flush ONCE while the server is still idle (before the replayer),
            # clearing any probes cached by a previous rate.
            try:
                ctrl.flush_cache()
            except Exception:
                pass
            source = PromptSource(background, seed=cfg.seed)
            replayer = BackgroundReplayer(
                source,
                BackgroundConfig(
                    base_url=cfg.server.base_url,
                    mean_gap_s=cfg.background.base_mean_gap_s,
                    rate_multiplier=rate,
                    seed=cfg.seed,
                ),
            )
            replayer.start()
            time.sleep(cfg.background.warmup_s)  # let the load ramp
            batch: list[dict] = []
            skipped = 0
            t0 = time.time()
            for probe in tqdm(probes, desc=f"rate {rate}x"):
                text = probe["text"]
                try:
                    miss_ms = client.ttft(text)  # first send: cold miss
                    hit_ms = client.ttft(text)   # second send: warm hit
                except Exception:
                    skipped += 1
                    continue
                for label, ttft_ms in (("miss", miss_ms), ("hit", hit_ms)):
                    batch.append(
                        {
                            "prefix_id": probe["id"],
                            "bucket": probe["bucket"],
                            "n_tokens": probe["n_tokens"],
                            "label": label,
                            "ttft_ms": ttft_ms,
                            "rate": rate,
                        }
                    )
            dur = time.time() - t0
            replayer.stop()
            with open(out_path, "a") as f:  # checkpoint this rate
                for rec in batch:
                    f.write(json.dumps(rec) + "\n")
            total += len(batch)
            achieved = replayer.sent / dur if dur > 0 else 0.0
            rate_stats.append(
                {"rate": rate, "bg_sent": replayer.sent, "duration_s": dur,
                 "achieved_bg_req_per_s": achieved, "probes_skipped": skipped})
            print(f"rate {rate}x: {len(batch)//2} probes, {skipped} skipped; "
                  f"background sent {replayer.sent} ({achieved:.2f} req/s) over {dur:.0f}s")
    finally:
        client.close()

    with open(cfg.results_path / "background_rate_stats.json", "w") as f:
        json.dump(rate_stats, f, indent=2)
    print(f"Wrote {total} measurements -> {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: baseline under background load")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_background_baseline(cfg)


if __name__ == "__main__":
    main()
