"""LMSYS-Chat-1M background-traffic replay harness (Phase-4 bridge).

NOT used by the clean baseline (Experiment 1). This is the scaffold for Phase 4,
where victim/attacker probes are interleaved with realistic background load to
test whether queueing jitter masks the timing signal.

Design notes for when this is wired in:
  - Background requests are replayed on a SEPARATE thread/process from the
    sequential probe path so they create cache pressure without serializing
    behind the timed probe. The probe path itself stays single-request.
  - Inter-arrival gaps are drawn to match observed LMSYS timing (~1.8s mean at
    target load); rate multipliers 0.5x / 1x / 2x scale the mean gap.
  - No prompt is replayed twice (the report's "no prompt reused" constraint).

Left intentionally minimal; the baseline does not import or run it.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .client import ProbeClient


def load_background(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class BackgroundConfig:
    base_url: str
    mean_gap_s: float = 1.8  # observed LMSYS mean inter-arrival at target load
    rate_multiplier: float = 1.0  # 0.5x / 1x / 2x
    seed: int = 1234


class BackgroundReplayer:
    """Replays background prompts at a Poisson-ish rate on a worker thread.

    TODO(phase4): integrate with the probe loop; record achieved request rate
    and server queue depth so we can report load alongside AUC.
    """

    def __init__(self, prompts: list[dict], cfg: BackgroundConfig):
        self.prompts = list(prompts)
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sent = 0

    def _run(self) -> None:
        client = ProbeClient(self.cfg.base_url)
        idx = 0
        mean_gap = self.cfg.mean_gap_s / max(self.cfg.rate_multiplier, 1e-6)
        try:
            while not self._stop.is_set() and idx < len(self.prompts):
                try:
                    # Reuse the streaming generate path; we don't time these.
                    client.ttft(self.prompts[idx]["text"])
                    self._sent += 1
                except Exception:
                    pass
                idx += 1
                self._stop.wait(self._rng.expovariate(1.0 / mean_gap))
        finally:
            client.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    @property
    def sent(self) -> int:
        return self._sent
