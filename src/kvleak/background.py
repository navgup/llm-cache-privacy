"""LMSYS-Chat-1M background-traffic harness.

Two ways background traffic is used:

  * **Concurrent replay** (Phase 4): a worker thread fires background requests at
    a Poisson rate *while* the probe thread measures TTFT, so the probe inherits
    realistic queueing jitter. Tests whether load masks the timing signal.
  * **Synchronous injection** (Experiment 2): between caching a victim prefix and
    probing it, we inject N distinct background prompts back-to-back to create
    cache pressure / eviction. The "attack window" is the volume N at which the
    victim gets evicted.

Both draw from a shared :class:`PromptSource` so no prompt is reused (the report's
"no prompt replayed twice" constraint), wrapping around only if the pool runs out.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path

from .client import ProbeClient


def load_background(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class PromptSource:
    """A cursor over background prompts that hands out fresh, non-repeating texts."""

    def __init__(self, prompts: list[dict], seed: int = 1234):
        self._texts = [p["text"] for p in prompts]
        random.Random(seed).shuffle(self._texts)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._texts)

    def next_batch(self, n: int) -> list[str]:
        out = []
        for _ in range(n):
            out.append(self._texts[self._cursor % len(self._texts)])
            self._cursor += 1
        return out


def inject(client: ProbeClient, source: PromptSource, n: int) -> None:
    """Synchronously send ``n`` fresh background prompts (cache-pressure injection)."""
    for text in source.next_batch(n):
        try:
            client.ttft(text)  # the prefill caches the prompt; we ignore timing
        except Exception:
            pass


@dataclass
class BackgroundConfig:
    base_url: str
    mean_gap_s: float = 1.8  # observed LMSYS mean inter-arrival at 1x load
    rate_multiplier: float = 1.0  # 0.5x / 1x / 2x
    seed: int = 1234


class BackgroundReplayer:
    """Replays background prompts at a Poisson rate on a worker thread (Phase 4)."""

    def __init__(self, source: PromptSource, cfg: BackgroundConfig):
        self.source = source
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sent = 0

    def _run(self) -> None:
        client = ProbeClient(self.cfg.base_url)
        mean_gap = self.cfg.mean_gap_s / max(self.cfg.rate_multiplier, 1e-6)
        try:
            while not self._stop.is_set():
                try:
                    client.ttft(self.source.next_batch(1)[0])
                    self._sent += 1
                except Exception:
                    pass
                self._stop.wait(self._rng.expovariate(1.0 / mean_gap))
        finally:
            client.close()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    @property
    def sent(self) -> int:
        return self._sent
