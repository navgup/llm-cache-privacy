"""TTFT probe client.

Measures client-side time-to-first-token against SGLang's native ``/generate``
endpoint. The first streamed chunk falls out of the prefill forward pass, so
with ``max_new_tokens=1`` the TTFT *is* the prefill time plus fixed overhead —
exactly the cache-hit/miss signal we want.

Probes MUST be sent strictly sequentially: SGLang batches concurrent requests,
which mixes prefills and destroys per-request timing.
"""

from __future__ import annotations

import time

import httpx


class ProbeClient:
    def __init__(self, base_url: str, request_timeout_s: float = 120.0):
        # http2 disabled: one-at-a-time requests, plain HTTP/1.1 keeps timing simple.
        self._client = httpx.Client(base_url=base_url, timeout=request_timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ProbeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _payload(self, text: str) -> dict:
        return {
            "text": text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "stream": True,
        }

    def ttft(self, text: str) -> float:
        """Return time-to-first-token in **milliseconds** for ``text``.

        Timing starts immediately before the request is sent and stops on the
        first non-empty streamed chunk.
        """
        start = time.perf_counter()
        with self._client.stream("POST", "/generate", json=self._payload(text)) as r:
            r.raise_for_status()
            for chunk in r.iter_lines():
                if chunk:  # first SSE data line == first token emitted
                    elapsed = time.perf_counter() - start
                    return elapsed * 1000.0
        raise RuntimeError("stream ended without producing a token")

    def warmup(self, text: str = "warmup", n: int = 1) -> None:
        """Prime the connection / CUDA graphs so the first real probe isn't biased."""
        for _ in range(n):
            try:
                self.ttft(text)
            except httpx.HTTPError:
                pass
