"""SGLang server lifecycle manager.

Spawns ``python -m sglang.launch_server`` from an :class:`ServerConfig`, waits
for ``/health``, and exposes the control endpoints the experiment needs
(``/flush_cache``, ``/server_info``). This only meaningfully runs on the GPU VM;
on a CPU-only box importing it is safe but ``start()`` will fail when SGLang
tries to load the model.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

import httpx

from .config import ServerConfig


def build_launch_args(cfg: ServerConfig) -> list[str]:
    """Construct the ``sglang.launch_server`` argv from config."""
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model_path,
        "--dtype",
        cfg.dtype,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--page-size",
        str(cfg.page_size),
        "--attention-backend",
        cfg.attention_backend,
        "--radix-eviction-policy",
        cfg.radix_eviction_policy,
    ]
    if cfg.mem_fraction_static is not None:
        args += ["--mem-fraction-static", str(cfg.mem_fraction_static)]
    if cfg.max_total_tokens is not None:
        args += ["--max-total-tokens", str(cfg.max_total_tokens)]
    if cfg.disable_radix_cache:
        args += ["--disable-radix-cache"]
    args += list(cfg.extra_args)
    return args


class SGLangServer:
    """Manage a local SGLang server process.

    Usable as a context manager::

        with SGLangServer(cfg) as server:
            server.flush_cache()
            ...
    """

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self._client = httpx.Client(base_url=cfg.base_url, timeout=30.0)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the server and block until ``/health`` returns 200."""
        if self.proc is not None:
            raise RuntimeError("server already started")
        args = build_launch_args(self.cfg)
        self.proc = subprocess.Popen(args)
        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server process exited early (code {self.proc.returncode})"
                )
            try:
                r = self._client.get("/health", timeout=5.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise TimeoutError(
            f"server not healthy within {self.cfg.startup_timeout_s}s"
        )

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        self._client.close()

    def __enter__(self) -> "SGLangServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- control endpoints -------------------------------------------------

    def flush_cache(self, timeout_s: int = 30) -> None:
        """Flush the radix cache, waiting up to ``timeout_s`` for idle."""
        r = self._client.post("/flush_cache", params={"timeout": timeout_s})
        r.raise_for_status()

    def server_info(self) -> dict[str, Any]:
        r = self._client.get("/server_info")
        r.raise_for_status()
        return r.json()

    def model_info(self) -> dict[str, Any]:
        r = self._client.get("/get_model_info")
        r.raise_for_status()
        return r.json()
