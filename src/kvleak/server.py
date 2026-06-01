"""SGLang server lifecycle manager.

Spawns ``python -m sglang.launch_server`` from an :class:`ServerConfig`, waits
for ``/health``, and exposes the control endpoints the experiment needs
(``/flush_cache``, ``/server_info``). This only meaningfully runs on the GPU VM;
on a CPU-only box importing it is safe but ``start()`` will fail when SGLang
tries to load the model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .config import ServerConfig


def sglang_args(cfg: ServerConfig, host: str) -> list[str]:
    """The ``sglang.launch_server`` flags (single source of truth for both
    local and Docker launches). ``host`` differs: 127.0.0.1 for a local
    process, 0.0.0.0 inside a container so the published port is reachable."""
    args = [
        "--model-path",
        cfg.model_path,
        "--dtype",
        cfg.dtype,
        "--host",
        host,
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


def build_local_command(cfg: ServerConfig) -> list[str]:
    """Full argv to launch SGLang as a local Python process."""
    return [sys.executable, "-m", "sglang.launch_server", *sglang_args(cfg, cfg.host)]


def build_docker_command(cfg: ServerConfig) -> list[str]:
    """Full argv (starting with ``docker``) to launch SGLang in a container.

    Binds the host HF cache so weights persist, publishes the port on loopback,
    and passes HF_TOKEN through from the environment (no token in the argv).
    """
    hf_cache = str(Path(os.path.expanduser(cfg.hf_cache_dir)))
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        cfg.container_name,
        "--shm-size",
        cfg.shm_size,
        "-p",
        f"127.0.0.1:{cfg.port}:{cfg.port}",
        "-v",
        f"{hf_cache}:/root/.cache/huggingface",
        "-e",
        "HF_TOKEN",  # value inherited from the host environment
        cfg.docker_image,
        "python3",
        "-m",
        "sglang.launch_server",
        *sglang_args(cfg, "0.0.0.0"),
    ]
    return cmd


def build_launch_command(cfg: ServerConfig) -> list[str]:
    """Pick Docker or local launch based on config."""
    return build_docker_command(cfg) if cfg.docker_image else build_local_command(cfg)


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
        args = build_launch_command(self.cfg)
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
        # For a container, `docker stop` is the reliable teardown (`--rm` cleans
        # it up); terminating the `docker run` client alone can orphan it.
        if self.cfg.docker_image is not None:
            subprocess.run(
                ["docker", "stop", self.cfg.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
        """Best-effort server info. The endpoint name varies across SGLang
        versions (and may be absent), so this never raises — returns {} if no
        known endpoint responds 200."""
        for path in ("/get_server_info", "/server_info"):
            try:
                r = self._client.get(path)
                if r.status_code == 200:
                    return r.json()
            except httpx.HTTPError:
                pass
        return {}

    def model_info(self) -> dict[str, Any]:
        r = self._client.get("/get_model_info")
        r.raise_for_status()
        return r.json()
