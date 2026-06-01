"""Experiment configuration: dataclasses + YAML loader.

A single ``ExperimentConfig`` drives the server launch, the data pipeline, and
the baseline experiment so every component reads the same knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# Repo root = three parents up from this file (src/kvleak/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ServerConfig:
    """SGLang launch + connection settings."""

    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    host: str = "127.0.0.1"
    port: int = 30000
    page_size: int = 1
    attention_backend: str = "flashinfer"
    # None -> let SGLang auto-size the KV pool from free memory.
    mem_fraction_static: float | None = None
    max_total_tokens: int | None = None
    # When True, launch with --disable-radix-cache (hard-uncached control run).
    disable_radix_cache: bool = False
    radix_eviction_policy: str = "lru"  # used by later experiments
    # Seconds to wait for /health after spawning the server.
    startup_timeout_s: float = 600.0
    # Extra raw args appended verbatim to the launch command.
    extra_args: list[str] = field(default_factory=list)

    # --- Docker deployment (the L4 runs SGLang in a container, not via pip) ---
    # If set, the server is launched as a Docker container; if None, launched as
    # a local `python -m sglang.launch_server` process (rarely used now).
    docker_image: str | None = "lmsysorg/sglang:v0.5.3-cu129"
    container_name: str = "sglang-kvleak"
    shm_size: str = "16g"
    # Host HuggingFace cache, bind-mounted into the container so weights persist
    # across container restarts (~ expands to the invoking user's home).
    hf_cache_dir: str = "~/.cache/huggingface"

    @property
    def base_url(self) -> str:
        # Always connect over the loopback published port from the host.
        return f"http://127.0.0.1:{self.port}"


@dataclass
class DataConfig:
    """Dataset selection, filtering, and length bucketing."""

    # Tokenizer for exact token counts. Defaults to the served model's tokenizer.
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct"

    # MedQA probe set.
    medqa_dataset: str = "GBaker/MedQA-USMLE-4-options"
    medqa_split: str = "test"
    n_probes: int = 500
    # Length buckets in tokens: [low, high). A prefix is truncated to `high - 1`
    # worth of tokens (or kept if already shorter) and assigned to the bucket it
    # falls into. See data/tokenize_utils.py:bucketize.
    buckets: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "short": (32, 64),
            "medium": (64, 128),
            "long": (128, 256),
        }
    )
    # Window (in tokens) scanned for sensitive terms ("victim prefix").
    sensitivity_window: int = 128
    sensitive_terms: list[str] = field(
        default_factory=lambda: [
            "diagnosis",
            "diagnosed",
            "medication",
            "prescribed",
            "treatment",
            "symptoms",
            "disease",
            "cancer",
            "tumor",
            "infection",
            "therapy",
            "dose",
            "mg",
            "patient presents",
            "history of",
        ]
    )

    # Background-traffic corpus (diverse real user prompts that create cache
    # pressure + concurrency jitter). LMSYS-Chat-1M is the report's choice but its
    # gated parquet trips a datasets-library load bug; Open-Orca/OpenOrca is an
    # open, diverse stand-in. The loader auto-detects the prompt field, so either
    # works — set lmsys_dataset back to lmsys/lmsys-chat-1m if access is resolved.
    lmsys_dataset: str = "Open-Orca/OpenOrca"
    lmsys_split: str = "train"
    lmsys_min_tokens: int = 32
    lmsys_max_prompts: int = 10000  # enough for the sweep (~8k) + Phase-4 replay


@dataclass
class AnalysisConfig:
    """Threshold calibration + metric reporting."""

    calib_frac: float = 0.20
    target_recall: float = 0.80  # for precision@recall


@dataclass
class BackgroundExpConfig:
    """Phase 4: rerun the probe protocol under concurrent LMSYS background load."""

    rates: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    base_mean_gap_s: float = 1.8  # observed LMSYS mean inter-arrival at 1x load
    n_probes: int = 150  # subsample of MedQA probes per rate (keeps runtime sane)
    warmup_s: float = 5.0  # let the replayer ramp before probing


@dataclass
class SweepConfig:
    """Experiment 2: eviction-policy + cache-size attack-surface sweep.

    The attack window is measured as background *volume*: how many injected LMSYS
    background prompts it takes to evict a freshly-cached victim prefix. Cache
    size (``--max-total-tokens``) and eviction policy are what control this.
    """

    fixed_length: int = 512  # past the ~259-token crossover; clean classic signal
    eviction_policies: list[str] = field(
        default_factory=lambda: ["lru", "lfu", "fifo"]
    )
    policy_cache_tokens: int = 4096  # max_total_tokens used for the policy sweep
    cache_sizes: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    size_policy: str = "lru"  # eviction policy used for the cache-size sweep
    # Background volume ladder: # of distinct background prompts injected after
    # the victim is cached, before the single attacker probe.
    volume_ladder: list[int] = field(
        default_factory=lambda: [0, 4, 8, 16, 32, 64, 128]
    )
    n_victims: int = 6  # distinct victim prefixes per (config, volume)


@dataclass
class LengthSweepConfig:
    """Controlled prefix-length sweep to locate the ΔTTFT sign crossover.

    Long prefixes are built by concatenating MedQA stems (a long sensitive
    medical context) and truncating to each ladder length.
    """

    lengths: list[int] = field(
        default_factory=lambda: [
            32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096,
        ]
    )
    n_base_texts: int = 20  # independent long contexts (content diversity)
    repeats: int = 2  # cold/warm pairs per (base_text, length)


@dataclass
class ExperimentConfig:
    seed: int = 1234
    server: ServerConfig = field(default_factory=ServerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    length_sweep: LengthSweepConfig = field(default_factory=LengthSweepConfig)
    background: BackgroundExpConfig = field(default_factory=BackgroundExpConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    # Paths (relative entries are resolved against the repo root).
    data_dir: str = "data"
    results_dir: str = "results"

    @property
    def processed_dir(self) -> Path:
        return _resolve(self.data_dir) / "processed"

    @property
    def raw_dir(self) -> Path:
        return _resolve(self.data_dir) / "raw"

    @property
    def results_path(self) -> Path:
        return _resolve(self.results_dir)

    @property
    def medqa_probes_file(self) -> Path:
        return self.processed_dir / "medqa_probes.jsonl"

    @property
    def lmsys_background_file(self) -> Path:
        return self.processed_dir / "lmsys_background.jsonl"

    @property
    def length_prefixes_file(self) -> Path:
        return self.processed_dir / "length_prefixes.jsonl"


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not isinstance(data, dict):
        return data
    kwargs: dict[str, Any] = {}
    # Resolve real types (with `from __future__ import annotations`, f.type is a str).
    type_hints = get_type_hints(cls)
    type_hints = {f.name: type_hints[f.name] for f in fields(cls)}
    for key, value in data.items():
        if key not in type_hints:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = type_hints[key]
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        elif key == "buckets" and isinstance(value, dict):
            kwargs[key] = {k: tuple(v) for k, v in value.items()}
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(ExperimentConfig, raw)


def hf_token() -> str | None:
    """Resolve the HuggingFace token from the environment (.env or shell)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
