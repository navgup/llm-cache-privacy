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

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


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

    # LMSYS background corpus (prepared now; consumed by Phase-4 replay later).
    lmsys_dataset: str = "lmsys/lmsys-chat-1m"
    lmsys_split: str = "train"
    lmsys_min_tokens: int = 32
    lmsys_max_prompts: int = 50000  # cap how many to materialize locally


@dataclass
class AnalysisConfig:
    """Threshold calibration + metric reporting."""

    calib_frac: float = 0.20
    target_recall: float = 0.80  # for precision@recall


@dataclass
class ExperimentConfig:
    seed: int = 1234
    server: ServerConfig = field(default_factory=ServerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

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
