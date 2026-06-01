"""LMSYS-Chat-1M background-traffic preparation.

Takes the first user message of each conversation, normalizes it, drops messages
under ``lmsys_min_tokens`` tokens, and writes a flat prompt list. This corpus is
prepared now but only *consumed* by the Phase-4 background replay harness
(kvleak.background); the clean baseline does not interleave it.

Run as a module::

    uv run python -m kvleak.data.lmsys --config configs/baseline.yaml

The dataset is gated on HuggingFace; a valid HF_TOKEN must be set.
"""

from __future__ import annotations

import argparse
import json

from ..config import ExperimentConfig, load_config
from .tokenize_utils import count_tokens, get_tokenizer, normalize


def _first_user_message(conversation) -> str | None:
    """Extract the first user turn from an LMSYS conversation field."""
    if not isinstance(conversation, list):
        return None
    for turn in conversation:
        if isinstance(turn, dict) and turn.get("role") == "user":
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def extract_background(cfg: ExperimentConfig) -> list[dict]:
    from datasets import load_dataset

    dcfg = cfg.data
    tokenizer = get_tokenizer(dcfg.tokenizer)

    # Stream to avoid materializing all of LMSYS-Chat-1M (~1M convos) on disk.
    ds = load_dataset(dcfg.lmsys_dataset, split=dcfg.lmsys_split, streaming=True)

    prompts: list[dict] = []
    for i, row in enumerate(ds):
        if len(prompts) >= dcfg.lmsys_max_prompts:
            break
        msg = _first_user_message(row.get("conversation"))
        if msg is None:
            continue
        text = normalize(msg)
        n_tokens = count_tokens(text, tokenizer)
        if n_tokens < dcfg.lmsys_min_tokens:
            continue
        prompts.append(
            {
                "id": f"lmsys-{i}",
                "text": text,
                "n_tokens": n_tokens,
                "source": "lmsys",
            }
        )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LMSYS background prompts")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = extract_background(cfg)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.lmsys_background_file
    with open(out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(prompts)} background prompts -> {out}")


if __name__ == "__main__":
    main()
