"""Build long medical-context prefixes for the length-crossover sweep.

MedQA question stems are only ~100-250 tokens, too short to study prefix lengths
up to a few thousand tokens. We concatenate distinct stems into longer "medical
context" base texts (a stand-in for a long sensitive RAG / system-prompt prefix),
then truncate each base text to every length in the configured ladder.

Each base text uses a DISJOINT set of stems so the base texts are independent.
Different lengths of the *same* base text do overlap as radix prefixes, but the
sweep flushes the cache before every measurement, so a "cold" probe is always
genuinely cold regardless of overlap.

Run as a module::

    uv run python -m kvleak.data.long_prefixes --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random

from ..config import ExperimentConfig, load_config
from .medqa import _question_text
from .tokenize_utils import count_tokens, get_tokenizer, normalize, truncate_to_tokens

SEP = "\n\n"


def build_length_prefixes(cfg: ExperimentConfig) -> list[dict]:
    from datasets import load_dataset

    dcfg = cfg.data
    scfg = cfg.length_sweep
    tokenizer = get_tokenizer(dcfg.tokenizer)
    rng = random.Random(cfg.seed)
    max_len = max(scfg.lengths)

    ds = load_dataset(dcfg.medqa_dataset, split=dcfg.medqa_split)
    stems: list[str] = []
    for row in ds:
        raw = _question_text(row)
        if raw:
            stems.append(normalize(raw))
    rng.shuffle(stems)

    rows: list[dict] = []
    cursor = 0  # consume stems disjointly across base texts
    for b in range(scfg.n_base_texts):
        # Accumulate distinct stems until the base text clears the max length.
        parts: list[str] = []
        tok_total = 0
        while tok_total < max_len + 32:  # small margin over the longest bucket
            if cursor >= len(stems):
                rng.shuffle(stems)  # pool exhausted: reshuffle (rare)
                cursor = 0
            stem = stems[cursor]
            cursor += 1
            parts.append(stem)
            tok_total += count_tokens(stem, tokenizer) + 2  # ~SEP tokens
        base_text = SEP.join(parts)
        if count_tokens(base_text, tokenizer) < max_len:
            continue  # safety; should not happen given the margin
        for length in scfg.lengths:
            prefix = truncate_to_tokens(base_text, length, tokenizer)
            n_tokens = count_tokens(prefix, tokenizer)
            rows.append(
                {
                    "base_id": f"ctx-{b}",
                    "length": length,
                    "n_tokens": n_tokens,
                    "text": prefix,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build long-prefix length ladder")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = build_length_prefixes(cfg)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.length_prefixes_file
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_len: dict[int, int] = {}
    for r in rows:
        by_len[r["length"]] = by_len.get(r["length"], 0) + 1
    print(f"Wrote {len(rows)} length-prefixes -> {out}")
    print(f"Lengths (target -> count): {dict(sorted(by_len.items()))}")


if __name__ == "__main__":
    main()
