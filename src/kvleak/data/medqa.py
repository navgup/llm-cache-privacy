"""MedQA-USMLE sensitive-prefix extraction.

Builds the attacker probe set: clinically-sensitive question stems, normalized,
truncated into length buckets (short/medium/long), one probe per question so
distinct probes don't overlap as radix prefixes.

Run as a module to materialize the probe file::

    uv run python -m kvleak.data.medqa --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re

from ..config import ExperimentConfig, load_config
from .tokenize_utils import (
    bucketize,
    count_tokens,
    get_tokenizer,
    normalize,
    truncate_to_tokens,
)

# Candidate column names for the question stem across MedQA HF variants.
_QUESTION_COLUMNS = ("question", "Question", "sent1", "query")


def _question_text(row: dict) -> str | None:
    for col in _QUESTION_COLUMNS:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return None


def _build_sensitivity_regex(terms: list[str]) -> re.Pattern:
    # Word-ish boundaries; terms may contain spaces ("patient presents").
    escaped = [re.escape(t) for t in terms]
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(escaped) + r")", re.IGNORECASE)


def is_sensitive(text: str, regex: re.Pattern, tokenizer, window: int) -> bool:
    """True if a sensitive term appears within the first ``window`` tokens."""
    head = truncate_to_tokens(text, window, tokenizer)
    return regex.search(head) is not None


def extract_probes(cfg: ExperimentConfig) -> list[dict]:
    """Extract, filter, and bucket MedQA probes. Returns probe dicts."""
    from datasets import load_dataset

    dcfg = cfg.data
    tokenizer = get_tokenizer(dcfg.tokenizer)
    regex = _build_sensitivity_regex(dcfg.sensitive_terms)
    rng = random.Random(cfg.seed)

    ds = load_dataset(dcfg.medqa_dataset, split=dcfg.medqa_split)

    # Per-bucket quota: split n_probes roughly evenly across buckets.
    bucket_names = list(dcfg.buckets.keys())
    per_bucket = {b: cfg.data.n_probes // len(bucket_names) for b in bucket_names}
    # Distribute remainder to the first buckets.
    for i in range(cfg.data.n_probes % len(bucket_names)):
        per_bucket[bucket_names[i]] += 1

    filled: dict[str, list[dict]] = {b: [] for b in bucket_names}
    seen_texts: set[str] = set()

    # Shuffle row order for an unbiased draw across the split.
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        if all(len(filled[b]) >= per_bucket[b] for b in bucket_names):
            break
        raw = _question_text(ds[idx])
        if raw is None:
            continue
        text = normalize(raw)
        n_full = count_tokens(text, tokenizer)
        if n_full < min(low for low, _ in dcfg.buckets.values()):
            continue  # too short for even the smallest bucket
        if not is_sensitive(text, regex, tokenizer, dcfg.sensitivity_window):
            continue

        # Pick a bucket that still needs probes AND that this question can fill,
        # then truncate to a target length sampled within that bucket.
        candidate_buckets = [
            b
            for b in bucket_names
            if len(filled[b]) < per_bucket[b] and n_full >= dcfg.buckets[b][0]
        ]
        if not candidate_buckets:
            continue
        bucket = rng.choice(candidate_buckets)
        low, high = dcfg.buckets[bucket]
        target = rng.randint(low, min(high - 1, n_full))
        prefix = truncate_to_tokens(text, target, tokenizer)
        n_tokens = count_tokens(prefix, tokenizer)
        # Re-bucket on the exact post-truncation count (decode can shift it).
        actual_bucket = bucketize(n_tokens, dcfg.buckets)
        if actual_bucket is None or prefix in seen_texts:
            continue
        seen_texts.add(prefix)
        filled[actual_bucket].append(
            {
                "id": f"medqa-{idx}",
                "text": prefix,
                "n_tokens": n_tokens,
                "bucket": actual_bucket,
                "sensitive": True,
                "source": "medqa",
            }
        )

    probes = [p for b in bucket_names for p in filled[b]]
    return probes


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MedQA sensitive probes")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    probes = extract_probes(cfg)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.medqa_probes_file
    with open(out, "w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")

    counts: dict[str, int] = {}
    for p in probes:
        counts[p["bucket"]] = counts.get(p["bucket"], 0) + 1
    print(f"Wrote {len(probes)} probes -> {out}")
    print(f"Per-bucket counts: {counts}")


if __name__ == "__main__":
    main()
