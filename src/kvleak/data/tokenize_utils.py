"""Tokenization + normalization helpers for prefix preparation.

The tokenizer is loaded lazily (and cached) so importing this module never
forces a HuggingFace download. The Llama-3.1 tokenizer is gated, so a valid
``HF_TOKEN`` must be in the environment when ``get_tokenizer`` first runs with
the default model. Tests pass an open tokenizer (e.g. ``gpt2``) to avoid that.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from ..config import hf_token


@lru_cache(maxsize=4)
def get_tokenizer(name: str):
    """Load and cache a HuggingFace tokenizer by name."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, token=hf_token())


def normalize(text: str) -> str:
    """NFC-normalize unicode and strip trailing whitespace.

    Done so cosmetic formatting differences don't perturb the token count or
    the prefix bytes SGLang hashes for cache matching.
    """
    return unicodedata.normalize("NFC", text).rstrip()


def count_tokens(text: str, tokenizer) -> int:
    """Exact token count (no special tokens added)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, n: int, tokenizer) -> str:
    """Truncate ``text`` to at most ``n`` tokens, decoding back to a string.

    Re-normalized after decoding so the returned text round-trips to the same
    token count we report.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return normalize(tokenizer.decode(ids))


def bucketize(n_tokens: int, buckets: dict[str, tuple[int, int]]) -> str | None:
    """Return the bucket name whose ``[low, high)`` range contains ``n_tokens``.

    Returns ``None`` if the count falls outside every bucket.
    """
    for name, (low, high) in buckets.items():
        if low <= n_tokens < high:
            return name
    return None
