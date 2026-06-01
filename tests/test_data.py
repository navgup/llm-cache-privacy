"""Data-pipeline logic tests with a fake whitespace tokenizer (no network/GPU).

These exercise normalization, token counting/truncation, bucketing, and the
MedQA sensitivity filter without downloading the gated Llama tokenizer.
"""

from __future__ import annotations

from kvleak.data import medqa
from kvleak.data.tokenize_utils import (
    bucketize,
    count_tokens,
    normalize,
    truncate_to_tokens,
)


class FakeTokenizer:
    """Whitespace tokenizer: one 'token' per word. Good enough for logic tests."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_normalize_strips_trailing_ws():
    assert normalize("hello world   \n") == "hello world"


def test_count_and_truncate():
    tok = FakeTokenizer()
    text = "one two three four five"
    assert count_tokens(text, tok) == 5
    assert truncate_to_tokens(text, 3, tok) == "one two three"
    # Truncating beyond length is a no-op.
    assert count_tokens(truncate_to_tokens(text, 99, tok), tok) == 5


def test_bucketize_ranges():
    buckets = {"short": (32, 64), "medium": (64, 128), "long": (128, 256)}
    assert bucketize(32, buckets) == "short"
    assert bucketize(63, buckets) == "short"
    assert bucketize(64, buckets) == "medium"
    assert bucketize(200, buckets) == "long"
    assert bucketize(10, buckets) is None  # below all buckets
    assert bucketize(256, buckets) is None  # at/above the top edge


def test_sensitivity_filter_matches_terms_in_window():
    tok = FakeTokenizer()
    regex = medqa._build_sensitivity_regex(["diagnosis", "medication"])
    sensitive = "The patient received a diagnosis of pneumonia after admission"
    bland = "The capital city has many tall buildings and busy streets today"
    assert medqa.is_sensitive(sensitive, regex, tok, window=128)
    assert not medqa.is_sensitive(bland, regex, tok, window=128)


def test_sensitivity_filter_respects_window():
    tok = FakeTokenizer()
    regex = medqa._build_sensitivity_regex(["diagnosis"])
    # 'diagnosis' sits past a 3-token window -> should not match.
    text = "aaa bbb ccc diagnosis ddd"
    assert not medqa.is_sensitive(text, regex, tok, window=3)
    assert medqa.is_sensitive(text, regex, tok, window=10)


def test_question_text_extraction():
    assert medqa._question_text({"question": "stem here"}) == "stem here"
    assert medqa._question_text({"Question": "alt col"}) == "alt col"
    assert medqa._question_text({"answer": "no stem"}) is None
