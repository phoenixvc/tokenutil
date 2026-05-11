"""Tests for count_tokens()."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from tokenutil import count_tokens


# ---------------------------------------------------------------------------
# Plain string inputs
# ---------------------------------------------------------------------------


def test_plain_string_gpt4o() -> None:
    n = count_tokens("Hello world", model="gpt-4o")
    assert n == 2


def test_plain_string_cl100k() -> None:
    n = count_tokens("Hello world", model="gpt-4")
    assert n == 2


def test_empty_string_returns_zero() -> None:
    assert count_tokens("", model="gpt-4o") == 0


def test_none_returns_zero() -> None:
    assert count_tokens(None, model="gpt-4o") == 0


def test_whitespace_only_returns_zero() -> None:
    # flatten_messages strips to "", backend returns 0
    assert count_tokens("   ", model="gpt-4o") == 0


# ---------------------------------------------------------------------------
# Chat messages list
# ---------------------------------------------------------------------------


def test_messages_list_string_content() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    n = count_tokens(messages, model="gpt-4o")
    assert n > 0


def test_messages_list_multipart_content() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
    ]
    n = count_tokens(messages, model="gpt-4o")
    # Only the text part should be counted, not the image URL
    assert n == count_tokens("What is in this image?", model="gpt-4o")


def test_messages_list_empty_returns_zero() -> None:
    assert count_tokens([], model="gpt-4o") == 0


# ---------------------------------------------------------------------------
# Responses API input list
# ---------------------------------------------------------------------------


def test_responses_api_input_text_type() -> None:
    inp = [{"role": "user", "content": [{"type": "input_text", "text": "Hello world"}]}]
    n = count_tokens(inp, model="gpt-4o")
    assert n == count_tokens("Hello world", model="gpt-4o")


# ---------------------------------------------------------------------------
# Model aliases (sluice policy aliases)
# ---------------------------------------------------------------------------


def test_sluice_alias_cheap_fast_uses_cl100k() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    n = count_tokens(text, model="cheap-fast")
    assert n == count_tokens(text, model="gpt-4")


def test_sluice_alias_groq_model() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    n = count_tokens(text, model="groq/moonshotai/kimi-k2")
    # groq keyword → cl100k_base, exact count is known
    assert n == count_tokens(text, model="gpt-4")


def test_kimi_coding_alias() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    n = count_tokens(text, model="kimi-coding")
    assert n == count_tokens(text, model="gpt-4")  # kimi → cl100k_base


def test_gemini_sentencepiece_load_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSentencePieceProcessor:
        def __init__(self, model_file: str) -> None:
            raise OSError(f"missing model: {model_file}")

    monkeypatch.setenv("TOKENUTIL_SENTENCEPIECE_MODEL", "missing.model")
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=BrokenSentencePieceProcessor),
    )

    with pytest.warns(UserWarning, match="using 4 chars/token heuristic"):
        n = count_tokens("a" * 400, model="gemini-flash")

    assert n == 100


# ---------------------------------------------------------------------------
# Unknown model — should warn and return a positive heuristic count
# ---------------------------------------------------------------------------


def test_unknown_model_warns() -> None:
    with pytest.warns(UserWarning, match="no tokenizer found"):
        n = count_tokens("Hello world", model="totally-unknown-model-xyz")
    assert n > 0


def test_unknown_model_heuristic_reasonable() -> None:
    text = "a" * 400  # 400 chars → heuristic gives 100 tokens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        n = count_tokens(text, model="totally-unknown-model-xyz")
    assert n == 100


# ---------------------------------------------------------------------------
# Consistency: same text, similar models → same count
# ---------------------------------------------------------------------------


def test_consistency_gpt4_variants() -> None:
    text = "Tokenization is important for routing."
    assert count_tokens(text, "gpt-4") == count_tokens(text, "gpt-4-turbo")
