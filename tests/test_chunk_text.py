"""Tests for chunk_text()."""

from __future__ import annotations

import pytest

from tokenutil import Chunk, chunk_text, count_tokens

MODEL = "text-embedding-3-large"  # cl100k_base


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty_list() -> None:
    assert chunk_text("", MODEL) == []


def test_single_short_sentence() -> None:
    chunks = chunk_text("Hello world.", MODEL)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].token_count > 0


def test_chunk_is_dataclass() -> None:
    chunks = chunk_text("Hello world.", MODEL)
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert isinstance(c.text, str)
    assert isinstance(c.token_count, int)
    assert isinstance(c.start_byte, int)
    assert isinstance(c.end_byte, int)


# ---------------------------------------------------------------------------
# Chunk size constraint
# ---------------------------------------------------------------------------


def _make_long_text(sentences: int = 200, words_per_sentence: int = 12) -> str:
    """Generate deterministic text long enough to produce multiple chunks."""
    sentence = "The quick brown fox jumps over the lazy dog near the river bank."
    return " ".join([sentence] * sentences)


def test_all_chunks_within_size_limit() -> None:
    text = _make_long_text()
    chunks = chunk_text(text, MODEL, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 500, f"Chunk exceeds size limit: {c.token_count}"


def test_mystira_defaults_500_100() -> None:
    """Mystira/OmniPost/ConvoLens production defaults must be honoured."""
    text = _make_long_text(sentences=300)
    chunks = chunk_text(text, MODEL, chunk_size=500, overlap=100)
    assert all(c.token_count <= 500 for c in chunks)


def test_code_diff_defaults_256_32() -> None:
    """Code diff chunking (codeflow-engine convention): 256/32."""
    code = "def foo():\n    return 1\n" * 80
    chunks = chunk_text(code, MODEL, chunk_size=256, overlap=32)
    assert all(c.token_count <= 256 for c in chunks)


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_overlap_between_consecutive_chunks() -> None:
    text = _make_long_text(sentences=100)
    chunks = chunk_text(text, MODEL, chunk_size=100, overlap=20)
    if len(chunks) < 2:
        pytest.skip("Text too short to produce two chunks at this chunk_size")
    # The end of chunk[i] and start of chunk[i+1] should share some text
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].text.split())
        words_b = set(chunks[i + 1].text.split())
        assert words_a & words_b, (
            f"No overlap between chunk {i} and {i + 1}"
        )


def test_no_overlap_when_zero() -> None:
    text = _make_long_text(sentences=50)
    chunks = chunk_text(text, MODEL, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    # With no overlap the union of all chunk texts should not have duplicates
    # (check that consecutive chunks share NO words from the previous chunk's
    # trailing content — we do a weaker check: byte ranges don't overlap)
    for i in range(len(chunks) - 1):
        assert chunks[i].end_byte <= chunks[i + 1].start_byte


# ---------------------------------------------------------------------------
# Byte offsets
# ---------------------------------------------------------------------------


def test_byte_offsets_non_negative() -> None:
    chunks = chunk_text(_make_long_text(), MODEL)
    for c in chunks:
        assert c.start_byte >= 0
        assert c.end_byte > c.start_byte


def test_byte_offsets_monotonically_increasing() -> None:
    chunks = chunk_text(_make_long_text(sentences=100), MODEL, chunk_size=100, overlap=0)
    for i in range(len(chunks) - 1):
        assert chunks[i].end_byte <= chunks[i + 1].start_byte


# ---------------------------------------------------------------------------
# Oversized single sentence — word-level fallback
# ---------------------------------------------------------------------------


def test_oversized_sentence_word_split() -> None:
    """A single very-long sentence must be split at word level."""
    # Build a single sentence with ~600 tokens worth of unique words
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words) + "."
    chunks = chunk_text(text, MODEL, chunk_size=200, overlap=0)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.token_count <= 200


# ---------------------------------------------------------------------------
# token_count field accuracy
# ---------------------------------------------------------------------------


def test_token_count_matches_recount() -> None:
    """Chunk.token_count should equal count_tokens(chunk.text, model)."""
    text = _make_long_text(sentences=50)
    chunks = chunk_text(text, MODEL, chunk_size=200, overlap=50)
    for c in chunks:
        recounted = count_tokens(c.text, MODEL)
        # Allow ±2 tokens due to join-space tokenisation differences
        assert abs(c.token_count - recounted) <= 2, (
            f"token_count {c.token_count} vs recount {recounted}"
        )
