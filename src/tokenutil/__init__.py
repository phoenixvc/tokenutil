"""phoenixvc-tokenutil — token counting and document chunking primitives.

Public API
----------
count_tokens(text_or_messages, model) -> int
    Count tokens in a string, a chat messages list, or a Responses API
    input list.  Selects the right tokenizer backend by model family.

chunk_text(text, model, chunk_size, overlap) -> list[Chunk]
    Split a document into token-bounded chunks with optional overlap.
    Splits at sentence/paragraph boundaries where possible.

Chunk
    Dataclass returned by chunk_text: text, token_count, start_byte, end_byte.

Install
-------
    pip install git+https://github.com/phoenixvc/tokenutil@v0.1.0
    pip install "git+https://github.com/phoenixvc/tokenutil@v0.1.0#egg=phoenixvc-tokenutil[gemini]"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Union

from tokenutil._backends import count_tokens_for_text
from tokenutil._messages import flatten_messages

__version__ = "0.1.0"
__all__ = ["count_tokens", "chunk_text", "Chunk"]

# ---------------------------------------------------------------------------
# Sentence / paragraph boundary pattern.
# Splits on:  ". " / "! " / "? " (end of sentence)
#             "\n\n" or longer  (paragraph break)
#             single "\n"       (line break — less preferred but used as fallback)
# ---------------------------------------------------------------------------
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A text segment produced by :func:`chunk_text`.

    Attributes:
        text:        The chunk content.
        token_count: Exact token count for this chunk (using the same backend
                     that was used to decide the split).
        start_byte:  Byte offset of the first character in the original *text*
                     argument (UTF-8).  Useful for mapping chunks back to source
                     positions in the original document.
        end_byte:    Byte offset one past the last character (exclusive).
    """

    text: str
    token_count: int
    start_byte: int
    end_byte: int


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def count_tokens(
    text_or_messages: Union[str, list[Any], None],
    model: str = "gpt-4o",
) -> int:
    """Count the number of tokens in *text_or_messages* for the given *model*.

    Args:
        text_or_messages: A plain string, a Chat Completions ``messages`` list,
            a Responses API ``input`` list, or ``None``.
        model: Model name used to select the tokenizer backend.  Defaults to
            ``"gpt-4o"`` (o200k_base).  Any sluice alias
            (``"cheap-fast"``, ``"kimi-coding"``, etc.) is accepted; unknown
            aliases fall back to a ``chars ÷ 4`` heuristic with a warning.

    Returns:
        Non-negative integer token count.  Returns ``0`` for empty / None input.
    """
    text = flatten_messages(text_or_messages)
    if not text or not text.strip():
        return 0
    return count_tokens_for_text(text, model)


# ---------------------------------------------------------------------------
# chunk_text helpers
# ---------------------------------------------------------------------------


def _split_with_positions(text: str) -> list[tuple[str, int, int]]:
    """Split *text* at sentence/paragraph boundaries.

    Returns a list of ``(segment, start, end)`` where *start* and *end* are
    character offsets into the original *text*.
    """
    segments: list[tuple[str, int, int]] = []
    pos = 0
    for match in _BOUNDARY_RE.finditer(text):
        raw = text[pos : match.start()]
        seg = raw.strip()
        if seg:
            # Locate the stripped text within raw to get the correct offset
            offset = raw.index(seg)
            abs_start = pos + offset
            segments.append((seg, abs_start, abs_start + len(seg)))
        pos = match.end()

    # Trailing segment after the last boundary
    raw = text[pos:]
    seg = raw.strip()
    if seg:
        offset = raw.index(seg)
        abs_start = pos + offset
        segments.append((seg, abs_start, abs_start + len(seg)))

    return segments


def _trim_to_overlap(
    window: list[tuple[str, int, int, int]],
    overlap: int,
    model: str,
) -> tuple[list[tuple[str, int, int, int]], int]:
    """Return the trailing elements of *window* that fit within *overlap* tokens.

    Each element is ``(text, tok, char_start, char_end)``.
    """
    if overlap <= 0 or not window:
        return [], 0
    kept: list[tuple[str, int, int, int]] = []
    kept_tok = 0
    for item in reversed(window):
        t = item[1]
        if kept_tok + t > overlap:
            break
        kept.insert(0, item)
        kept_tok += t
    return kept, kept_tok


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    model: str = "text-embedding-3-large",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Split *text* into token-bounded chunks with optional sliding overlap.

    Splits at sentence/paragraph boundaries (``". "``, ``"\\n\\n"``, ``"\\n"``)
    where possible.  Falls back to word-level splitting when a single sentence
    exceeds *chunk_size* tokens.  No heavy NLP dependencies are required.

    Args:
        text:       The source document text.
        model:      Model name for tokenizer selection.  Defaults to
                    ``"text-embedding-3-large"`` (cl100k_base).
        chunk_size: Maximum tokens per chunk.  Default ``500`` matches the
                    Mystira / OmniPost / ConvoLens production standard.
        overlap:    Token overlap between consecutive chunks.  Default ``100``.
                    Use ``0`` to disable overlap.

    Returns:
        List of :class:`Chunk` objects.  Empty list if *text* is empty.
    """
    if not text:
        return []

    raw_segments = _split_with_positions(text)
    if not raw_segments:
        return []

    # window: list of (text, token_count, char_start, char_end)
    window: list[tuple[str, int, int, int]] = []
    window_tok = 0
    chunks: list[Chunk] = []

    def _flush() -> None:
        nonlocal window, window_tok
        if not window:
            return
        chunk_str = " ".join(w[0] for w in window)
        # Byte offsets from the first and last segments in the window
        start_b = len(text[: window[0][2]].encode("utf-8"))
        end_b = len(text[: window[-1][3]].encode("utf-8"))
        chunks.append(Chunk(chunk_str, window_tok, start_b, end_b))
        window, window_tok = _trim_to_overlap(window, overlap, model)

    for seg, seg_start, seg_end in raw_segments:
        seg_tok = count_tokens_for_text(seg, model)

        # A single sentence is larger than chunk_size: split word by word
        if seg_tok > chunk_size:
            words = seg.split()
            word_pos = seg_start
            for word in words:
                w_tok = count_tokens_for_text(word, model)
                if window_tok + w_tok > chunk_size:
                    _flush()
                # Approximate word start position within original text
                found = text.find(word, word_pos)
                w_start = found if found >= 0 else word_pos
                w_end = w_start + len(word)
                window.append((word, w_tok, w_start, w_end))
                window_tok += w_tok
                word_pos = w_end
            continue

        # Normal segment: flush before appending if it would overflow
        if window_tok + seg_tok > chunk_size:
            _flush()

        window.append((seg, seg_tok, seg_start, seg_end))
        window_tok += seg_tok

    # Final flush
    if window:
        chunk_str = " ".join(w[0] for w in window)
        start_b = len(text[: window[0][2]].encode("utf-8"))
        end_b = len(text[: window[-1][3]].encode("utf-8"))
        chunks.append(Chunk(chunk_str, window_tok, start_b, end_b))

    return chunks
