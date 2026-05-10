"""Tokenizer backend selection by model family.

Priority order:
  1. tiktoken.encoding_for_model() — exact model match
  2. Prefix/keyword matching → cl100k_base or o200k_base
  3. Optional SentencePiece (tokenutil[gemini] extra) for Gemini-family
  4. Conservative chars ÷ 4 heuristic with a UserWarning
"""

from __future__ import annotations

import warnings
from typing import Optional

import tiktoken

# Models / prefixes that use o200k_base (gpt-4o family)
_O200K_PREFIXES = frozenset(
    [
        "gpt-4o",
        "o1",
        "o3",
        "o4",
        "chatgpt-4o",
    ]
)

# Models / keywords that map to cl100k_base
_CL100K_KEYWORDS = frozenset(
    [
        "gpt-4",
        "gpt-3.5",
        "text-embedding-3",
        "text-embedding-ada",
        # Inference aggregators hosting OpenAI-compatible open-weight models
        "groq",
        "together",
        "fireworks",
        "openrouter",
        # Moonshot / Kimi — uses its own tokenizer but cl100k_base is a close
        # approximation and accurate enough for routing threshold decisions.
        "kimi",
        "moonshot",
        # Generic open-weight model family names seen in sluice aliases
        "gpt-oss",
        "llama",
        "mistral",
        "qwen",
        "deepseek",
        "phi",
        "gemma",
    ]
)


def _tiktoken_for_model(model: str) -> Optional[tiktoken.Encoding]:
    """Return a tiktoken Encoding for *model*, or None if not applicable."""
    m = model.lower()

    # Skip Gemini — handled by optional SentencePiece backend
    if "gemini" in m:
        return None

    # Try tiktoken's own registry first (handles official OpenAI model names)
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass

    # o200k_base family
    for prefix in _O200K_PREFIXES:
        if m.startswith(prefix) or f"/{prefix}" in m:
            return tiktoken.get_encoding("o200k_base")

    # cl100k_base family (covers everything else including aggregator-hosted models)
    for kw in _CL100K_KEYWORDS:
        if kw in m:
            return tiktoken.get_encoding("cl100k_base")

    return None


def count_tokens_for_text(text: str, model: str) -> int:
    """Return token count for a plain *text* string using the best available backend.

    Falls back to the SentencePiece backend for Gemini models when the
    ``tokenutil[gemini]`` optional extra is installed, and to a conservative
    ``len(text) // 4`` heuristic for completely unknown models.
    """
    if not text:
        return 0

    enc = _tiktoken_for_model(model)
    if enc is not None:
        return len(enc.encode(text))

    # Optional SentencePiece path for Gemini
    if "gemini" in model.lower():
        try:
            from tokenutil._sentencepiece import count_sp  # type: ignore[import-not-found]

            return count_sp(text)
        except ImportError:
            pass  # fall through to heuristic

    # Heuristic fallback — warn once per model per process
    warnings.warn(
        f"tokenutil: no tokenizer found for model {model!r}; "
        "using 4 chars/token heuristic. "
        "Install tokenutil[gemini] for Gemini support.",
        UserWarning,
        stacklevel=4,
    )
    return max(1, len(text) // 4)
