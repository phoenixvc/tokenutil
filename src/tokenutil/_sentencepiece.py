"""Optional SentencePiece backend for Gemini-family models.

This module is imported only when a Gemini model is requested and the
``tokenutil[gemini]`` extra is installed.  The initial implementation uses a
generic SentencePieceProcessor path configured by environment variable because
Google's Gemini tokenizer model is not distributed by this package.
"""

from __future__ import annotations

import os
import warnings


def count_sp(text: str) -> int:
    """Count tokens with a SentencePiece model configured at runtime.

    Set ``TOKENUTIL_SENTENCEPIECE_MODEL`` to the local ``.model`` file.  If the
    optional dependency or model file is missing, this function raises
    ``ImportError`` so the caller can fall back to the conservative heuristic.
    """
    try:
        import sentencepiece as spm  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("sentencepiece is not installed") from exc

    model_path = os.getenv("TOKENUTIL_SENTENCEPIECE_MODEL", "")
    if not model_path:
        warnings.warn(
            "tokenutil: TOKENUTIL_SENTENCEPIECE_MODEL is not set; "
            "using fallback tokenizer for Gemini.",
            UserWarning,
            stacklevel=2,
        )
        raise ImportError("TOKENUTIL_SENTENCEPIECE_MODEL is not set")

    processor = spm.SentencePieceProcessor(model_file=model_path)
    return len(processor.encode(text, out_type=int))
