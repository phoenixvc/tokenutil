"""Flatten OpenAI-style message payloads to a single plain-text string.

Handles three input shapes:
  - Plain ``str``
  - Chat Completions API: ``list[{"role": ..., "content": str | list[part]}]``
  - Responses API: ``list[{"role": ..., "content": str | list[part]}]``
    where parts may have ``type`` of ``"text"`` or ``"input_text"``

Only text parts are counted; image URLs, tool calls, and binary blobs are
deliberately excluded because they don't meaningfully predict context-window
consumption in the models sluice routes.
"""

from __future__ import annotations

from typing import Any


def flatten_messages(data: Any) -> str:  # noqa: ANN401
    """Return a single string containing all countable text from *data*.

    Args:
        data: A ``str``, a messages/input ``list``, or ``None``.

    Returns:
        Concatenated text joined by newlines, or ``""`` for empty/None input.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if not isinstance(data, list):
        # Unexpected shape — convert to string as best-effort
        return str(data)

    parts: list[str] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        content = item.get("content")

        if isinstance(content, str):
            if content:
                parts.append(content)

        elif isinstance(content, list):
            # Multi-part content blocks (vision, tool results, etc.)
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type in ("text", "input_text"):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)

        # Responses API top-level "input" field (bare string variant)
        input_val = item.get("input")
        if isinstance(input_val, str) and input_val:
            parts.append(input_val)

    return "\n".join(parts)
