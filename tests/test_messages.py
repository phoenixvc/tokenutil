"""Tests for flatten_messages()."""

from __future__ import annotations

from tokenutil._messages import flatten_messages


def test_plain_string_passthrough() -> None:
    assert flatten_messages("hello") == "hello"


def test_none_returns_empty() -> None:
    assert flatten_messages(None) == ""


def test_empty_list_returns_empty() -> None:
    assert flatten_messages([]) == ""


def test_chat_string_content() -> None:
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi!"},
    ]
    result = flatten_messages(msgs)
    assert "Be helpful." in result
    assert "Hi!" in result


def test_chat_multipart_text_only() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
    ]
    result = flatten_messages(msgs)
    assert "Describe this." in result
    assert "image_url" not in result
    assert "http" not in result


def test_responses_api_input_text_type() -> None:
    inp = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
    assert "Hello" in flatten_messages(inp)


def test_responses_api_bare_input_string() -> None:
    inp = [{"role": "user", "input": "Direct input string"}]
    assert "Direct input string" in flatten_messages(inp)


def test_ignores_non_dict_items() -> None:
    result = flatten_messages(["not a dict", 42, None])
    assert result == ""


def test_empty_content_strings_excluded() -> None:
    msgs = [{"role": "user", "content": ""}]
    assert flatten_messages(msgs) == ""


def test_multiple_messages_joined_by_newline() -> None:
    msgs = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
    ]
    result = flatten_messages(msgs)
    assert result == "First\nSecond"
