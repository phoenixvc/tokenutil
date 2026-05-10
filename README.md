# phoenixvc-tokenutil

Token counting and document chunking primitives for phoenixvc LLM workloads.

## Install

```bash
pip install git+https://github.com/phoenixvc/tokenutil@v0.1.0
# Gemini SentencePiece support (optional):
pip install "git+https://github.com/phoenixvc/tokenutil@v0.1.0#egg=phoenixvc-tokenutil[gemini]"
```

## Public API

### `count_tokens(text_or_messages, model) -> int`

Count tokens in a string, a Chat Completions `messages` list, or a Responses API `input` list.

```python
from tokenutil import count_tokens

# Plain string
count_tokens("Hello world", model="gpt-4o")          # → 2

# Chat messages list
count_tokens(
    [{"role": "user", "content": "Hello world"}],
    model="gpt-4o",
)  # → 2

# Sluice policy aliases
count_tokens("some text", model="kimi-coding")        # uses cl100k_base
count_tokens("some text", model="groq/openai/gpt-oss-20b")  # uses cl100k_base
```

Backends selected by model family:

| Model family | Encoding |
|---|---|
| `gpt-4o`, `o1`, `o3`, `o4` | `o200k_base` |
| `gpt-4`, `gpt-3.5`, `text-embedding-3*`, `groq/*`, `kimi*`, `moonshot*`, `llama*`, `mistral*`, … | `cl100k_base` |
| `gemini*` | SentencePiece (`tokenutil[gemini]`) or `chars ÷ 4` fallback |
| Unknown | `chars ÷ 4` heuristic + `UserWarning` |

### `chunk_text(text, model, chunk_size, overlap) -> list[Chunk]`

Split a document into token-bounded segments with sliding overlap. Splits at sentence/paragraph boundaries; falls back to word-level splitting for oversized sentences.

```python
from tokenutil import chunk_text

# Mystira / OmniPost / ConvoLens production defaults
chunks = chunk_text(document, model="text-embedding-3-large", chunk_size=500, overlap=100)

# codeflow-engine code-diff convention
chunks = chunk_text(diff, model="text-embedding-3-large", chunk_size=256, overlap=32)

for c in chunks:
    print(c.token_count, c.start_byte, c.end_byte, c.text[:60])
```

`Chunk` fields:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Chunk content |
| `token_count` | `int` | Tokens in this chunk |
| `start_byte` | `int` | UTF-8 byte offset of first char in original `text` |
| `end_byte` | `int` | UTF-8 byte offset one past the last char (exclusive) |

## Chunking defaults across phoenixvc repos

See [ADR 12](https://github.com/phoenixvc/sluice/blob/main/docs/architecture/12-tokenisation-conventions.md) for the full cross-language conventions document.

| Use case | `chunk_size` | `overlap` |
|---|---|---|
| Narrative / conversation (Mystira, OmniPost, ConvoLens) | 500 | 100 |
| Code diffs (codeflow-engine) | 256 | 32 |

## Development

```bash
pip install -e ".[dev]"
mypy src/tokenutil
pytest tests/ -v
```

## Related

- [phoenixvc/sluice](https://github.com/phoenixvc/sluice) — AI gateway (consumer of `count_tokens`)
- [ADR 11](https://github.com/phoenixvc/sluice/blob/main/docs/architecture/11-multi-provider-routing.md) — Multi-provider routing
- [ADR 12](https://github.com/phoenixvc/sluice/blob/main/docs/architecture/12-tokenisation-conventions.md) — Tokenisation conventions
