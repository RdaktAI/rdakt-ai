---
name: rdakt-codebase
description: >
  Rdakt AI client codebase architecture and internals. Use when working on the
  middleware, streaming pipeline, provider integrations, session/anonymization
  logic, or when debugging issues with PII detection, deanonymization, SSE
  streaming, or gzip handling.
user-invocable: true
---

# Rdakt AI — Client Codebase Architecture & Internals

## Project Structure

```
rdakt_ai/
├── __init__.py            # Public API surface and exports
├── middleware.py          # RdaktMiddleware (async) + RdaktSyncMiddleware — httpx transports
├── session.py             # RdaktSession — entity map, deanonymize(), streaming deanonymize_chunk()
├── streaming.py           # AsyncDeanonymizingStream, gzip decompression, SSE chunk reassembly
├── formats.py             # Provider-agnostic SSE content extraction (OpenAI/Anthropic/Gemini)
├── pipeline.py            # Detection pipeline — RdaktStage chain, overlap resolution
├── detectors/
│   └── regex.py           # RegexDetector — EMAIL, SSN, CREDIT_CARD, IBAN, JWT, etc.
├── anonymizer.py          # Token replacement, synthetic substitution (Faker), hybrid
├── config.py              # RdaktConfig from rdakt.yaml — validated mode/on_error
├── stores.py              # SessionStore ABC, thread-safe MemoryStore
├── models.py              # Entity, RdaktContext dataclasses
├── logging.py             # Structured JSON logging
└── cli.py                 # CLI: rdakt-ai init, rdakt-ai demo

tests/
├── test_middleware.py             # Async middleware (passthrough, anonymize, deanonymize, audit, fail-open)
├── test_middleware_sync.py        # Sync middleware (same coverage, sync paths)
├── test_middleware_streaming.py   # SSE streaming deanonymization, HTTP chunk splitting, gzip
├── test_middleware_callbacks.py   # Event callback firing and error isolation
├── test_session.py                # Session mapping, deanonymize, streaming chunks, escaped tokens
├── test_formats.py                # Provider-agnostic SSE extraction (OpenAI/Anthropic/Gemini)
├── test_anonymizer.py             # Token/synthetic/hybrid strategies, determinism
├── test_pipeline.py               # Overlap resolution
├── test_pipeline_chain.py         # Stage chaining, async/sync
├── test_detectors/test_regex.py   # Built-in + custom pattern detection
├── test_config.py                 # YAML loading, validation
├── test_stores.py                 # MemoryStore CRUD
├── test_integration.py            # End-to-end pipeline without network
├── test_init.py                   # Package exports
├── test_models.py                 # Data model construction
├── test_logging.py                # JSON formatter
├── test_cli.py                    # CLI init + demo commands
└── test_examples.py               # Examples in examples/ run without error
```

## Core Data Flow

```
Request:   SDK → RdaktMiddleware (httpx transport) → detect PII → anonymize → LLM
Response:  LLM → SSE stream → AsyncDeanonymizingStream → deanonymize tokens → SDK
```

Detailed:

```
User prompt → SDK builds HTTP request
  → RdaktMiddleware.handle_async_request()
    → Parse JSON body
    → _extract_text_refs(): text in messages[].content OR contents[].parts[].text
    → pipeline.detect(): regex → Entity list
    → anonymizer.anonymize(): replace PII with <TYPE_N> tokens or synthetic values
    → _rebuild_request(): update body + Content-Length
  → Forward to inner transport (real HTTP)
  → LLM returns response
    → SSE stream: wrap with AsyncDeanonymizingStream
    → Non-streaming: deanonymize_structured() on full JSON body
  → SDK receives response with real values restored
```

## Middleware (`middleware.py`)

### Transport Chain

The middleware implements `httpx.AsyncBaseTransport`. SDKs inject it as:

```python
client = AsyncOpenAI(http_client=httpx.AsyncClient(transport=middleware))
```

### Multi-Provider Request Formats

`_TextRef` + `_extract_text_refs()` abstract over provider wire differences:

| Provider | Text location |
|----------|---------------|
| OpenAI / Anthropic | `messages[].content` |
| Gemini | `contents[].parts[].text` |

`_TextRef` holds a mutable reference — setting `.text` modifies the original dict in-place.

### Active vs Audit Mode

- **active** (default): detect → anonymize → forward anonymized → deanonymize response
- **audit**: detect → log entities → forward original unanonymized

### Error Policy

- `warn_and_forward` (default): catch, log, forward unanonymized
- `block`: re-raise — SDK call fails

Validated at construction (`RdaktConfig.__post_init__`).

### Callbacks

- `on_entities_detected(entities, text)` — after detection
- `on_anonymized(original, anonymized, mapping)` — after anonymization
- `on_deanonymized(anon_response, restored_response)` — non-streaming only
- `on_error(exception, phase)` — on fail-open

All wrapped in `_fire_callback()` — exceptions caught and logged with callback name.

## Session (`session.py`)

Bidirectional entity mapping:

- `_entity_map`: `{"<EMAIL_1>": "john@example.com"}`
- `_reverse_map`: `{"john@example.com": "<EMAIL_1>"}`
- Public: `session.entity_map` (copy), `session.has_mappings` (bool)

### Deanonymization Regexes

```python
_TOKEN_RE = re.compile(r"\\?<([A-Z][A-Z0-9_]*)\\?>")     # <TOKEN> and \<TOKEN\>
_PARTIAL_RE = re.compile(r"(?:\\?<(?:[A-Z][A-Z0-9_]*\\?)?|\\)\Z")  # Trailing partial
```

- `deanonymize(text)` — single-pass regex replace on complete text
- `deanonymize_chunk(chunk)` — stateful streaming: buffer → find tokens → emit fragments → keep partial
- `flush()` — emit remaining buffer at stream end
- `MAX_BUFFER = 256` — safety valve against unbounded buffering

## Streaming Pipeline (`streaming.py`)

Three layers of complexity:

1. **HTTP chunked transfer** splits SSE `data:` lines across chunks
2. **Gzip** (Anthropic) requires decompression before parsing
3. **Token splitting** — `<EMAIL_1>` can arrive as `<EM` + `AIL_1>` across events

### Processing Chain

```
Raw bytes → _AsyncGunzipStream (if gzip) → AsyncDeanonymizingStream
  → _process_sse_chunk():
      prepend state["leftover"] → split lines → buffer incomplete trailing line
      → _process_data_lines(): parse JSON → _deanonymize_sse_content()
        → extract_sse_content() → session.deanonymize_chunk() → set_sse_content()
  → flush() at stream end
```

### SSE Content Extraction (`formats.py`)

| Function | OpenAI | Anthropic | Gemini |
|----------|--------|-----------|--------|
| `extract_sse_content()` | `choices[].delta.content` | `delta.text` (content_block_delta only) | `candidates[].content.parts[].text` |
| `set_sse_content()` | same paths | same | same |
| `extract_response_content()` | `choices[0].message.content` | `content[0].text` | `candidates[0].content.parts[0].text` |

Only `content_block_delta` events carry text for Anthropic — other event types return `None`, preventing non-content fields from corrupting the chunk buffer.

### Gzip

`_AsyncGunzipStream` / `_SyncGunzipStream`:
- `zlib.decompressobj(zlib.MAX_WBITS | 16)` for gzip
- `zlib.error` caught per-chunk, falls back to raw bytes
- Middleware strips `content-encoding` + `content-length` from wrapped response

## Detection (`pipeline.py`, `detectors/regex.py`)

Composable `RdaktStage` chain with `call_next` pattern. After all stages: `resolve_overlaps()` keeps longest span (priority tiebreaker).

Built-in patterns: `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, `API_KEY`, `AWS_KEY`, `IP_ADDRESS`, `IBAN`, `JWT`.

Custom patterns via config:
```yaml
entities:
  EMPLOYEE_ID:
    pattern: "EMP-\\d{6}"
    strategy: token
```

## Anonymizer (`anonymizer.py`)

| Strategy | Output | Use case |
|----------|--------|----------|
| `token` | `<TYPE_N>` | Structured data (emails, SSNs) |
| `synthetic` | Faker value | Names, orgs |
| `hybrid` | Synthetic for PERSON/ORG, token for rest | Default |

Same real value → same token within a session. Synthetic values are deterministic (seeded Faker: session_id + value hash).

## Common Pitfalls

**Tokens in streaming output**: Check `session.has_mappings` is true → `_wrap_streaming_response()` fires. Check `state["leftover"]` handles split SSE lines. Check gzip decompression for Anthropic.

**Gemini not anonymized**: Wire format uses `contents[].parts[].text` not `messages[].content`. `_extract_text_refs()` handles both.

**Gzip errors**: Anthropic gzip SSE. Ensure `_AsyncGunzipStream` wraps stream, headers stripped.

**Config validation**: `mode` must be `"active"` or `"audit"`. `on_error` must be `"warn_and_forward"` or `"block"`. Constants: `VALID_MODES`, `VALID_ERROR_POLICIES`.

## Using the Client With the Rdakt AI Gateway

When the user is going through the hosted gateway (`gateway.rdakt.ai` in prod, `localhost:8001` in dev), they do **not** instantiate `RdaktMiddleware` at all. Anonymization, deanonymization, tracing, and BYOK key decryption all happen on the gateway side, which itself runs this same library server-side.

### Client-side config (gateway mode)

The gateway exposes a per-provider proxy at `/v1/{provider}/{path:path}`. The client retargets the SDK's `base_url` and authenticates with a Bearer `rk_<prefix>.<tail>` API key from the dashboard.

| Provider | SDK option | Value |
|---|---|---|
| OpenAI | `base_url=` | `https://gateway.rdakt.ai/v1/openai` |
| Anthropic | `base_url=` | `https://gateway.rdakt.ai/v1/anthropic` |
| Gemini | `http_options={"base_url": ...}` | `https://gateway.rdakt.ai/v1/gemini` |

The OpenAI base URL deliberately stops at `/v1/openai` (no trailing `/v1`) because the OpenAI SDK appends paths like `/chat/completions` directly onto whatever `base_url` you give it. Anthropic's SDK adds its own `/v1/messages` to the path, so the same prefix works.

### What changes on the client

- **No `RdaktMiddleware`** — drop the httpx transport entirely. The local middleware is only for direct-to-provider setups.
- **No `rdakt.yaml`** — detection rules and per-entity strategies come from the **project policy** stored on the gateway. The dashboard's `Project → Policy` page is the source of truth.
- **No provider secrets** in the SDK call — the BYOK upstream key is uploaded once via the dashboard and decrypted server-side per request. The client only ever holds the `rk_` key.
- **Streaming, multi-turn consistency, audit mode** — all still work, because the gateway is running `RdaktMiddleware` for you. Tokens are stable per `session_key`, configured server-side per project.
- **`rk_` keys authorize `/v1/*` only.** They do not work against `/api/v1/*` (dashboard / control plane), which requires a session cookie + CSRF.

### When to keep using the local middleware

The local `RdaktMiddleware` / `RdaktSyncMiddleware` is the right answer when:

- The user can't / won't route traffic through the gateway (air-gapped, self-hosted, on-prem).
- They want PII redaction without a hosted dependency.
- They need to inspect or extend detection logic in-process (custom detectors, callbacks, programmatic config).

If the user is on the gateway, recommendations like "wrap the client in `RdaktMiddleware`" are wrong — they'd double-anonymize (client tokens → gateway sees pre-anonymized text → no real PII to detect). Lead with the `base_url` swap instead.

## Testing Patterns

- **Middleware**: Mock with `FakeTransport`. Both async + sync paths.
- **Streaming**: Build raw SSE bytes, test via `_process_sse_chunk()` or full stream. Break bytes at arbitrary boundaries for chunk-splitting tests.
- **Session**: Direct `deanonymize_chunk()` with split tokens. Test `MAX_BUFFER` safety valve.
- **Formats**: Each provider's SSE format in `test_formats.py`.
- **Integration**: Full pipeline (detect → anonymize → session → deanonymize) without network.
