# Rdakt AI

Composable anonymization middleware for LLM interactions. Rdakt AI sits between your application and the LLM provider as an `httpx` transport: it detects sensitive data in outbound requests, anonymizes it with tokens or synthetic values before the payload leaves your process, and de-anonymizes the response — including streaming responses — transparently on the way back.

Works out of the box with the official **OpenAI**, **Anthropic**, and **Google Gemini** SDKs (sync and async), plus first-class integrations for **LangChain** and **Pydantic AI** agent frameworks. Streaming responses, multi-turn token consistency, custom regex patterns, and pluggable session stores (memory, SQLite, Redis) are all supported — and a hosted gateway is available if you'd rather not run the middleware in-process.

## Installation

```bash
pip install rdakt-ai
```

Or with `uv`:

```bash
uv add rdakt-ai
```

Optional extras: `redis` (Redis session store), `ner` (spaCy NER), `langchain`, `pydantic-ai`, `cli` (scaffolding + offline demo command), and `all` for everything.

```bash
pip install "rdakt-ai[redis]"
pip install "rdakt-ai[all]"
```

### CLI

Install the `cli` extra to get the `rdakt-ai` command:

```bash
pip install "rdakt-ai[cli]"
rdakt-ai demo
rdakt-ai init
```

If you're developing against a local checkout, `uv run rdakt-ai demo` resolves the CLI dependencies against the project's virtual environment.

## Quick Start

### OpenAI

```python
import httpx
from openai import AsyncOpenAI
from rdakt_ai import RdaktMiddleware

client = AsyncOpenAI(
    http_client=httpx.AsyncClient(
        transport=RdaktMiddleware(inner=httpx.AsyncHTTPTransport()),
    ),
)

# PII in your prompt is automatically anonymized before reaching OpenAI,
# and de-anonymized in the response.
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize the case for John Smith (SSN 123-45-6789)"}],
)
```

### Anthropic

```python
import httpx
from anthropic import AsyncAnthropic
from rdakt_ai import RdaktMiddleware

client = AsyncAnthropic(
    http_client=httpx.AsyncClient(
        transport=RdaktMiddleware(inner=httpx.AsyncHTTPTransport()),
    ),
)

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize the case for John Smith (SSN 123-45-6789)"}],
)
```

### Sync Usage

```python
import httpx
from openai import OpenAI
from rdakt_ai import RdaktSyncMiddleware

client = OpenAI(
    http_client=httpx.Client(
        transport=RdaktSyncMiddleware(inner=httpx.HTTPTransport()),
    ),
)
```

### Google Gemini

```python
import httpx
from google import genai
from rdakt_ai import RdaktMiddleware

client = genai.Client(
    http_options={"transport": RdaktMiddleware(inner=httpx.AsyncHTTPTransport())},
)

response = await client.aio.models.generate_content(
    model="gemini-2.0-flash",
    contents="Summarize the case for John Smith (SSN 123-45-6789)",
)
```

## Using via the Rdakt AI Gateway

If you're using the hosted Rdakt AI Gateway, **don't install `RdaktMiddleware`** — anonymization, deanonymization, and tracing run server-side. Just point your provider SDK at the gateway and authenticate with an `rk_*` API key issued from the dashboard.

The gateway exposes a per-provider proxy at `/v1/{provider}/...`. Production base URL is `https://gateway.rdakt.ai`; local dev is `http://localhost:8001`.

### OpenAI

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://gateway.rdakt.ai/v1/openai",
    api_key="rk_<prefix>.<tail>",  # issued in dashboard → API keys
)

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize the case for John Smith (SSN 123-45-6789)"}],
)
```

### Anthropic

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    base_url="https://gateway.rdakt.ai/v1/anthropic",
    api_key="rk_<prefix>.<tail>",
)
```

### Google Gemini

```python
from google import genai

client = genai.Client(
    api_key="rk_<prefix>.<tail>",
    http_options={"base_url": "https://gateway.rdakt.ai/v1/gemini"},
)
```

### Notes

- The `rk_*` key authorizes the data plane only (`/v1/*`). It does **not** grant access to the dashboard / control-plane API.
- Provider credentials (BYOK) are uploaded once to the gateway from the dashboard and decrypted per-request. Don't ship provider secrets from the client.
- Detection rules and anonymization strategies come from the **project policy** on the gateway, not from a local `rdakt.yaml`. Configure them in the dashboard under `Project → Policy`.
- Streaming, multi-turn token consistency, and audit mode all work the same way — the gateway runs `RdaktMiddleware` on your behalf.
- Use the local middleware (`RdaktMiddleware` / `RdaktSyncMiddleware`) only if you're **not** going through the gateway (e.g. air-gapped deployments or self-hosted setups that talk to providers directly).

## Configuration

Generate a config file:

```bash
uv run rdakt-ai init
```

This creates `rdakt.yaml`:

```yaml
detection:
  layers:
    - regex

entities:
  PERSON:
    strategy: synthetic
  EMAIL:
    strategy: token
  SSN:
    strategy: token
  CREDIT_CARD:
    strategy: token
  API_KEY:
    strategy: token

session:
  store: memory

on_error: warn_and_forward
mode: active  # or "audit" for dry-run
```

### Custom Patterns

```yaml
entities:
  ACCOUNT_NUMBER:
    pattern: '\d{4}-\d{4}'
    strategy: token
```

## How It Works

1. **Detect** — Regex patterns identify PII (emails, SSNs, credit cards, API keys, etc.)
2. **Anonymize** — Entities are replaced with tokens (`<EMAIL_1>`) or synthetic values (fake names via Faker)
3. **Forward** — Anonymized request is sent to the LLM provider
4. **De-anonymize** — Response tokens are mapped back to original values

### Anonymization Strategies

| Strategy | Used For | Example |
|----------|----------|---------|
| `token` | Structured data (emails, SSNs, IPs) | `john@example.com` -> `<EMAIL_1>` |
| `synthetic` | Names, organizations, locations | `John Smith` -> `Maria Garcia` |
| `hybrid` (default) | Best of both | Synthetic for names, tokens for structured data |

## Streaming Support

Streaming responses (`stream=True`) are automatically deanonymized chunk-by-chunk. SSE events are parsed, content fields are restored, and structural fields are left untouched.

## Multi-Turn Conversations

Token mappings are consistent within a session scope. The same PII value always gets the same token, and different values never collide:

```python
# Turn 1: "Email john@example.com" → "Email <EMAIL_1>"
# Turn 2: "Also contact jane@example.com" → "Also contact <EMAIL_2>"
# Turn 3: "Remind john@example.com" → "Remind <EMAIL_1>"  (reused, not <EMAIL_3>)
```

The scope is controlled by `session_key` — set it to a conversation ID, user ID, or any identifier that defines the boundary:

```python
# Per-conversation (default playground behavior)
middleware = RdaktMiddleware(inner=..., session_key=conversation_id)

# Per-user — all conversations for a user share token mappings
middleware = RdaktMiddleware(inner=..., session_key=user_id)
```

Token mappings are persisted via the session store. Use `MemoryStore` (default) for ephemeral sessions, `SQLiteStore` for local persistence, or `RedisStore` for distributed setups:

```python
from rdakt_ai import RdaktMiddleware, SQLiteStore

store = SQLiteStore("sessions.db")
middleware = RdaktMiddleware(inner=..., store=store, session_key="user-123")
```

## Event Callbacks

Hook into the detection/anonymization lifecycle for monitoring and debugging:

```python
middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    on_entities_detected=lambda entities, text: print(f"Found {len(entities)} entities"),
    on_anonymized=lambda orig, anon, mapping: print(f"Anonymized {len(mapping)} values"),
    on_deanonymized=lambda anon, restored: print("Response restored"),
    on_error=lambda error, stage: print(f"Error in {stage}: {error}"),
)
```

## Inspecting What Was Redacted

Use the `on_anonymized` callback to see exactly what was replaced and what it was replaced with:

```python
import httpx
from rdakt_ai import RdaktMiddleware

def log_redactions(original_text, anonymized_text, mapping):
    print(f"Original:    {original_text}")
    print(f"Sent to LLM: {anonymized_text}")
    for real_value, token in mapping.items():
        print(f"  {real_value} -> {token}")

def log_restored(anonymized_text, restored_text):
    print(f"LLM saw:  {anonymized_text}")
    print(f"You see:  {restored_text}")

middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    on_anonymized=log_redactions,
    on_deanonymized=log_restored,
)
```

The same middleware instance works with any provider:

### OpenAI

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(http_client=httpx.AsyncClient(transport=middleware))

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize the case for John Smith (SSN 123-45-6789)"}],
)
```

### Anthropic

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(http_client=httpx.AsyncClient(transport=middleware))

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize the case for John Smith (SSN 123-45-6789)"}],
)
```

### Google Gemini

```python
from google import genai

client = genai.Client(http_options={"transport": middleware})

response = await client.aio.models.generate_content(
    model="gemini-2.0-flash",
    contents="Summarize the case for John Smith (SSN 123-45-6789)",
)
```

All three print the same inspection output:

```
Original:    Summarize the case for John Smith (SSN 123-45-6789)
Sent to LLM: Summarize the case for <PERSON_1> (SSN <SSN_1>)
  John Smith -> <PERSON_1>
  123-45-6789 -> <SSN_1>

LLM saw:  ... <PERSON_1> ... <SSN_1> ...
You see:  ... John Smith ... 123-45-6789 ...
```

You can also access the full entity map from the middleware's session at any time:

```python
entity_map = middleware._session.entity_map
# {'<PERSON_1>': 'John Smith', '<SSN_1>': '123-45-6789'}
```

## Audit Mode

Run in audit mode to detect PII without modifying requests — useful for assessing exposure before enabling anonymization:

```python
middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    mode="audit",
)
```

## Demo

See what rdakt-ai does without API keys:

```bash
uv run rdakt-ai demo                    # all scenarios
uv run rdakt-ai demo --scenario pii     # PII only
uv run rdakt-ai demo --scenario financial
uv run rdakt-ai demo --scenario secrets
```

## Development

```bash
# Install dependencies
make init

# Run tests
make test

# Run tests with coverage
make test/cov

# Run linting + type checking
make audit
```

## License

MIT
