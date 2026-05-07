# Configuration Guide

Rdakt AI is configured via a YAML file (`rdakt.yaml`) or programmatically through `RdaktConfig`.

## Quick Start

Create a `rdakt.yaml` in your project root:

```yaml
mode: active
pipeline:
  - regex
session:
  store: memory
```

Load it in your application:

```python
from pathlib import Path
from rdakt_ai import load_config, create_store, RdaktMiddleware
import httpx

config = load_config(Path("rdakt.yaml"))
store = create_store(config)
middleware = RdaktMiddleware(inner=httpx.AsyncHTTPTransport(), config=config, store=store)
```

Or configure programmatically:

```python
from rdakt_ai import RdaktConfig, EntityConfig, SessionConfig

config = RdaktConfig(
    mode="active",
    pipeline=["regex"],
    entities={
        "EMAIL": EntityConfig(strategy="token"),
        "PERSON": EntityConfig(strategy="synthetic"),
    },
    session=SessionConfig(store="memory"),
)
```

## Configuration Reference

### `mode`

Controls whether PII is actually anonymized or just detected and logged.

| Value | Description |
|-------|-------------|
| `active` (default) | Detect and anonymize PII before forwarding to the LLM |
| `audit` | Detect and log PII but forward the original text unchanged |

### `on_error`

What to do when the anonymization pipeline encounters an error.

| Value | Description |
|-------|-------------|
| `warn_and_forward` (default) | Log a warning and forward the original unanonymized request |
| `block` | Raise an exception — the SDK call fails |

### `pipeline`

List of detection stages to run, in order. Each entry is either a string (detector name) or a dict with detector name and options.

```yaml
# Simple — regex only (default)
pipeline:
  - regex

# Regex + NER
pipeline:
  - regex
  - ner:
      model: en_core_web_sm

# NER only
pipeline:
  - ner:
      model: en_core_web_sm
```

Available detectors:

| Detector | Description | Extra dependency |
|----------|-------------|------------------|
| `regex` | Built-in regex patterns for EMAIL, PHONE, SSN, CREDIT_CARD, API_KEY, AWS_KEY, IP_ADDRESS, IBAN, JWT | None |
| `ner` | spaCy Named Entity Recognition for PERSON, ORG, GPE, etc. | `pip install rdakt-ai[ner]` + `python -m spacy download <model>` |

### `session`

Session store configuration. The `store` field selects the backend; additional fields are passed as options to the store constructor.

#### Memory Store (default)

```yaml
session:
  store: memory
```

In-memory, single-process. Data is lost when the process exits. Suitable for development and testing.

#### SQLite Store

```yaml
session:
  store: sqlite
  path: /path/to/sessions.db
```

Persistent file-based storage with WAL mode enabled. Suitable for single-server production deployments.

| Option | Type | Description |
|--------|------|-------------|
| `path` | string | Path to the SQLite database file |

#### Redis Store

```yaml
session:
  store: redis
  url: redis://localhost:6379
  ttl: 3600
  prefix: "rdakt:"
```

Distributed storage. Requires `pip install rdakt-ai[redis]`.

| Option | Type | Description |
|--------|------|-------------|
| `url` | string | Redis connection URL |
| `ttl` | int | Time-to-live in seconds for session keys (optional) |
| `prefix` | string | Key prefix for namespacing (default: `"rdakt:"`) |

### `entities`

Per-entity-type configuration. Each entry maps an entity type name to its settings.

```yaml
entities:
  EMAIL:
    strategy: token
  PERSON:
    strategy: synthetic
  EMPLOYEE_ID:
    pattern: "EMP-\\d{6}"
    strategy: token
```

#### `strategy`

How to anonymize detected entities of this type.

| Strategy | Output | Use case |
|----------|--------|----------|
| `token` | `<TYPE_N>` (e.g., `<EMAIL_1>`) | Structured data — emails, SSNs, IDs |
| `synthetic` | Faker-generated value | Names, organizations |
| `hybrid` (default) | `synthetic` for PERSON/ORG, `token` for everything else | General-purpose |

#### `pattern`

Custom regex pattern for detecting entities of this type. Use this to detect domain-specific PII that the built-in patterns don't cover.

```yaml
entities:
  EMPLOYEE_ID:
    pattern: "EMP-\\d{6}"
  ACCOUNT_NUMBER:
    pattern: "\\d{4}-\\d{4}-\\d{4}"
```

You can use `pattern` alone (detection only, default strategy), `strategy` alone (override strategy for a built-in entity type), or both together.

## Example Configurations

Example YAML files are in [`examples/configs/`](../examples/configs/):

| File | Description |
|------|-------------|
| [`minimal.yaml`](../examples/configs/minimal.yaml) | Default config — regex + memory store |
| [`production.yaml`](../examples/configs/production.yaml) | Regex + NER, SQLite, custom patterns, fail-closed |
| [`redis.yaml`](../examples/configs/redis.yaml) | Redis-backed session store for distributed deployments |
| [`audit.yaml`](../examples/configs/audit.yaml) | Audit mode — detect and log without anonymizing |
| [`ner_only.yaml`](../examples/configs/ner_only.yaml) | spaCy NER without regex |

## Using the Middleware in Your Project

Rdakt AI works as an [httpx transport](https://www.python-httpx.org/advanced/transports/). You create the middleware, then pass it to any SDK that accepts an `httpx` client — OpenAI, Anthropic, and Google Gemini all do.

The pattern is always the same:

1. Build a `RdaktConfig` (from YAML or in code)
2. Create a session store from the config
3. Create the middleware with the config and store
4. Pass the middleware as the `transport` to an `httpx` client
5. Pass that client to your SDK

### From a YAML file

```python
from pathlib import Path
from rdakt_ai import load_config, create_store, RdaktMiddleware

config = load_config(Path("rdakt.yaml"))
store = create_store(config)
middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    config=config,
    store=store,
)
```

### From code (no YAML)

```python
from rdakt_ai import RdaktConfig, EntityConfig, SessionConfig, create_store, RdaktMiddleware

config = RdaktConfig(
    mode="active",
    on_error="block",
    pipeline=["regex"],
    entities={
        "EMAIL": EntityConfig(strategy="token"),
        "PERSON": EntityConfig(strategy="synthetic"),
    },
    session=SessionConfig(store="sqlite", path="sessions.db"),
)
store = create_store(config)
middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    config=config,
    store=store,
)
```

### OpenAI (async)

```python
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    http_client=httpx.AsyncClient(transport=middleware),
)

response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My email is john@example.com"}],
)
# The LLM received "<EMAIL_1>" — the response has the real email restored.
```

### OpenAI (sync)

```python
import httpx
from openai import OpenAI
from rdakt_ai import RdaktSyncMiddleware

sync_middleware = RdaktSyncMiddleware(
    inner=httpx.HTTPTransport(),
    config=config,
    store=store,
)

client = OpenAI(
    http_client=httpx.Client(transport=sync_middleware),
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
)
```

### Anthropic

```python
import httpx
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    http_client=httpx.AsyncClient(transport=middleware),
)

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
)
```

### Google Gemini

```python
import httpx
from google import genai

client = genai.Client(
    api_key="...",
    http_options={"httpx_client": httpx.AsyncClient(transport=middleware)},
)

response = await client.aio.models.generate_content(
    model="gemini-2.0-flash",
    contents="My phone number is (555) 123-4567",
)
```

### FastAPI

```python
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI
from rdakt_ai import RdaktMiddleware, create_store, load_config

config = load_config(Path("rdakt.yaml"))
store = create_store(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    store.close()

app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(prompt: str):
    # Create middleware per-request for independent session state,
    # or once at startup if you want shared state across requests.
    middleware = RdaktMiddleware(
        inner=httpx.AsyncHTTPTransport(),
        config=config,
        store=store,
    )
    client = AsyncOpenAI(
        http_client=httpx.AsyncClient(transport=middleware),
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return {"response": response.choices[0].message.content}
```

### Event Callbacks

The middleware accepts optional callbacks for observability:

```python
def on_entities_detected(entities, text):
    print(f"Found {len(entities)} entities")

def on_anonymized(original, anonymized, mapping):
    print(f"Mapping: {mapping}")  # e.g. {"<EMAIL_1>": "john@example.com"}

def on_deanonymized(anon_response, restored_response):
    print("Response restored")

def on_error(exception, phase):
    print(f"Error in {phase}: {exception}")

middleware = RdaktMiddleware(
    inner=httpx.AsyncHTTPTransport(),
    config=config,
    store=store,
    on_entities_detected=on_entities_detected,
    on_anonymized=on_anonymized,
    on_deanonymized=on_deanonymized,
    on_error=on_error,
)
```

### With the CLI

```bash
# Generate a default config file
rdakt-ai init

# Run a demo against a live LLM with your config
rdakt-ai demo --config rdakt.yaml --provider openai
```

## Backwards Compatibility

The following constructor arguments still work but map to the new nested structure:

| Old style | New style |
|-----------|-----------|
| `RdaktConfig(session_store="redis")` | `RdaktConfig(session=SessionConfig(store="redis"))` |
| `RdaktConfig(store_options={"url": "..."})` | `RdaktConfig(session=SessionConfig(store="redis", url="..."))` |
| `RdaktConfig(entity_strategies={"EMAIL": "token"})` | `RdaktConfig(entities={"EMAIL": EntityConfig(strategy="token")})` |
| `RdaktConfig(custom_patterns={"X": "..."})` | `RdaktConfig(entities={"X": EntityConfig(pattern="...")})` |

The YAML key `detection.layers` is deprecated in favor of `pipeline`:

```yaml
# Old (deprecated)
detection:
  layers:
    - regex

# New
pipeline:
  - regex
```
