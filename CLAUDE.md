# CLAUDE.md

## Project

**Rdakt AI** — composable anonymization middleware for LLM interactions. Intercepts outbound HTTP requests via httpx transport, detects PII with regex, anonymizes it with tokens/synthetic values, and deanonymizes streaming responses. Works with OpenAI, Anthropic, and Google Gemini.

- **Spec**: `docs/specs/2026-03-24-rdakt-ai-design.md`
- **Implementation plan**: `docs/plans/2026-03-24-rdakt-ai-mvp.md`
- **Issue tracker**: Linear workspace `rdakt-ai`, team prefix `RDA` (e.g. `RDA-12`). Use the Linear MCP tools (`list_issues`, `get_issue`, `save_issue`) to read and update tickets.
- **Detailed architecture skill**: `.claude/skills/rdakt-codebase/SKILL.md` — provider internals, streaming pipeline, common pitfalls

## Commands

Use `uv` to run all commands except make commands below.

```bash
# Install all dependencies
make init

# Run tests
make test

# Run tests with coverage
make test/cov

# Run the full audit pipeline (pre-commit + mypy)
make audit

# Run pre-commit hooks only
make pre_commit

# Run mypy only
make mypy

# Build the package
make build
```

## Workflow

### Task Execution

Work on **one Linear issue at a time**. Use the Linear MCP tools to manage state:

- See what's open / pick next: `list_issues` (filter by `assignee: "me"`, `state: "Todo"` or `"In Progress"`)
- Read full details for an issue: `get_issue` with the identifier (e.g. `RDA-12`)
- **Before starting work**, ensure the work is tracked under a Linear **project**:
  - If the issue belongs to a project, use that project.
  - If the project has **no child issues** for the work being started, create the issue(s) on the project first (`save_issue` with `project: "<project-id>"`) before beginning implementation.
- Move to in-progress before starting: `save_issue` with `state: "In Progress"`
- Mark done after merging: `save_issue` with `state: "Done"`
- **Also update the parent project's status** (not just the issue) as work progresses — move the project to `In Progress` when its first child issue starts, and to `Completed` when all its child issues are done. Use `save_project` with the corresponding `state`.

### Git & Commits

- **Prefer git worktrees** for issue work — create a dedicated worktree per Linear issue under `~/worktrees/rdakt-ai/` (e.g. `git worktree add ~/worktrees/rdakt-ai/RDA-12 -b RDA-12`) so concurrent issues stay isolated. Remove the worktree (`git worktree remove ~/worktrees/rdakt-ai/RDA-12`) once the branch is merged.
- **Work directly on master** (no feature branches for MVP phase) is acceptable when not using a worktree, but worktrees are preferred
- **Before committing**, always run in this order:
  1. `make test` — all tests must pass
  2. `make audit` — pre-commit + mypy must pass
- **Always push to remote after committing** (`git push`)
- **Commit after completing each issue** with the format:

  ```
  [RDA-<issue_id>] <type>: <description>
  ```

  Linear auto-links commits whose message contains the `RDA-<n>` identifier, so the commit shows up on the issue.

  **Types**:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation only changes
  - `test`: Adding tests, refactoring tests; no production code change
  - `refactor`: Refactoring production code
  - `chore`: Build, tooling, config changes; no production code change

  **Examples**:
  ```
  [RDA-1] chore: initialize project scaffold with pyproject.toml and tooling
  [RDA-2] feat: add core data models — Entity and RdaktContext
  [RDA-4] feat: add RegexDetector with built-in patterns
  ```

- **Pre-commit hooks are installed** — they run ruff check, ruff format, and file checks automatically on commit. Do not skip them with `--no-verify`.

### Testing & TDD

Follow **strict test-driven development** for every task:

1. **Write the failing test first** — define the expected behavior before writing any implementation code
2. **Run the test to confirm it fails** — `uv run pytest tests/<test_file>.py -v`. Verify the failure is for the right reason (e.g., `ImportError` or `AssertionError`, not a syntax error)
3. **Write the minimal implementation** to make the test pass — no more, no less
4. **Run the test to confirm it passes** — `uv run pytest tests/<test_file>.py -v`
5. **Refactor if needed** — clean up while keeping tests green
6. **Run the full suite** — `make test` to ensure nothing else broke

- Tests use `pytest` with `pytest-asyncio` for async tests
- Test files go in `tests/` mirroring the source structure
- Run a single test: `uv run pytest tests/<file>.py::<TestClass>::<test_name> -v`
- Run full suite: `make test`
- **Never weaken tests to make them pass** — fix the implementation instead
- **Never skip or delete a failing test** — understand why it fails and fix the root cause

### Examples & Documentation

Whenever a change introduces a **new public-facing surface** — a new
config field/section, a new public class or strategy, a new entry on
`RdaktConfig`, a new integration, a new yaml schema option — you must
ship it with discoverable documentation and a runnable example, not just
unit tests:

1. **Add a runnable script in `examples/`** (e.g. `examples/<feature>.py`)
   that exercises the feature end-to-end with realistic data. Every
   script under `examples/` is auto-run by `tests/test_examples.py`, so
   it must succeed with `uv run python examples/<feature>.py` and have
   no external dependencies (no API keys, no network) — use the applier
   / session / middleware in isolation if needed.
2. **Add a yaml file in `examples/configs/`** if the feature surfaces in
   `rdakt.yaml`, mirroring the style of `production.yaml` / `redis.yaml`
   / `audit.yaml`.
3. Treat the example and the yaml as part of the feature — not
   follow-ups. A PR that adds a config field but no example is
   incomplete.

### Code Style

- Dev environment is Python **3.14** (managed via `uv`).
- Client library (`rdakt_ai/`) must remain **importable on Python 3.11+** for downstream consumers — avoid 3.12-only syntax (PEP 695 `type` statements, generic `class Foo[T]`, `typing.override`) and 3.13-only typing (`TypeIs`, `ReadOnly`, `warnings.deprecated`). `X | Y` unions and `match` statements are fine (3.10+). Ruff `target-version = "py311"` enforces this — don't bump it without reviewing `requires-python` in `pyproject.toml`.
- Ruff for linting and formatting (line length 120)
- Mypy for type checking
- Library source in `rdakt_ai/` (top-level), tests in `tests/`
- No magic numbers: define constants
- Imports at the top of the module, not inside functions

### Typing conventions

Every internal data structure that crosses a function boundary must be a **named typed structure** — no anonymous positional tuples, no untyped dicts, no `Any` smuggling implicit schemas.

- **Internal data**: prefer `@dataclass(frozen=True, slots=True)` by default (named fields, immutable, cheap). For tiny ad-hoc shapes, `typing.NamedTuple` is acceptable.
- **External / parsed input** (yaml config, JSON request/response bodies, anything crossing a process boundary): use `pydantic` models with strict validation.
- **No `Any`** in function signatures unless it's a genuine opaque-passthrough (third-party untyped value, JSON-walk recursion). Every `Any` must have a comment explaining why.
- **No `dict[str, Any]` as an internal function parameter** — replace with a dataclass or `TypedDict`. At JSON parse boundaries, define a module-level `TypeAlias` (e.g. `JsonBody: TypeAlias = dict[str, Any]`) so the boundary is named, not implicit.
- **No anonymous tuples of length > 2** in function signatures or returns — promote to `NamedTuple` or `@dataclass`. Length-2 tuples are tolerated when the meaning is obvious from context.
- **Type all return values** explicitly, including `None`.
- **mypy must stay clean** (`make mypy`).

Example of the preferred pattern (replacing an 8-tuple builder):

```python
@dataclass(frozen=True, slots=True)
class MiddlewareState:
    config: RdaktConfig
    store: SessionStore
    pipeline: RdaktPipeline
    anonymizer: Anonymizer
    session: RdaktSession
    mode: str
    error_policy: str
    ontology: OntologyApplier | None

def _build_middleware_state(...) -> MiddlewareState:
    ...
```

instead of `-> tuple[RdaktConfig, SessionStore, RdaktPipeline, Anonymizer, RdaktSession, str, str, OntologyApplier | None]`.

## Architecture

```
Request flow:  SDK → RdaktMiddleware (httpx transport) → detect → anonymize → LLM
Response flow: LLM → SSE stream → AsyncDeanonymizingStream → deanonymize → SDK
```

```
rdakt_ai/
├── middleware.py      # RdaktMiddleware (async) + RdaktSyncMiddleware — entry points
├── session.py         # RdaktSession — entity map, deanonymize(), deanonymize_chunk()
├── streaming.py       # SSE stream wrappers, gzip decompression, chunk reassembly
├── formats.py         # Provider-agnostic SSE content extraction (OpenAI/Anthropic/Gemini)
├── pipeline.py        # Detection pipeline with RdaktStage chain
├── detectors/regex.py # RegexDetector — built-in + custom patterns
├── anonymizer.py      # Token replacement, synthetic substitution, hybrid strategies
├── config.py          # RdaktConfig from rdakt.yaml, validated mode/on_error
├── stores.py          # SessionStore ABC, thread-safe MemoryStore
├── models.py          # Entity, RdaktContext data models
├── logging.py         # Structured JSON logging
├── cli.py             # CLI: rdakt-ai init, rdakt-ai demo
└── __init__.py        # Public API surface
```

Detailed internals — provider wire formats, streaming pipeline, common pitfalls — are in `.claude/skills/rdakt-codebase/SKILL.md`.
