"""PydanticAI integration — protect agents with Rdakt AI anonymization.

Install: ``pip install rdakt-ai[pydantic-ai]``

Usage::

    from rdakt_ai.integrations.pydantic_ai import protect_agent, create_http_client, RdaktCapability

    # High-level: wrap an existing agent
    protected = protect_agent(agent, config=config)

    # Low-level: get a wired httpx client
    http_client = create_http_client(config=config)

    # Message-level: use as a PydanticAI capability
    cap = RdaktCapability(config=config)
    agent = Agent("openai:gpt-4o", capabilities=[cap])
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from collections.abc import AsyncIterable
from typing import Any

import httpx

try:
    from pydantic_ai import Agent
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.messages import (
        AgentStreamEvent,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        PartDeltaEvent,
        SystemPromptPart,
        TextPart,
        TextPartDelta,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models import infer_model
except ImportError as exc:
    raise ImportError(
        "pydantic-ai is required for this integration — install with: pip install rdakt-ai[pydantic-ai]"
    ) from exc

from rdakt_ai.anonymizer import Anonymizer
from rdakt_ai.config import RdaktConfig
from rdakt_ai.middleware import RdaktMiddleware, _build_middleware_state
from rdakt_ai.pipeline import RdaktPipeline
from rdakt_ai.session import RdaktSession
from rdakt_ai.stores import SessionStore


def create_http_client(
    *,
    config: RdaktConfig | None = None,
    session_key: str | None = None,
    store: SessionStore | None = None,
    **middleware_kwargs: Any,
) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` with Rdakt middleware as the transport.

    Pass the returned client to any PydanticAI provider constructor::

        from pydantic_ai.providers.openai import OpenAIProvider

        http_client = create_http_client(config=config)
        provider = OpenAIProvider(http_client=http_client)

    Args:
        config: Rdakt configuration. Defaults to ``RdaktConfig()``.
        session_key: Session identifier for token mapping persistence.
            If omitted, a UUID is generated (one session per call).
        store: Session store for mapping persistence. If omitted, uses
            the store configured in ``config``.
        **middleware_kwargs: Extra keyword arguments passed to
            ``RdaktMiddleware`` (e.g., ``on_error``, ``on_anonymized``).

    Returns:
        An async httpx client wired with Rdakt anonymization.
    """
    key = session_key or str(uuid.uuid4())
    middleware = RdaktMiddleware(
        config=config,
        store=store,
        session_key=key,
        **middleware_kwargs,
    )
    return httpx.AsyncClient(transport=middleware)


def protect_agent(
    agent: Agent,  # type: ignore[type-arg]
    *,
    config: RdaktConfig | None = None,
    session_key: str | None = None,
    store: SessionStore | None = None,
    **middleware_kwargs: Any,
) -> Agent:  # type: ignore[type-arg]
    """Wrap a PydanticAI ``Agent`` with Rdakt AI anonymization.

    Returns a **new** agent instance with a Rdakt middleware transport
    injected into the provider's httpx client.  The original agent is
    not mutated.

    The provider is reconstructed via its public constructor with
    ``http_client=``.  Credentials (API key, base URL) are read from
    environment variables by the provider — if you need to pass them
    explicitly, use Tier 1 (``create_http_client``) instead.

    Args:
        agent: A ``pydantic_ai.Agent`` instance.
        config: Rdakt configuration. Defaults to ``RdaktConfig()``.
        session_key: Session identifier for token mapping persistence.
            If omitted, a UUID is generated (one session per call).
        store: Session store for mapping persistence. If omitted, uses
            the store configured in ``config``.
        **middleware_kwargs: Extra keyword arguments passed to
            ``RdaktMiddleware`` (e.g., ``on_error``, ``on_anonymized``).

    Returns:
        A new ``Agent`` with the Rdakt middleware injected.

    Raises:
        TypeError: If *agent* is not a ``pydantic_ai.Agent``.
        ValueError: If the agent has no model configured.
    """
    if not isinstance(agent, Agent):
        msg = f"Expected a pydantic_ai.Agent, got {type(agent).__name__}"
        raise TypeError(msg)

    model = agent.model
    if model is None:
        msg = "Cannot protect an agent with no model configured — pass a model when creating the Agent"
        raise ValueError(msg)

    # Resolve string model names (e.g. "openai:gpt-4o") to model objects.
    if isinstance(model, str):
        model = infer_model(model)

    wired_client = create_http_client(
        config=config,
        session_key=session_key,
        store=store,
        **middleware_kwargs,
    )

    # PydanticAI doesn't expose a public API for the provider yet,
    # so we access _provider to reconstruct it with our transport.
    new_provider = type(model._provider)(http_client=wired_client)  # type: ignore[attr-defined]
    new_model = type(model)(model.model_name, provider=new_provider)  # type: ignore[call-arg,arg-type,misc]

    new_agent = copy.copy(agent)
    new_agent.model = new_model
    return new_agent


# ---------------------------------------------------------------------------
# RdaktCapability — message-level interception via PydanticAI capabilities
# ---------------------------------------------------------------------------


class RdaktCapability(AbstractCapability):  # type: ignore[type-arg]
    """PydanticAI capability that anonymizes request messages and deanonymizes responses.

    Uses the core Rdakt detection pipeline at the message level, providing
    an alternative to the transport-level ``RdaktMiddleware`` approach.

    Usage::

        from rdakt_ai.integrations.pydantic_ai import RdaktCapability

        cap = RdaktCapability(config=config, session_key="conv-1")
        agent = Agent("openai:gpt-4o", capabilities=[cap])
        result = await agent.run("My email is john@example.com")
    """

    def __init__(
        self,
        *,
        config: RdaktConfig | None = None,
        session_key: str | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self._config = config
        self._session_key = session_key
        self._store_param = store
        # Lazily initialized
        self._pipeline: RdaktPipeline | None = None
        self._anonymizer: Anonymizer | None = None
        self._session: RdaktSession | None = None
        self._store: SessionStore | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily build pipeline, anonymizer, and session from config."""
        if self._initialized:
            return
        state = _build_middleware_state(
            self._config,
            self._store_param,
            self._session_key,
            None,
            None,
        )
        self._pipeline = state.pipeline
        self._anonymizer = state.anonymizer
        self._session = state.session
        self._store = state.store
        self._initialized = True

    def _anonymize_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Walk ModelRequest parts, detect PII, and anonymize text content."""
        self._ensure_initialized()
        assert self._pipeline is not None
        assert self._anonymizer is not None
        assert self._session is not None

        result: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[Any] = []
                for part in msg.parts:
                    if isinstance(part, (UserPromptPart, SystemPromptPart)):
                        content = part.content
                        if isinstance(content, str):
                            entities = self._pipeline.detect_sync(content)
                            if entities:
                                anon_text, mapping = self._anonymizer.anonymize(
                                    content, entities, session=self._session
                                )
                                self._session.add_mappings(mapping)
                                new_parts.append(dataclasses.replace(part, content=anon_text))
                            else:
                                new_parts.append(part)
                        else:
                            new_parts.append(part)
                    elif isinstance(part, ToolReturnPart) and isinstance(part.content, str):
                        entities = self._pipeline.detect_sync(part.content)
                        if entities:
                            anon_text, mapping = self._anonymizer.anonymize(
                                part.content, entities, session=self._session
                            )
                            self._session.add_mappings(mapping)
                            new_parts.append(dataclasses.replace(part, content=anon_text))
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                result.append(dataclasses.replace(msg, parts=new_parts))
            else:
                # ModelResponse messages pass through unchanged
                result.append(msg)
        return result

    def _deanonymize_response(self, response: ModelResponse) -> ModelResponse:
        """Walk ModelResponse parts and deanonymize text content."""
        self._ensure_initialized()
        assert self._session is not None

        if not self._session.has_mappings:
            return response

        new_parts: list[Any] = []
        for part in response.parts:
            if isinstance(part, TextPart):
                new_parts.append(dataclasses.replace(part, content=self._session.deanonymize(part.content)))
            else:
                new_parts.append(part)
        return dataclasses.replace(response, parts=new_parts)

    def _save_session(self) -> None:
        """Persist the session's entity map to the store."""
        self._ensure_initialized()
        assert self._session is not None
        assert self._store is not None

        if self._session.has_mappings:
            self._store.save(self._session.session_id, self._session.entity_map)

    def _load_session(self) -> None:
        """Load an existing session from the store (done automatically during init)."""
        self._ensure_initialized()

    async def before_model_request(
        self,
        ctx: Any,
        request_context: Any,
    ) -> Any:
        """Anonymize message text parts before the model call."""
        request_context.messages = self._anonymize_messages(request_context.messages)
        return request_context

    async def after_model_request(
        self,
        ctx: Any,
        *,
        request_context: Any,
        response: Any,
    ) -> Any:
        """Deanonymize response text parts after the model call and save session."""
        deanonymized = self._deanonymize_response(response)
        self._save_session()
        return deanonymized

    async def wrap_run_event_stream(
        self,
        ctx: Any,
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Deanonymize streaming TextPartDelta events."""
        self._ensure_initialized()
        assert self._session is not None

        async for event in stream:
            if (
                isinstance(event, PartDeltaEvent)
                and isinstance(event.delta, TextPartDelta)
                and self._session.has_mappings
            ):
                chunks = self._session.deanonymize_chunk(event.delta.content_delta)
                if chunks:
                    new_content = "".join(chunks)
                    new_delta = dataclasses.replace(event.delta, content_delta=new_content)
                    yield dataclasses.replace(event, delta=new_delta)
                else:
                    # Buffer absorbed the chunk, emit empty delta
                    yield dataclasses.replace(
                        event,
                        delta=dataclasses.replace(event.delta, content_delta=""),
                    )
            else:
                yield event

        # Flush any remaining buffer
        if self._session.has_mappings:
            remaining = self._session.flush()
            if remaining:
                flush_text = "".join(remaining)
                flush_delta = TextPartDelta(content_delta=flush_text)
                yield PartDeltaEvent(index=0, delta=flush_delta)
