"""LangChain/LangGraph integration — protect chat models with Rdakt AI anonymization.

Install: ``pip install rdakt-ai[langchain]``

Usage::

    from rdakt_ai.integrations.langchain import protect_chat_model, create_http_clients

    # High-level: wrap a ChatOpenAI model
    protected = protect_chat_model(ChatOpenAI(model="gpt-4o"), config=config)

    # Low-level: get wired httpx clients
    http_client, async_http_client = create_http_clients(config=config)
    model = ChatOpenAI(model="gpt-4o", http_client=http_client, http_async_client=async_http_client)

    # Message-level: use as a LangChain AgentMiddleware
    from rdakt_ai.integrations.langchain import RdaktLangChainMiddleware
    mw = RdaktLangChainMiddleware(config=config)
    agent = create_agent(..., middleware=[mw])

Supported models:
    - ``ChatOpenAI`` — fully supported (clean httpx client injection)
    - ``ChatAnthropic`` — not yet supported (no public httpx param)
    - ``ChatGoogleGenerativeAI`` — not yet supported (async uses aiohttp)
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import BaseMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise ImportError(
        "langchain is required for this integration — install with: pip install rdakt-ai[langchain]"
    ) from exc

from rdakt_ai.anonymizer import Anonymizer
from rdakt_ai.config import RdaktConfig
from rdakt_ai.integrations import NotSupportedError
from rdakt_ai.middleware import RdaktMiddleware, RdaktSyncMiddleware, _build_middleware_state
from rdakt_ai.pipeline import RdaktPipeline
from rdakt_ai.session import RdaktSession
from rdakt_ai.stores import SessionStore

# Models that don't support httpx client injection via protect_chat_model().
_UNSUPPORTED_MODELS: dict[str, str] = {
    "ChatAnthropic": (
        "ChatAnthropic does not expose a public http_client parameter. Use create_http_clients() for manual wiring."
    ),
    "ChatGoogleGenerativeAI": (
        "ChatGoogleGenerativeAI uses aiohttp for async calls (not httpx). Use create_http_clients() for manual wiring."
    ),
}


def create_http_clients(
    *,
    config: RdaktConfig | None = None,
    session_key: str | None = None,
    store: SessionStore | None = None,
    **middleware_kwargs: Any,
) -> tuple[httpx.Client, httpx.AsyncClient]:
    """Create sync and async httpx clients with Rdakt middleware.

    Returns a ``(sync_client, async_client)`` tuple for use with
    ``ChatOpenAI(http_client=..., http_async_client=...)``.

    Args:
        config: Rdakt configuration. Defaults to ``RdaktConfig()``.
        session_key: Session identifier for token mapping persistence.
            If omitted, a UUID is generated (one session per call).
        store: Session store for mapping persistence.
        **middleware_kwargs: Extra kwargs passed to middleware constructors
            (e.g., ``on_error``, ``on_anonymized``).

    Returns:
        A ``(httpx.Client, httpx.AsyncClient)`` tuple, both wired with
        Rdakt anonymization.
    """
    key = session_key or str(uuid.uuid4())
    sync_middleware = RdaktSyncMiddleware(
        config=config,
        store=store,
        session_key=key,
        **middleware_kwargs,
    )
    async_middleware = RdaktMiddleware(
        config=config,
        store=store,
        session_key=key,
        **middleware_kwargs,
    )
    return httpx.Client(transport=sync_middleware), httpx.AsyncClient(transport=async_middleware)


def protect_chat_model(
    model: Any,
    *,
    config: RdaktConfig | None = None,
    session_key: str | None = None,
    store: SessionStore | None = None,
    **middleware_kwargs: Any,
) -> Any:
    """Wrap a LangChain chat model with Rdakt anonymization middleware.

    Returns a copy of *model* whose HTTP clients are wired through
    Rdakt middleware so that outbound requests are anonymized and
    streaming responses are deanonymized.

    Currently supports:
        - ``ChatOpenAI`` (from ``langchain-openai``)

    Args:
        model: A LangChain chat model instance.
        config: Rdakt configuration. Defaults to ``RdaktConfig()``.
        session_key: Session identifier for token mapping persistence.
            If omitted, a UUID is generated (one session per call).
        store: Session store for mapping persistence.
        **middleware_kwargs: Extra kwargs passed to middleware constructors
            (e.g., ``on_error``, ``on_anonymized``).

    Returns:
        A new model instance of the same type with Rdakt-protected
        httpx clients injected.

    Raises:
        NotSupportedError: If the model class is not supported.
    """
    class_name = type(model).__name__
    if class_name in _UNSUPPORTED_MODELS:
        raise NotSupportedError(_UNSUPPORTED_MODELS[class_name])

    if not isinstance(model, ChatOpenAI):
        raise NotSupportedError(
            f"{class_name} is not supported by protect_chat_model(). "
            f"Supported: ChatOpenAI. Use create_http_clients() for manual wiring."
        )

    sync_client, async_client = create_http_clients(
        config=config,
        session_key=session_key,
        store=store,
        **middleware_kwargs,
    )

    return model.model_copy(update={"http_client": sync_client, "http_async_client": async_client})


# ---------------------------------------------------------------------------
# RdaktLangChainMiddleware — message-level interception via LangChain AgentMiddleware
# ---------------------------------------------------------------------------


class RdaktLangChainMiddleware(AgentMiddleware):
    """LangChain middleware that anonymizes messages and deanonymizes responses.

    Only works with ``create_agent()``. For direct ``ChatModel`` usage, use
    ``create_http_clients()`` or ``protect_chat_model()`` instead.

    Usage::

        from rdakt_ai.integrations.langchain import RdaktLangChainMiddleware

        mw = RdaktLangChainMiddleware(config=config, session_key="conv-1")
        agent = create_agent(..., middleware=[mw])
    """

    def __init__(
        self,
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
        self._store_resolved: SessionStore | None = None

    def _ensure_initialized(self) -> None:
        """Lazily build pipeline, anonymizer, and session from config."""
        if self._pipeline is not None:
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
        self._store_resolved = state.store

    def _save_session(self) -> None:
        """Persist the session's entity map to the store."""
        self._ensure_initialized()
        assert self._session is not None
        assert self._store_resolved is not None

        if self._session.has_mappings:
            self._store_resolved.save(self._session.session_id, self._session.entity_map)

    def _anonymize_messages(self, messages: list[Any]) -> list[Any]:
        """Walk LangChain messages, detect PII, and anonymize text content."""
        self._ensure_initialized()
        assert self._pipeline is not None
        assert self._anonymizer is not None
        assert self._session is not None

        result: list[Any] = []
        for msg in messages:
            if not isinstance(msg, BaseMessage) or not isinstance(msg.content, str):
                result.append(msg)
                continue
            entities = self._pipeline.detect_sync(msg.content)
            if entities:
                anon_text, mapping = self._anonymizer.anonymize(
                    msg.content,
                    entities,
                    session=self._session,
                )
                self._session.add_mappings(mapping)
                result.append(msg.model_copy(update={"content": anon_text}))
            else:
                result.append(msg)
        return result

    def _deanonymize_messages(self, messages: list[Any]) -> list[Any]:
        """Walk LangChain messages and deanonymize token placeholders."""
        self._ensure_initialized()
        assert self._session is not None

        if not self._session.has_mappings:
            return messages
        result: list[Any] = []
        for msg in messages:
            if isinstance(msg, BaseMessage) and isinstance(msg.content, str):
                result.append(msg.model_copy(update={"content": self._session.deanonymize(msg.content)}))
            else:
                result.append(msg)
        return result

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Sync: anonymize request messages, call handler, deanonymize response."""
        self._ensure_initialized()
        assert self._pipeline is not None
        assert self._anonymizer is not None
        assert self._session is not None

        anon_messages = self._anonymize_messages(list(request.messages))
        anon_sys = None
        if request.system_message and isinstance(request.system_message.content, str):
            entities = self._pipeline.detect_sync(request.system_message.content)
            if entities:
                anon_text, mapping = self._anonymizer.anonymize(
                    request.system_message.content,
                    entities,
                    session=self._session,
                )
                self._session.add_mappings(mapping)
                anon_sys = request.system_message.model_copy(update={"content": anon_text})

        overrides: dict[str, Any] = {"messages": anon_messages}
        if anon_sys is not None:
            overrides["system_message"] = anon_sys
        anon_request = request.override(**overrides)

        response = handler(anon_request)
        response.result = self._deanonymize_messages(response.result)
        self._save_session()
        return response

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Async: anonymize request messages, call handler, deanonymize response."""
        self._ensure_initialized()
        assert self._pipeline is not None
        assert self._anonymizer is not None
        assert self._session is not None

        anon_messages = self._anonymize_messages(list(request.messages))
        anon_sys = None
        if request.system_message and isinstance(request.system_message.content, str):
            entities = self._pipeline.detect_sync(request.system_message.content)
            if entities:
                anon_text, mapping = self._anonymizer.anonymize(
                    request.system_message.content,
                    entities,
                    session=self._session,
                )
                self._session.add_mappings(mapping)
                anon_sys = request.system_message.model_copy(update={"content": anon_text})

        overrides: dict[str, Any] = {"messages": anon_messages}
        if anon_sys is not None:
            overrides["system_message"] = anon_sys
        anon_request = request.override(**overrides)

        response = await handler(anon_request)
        response.result = self._deanonymize_messages(response.result)
        self._save_session()
        return response
