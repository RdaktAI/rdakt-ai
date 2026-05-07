"""Tests for LangChain/LangGraph integration connector."""

from __future__ import annotations

import httpx
import pytest

from rdakt_ai import RdaktConfig, RdaktMiddleware, RdaktSyncMiddleware
from rdakt_ai.integrations import NotSupportedError
from rdakt_ai.stores import MemoryStore

# Gate all langchain tests on import availability — must come before
# importing our integration module (which raises if langchain is missing).
pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")

from rdakt_ai.integrations.langchain import (
    RdaktLangChainMiddleware,
    create_http_clients,
    protect_chat_model,
)

# ChatOpenAI requires OPENAI_API_KEY to instantiate (even without making calls).
_FAKE_KEY = "sk-test-key-for-unit-tests"


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure OPENAI_API_KEY is set for every test in the module."""
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)


class TestCreateHttpClients:
    def test_returns_sync_and_async_clients(self) -> None:
        sync_client, async_client = create_http_clients()
        assert isinstance(sync_client, httpx.Client)
        assert isinstance(async_client, httpx.AsyncClient)

    def test_sync_transport_is_rdakt(self) -> None:
        sync_client, _ = create_http_clients()
        assert isinstance(sync_client._transport, RdaktSyncMiddleware)

    def test_async_transport_is_rdakt(self) -> None:
        _, async_client = create_http_clients()
        assert isinstance(async_client._transport, RdaktMiddleware)

    def test_accepts_config(self) -> None:
        config = RdaktConfig(mode="audit")
        sync_client, async_client = create_http_clients(config=config)
        sync_transport = sync_client._transport
        async_transport = async_client._transport
        assert isinstance(sync_transport, RdaktSyncMiddleware)
        assert isinstance(async_transport, RdaktMiddleware)
        assert sync_transport._mode == "audit"
        assert async_transport._mode == "audit"

    def test_session_key_consistent_across_sync_and_async(self) -> None:
        sync_client, async_client = create_http_clients(session_key="shared")
        sync_transport = sync_client._transport
        async_transport = async_client._transport
        assert isinstance(sync_transport, RdaktSyncMiddleware)
        assert isinstance(async_transport, RdaktMiddleware)
        assert sync_transport._session.session_id == "shared"
        assert async_transport._session.session_id == "shared"

    def test_auto_session_key_when_omitted(self) -> None:
        s1, _a1 = create_http_clients()
        s2, _a2 = create_http_clients()
        t1 = s1._transport
        t2 = s2._transport
        assert isinstance(t1, RdaktSyncMiddleware)
        assert isinstance(t2, RdaktSyncMiddleware)
        assert t1._session.session_id != t2._session.session_id

    def test_passes_middleware_kwargs(self) -> None:
        def on_error(exc: Exception, text: str) -> None:
            pass

        sync_client, async_client = create_http_clients(on_error=on_error)
        sync_transport = sync_client._transport
        async_transport = async_client._transport
        assert isinstance(sync_transport, RdaktSyncMiddleware)
        assert isinstance(async_transport, RdaktMiddleware)
        assert sync_transport._on_error_cb is on_error
        assert async_transport._on_error_cb is on_error


class TestProtectChatModel:
    def test_returns_chat_openai_instance(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        protected = protect_chat_model(model)
        assert isinstance(protected, ChatOpenAI)

    def test_preserves_model_name_and_temperature(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        protected = protect_chat_model(model)
        assert protected.model_name == "gpt-4o-mini"
        assert protected.temperature == 0.3

    def test_injects_sync_rdakt_transport(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        protected = protect_chat_model(model)
        assert protected.http_client is not None
        assert isinstance(protected.http_client._transport, RdaktSyncMiddleware)

    def test_injects_async_rdakt_transport(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        protected = protect_chat_model(model)
        assert protected.http_async_client is not None
        assert isinstance(protected.http_async_client._transport, RdaktMiddleware)

    def test_accepts_config(self) -> None:
        from langchain_openai import ChatOpenAI

        config = RdaktConfig(mode="audit")
        model = ChatOpenAI(model="gpt-4o")
        protected = protect_chat_model(model, config=config)
        sync_transport = protected.http_client._transport
        async_transport = protected.http_async_client._transport
        assert isinstance(sync_transport, RdaktSyncMiddleware)
        assert isinstance(async_transport, RdaktMiddleware)
        assert sync_transport._mode == "audit"
        assert async_transport._mode == "audit"

    def test_passes_session_key(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        protected = protect_chat_model(model, session_key="my-session")
        sync_transport = protected.http_client._transport
        async_transport = protected.http_async_client._transport
        assert isinstance(sync_transport, RdaktSyncMiddleware)
        assert isinstance(async_transport, RdaktMiddleware)
        assert sync_transport._session.session_id == "my-session"
        assert async_transport._session.session_id == "my-session"

    def test_auto_generates_unique_session_keys(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        p1 = protect_chat_model(model)
        p2 = protect_chat_model(model)
        t1 = p1.http_client._transport
        t2 = p2.http_client._transport
        assert isinstance(t1, RdaktSyncMiddleware)
        assert isinstance(t2, RdaktSyncMiddleware)
        assert t1._session.session_id != t2._session.session_id

    def test_raises_not_supported_for_unknown_class(self) -> None:
        class FakeModel:
            pass

        with pytest.raises(NotSupportedError, match="FakeModel"):
            protect_chat_model(FakeModel())  # type: ignore[arg-type]

    def test_raises_not_supported_for_unknown_class_includes_class_name(self) -> None:
        class MyCustomChat:
            pass

        with pytest.raises(NotSupportedError, match="MyCustomChat is not supported"):
            protect_chat_model(MyCustomChat())  # type: ignore[arg-type]


class TestProtectChatModelAnthropic:
    def test_raises_not_supported_for_chat_anthropic(self) -> None:
        """ChatAnthropic should raise NotSupportedError if langchain-anthropic is installed."""
        langchain_anthropic = pytest.importorskip("langchain_anthropic")
        model = langchain_anthropic.ChatAnthropic(model="claude-sonnet-4-20250514")
        with pytest.raises(NotSupportedError, match="ChatAnthropic"):
            protect_chat_model(model)


class TestRdaktLangChainMiddleware:
    def test_is_agent_middleware_subclass(self) -> None:
        from langchain.agents.middleware import AgentMiddleware

        assert issubclass(RdaktLangChainMiddleware, AgentMiddleware)

    def test_default_config(self) -> None:
        mw = RdaktLangChainMiddleware()
        assert mw._config is None
        assert mw._session_key is None

    def test_custom_config_and_session_key(self) -> None:
        config = RdaktConfig(mode="audit")
        mw = RdaktLangChainMiddleware(config=config, session_key="conv-1")
        assert mw._config is config
        assert mw._session_key == "conv-1"

    def test_anonymize_human_message(self) -> None:
        from langchain_core.messages import HumanMessage

        mw = RdaktLangChainMiddleware()
        messages = [HumanMessage(content="My email is john@example.com")]
        anonymized = mw._anonymize_messages(messages)
        assert "john@example.com" not in anonymized[0].content
        assert "EMAIL" in anonymized[0].content

    def test_anonymize_system_message(self) -> None:
        from langchain_core.messages import SystemMessage

        mw = RdaktLangChainMiddleware()
        messages = [SystemMessage(content="User SSN is 123-45-6789")]
        anonymized = mw._anonymize_messages(messages)
        assert "123-45-6789" not in anonymized[0].content

    def test_deanonymize_ai_message(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        mw = RdaktLangChainMiddleware()
        mw._anonymize_messages([HumanMessage(content="My email is john@example.com")])
        response_msgs = [AIMessage(content="Your email is <EMAIL_1>")]
        deanonymized = mw._deanonymize_messages(response_msgs)
        assert "john@example.com" in deanonymized[0].content

    def test_non_string_content_passed_through(self) -> None:
        from langchain_core.messages import HumanMessage

        mw = RdaktLangChainMiddleware()
        messages = [HumanMessage(content=[{"type": "text", "text": "hello"}])]
        anonymized = mw._anonymize_messages(messages)
        assert anonymized[0].content == [{"type": "text", "text": "hello"}]

    def test_session_key_persists(self) -> None:
        from langchain_core.messages import HumanMessage

        store = MemoryStore()
        mw1 = RdaktLangChainMiddleware(session_key="conv-1", store=store)
        mw1._anonymize_messages([HumanMessage(content="My email is john@example.com")])
        mw1._save_session()

        mw2 = RdaktLangChainMiddleware(session_key="conv-1", store=store)
        anonymized = mw2._anonymize_messages([HumanMessage(content="Email: john@example.com")])
        assert "<EMAIL_1>" in anonymized[0].content
