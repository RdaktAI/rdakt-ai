"""Tests for PydanticAI integration connector."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from rdakt_ai import RdaktConfig, RdaktMiddleware
from rdakt_ai.stores import MemoryStore

# Gate pydantic-ai tests on import availability — must come before
# importing our integration module (which raises if pydantic-ai is missing).
pydantic_ai = pytest.importorskip("pydantic_ai")

from rdakt_ai.integrations.pydantic_ai import RdaktCapability, create_http_client, protect_agent  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure provider constructors don't fail due to missing API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tests")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-tests")


class TestCreateHttpClient:
    def test_returns_async_client(self) -> None:
        client = create_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_transport_is_rdakt_middleware(self) -> None:
        client = create_http_client()
        transport = client._transport
        assert isinstance(transport, RdaktMiddleware)

    def test_accepts_config(self) -> None:
        config = RdaktConfig(mode="audit")
        client = create_http_client(config=config)
        transport = client._transport
        assert isinstance(transport, RdaktMiddleware)
        assert transport._mode == "audit"

    def test_session_key_passed_through(self) -> None:
        client = create_http_client(session_key="my-session")
        transport = client._transport
        assert isinstance(transport, RdaktMiddleware)
        assert transport._session.session_id == "my-session"

    def test_auto_session_key_when_omitted(self) -> None:
        client1 = create_http_client()
        client2 = create_http_client()
        t1 = client1._transport
        t2 = client2._transport
        assert isinstance(t1, RdaktMiddleware)
        assert isinstance(t2, RdaktMiddleware)
        # Auto-generated UUIDs should differ
        assert t1._session.session_id != t2._session.session_id

    def test_passes_middleware_kwargs(self) -> None:
        callback_called = False

        def on_error(exc: Exception, text: str) -> None:
            nonlocal callback_called
            callback_called = True

        client = create_http_client(on_error=on_error)
        transport = client._transport
        assert isinstance(transport, RdaktMiddleware)
        assert transport._on_error_cb is on_error


class TestProtectAgent:
    """Tests for the high-level protect_agent factory."""

    def _get_transport(self, agent: Any) -> Any:
        """Extract the httpx transport from a protected agent."""
        return agent.model._provider._client._client._transport

    def test_returns_agent_instance(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="hello")
        protected = protect_agent(agent)
        assert isinstance(protected, Agent)

    def test_returns_new_agent_not_original(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="hello")
        protected = protect_agent(agent)
        assert protected is not agent

    def test_preserves_system_prompt(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")
        protected = protect_agent(agent)
        assert protected._system_prompts == ("You are a helpful assistant.",)

    def test_injects_rdakt_middleware_transport(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="test")
        protected = protect_agent(agent)
        transport = self._get_transport(protected)
        assert isinstance(transport, RdaktMiddleware)

    def test_does_not_mutate_original_agent(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="test")
        _protected = protect_agent(agent)
        original_transport = agent.model._provider._client._client._transport  # type: ignore[union-attr]
        assert not isinstance(original_transport, RdaktMiddleware)

    def test_accepts_rdakt_config(self) -> None:
        from pydantic_ai import Agent

        config = RdaktConfig(mode="audit")
        agent = Agent("openai:gpt-4o", system_prompt="test")
        protected = protect_agent(agent, config=config)
        transport = self._get_transport(protected)
        assert isinstance(transport, RdaktMiddleware)
        assert transport._mode == "audit"

    def test_session_key_passed_through(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="test")
        protected = protect_agent(agent, session_key="my-session")
        transport = self._get_transport(protected)
        assert isinstance(transport, RdaktMiddleware)
        assert transport._session.session_id == "my-session"

    def test_auto_generates_unique_session_keys(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o", system_prompt="test")
        p1 = protect_agent(agent)
        p2 = protect_agent(agent)
        t1 = self._get_transport(p1)
        t2 = self._get_transport(p2)
        assert isinstance(t1, RdaktMiddleware)
        assert isinstance(t2, RdaktMiddleware)
        assert t1._session.session_id != t2._session.session_id

    def test_works_with_anthropic_provider(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("anthropic:claude-3-5-sonnet-latest", system_prompt="test")
        protected = protect_agent(agent)
        transport = self._get_transport(protected)
        assert isinstance(transport, RdaktMiddleware)

    def test_raises_on_non_agent_input(self) -> None:
        with pytest.raises(TypeError, match=r"Expected a pydantic_ai\.Agent"):
            protect_agent("not an agent")  # type: ignore[arg-type]

    def test_raises_when_model_not_set(self) -> None:
        from pydantic_ai import Agent

        agent = Agent(system_prompt="test")
        with pytest.raises(ValueError, match="no model configured"):
            protect_agent(agent)

    def test_preserves_model_name(self) -> None:
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o-mini", system_prompt="test")
        protected = protect_agent(agent)
        assert not isinstance(protected.model, str)
        assert protected.model is not None
        assert protected.model.model_name == "gpt-4o-mini"

    def test_passes_middleware_kwargs(self) -> None:
        from pydantic_ai import Agent

        def on_error(exc: Exception, text: str) -> None:
            pass

        agent = Agent("openai:gpt-4o", system_prompt="test")
        protected = protect_agent(agent, on_error=on_error)
        transport = self._get_transport(protected)
        assert isinstance(transport, RdaktMiddleware)
        assert transport._on_error_cb is on_error


class TestRdaktCapability:
    """Tests for the PydanticAI capability (message-level interception)."""

    def test_is_abstract_capability_subclass(self) -> None:
        from pydantic_ai.capabilities.abstract import AbstractCapability

        assert issubclass(RdaktCapability, AbstractCapability)

    def test_default_config(self) -> None:
        cap = RdaktCapability()
        assert cap._config is None
        assert cap._session_key is None

    def test_custom_config_and_session_key(self) -> None:
        config = RdaktConfig(mode="audit")
        cap = RdaktCapability(config=config, session_key="conv-1")
        assert cap._config is config
        assert cap._session_key == "conv-1"

    def test_anonymize_text_in_user_prompt(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = RdaktCapability()
        parts = [UserPromptPart(content="My email is john@example.com")]
        request = ModelRequest(parts=parts)
        anonymized = cap._anonymize_messages([request])
        anon_msg = anonymized[0]
        assert isinstance(anon_msg, ModelRequest)
        anon_part = anon_msg.parts[0]
        assert isinstance(anon_part, UserPromptPart)
        text = anon_part.content
        assert isinstance(text, str)
        assert "john@example.com" not in text
        assert "<EMAIL_1>" in text or "EMAIL" in text

    def test_anonymize_text_in_system_prompt(self) -> None:
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        cap = RdaktCapability()
        parts = [SystemPromptPart(content="User SSN is 123-45-6789")]
        request = ModelRequest(parts=parts)
        anonymized = cap._anonymize_messages([request])
        anon_msg = anonymized[0]
        assert isinstance(anon_msg, ModelRequest)
        anon_part = anon_msg.parts[0]
        assert isinstance(anon_part, SystemPromptPart)
        assert "123-45-6789" not in anon_part.content

    def test_deanonymize_text_part(self) -> None:
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = RdaktCapability()
        # Anonymize first to establish mappings
        parts = [UserPromptPart(content="My email is john@example.com")]
        cap._anonymize_messages([ModelRequest(parts=parts)])
        # Then deanonymize
        response = ModelResponse(parts=[TextPart(content="Your email is <EMAIL_1>")])
        deanonymized = cap._deanonymize_response(response)
        depart = deanonymized.parts[0]
        assert isinstance(depart, TextPart)
        assert "john@example.com" in depart.content

    def test_session_key_persists_across_calls(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        store = MemoryStore()
        cap1 = RdaktCapability(session_key="conv-1", store=store)
        cap1._anonymize_messages([ModelRequest(parts=[UserPromptPart(content="My email is john@example.com")])])
        cap1._save_session()
        # Second turn
        cap2 = RdaktCapability(session_key="conv-1", store=store)
        anonymized = cap2._anonymize_messages([ModelRequest(parts=[UserPromptPart(content="Email: john@example.com")])])
        anon_msg = anonymized[0]
        assert isinstance(anon_msg, ModelRequest)
        anon_part = anon_msg.parts[0]
        assert isinstance(anon_part, UserPromptPart)
        text = anon_part.content
        assert isinstance(text, str)
        assert "<EMAIL_1>" in text

    def test_non_text_parts_passed_through(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = RdaktCapability()
        parts = [UserPromptPart(content="No PII here")]
        request = ModelRequest(parts=parts)
        anonymized = cap._anonymize_messages([request])
        anon_msg = anonymized[0]
        assert isinstance(anon_msg, ModelRequest)
        anon_part = anon_msg.parts[0]
        assert isinstance(anon_part, UserPromptPart)
        assert anon_part.content == "No PII here"
