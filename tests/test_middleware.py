"""Tests for RdaktMiddleware — httpx transport integration."""

import json

import httpx
import pytest

from rdakt_ai.config import RdaktConfig
from rdakt_ai.middleware import RdaktMiddleware
from rdakt_ai.stores import SQLiteStore


class CapturingTransport(httpx.AsyncBaseTransport):
    """Test transport that captures request body and returns a fixed response."""

    def __init__(self, response_json: dict | None = None) -> None:
        self.captured_body: dict = {}
        self._response_json = response_json or {"choices": [{"message": {"content": "OK"}}]}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured_body = json.loads(request.content.decode())
        return httpx.Response(200, json=self._response_json)


class EchoTokenTransport(httpx.AsyncBaseTransport):
    """Test transport that echoes back the request message content."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"I see {content}"}}]},
        )


class TestRdaktMiddlewareBasic:
    async def test_passthrough_no_pii(self) -> None:
        transport = CapturingTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello world"}]},
            )
        assert response.status_code == 200
        # Content should be unchanged — no PII detected
        assert transport.captured_body["messages"][0]["content"] == "Hello world"

    async def test_anonymizes_email_in_request(self) -> None:
        transport = CapturingTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email me at john@example.com"}]},
            )
        content = transport.captured_body["messages"][0]["content"]
        assert "john@example.com" not in content
        assert "<EMAIL_1>" in content

    async def test_deanonymizes_response(self) -> None:
        middleware = RdaktMiddleware(inner=EchoTokenTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        data = response.json()
        response_content = data["choices"][0]["message"]["content"]
        assert "john@example.com" in response_content


class TestAuditMode:
    async def test_audit_mode_does_not_anonymize(self) -> None:
        transport = CapturingTransport()
        middleware = RdaktMiddleware(inner=transport, mode="audit")
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        content = transport.captured_body["messages"][0]["content"]
        assert "john@example.com" in content  # NOT anonymized in audit mode


class TestMiddlewareEdgeCases:
    async def test_empty_request_body_passthrough(self) -> None:
        class OkTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        middleware = RdaktMiddleware(inner=OkTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
            )
        assert response.status_code == 200

    async def test_non_json_body_passthrough(self) -> None:
        class OkTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        middleware = RdaktMiddleware(inner=OkTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=b"not json at all",
            )
        assert response.status_code == 200

    async def test_malformed_response_returns_original(self) -> None:
        class BadResponseTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, content=b"not json")

        middleware = RdaktMiddleware(inner=BadResponseTransport())
        # Seed the session so deanonymization is attempted
        middleware._session.add_mappings({"<EMAIL_1>": "john@example.com"})
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        assert response.status_code == 200


class TestFailOpen:
    async def test_detection_error_forwards_unanonymized(self) -> None:
        """If detection fails, middleware should forward the request as-is."""

        class OkTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        middleware = RdaktMiddleware(inner=OkTransport(), error_policy="warn_and_forward")
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 200


class TestMiddlewareConfigDrivenStore:
    async def test_uses_create_store_from_config(self, tmp_path) -> None:
        """When no store kwarg, middleware uses create_store(config)."""
        transport = CapturingTransport()
        config = RdaktConfig(  # type: ignore[call-arg]
            session_store="sqlite",
            store_options={"path": str(tmp_path / "test.db")},
        )
        middleware = RdaktMiddleware(inner=transport, config=config)
        assert isinstance(middleware._store, SQLiteStore)

    async def test_explicit_store_overrides_config(self, tmp_path) -> None:
        """Explicit store kwarg takes precedence over config."""
        from rdakt_ai.stores import MemoryStore

        transport = CapturingTransport()
        config = RdaktConfig(session_store="sqlite", store_options={"path": str(tmp_path / "t.db")})  # type: ignore[call-arg]
        explicit_store = MemoryStore()
        middleware = RdaktMiddleware(inner=transport, config=config, store=explicit_store)
        assert middleware._store is explicit_store


class TestMiddlewareConfigDrivenPipeline:
    async def test_regex_only_pipeline_from_config(self) -> None:
        """Default config builds regex-only pipeline."""
        transport = CapturingTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email me at john@example.com"}]},
            )
        content = transport.captured_body["messages"][0]["content"]
        assert "<EMAIL_1>" in content

    async def test_ner_pipeline_from_config(self) -> None:
        """Config with pipeline: [ner] builds NER-only pipeline."""
        from unittest.mock import MagicMock, patch

        from rdakt_ai.models import Entity

        mock_ner = MagicMock()
        mock_ner.return_value = MagicMock()
        mock_ner.return_value.detect.return_value = [
            Entity(value="John Smith", type="PERSON", start=0, end=10, detector_priority=10),
        ]

        transport = CapturingTransport()
        # Use token strategy for PERSON so we get a deterministic <PERSON_1> token
        config = RdaktConfig(
            pipeline=[{"ner": {"model": "en_core_web_sm"}}],
            entity_strategies={"PERSON": "token"},  # type: ignore[call-arg]
        )

        with patch("rdakt_ai.detectors.ner.SpacyDetector", mock_ner):
            middleware = RdaktMiddleware(inner=transport, config=config)
            async with httpx.AsyncClient(transport=middleware) as client:
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "John Smith works at Acme"}]},
                )
        content = transport.captured_body["messages"][0]["content"]
        assert "John Smith" not in content
        assert "<PERSON_1>" in content


class TestMiddlewareConfigDrivenEdgeCases:
    def test_empty_pipeline_defaults_to_regex(self):
        """Empty pipeline config defaults to regex."""
        config = RdaktConfig(pipeline=[])
        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            config=config,
        )
        assert middleware is not None

    def test_invalid_entity_strategy_in_config(self):
        """Invalid strategy in config raises at construction."""
        with pytest.raises(ValueError, match="not_real"):
            RdaktConfig(entity_strategies={"EMAIL": "not_real"})  # type: ignore[call-arg]
