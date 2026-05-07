"""Integration tests: full pipeline from detection to deanonymization."""

import json

import httpx

from rdakt_ai import RdaktMiddleware
from rdakt_ai.config import RdaktConfig


class EchoTransport(httpx.AsyncBaseTransport):
    """Echoes back message content in OpenAI response format."""

    def __init__(self) -> None:
        self.last_request_body: dict = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.last_request_body = body
        content = body.get("messages", [{}])[0].get("content", "")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"Response about: {content}"}}],
            },
        )


class TestFullPipeline:
    async def test_email_anonymized_and_restored(self) -> None:
        transport = EchoTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Contact john@example.com please"}]},
            )
        inner_content = transport.last_request_body["messages"][0]["content"]
        assert "john@example.com" not in inner_content
        assert "<EMAIL_1>" in inner_content

        data = response.json()
        assert "john@example.com" in data["choices"][0]["message"]["content"]

    async def test_multiple_entity_types(self) -> None:
        transport = EchoTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "SSN 123-45-6789, email test@foo.com"}]},
            )
        inner_content = transport.last_request_body["messages"][0]["content"]
        assert "123-45-6789" not in inner_content
        assert "test@foo.com" not in inner_content

    async def test_clean_text_passes_through(self) -> None:
        transport = EchoTransport()
        middleware = RdaktMiddleware(inner=transport)
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "What is the weather today?"}]},
            )
        inner_content = transport.last_request_body["messages"][0]["content"]
        assert inner_content == "What is the weather today?"

    async def test_custom_config(self) -> None:
        config = RdaktConfig(  # type: ignore[call-arg]
            custom_patterns={"ACCT": r"\d{4}-\d{4}"},
            entity_strategies={"ACCT": "token"},
        )
        transport = EchoTransport()
        middleware = RdaktMiddleware(inner=transport, config=config)
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Account 1234-5678"}]},
            )
        inner_content = transport.last_request_body["messages"][0]["content"]
        assert "1234-5678" not in inner_content

    async def test_audit_mode_no_modification(self) -> None:
        transport = EchoTransport()
        middleware = RdaktMiddleware(inner=transport, mode="audit")
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@secret.com"}]},
            )
        inner_content = transport.last_request_body["messages"][0]["content"]
        assert "john@secret.com" in inner_content
