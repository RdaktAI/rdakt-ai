"""Tests for middleware event callbacks."""

import json

import httpx
import pytest

from rdakt_ai.middleware import RdaktMiddleware, RdaktSyncMiddleware


class CapturingTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"Echo: {content}"}}]},
        )


class SyncCapturingTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"Echo: {content}"}}]},
        )


class TestAsyncCallbacks:
    @pytest.mark.asyncio
    async def test_on_entities_detected_fires(self) -> None:
        detected: list[tuple] = []
        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_entities_detected=lambda entities, text: detected.append((len(entities), text)),
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert len(detected) == 1
        assert detected[0][0] >= 1  # at least one entity

    @pytest.mark.asyncio
    async def test_on_anonymized_fires(self) -> None:
        anonymized_calls: list[tuple] = []
        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_anonymized=lambda orig, anon, mapping: anonymized_calls.append((orig, anon, mapping)),
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert len(anonymized_calls) == 1
        assert "john@example.com" in anonymized_calls[0][0]
        assert "<EMAIL_1>" in anonymized_calls[0][1]

    @pytest.mark.asyncio
    async def test_on_deanonymized_fires(self) -> None:
        deanon_calls: list[tuple] = []
        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_deanonymized=lambda anon, restored: deanon_calls.append((anon, restored)),
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert len(deanon_calls) == 1

    @pytest.mark.asyncio
    async def test_on_error_fires_on_failure(self) -> None:
        error_calls: list[tuple] = []

        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_error=lambda error, stage: error_calls.append((str(error), stage)),
        )
        # Override pipeline to fail during detection
        middleware._pipeline = None  # type: ignore[assignment]
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert len(error_calls) >= 1

    @pytest.mark.asyncio
    async def test_broken_callback_does_not_crash(self) -> None:
        def bad_callback(entities, text):
            raise ValueError("callback exploded")

        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_entities_detected=bad_callback,
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_no_callback_when_no_entities(self) -> None:
        detected: list = []
        middleware = RdaktMiddleware(
            inner=CapturingTransport(),
            on_entities_detected=lambda entities, text: detected.append(1),
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello world"}]},
            )
        assert len(detected) == 0


class TestSyncCallbacks:
    def test_on_entities_detected_fires(self) -> None:
        detected: list = []
        middleware = RdaktSyncMiddleware(
            inner=SyncCapturingTransport(),
            on_entities_detected=lambda entities, text: detected.append(len(entities)),
        )
        with httpx.Client(transport=middleware) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        assert len(detected) == 1
