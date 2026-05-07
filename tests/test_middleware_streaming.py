"""Tests for streaming SSE response deanonymization."""

import json

import httpx

from rdakt_ai.formats import set_sse_content
from rdakt_ai.middleware import RdaktMiddleware, RdaktSyncMiddleware
from rdakt_ai.session import RdaktSession
from rdakt_ai.streaming import (
    SyncDeanonymizingStream,
    _deanonymize_sse_content,
    _process_data_lines,
    _process_sse_chunk,
    _StreamState,
    _wrap_as_sse,
)


def _make_sse_chunk(content: str) -> bytes:
    """Helper to create a single SSE data line."""
    data = json.dumps({"choices": [{"delta": {"content": content}}]})
    return f"data: {data}\n\n".encode()


class _AsyncIterableStream(httpx.AsyncByteStream):
    """Wraps an async iterable into a proper AsyncByteStream for httpx."""

    def __init__(self, iterable):
        self._iterable = iterable

    async def __aiter__(self):
        async for chunk in self._iterable:
            yield chunk


class StreamingEchoTransport(httpx.AsyncBaseTransport):
    """Async transport that streams back request content as SSE chunks."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        words = content.split()

        async def stream():
            for word in words:
                yield _make_sse_chunk(word + " ")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncIterableStream(stream()),
        )


class TestStreamingDeanonymization:
    async def test_streaming_deanonymizes_tokens(self) -> None:
        middleware = RdaktMiddleware(inner=StreamingEchoTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        # Read the full streamed response
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        # The email should be restored in the stream
        assert "john@example.com" in full_response
        assert "<EMAIL_1>" not in full_response

    async def test_streaming_passthrough_no_entities(self) -> None:
        middleware = RdaktMiddleware(inner=StreamingEchoTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello world"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "Hello" in full_response

    async def test_streaming_done_sentinel_passes_through(self) -> None:
        middleware = RdaktMiddleware(inner=StreamingEchoTransport())
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "[DONE]" in full_response

    async def test_streaming_skips_structural_fields(self) -> None:
        """Structural fields like 'role' and 'model' should not be deanonymized."""

        class StructuralFieldTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                async def stream():
                    data = json.dumps({"choices": [{"delta": {"role": "assistant"}}]})
                    yield f"data: {data}\n\n".encode()
                    data2 = json.dumps({"choices": [{"delta": {"content": "<EMAIL_1>"}}]})
                    yield f"data: {data2}\n\n".encode()
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_AsyncIterableStream(stream()),
                )

        middleware = RdaktMiddleware(inner=StructuralFieldTransport())
        middleware._session.add_mappings({"<EMAIL_1>": "john@example.com"})
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        # Content field deanonymized, but role field untouched
        assert "john@example.com" in full_response
        assert '"role": "assistant"' in full_response or '"role":"assistant"' in full_response

    async def test_streaming_deanonymizes_split_tokens(self) -> None:
        """Tokens split across SSE events (like real LLM tokenizers) must be deanonymized."""

        class SplitTokenTransport(httpx.AsyncBaseTransport):
            """Simulates an LLM that splits <SSN_1> across multiple SSE events."""

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                async def stream():
                    # LLM echoes back the anonymized token split across subword tokens
                    for fragment in ["- SSN: ", "<", "SS", "N_", "1", ">"]:
                        yield _make_sse_chunk(fragment)
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_AsyncIterableStream(stream()),
                )

        middleware = RdaktMiddleware(inner=SplitTokenTransport())
        middleware._session.add_mappings({"<SSN_1>": "123-45-6789"})
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "123-45-6789" in full_response
        assert "<SSN_1>" not in full_response

    async def test_streaming_non_content_fields_dont_corrupt_buffer(self) -> None:
        """Non-content string fields (service_tier etc.) must not interfere with chunk buffer."""

        class InterleavedFieldTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                async def stream():
                    # Content event ending with '<' (will be buffered)
                    yield _make_sse_chunk("SSN: <")
                    # Event with non-content string fields (service_tier, etc.)
                    data = json.dumps(
                        {
                            "id": "chatcmpl-1",
                            "service_tier": "default",
                            "choices": [{"delta": {}}],
                        }
                    )
                    yield f"data: {data}\n\n".encode()
                    # Content event completing the token
                    yield _make_sse_chunk("SSN_1>")
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_AsyncIterableStream(stream()),
                )

        middleware = RdaktMiddleware(inner=InterleavedFieldTransport())
        middleware._session.add_mappings({"<SSN_1>": "123-45-6789"})
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "123-45-6789" in full_response
        assert "SSN_1>" not in full_response

    async def test_streaming_deanonymizes_escaped_tokens(self) -> None:
        """LLMs may backslash-escape angle brackets: \\<SSN_1\\> must still be deanonymized."""

        class EscapedTokenTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                async def stream():
                    # LLM escapes angle brackets in markdown streaming
                    for fragment in ["- SSN: ", "\\<", "SSN_1", "\\>"]:
                        yield _make_sse_chunk(fragment)
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_AsyncIterableStream(stream()),
                )

        middleware = RdaktMiddleware(inner=EscapedTokenTransport())
        middleware._session.add_mappings({"<SSN_1>": "123-45-6789"})
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "123-45-6789" in full_response
        assert "SSN_1" not in full_response

    async def test_streaming_deanonymizes_when_http_chunk_splits_sse_line(self) -> None:
        """HTTP chunked transfer may split an SSE data: line across chunks.

        When this happens, _process_sse_chunk must buffer the incomplete line
        and prepend it to the next chunk, so JSON parsing succeeds and
        deanonymize_chunk is called on the content.
        """

        class SplitHttpChunkTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # Build three SSE events with tokens
                event1 = _make_sse_chunk("- SSN: <SSN_1>")
                event2 = _make_sse_chunk("- CC: <CREDIT_CARD_1>")
                done = b"data: [DONE]\n\n"

                # Simulate HTTP chunks splitting mid-line:
                # chunk 1 = complete event1 + first half of event2's data: line
                raw = event1 + event2
                mid = len(raw) // 2

                async def stream():
                    yield raw[:mid]  # cuts event2's data: line in half
                    yield raw[mid:]  # rest of event2
                    yield done

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_AsyncIterableStream(stream()),
                )

        middleware = RdaktMiddleware(inner=SplitHttpChunkTransport())
        middleware._session.add_mappings(
            {
                "<SSN_1>": "123-45-6789",
                "<CREDIT_CARD_1>": "4111-1111-1111-1111",
            }
        )
        async with httpx.AsyncClient(transport=middleware) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        full_response = b"".join(chunks).decode()
        assert "123-45-6789" in full_response
        assert "4111-1111-1111-1111" in full_response
        assert "<SSN_1>" not in full_response
        assert "<CREDIT_CARD_1>" not in full_response


class TestStreamingHelpers:
    def test_deanonymize_sse_content_openai(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data = {"choices": [{"delta": {"content": "<EMAIL_1>"}}]}
        result = _deanonymize_sse_content(data, session)
        assert result["choices"][0]["delta"]["content"] == "a@b.com"

    def test_deanonymize_sse_content_anthropic(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "<EMAIL_1>"}}
        result = _deanonymize_sse_content(data, session)
        assert result["delta"]["text"] == "a@b.com"

    def test_deanonymize_sse_content_ignores_non_content_fields(self) -> None:
        """Non-content string fields must not corrupt the deanonymize_chunk buffer."""
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        # Process an event that ends with '<' buffered
        data1 = {"choices": [{"delta": {"content": "hello <"}}]}
        _deanonymize_sse_content(data1, session)
        # A non-content event with extra string fields should NOT flush the buffer
        data2 = {"id": "chatcmpl-1", "service_tier": "default", "choices": [{"delta": {}}]}
        _deanonymize_sse_content(data2, session)
        # The next content event should complete the token
        data3 = {"choices": [{"delta": {"content": "EMAIL_1> bye"}}]}
        _deanonymize_sse_content(data3, session)
        assert data3["choices"][0]["delta"]["content"] == "a@b.com bye"

    def test_process_data_lines_invalid_json(self) -> None:
        session = RdaktSession(session_id="t")
        state = _StreamState()
        result = _process_data_lines(["not valid json"], session, state)
        assert result == ["data: not valid json"]

    def test_wrap_as_sse(self) -> None:
        structure = {"choices": [{"delta": {"content": "old"}}]}
        result = _wrap_as_sse("new content", structure)
        parsed = json.loads(result.decode().removeprefix("data: ").strip())
        assert parsed["choices"][0]["delta"]["content"] == "new content"

    def test_set_sse_content_openai(self) -> None:
        data = {"choices": [{"delta": {"content": "old"}}]}
        assert set_sse_content(data, "new") is True
        assert data["choices"][0]["delta"]["content"] == "new"

    def test_set_sse_content_no_target(self) -> None:
        data: dict = {"choices": [{"delta": {}}]}
        assert set_sse_content(data, "x") is False

    def test_deanonymize_sse_content_gemini(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data = {"candidates": [{"content": {"parts": [{"text": "<EMAIL_1>"}]}}]}
        _deanonymize_sse_content(data, session)
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "a@b.com"

    def test_deanonymize_sse_content_escaped_openai(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data = {"choices": [{"delta": {"content": "\\<EMAIL_1\\>"}}]}
        _deanonymize_sse_content(data, session)
        assert data["choices"][0]["delta"]["content"] == "a@b.com"

    def test_deanonymize_sse_content_escaped_anthropic(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data: dict = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "\\<EMAIL_1\\>"}}
        _deanonymize_sse_content(data, session)
        assert data["delta"]["text"] == "a@b.com"

    def test_deanonymize_sse_content_escaped_gemini(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        data = {"candidates": [{"content": {"parts": [{"text": "\\<EMAIL_1\\>"}]}}]}
        _deanonymize_sse_content(data, session)
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "a@b.com"

    def test_deanonymize_sse_content_skips_non_content_events(self) -> None:
        """Non-content SSE events (message_start, etc.) should pass through unchanged."""
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        # Anthropic message_start
        data = {"type": "message_start", "message": {"id": "msg_1", "role": "assistant"}}
        _deanonymize_sse_content(data, session)
        assert data == {"type": "message_start", "message": {"id": "msg_1", "role": "assistant"}}

    def test_process_sse_chunk_openai_format(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        state = _StreamState()
        chunk = b'data: {"choices": [{"delta": {"content": "<EMAIL_1>"}}]}\n\n'
        result = _process_sse_chunk(chunk, session, state)
        assert b"a@b.com" in result
        assert state.last_event_structure is not None

    def test_process_sse_chunk_anthropic_format(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        state = _StreamState()
        data = json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "<EMAIL_1>"}})
        chunk = f"data: {data}\n\n".encode()
        result = _process_sse_chunk(chunk, session, state)
        assert b"a@b.com" in result

    def test_process_sse_chunk_gemini_format(self) -> None:
        session = RdaktSession(session_id="t", entity_map={"<EMAIL_1>": "a@b.com"})
        state = _StreamState()
        data = json.dumps({"candidates": [{"content": {"parts": [{"text": "<EMAIL_1>"}]}}]})
        chunk = f"data: {data}\n\n".encode()
        result = _process_sse_chunk(chunk, session, state)
        assert b"a@b.com" in result


class TestSyncStreamingDeanonymization:
    def test_sync_streaming_deanonymizes_tokens(self) -> None:
        class SyncStreamingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                content = body["messages"][0]["content"]

                class IterStream(httpx.SyncByteStream):
                    def __iter__(self):
                        for word in content.split():
                            yield _make_sse_chunk(word + " ")
                        yield b"data: [DONE]\n\n"

                    def close(self) -> None:
                        pass

                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=IterStream(),
                )

        middleware = RdaktSyncMiddleware(inner=SyncStreamingTransport())
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        chunks = list(response.iter_bytes())
        full_response = b"".join(chunks).decode()
        assert "john@example.com" in full_response
        assert "<EMAIL_1>" not in full_response

    def test_sync_deanonymizing_stream_close(self) -> None:
        closed = []

        class FakeStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"data: [DONE]\n\n"

            def close(self) -> None:
                closed.append(True)

        session = RdaktSession(session_id="t")
        stream = SyncDeanonymizingStream(FakeStream(), session)
        list(stream)  # consume
        stream.close()
        assert len(closed) == 1
