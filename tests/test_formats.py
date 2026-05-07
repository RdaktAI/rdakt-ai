"""Tests for provider-agnostic response format helpers."""

from typing import Any

from rdakt_ai.formats import extract_response_content, extract_sse_content, set_sse_content


class TestExtractSseContent:
    def test_openai_format(self) -> None:
        data = {"choices": [{"delta": {"content": "hello"}}]}
        assert extract_sse_content(data) == "hello"

    def test_openai_empty_delta(self) -> None:
        data: dict[str, Any] = {"choices": [{"delta": {}}]}
        assert extract_sse_content(data) is None

    def test_anthropic_format(self) -> None:
        data = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}
        assert extract_sse_content(data) == "hello"

    def test_anthropic_non_content_event(self) -> None:
        data = {"type": "message_start", "message": {"id": "msg_1"}}
        assert extract_sse_content(data) is None

    def test_gemini_format(self) -> None:
        data = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert extract_sse_content(data) == "hello"

    def test_gemini_empty_parts(self) -> None:
        data: dict[str, Any] = {"candidates": [{"content": {"parts": []}}]}
        assert extract_sse_content(data) is None

    def test_unknown_format(self) -> None:
        data = {"foo": "bar"}
        assert extract_sse_content(data) is None

    def test_openai_content_null(self) -> None:
        data = {"choices": [{"delta": {"content": None}}]}
        assert extract_sse_content(data) is None


class TestSetSseContent:
    def test_openai_format(self) -> None:
        data = {"choices": [{"delta": {"content": "old"}}]}
        assert set_sse_content(data, "new") is True
        assert data["choices"][0]["delta"]["content"] == "new"

    def test_anthropic_format(self) -> None:
        data: dict[str, Any] = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "old"}}
        assert set_sse_content(data, "new") is True
        assert data["delta"]["text"] == "new"

    def test_gemini_format(self) -> None:
        data: dict[str, Any] = {"candidates": [{"content": {"parts": [{"text": "old"}]}}]}
        assert set_sse_content(data, "new") is True
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "new"

    def test_no_content_field(self) -> None:
        data: dict[str, Any] = {"choices": [{"delta": {}}]}
        assert set_sse_content(data, "new") is False

    def test_unknown_format(self) -> None:
        data = {"foo": "bar"}
        assert set_sse_content(data, "new") is False


class TestExtractResponseContent:
    def test_openai_format(self) -> None:
        body = {"choices": [{"message": {"content": "hello"}}]}
        assert extract_response_content(body) == "hello"

    def test_anthropic_format(self) -> None:
        body = {"content": [{"type": "text", "text": "hello"}]}
        assert extract_response_content(body) == "hello"

    def test_gemini_format(self) -> None:
        body = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert extract_response_content(body) == "hello"

    def test_unknown_format(self) -> None:
        assert extract_response_content({"foo": "bar"}) is None

    def test_malformed_openai(self) -> None:
        assert extract_response_content({"choices": []}) is None

    def test_malformed_anthropic(self) -> None:
        assert extract_response_content({"content": "not a list"}) is None
