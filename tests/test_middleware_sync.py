"""Tests for RdaktSyncMiddleware — sync httpx transport."""

import gzip
import json

import httpx

from rdakt_ai.middleware import RdaktSyncMiddleware, _rebuild_request, _strip_encoding_headers


class SyncCapturingTransport(httpx.BaseTransport):
    """Sync test transport that captures request body."""

    def __init__(self, response_json: dict | None = None) -> None:
        self.captured_body: dict = {}
        self._response_json = response_json or {"choices": [{"message": {"content": "OK"}}]}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.captured_body = json.loads(request.content.decode())
        return httpx.Response(200, json=self._response_json)


class SyncEchoTransport(httpx.BaseTransport):
    """Sync test transport that echoes back message content."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"I see {content}"}}]},
        )


class TestSyncMiddlewareBasic:
    def test_passthrough_no_pii(self) -> None:
        transport = SyncCapturingTransport()
        middleware = RdaktSyncMiddleware(inner=transport)
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello world"}]},
            )
        assert response.status_code == 200
        assert transport.captured_body["messages"][0]["content"] == "Hello world"

    def test_anonymizes_email(self) -> None:
        transport = SyncCapturingTransport()
        middleware = RdaktSyncMiddleware(inner=transport)
        with httpx.Client(transport=middleware) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email me at john@example.com"}]},
            )
        content = transport.captured_body["messages"][0]["content"]
        assert "john@example.com" not in content
        assert "<EMAIL_1>" in content

    def test_deanonymizes_response(self) -> None:
        middleware = RdaktSyncMiddleware(inner=SyncEchoTransport())
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        data = response.json()
        assert "john@example.com" in data["choices"][0]["message"]["content"]

    def test_audit_mode(self) -> None:
        transport = SyncCapturingTransport()
        middleware = RdaktSyncMiddleware(inner=transport, mode="audit")
        with httpx.Client(transport=middleware) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        content = transport.captured_body["messages"][0]["content"]
        assert "john@example.com" in content

    def test_fail_open(self) -> None:
        class OkTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        middleware = RdaktSyncMiddleware(inner=OkTransport(), error_policy="warn_and_forward")
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 200


class TestRebuildRequestContentLength:
    """Regression: _rebuild_request must update Content-Length after anonymization."""

    def test_content_length_matches_new_body(self) -> None:
        original_body = b'{"messages":[{"role":"user","content":"Email john.smith@example.com please"}]}'
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=original_body)
        # Simulate a shorter body after anonymization
        new_body = b'{"messages":[{"role":"user","content":"Email <EMAIL_1> please"}]}'
        rebuilt = _rebuild_request(request, new_body)
        assert rebuilt.headers["content-length"] == str(len(new_body))
        assert rebuilt.content == new_body

    def test_anonymized_request_has_correct_content_length(self) -> None:
        """End-to-end: the request arriving at the inner transport must have matching Content-Length."""
        captured_headers: dict[str, str] = {}

        class HeaderCapturingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured_headers.update(dict(request.headers))
                captured_headers["_body_len"] = str(len(request.content))
                return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        middleware = RdaktSyncMiddleware(inner=HeaderCapturingTransport())
        with httpx.Client(transport=middleware) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Contact john@example.com today"}]},
            )
        assert captured_headers["content-length"] == captured_headers["_body_len"]


class TestResponseNotRead:
    """Regression: sync middleware must read() the response before accessing .content."""

    def test_deanonymizes_unread_response(self) -> None:
        """Transport returns a response with stream (not pre-read) — middleware must handle it."""

        class StreamingJsonTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                content = body["messages"][0]["content"]
                resp_bytes = json.dumps({"choices": [{"message": {"content": f"echo: {content}"}}]}).encode()
                # Return a streaming response (not pre-read) by using stream=
                return httpx.Response(
                    200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(resp_bytes)
                )

        middleware = RdaktSyncMiddleware(inner=StreamingJsonTransport())
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        data = response.json()
        assert "john@example.com" in data["choices"][0]["message"]["content"]


class TestGzipResponseDeanonymization:
    """Regression: rebuilt response must not carry content-encoding from the original."""

    def test_strip_encoding_headers_removes_gzip(self) -> None:
        headers = httpx.Headers(
            {"content-type": "application/json", "content-encoding": "gzip", "content-length": "99"}
        )
        cleaned = _strip_encoding_headers(headers, 42)
        assert "content-encoding" not in cleaned
        assert cleaned["content-length"] == "42"
        assert cleaned["content-type"] == "application/json"

    def test_deanonymize_gzip_response(self) -> None:
        """Inner transport returns gzip-encoded body — middleware must deanonymize without error."""

        class GzipEchoTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                content = body["messages"][0]["content"]
                resp_json = json.dumps({"choices": [{"message": {"content": f"echo: {content}"}}]})
                compressed = gzip.compress(resp_json.encode())
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json", "content-encoding": "gzip"},
                    content=compressed,
                )

        middleware = RdaktSyncMiddleware(inner=GzipEchoTransport())
        with httpx.Client(transport=middleware) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Email john@example.com"}]},
            )
        data = response.json()
        assert "john@example.com" in data["choices"][0]["message"]["content"]
