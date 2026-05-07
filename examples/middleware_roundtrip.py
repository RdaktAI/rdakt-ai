"""Middleware roundtrip — full request/response cycle with a fake transport."""

import json

import httpx
from rdakt_ai import RdaktSyncMiddleware


class FakeTransport(httpx.BaseTransport):
    """Simulates an LLM API that echoes back the message content."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        print(f"  [LLM received]: {content}")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"Noted: {content}"}}]},
        )


print("=== Middleware Roundtrip ===")
print()

middleware = RdaktSyncMiddleware(inner=FakeTransport())

with httpx.Client(transport=middleware) as client:
    response = client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "My email is john@example.com and SSN is 123-45-6789"}]},
    )

data = response.json()
print(f"  [App sees]:    {data['choices'][0]['message']['content']}")
