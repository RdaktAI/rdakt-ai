"""Event callbacks — hook into detection and anonymization events."""

import json

import httpx
from rdakt_ai import RdaktSyncMiddleware


class FakeTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"Got it: {content}"}}]},
        )


print("=== Event Callbacks ===")
print()

middleware = RdaktSyncMiddleware(
    inner=FakeTransport(),
    on_entities_detected=lambda entities, text: print(f"  [detected] {len(entities)} entities in: {text[:50]}..."),
    on_anonymized=lambda orig, anon, mapping: print(f"  [anonymized] {len(mapping)} replacements"),
    on_deanonymized=lambda anon, restored: print("  [deanonymized] response restored"),
)

with httpx.Client(transport=middleware) as client:
    response = client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Email john@example.com, card 4111-1111-1111-1111"}]},
    )

print()
data = response.json()
print(f"Final response: {data['choices'][0]['message']['content']}")
