"""Audit mode — detect PII without modifying requests."""

import logging

import httpx
from rdakt_ai import RdaktSyncMiddleware
from rdakt_ai.logging import setup_logging

# Enable logging so we can see audit output
setup_logging(level=logging.INFO)


class PassthroughTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})


print("=== Audit Mode ===")
print()
print("Sending request with PII in audit mode (detect-only, no modification):")
print()

middleware = RdaktSyncMiddleware(inner=PassthroughTransport(), mode="audit")

with httpx.Client(transport=middleware) as client:
    client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Contact john@example.com, SSN 123-45-6789"}]},
    )

print()
print("(PII was detected and logged but NOT anonymized in the request)")
