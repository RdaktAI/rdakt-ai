"""Provider-agnostic helpers for extracting/setting content in LLM response formats.

Supports OpenAI, Anthropic, and Gemini — both streaming (SSE events) and
non-streaming (full response bodies).  Used by the middleware streaming layer,
the CLI demo, and the playground backend so the provider detection logic
lives in one place.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Streaming SSE events
# ---------------------------------------------------------------------------


def extract_sse_content(data: dict[str, Any]) -> str | None:
    """Extract the text content field from a parsed SSE event JSON.

    Returns the text string, or None if the event carries no text content
    (e.g. a ``message_start`` or ``finish`` event).
    """
    # OpenAI: choices[].delta.content
    if "choices" in data:
        for choice in data.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str):
                return content
    # Anthropic: delta.text (only in content_block_delta events)
    elif data.get("type") == "content_block_delta":
        text = data.get("delta", {}).get("text")
        if isinstance(text, str):
            return text
    # Gemini: candidates[].content.parts[].text
    elif "candidates" in data:
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    return part["text"]
    return None


def set_sse_content(data: dict[str, Any], content: str) -> bool:
    """Set the text content field in a parsed SSE event JSON.

    Returns True if the field was found and updated, False otherwise.
    """
    # OpenAI: choices[].delta.content
    if "choices" in data:
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta:
                delta["content"] = content
                return True
    # Anthropic: delta.text
    elif data.get("type") == "content_block_delta":
        delta = data.get("delta", {})
        if "text" in delta:
            delta["text"] = content
            return True
    # Gemini: candidates[].content.parts[].text
    elif "candidates" in data:
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    part["text"] = content
                    return True
    return False


# ---------------------------------------------------------------------------
# Non-streaming response bodies
# ---------------------------------------------------------------------------


def extract_response_content(body: dict[str, Any]) -> str | None:
    """Extract the text content from a complete (non-streaming) LLM response body."""
    try:
        # OpenAI: choices[0].message.content
        if "choices" in body:
            return body["choices"][0]["message"]["content"]
        # Anthropic: content[0].text
        if "content" in body and isinstance(body["content"], list):
            return body["content"][0]["text"]
        # Gemini: candidates[0].content.parts[0].text
        if "candidates" in body:
            return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        pass
    return None
