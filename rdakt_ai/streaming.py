"""SSE stream wrappers for deanonymizing streaming LLM responses."""

from __future__ import annotations

import copy
import json
import logging
import zlib
from dataclasses import dataclass
from typing import Any, TypeAlias

import httpx

from rdakt_ai.formats import extract_sse_content, set_sse_content
from rdakt_ai.session import RdaktSession

logger = logging.getLogger("rdakt_ai.streaming")

# Parsed SSE event payload (`data:` JSON). `Any` is intentional at this
# JSON-decode boundary — provider event shapes vary.
SseEvent: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class _StreamState:
    """Mutable state carried between SSE chunks within a single stream.

    HTTP chunked transfer can split an SSE ``data:`` line across two
    raw chunks; ``leftover`` carries the unfinished tail. Once an event
    has been parsed, ``last_event_structure`` keeps a deep-copied JSON
    skeleton so the trailing flush can re-emit a valid SSE frame.
    """

    leftover: str = ""
    last_event_structure: SseEvent | None = None


def _deanonymize_sse_content(data: SseEvent, session: RdaktSession) -> SseEvent:
    """Deanonymize the text content field of an SSE event.

    Uses session.deanonymize_chunk() (stateful) to handle anonymized tokens
    split across multiple SSE events (e.g. ``<SS`` / ``N_1>``).

    Only targets the actual text content field — OpenAI ``delta.content``,
    Anthropic ``delta.text``, Gemini ``candidates[].content.parts[].text`` —
    so that non-content string fields don't corrupt the chunk buffer.
    """
    text = extract_sse_content(data)
    if text is not None:
        fragments = session.deanonymize_chunk(text)
        set_sse_content(data, "".join(fragments))
    return data


def _process_sse_chunk(raw: bytes, session: RdaktSession, state: _StreamState) -> bytes:
    """Process a raw SSE chunk — shared by both async and sync streams.

    HTTP chunked transfer encoding may split an SSE ``data:`` line across
    chunks. We carry incomplete trailing lines in ``state.leftover`` so
    the next call can reassemble the full line before JSON parsing.
    """
    text = raw.decode("utf-8", errors="replace")

    # Prepend any incomplete line left over from the previous chunk
    text = state.leftover + text
    state.leftover = ""

    lines = text.split("\n")

    # If the text does not end with '\n', the last element is an
    # incomplete line — save it for the next chunk.
    if text and not text.endswith("\n"):
        state.leftover = lines.pop()

    result_lines: list[str] = []
    data_buffer: list[str] = []

    for line in lines:
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                if data_buffer:
                    result_lines.extend(_process_data_lines(data_buffer, session, state))
                    data_buffer = []
                result_lines.append(line)
            else:
                data_buffer.append(payload)
        else:
            if data_buffer:
                result_lines.extend(_process_data_lines(data_buffer, session, state))
                data_buffer = []
            result_lines.append(line)

    if data_buffer:
        result_lines.extend(_process_data_lines(data_buffer, session, state))

    return "\n".join(result_lines).encode()


def _process_data_lines(data_lines: list[str], session: RdaktSession, state: _StreamState) -> list[str]:
    """Process accumulated data: lines (multi-line SSE support)."""
    combined = "\n".join(data_lines)
    try:
        parsed = json.loads(combined)
        state.last_event_structure = copy.deepcopy(parsed)
        deanonymized = _deanonymize_sse_content(parsed, session)
        return [f"data: {json.dumps(deanonymized)}"]
    except json.JSONDecodeError:
        logger.debug("Malformed SSE data line (not valid JSON), passing through: %s", combined[:200])
        return [f"data: {line}" for line in data_lines]


def _wrap_as_sse(content: str, last_event_structure: SseEvent) -> bytes:
    """Wrap flushed content in the last seen event structure."""
    structure = copy.deepcopy(last_event_structure)
    set_sse_content(structure, content)
    return f"data: {json.dumps(structure)}\n\n".encode()


class _AsyncGunzipStream(httpx.AsyncByteStream):
    """Decompress a gzip-encoded async byte stream on the fly."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream = stream
        self._decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)

    async def __aiter__(self):
        async for chunk in self._stream:
            try:
                yield self._decompressor.decompress(chunk)
            except zlib.error:
                logger.warning("Gzip decompression failed mid-stream, passing raw chunk")
                yield chunk
        try:
            trailing = self._decompressor.flush()
            if trailing:
                yield trailing
        except zlib.error:
            logger.warning("Gzip flush failed at end of stream")

    async def aclose(self) -> None:
        await self._stream.aclose()


class _SyncGunzipStream(httpx.SyncByteStream):
    """Decompress a gzip-encoded sync byte stream on the fly."""

    def __init__(self, stream: httpx.SyncByteStream) -> None:
        self._stream = stream
        self._decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)

    def __iter__(self):
        for chunk in self._stream:
            try:
                yield self._decompressor.decompress(chunk)
            except zlib.error:
                logger.warning("Gzip decompression failed mid-stream, passing raw chunk")
                yield chunk
        try:
            trailing = self._decompressor.flush()
            if trailing:
                yield trailing
        except zlib.error:
            logger.warning("Gzip flush failed at end of stream")

    def close(self) -> None:
        self._stream.close()


class AsyncDeanonymizingStream(httpx.AsyncByteStream):
    """Wraps an async byte stream, deanonymizing SSE content chunks."""

    def __init__(self, stream: httpx.AsyncByteStream, session: RdaktSession, content_encoding: str = "") -> None:
        if content_encoding.lower() == "gzip":
            stream = _AsyncGunzipStream(stream)
        self._stream = stream
        self._session = session
        self._state = _StreamState()

    async def __aiter__(self):
        async for raw_chunk in self._stream:
            yield _process_sse_chunk(raw_chunk, self._session, self._state)
        # Flush remaining buffer as valid SSE
        flush_fragments = self._session.flush()
        if flush_fragments and self._state.last_event_structure is not None:
            yield _wrap_as_sse("".join(flush_fragments), self._state.last_event_structure)

    async def aclose(self) -> None:
        await self._stream.aclose()


class SyncDeanonymizingStream(httpx.SyncByteStream):
    """Wraps a sync byte stream, deanonymizing SSE content chunks."""

    def __init__(self, stream: httpx.SyncByteStream, session: RdaktSession, content_encoding: str = "") -> None:
        if content_encoding.lower() == "gzip":
            stream = _SyncGunzipStream(stream)
        self._stream = stream
        self._session = session
        self._state = _StreamState()

    def __iter__(self):
        for raw_chunk in self._stream:
            yield _process_sse_chunk(raw_chunk, self._session, self._state)
        flush_fragments = self._session.flush()
        if flush_fragments and self._state.last_event_structure is not None:
            yield _wrap_as_sse("".join(flush_fragments), self._state.last_event_structure)

    def close(self) -> None:
        self._stream.close()
