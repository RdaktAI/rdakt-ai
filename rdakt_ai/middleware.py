"""RdaktMiddleware — httpx transports that anonymize LLM API requests."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from rdakt_ai.anonymizer import AnonymizationStrategy, Anonymizer
from rdakt_ai.config import RdaktConfig, create_store
from rdakt_ai.detectors.regex import RegexDetector
from rdakt_ai.models import Entity
from rdakt_ai.ontology import OntologyApplier, OwnedKey
from rdakt_ai.pipeline import DetectorStage, RdaktPipeline, RdaktStage
from rdakt_ai.session import RdaktSession
from rdakt_ai.stores import SessionStore

logger = logging.getLogger("rdakt_ai")


# Parsed JSON request/response body. Treated as opaque because the shape
# is dictated by each provider's wire format, not by us — see the
# ``formats`` module for per-provider extractors that turn this into
# typed values.
JsonBody = dict[str, Any]


# ---------------------------------------------------------------------------
# Shared helpers (used by both async and sync middleware)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MiddlewareState:
    """Everything both middleware classes need to handle one request.

    Built once at construction time by :func:`_build_middleware_state` and
    reused per-request thereafter.
    """

    config: RdaktConfig
    store: SessionStore
    pipeline: RdaktPipeline
    anonymizer: Anonymizer
    session: RdaktSession
    mode: str
    error_policy: str
    ontology: OntologyApplier | None


def _build_middleware_state(
    config: RdaktConfig | None,
    store: SessionStore | None,
    session_key: str | None,
    mode: str | None,
    error_policy: str | None,
) -> _MiddlewareState:
    """Build shared state for both sync and async middleware."""
    cfg = config or RdaktConfig()
    st = store or create_store(cfg)

    # Build pipeline from config
    stages: list[RdaktStage] = []
    for entry in cfg.pipeline:
        if isinstance(entry, str):
            name = entry
            opts: dict[str, Any] = {}
        elif isinstance(entry, dict):
            name = next(iter(entry))
            opts = entry[name] or {}
        else:
            raise ValueError(f"Invalid pipeline entry: {entry!r}")

        if name == "regex":
            regex_detector = RegexDetector(
                custom_patterns=cfg.custom_patterns,
                **{k: v for k, v in opts.items() if k != "custom_patterns"},
            )
            stages.append(DetectorStage(regex_detector))
        elif name == "ner":
            from rdakt_ai.detectors.ner import SpacyDetector

            ner_detector = SpacyDetector(**opts)
            stages.append(DetectorStage(ner_detector))
        else:
            raise ValueError(f"Unknown pipeline detector {name!r}. Supported: regex, ner")

    if not stages:
        stages.append(DetectorStage(RegexDetector(custom_patterns=cfg.custom_patterns)))

    pipeline = RdaktPipeline(stages=stages)
    type_strategies: dict[str, AnonymizationStrategy] = {}
    for entity_type, strategy_name in cfg.entity_strategies.items():
        type_strategies[entity_type] = AnonymizationStrategy(strategy_name)
    placeholder_format = cfg.placeholders.build()
    anonymizer = Anonymizer(type_strategies=type_strategies, placeholder_format=placeholder_format)
    session = RdaktSession(session_id=session_key, placeholder_format=placeholder_format)
    if session_key:
        existing = st.load(session_key)
        if existing:
            session.add_mappings(existing)
    md = mode or cfg.mode
    err = error_policy or cfg.on_error
    ontology_applier: OntologyApplier | None = None
    if cfg.ontology is not None and cfg.ontology.fields:
        ontology_applier = OntologyApplier(cfg.ontology, anonymizer=anonymizer, session=session)
    return _MiddlewareState(
        config=cfg,
        store=st,
        pipeline=pipeline,
        anonymizer=anonymizer,
        session=session,
        mode=md,
        error_policy=err,
        ontology=ontology_applier,
    )


class _TextRef:
    """Mutable reference to a text field inside a request body.

    Supports both OpenAI/Anthropic format (``messages[].content``)
    and Gemini format (``contents[].parts[].text``).
    """

    __slots__ = ("_key", "_obj")

    def __init__(self, obj: dict[str, Any], key: str) -> None:
        self._obj = obj
        self._key = key

    @property
    def text(self) -> str:
        return self._obj[self._key]

    @text.setter
    def text(self, value: str) -> None:
        self._obj[self._key] = value


def _extract_text_refs(
    body_json: JsonBody,
    *,
    owned: set[OwnedKey] | None = None,
) -> list[_TextRef]:
    """Extract mutable text references from a request body.

    Handles both ``messages[].content`` (OpenAI/Anthropic) and
    ``contents[].parts[].text`` (Gemini) formats. Positions in *owned*
    (the ontology pass's :class:`~rdakt_ai.ontology.OwnedKey` set) are
    skipped, so the global pipeline doesn't re-process fields the
    ontology already handled.
    """
    refs: list[_TextRef] = []
    owned_keys = owned or set()

    # OpenAI / Anthropic format
    for msg in body_json.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str) and content and OwnedKey(id(msg), "content") not in owned_keys:
            refs.append(_TextRef(msg, "content"))

    # Gemini format: contents[].parts[].text
    for content_item in body_json.get("contents", []):
        for part in content_item.get("parts", []):
            text = part.get("text", "")
            if isinstance(text, str) and text and OwnedKey(id(part), "text") not in owned_keys:
                refs.append(_TextRef(part, "text"))

    return refs


def _fire_callback(callback: Callable | None, *args: Any) -> None:
    """Fire a callback, catching and logging any exceptions."""
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        name = getattr(callback, "__name__", repr(callback))
        logger.warning("Event callback %s failed", name, exc_info=True)


async def _process_text_refs_async(
    text_refs: list[_TextRef],
    pipeline: RdaktPipeline,
    anonymizer: Anonymizer,
    session: RdaktSession,
    on_entities_detected: Callable[[list[Entity], str], None] | None = None,
    on_anonymized: Callable[[str, str, dict[str, str]], None] | None = None,
) -> bool:
    """Detect and anonymize text refs asynchronously. Returns True if any were modified."""
    modified = False
    for ref in text_refs:
        entities = await pipeline.detect(ref.text)
        if not entities:
            continue
        _fire_callback(on_entities_detected, entities, ref.text)
        anonymized_text, mapping = anonymizer.anonymize(ref.text, entities, session=session)
        session.add_mappings(mapping)
        _fire_callback(on_anonymized, ref.text, anonymized_text, mapping)
        ref.text = anonymized_text
        modified = True
    return modified


def _process_text_refs_sync(
    text_refs: list[_TextRef],
    pipeline: RdaktPipeline,
    anonymizer: Anonymizer,
    session: RdaktSession,
    on_entities_detected: Callable[[list[Entity], str], None] | None = None,
    on_anonymized: Callable[[str, str, dict[str, str]], None] | None = None,
) -> bool:
    """Detect and anonymize text refs synchronously. Returns True if any were modified."""
    modified = False
    for ref in text_refs:
        entities = pipeline.detect_sync(ref.text)
        if not entities:
            continue
        _fire_callback(on_entities_detected, entities, ref.text)
        anonymized_text, mapping = anonymizer.anonymize(ref.text, entities, session=session)
        session.add_mappings(mapping)
        _fire_callback(on_anonymized, ref.text, anonymized_text, mapping)
        ref.text = anonymized_text
        modified = True
    return modified


def _deanonymize_response_body(body_bytes: bytes, session: RdaktSession) -> bytes | None:
    """Deanonymize response body. Returns new body bytes or None on failure."""
    try:
        body_json = json.loads(body_bytes.decode("utf-8"))
        deanonymized = session.deanonymize_structured(body_json)
        return json.dumps(deanonymized).encode("utf-8")
    except Exception:
        logger.warning("Failed to deanonymize response", exc_info=True)
        return None


def _rebuild_request(request: httpx.Request, new_body: bytes) -> httpx.Request:
    """Create a new request with the modified body and updated Content-Length."""
    headers = dict(request.headers)
    headers["content-length"] = str(len(new_body))
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=headers,
        content=new_body,
    )


def _strip_encoding_headers(headers: httpx.Headers, content_length: int) -> dict[str, str]:
    """Build response headers with content-encoding removed and content-length updated.

    After decoding + deanonymizing, the body is plain bytes — keeping the original
    content-encoding (e.g. gzip) would cause httpx to try decompressing again.
    """
    cleaned = {k: v for k, v in headers.items() if k.lower() not in ("content-encoding", "content-length")}
    cleaned["content-length"] = str(content_length)
    return cleaned


# ---------------------------------------------------------------------------
# Async middleware
# ---------------------------------------------------------------------------


class RdaktMiddleware(httpx.AsyncBaseTransport):
    """Composable anonymization middleware for LLM API calls.

    Implements httpx.AsyncBaseTransport so it can be passed directly to
    OpenAI, Anthropic, and Google Gemini SDK constructors.
    """

    def __init__(
        self,
        *,
        inner: httpx.AsyncBaseTransport | None = None,
        config: RdaktConfig | None = None,
        store: SessionStore | None = None,
        session_key: str | None = None,
        mode: str | None = None,
        error_policy: str | None = None,
        on_entities_detected: Callable[[list[Entity], str], None] | None = None,
        on_anonymized: Callable[[str, str, dict[str, str]], None] | None = None,
        on_deanonymized: Callable[[str, str], None] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
    ) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()
        state = _build_middleware_state(config, store, session_key, mode, error_policy)
        self._config = state.config
        self._store = state.store
        self._pipeline = state.pipeline
        self._anonymizer = state.anonymizer
        self._session = state.session
        self._mode = state.mode
        self._error_policy = state.error_policy
        self._ontology = state.ontology
        self._on_entities_detected = on_entities_detected
        self._on_anonymized = on_anonymized
        self._on_deanonymized = on_deanonymized
        self._on_error_cb = on_error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await self._handle_request(request)
        except Exception as exc:
            if self._error_policy == "block":
                raise
            _fire_callback(self._on_error_cb, exc, "detection")
            logger.warning("Rdakt AI detection error, forwarding unanonymized", exc_info=True)
            return await self._inner.handle_async_request(request)

    async def _handle_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = request.content
        if not body_bytes:
            return await self._inner.handle_async_request(request)

        body_str = body_bytes.decode("utf-8")
        try:
            body_json: dict[str, Any] = json.loads(body_str)
        except json.JSONDecodeError:
            return await self._inner.handle_async_request(request)

        ontology_owned: set[OwnedKey] = set()
        if self._ontology is not None and self._mode != "audit":
            ontology_owned = self._ontology.apply(body_json)

        text_refs = _extract_text_refs(body_json, owned=ontology_owned)
        if not text_refs and not ontology_owned:
            return await self._inner.handle_async_request(request)

        if self._mode == "audit":
            for ref in text_refs:
                entities = await self._pipeline.detect(ref.text)
                if entities:
                    logger.info(
                        "Audit mode: detected %d entities in message: %s",
                        len(entities),
                        [(e.type, e.value) for e in entities],
                    )
            return await self._inner.handle_async_request(request)

        modified = await _process_text_refs_async(
            text_refs,
            self._pipeline,
            self._anonymizer,
            self._session,
            on_entities_detected=self._on_entities_detected,
            on_anonymized=self._on_anonymized,
        )
        if ontology_owned:
            modified = True

        if modified:
            self._store.save(self._session.session_id, self._session.entity_map)
            new_body = json.dumps(body_json).encode("utf-8")
            request = _rebuild_request(request, new_body)

        response = await self._inner.handle_async_request(request)

        # Deanonymize the response
        if self._session.has_mappings:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                response = self._wrap_streaming_response(response)
            else:
                await response.aread()
                anon_body = response.content
                new_body_bytes = _deanonymize_response_body(anon_body, self._session)
                if new_body_bytes is not None:
                    _fire_callback(
                        self._on_deanonymized,
                        anon_body.decode("utf-8"),
                        new_body_bytes.decode("utf-8"),
                    )
                    response = httpx.Response(
                        status_code=response.status_code,
                        headers=_strip_encoding_headers(response.headers, len(new_body_bytes)),
                        content=new_body_bytes,
                    )

        return response

    def _wrap_streaming_response(self, response: httpx.Response) -> httpx.Response:
        """Wrap a streaming response with deanonymizing stream.

        Strips ``content-encoding`` so the SDK's automatic decompression
        layer sits *below* the stream we read, and we always see plain-text
        SSE bytes (Anthropic, for example, returns gzip-compressed streams).
        """
        from rdakt_ai.streaming import AsyncDeanonymizingStream

        stream = response.stream
        if not isinstance(stream, httpx.AsyncByteStream):
            logger.warning("Expected AsyncByteStream, got %s; skipping deanonymization", type(stream).__name__)
            return response
        content_encoding = response.headers.get("content-encoding", "")
        wrapped_stream = AsyncDeanonymizingStream(stream, self._session, content_encoding=content_encoding)
        cleaned_headers = {
            k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length")
        }
        return httpx.Response(
            status_code=response.status_code,
            headers=cleaned_headers,
            stream=wrapped_stream,
        )


# ---------------------------------------------------------------------------
# Sync middleware
# ---------------------------------------------------------------------------


class RdaktSyncMiddleware(httpx.BaseTransport):
    """Sync anonymization middleware for LLM API calls.

    Implements httpx.BaseTransport for use with httpx.Client (synchronous).
    """

    def __init__(
        self,
        *,
        inner: httpx.BaseTransport | None = None,
        config: RdaktConfig | None = None,
        store: SessionStore | None = None,
        session_key: str | None = None,
        mode: str | None = None,
        error_policy: str | None = None,
        on_entities_detected: Callable[[list[Entity], str], None] | None = None,
        on_anonymized: Callable[[str, str, dict[str, str]], None] | None = None,
        on_deanonymized: Callable[[str, str], None] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
    ) -> None:
        self._inner = inner or httpx.HTTPTransport()
        state = _build_middleware_state(config, store, session_key, mode, error_policy)
        self._config = state.config
        self._store = state.store
        self._pipeline = state.pipeline
        self._anonymizer = state.anonymizer
        self._session = state.session
        self._mode = state.mode
        self._error_policy = state.error_policy
        self._ontology = state.ontology
        self._on_entities_detected = on_entities_detected
        self._on_anonymized = on_anonymized
        self._on_deanonymized = on_deanonymized
        self._on_error_cb = on_error

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._handle_request(request)
        except Exception as exc:
            if self._error_policy == "block":
                raise
            _fire_callback(self._on_error_cb, exc, "detection")
            logger.warning("Rdakt AI detection error, forwarding unanonymized", exc_info=True)
            return self._inner.handle_request(request)

    def _handle_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = request.content
        if not body_bytes:
            return self._inner.handle_request(request)

        try:
            body_json: dict[str, Any] = json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            return self._inner.handle_request(request)

        ontology_owned: set[OwnedKey] = set()
        if self._ontology is not None and self._mode != "audit":
            ontology_owned = self._ontology.apply(body_json)

        text_refs = _extract_text_refs(body_json, owned=ontology_owned)
        if not text_refs and not ontology_owned:
            return self._inner.handle_request(request)

        if self._mode == "audit":
            for ref in text_refs:
                entities = self._pipeline.detect_sync(ref.text)
                if entities:
                    logger.info(
                        "Audit mode: detected %d entities in message: %s",
                        len(entities),
                        [(e.type, e.value) for e in entities],
                    )
            return self._inner.handle_request(request)

        modified = _process_text_refs_sync(
            text_refs,
            self._pipeline,
            self._anonymizer,
            self._session,
            on_entities_detected=self._on_entities_detected,
            on_anonymized=self._on_anonymized,
        )
        if ontology_owned:
            modified = True

        if modified:
            self._store.save(self._session.session_id, self._session.entity_map)
            new_body = json.dumps(body_json).encode("utf-8")
            request = _rebuild_request(request, new_body)

        response = self._inner.handle_request(request)

        if self._session.has_mappings:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                response = self._wrap_streaming_response(response)
            else:
                response.read()
                anon_body = response.content
                new_body_bytes = _deanonymize_response_body(anon_body, self._session)
                if new_body_bytes is not None:
                    _fire_callback(
                        self._on_deanonymized,
                        anon_body.decode("utf-8"),
                        new_body_bytes.decode("utf-8"),
                    )
                    response = httpx.Response(
                        status_code=response.status_code,
                        headers=_strip_encoding_headers(response.headers, len(new_body_bytes)),
                        content=new_body_bytes,
                    )

        return response

    def _wrap_streaming_response(self, response: httpx.Response) -> httpx.Response:
        """Wrap a streaming response with deanonymizing stream."""
        from rdakt_ai.streaming import SyncDeanonymizingStream

        stream = response.stream
        if not isinstance(stream, httpx.SyncByteStream):
            logger.warning("Expected SyncByteStream, got %s; skipping deanonymization", type(stream).__name__)
            return response
        content_encoding = response.headers.get("content-encoding", "")
        wrapped_stream = SyncDeanonymizingStream(stream, self._session, content_encoding=content_encoding)
        cleaned_headers = {
            k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length")
        }
        return httpx.Response(
            status_code=response.status_code,
            headers=cleaned_headers,
            stream=wrapped_stream,
        )
