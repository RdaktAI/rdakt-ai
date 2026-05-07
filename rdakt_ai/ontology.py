"""Per-field PII ontology: JSONPath-keyed rules that override or augment
the global detection pipeline at specific positions in a request body.

A rule selects nodes via a small JSONPath subset (``$.a.b[*].c``) and
declares one of:

* ``detect``: restrict the global detector pipeline to a subset of types.
* ``detect_via_regex``: an inline pattern that produces entities locally.
* ``replace_with_synthetic``: replace the entire field value with a
  synthetic generated from a ``format`` string (e.g. ``"user-{n:06d}"``).

Rules can also override the masking strategy (``token`` / ``synthetic`` /
``hybrid`` / ``hash``). ``hash`` is deterministic within a session via a
per-session salt so the same input maps to the same opaque token across
turns.

The applier mutates the body in place and returns the set of
``(id(container), key)`` pairs it owns, so the middleware can skip those
positions when running global detection on the rest of the body.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Union

from rdakt_ai.anonymizer import AnonymizationStrategy, Anonymizer
from rdakt_ai.detectors.regex import RegexDetector
from rdakt_ai.models import Entity

if TYPE_CHECKING:
    from rdakt_ai.config import FieldRule, OntologyConfig
    from rdakt_ai.session import RdaktSession


# `Any` here is the parsed-JSON document tree — opaque by definition since
# the shape is decided by the caller's payload, not us. All public callers
# treat it as such.
JsonValue = Any


# A position in a JSON document a rule has already handled. ``container``
# is the parent dict or list (identified by ``id()`` so we can compare
# without hashing the container itself), ``key`` is its dict-key or
# list-index. Used by the middleware to skip positions during the global
# pipeline pass.
class OwnedKey(NamedTuple):
    container_id: int
    key: Union[str, int]


# ---------------------------------------------------------------------------
# JSONPath subset
# ---------------------------------------------------------------------------

# Valid syntax (a deliberately small slice of full JSONPath):
#   $                      root
#   .name                  child by key
#   [*]                    array wildcard
# Anything else is rejected at config load with a clear error.

_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class _Field:
    name: str


@dataclass(frozen=True)
class _ArrayWildcard:
    pass


_Segment = _Field | _ArrayWildcard


def parse_jsonpath(path: str) -> list[_Segment]:
    if not path.startswith("$"):
        raise ValueError(f"JSONPath {path!r} must start with '$'")
    rest = path[1:]
    segments: list[_Segment] = []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == ".":
            i += 1
            m = _FIELD_RE.match(rest, i)
            if not m:
                raise ValueError(f"JSONPath {path!r}: expected field name after '.' at offset {i + 1}")
            segments.append(_Field(m.group()))
            i = m.end()
        elif ch == "[":
            if rest[i : i + 3] != "[*]":
                raise ValueError(f"JSONPath {path!r}: only '[*]' array wildcard is supported (got {rest[i : i + 3]!r})")
            segments.append(_ArrayWildcard())
            i += 3
        else:
            raise ValueError(f"JSONPath {path!r}: unexpected character {ch!r} at offset {i + 1}")
    return segments


# Callback invoked when a JSONPath fully matches. Receives the parent
# container (dict or list) and the dict-key / list-index of the matched
# node so the caller can mutate the slot in place.
MatchCallback = Callable[[JsonValue, Union[str, int], JsonValue], None]


def _walk(
    node: JsonValue,
    segments: list[_Segment],
    on_match: MatchCallback,
) -> None:
    """Visit every node matching *segments* under *node* and invoke *on_match*."""
    _walk_segments(node, segments, 0, on_match, parent=None, key=None)


def _walk_segments(
    node: JsonValue,
    segments: list[_Segment],
    idx: int,
    on_match: MatchCallback,
    parent: JsonValue,
    key: Union[str, int, None],
) -> None:
    if idx == len(segments):
        if parent is not None and key is not None:
            on_match(parent, key, node)
        return

    segment = segments[idx]
    if isinstance(segment, _Field):
        if isinstance(node, dict) and segment.name in node:
            _walk_segments(node[segment.name], segments, idx + 1, on_match, parent=node, key=segment.name)
    elif isinstance(segment, _ArrayWildcard) and isinstance(node, list):
        for i, item in enumerate(node):
            _walk_segments(item, segments, idx + 1, on_match, parent=node, key=i)


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


@dataclass
class _CompiledRule:
    rule: FieldRule
    segments: list[_Segment]
    regex: re.Pattern[str] | None = None
    type_counters: dict[str, int] = field(default_factory=dict)


class OntologyApplier:
    """Applies an :class:`OntologyConfig` to a request body in place.

    Returns the set of ``(id(container), key)`` pairs it owns so the
    global-pipeline pass can skip those positions.
    """

    def __init__(
        self,
        ontology: OntologyConfig,
        *,
        anonymizer: Anonymizer,
        session: RdaktSession,
    ) -> None:
        self._anonymizer = anonymizer
        self._session = session
        self._compiled: list[_CompiledRule] = []
        for rule in ontology.fields:
            segments = parse_jsonpath(rule.path)
            compiled = _CompiledRule(rule=rule, segments=segments)
            if rule.detect_via_regex:
                # Already validated at config load, but compile here too.
                compiled.regex = re.compile(rule.detect_via_regex)
            self._compiled.append(compiled)

    def apply(self, body: JsonValue) -> set[OwnedKey]:
        """Apply every compiled rule to *body* in place.

        Returns the set of (container_id, key) positions the ontology
        rewrote — used by the middleware so the global pipeline pass
        skips them.
        """
        owned: set[OwnedKey] = set()

        for compiled in self._compiled:

            def on_match(
                parent: JsonValue,
                key: Union[str, int],
                value: JsonValue,
                _compiled: _CompiledRule = compiled,
            ) -> None:
                if not isinstance(value, str):
                    # Ontology rules only target string-valued fields.
                    return
                new_value = self._apply_rule(_compiled, value)
                parent[key] = new_value
                owned.add(OwnedKey(container_id=id(parent), key=key))

            _walk(body, compiled.segments, on_match)
        return owned

    # -- per-rule application -------------------------------------------------

    def _apply_rule(self, compiled: _CompiledRule, value: str) -> str:
        rule = compiled.rule

        if rule.replace_with_synthetic:
            return self._render_synthetic(compiled, value)

        if rule.detect_via_regex:
            assert compiled.regex is not None
            entity_type = rule.detect_via_regex_as or "CUSTOM"
            entities = [
                Entity(
                    value=m.group(),
                    type=entity_type,
                    start=m.start(),
                    end=m.end(),
                )
                for m in compiled.regex.finditer(value)
            ]
            return self._anonymize_with_entities(value, entities, rule.mask_strategy)

        if rule.detect:
            entities = self._detect_subset(value, rule.detect)
            return self._anonymize_with_entities(value, entities, rule.mask_strategy)

        # No detection source declared: rule is a strategy override only,
        # which without entities is a no-op. Caller should have ensured at
        # least one source via config validation.
        return value

    def _render_synthetic(self, compiled: _CompiledRule, original: str) -> str:
        spec = compiled.rule.replace_with_synthetic
        assert spec is not None
        existing = self._session.get_token_for_value(original)
        if existing is not None:
            return existing
        compiled.type_counters[spec.type] = compiled.type_counters.get(spec.type, 0) + 1
        n = compiled.type_counters[spec.type]
        rendered = spec.format.format(n=n)
        self._session.add_mappings({rendered: original})
        return rendered

    def _detect_subset(self, value: str, allowed_types: list[str]) -> list[Entity]:
        # Use a one-shot regex detector restricted to the allowed types so we
        # don't spend cycles on types the user has explicitly excluded for
        # this field. NER-typed entities aren't supported by ontology rules
        # in this iteration — users who need NER for a specific path can use
        # detect_via_regex or fall through to the global pipeline.
        custom = {t: RegexDetector.builtin_patterns[t] for t in allowed_types if t in RegexDetector.builtin_patterns}
        if not custom:
            return []
        return _SubsetRegexDetector(custom).detect(value)

    def _anonymize_with_entities(
        self,
        value: str,
        entities: list[Entity],
        mask_strategy: str | None,
    ) -> str:
        if not entities:
            return value

        if mask_strategy == "hash":
            return self._hash_replace(value, entities)

        # Strategy override: temporarily swap the anonymizer's per-type
        # strategy so this rule's call uses the requested strategy without
        # leaking into other paths.
        if mask_strategy is None:
            anon_text, mapping = self._anonymizer.anonymize(value, entities, session=self._session)
        else:
            strategy = AnonymizationStrategy(mask_strategy)
            saved = dict(self._anonymizer._type_strategies)  # type: ignore[attr-defined]
            try:
                for e in entities:
                    self._anonymizer._type_strategies[e.type] = strategy  # type: ignore[attr-defined]
                anon_text, mapping = self._anonymizer.anonymize(value, entities, session=self._session)
            finally:
                self._anonymizer._type_strategies = saved  # type: ignore[attr-defined]

        self._session.add_mappings(mapping)
        return anon_text

    def _hash_replace(self, value: str, entities: list[Entity]) -> str:
        """Replace each entity span with a deterministic per-session hash token.

        The hash is salted with the session id so the same value in two
        different sessions yields different tokens — preventing cross-session
        correlation by an LLM.
        """
        # Apply right-to-left so spans don't shift.
        result = value
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            existing = self._session.get_token_for_value(entity.value)
            if existing is not None:
                token = existing
            else:
                digest = hashlib.sha256(f"{self._session.session_id}:{entity.value}".encode()).hexdigest()[:12]
                token = self._anonymizer._placeholder.format(  # type: ignore[attr-defined]
                    f"{entity.type}_HASH", int(digest, 16) % 10**8
                )
                self._session.add_mappings({token: entity.value})
            result = result[: entity.start] + token + result[entity.end :]
        return result


class _SubsetRegexDetector:
    """A RegexDetector limited to a fixed subset of built-in patterns.

    Used internally by :class:`OntologyApplier` so that a field rule with
    ``detect: [EMAIL, PHONE]`` doesn't run the full built-in set.
    """

    def __init__(self, patterns: dict[str, str]) -> None:
        self._compiled = {name: re.compile(pat) for name, pat in patterns.items()}

    def detect(self, text: str) -> list[Entity]:
        out: list[Entity] = []
        for entity_type, pattern in self._compiled.items():
            for m in pattern.finditer(text):
                out.append(
                    Entity(
                        value=m.group(),
                        type=entity_type,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return out
