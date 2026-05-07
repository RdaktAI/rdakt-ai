"""Session management: entity mapping, deanonymization, and streaming."""

from __future__ import annotations

import re
import uuid
from typing import Any

from rdakt_ai.placeholders import PlaceholderFormat


class RdaktSession:
    """Holds bidirectional entity mapping for a conversation.

    Handles deanonymization of text, streaming chunks, and structured data
    (tool call arguments).
    """

    MAX_BUFFER = 256

    def __init__(
        self,
        *,
        session_id: str | None = None,
        entity_map: dict[str, str] | None = None,
        placeholder_format: PlaceholderFormat | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._entity_map: dict[str, str] = dict(entity_map) if entity_map else {}
        self._reverse_map: dict[str, str] = {v: k for k, v in self._entity_map.items()}
        self._buffer: str = ""
        self._placeholder = placeholder_format or PlaceholderFormat()

    def add_mappings(self, new_mappings: dict[str, str]) -> None:
        """Add new entity mappings (for multi-turn accumulation)."""
        self._entity_map.update(new_mappings)
        self._reverse_map.update({v: k for k, v in new_mappings.items()})

    @property
    def entity_map(self) -> dict[str, str]:
        """Read-only view of the current token-to-value mappings."""
        return dict(self._entity_map)

    @property
    def has_mappings(self) -> bool:
        """True if any entity mappings exist."""
        return bool(self._entity_map)

    @property
    def real_values(self) -> set[str]:
        return set(self._entity_map.values())

    def get_anonymized(self, real_value: str) -> str | None:
        return self._reverse_map.get(real_value)

    def get_token_for_value(self, value: str) -> str | None:
        """Return the existing token for a real value, or None if not mapped."""
        return self._reverse_map.get(value)

    @property
    def token_names(self) -> set[str]:
        """All token names currently in the session."""
        return set(self._entity_map.keys())

    def deanonymize(self, text: str) -> str:
        """Deanonymize a complete text by replacing all tokens.

        For the canonical template, also handles backslash-escaped
        (``\\<SSN_1\\>``) forms via the placeholder's ``full_re``.

        Literal whole-value tokens that don't fit the placeholder template
        (produced by ontology ``replace_with_synthetic`` rules, e.g.
        ``user-000001``) are also restored via a longest-first literal pass.
        """

        def _replace(m: re.Match[str]) -> str:
            entity_type = m.group(1).upper()
            n = int(m.group(2))
            canonical = self._placeholder.format(entity_type, n)
            return self._entity_map.get(canonical, m.group(0))

        result = self._placeholder.full_re.sub(_replace, text)

        # Second pass: restore tokens that don't match the placeholder regex.
        literal_tokens = [tok for tok in self._entity_map if not self._placeholder.full_re.fullmatch(tok)]
        if literal_tokens:
            for tok in sorted(literal_tokens, key=len, reverse=True):
                if tok in result:
                    result = result.replace(tok, self._entity_map[tok])
        return result

    def deanonymize_structured(self, data: Any) -> Any:
        """Recursively deanonymize structured data (dicts, lists, strings)."""
        if isinstance(data, str):
            return self.deanonymize(data)
        if isinstance(data, dict):
            return {k: self.deanonymize_structured(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.deanonymize_structured(item) for item in data]
        return data

    def deanonymize_chunk(self, chunk: str) -> list[str]:
        """Deanonymize a streaming chunk. Returns list of text fragments to emit.

        Uses ``_TOKEN_RE`` to find complete tokens (escaped or not) and
        ``_PARTIAL_RE`` to detect a trailing partial token that should be
        buffered until the next chunk arrives.
        """
        if not chunk:
            return []

        self._buffer += chunk
        results: list[str] = []
        last_end = 0

        for m in self._placeholder.full_re.finditer(self._buffer):
            entity_type = m.group(1).upper()
            n = int(m.group(2))
            canonical = self._placeholder.format(entity_type, n)
            if canonical not in self._entity_map:
                continue
            if m.start() > last_end:
                results.append(self._buffer[last_end : m.start()])
            results.append(self._entity_map[canonical])
            last_end = m.end()

        remaining = self._buffer[last_end:]

        # Check for trailing partial token
        partial = self._placeholder.partial_re.search(remaining)
        if partial:
            safe = remaining[: partial.start()]
            if safe:
                results.append(safe)
            self._buffer = remaining[partial.start() :]
        else:
            if remaining:
                results.append(remaining)
            self._buffer = ""

        # Safety valve
        if len(self._buffer) > self.MAX_BUFFER:
            results.append(self._buffer)
            self._buffer = ""

        return results

    def flush(self) -> list[str]:
        """Flush any remaining buffered content."""
        if self._buffer:
            remaining = self._buffer
            self._buffer = ""
            return [remaining]
        return []
