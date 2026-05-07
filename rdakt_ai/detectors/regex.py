"""Regex-based entity detector for structured patterns."""

from __future__ import annotations

import re
from typing import ClassVar

from rdakt_ai.models import Entity

_BUILTIN_PATTERNS: dict[str, str] = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
    "JWT": r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    "API_KEY": (
        r"(?:"
        r"sk-proj-[A-Za-z0-9]{20,}"
        r"|sk-[A-Za-z0-9]{20,}"
        r"|AKIA[A-Z0-9]{16}"
        r"|ghp_[A-Za-z0-9]{36}"
        r"|glpat-[A-Za-z0-9\-]{20,}"
        r")"
    ),
}


class RegexDetector:
    """Detects sensitive entities using regex patterns.

    Supports built-in patterns for common entity types and custom patterns.
    """

    builtin_patterns: ClassVar[dict[str, str]] = _BUILTIN_PATTERNS

    def __init__(
        self,
        *,
        custom_patterns: dict[str, str] | None = None,
        detector_priority: int = 0,
    ) -> None:
        self._priority = detector_priority
        self._compiled: dict[str, re.Pattern[str]] = {}

        for name, pattern in self.builtin_patterns.items():
            self._compiled[name] = re.compile(pattern)

        if custom_patterns:
            for name, pattern in custom_patterns.items():
                try:
                    self._compiled[name] = re.compile(pattern)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern for {name!r}: {e}") from e

    def detect(self, text: str) -> list[Entity]:
        """Detect all entities in the given text."""
        entities: list[Entity] = []
        for entity_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                entities.append(
                    Entity(
                        value=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        detector_priority=self._priority,
                    )
                )
        return entities
