"""Core data models for Rdakt AI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    """A detected sensitive entity with its position in the text."""

    value: str
    type: str
    start: int
    end: int
    locale: str | None = None
    detector_priority: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class RdaktContext:
    """Mutable context passed through the detection pipeline."""

    entities: list[Entity] = field(default_factory=list)
