"""Detection stages for the Rdakt AI pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rdakt_ai.models import Entity


@runtime_checkable
class Detector(Protocol):
    """Protocol for entity detectors.

    Any class with a ``detect(text: str) -> list[Entity]`` method satisfies this.
    """

    def detect(self, text: str) -> list[Entity]: ...
