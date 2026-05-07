"""Detection pipeline: composable call_next chain with overlap resolution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from rdakt_ai.detectors import Detector
from rdakt_ai.models import Entity, RdaktContext

logger = logging.getLogger("rdakt_ai.pipeline")

CallNext = Callable[[str, RdaktContext], Coroutine[Any, Any, None]]
CallNextSync = Callable[[str, RdaktContext], None]


def resolve_overlaps(entities: list[Entity]) -> list[Entity]:
    """Resolve overlapping entity spans.

    Strategy: longest match wins, pipeline order (detector_priority) breaks ties.
    Returns non-overlapping entities sorted by position.
    """
    if not entities:
        return []

    sorted_entities = sorted(
        entities,
        key=lambda e: (e.start, -e.length, e.detector_priority),
    )

    resolved: list[Entity] = []
    last_end = -1
    for entity in sorted_entities:
        if entity.start >= last_end:
            resolved.append(entity)
            last_end = entity.end

    return resolved


class RdaktStage(ABC):
    """Base class for detection pipeline stages."""

    @abstractmethod
    async def process(self, text: str, context: RdaktContext, call_next: CallNext) -> None: ...

    def process_sync(self, text: str, context: RdaktContext, call_next_sync: CallNextSync) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support sync execution")


class DetectorStage(RdaktStage):
    """Wraps any Detector as a pipeline stage."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    async def process(self, text: str, context: RdaktContext, call_next: CallNext) -> None:
        entities = self._detector.detect(text)
        context.entities.extend(entities)
        await call_next(text, context)

    def process_sync(self, text: str, context: RdaktContext, call_next_sync: CallNextSync) -> None:
        entities = self._detector.detect(text)
        context.entities.extend(entities)
        call_next_sync(text, context)


# Backwards-compatible aliases
RegexStage = DetectorStage
NerStage = DetectorStage


class RdaktPipeline:
    """Composable detection pipeline with call_next chain pattern."""

    def __init__(self, *, stages: list[RdaktStage]) -> None:
        self._stages = stages

    async def detect(self, text: str) -> list[Entity]:
        """Run all stages and return resolved (non-overlapping) entities."""
        context = RdaktContext()

        async def terminal(text: str, context: RdaktContext) -> None:
            pass

        chain: CallNext = terminal
        for stage in reversed(self._stages):
            chain = _make_handler(stage, chain)

        await chain(text, context)

        return resolve_overlaps(context.entities)

    def detect_sync(self, text: str) -> list[Entity]:
        """Run all stages synchronously and return resolved entities."""
        context = RdaktContext()

        def terminal(text: str, context: RdaktContext) -> None:
            pass

        chain: CallNextSync = terminal
        for stage in reversed(self._stages):
            chain = _make_sync_handler(stage, chain)

        chain(text, context)
        return resolve_overlaps(context.entities)


def _make_handler(stage: RdaktStage, next_fn: CallNext) -> CallNext:
    async def handler(text: str, context: RdaktContext) -> None:
        await stage.process(text, context, next_fn)

    return handler


def _make_sync_handler(stage: RdaktStage, next_fn: CallNextSync) -> CallNextSync:
    def handler(text: str, context: RdaktContext) -> None:
        stage.process_sync(text, context, next_fn)

    return handler
