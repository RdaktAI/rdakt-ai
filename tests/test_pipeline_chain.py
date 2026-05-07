"""Tests for the composable detection pipeline with call_next pattern."""

from unittest.mock import MagicMock

import pytest

from rdakt_ai.models import Entity, RdaktContext
from rdakt_ai.pipeline import CallNext, CallNextSync, NerStage, RdaktPipeline, RdaktStage, RegexStage


class CounterStage(RdaktStage):
    """Test stage that tracks how many times it was called."""

    def __init__(self) -> None:
        self.call_count = 0

    async def process(self, text: str, context: RdaktContext, call_next: CallNext) -> None:
        self.call_count += 1
        await call_next(text, context)


class FixedEntityStage(RdaktStage):
    """Test stage that adds a fixed entity."""

    def __init__(self, entity: Entity) -> None:
        self._entity = entity

    async def process(self, text: str, context: RdaktContext, call_next: CallNext) -> None:
        context.entities.append(self._entity)
        await call_next(text, context)

    def process_sync(self, text: str, context: RdaktContext, call_next_sync: CallNextSync) -> None:
        context.entities.append(self._entity)
        call_next_sync(text, context)


class TestRdaktPipeline:
    async def test_empty_pipeline(self) -> None:
        pipeline = RdaktPipeline(stages=[])
        entities = await pipeline.detect("Hello world")
        assert entities == []

    async def test_single_stage(self) -> None:
        entity = Entity(value="John", type="PERSON", start=0, end=4)
        pipeline = RdaktPipeline(stages=[FixedEntityStage(entity)])
        entities = await pipeline.detect("John is here")
        assert len(entities) == 1

    async def test_multiple_stages_chain(self) -> None:
        e1 = Entity(value="John", type="PERSON", start=0, end=4, detector_priority=0)
        e2 = Entity(value="john@test.com", type="EMAIL", start=10, end=23, detector_priority=1)
        pipeline = RdaktPipeline(
            stages=[
                FixedEntityStage(e1),
                FixedEntityStage(e2),
            ]
        )
        entities = await pipeline.detect("John says john@test.com")
        assert len(entities) == 2

    async def test_call_next_order(self) -> None:
        s1 = CounterStage()
        s2 = CounterStage()
        pipeline = RdaktPipeline(stages=[s1, s2])
        await pipeline.detect("test")
        assert s1.call_count == 1
        assert s2.call_count == 1

    async def test_overlap_resolution_applied(self) -> None:
        e1 = Entity(value="John", type="PERSON", start=0, end=4, detector_priority=0)
        e2 = Entity(value="John Smith", type="PERSON", start=0, end=10, detector_priority=1)
        pipeline = RdaktPipeline(
            stages=[
                FixedEntityStage(e1),
                FixedEntityStage(e2),
            ]
        )
        entities = await pipeline.detect("John Smith is here")
        assert len(entities) == 1
        assert entities[0].value == "John Smith"


class TestRegexInPipeline:
    async def test_regex_as_stage(self) -> None:
        from rdakt_ai.detectors.regex import RegexDetector
        from rdakt_ai.pipeline import RegexStage

        detector = RegexDetector()
        pipeline = RdaktPipeline(stages=[RegexStage(detector)])
        entities = await pipeline.detect("Email: test@example.com")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert len(emails) == 1


class TestSyncPipeline:
    def test_detect_sync_empty_pipeline(self) -> None:
        pipeline = RdaktPipeline(stages=[])
        entities = pipeline.detect_sync("Hello world")
        assert entities == []

    def test_detect_sync_single_stage(self) -> None:
        entity = Entity(value="John", type="PERSON", start=0, end=4)
        pipeline = RdaktPipeline(stages=[FixedEntityStage(entity)])
        entities = pipeline.detect_sync("John is here")
        assert len(entities) == 1

    def test_detect_sync_with_regex_stage(self) -> None:
        from rdakt_ai.detectors.regex import RegexDetector
        from rdakt_ai.pipeline import RegexStage

        detector = RegexDetector()
        pipeline = RdaktPipeline(stages=[RegexStage(detector)])
        entities = pipeline.detect_sync("Email: test@example.com")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert len(emails) == 1

    def test_async_only_stage_raises_on_sync(self) -> None:
        pipeline = RdaktPipeline(stages=[CounterStage()])
        with pytest.raises(NotImplementedError):
            pipeline.detect_sync("test")


class TestNerStage:
    async def test_ner_stage_appends_entities(self) -> None:
        entity = Entity(value="John", type="PERSON", start=0, end=4, detector_priority=10)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [entity]

        stage = NerStage(detector=mock_detector)
        context = RdaktContext()
        called = []

        async def mock_next(text, ctx):
            called.append(True)

        await stage.process("John is here", context, mock_next)
        assert len(context.entities) == 1
        assert context.entities[0].value == "John"
        assert called == [True]

    def test_ner_stage_sync(self) -> None:
        entity = Entity(value="John", type="PERSON", start=0, end=4, detector_priority=10)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [entity]

        stage = NerStage(detector=mock_detector)
        context = RdaktContext()
        called = []

        def mock_next(text, ctx):
            called.append(True)

        stage.process_sync("John is here", context, mock_next)
        assert len(context.entities) == 1
        assert called == [True]

    async def test_ner_in_pipeline_with_regex(self) -> None:
        """Regex + NER in same pipeline, overlap resolution picks longest."""
        from rdakt_ai.detectors.regex import RegexDetector

        regex_detector = RegexDetector()
        regex_stage = RegexStage(regex_detector)

        ner_entity = Entity(value="John Smith", type="PERSON", start=0, end=10, detector_priority=10)
        mock_ner = MagicMock()
        mock_ner.detect.return_value = [ner_entity]
        ner_stage = NerStage(detector=mock_ner)

        pipeline = RdaktPipeline(stages=[regex_stage, ner_stage])
        entities = await pipeline.detect("John Smith email john@example.com")

        types = {e.type for e in entities}
        assert "PERSON" in types
        assert "EMAIL" in types

    async def test_ner_only_pipeline(self) -> None:
        entity = Entity(value="Acme Corp", type="ORG", start=0, end=9, detector_priority=10)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [entity]

        pipeline = RdaktPipeline(stages=[NerStage(detector=mock_detector)])
        entities = await pipeline.detect("Acme Corp is hiring")
        assert len(entities) == 1
        assert entities[0].type == "ORG"


class TestDetectorStage:
    """Tests for the unified DetectorStage."""

    def test_detector_stage_accepts_detector_protocol(self):
        from rdakt_ai.pipeline import DetectorStage

        class CustomDetector:
            def detect(self, text: str) -> list[Entity]:
                return [Entity(value="test", type="CUSTOM", start=0, end=4)]

        stage = DetectorStage(CustomDetector())
        assert stage is not None

    @pytest.mark.asyncio
    async def test_detector_stage_appends_entities(self):
        from rdakt_ai.pipeline import DetectorStage

        class StubDetector:
            def detect(self, text: str) -> list[Entity]:
                return [Entity(value="Alice", type="PERSON", start=0, end=5, detector_priority=10)]

        stage = DetectorStage(StubDetector())
        context = RdaktContext()
        called = False

        async def next_fn(text, ctx):
            nonlocal called
            called = True

        await stage.process("Alice works at ACME", context, next_fn)
        assert called
        assert len(context.entities) == 1
        assert context.entities[0].type == "PERSON"

    def test_detector_stage_sync(self):
        from rdakt_ai.pipeline import DetectorStage

        class StubDetector:
            def detect(self, text: str) -> list[Entity]:
                return [Entity(value="test@x.com", type="EMAIL", start=0, end=10)]

        stage = DetectorStage(StubDetector())
        context = RdaktContext()

        def next_fn(text, ctx):
            pass

        stage.process_sync("test@x.com", context, next_fn)
        assert len(context.entities) == 1

    def test_regex_stage_is_detector_stage_alias(self):
        from rdakt_ai.pipeline import DetectorStage, RegexStage

        assert RegexStage is DetectorStage

    def test_ner_stage_is_detector_stage_alias(self):
        from rdakt_ai.pipeline import DetectorStage, NerStage

        assert NerStage is DetectorStage
