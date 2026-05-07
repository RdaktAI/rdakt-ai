"""Tests for core data models."""

from rdakt_ai.models import Entity, RdaktContext


class TestEntity:
    def test_create_entity(self) -> None:
        entity = Entity(value="John Smith", type="PERSON", start=0, end=10)
        assert entity.value == "John Smith"
        assert entity.type == "PERSON"
        assert entity.start == 0
        assert entity.end == 10
        assert entity.length == 10

    def test_entity_with_locale(self) -> None:
        entity = Entity(value="田中太郎", type="PERSON", start=0, end=4, locale="ja_JP")
        assert entity.locale == "ja_JP"

    def test_entity_with_detector_priority(self) -> None:
        entity = Entity(value="test@example.com", type="EMAIL", start=0, end=16, detector_priority=0)
        assert entity.detector_priority == 0

    def test_entity_default_locale_is_none(self) -> None:
        entity = Entity(value="test", type="CUSTOM", start=0, end=4)
        assert entity.locale is None

    def test_entity_default_priority_is_zero(self) -> None:
        entity = Entity(value="test", type="CUSTOM", start=0, end=4)
        assert entity.detector_priority == 0


class TestRdaktContext:
    def test_empty_context(self) -> None:
        ctx = RdaktContext()
        assert ctx.entities == []

    def test_add_entities(self) -> None:
        ctx = RdaktContext()
        entity = Entity(value="John", type="PERSON", start=0, end=4)
        ctx.entities.append(entity)
        assert len(ctx.entities) == 1
