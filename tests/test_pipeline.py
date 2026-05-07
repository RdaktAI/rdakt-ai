"""Tests for entity overlap resolution."""

from rdakt_ai.models import Entity
from rdakt_ai.pipeline import resolve_overlaps


class TestOverlapResolution:
    def test_no_overlaps(self) -> None:
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="test@example.com", type="EMAIL", start=10, end=26),
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 2

    def test_overlapping_longest_wins(self) -> None:
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="John Smith", type="PERSON", start=0, end=10),
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 1
        assert result[0].value == "John Smith"

    def test_overlapping_priority_tiebreaker(self) -> None:
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4, detector_priority=0),
            Entity(value="John", type="NAME", start=0, end=4, detector_priority=1),
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 1
        assert result[0].type == "PERSON"

    def test_partial_overlap_longest_wins(self) -> None:
        entities = [
            Entity(value="John Smith", type="PERSON", start=0, end=10),
            Entity(value="john.smith@acme.com", type="EMAIL", start=0, end=19),
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 1
        assert result[0].type == "EMAIL"

    def test_adjacent_no_overlap(self) -> None:
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="Smith", type="PERSON", start=4, end=9),
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert resolve_overlaps([]) == []

    def test_result_sorted_by_position(self) -> None:
        entities = [
            Entity(value="Smith", type="PERSON", start=10, end=15),
            Entity(value="John", type="PERSON", start=0, end=4),
        ]
        result = resolve_overlaps(entities)
        assert result[0].start == 0
        assert result[1].start == 10
