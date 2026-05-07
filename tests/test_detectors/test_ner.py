"""Tests for SpacyDetector — NER-based entity detection."""

from unittest.mock import MagicMock, patch

import pytest

from rdakt_ai.detectors import Detector
from rdakt_ai.detectors.ner import SpacyDetector


class MockDoc:
    """Mock spaCy Doc with configurable entities."""

    def __init__(self, ents):
        self.ents = ents


class MockSpan:
    """Mock spaCy Span (entity)."""

    def __init__(self, text, label_, start_char, end_char):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


def make_mock_nlp(entities):
    """Create a mock spaCy nlp object that returns given entities."""
    nlp = MagicMock()
    nlp.return_value = MockDoc(entities)
    return nlp


class TestSpacyDetectorProtocol:
    @patch("rdakt_ai.detectors.ner.spacy")
    def test_satisfies_detector_protocol(self, mock_spacy) -> None:
        mock_spacy.load.return_value = MagicMock()
        detector = SpacyDetector()
        assert isinstance(detector, Detector)


class TestSpacyDetectorDetect:
    @patch("rdakt_ai.detectors.ner.spacy")
    def test_detect_person(self, mock_spacy) -> None:
        mock_nlp = make_mock_nlp(
            [
                MockSpan("John Smith", "PERSON", 0, 10),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector()
        entities = detector.detect("John Smith works at Acme")
        assert len(entities) == 1
        assert entities[0].type == "PERSON"
        assert entities[0].value == "John Smith"
        assert entities[0].start == 0
        assert entities[0].end == 10

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_detect_org(self, mock_spacy) -> None:
        mock_nlp = make_mock_nlp(
            [
                MockSpan("Acme Corp", "ORG", 20, 29),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector()
        entities = detector.detect("John works at Acme Corp")
        assert len(entities) == 1
        assert entities[0].type == "ORG"

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_default_label_mapping(self, mock_spacy) -> None:
        """GPE and LOC both map to LOCATION."""
        mock_nlp = make_mock_nlp(
            [
                MockSpan("London", "GPE", 0, 6),
                MockSpan("Mount Everest", "LOC", 10, 23),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector()
        entities = detector.detect("London and Mount Everest")
        assert all(e.type == "LOCATION" for e in entities)

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_unmapped_labels_skipped(self, mock_spacy) -> None:
        """Labels not in the mapping should be ignored."""
        mock_nlp = make_mock_nlp(
            [
                MockSpan("John", "PERSON", 0, 4),
                MockSpan("100%", "PERCENT", 10, 14),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector()
        entities = detector.detect("John scored 100%")
        assert len(entities) == 1
        assert entities[0].type == "PERSON"

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_custom_label_mapping(self, mock_spacy) -> None:
        mock_nlp = make_mock_nlp(
            [
                MockSpan("Acme", "ORG", 0, 4),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector(label_mapping={"ORG": "COMPANY"})
        entities = detector.detect("Acme is great")
        assert entities[0].type == "COMPANY"

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_detector_priority(self, mock_spacy) -> None:
        mock_nlp = make_mock_nlp(
            [
                MockSpan("John", "PERSON", 0, 4),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector(detector_priority=10)
        entities = detector.detect("John")
        assert entities[0].detector_priority == 10

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_default_priority_is_10(self, mock_spacy) -> None:
        mock_nlp = make_mock_nlp(
            [
                MockSpan("John", "PERSON", 0, 4),
            ]
        )
        mock_spacy.load.return_value = mock_nlp
        detector = SpacyDetector()
        entities = detector.detect("John")
        assert entities[0].detector_priority == 10


class TestSpacyDetectorLazyLoading:
    @patch("rdakt_ai.detectors.ner.spacy")
    def test_model_not_loaded_until_detect(self, mock_spacy) -> None:
        SpacyDetector()
        mock_spacy.load.assert_not_called()

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_model_loaded_on_first_detect(self, mock_spacy) -> None:
        mock_spacy.load.return_value = make_mock_nlp([])
        detector = SpacyDetector()
        detector.detect("Hello")
        mock_spacy.load.assert_called_once_with("en_core_web_sm")

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_model_cached_after_first_detect(self, mock_spacy) -> None:
        mock_spacy.load.return_value = make_mock_nlp([])
        detector = SpacyDetector()
        detector.detect("Hello")
        detector.detect("World")
        mock_spacy.load.assert_called_once()

    @patch("rdakt_ai.detectors.ner.spacy")
    def test_custom_model_name(self, mock_spacy) -> None:
        mock_spacy.load.return_value = make_mock_nlp([])
        detector = SpacyDetector(model="en_core_web_trf")
        detector.detect("Hello")
        mock_spacy.load.assert_called_once_with("en_core_web_trf")


class TestSpacyDetectorErrors:
    @patch("rdakt_ai.detectors.ner.spacy")
    def test_model_not_found_raises_clear_error(self, mock_spacy) -> None:
        mock_spacy.load.side_effect = OSError("Can't find model")
        detector = SpacyDetector(model="en_core_web_sm")
        with pytest.raises(OSError, match="python -m spacy download en_core_web_sm"):
            detector.detect("Hello")

    def test_import_error_without_spacy(self) -> None:
        with patch("rdakt_ai.detectors.ner.spacy", None), pytest.raises(ImportError, match="ner"):
            SpacyDetector()
