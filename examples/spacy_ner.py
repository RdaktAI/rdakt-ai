"""SpacyDetector: NER-based entity detection.

Demonstrates SpacyDetector construction and label mapping.
Uses mocks to avoid requiring spaCy model download.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rdakt_ai.detectors.ner import DEFAULT_LABEL_MAPPING, SpacyDetector

# Show default label mapping
print("Default label mapping:")
for spacy_label, rdakt_type in DEFAULT_LABEL_MAPPING.items():
    print(f"  {spacy_label} -> {rdakt_type}")

# Demonstrate with mocked spaCy model
mock_nlp = MagicMock()
mock_ent = MagicMock()
mock_ent.text = "Alice"
mock_ent.label_ = "PERSON"
mock_ent.start_char = 0
mock_ent.end_char = 5
mock_doc = MagicMock()
mock_doc.ents = [mock_ent]
mock_nlp.return_value = mock_doc

with patch("rdakt_ai.detectors.ner.spacy") as mock_spacy:
    mock_spacy.load.return_value = mock_nlp
    detector = SpacyDetector(model="en_core_web_sm", detector_priority=10)
    entities = detector.detect("Alice went to Paris")
    print(f"\nDetected: {[(e.type, e.value, e.detector_priority) for e in entities]}")

print("Done.")
