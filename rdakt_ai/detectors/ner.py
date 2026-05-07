"""spaCy NER-based entity detector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdakt_ai.models import Entity

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from spacy.language import Language

DEFAULT_LABEL_MAPPING: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "NORP": "GROUP",
}


class SpacyDetector:
    """Detects entities using spaCy NER models.

    Requires the ``ner`` extra: ``pip install rdakt-ai[ner]``.
    Model is loaded lazily on the first ``detect()`` call.
    """

    def __init__(
        self,
        *,
        model: str = "en_core_web_sm",
        label_mapping: dict[str, str] | None = None,
        detector_priority: int = 10,
    ) -> None:
        if spacy is None:
            raise ImportError("NER support requires the 'ner' extra. Install it with: pip install rdakt-ai[ner]")
        self._model_name = model
        self._label_mapping = label_mapping or dict(DEFAULT_LABEL_MAPPING)
        self._priority = detector_priority
        self._nlp: Language | None = None

    def _load_model(self) -> None:
        try:
            self._nlp = spacy.load(self._model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{self._model_name}' not found. "
                f"Install it with: python -m spacy download {self._model_name}"
            ) from e

    def detect(self, text: str) -> list[Entity]:
        """Detect entities in text using the spaCy NER model."""
        if self._nlp is None:
            self._load_model()
        assert self._nlp is not None

        doc = self._nlp(text)
        entities: list[Entity] = []
        for ent in doc.ents:
            mapped_type = self._label_mapping.get(ent.label_)
            if mapped_type is None:
                continue
            entities.append(
                Entity(
                    value=ent.text,
                    type=mapped_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    detector_priority=self._priority,
                )
            )
        return entities
