"""Anonymization strategies: token replacement, synthetic substitution, hybrid."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from faker import Faker

from rdakt_ai.models import Entity
from rdakt_ai.placeholders import PlaceholderFormat

if TYPE_CHECKING:
    from rdakt_ai.session import RdaktSession


class AnonymizationStrategy(Enum):
    TOKEN = "token"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"


_HYBRID_SYNTHETIC_TYPES: set[str] = {"PERSON", "ORGANIZATION", "LOCATION", "ADDRESS"}

_FAKER_TYPE_MAP: dict[str, str] = {
    "PERSON": "name",
    "ORGANIZATION": "company",
    "LOCATION": "city",
    "ADDRESS": "address",
}


class Anonymizer:
    """Applies anonymization strategies to detected entities."""

    hybrid_synthetic_types: ClassVar[set[str]] = _HYBRID_SYNTHETIC_TYPES

    def __init__(
        self,
        *,
        default_strategy: AnonymizationStrategy = AnonymizationStrategy.HYBRID,
        type_strategies: dict[str, AnonymizationStrategy] | None = None,
        placeholder_format: PlaceholderFormat | None = None,
    ) -> None:
        self._default_strategy = default_strategy
        self._type_strategies = type_strategies or {}
        self._placeholder = placeholder_format or PlaceholderFormat()

    def _get_strategy(self, entity_type: str) -> AnonymizationStrategy:
        if entity_type in self._type_strategies:
            return self._type_strategies[entity_type]
        if self._default_strategy == AnonymizationStrategy.HYBRID:
            if entity_type in self.hybrid_synthetic_types:
                return AnonymizationStrategy.SYNTHETIC
            return AnonymizationStrategy.TOKEN
        return self._default_strategy

    def _infer_type_counters(self, token_names: set[str]) -> dict[str, int]:
        """Extract the highest counter per entity type from existing token names.

        Uses the active placeholder format so counters survive a session that
        was created with a custom template.
        """
        counters: dict[str, int] = {}
        for token in token_names:
            parsed = self._placeholder.parse(token)
            if parsed is None:
                continue
            entity_type, num = parsed
            if num > counters.get(entity_type, 0):
                counters[entity_type] = num
        return counters

    def anonymize(
        self,
        text: str,
        entities: list[Entity],
        *,
        session: RdaktSession | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Anonymize text by replacing entities.

        Returns (anonymized_text, mapping) where mapping contains only NEW
        mappings ``{anonymized_value: original_value}`` (values already in
        the session are reused but not included in the returned mapping).

        When ``session`` is provided, existing token mappings are reused for
        known values, and type counters start after the highest existing
        number to avoid collisions.
        """
        if not entities:
            return text, {}

        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        mapping: dict[str, str] = {}
        value_to_anon: dict[str, str] = {}
        type_counters = self._infer_type_counters(session.token_names) if session else {}

        for entity in sorted_entities:
            if entity.value in value_to_anon:
                replacement = value_to_anon[entity.value]
            elif session and (existing := session.get_token_for_value(entity.value)):
                replacement = existing
                value_to_anon[entity.value] = replacement
            else:
                strategy = self._get_strategy(entity.type)
                if strategy == AnonymizationStrategy.SYNTHETIC:
                    session_id = session.session_id if session else ""
                    replacement = self._generate_synthetic(entity, session_id, set(mapping.values()))
                else:
                    count = type_counters.get(entity.type, 0) + 1
                    type_counters[entity.type] = count
                    replacement = self._placeholder.format(entity.type, count)

                value_to_anon[entity.value] = replacement
                mapping[replacement] = entity.value

            text = text[: entity.start] + replacement + text[entity.end :]

        return text, mapping

    def _generate_synthetic(
        self,
        entity: Entity,
        session_id: str,
        existing_real_values: set[str],
    ) -> str:
        seed = hash((session_id, entity.value))
        locale = entity.locale or "en_US"
        fake = Faker(locale)
        faker_method = _FAKER_TYPE_MAP.get(entity.type, "name")

        for attempt in range(5):
            fake.seed_instance(seed + attempt)
            candidate: str = getattr(fake, faker_method)()
            if candidate not in existing_real_values and candidate != entity.value:
                return candidate

        return f"<{entity.type}_SYNTH>"
