"""Tests for anonymization strategies."""

from unittest.mock import patch

from rdakt_ai.anonymizer import AnonymizationStrategy, Anonymizer
from rdakt_ai.models import Entity
from rdakt_ai.session import RdaktSession


class TestTokenReplacement:
    def test_replace_single_entity(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        text = "Hello John Smith"
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        result, mapping = anon.anonymize(text, entities)
        assert "John Smith" not in result
        assert "<PERSON_1>" in result
        assert mapping["<PERSON_1>"] == "John Smith"

    def test_replace_multiple_same_type(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        text = "John called Jane"
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="Jane", type="PERSON", start=12, end=16),
        ]
        result, mapping = anon.anonymize(text, entities)
        assert "<PERSON_1>" in result
        assert "<PERSON_2>" in result
        assert len(mapping) == 2

    def test_replace_different_types(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        text = "John's email is john@example.com"
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="john@example.com", type="EMAIL", start=16, end=32),
        ]
        result, _ = anon.anonymize(text, entities)
        assert "<PERSON_1>" in result
        assert "<EMAIL_1>" in result

    def test_same_value_same_token(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        text = "John said John is here"
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="John", type="PERSON", start=10, end=14),
        ]
        result, _ = anon.anonymize(text, entities)
        assert result.count("<PERSON_1>") == 2


class TestSyntheticSubstitution:
    def test_synthetic_produces_different_value(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.SYNTHETIC)
        text = "Hello John Smith"
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        result, _ = anon.anonymize(text, entities, session=RdaktSession(session_id="test-session"))
        assert "John Smith" not in result
        assert "<PERSON" not in result

    def test_synthetic_deterministic_same_session(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.SYNTHETIC)
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        _, map1 = anon.anonymize("Hello John Smith", entities, session=RdaktSession(session_id="s1"))
        _, map2 = anon.anonymize("Hello John Smith", entities, session=RdaktSession(session_id="s1"))
        assert list(map1.values()) == list(map2.values())

    def test_synthetic_different_sessions_differ(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.SYNTHETIC)
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        _, map1 = anon.anonymize("Hello John Smith", entities, session=RdaktSession(session_id="s1"))
        _, map2 = anon.anonymize("Hello John Smith", entities, session=RdaktSession(session_id="s2"))
        synth1 = next(iter(map1.values()))
        synth2 = next(iter(map2.values()))
        assert synth1
        assert synth2


class TestHybridStrategy:
    def test_hybrid_uses_synthetic_for_person(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.HYBRID)
        text = "Hello John Smith"
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        result, _ = anon.anonymize(text, entities, session=RdaktSession(session_id="test"))
        assert "John Smith" not in result
        assert "<PERSON" not in result

    def test_hybrid_uses_token_for_email(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.HYBRID)
        text = "Email: john@example.com"
        entities = [Entity(value="john@example.com", type="EMAIL", start=7, end=23)]
        result, _ = anon.anonymize(text, entities)
        assert "<EMAIL_1>" in result

    def test_hybrid_uses_token_for_credit_card(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.HYBRID)
        text = "Card: 4111-1111-1111-1111"
        entities = [
            Entity(value="4111-1111-1111-1111", type="CREDIT_CARD", start=6, end=25),
        ]
        result, _ = anon.anonymize(text, entities)
        assert "<CREDIT_CARD_1>" in result


class TestPerEntityConfig:
    def test_override_strategy_per_type(self) -> None:
        anon = Anonymizer(
            default_strategy=AnonymizationStrategy.TOKEN,
            type_strategies={"PERSON": AnonymizationStrategy.SYNTHETIC},
        )
        text = "John's email john@example.com"
        entities = [
            Entity(value="John", type="PERSON", start=0, end=4),
            Entity(value="john@example.com", type="EMAIL", start=14, end=30),
        ]
        result, _ = anon.anonymize(text, entities, session=RdaktSession(session_id="test"))
        assert "<PERSON" not in result
        assert "<EMAIL_1>" in result


class TestSyntheticFallback:
    def test_collision_fallback_to_token(self) -> None:
        anon = Anonymizer(default_strategy=AnonymizationStrategy.SYNTHETIC)
        text = "Hello John"
        entities = [Entity(value="John", type="PERSON", start=6, end=10)]
        # Mock Faker to always return the original value (forcing collision)
        with patch("rdakt_ai.anonymizer.Faker") as mock_faker_class:
            mock_instance = mock_faker_class.return_value
            mock_instance.name.return_value = "John"
            _, mapping = anon.anonymize(text, entities, session=RdaktSession(session_id="test"))
        # Should fall back to <PERSON_SYNTH> after 5 collisions
        assert "<PERSON_SYNTH>" in next(iter(mapping.keys()))


class TestAnonymizeEmpty:
    def test_no_entities(self) -> None:
        anon = Anonymizer()
        result, mapping = anon.anonymize("Hello world", [])
        assert result == "Hello world"
        assert mapping == {}


class TestSessionAwareAnonymization:
    def test_reuses_token_for_known_value(self) -> None:
        """Same value seen in a prior turn gets the same token."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "john@test.com"})
        text = "Contact john@test.com"
        entities = [Entity(value="john@test.com", type="EMAIL", start=8, end=21)]
        result, mapping = anon.anonymize(text, entities, session=session)
        assert result == "Contact <EMAIL_1>"
        assert mapping == {}  # not new — already in session

    def test_new_value_avoids_collision(self) -> None:
        """New value of same type gets next counter, not <EMAIL_1> again."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "john@test.com"})
        text = "Contact jane@test.com"
        entities = [Entity(value="jane@test.com", type="EMAIL", start=8, end=21)]
        result, mapping = anon.anonymize(text, entities, session=session)
        assert result == "Contact <EMAIL_2>"
        assert mapping == {"<EMAIL_2>": "jane@test.com"}

    def test_mixed_reuse_and_new(self) -> None:
        """One known value reused, one new value gets next counter."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "john@test.com"})
        text = "john@test.com and jane@test.com"
        entities = [
            Entity(value="john@test.com", type="EMAIL", start=0, end=13),
            Entity(value="jane@test.com", type="EMAIL", start=18, end=31),
        ]
        result, mapping = anon.anonymize(text, entities, session=session)
        assert "<EMAIL_1>" in result
        assert "<EMAIL_2>" in result
        assert mapping == {"<EMAIL_2>": "jane@test.com"}

    def test_no_session_works_statelessly(self) -> None:
        """Without a session, anonymize works exactly as before."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        text = "Email john@test.com"
        entities = [Entity(value="john@test.com", type="EMAIL", start=6, end=19)]
        result, mapping = anon.anonymize(text, entities)
        assert "<EMAIL_1>" in result
        assert mapping == {"<EMAIL_1>": "john@test.com"}

    def test_counter_continues_across_types(self) -> None:
        """Counters are per-type: EMAIL_2 and SSN_1 are independent."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.TOKEN)
        session = RdaktSession(
            session_id="test",
            entity_map={
                "<EMAIL_1>": "john@test.com",
                "<SSN_1>": "111-11-1111",
            },
        )
        text = "jane@test.com 222-22-2222"
        entities = [
            Entity(value="jane@test.com", type="EMAIL", start=0, end=13),
            Entity(value="222-22-2222", type="SSN", start=14, end=25),
        ]
        result, _mapping = anon.anonymize(text, entities, session=session)
        assert "<EMAIL_2>" in result
        assert "<SSN_2>" in result

    def test_synthetic_reuses_for_known_value(self) -> None:
        """Synthetic strategy also reuses tokens for known values."""
        anon = Anonymizer(default_strategy=AnonymizationStrategy.SYNTHETIC)
        session = RdaktSession(session_id="test", entity_map={"Maria Garcia": "John Smith"})
        text = "Hello John Smith"
        entities = [Entity(value="John Smith", type="PERSON", start=6, end=16)]
        result, mapping = anon.anonymize(text, entities, session=session)
        assert result == "Hello Maria Garcia"
        assert mapping == {}  # reused, not new
