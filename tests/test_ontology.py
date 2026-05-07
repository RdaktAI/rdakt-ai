"""Tests for the per-field PII ontology."""

from __future__ import annotations

import re
from typing import Any

import pytest

from rdakt_ai.anonymizer import Anonymizer
from rdakt_ai.config import FieldRule, OntologyConfig, RdaktConfig, SyntheticReplacement
from rdakt_ai.ontology import OntologyApplier, parse_jsonpath
from rdakt_ai.session import RdaktSession

# ---------------------------------------------------------------------------
# JSONPath parser
# ---------------------------------------------------------------------------


class TestParseJsonPath:
    def test_root_only_is_empty_segments(self) -> None:
        assert parse_jsonpath("$") == []

    def test_simple_field(self) -> None:
        segments = parse_jsonpath("$.foo")
        assert len(segments) == 1

    def test_nested_field(self) -> None:
        segments = parse_jsonpath("$.foo.bar.baz")
        assert len(segments) == 3

    def test_array_wildcard(self) -> None:
        segments = parse_jsonpath("$.messages[*].content")
        assert len(segments) == 3

    def test_must_start_with_dollar(self) -> None:
        with pytest.raises(ValueError, match=r"\$"):
            parse_jsonpath(".foo")

    def test_only_array_wildcard_supported(self) -> None:
        with pytest.raises(ValueError, match="wildcard"):
            parse_jsonpath("$.messages[0].content")

    def test_invalid_field_name_raises(self) -> None:
        with pytest.raises(ValueError, match="field name"):
            parse_jsonpath("$.")


# ---------------------------------------------------------------------------
# FieldRule validation (config-load)
# ---------------------------------------------------------------------------


class TestFieldRuleValidation:
    def test_detect_rule_valid(self) -> None:
        rule = FieldRule(path="$.messages[*].content", detect=["EMAIL"])
        assert rule.detect == ["EMAIL"]

    def test_synthetic_rule_valid(self) -> None:
        rule = FieldRule(
            path="$.messages[*].metadata.user_id",
            replace_with_synthetic=SyntheticReplacement(type="USER_ID", format="user-{n:06d}"),
        )
        assert rule.replace_with_synthetic is not None

    def test_must_declare_exactly_one_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            FieldRule(path="$.x")

    def test_cannot_declare_two_sources(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            FieldRule(
                path="$.x",
                detect=["EMAIL"],
                detect_via_regex=r"\bX\d+\b",
            )

    def test_invalid_jsonpath_rejected_at_load(self) -> None:
        with pytest.raises(ValueError, match="path"):
            FieldRule(path="not.a.path", detect=["EMAIL"])

    def test_invalid_regex_rejected(self) -> None:
        with pytest.raises(ValueError, match="regex"):
            FieldRule(path="$.x", detect_via_regex="(unclosed")

    def test_invalid_synthetic_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="format"):
            SyntheticReplacement(type="X", format="bad-{wrong}")

    def test_preserve_components_requires_detect(self) -> None:
        with pytest.raises(ValueError, match="preserve_components"):
            FieldRule(
                path="$.x",
                detect_via_regex=r"\bX\b",
                preserve_components=["city"],
            )


class TestRdaktConfigOntology:
    def test_default_ontology_is_none(self) -> None:
        cfg = RdaktConfig()
        assert cfg.ontology is None

    def test_ontology_constructed(self) -> None:
        cfg = RdaktConfig(ontology=OntologyConfig(fields=[FieldRule(path="$.messages[*].content", detect=["EMAIL"])]))
        assert cfg.ontology is not None
        assert len(cfg.ontology.fields) == 1


# ---------------------------------------------------------------------------
# Applier behaviour
# ---------------------------------------------------------------------------


_session_seq = 0


def _make_applier(rules: list[FieldRule]) -> tuple[OntologyApplier, RdaktSession, Anonymizer]:
    global _session_seq
    _session_seq += 1
    session = RdaktSession(session_id=f"sess-{_session_seq}")
    anonymizer = Anonymizer()
    applier = OntologyApplier(OntologyConfig(fields=rules), anonymizer=anonymizer, session=session)
    return applier, session, anonymizer


class TestApplierDetectSubset:
    def test_restricts_to_named_types(self) -> None:
        applier, _, _ = _make_applier([FieldRule(path="$.messages[*].content", detect=["EMAIL"])])
        body = {"messages": [{"content": "Email me at a@b.com or call 555-123-4567"}]}
        applier.apply(body)
        # EMAIL gets replaced; PHONE pattern is NOT in the detect subset
        assert "a@b.com" not in body["messages"][0]["content"]
        assert "555-123-4567" in body["messages"][0]["content"]


class TestApplierDetectViaRegex:
    def test_inline_regex_anonymizes(self) -> None:
        applier, _, _ = _make_applier(
            [
                FieldRule(
                    path="$.messages[*].context.case_number",
                    detect_via_regex=r"\bCASE-\d{4}-\d{6}\b",
                    detect_via_regex_as="CASE_NUMBER",
                )
            ]
        )
        body = {"messages": [{"context": {"case_number": "Refer to CASE-2026-000123 in the file"}}]}
        applier.apply(body)
        assert "CASE-2026-000123" not in body["messages"][0]["context"]["case_number"]


class TestApplierReplaceSynthetic:
    def test_whole_value_swapped(self) -> None:
        applier, session, _ = _make_applier(
            [
                FieldRule(
                    path="$.messages[*].metadata.user_id",
                    replace_with_synthetic=SyntheticReplacement(type="USER_ID", format="user-{n:06d}"),
                )
            ]
        )
        body = {"messages": [{"metadata": {"user_id": "internal-id-9f8a3c"}}]}
        applier.apply(body)
        new_value = body["messages"][0]["metadata"]["user_id"]
        # The whole value is replaced — original never reaches downstream
        assert "9f8a3c" not in new_value
        assert re.fullmatch(r"user-\d{6}", new_value)
        # Reverse map populated for deanonymization
        assert session.deanonymize(new_value) == "internal-id-9f8a3c"

    def test_repeated_value_reuses_token(self) -> None:
        applier, _, _ = _make_applier(
            [
                FieldRule(
                    path="$.messages[*].metadata.user_id",
                    replace_with_synthetic=SyntheticReplacement(type="USER_ID", format="user-{n:06d}"),
                )
            ]
        )
        body = {
            "messages": [
                {"metadata": {"user_id": "abc"}},
                {"metadata": {"user_id": "abc"}},
                {"metadata": {"user_id": "def"}},
            ]
        }
        applier.apply(body)
        u0 = body["messages"][0]["metadata"]["user_id"]
        u1 = body["messages"][1]["metadata"]["user_id"]
        u2 = body["messages"][2]["metadata"]["user_id"]
        assert u0 == u1
        assert u0 != u2


class TestApplierHashStrategy:
    def test_hash_strategy_is_deterministic_within_session(self) -> None:
        rules = [
            FieldRule(
                path="$.messages[*].context.case_number",
                detect_via_regex=r"\bCASE-\d{4}-\d{6}\b",
                detect_via_regex_as="CASE_NUMBER",
                mask_strategy="hash",
            )
        ]

        applier1, _, _ = _make_applier(rules)
        body1 = {"messages": [{"context": {"case_number": "CASE-2026-000123"}}]}
        applier1.apply(body1)
        v1 = body1["messages"][0]["context"]["case_number"]

        # New session via fresh applier
        applier2, _, _ = _make_applier(rules)
        body2 = {"messages": [{"context": {"case_number": "CASE-2026-000123"}}]}
        applier2.apply(body2)
        v2 = body2["messages"][0]["context"]["case_number"]

        # Identity within session: re-apply same value, same hash
        applier1.apply(body1)
        v1_again = body1["messages"][0]["context"]["case_number"]
        assert v1 == v1_again

        # Cross-session: different (per-session salt)
        assert v1 != v2


class TestApplierNoMatch:
    def test_no_matching_path_is_noop_not_error(self) -> None:
        applier, _, _ = _make_applier([FieldRule(path="$.does.not.exist", detect=["EMAIL"])])
        body = {"messages": [{"content": "a@b.com"}]}
        # Must not raise; must not mutate
        applier.apply(body)
        assert body["messages"][0]["content"] == "a@b.com"


class TestFullExamplePayload:
    """End-to-end example from the project description: 4 different rules
    over 4 different paths in one payload."""

    def test_four_rule_payload(self) -> None:
        rules = [
            FieldRule(path="$.messages[*].content", detect=["EMAIL", "PHONE"]),
            FieldRule(
                path="$.messages[*].metadata.user_id",
                replace_with_synthetic=SyntheticReplacement(type="USER_ID", format="user-{n:06d}"),
            ),
            FieldRule(
                path="$.messages[*].context.case_number",
                detect_via_regex=r"\bCASE-\d{4}-\d{6}\b",
                detect_via_regex_as="CASE_NUMBER",
                mask_strategy="hash",
            ),
        ]
        applier, _, _ = _make_applier(rules)
        body: dict[str, list[dict[str, Any]]] = {
            "messages": [
                {
                    "content": "Hi, my email is alice@x.com",
                    "metadata": {"user_id": "u-1"},
                    "context": {"case_number": "CASE-2026-000001"},
                },
                {
                    "content": "Call 555-100-2000",
                    "metadata": {"user_id": "u-2"},
                    "context": {"case_number": "CASE-2026-000002"},
                },
            ]
        }
        owned = applier.apply(body)

        for msg in body["messages"]:
            assert "@x.com" not in msg["content"] if "alice" in msg["content"] else True
            assert "alice@x.com" not in msg["content"]
            assert "555-100-2000" not in msg["content"] if "Call" in msg["content"] else True
            assert msg["metadata"]["user_id"].startswith("user-")
            assert "CASE-2026" not in msg["context"]["case_number"]

        # owned positions cover all six (2 messages, 3 rule kinds each)
        assert len(owned) == 6
