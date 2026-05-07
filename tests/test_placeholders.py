"""Tests for the configurable placeholder format."""

from __future__ import annotations

import pytest

from rdakt_ai.placeholders import PlaceholderFormat


class TestFormatting:
    def test_default_template_produces_canonical_token(self) -> None:
        fmt = PlaceholderFormat()
        assert fmt.format("EMAIL", 1) == "<EMAIL_1>"

    def test_lowercase_template(self) -> None:
        fmt = PlaceholderFormat("<{type}_{n}>")
        assert fmt.format("EMAIL", 2) == "<email_2>"

    def test_double_brace_template(self) -> None:
        fmt = PlaceholderFormat("{{{type}_{n}}}")
        assert fmt.format("EMAIL", 1) == "{{email_1}}"

    def test_bracketed_hash_template(self) -> None:
        fmt = PlaceholderFormat("[{TYPE}#{n}]")
        assert fmt.format("SSN", 7) == "[SSN#7]"

    def test_underscore_wrapped_template(self) -> None:
        fmt = PlaceholderFormat("__{TYPE}_{N}__")
        assert fmt.format("PERSON", 3) == "__PERSON_3__"

    def test_case_upper_overrides_lowercase_slot(self) -> None:
        fmt = PlaceholderFormat("<{type}_{n}>", case="upper")
        assert fmt.format("EMAIL", 1) == "<EMAIL_1>"

    def test_case_lower_overrides_uppercase_slot(self) -> None:
        fmt = PlaceholderFormat("<{TYPE}_{N}>", case="lower")
        assert fmt.format("EMAIL", 1) == "<email_1>"


class TestRoundTrip:
    @pytest.mark.parametrize(
        "template",
        [
            "<{TYPE}_{N}>",
            "<{type}_{n}>",
            "{{{type}_{n}}}",
            "[{TYPE}#{N}]",
            "__{TYPE}_{N}__",
            "[[{TYPE}::{N}]]",
        ],
    )
    def test_format_then_parse_roundtrip(self, template: str) -> None:
        fmt = PlaceholderFormat(template)
        rendered = fmt.format("EMAIL", 42)
        assert fmt.parse(rendered) == ("EMAIL", 42)

    def test_parse_normalises_type_to_upper(self) -> None:
        fmt = PlaceholderFormat("<{type}_{n}>")
        rendered = fmt.format("EMAIL", 1)
        assert rendered == "<email_1>"
        # Parse always returns canonical (upper) entity type
        assert fmt.parse(rendered) == ("EMAIL", 1)


class TestValidation:
    def test_missing_type_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\{TYPE\}"):
            PlaceholderFormat("<TOKEN_{N}>")

    def test_missing_n_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\{N\}"):
            PlaceholderFormat("<{TYPE}_X>")

    def test_both_type_cases_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            PlaceholderFormat("<{TYPE}{type}_{N}>")

    def test_n_before_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="before"):
            PlaceholderFormat("<{N}_{TYPE}>")

    def test_empty_middle_rejected_for_ambiguity(self) -> None:
        with pytest.raises(ValueError, match="separator"):
            PlaceholderFormat("<{TYPE}{N}>")

    def test_empty_suffix_rejected(self) -> None:
        with pytest.raises(ValueError, match="suffix"):
            PlaceholderFormat("<{TYPE}_{N}")

    def test_empty_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="prefix"):
            PlaceholderFormat("{TYPE}_{N}>")

    def test_invalid_case_rejected(self) -> None:
        with pytest.raises(ValueError, match="case"):
            PlaceholderFormat(case="weird")  # type: ignore[arg-type]


class TestFullRegex:
    def test_default_full_re_matches_canonical_token(self) -> None:
        fmt = PlaceholderFormat()
        m = fmt.full_re.search("hello <EMAIL_1> world")
        assert m is not None
        assert m.group(0) == "<EMAIL_1>"
        assert m.group(1) == "EMAIL"
        assert m.group(2) == "1"

    def test_default_full_re_matches_backslash_escaped(self) -> None:
        """Canonical template preserves backslash-escape support for providers
        that escape angle brackets in JSON-encoded streams."""
        fmt = PlaceholderFormat()
        m = fmt.full_re.search(r"hello \<EMAIL_1\> world")
        assert m is not None

    def test_custom_template_full_re_does_not_match_default(self) -> None:
        fmt = PlaceholderFormat("[{TYPE}#{N}]")
        assert fmt.full_re.search("<EMAIL_1>") is None
        assert fmt.full_re.search("[EMAIL#1]") is not None


class TestPartialRegex:
    """Streaming-buffer partial matcher: should match a buffer ending in an
    in-progress placeholder, so the chunk-emitter knows to keep buffering."""

    def test_partial_at_prefix_of_default(self) -> None:
        fmt = PlaceholderFormat()
        assert fmt.partial_re.search("plain text <") is not None
        assert fmt.partial_re.search("plain text <EMA") is not None
        assert fmt.partial_re.search("plain text <EMAIL_1") is not None

    def test_no_partial_after_complete_token(self) -> None:
        fmt = PlaceholderFormat()
        # Trailing period is unrelated to a placeholder
        assert fmt.partial_re.search("just text.") is None

    def test_partial_for_custom_template(self) -> None:
        fmt = PlaceholderFormat("[{TYPE}#{N}]")
        assert fmt.partial_re.search("hello [") is not None
        assert fmt.partial_re.search("hello [EMA") is not None
        assert fmt.partial_re.search("hello [EMAIL#") is not None
        assert fmt.partial_re.search("hello [EMAIL#1") is not None

    def test_partial_for_multichar_prefix(self) -> None:
        fmt = PlaceholderFormat("[[{TYPE}::{N}]]")
        # Just the first opening bracket is a partial of the prefix
        assert fmt.partial_re.search("hi [") is not None
        assert fmt.partial_re.search("hi [[") is not None
        assert fmt.partial_re.search("hi [[EMAIL::1]") is not None


class TestEndToEnd:
    """End-to-end: anonymize then deanonymize using a custom template."""

    @pytest.mark.parametrize(
        "template",
        [
            "<{TYPE}_{N}>",
            "<{type}_{n}>",
            "{{{type}_{n}}}",
            "[{TYPE}#{N}]",
            "__{TYPE}_{N}__",
        ],
    )
    def test_roundtrip_through_anonymizer_and_session(self, template: str) -> None:
        from rdakt_ai.anonymizer import Anonymizer
        from rdakt_ai.models import Entity
        from rdakt_ai.session import RdaktSession

        fmt = PlaceholderFormat(template)
        anonymizer = Anonymizer(placeholder_format=fmt)
        session = RdaktSession(placeholder_format=fmt)

        text = "Email me at john@example.com"
        entities = [Entity(type="EMAIL", value="john@example.com", start=12, end=28)]

        anonymized, mapping = anonymizer.anonymize(text, entities)
        session.add_mappings(mapping)

        # The rendered token must follow the template
        rendered_token = fmt.format("EMAIL", 1)
        assert rendered_token in anonymized
        assert "john@example.com" not in anonymized

        # Deanonymize round-trips
        assert session.deanonymize(anonymized) == text

    def test_streaming_chunk_reassembly_with_custom_template(self) -> None:
        from rdakt_ai.session import RdaktSession

        fmt = PlaceholderFormat("[{TYPE}#{N}]")
        session = RdaktSession(
            entity_map={"[EMAIL#1]": "real@example.com"},
            placeholder_format=fmt,
        )
        # Split the token across two chunks: the first ends mid-token,
        # so the buffer must hold it until the closing `]` arrives.
        first = session.deanonymize_chunk("Hi [EMAIL#")
        second = session.deanonymize_chunk("1] there")

        joined = "".join(first) + "".join(second)
        assert joined == "Hi real@example.com there"

    def test_counter_continues_across_session_with_lowercase_template(self) -> None:
        from rdakt_ai.anonymizer import Anonymizer
        from rdakt_ai.models import Entity
        from rdakt_ai.session import RdaktSession

        fmt = PlaceholderFormat("<{type}_{n}>")
        anonymizer = Anonymizer(placeholder_format=fmt)
        session = RdaktSession(
            entity_map={"<email_1>": "first@example.com", "<email_2>": "second@example.com"},
            placeholder_format=fmt,
        )

        text = "Send to third@example.com"
        entities = [Entity(type="EMAIL", value="third@example.com", start=8, end=25)]  # 'third@example.com' length = 17
        anonymized, mapping = anonymizer.anonymize(text, entities, session=session)

        # Counter must continue after the highest existing one (2 → 3),
        # not reset to 1, and must use the lowercase rendering.
        assert "<email_3>" in anonymized
        assert mapping == {"<email_3>": "third@example.com"}
