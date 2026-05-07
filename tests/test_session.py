"""Tests for RdaktSession — mapping, deanonymization, and streaming."""

from rdakt_ai.session import RdaktSession


class TestDeanonymize:
    def test_deanonymize_text(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John Smith", "<EMAIL_1>": "john@example.com"},
        )
        text = "I recommend <PERSON_1> contacts <EMAIL_1>"
        result = session.deanonymize(text)
        assert result == "I recommend John Smith contacts john@example.com"

    def test_deanonymize_no_tokens(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<PERSON_1>": "John"})
        assert session.deanonymize("Hello world") == "Hello world"

    def test_deanonymize_empty_map(self) -> None:
        session = RdaktSession(session_id="test", entity_map={})
        assert session.deanonymize("Hello world") == "Hello world"


class TestDeanonymizeStructured:
    def test_deanonymize_dict(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John Smith"},
        )
        data = {"to": "<PERSON_1>", "subject": "Hello"}
        result = session.deanonymize_structured(data)
        assert result == {"to": "John Smith", "subject": "Hello"}

    def test_deanonymize_nested_dict(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        data = {"args": {"name": "<PERSON_1>", "count": 5}}
        result = session.deanonymize_structured(data)
        assert result["args"]["name"] == "John"
        assert result["args"]["count"] == 5

    def test_deanonymize_list(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        data = ["<PERSON_1>", "other"]
        result = session.deanonymize_structured(data)
        assert result == ["John", "other"]


class TestEscapedTokenDeanonymize:
    """LLMs may backslash-escape angle brackets in responses (\\<TOKEN\\>)."""

    def test_deanonymize_escaped_token(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<SSN_1>": "123-45-6789"})
        assert session.deanonymize("SSN: \\<SSN_1\\>") == "SSN: 123-45-6789"

    def test_deanonymize_mixed_escaped_and_plain(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<SSN_1>": "123-45-6789", "<EMAIL_1>": "a@b.com"},
        )
        assert session.deanonymize("\\<SSN_1\\> and <EMAIL_1>") == "123-45-6789 and a@b.com"

    def test_deanonymize_structured_escaped(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<SSN_1>": "123-45-6789"})
        data = {"text": "SSN: \\<SSN_1\\>"}
        assert session.deanonymize_structured(data) == {"text": "SSN: 123-45-6789"}

    def test_deanonymize_chunk_escaped_complete(self) -> None:
        """Escaped token arriving in a single chunk."""
        session = RdaktSession(session_id="test", entity_map={"<SSN_1>": "123-45-6789"})
        result = session.deanonymize_chunk("SSN: \\<SSN_1\\>")
        assert "".join(result) == "SSN: 123-45-6789"

    def test_deanonymize_chunk_escaped_split(self) -> None:
        """Escaped token split across chunks: backslash in one, angle bracket in next."""
        session = RdaktSession(session_id="test", entity_map={"<SSN_1>": "123-45-6789"})
        r1 = session.deanonymize_chunk("SSN: \\<")
        r2 = session.deanonymize_chunk("SSN_1\\>")
        assert "".join(r1) + "".join(r2) == "SSN: 123-45-6789"

    def test_deanonymize_chunk_escaped_split_at_backslash(self) -> None:
        """Escaped token split right before the backslash."""
        session = RdaktSession(session_id="test", entity_map={"<SSN_1>": "123-45-6789"})
        r1 = session.deanonymize_chunk("SSN: \\")
        r2 = session.deanonymize_chunk("<SSN_1\\>")
        assert "".join(r1) + "".join(r2) == "SSN: 123-45-6789"


class TestStreamingDeanonymize:
    def test_complete_token_in_single_chunk(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John Smith"},
        )
        result = session.deanonymize_chunk("Hello <PERSON_1>!")
        assert "".join(result) == "Hello John Smith!"

    def test_token_split_across_chunks(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John Smith"},
        )
        r1 = session.deanonymize_chunk("Hello <PERS")
        r2 = session.deanonymize_chunk("ON_1> world")
        combined = "".join(r1) + "".join(r2)
        assert combined == "Hello John Smith world"

    def test_literal_angle_bracket_not_stalled(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        result = session.deanonymize_chunk("x < 5 is true")
        assert "".join(result) == "x < 5 is true"

    def test_multiple_tokens_in_stream(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John", "<EMAIL_1>": "john@test.com"},
        )
        result = session.deanonymize_chunk("<PERSON_1> has email <EMAIL_1>")
        assert "".join(result) == "John has email john@test.com"

    def test_flush_remaining(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        session.deanonymize_chunk("Hello <PERS")
        remaining = session.flush()
        assert "".join(remaining) == "<PERS"

    def test_empty_chunk(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        result = session.deanonymize_chunk("")
        assert result == []


class TestSessionEdgeCases:
    def test_real_values_property(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John", "<EMAIL_1>": "john@test.com"},
        )
        assert session.real_values == {"John", "john@test.com"}

    def test_buffer_safety_valve(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        # Send text longer than MAX_BUFFER (256) with no token match
        long_text = "x" * 300
        result = session.deanonymize_chunk(long_text)
        combined = "".join(result)
        assert combined == long_text


class TestSessionAccumulation:
    def test_add_mappings(self) -> None:
        session = RdaktSession(session_id="test", entity_map={})
        session.add_mappings({"<PERSON_1>": "John"})
        assert session.deanonymize("<PERSON_1>") == "John"

    def test_accumulate_across_turns(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<PERSON_1>": "John"})
        session.add_mappings({"<EMAIL_1>": "john@test.com"})
        result = session.deanonymize("<PERSON_1> at <EMAIL_1>")
        assert result == "John at john@test.com"

    def test_reverse_map(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={"<PERSON_1>": "John"},
        )
        assert session.get_anonymized("John") == "<PERSON_1>"
        assert session.get_anonymized("Unknown") is None


class TestSessionLookups:
    def test_get_token_for_value_found(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "john@test.com"})
        assert session.get_token_for_value("john@test.com") == "<EMAIL_1>"

    def test_get_token_for_value_not_found(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "john@test.com"})
        assert session.get_token_for_value("other@test.com") is None

    def test_get_token_for_value_empty_session(self) -> None:
        session = RdaktSession(session_id="test")
        assert session.get_token_for_value("anything") is None

    def test_token_names_property(self) -> None:
        session = RdaktSession(
            session_id="test",
            entity_map={
                "<EMAIL_1>": "john@test.com",
                "<SSN_1>": "123-45-6789",
            },
        )
        assert session.token_names == {"<EMAIL_1>", "<SSN_1>"}

    def test_token_names_empty(self) -> None:
        session = RdaktSession(session_id="test")
        assert session.token_names == set()

    def test_token_names_updates_after_add_mappings(self) -> None:
        session = RdaktSession(session_id="test", entity_map={"<EMAIL_1>": "a@b.com"})
        session.add_mappings({"<SSN_1>": "123-45-6789"})
        assert session.token_names == {"<EMAIL_1>", "<SSN_1>"}
