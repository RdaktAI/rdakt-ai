"""Tests for RegexDetector."""

import pytest

from rdakt_ai.detectors import Detector
from rdakt_ai.detectors.regex import RegexDetector


class TestRegexDetector:
    def setup_method(self) -> None:
        self.detector = RegexDetector()

    def test_detect_email(self) -> None:
        text = "Contact john.smith@example.com for details"
        entities = self.detector.detect(text)
        emails = [e for e in entities if e.type == "EMAIL"]
        assert len(emails) == 1
        assert emails[0].value == "john.smith@example.com"

    def test_detect_phone_us(self) -> None:
        text = "Call me at (555) 123-4567"
        entities = self.detector.detect(text)
        phones = [e for e in entities if e.type == "PHONE"]
        assert len(phones) == 1

    def test_detect_ssn(self) -> None:
        text = "SSN: 123-45-6789"
        entities = self.detector.detect(text)
        ssns = [e for e in entities if e.type == "SSN"]
        assert len(ssns) == 1
        assert ssns[0].value == "123-45-6789"

    def test_detect_credit_card(self) -> None:
        text = "Card: 4111-1111-1111-1111"
        entities = self.detector.detect(text)
        cards = [e for e in entities if e.type == "CREDIT_CARD"]
        assert len(cards) == 1

    def test_detect_credit_card_no_dashes(self) -> None:
        text = "Card: 4111111111111111"
        entities = self.detector.detect(text)
        cards = [e for e in entities if e.type == "CREDIT_CARD"]
        assert len(cards) == 1

    def test_detect_api_key_openai(self) -> None:
        text = "key: sk-proj-abc123def456ghi789jkl012mno345pqr678"
        entities = self.detector.detect(text)
        keys = [e for e in entities if e.type == "API_KEY"]
        assert len(keys) == 1

    def test_detect_aws_key(self) -> None:
        text = "AKIAIOSFODNN7EXAMPLE"
        entities = self.detector.detect(text)
        keys = [e for e in entities if e.type == "API_KEY"]
        assert len(keys) == 1

    def test_detect_ip_address(self) -> None:
        text = "Server at 192.168.1.100"
        entities = self.detector.detect(text)
        ips = [e for e in entities if e.type == "IP_ADDRESS"]
        assert len(ips) == 1
        assert ips[0].value == "192.168.1.100"

    def test_detect_iban(self) -> None:
        text = "IBAN: DE89370400440532013000"
        entities = self.detector.detect(text)
        ibans = [e for e in entities if e.type == "IBAN"]
        assert len(ibans) == 1

    def test_detect_jwt(self) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        text = f"token: {jwt}"
        entities = self.detector.detect(text)
        jwts = [e for e in entities if e.type == "JWT"]
        assert len(jwts) == 1

    def test_detect_multiple_types(self) -> None:
        text = "Email john@example.com, SSN 123-45-6789, IP 10.0.0.1"
        entities = self.detector.detect(text)
        types = {e.type for e in entities}
        assert "EMAIL" in types
        assert "SSN" in types
        assert "IP_ADDRESS" in types

    def test_no_false_positives_on_clean_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        entities = self.detector.detect(text)
        assert len(entities) == 0

    def test_entity_positions_are_correct(self) -> None:
        text = "Email: test@example.com here"
        entities = self.detector.detect(text)
        email = entities[0]
        assert text[email.start : email.end] == "test@example.com"

    def test_custom_pattern(self) -> None:
        detector = RegexDetector(custom_patterns={"ACCOUNT_NUMBER": r"\d{4}-\d{4}"})
        text = "Account 1234-5678"
        entities = detector.detect(text)
        accts = [e for e in entities if e.type == "ACCOUNT_NUMBER"]
        assert len(accts) == 1
        assert accts[0].value == "1234-5678"


class TestRegexDetectorValidation:
    def test_invalid_custom_pattern_raises_with_name(self):
        """Invalid regex in custom_patterns raises ValueError with pattern name."""
        with pytest.raises(ValueError, match="BROKEN"):
            RegexDetector(custom_patterns={"BROKEN": "[invalid("})

    def test_valid_custom_pattern_accepted(self):
        """Valid custom patterns compile without error."""
        detector = RegexDetector(custom_patterns={"EMPLOYEE_ID": r"EMP-\d{6}"})
        result = detector.detect("Employee EMP-123456 is active")
        assert len(result) == 1
        assert result[0].type == "EMPLOYEE_ID"


class TestDetectorProtocol:
    def test_regex_detector_satisfies_protocol(self) -> None:
        detector = RegexDetector()
        assert isinstance(detector, Detector)

    def test_protocol_requires_detect_method(self) -> None:
        class BadDetector:
            pass

        assert not isinstance(BadDetector(), Detector)
