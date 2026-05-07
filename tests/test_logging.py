"""Tests for structured logging."""

import json
import logging

from rdakt_ai.logging import RdaktLogFormatter, setup_logging


class TestRdaktLogFormatter:
    def test_formats_as_json(self) -> None:
        formatter = RdaktLogFormatter()
        record = logging.LogRecord(
            name="rdakt_ai",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Detected %d entities",
            args=(3,),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Detected 3 entities"
        assert data["level"] == "INFO"
        assert data["logger"] == "rdakt_ai"

    def test_includes_extra_fields(self) -> None:
        formatter = RdaktLogFormatter()
        record = logging.LogRecord(
            name="rdakt_ai",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.entity_count = 5  # type: ignore[attr-defined]
        record.session_id = "abc"  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert data["entity_count"] == 5
        assert data["session_id"] == "abc"


class TestExceptionFormatting:
    def test_includes_exception_string(self) -> None:
        formatter = RdaktLogFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="rdakt_ai",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Something failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["exception"] == "test error"


class TestSetupLogging:
    def test_sets_up_logger(self) -> None:
        logger = setup_logging(level=logging.DEBUG)
        assert logger.name == "rdakt_ai"
        assert logger.level == logging.DEBUG
