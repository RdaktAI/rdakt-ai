"""Structured logging for Rdakt AI."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_EXTRA_FIELDS = {"entity_count", "session_id", "detection_ms", "entities_by_type", "mode"}


class RdaktLogFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                data[field] = getattr(record, field)

        if record.exc_info and record.exc_info[1]:
            data["exception"] = str(record.exc_info[1])

        return json.dumps(data)


def setup_logging(*, level: int = logging.INFO) -> logging.Logger:
    """Configure the rdakt_ai logger with structured JSON output."""
    logger = logging.getLogger("rdakt_ai")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(RdaktLogFormatter())
        logger.addHandler(handler)

    return logger
