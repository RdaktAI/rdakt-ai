"""Custom Detector: implement the Detector protocol for your own detection logic.

Any class with a `detect(text: str) -> list[Entity]` method satisfies the protocol.
"""

from __future__ import annotations

import asyncio
import re

from rdakt_ai import Detector, Entity, RdaktPipeline
from rdakt_ai.pipeline import DetectorStage


class EmployeeIdDetector:
    """Detects employee IDs in the format EMP-XXXXXX."""

    def detect(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for match in re.finditer(r"EMP-\d{6}", text):
            entities.append(
                Entity(
                    value=match.group(),
                    type="EMPLOYEE_ID",
                    start=match.start(),
                    end=match.end(),
                )
            )
        return entities


# Verify it satisfies the protocol
assert isinstance(EmployeeIdDetector(), Detector)

# Use it in a pipeline
pipeline = RdaktPipeline(stages=[DetectorStage(EmployeeIdDetector())])

text = "Contact EMP-123456 or EMP-789012 for details"
entities = asyncio.run(pipeline.detect(text))
print(f"Input:    {text}")
print(f"Detected: {[(e.type, e.value) for e in entities]}")
print("Done.")
