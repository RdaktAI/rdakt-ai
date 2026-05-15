"""Rdakt AI — composable anonymization middleware for LLM interactions."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("rdakt-ai")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from rdakt_ai.anonymizer import AnonymizationStrategy, Anonymizer
from rdakt_ai.config import (
    VALID_ERROR_POLICIES,
    VALID_MODES,
    VALID_PIPELINE_DETECTORS,
    VALID_STRATEGIES,
    EntityConfig,
    RdaktConfig,
    SessionConfig,
    create_store,
    load_config,
)
from rdakt_ai.detectors import Detector
from rdakt_ai.detectors.ner import SpacyDetector
from rdakt_ai.detectors.regex import RegexDetector
from rdakt_ai.formats import extract_response_content, extract_sse_content, set_sse_content
from rdakt_ai.integrations import NotSupportedError
from rdakt_ai.middleware import RdaktMiddleware, RdaktSyncMiddleware
from rdakt_ai.models import Entity, RdaktContext
from rdakt_ai.pipeline import DetectorStage, NerStage, RdaktPipeline, RdaktStage, RegexStage
from rdakt_ai.session import RdaktSession
from rdakt_ai.stores import MemoryStore, RedisStore, SessionStore, SQLiteStore

__all__ = [
    "VALID_ERROR_POLICIES",
    "VALID_MODES",
    "VALID_PIPELINE_DETECTORS",
    "VALID_STRATEGIES",
    "AnonymizationStrategy",
    "Anonymizer",
    "Detector",
    "DetectorStage",
    "Entity",
    "EntityConfig",
    "MemoryStore",
    "NerStage",
    "NotSupportedError",
    "RdaktConfig",
    "RdaktContext",
    "RdaktMiddleware",
    "RdaktPipeline",
    "RdaktSession",
    "RdaktStage",
    "RdaktSyncMiddleware",
    "RedisStore",
    "RegexDetector",
    "RegexStage",
    "SQLiteStore",
    "SessionConfig",
    "SessionStore",
    "SpacyDetector",
    "create_store",
    "extract_response_content",
    "extract_sse_content",
    "load_config",
    "set_sse_content",
]
