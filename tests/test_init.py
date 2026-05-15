"""Tests for package exports."""


def test_version() -> None:
    from rdakt_ai import __version__

    assert __version__


def test_all_exports_importable() -> None:
    from rdakt_ai import (
        AnonymizationStrategy,
        Anonymizer,
        Entity,
        MemoryStore,
        RdaktConfig,
        RdaktContext,
        RdaktMiddleware,
        RdaktPipeline,
        RdaktSession,
        RdaktStage,
        RdaktSyncMiddleware,
        RegexDetector,
        RegexStage,
        SessionStore,
        load_config,
    )

    # Verify they're all the real classes, not None
    assert RdaktMiddleware is not None
    assert RdaktSyncMiddleware is not None
    assert Entity is not None
    assert load_config is not None
    assert Anonymizer is not None
    assert AnonymizationStrategy is not None
    assert MemoryStore is not None
    assert RdaktConfig is not None
    assert RdaktContext is not None
    assert RdaktPipeline is not None
    assert RdaktSession is not None
    assert RdaktStage is not None
    assert RegexDetector is not None
    assert RegexStage is not None
    assert SessionStore is not None


def test_all_list_complete() -> None:
    import rdakt_ai

    expected = {
        "RdaktMiddleware",
        "RdaktSyncMiddleware",
        "RdaktConfig",
        "load_config",
        "RegexDetector",
        "RdaktPipeline",
        "DetectorStage",
        "RegexStage",
        "RdaktStage",
        "Anonymizer",
        "AnonymizationStrategy",
        "RdaktSession",
        "SessionStore",
        "MemoryStore",
        "Entity",
        "RdaktContext",
        "VALID_MODES",
        "VALID_ERROR_POLICIES",
        "extract_sse_content",
        "set_sse_content",
        "extract_response_content",
        "RedisStore",
        "SQLiteStore",
        "SpacyDetector",
        "NerStage",
        "NotSupportedError",
        "Detector",
        "create_store",
        "VALID_PIPELINE_DETECTORS",
        "VALID_STRATEGIES",
        "EntityConfig",
        "SessionConfig",
    }
    assert set(rdakt_ai.__all__) == expected


def test_exports_redis_store() -> None:
    from rdakt_ai import RedisStore

    assert RedisStore is not None


def test_exports_sqlite_store() -> None:
    from rdakt_ai import SQLiteStore

    assert SQLiteStore is not None


def test_exports_spacy_detector() -> None:
    from rdakt_ai import SpacyDetector

    assert SpacyDetector is not None


def test_exports_ner_stage() -> None:
    from rdakt_ai import NerStage

    assert NerStage is not None


def test_exports_detector_protocol() -> None:
    from rdakt_ai import Detector

    assert Detector is not None


def test_exports_create_store() -> None:
    from rdakt_ai import create_store

    assert create_store is not None
