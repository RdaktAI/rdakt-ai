"""Tests for configuration loading."""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from rdakt_ai.config import RdaktConfig, create_store, load_config


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            detection:
              layers:
                - regex
            entities:
              PERSON:
                strategy: synthetic
              EMAIL:
                strategy: token
            session:
              store: memory
            on_error: warn_and_forward
        """)
        )
        config = load_config(config_file)
        assert config.detection_layers == ["regex"]
        assert config.entity_strategies["PERSON"] == "synthetic"
        assert config.entity_strategies["EMAIL"] == "token"
        assert config.session_store == "memory"
        assert config.on_error == "warn_and_forward"

    def test_load_defaults(self) -> None:
        config = RdaktConfig()
        assert config.detection_layers == ["regex"]
        assert config.session_store == "memory"
        assert config.on_error == "warn_and_forward"
        assert config.mode == "active"

    def test_load_with_custom_patterns(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            entities:
              ACCOUNT_NUMBER:
                pattern: '\\d{4}-\\d{4}'
                strategy: token
        """)
        )
        config = load_config(config_file)
        assert "ACCOUNT_NUMBER" in config.custom_patterns
        assert config.custom_patterns["ACCOUNT_NUMBER"] == r"\d{4}-\d{4}"

    def test_load_audit_mode(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("mode: audit\n")
        config = load_config(config_file)
        assert config.mode == "audit"

    def test_load_fail_closed(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("on_error: block\n")
        config = load_config(config_file)
        assert config.on_error == "block"

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config == RdaktConfig()


class TestRdaktConfigPipeline:
    def test_default_pipeline(self) -> None:
        config = RdaktConfig()
        assert config.pipeline == ["regex"]

    def test_store_options_default(self) -> None:
        config = RdaktConfig()
        assert config.store_options == {}


class TestLoadConfigPipeline:
    def test_load_pipeline_from_yaml(self, tmp_path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            pipeline:
              - regex
              - ner:
                  model: en_core_web_sm
        """)
        )
        config = load_config(config_file)
        assert config.pipeline == ["regex", {"ner": {"model": "en_core_web_sm"}}]

    def test_load_pipeline_simple_entries(self, tmp_path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            pipeline:
              - regex
        """)
        )
        config = load_config(config_file)
        assert config.pipeline == ["regex"]

    def test_load_session_store_options(self, tmp_path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            session:
              store: redis
              url: redis://localhost:6379
              ttl: 3600
              prefix: "myapp:"
        """)
        )
        config = load_config(config_file)
        assert config.session_store == "redis"
        assert config.store_options == {
            "url": "redis://localhost:6379",
            "ttl": 3600,
            "prefix": "myapp:",
        }

    def test_backwards_compat_detection_layers(self, tmp_path) -> None:
        """Old detection.layers format maps to pipeline field."""
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            detection:
              layers:
                - regex
        """)
        )
        config = load_config(config_file)
        assert config.pipeline == ["regex"]

    def test_pipeline_key_takes_precedence_over_detection_layers(self, tmp_path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            detection:
              layers:
                - regex
            pipeline:
              - ner
        """)
        )
        config = load_config(config_file)
        assert config.pipeline == ["ner"]

    def test_no_pipeline_no_detection_defaults_to_regex(self, tmp_path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("mode: active\n")
        config = load_config(config_file)
        assert config.pipeline == ["regex"]


class TestCreateStore:
    def test_default_creates_memory_store(self) -> None:
        from rdakt_ai.stores import MemoryStore

        config = RdaktConfig()
        store = create_store(config)
        assert isinstance(store, MemoryStore)

    def test_memory_store_explicit(self) -> None:
        from rdakt_ai.stores import MemoryStore

        config = RdaktConfig(session_store="memory")  # type: ignore[call-arg]
        store = create_store(config)
        assert isinstance(store, MemoryStore)

    def test_sqlite_store(self, tmp_path) -> None:
        from rdakt_ai.stores import SQLiteStore

        config = RdaktConfig(  # type: ignore[call-arg]
            session_store="sqlite",
            store_options={"path": str(tmp_path / "test.db")},
        )
        store = create_store(config)
        assert isinstance(store, SQLiteStore)

    def test_redis_store(self) -> None:
        from unittest.mock import MagicMock, patch

        from rdakt_ai.stores import RedisStore

        with patch("rdakt_ai.stores.redis") as mock_redis:
            mock_redis.Redis.from_url.return_value = MagicMock()
            config = RdaktConfig(  # type: ignore[call-arg]
                session_store="redis",
                store_options={"url": "redis://localhost:6379", "ttl": 3600},
            )
            store = create_store(config)
            assert isinstance(store, RedisStore)

    def test_unknown_store_raises(self) -> None:
        with pytest.raises(ValidationError, match="dynamodb"):
            RdaktConfig(session_store="dynamodb")  # type: ignore[call-arg]


class TestRdaktConfigValidation:
    def test_invalid_pipeline_entry_type_raises(self) -> None:
        with pytest.raises(ValueError, match="pipeline"):
            RdaktConfig(pipeline=[42])  # type: ignore[list-item]

    def test_invalid_pipeline_detector_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown_detector"):
            RdaktConfig(pipeline=["unknown_detector"])

    def test_invalid_pipeline_dict_detector_name_raises(self) -> None:
        with pytest.raises(ValueError, match="bad_detector"):
            RdaktConfig(pipeline=[{"bad_detector": {"model": "x"}}])

    def test_valid_pipeline_entries_accepted(self) -> None:
        config = RdaktConfig(pipeline=["regex", {"ner": {"model": "en_core_web_sm"}}])
        assert config.pipeline == ["regex", {"ner": {"model": "en_core_web_sm"}}]

    def test_invalid_entity_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="not_a_strategy"):
            RdaktConfig(entity_strategies={"EMAIL": "not_a_strategy"})  # type: ignore[call-arg]

    def test_valid_entity_strategies_accepted(self) -> None:
        config = RdaktConfig(entity_strategies={"EMAIL": "token", "PERSON": "synthetic"})  # type: ignore[call-arg]
        assert config.entity_strategies == {"EMAIL": "token", "PERSON": "synthetic"}

    def test_detection_layers_deprecation_warning(self, tmp_path: Path) -> None:
        import warnings

        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("detection:\n  layers:\n    - regex\n")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(config_file)

        assert config.pipeline == ["regex"]
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        assert "pipeline" in str(dep_warnings[0].message)


class TestLoadConfigEdgeCases:
    def test_empty_yaml_file_returns_defaults(self, tmp_path):
        """An empty YAML file returns default config."""
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config.pipeline == ["regex"]
        assert config.session_store == "memory"

    def test_yaml_with_only_mode(self, tmp_path):
        """YAML with only mode set uses defaults for everything else."""
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("mode: audit\n")
        config = load_config(config_file)
        assert config.mode == "audit"
        assert config.pipeline == ["regex"]

    def test_invalid_mode_in_yaml_raises(self, tmp_path):
        """YAML with invalid mode raises ValueError."""
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("mode: stealth\n")
        with pytest.raises(ValueError, match="stealth"):
            load_config(config_file)

    def test_entity_with_pattern_only(self, tmp_path):
        """Entity config with pattern but no strategy is accepted."""
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text('entities:\n  CUSTOM:\n    pattern: "X-\\\\d+"\n')
        config = load_config(config_file)
        assert "CUSTOM" in config.custom_patterns
        assert "CUSTOM" not in config.entity_strategies


class TestPlaceholderConfig:
    def test_default_placeholders(self) -> None:
        config = RdaktConfig()
        assert config.placeholders.template == "<{TYPE}_{N}>"
        assert config.placeholders.case == "preserve"

    def test_load_placeholders_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                placeholders:
                  template: "[{TYPE}#{N}]"
                  case: upper
            """)
        )
        config = load_config(config_file)
        assert config.placeholders.template == "[{TYPE}#{N}]"
        assert config.placeholders.case == "upper"

    def test_invalid_template_rejected_at_load(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        # Empty middle → ambiguous tokens (EMAIL1 vs EMAIL11)
        config_file.write_text('placeholders:\n  template: "<{TYPE}{N}>"\n')
        with pytest.raises(ValueError, match="separator"):
            load_config(config_file)

    def test_unknown_placeholders_field_rejected(self, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text('placeholders:\n  template: "<{TYPE}_{N}>"\n  bogus: 1\n')
        with pytest.raises(ValueError, match=r"bogus|extra"):
            load_config(config_file)

    def test_build_returns_placeholder_format(self) -> None:
        config = RdaktConfig()
        fmt = config.placeholders.build()
        assert fmt.format("EMAIL", 1) == "<EMAIL_1>"
