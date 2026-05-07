"""Tests for the pure helpers behind the CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdakt_ai.cli_helpers import (
    _redact_secrets,
    analyze_sample,
    resolved_config_dict,
    summarise_hits,
    validate_paths,
)
from rdakt_ai.config import (
    FieldRule,
    OntologyConfig,
    RdaktConfig,
    SyntheticReplacement,
)

# ---------------------------------------------------------------------------
# Sample analysis
# ---------------------------------------------------------------------------


class TestAnalyzeSample:
    def test_finds_email_in_nested_message_content(self) -> None:
        body = {
            "messages": [
                {"role": "user", "content": "ping me at alice@example.com"},
                {"role": "user", "content": "no PII here"},
            ]
        }
        hits = analyze_sample(body)
        assert any(h.type == "EMAIL" and "alice@example.com" in h.value for h in hits)
        assert all(h.path.startswith("$.messages[") for h in hits)

    def test_summary_groups_by_type_and_path(self) -> None:
        from rdakt_ai.cli_helpers import HitKey

        body = {"messages": [{"content": "a@b.com and c@d.com"}]}
        summary = summarise_hits(analyze_sample(body))
        assert summary[HitKey(type="EMAIL", path="$.messages[0].content")] == 2

    def test_no_pii_returns_empty(self) -> None:
        body = {"messages": [{"content": "boring"}], "model": "gpt-4o"}
        assert analyze_sample(body) == []


# ---------------------------------------------------------------------------
# `validate`
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    (tmp_path / "with_pii.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Email me at alice@example.com please"}]})
    )
    (tmp_path / "no_pii.json").write_text(json.dumps({"messages": [{"role": "user", "content": "all good"}]}))
    return tmp_path


class TestValidatePaths:
    def test_reports_redaction_for_email(self, sample_dir: Path) -> None:
        config = RdaktConfig()
        report = validate_paths(
            [sample_dir / "with_pii.json", sample_dir / "no_pii.json"],
            config,
        )
        assert len(report.files) == 2
        with_pii = next(f for f in report.files if f.path.name == "with_pii.json")
        no_pii = next(f for f in report.files if f.path.name == "no_pii.json")
        assert with_pii.total == 1
        assert no_pii.total == 0
        row = with_pii.redactions[0]
        assert row.type == "EMAIL"
        assert row.original == "alice@example.com"
        assert row.replacement == "<EMAIL_1>"
        assert row.path == "$.messages[0].content"

    def test_malformed_json_is_recorded_as_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        report = validate_paths([bad], RdaktConfig())
        assert report.error_count == 1
        assert report.files[0].error is not None

    def test_ontology_rule_is_honoured(self, tmp_path: Path) -> None:
        sample = tmp_path / "ont.json"
        sample.write_text(json.dumps({"messages": [{"metadata": {"user_id": "internal-9f8a"}, "content": "hi"}]}))
        config = RdaktConfig(
            ontology=OntologyConfig(
                fields=[
                    FieldRule(
                        path="$.messages[*].metadata.user_id",
                        replace_with_synthetic=SyntheticReplacement(type="USER_ID", format="user-{n:06d}"),
                    )
                ]
            )
        )
        report = validate_paths([sample], config)
        rows = report.files[0].redactions
        # The user_id was rewritten by the ontology pass — should appear in the diff.
        assert any("internal-9f8a" in row.original and row.replacement.startswith("user-") for row in rows)


# ---------------------------------------------------------------------------
# `show`
# ---------------------------------------------------------------------------


class TestResolvedConfig:
    def test_defaults_when_no_path(self) -> None:
        resolved = resolved_config_dict(None)
        assert resolved["mode"] == "active"
        assert resolved["pipeline"] == ["regex"]
        assert resolved["session"]["store"] == "memory"
        assert resolved["placeholders"]["template"] == "<{TYPE}_{N}>"
        assert "detection_layers" not in resolved

    def test_preserves_ontology_rules(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "rdakt.yaml"
        cfg_path.write_text(
            "ontology:\n"
            "  fields:\n"
            '    - path: "$.messages[*].metadata.user_id"\n'
            "      replace_with_synthetic:\n"
            "        type: USER_ID\n"
            '        format: "user-{n:06d}"\n'
        )
        resolved = resolved_config_dict(cfg_path)
        assert resolved["ontology"]["fields"][0]["path"] == "$.messages[*].metadata.user_id"

    def test_redis_password_is_masked(self) -> None:
        node = {"session": {"url": "redis://user:supersecret@host:6379"}}
        out = _redact_secrets(node)
        assert "supersecret" not in out["session"]["url"]
        assert "***" in out["session"]["url"]
