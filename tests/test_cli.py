"""Tests for rdakt-ai CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from rdakt_ai.cli import _DemoResult, generate_config, main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestGenerateConfig:
    def test_generates_yaml_with_defaults(self, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        generate_config(
            output_path=output,
            layers=["regex"],
            entity_groups=["pii", "financial"],
            strategy="hybrid",
            store="memory",
        )
        assert output.exists()
        content = output.read_text()
        assert "regex" in content
        assert "PERSON" in content
        assert "CREDIT_CARD" in content
        assert "memory" in content

    def test_generates_with_secrets(self, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        generate_config(
            output_path=output,
            layers=["regex"],
            entity_groups=["secrets"],
            strategy="token",
            store="memory",
        )
        content = output.read_text()
        assert "API_KEY" in content

    def test_does_not_overwrite_without_force(self, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        output.write_text("existing")
        result = generate_config(
            output_path=output,
            layers=["regex"],
            entity_groups=["pii"],
            strategy="hybrid",
            store="memory",
            force=False,
        )
        assert result is False
        assert output.read_text() == "existing"


class TestCLIEndToEnd:
    def test_init_non_interactive_generates_config(self, runner: CliRunner, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        result = runner.invoke(main, ["init", "-o", str(output), "--non-interactive"])
        assert result.exit_code == 0, result.output
        assert output.exists()
        content = output.read_text()
        assert "regex" in content
        assert "EMAIL" in content

    def test_init_wizard_with_scripted_answers(self, runner: CliRunner, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        # Answers in order: providers, sample path, strategy, mode, RDAKT_TOKEN
        scripted = "1\nskip\ntoken\nactive\n\n"
        result = runner.invoke(main, ["init", "-o", str(output)], input=scripted)
        assert result.exit_code == 0, result.output
        assert output.exists()
        content = output.read_text()
        assert "mode: active" in content
        assert "EMAIL" in content
        assert "Wrote" in result.output

    def test_init_wizard_with_sample_runs_analysis(self, runner: CliRunner, tmp_path: Path) -> None:
        sample = tmp_path / "sample.json"
        sample.write_text('{"messages": [{"content": "ping me at alice@example.com"}]}')
        output = tmp_path / "rdakt.yaml"
        # providers, sample path is preset via --sample, then ontology y/n,
        # strategy, mode, token. We pass `n` to skip ontology because the
        # sample has no obvious structured paths.
        scripted = "1\nn\ntoken\nactive\n\n"
        result = runner.invoke(main, ["init", "-o", str(output), "--sample", str(sample)], input=scripted)
        assert result.exit_code == 0, result.output
        assert "EMAIL: 1 instance(s)" in result.output

    def test_init_no_overwrite_refuses_existing(self, runner: CliRunner, tmp_path: Path) -> None:
        output = tmp_path / "rdakt.yaml"
        output.write_text("existing")
        result = runner.invoke(main, ["init", "-o", str(output), "--no-overwrite"])
        assert result.exit_code == 1
        assert output.read_text() == "existing"

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, [])
        assert "Commands:" in result.output


class TestValidateCommand:
    def test_validate_reports_redactions(self, runner: CliRunner, tmp_path: Path) -> None:
        sample = tmp_path / "s.json"
        sample.write_text('{"messages": [{"content": "alice@example.com"}]}')
        result = runner.invoke(main, ["validate", str(sample)])
        assert result.exit_code == 0, result.output
        assert "EMAIL" in result.output
        assert "alice@example.com" in result.output
        assert "1 entit" in result.output

    def test_validate_summary_flag_hides_rows(self, runner: CliRunner, tmp_path: Path) -> None:
        sample = tmp_path / "s.json"
        sample.write_text('{"messages": [{"content": "alice@example.com"}]}')
        result = runner.invoke(main, ["validate", "--summary", str(sample)])
        assert result.exit_code == 0
        assert "EMAIL" not in result.output  # detail rows hidden
        assert "redaction(s)" in result.output

    def test_validate_directory_walks_json(self, runner: CliRunner, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text('{"messages": [{"content": "a@b.com"}]}')
        (tmp_path / "b.json").write_text('{"messages": [{"content": "ok"}]}')
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert result.exit_code == 0
        assert "2 file(s) processed" in result.output

    def test_validate_no_json_returns_nonzero(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert result.exit_code == 2

    def test_validate_malformed_json_exits_one(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        result = runner.invoke(main, ["validate", str(bad)])
        assert result.exit_code == 1


class TestShowCommand:
    def test_show_yaml_default(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["show"])
        assert result.exit_code == 0, result.output
        assert "mode: active" in result.output
        assert "Effective config" in result.output

    def test_show_json_format(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["show", "--format", "json"])
        assert result.exit_code == 0
        # The output starts with the comment line; the JSON body follows.
        assert '"mode": "active"' in result.output


class TestDemoCommand:
    def test_demo_requires_api_key(self, runner: CliRunner) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(main, ["demo"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    def test_demo_requires_provider_api_key(self, runner: CliRunner) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(main, ["demo", "--provider", "anthropic"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    def test_demo_single_scenario_requires_key(self, runner: CliRunner) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(main, ["demo", "--scenario", "pii"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    def test_demo_calls_llm(self, runner: CliRunner) -> None:
        mock_result = _DemoResult(
            anonymized="<EMAIL_1>",
            raw_response="I'll contact <EMAIL_1>",
            restored="I'll contact john@example.com",
            mapping={"<EMAIL_1>": "john@example.com"},
        )
        with (
            patch("rdakt_ai.cli._call_llm", return_value=mock_result) as mock_call,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            result = runner.invoke(main, ["demo", "--scenario", "pii"])
        assert result.exit_code == 0
        assert "PII" in result.output
        assert "Anonymized" in result.output
        assert "Restored" in result.output
        mock_call.assert_called_once()

    def test_demo_all_scenarios(self, runner: CliRunner) -> None:
        mock_result = _DemoResult(
            anonymized="<TOKEN_1>",
            raw_response="anon response",
            restored="restored",
            mapping={"<TOKEN_1>": "original"},
        )
        with (
            patch("rdakt_ai.cli._call_llm", return_value=mock_result),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            result = runner.invoke(main, ["demo"])
        assert result.exit_code == 0
        assert "PII" in result.output
        assert "Financial" in result.output
        assert "Secrets" in result.output

    def test_demo_with_config_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        config_file = tmp_path / "rdakt.yaml"
        config_file.write_text("mode: active\non_error: block\n")

        mock_result = _DemoResult(
            anonymized="<EMAIL_1>",
            raw_response="anon",
            restored="restored",
            mapping={"<EMAIL_1>": "john@example.com"},
        )
        with (
            patch("rdakt_ai.cli._call_llm", return_value=mock_result) as mock_call,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            result = runner.invoke(main, ["demo", "--scenario", "pii", "-c", str(config_file)])
        assert result.exit_code == 0
        assert "Loaded config" in result.output
        # Verify config was passed through to _call_llm
        _, kwargs = mock_call.call_args
        assert kwargs["config"] is not None
        assert kwargs["config"].on_error == "block"

    def test_demo_config_file_not_found(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["demo", "-c", "/nonexistent/rdakt.yaml"])
        assert result.exit_code != 0
