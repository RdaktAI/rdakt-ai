"""CLI for rdakt-ai: project setup, validation, and demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import httpx
import yaml

from rdakt_ai.cli_helpers import (
    Redaction,
    ValidateReport,
    analyze_sample,
    resolved_config_dict,
    summarise_hits,
    validate_paths,
)
from rdakt_ai.config import RdaktConfig, create_store, load_config
from rdakt_ai.middleware import RdaktSyncMiddleware

# Entity groups for the wizard
_ENTITY_GROUPS: dict[str, dict[str, str]] = {
    "pii": {
        "PERSON": "synthetic",
        "EMAIL": "token",
        "PHONE": "token",
        "IP_ADDRESS": "token",
    },
    "financial": {
        "CREDIT_CARD": "token",
        "SSN": "token",
        "IBAN": "token",
    },
    "secrets": {
        "API_KEY": "token",
        "JWT": "token",
    },
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

_DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "pii": {
        "title": "PII Detection & Anonymization",
        "prompt": (
            "Rewrite this message in a more formal tone, keeping all details exactly as they are: "
            '"Hi, please email john.smith@example.com or call (555) 123-4567 to follow up on the case."'
        ),
    },
    "financial": {
        "title": "Financial Data Protection",
        "prompt": (
            "Format the following customer record as a bullet list, keeping all values exactly as provided: "
            "SSN 123-45-6789, credit card 4111-1111-1111-1111, IBAN DE89370400440532013000."
        ),
    },
    "secrets": {
        "title": "Secrets & API Key Detection",
        "prompt": (
            "Write a curl command that calls https://api.example.com/v1/data using "
            "API key sk-proj-abc123def456ghi789jkl012mno345pqr678 from host 192.168.1.100."
        ),
    },
}


def generate_config(
    *,
    output_path: Path,
    layers: list[str],
    entity_groups: list[str],
    strategy: str,
    store: str,
    force: bool = True,
) -> bool:
    """Generate a rdakt.yaml config file.

    Returns True if file was written, False if skipped.
    """
    if output_path.exists() and not force:
        return False

    entities: dict[str, dict[str, str]] = {}
    for group in entity_groups:
        if group in _ENTITY_GROUPS:
            for name, default_strategy in _ENTITY_GROUPS[group].items():
                s = default_strategy if strategy == "hybrid" else strategy
                entities[name] = {"strategy": s}

    config = {
        "detection": {"layers": layers},
        "entities": entities,
        "session": {"store": store},
        "on_error": "warn_and_forward",
        "mode": "active",
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return True


class _DemoResult:
    """Holds the results of a demo LLM call."""

    def __init__(self, anonymized: str, raw_response: str, restored: str, mapping: dict[str, str]) -> None:
        self.anonymized = anonymized
        self.raw_response = raw_response
        self.restored = restored
        self.mapping = mapping  # token -> original value, e.g. {"<EMAIL_1>": "john@example.com"}


def _call_llm(provider: str, prompt: str, config: RdaktConfig | None = None) -> _DemoResult:
    """Call LLM through RdaktMiddleware and return structured demo result."""
    env_key = _PROVIDER_ENV_KEYS[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        click.echo(f"Error: {env_key} environment variable is not set.", err=True)
        sys.exit(1)

    captured_anonymized: list[str] = []
    captured_mapping: dict[str, str] = {}
    captured_deanonymized: list[str] = []

    def on_anonymized(_original: str, anonymized: str, mapping: dict[str, str]) -> None:
        captured_anonymized.append(anonymized)
        captured_mapping.update(mapping)

    def on_deanonymized(anon_response: str, _restored_response: str) -> None:
        import json

        from rdakt_ai.formats import extract_response_content

        try:
            body = json.loads(anon_response)
            text = extract_response_content(body) or anon_response
        except json.JSONDecodeError:
            text = anon_response
        captured_deanonymized.append(text)

    kwargs: dict[str, object] = {
        "inner": httpx.HTTPTransport(),
        "on_anonymized": on_anonymized,
        "on_deanonymized": on_deanonymized,
    }
    if config is not None:
        kwargs["config"] = config
        kwargs["store"] = create_store(config)

    middleware = RdaktSyncMiddleware(**kwargs)  # type: ignore[arg-type]
    restored = _call_provider(provider, api_key, prompt, middleware)

    return _DemoResult(
        anonymized=captured_anonymized[0] if captured_anonymized else prompt,
        raw_response=captured_deanonymized[0] if captured_deanonymized else restored,
        restored=restored,
        mapping=captured_mapping,
    )


def _call_provider(provider: str, api_key: str, prompt: str, middleware: RdaktSyncMiddleware) -> str:
    """Dispatch a prompt to the chosen LLM provider and return the response text."""
    with httpx.Client(transport=middleware) as http_client:
        if provider == "openai":
            from openai import OpenAI

            oai = OpenAI(api_key=api_key, http_client=http_client)
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        if provider == "anthropic":
            from anthropic import Anthropic

            ant = Anthropic(api_key=api_key, http_client=http_client)
            ant_resp = ant.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return ant_resp.content[0].text  # type: ignore[union-attr]

        # google
        from google import genai

        gclient = genai.Client(
            api_key=api_key,
            http_options={"httpx_client": http_client},  # type: ignore[arg-type]
        )
        gen_resp = gclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return gen_resp.text or ""  # type: ignore[union-attr]


def _highlight_tokens(text: str) -> str:
    """Highlight anonymization tokens like <EMAIL_1> in red."""
    import re

    return re.sub(r"(<[A-Z_]+_\d+>)", lambda m: click.style(m.group(1), fg="red", bold=True), text)


def _highlight_restored_values(text: str, mapping: dict[str, str]) -> str:
    """Highlight restored PII values in green using the token-to-original mapping."""
    result = text
    # Sort by length descending to replace longer values first (avoid partial matches)
    for original_value in sorted(mapping.values(), key=len, reverse=True):
        if original_value in result:
            result = result.replace(original_value, click.style(original_value, fg="green", bold=True))
    return result


def _run_demo_scenario(name: str, scenario: dict[str, str], provider: str, config: RdaktConfig | None = None) -> None:
    """Run a single demo scenario and print results."""
    click.echo(click.style(f"── {scenario['title']} ──", fg="cyan", bold=True))
    click.echo()

    try:
        result = _call_llm(provider, scenario["prompt"], config=config)
    except Exception as e:
        click.echo(click.style("Input:", fg="green", bold=True))
        click.echo(scenario["prompt"])
        click.echo()
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        click.echo()
        return

    click.echo(click.style("Input:", fg="green", bold=True))
    click.echo(_highlight_restored_values(scenario["prompt"], result.mapping))
    click.echo()

    click.echo(click.style("Anonymized:", fg="yellow", bold=True))
    click.echo(_highlight_tokens(result.anonymized))
    click.echo()
    click.echo(click.style("LLM response (anonymized):", fg="yellow", bold=True))
    for line in result.raw_response.splitlines():
        click.echo(_highlight_tokens(line))
    click.echo()
    click.echo(click.style("Restored:", fg="green", bold=True))
    restored_highlighted = _highlight_restored_values(result.restored, result.mapping)
    for line in restored_highlighted.splitlines():
        click.echo(line)
    click.echo()


@click.group()
def main() -> None:
    """Rdakt AI CLI — composable anonymization middleware for LLM interactions."""


_PROVIDERS = [
    "OpenAI / OpenAI-compatible",
    "Anthropic",
    "Google Gemini",
    "LiteLLM router",
    "OpenRouter",
]


@main.command()
@click.option("-o", "--output", default="rdakt.yaml", help="Output config file path.")
@click.option("--no-overwrite", is_flag=True, help="Don't overwrite existing config.")
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Skip the wizard and emit a sensible default config (CI-friendly).",
)
@click.option(
    "--sample",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Sample request body (JSON) to seed the wizard's analysis.",
)
def init(output: str, no_overwrite: bool, non_interactive: bool, sample: Path | None) -> None:
    """Initialize Rdakt AI configuration via an interactive wizard."""
    output_path = Path(output)

    if output_path.exists() and no_overwrite:
        click.echo(f"{output_path} already exists. Drop --no-overwrite to replace it.")
        sys.exit(1)

    click.echo()
    click.echo(click.style("Rdakt AI — Project Setup", bold=True))
    click.echo()

    if non_interactive:
        written = generate_config(
            output_path=output_path,
            layers=["regex"],
            entity_groups=["pii", "financial", "secrets"],
            strategy="hybrid",
            store="memory",
            force=True,
        )
        if written:
            click.echo(f"Wrote {output_path} (non-interactive defaults).")
        return

    # ---- Wizard ----------------------------------------------------------
    click.echo("? What providers do you call? (comma-separated indices, e.g. 1,2)")
    for i, prov in enumerate(_PROVIDERS, 1):
        click.echo(f"  {i}) {prov}")
    raw = click.prompt("  >", default="1", show_default=False)
    chosen_providers = _parse_choice_indices(raw, len(_PROVIDERS))
    click.echo()

    sample_path = sample
    if sample_path is None:
        path_str = click.prompt(
            "? Sample request body (path to JSON, or 'skip')",
            default="skip",
            show_default=False,
        )
        if path_str.strip().lower() not in ("skip", ""):
            sample_path = Path(path_str.strip())

    ontology_rules: list[dict[str, object]] = []
    if sample_path is not None and sample_path.exists():
        try:
            body = json.loads(sample_path.read_text())
        except json.JSONDecodeError as exc:
            click.echo(click.style(f"  Could not parse {sample_path}: {exc}", fg="yellow"))
        else:
            hits = analyze_sample(body)
            summary = summarise_hits(hits)
            if summary:
                click.echo()
                click.echo("  Detected the following PII in the sample:")
                for (etype, path), count in sorted(summary.items()):
                    click.echo(f"    {etype}: {count} instance(s) at {path}")
            else:
                click.echo("  No built-in entities detected in this sample.")
            click.echo()

            structured = _suggest_structured_paths(body)
            if structured and click.confirm(
                "? Want field-specific ontology rules for structured paths?",
                default=False,
            ):
                ontology_rules = _prompt_ontology_rules(structured)
    elif sample_path is not None:
        click.echo(click.style(f"  {sample_path} not found, skipping sample analysis.", fg="yellow"))

    click.echo()
    strategy = click.prompt(
        "? Anonymization strategy",
        type=click.Choice(["token", "synthetic", "hybrid"]),
        default="hybrid",
    )
    mode = click.prompt(
        "? Mode",
        type=click.Choice(["active", "audit"]),
        default="active",
    )

    token = click.prompt("? Set RDAKT_TOKEN now (optional, leave blank to skip)", default="", show_default=False)

    config_dict: dict[str, object] = {
        "mode": mode,
        "on_error": "warn_and_forward",
        "pipeline": ["regex"],
        "session": {"store": "memory"},
        "entities": {
            name: {"strategy": strategy if strategy != "hybrid" else default_s}
            for group in ("pii", "financial", "secrets")
            for name, default_s in _ENTITY_GROUPS[group].items()
        },
    }
    if ontology_rules:
        config_dict["ontology"] = {"fields": ontology_rules}
    # Validate before writing — turns a typo'd ontology rule into a clear
    # error rather than a runtime explosion later.
    RdaktConfig.model_validate(config_dict)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    click.echo()
    click.echo(click.style(f"✓ Wrote {output_path}", fg="green"))
    if token:
        click.echo("  Add `export RDAKT_TOKEN=..." + "*" * 8 + "` to your shell profile.")
    click.echo(f"  Providers selected: {[_PROVIDERS[i] for i in chosen_providers] or ['(none)']}")
    click.echo("  Run `rdakt-ai validate <path>` to dry-run the config against more samples.")


def _parse_choice_indices(raw: str, max_index: int) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            continue
        if 1 <= n <= max_index:
            out.append(n - 1)
    return out


def _suggest_structured_paths(body: object, parent_path: str = "$") -> list[str]:
    """Heuristic: flag paths whose key looks like an opaque ID / structured token."""
    suggestions: list[str] = []
    suspect_suffixes = ("_id", "id", "_number", "_code", "_ref")
    if isinstance(body, dict):
        for key, value in body.items():
            child = f"{parent_path}.{key}"
            if (
                isinstance(value, str)
                and any(key.lower().endswith(s) for s in suspect_suffixes)
                and key.lower() != "id"
            ):
                suggestions.append(child)
            else:
                suggestions.extend(_suggest_structured_paths(value, child))
    elif isinstance(body, list):
        # collapse [0] / [1] / ... into [*] for ontology rules
        for item in body:
            suggestions.extend(_suggest_structured_paths(item, f"{parent_path}[*]"))
    # de-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _prompt_ontology_rules(paths: list[str]) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for path in paths:
        click.echo()
        click.echo(f'? For path "{path}" — what is this?')
        click.echo("  1) Internal ID (replace with synthetic)")
        click.echo("  2) Domain identifier (custom regex)")
        click.echo("  3) Skip — leave to global pipeline")
        choice = click.prompt("  >", type=click.Choice(["1", "2", "3"]), default="3", show_default=False)
        if choice == "1":
            etype = click.prompt("    Entity type name", default="USER_ID")
            fmt = click.prompt("    Synthetic format", default=f"{etype.lower()}-{{n:06d}}")
            rules.append(
                {
                    "path": path,
                    "replace_with_synthetic": {"type": etype, "format": fmt},
                }
            )
        elif choice == "2":
            pattern = click.prompt("    Regex pattern")
            etype = click.prompt("    Entity type name", default="CUSTOM")
            rules.append(
                {
                    "path": path,
                    "detect_via_regex": pattern,
                    "detect_via_regex_as": etype,
                }
            )
    return rules


@main.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("rdakt.yaml"),
    show_default=True,
    help="Path to a rdakt.yaml config file.",
)
@click.option("--summary", is_flag=True, help="Per-file totals only — suppress per-entity rows.")
def validate(paths: tuple[Path, ...], config_path: Path, summary: bool) -> None:
    """Dry-run a config against sample request bodies (JSON files or directories)."""
    if not config_path.exists():
        click.echo(click.style("No rdakt.yaml found, using built-in defaults.", fg="yellow"))
        config = RdaktConfig()
    else:
        config = load_config(config_path)

    files = _expand_json_paths(paths)
    if not files:
        click.echo("No JSON files found.", err=True)
        sys.exit(2)

    report = validate_paths(files, config)
    _print_validate_report(report, summary=summary)
    if report.error_count > 0:
        sys.exit(1)


def _expand_json_paths(paths: tuple[Path, ...]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.json")))
        elif p.suffix == ".json":
            out.append(p)
    return out


def _print_validate_report(report: ValidateReport, *, summary: bool) -> None:
    for file_report in report.files:
        click.echo()
        click.echo(click.style(str(file_report.path), bold=True))
        if file_report.error is not None:
            click.echo(click.style(f"  error: {file_report.error}", fg="red"))
            continue
        if not file_report.redactions:
            click.echo("  (no redactions)")
            continue
        if summary:
            click.echo(f"  {file_report.total} redaction(s)")
            continue
        # Group rows by jsonpath for readability.
        by_path: dict[str, list[Redaction]] = {}
        for row in file_report.redactions:
            by_path.setdefault(row.path, []).append(row)
        for path, rows in by_path.items():
            click.echo(f"  {path}")
            for row in rows:
                click.echo(f"    {row.type:<10} {row.original!r:<40} -> {row.replacement!r}")
    click.echo()
    click.echo(
        f"{len(report.files)} file(s) processed, "
        f"{report.total_redactions} entit(ies) redacted, "
        f"{report.error_count} error(s)."
    )


@main.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a rdakt.yaml config file (default: rdakt.yaml if present).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    show_default=True,
)
def show(config_path: Path | None, fmt: str) -> None:
    """Print the resolved config as the middleware would see it.

    Credentials embedded in URLs (e.g. redis://user:password@host) are
    masked in the output.
    """
    actual_path = config_path
    if actual_path is None and Path("rdakt.yaml").exists():
        actual_path = Path("rdakt.yaml")

    resolved = resolved_config_dict(actual_path)

    click.echo("# Effective config (rdakt.yaml + defaults applied)")
    click.echo()
    if fmt == "json":
        click.echo(json.dumps(resolved, indent=2))
    else:
        click.echo(yaml.dump(resolved, default_flow_style=False, sort_keys=False).rstrip())


@main.command()
@click.option(
    "--scenario",
    type=click.Choice(["pii", "financial", "secrets", "all"]),
    default="all",
    help="Which scenario to run.",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a rdakt.yaml config file.",
)
def demo(scenario: str, provider: str, config_path: Path | None) -> None:
    """Run demo scenarios with a real LLM provider."""
    # Validate API key early
    env_key = _PROVIDER_ENV_KEYS[provider]
    if not os.environ.get(env_key):
        click.echo(f"Error: {env_key} environment variable is not set.", err=True)
        click.echo(f"Set it with: export {env_key}=your-key-here", err=True)
        sys.exit(1)

    config: RdaktConfig | None = None
    if config_path is not None:
        config = load_config(config_path)
        click.echo(f"Loaded config from {config_path}")

    click.echo()
    click.echo(f"Rdakt AI — Demo (provider: {provider})")
    click.echo()

    if scenario == "all":
        for name, sc in _DEMO_SCENARIOS.items():
            _run_demo_scenario(name, sc, provider, config=config)
    else:
        _run_demo_scenario(scenario, _DEMO_SCENARIOS[scenario], provider, config=config)


if __name__ == "__main__":
    main()
