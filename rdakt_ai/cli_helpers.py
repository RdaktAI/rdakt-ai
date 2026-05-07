"""Pure helpers behind the `rdakt` CLI commands.

Kept separate from ``cli.py`` so the wizard / validate / show logic can be
unit-tested without going through Click's runner. The helpers don't print
anything: they return structured results, and the CLI wrapper handles I/O.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from rdakt_ai.config import RdaktConfig, load_config
from rdakt_ai.detectors.regex import RegexDetector
from rdakt_ai.ontology import OntologyApplier, OwnedKey
from rdakt_ai.session import RdaktSession

# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------


class JsonStringLeaf(NamedTuple):
    """One string-valued leaf in a JSON tree, addressed by its concrete JSONPath."""

    path: str
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    """One regex match in one place."""

    type: str
    value: str
    path: str


class HitKey(NamedTuple):
    """Group key for :func:`summarise_hits`: ``(entity_type, jsonpath)``."""

    type: str
    path: str


@dataclass(frozen=True, slots=True)
class Redaction:
    """A single anonymization event in the validate report."""

    path: str
    type: str
    original: str
    replacement: str


# Public aliases — kept so external imports don't break.
_Hit = Hit


# ---------------------------------------------------------------------------
# Sample analysis (used by `init` wizard and `validate`)
# ---------------------------------------------------------------------------


JsonValue = Any
"""Opaque alias for parsed-JSON values: dict / list / str / int / float / bool / None.
Kept as ``Any`` because the structure is decided by the caller's payload, not us."""


def _walk_strings(node: JsonValue, path: str = "$") -> list[JsonStringLeaf]:
    """Yield every string-valued leaf in *node* with its concrete JSONPath.

    Array indices are emitted as ``[0]`` / ``[1]`` / ... — concrete, not
    wildcards — because the reports talk about *what was actually found*
    in the sample, not what the ontology rule would target.
    """
    out: list[JsonStringLeaf] = []
    if isinstance(node, str):
        out.append(JsonStringLeaf(path=path, text=node))
    elif isinstance(node, dict):
        for key, value in node.items():
            out.extend(_walk_strings(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            out.extend(_walk_strings(item, f"{path}[{i}]"))
    return out


def analyze_sample(body: JsonValue, detector: RegexDetector | None = None) -> list[Hit]:
    """Run the regex detector on every string in *body* and return all hits.

    Pure function with no side-effects so the wizard and `validate` can both
    consume it. Caller groups / formats as needed.
    """
    det = detector or RegexDetector()
    hits: list[Hit] = []
    for leaf in _walk_strings(body):
        for entity in det.detect(leaf.text):
            hits.append(Hit(type=entity.type, value=entity.value, path=leaf.path))
    return hits


def summarise_hits(hits: list[Hit]) -> dict[HitKey, int]:
    """Group hits by ``(type, path)`` with their counts."""
    return dict(Counter(HitKey(type=h.type, path=h.path) for h in hits))


# ---------------------------------------------------------------------------
# `validate` — dry-run report
# ---------------------------------------------------------------------------


@dataclass
class FileReport:
    """Per-file validate result."""

    path: Path
    redactions: list[Redaction] = field(default_factory=list)
    error: str | None = None

    @property
    def total(self) -> int:
        return len(self.redactions)


@dataclass
class ValidateReport:
    """Aggregate report across one or more files."""

    files: list[FileReport] = field(default_factory=list)

    @property
    def total_redactions(self) -> int:
        return sum(f.total for f in self.files)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.files if f.error is not None)


def validate_paths(paths: list[Path], config: RdaktConfig) -> ValidateReport:
    """Dry-run *config* against each JSON file in *paths*.

    Each file gets its own session so synthetic counters reset between
    files (matches the per-conversation runtime model). The per-file
    report records one row per detected entity (or whole-value
    replacement), tagged with the JSONPath where it was found.
    """
    placeholder_format = config.placeholders.build()
    report = ValidateReport()
    detector = RegexDetector(custom_patterns=config.custom_patterns)

    for file_path in paths:
        file_report = FileReport(path=file_path)
        try:
            body: JsonValue = json.loads(file_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            file_report.error = str(exc)
            report.files.append(file_report)
            continue

        before: JsonValue = copy.deepcopy(body)

        from rdakt_ai.anonymizer import AnonymizationStrategy, Anonymizer

        type_strategies = {t: AnonymizationStrategy(s) for t, s in config.entity_strategies.items()}
        session = RdaktSession(session_id=f"validate-{file_path.stem}", placeholder_format=placeholder_format)
        anonymizer = Anonymizer(type_strategies=type_strategies, placeholder_format=placeholder_format)

        owned: set[OwnedKey] = set()
        if config.ontology is not None and config.ontology.fields:
            applier = OntologyApplier(config.ontology, anonymizer=anonymizer, session=session)
            owned = applier.apply(body)

        if owned:
            file_report.redactions.extend(_ontology_rows(before, body))

        for leaf in _walk_strings(body):
            entities = detector.detect(leaf.text)
            if not entities:
                continue
            new_value, mapping = anonymizer.anonymize(leaf.text, entities, session=session)
            session.add_mappings(mapping)
            _set_at_path(body, leaf.path, new_value)
            reverse_mapping = {orig: tok for tok, orig in mapping.items()}
            for entity in entities:
                replacement = reverse_mapping.get(entity.value) or session.get_token_for_value(entity.value) or ""
                file_report.redactions.append(
                    Redaction(path=leaf.path, type=entity.type, original=entity.value, replacement=replacement)
                )

        report.files.append(file_report)

    return report


def _ontology_rows(before: JsonValue, after: JsonValue) -> list[Redaction]:
    """For every string leaf that the ontology pass changed, produce one row."""
    rows: list[Redaction] = []

    def walk(b: JsonValue, a: JsonValue, path: str) -> None:
        if isinstance(b, dict) and isinstance(a, dict):
            for key in b:
                if key in a:
                    walk(b[key], a[key], f"{path}.{key}")
        elif isinstance(b, list) and isinstance(a, list):
            for i in range(min(len(b), len(a))):
                walk(b[i], a[i], f"{path}[{i}]")
        elif isinstance(b, str) and isinstance(a, str) and b != a:
            rows.append(Redaction(path=path, type="REDACTED", original=b, replacement=a))

    walk(before, after, "$")
    return rows


_PATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def _set_at_path(body: JsonValue, path: str, new_value: str) -> None:
    """Set the value at a concrete (no-wildcard) JSONPath."""
    assert path.startswith("$")
    parent: JsonValue = body
    tokens = list(_PATH_TOKEN_RE.finditer(path))
    if not tokens:
        return
    for tok in tokens[:-1]:
        parent = parent[tok.group(1)] if tok.group(1) is not None else parent[int(tok.group(2))]
    last = tokens[-1]
    if last.group(1) is not None:
        parent[last.group(1)] = new_value
    else:
        parent[int(last.group(2))] = new_value


# ---------------------------------------------------------------------------
# `show` — resolved-config dump
# ---------------------------------------------------------------------------


_REDIS_URL_RE = re.compile(r"(redis://[^:]+:)([^@]+)(@)")


def _redact_secrets(node: JsonValue) -> JsonValue:
    """Walk a config dump and mask credentials in URL-shaped strings."""
    if isinstance(node, dict):
        return {k: _redact_secrets(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_secrets(item) for item in node]
    if isinstance(node, str):
        return _REDIS_URL_RE.sub(r"\1***\3", node)
    return node


def resolved_config_dict(config_path: Path | None) -> dict[str, JsonValue]:
    """Load *config_path* (or defaults) and return a redaction-safe dump.

    Returned shape is the post-defaults model dump suitable for yaml/json
    rendering — not a typed config object — because the consumer (``rdakt
    show``) is a dumb pretty-printer.
    """
    config = load_config(config_path) if config_path is not None else RdaktConfig()
    raw = config.model_dump(mode="json", exclude_none=True)
    raw.pop("detection_layers", None)
    return _redact_secrets(raw)
