"""Configuration loading from rdakt.yaml."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rdakt_ai.placeholders import PlaceholderFormat

if TYPE_CHECKING:
    from rdakt_ai.stores import SessionStore

logger = logging.getLogger("rdakt_ai.config")

VALID_MODES = frozenset({"active", "audit"})
VALID_ERROR_POLICIES = frozenset({"warn_and_forward", "block"})
VALID_PIPELINE_DETECTORS = frozenset({"regex", "ner"})
VALID_STRATEGIES = frozenset({"token", "synthetic", "hybrid"})


class EntityConfig(BaseModel):
    """Configuration for a single entity type."""

    strategy: Literal["token", "synthetic", "hybrid"] | None = None
    pattern: str | None = None


class PlaceholderConfig(BaseModel):
    """Placeholder rendering format.

    ``template`` controls the literal shape of the token the LLM sees, using
    ``{TYPE}`` / ``{type}`` for the entity-type slot and ``{N}`` / ``{n}``
    for the numeric counter. ``case`` overrides the case implied by the
    slot variant; ``preserve`` keeps it.
    """

    model_config = ConfigDict(extra="forbid")

    template: str = PlaceholderFormat.DEFAULT_TEMPLATE
    case: Literal["upper", "lower", "preserve"] = "preserve"

    @model_validator(mode="after")
    def _validate_template(self) -> PlaceholderConfig:
        # Constructing a PlaceholderFormat runs the same validation that will
        # be applied at runtime, so an invalid template fails at config load.
        PlaceholderFormat(self.template, self.case)
        return self

    def build(self) -> PlaceholderFormat:
        return PlaceholderFormat(self.template, self.case)


class SyntheticReplacement(BaseModel):
    """Spec for replacing a field's entire value with a generated synthetic.

    Used for opaque internal IDs that should never reach the LLM (e.g.
    ``user_id``). ``format`` is a Python format string with a single
    ``{n}`` slot that's filled with a per-(session, type) counter.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    format: str

    @field_validator("format")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        try:
            v.format(n=1)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"replace_with_synthetic.format {v!r} is invalid: must contain a single '{{n}}' slot ({exc})"
            ) from exc
        return v


class FieldRule(BaseModel):
    """One ontology rule keyed by a JSONPath.

    Exactly one of ``detect``, ``detect_via_regex``, or
    ``replace_with_synthetic`` declares the detection source.
    ``mask_strategy`` and ``preserve_components`` are optional modifiers.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    detect: list[str] | None = None
    detect_via_regex: str | None = None
    detect_via_regex_as: str | None = None
    replace_with_synthetic: SyntheticReplacement | None = None
    mask_strategy: Literal["token", "synthetic", "hybrid", "hash"] | None = None
    preserve_components: list[str] | None = None

    @model_validator(mode="after")
    def _validate(self) -> FieldRule:
        from rdakt_ai.ontology import parse_jsonpath

        try:
            parse_jsonpath(self.path)
        except ValueError as exc:
            raise ValueError(f"Invalid path in ontology rule: {exc}") from exc

        sources = [
            self.detect is not None,
            self.detect_via_regex is not None,
            self.replace_with_synthetic is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                f"Ontology rule for {self.path!r} must declare exactly one of "
                "'detect', 'detect_via_regex', or 'replace_with_synthetic'"
            )

        if self.detect_via_regex is not None:
            try:
                re.compile(self.detect_via_regex)
            except re.error as exc:
                raise ValueError(f"Invalid regex in ontology rule for {self.path!r}: {exc}") from exc

        if self.preserve_components and self.detect is None:
            raise ValueError(f"preserve_components is only valid alongside 'detect' (rule for {self.path!r})")

        return self


class OntologyConfig(BaseModel):
    """Project-level ontology: a list of :class:`FieldRule`s applied
    before the global detection pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[FieldRule] = Field(default_factory=list)


class SessionConfig(BaseModel):
    """Session store configuration.

    The ``store`` field selects the backend. Any additional fields
    (``url``, ``ttl``, ``prefix``, ``path``) are passed as keyword
    arguments to the store constructor.
    """

    model_config = ConfigDict(extra="allow")

    store: Literal["memory", "sqlite", "redis"] = "memory"

    @property
    def options(self) -> dict[str, Any]:
        """Extra fields passed to the store constructor."""
        return dict(self.model_extra) if self.model_extra else {}


class RdaktConfig(BaseModel):
    """Rdakt AI configuration.

    Can be constructed programmatically or loaded from YAML via :func:`load_config`.

    Example — programmatic::

        config = RdaktConfig(
            mode="active",
            pipeline=["regex", {"ner": {"model": "en_core_web_sm"}}],
            entities={"EMAIL": EntityConfig(strategy="token")},
            session=SessionConfig(store="redis", url="redis://localhost:6379"),
        )

    Example — YAML::

        mode: active
        pipeline:
          - regex
          - ner:
              model: en_core_web_sm
        entities:
          EMAIL:
            strategy: token
        session:
          store: redis
          url: redis://localhost:6379
    """

    model_config = ConfigDict(validate_default=True)

    mode: Literal["active", "audit"] = "active"
    on_error: Literal["warn_and_forward", "block"] = "warn_and_forward"
    pipeline: list[str | dict[str, Any]] = Field(default=["regex"])
    entities: dict[str, EntityConfig] = Field(default_factory=dict)
    session: SessionConfig = Field(default_factory=SessionConfig)
    placeholders: PlaceholderConfig = Field(default_factory=PlaceholderConfig)
    ontology: OntologyConfig | None = None

    # Deprecated — kept for backwards compatibility with old configs
    detection_layers: list[str] = Field(default_factory=lambda: ["regex"], exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_fields(cls, data: Any) -> Any:
        """Support old flat-style constructor args alongside new nested style.

        Transforms:
        - ``session_store="redis", store_options={...}`` → ``session=SessionConfig(...)``
        - ``entity_strategies={"EMAIL": "token"}`` → ``entities={"EMAIL": {"strategy": "token"}}``
        - ``custom_patterns={"X": "..."}`` → ``entities={"X": {"pattern": "..."}}``
        """
        if not isinstance(data, dict):
            return data

        # session_store / store_options → session
        if "session_store" in data or "store_options" in data:
            session = dict(data.get("session", {})) if isinstance(data.get("session"), dict) else {}
            if "session_store" in data:
                session["store"] = data.pop("session_store")
            if "store_options" in data:
                session.update(data.pop("store_options"))
            data["session"] = session

        # entity_strategies → entities
        if "entity_strategies" in data:
            entities: dict[str, Any] = dict(data.get("entities", {}))
            for name, strategy in data.pop("entity_strategies").items():
                entities.setdefault(name, {})
                if isinstance(entities[name], dict):
                    entities[name]["strategy"] = strategy
            data["entities"] = entities

        # custom_patterns → entities
        if "custom_patterns" in data:
            entities = dict(data.get("entities", {}))
            for name, pattern in data.pop("custom_patterns").items():
                entities.setdefault(name, {})
                if isinstance(entities[name], dict):
                    entities[name]["pattern"] = pattern
            data["entities"] = entities

        return data

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline_entries(cls, v: list[str | dict[str, Any]]) -> list[str | dict[str, Any]]:
        for entry in v:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict):
                name = next(iter(entry))
            else:
                raise ValueError(f"Invalid pipeline entry {entry!r}: must be str or dict")
            if name not in VALID_PIPELINE_DETECTORS:
                raise ValueError(f"Unknown pipeline detector {name!r}. Supported: {sorted(VALID_PIPELINE_DETECTORS)}")
        return v

    # ---- Backwards-compatible computed properties ----

    @property
    def entity_strategies(self) -> dict[str, str]:
        """Map of entity type → strategy name (for entities with a strategy set)."""
        return {name: cfg.strategy for name, cfg in self.entities.items() if cfg.strategy}

    @property
    def custom_patterns(self) -> dict[str, str]:
        """Map of entity type → regex pattern (for entities with a custom pattern)."""
        return {name: cfg.pattern for name, cfg in self.entities.items() if cfg.pattern}

    @property
    def session_store(self) -> str:
        """Session store backend name."""
        return self.session.store

    @property
    def store_options(self) -> dict[str, Any]:
        """Extra options passed to the session store constructor."""
        return self.session.options


def load_config(path: Path) -> RdaktConfig:
    """Load configuration from a YAML file.

    Returns defaults if file not found. Handles backwards-compatible
    ``detection.layers`` key (deprecated in favor of ``pipeline``).
    """
    if not path.exists():
        logger.debug("Config file %s not found, using defaults", path)
        return RdaktConfig()

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Backwards compat: detection.layers → pipeline
    if "pipeline" not in raw and "detection" in raw:
        warnings.warn(
            "Config key 'detection.layers' is deprecated. Use 'pipeline' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        layers = raw["detection"].get("layers", ["regex"])
        raw["pipeline"] = layers

    # Normalise: older flat entity format → EntityConfig dicts
    if "entities" in raw:
        for name, settings in raw["entities"].items():
            if isinstance(settings, str):
                # Allow shorthand: `EMAIL: token`
                raw["entities"][name] = {"strategy": settings}

    try:
        return RdaktConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def create_store(config: RdaktConfig) -> SessionStore:
    """Create a session store from configuration.

    Supported backends: ``memory``, ``sqlite``, ``redis``.
    """
    from rdakt_ai.stores import MemoryStore, RedisStore, SQLiteStore

    backend = config.session_store
    opts = config.store_options

    if backend == "memory":
        return MemoryStore()
    if backend == "sqlite":
        return SQLiteStore(**opts)
    if backend == "redis":
        return RedisStore(**opts)
    raise ValueError(f"Unknown session store backend {backend!r}. Supported: memory, sqlite, redis")
