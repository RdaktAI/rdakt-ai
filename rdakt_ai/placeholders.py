"""Configurable placeholder format for anonymization tokens.

The placeholder format controls how a detected entity is rendered in the
text the LLM sees. The default template ``<{TYPE}_{N}>`` produces tokens
like ``<EMAIL_1>``; users can pick alternatives such as ``<{type}_{n}>``
(``<email_1>``), ``[{TYPE}#{N}]`` (``[EMAIL#1]``), or supply a custom
template, because some models hallucinate around angle-bracket tokens or
have output-format preferences.

A :class:`PlaceholderFormat` is the single source of truth for both
forward (anonymizer) and reverse (deanonymizer / streaming-buffer)
matching, ensuring the two stay consistent within a session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CaseOption = Literal["upper", "lower", "preserve"]

_TYPE_UPPER = "{TYPE}"
_TYPE_LOWER = "{type}"
_N_UPPER = "{N}"
_N_LOWER = "{n}"


@dataclass(frozen=True)
class _Parsed:
    prefix: str
    middle: str
    suffix: str
    type_is_upper: bool


def _parse_template(template: str) -> _Parsed:
    upper_count = template.count(_TYPE_UPPER)
    lower_count = template.count(_TYPE_LOWER)
    if upper_count + lower_count != 1:
        raise ValueError(f"Template {template!r} must contain exactly one of {{TYPE}} or {{type}}")
    type_ph = _TYPE_UPPER if upper_count == 1 else _TYPE_LOWER
    type_is_upper = upper_count == 1

    n_upper = template.count(_N_UPPER)
    n_lower = template.count(_N_LOWER)
    if n_upper + n_lower != 1:
        raise ValueError(f"Template {template!r} must contain exactly one of {{N}} or {{n}}")
    n_ph = _N_UPPER if n_upper == 1 else _N_LOWER

    type_idx = template.index(type_ph)
    n_idx = template.index(n_ph)
    if type_idx > n_idx:
        raise ValueError(f"Template {template!r} must place {{TYPE}}/{{type}} before {{N}}/{{n}}")

    prefix = template[:type_idx]
    middle = template[type_idx + len(type_ph) : n_idx]
    suffix = template[n_idx + len(n_ph) :]

    if not prefix:
        raise ValueError(
            f"Template {template!r} must have a non-empty literal prefix before "
            "{TYPE}/{type} (else partial-token detection during streaming has "
            "no anchor)"
        )
    if not middle:
        raise ValueError(
            f"Template {template!r} must have a non-empty literal separator between "
            "{TYPE}/{type} and {N}/{n} (else tokens like EMAIL1 / EMAIL11 are ambiguous)"
        )
    if not suffix:
        raise ValueError(
            f"Template {template!r} must have a non-empty literal suffix after "
            "{N}/{n} (else the counter has no terminator and EMAIL_1 / EMAIL_11 "
            "would be ambiguous)"
        )
    return _Parsed(prefix=prefix, middle=middle, suffix=suffix, type_is_upper=type_is_upper)


def _proper_prefixes(s: str) -> list[str]:
    """All non-empty proper prefixes of *s*."""
    return [s[:i] for i in range(1, len(s))]


class PlaceholderFormat:
    """Renders and parses placeholder tokens using a template.

    The template uses ``{TYPE}`` / ``{type}`` for the entity-type slot and
    ``{N}`` / ``{n}`` for the numeric counter. ``case`` overrides the case
    chosen by the slot variant (``preserve`` keeps the slot's case).
    """

    DEFAULT_TEMPLATE = "<{TYPE}_{N}>"

    def __init__(
        self,
        template: str = DEFAULT_TEMPLATE,
        case: CaseOption = "preserve",
    ) -> None:
        if case not in ("upper", "lower", "preserve"):
            raise ValueError(f"Invalid placeholder case {case!r}; must be 'upper', 'lower', or 'preserve'")
        self._template = template
        self._case = case
        self._parsed = _parse_template(template)
        self._upper = True if case == "upper" else False if case == "lower" else self._parsed.type_is_upper
        self._full_re, self._partial_re = self._build_regexes()

    @property
    def template(self) -> str:
        return self._template

    @property
    def case(self) -> CaseOption:
        return self._case

    @property
    def is_canonical(self) -> bool:
        """True if this is the default ``<{TYPE}_{N}>`` upper-case template.

        Canonical mode preserves backslash-escape support for providers that
        emit JSON-encoded streams with ``\\<`` / ``\\>`` escaping.
        """
        return self._template == self.DEFAULT_TEMPLATE and self._upper

    @property
    def full_re(self) -> re.Pattern[str]:
        """Matches a complete placeholder anywhere in text.

        Captures: group(1) = entity type as rendered, group(2) = counter digits.
        """
        return self._full_re

    @property
    def partial_re(self) -> re.Pattern[str]:
        """Matches a trailing in-progress placeholder at end-of-string.

        Used by the streaming buffer to know when to keep buffering until the
        next chunk arrives.
        """
        return self._partial_re

    def format(self, entity_type: str, n: int) -> str:
        t = entity_type.upper() if self._upper else entity_type.lower()
        p = self._parsed
        return f"{p.prefix}{t}{p.middle}{n}{p.suffix}"

    def parse(self, token: str) -> tuple[str, int] | None:
        """Parse a fully-formed token into (entity_type, n).

        Always returns the entity type in upper-case (canonical form) so that
        callers keying counters by ``entity.type`` get consistent lookups
        regardless of the rendered case.
        """
        m = self._full_re.fullmatch(token)
        if not m:
            return None
        return m.group(1).upper(), int(m.group(2))

    # ---- internals ----

    def _type_charclass(self) -> str:
        return r"[A-Z][A-Z0-9_]*" if self._upper else r"[a-z][a-z0-9_]*"

    def _build_regexes(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        p = self._parsed
        prefix_e = re.escape(p.prefix)
        middle_e = re.escape(p.middle)
        suffix_e = re.escape(p.suffix)
        type_pat = self._type_charclass()
        n_pat = r"\d+"

        if self.is_canonical:
            # Preserve the original escape-aware patterns so providers that
            # JSON-escape angle brackets continue to work.
            full = re.compile(rf"\\?<({type_pat})_({n_pat})\\?>")
            partial = re.compile(r"(?:\\?<(?:[A-Z][A-Z0-9_]*\\?)?|\\)\Z")
            return full, partial

        full = re.compile(rf"{prefix_e}({type_pat}){middle_e}({n_pat}){suffix_e}")

        parts: list[str] = []
        # 1. partial within prefix (only if prefix is multi-char)
        for pp in _proper_prefixes(p.prefix):
            parts.append(re.escape(pp))
        # 2. prefix complete + optional partial type chars
        parts.append(rf"{prefix_e}(?:{type_pat})?")
        # 3. prefix + type + partial of multi-char middle
        for mp in _proper_prefixes(p.middle):
            parts.append(rf"{prefix_e}{type_pat}{re.escape(mp)}")
        # 4. prefix + type + middle + optional partial digits
        parts.append(rf"{prefix_e}{type_pat}{middle_e}(?:{n_pat})?")
        # 5. prefix + type + middle + digits + partial of multi-char suffix
        for sp in _proper_prefixes(p.suffix):
            parts.append(rf"{prefix_e}{type_pat}{middle_e}{n_pat}{re.escape(sp)}")

        partial = re.compile(r"(?:" + "|".join(parts) + r")\Z")
        return full, partial


CANONICAL = PlaceholderFormat()
"""Module-level singleton for the default ``<{TYPE}_{N}>`` template."""
