"""Deterministic per-field matchers. Pure functions; no LLM, no I/O."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any, Protocol

from core.task_spec import MatcherConfig, MatcherKind


class Matcher(Protocol):
    def matches(self, expected: Any, actual: Any, config: MatcherConfig) -> bool: ...


def normalize_text(value: str) -> str:
    """NFKC, casefold, collapse whitespace, strip surrounding punctuation."""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,;:!?\"'")


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def parse_date(value: Any) -> date | None:
    """ISO ``YYYY-MM-DD`` (a full ISO datetime is reduced to its date). Anything else -> None."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.strip()).date()
    except ValueError:
        return None


class DefaultMatcher:
    """Type-safe: a wrong-typed ``actual`` is simply a non-match (never an exception)."""

    def matches(self, expected: Any, actual: Any, config: MatcherConfig) -> bool:
        kind = config.kind
        if kind is MatcherKind.EXACT:
            return type(expected) is type(actual) and expected == actual
        if kind is MatcherKind.NORMALIZED_TEXT:
            return (
                isinstance(expected, str)
                and isinstance(actual, str)
                and normalize_text(expected) == normalize_text(actual)
            )
        if kind is MatcherKind.NUMERIC_TOLERANCE:
            if not (_is_number(expected) and _is_number(actual)):
                return False
            return math.isclose(
                expected, actual, rel_tol=config.rel_tol or 0.0, abs_tol=config.abs_tol or 0.0
            )
        if kind is MatcherKind.DATE:
            want, got = parse_date(expected), parse_date(actual)
            return want is not None and want == got
        if kind is MatcherKind.SET_EQUAL:
            if not (isinstance(expected, list | tuple) and isinstance(actual, list | tuple)):
                return False
            if not all(isinstance(x, str) for x in (*expected, *actual)):
                return False
            return {normalize_text(x) for x in expected} == {normalize_text(x) for x in actual}
        raise ValueError(f"unknown matcher kind {kind!r}")
