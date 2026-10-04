"""Deterministic per-field matchers (implementation: Track A)."""

from __future__ import annotations

from typing import Any, Protocol

from core.task_spec import MatcherConfig


class Matcher(Protocol):
    def matches(self, expected: Any, actual: Any, config: MatcherConfig) -> bool: ...
