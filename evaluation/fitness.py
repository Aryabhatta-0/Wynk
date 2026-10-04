"""Shaped search fitness (implementation: Track A). Pure and deterministic."""

from __future__ import annotations

from typing import Protocol

from core.results import BudgetUsage, FieldResult, Verdict


class FitnessFunction(Protocol):
    version: str

    def fitness(
        self, verdict: Verdict, field_results: tuple[FieldResult, ...], usage: BudgetUsage
    ) -> float: ...
