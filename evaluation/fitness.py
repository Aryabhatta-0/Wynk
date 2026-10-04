"""Shaped search fitness. Pure and deterministic; higher is better; always finite.

Bands (disjoint by construction, so ordering by verdict can never be inverted):

    INFEASIBLE                -1.0
    FAIL      [0.0, 0.8]      0.7 * fraction of fields matched + 0.1 * fraction with valid evidence
    PASS      (1.0, 1.2]      1.0 + cheapness bonus (<= 0.1 for tokens, <= 0.1 for wall time)

The FAIL ceiling (0.8) is below the PASS floor (> 1.0), so a FAIL can never outrank a PASS.
"""

from __future__ import annotations

from typing import Protocol

from core.results import BudgetUsage, FieldResult, Verdict

FITNESS_VERSION = "fitness/mvp-1"

INFEASIBLE_FITNESS = -1.0
_MATCH_WEIGHT = 0.7
_EVIDENCE_WEIGHT = 0.1
_TOKEN_SCALE = 5000.0
_TIME_SCALE = 30.0


class FitnessFunction(Protocol):
    version: str

    def fitness(
        self, verdict: Verdict, field_results: tuple[FieldResult, ...], usage: BudgetUsage
    ) -> float: ...


class ShapedFitness:
    version = FITNESS_VERSION

    def fitness(
        self, verdict: Verdict, field_results: tuple[FieldResult, ...], usage: BudgetUsage
    ) -> float:
        if verdict is Verdict.INFEASIBLE:
            return INFEASIBLE_FITNESS
        if verdict is Verdict.PASS:
            return (
                1.0
                + 0.1 / (1.0 + usage.tokens / _TOKEN_SCALE)
                + 0.1 / (1.0 + usage.wall_time_s / _TIME_SCALE)
            )
        n = len(field_results)
        if n == 0:
            return 0.0
        matched = sum(1 for r in field_results if r.matched) / n
        evidence = sum(1 for r in field_results if r.evidence_valid) / n
        return _MATCH_WEIGHT * matched + _EVIDENCE_WEIGHT * evidence
