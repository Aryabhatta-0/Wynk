"""Shaped search fitness. Pure and deterministic; higher is better; always finite.

Bands (disjoint by construction, so ordering by verdict can never be inverted):

    INFEASIBLE                -1.0
    FAIL      [0.0, 0.7]      0.7 * fraction of fields matched + 0.1 * fraction with valid evidence
    PASS      [1.0, 1.1]      1.0 + budget-headroom bonus (<= 0.1, vs the task's OWN token cap)

The FAIL ceiling is 0.7, not 0.8: ``matched = 1.0`` together with ``evidence = 1.0`` *is* the
PASS condition (see ``DeterministicEvaluator.evaluate``), so a FAIL can never earn the evidence
term on every field. 0.7 is the highest FAIL actually reachable.

Cost is deliberately ABSENT from the FAIL band. Budgets are hard constraints decided in code
(``usage_exceeds`` / ``ConstraintChecker``), so a run that fits its caps is already "cheap enough";
mixing a soft cost term in below PASS would let a cheap-but-hopeless FAIL outrank a nearly-correct
one and actively mislead the search. Cost only ever breaks ties *among* PASSes.

The headroom bonus is measured against ``caps.tokens`` - the cap of the task being run - not
against a fixed constant, so "cheap" means the same thing on a 2k-token task and a 20k-token one.
Wall-clock time is NOT scored: it is already a hard cap, and as a soft term it is pure
infrastructure noise (network, model-server load) feeding straight into the pheromone deposit.
"""

from __future__ import annotations

from typing import Protocol

from core.results import BudgetUsage, FieldResult, Verdict
from core.task_spec import Caps

FITNESS_VERSION = "fitness/mvp-2"

INFEASIBLE_FITNESS = -1.0
_MATCH_WEIGHT = 0.7
_EVIDENCE_WEIGHT = 0.1
_HEADROOM_WEIGHT = 0.1


class FitnessFunction(Protocol):
    version: str

    def fitness(
        self,
        verdict: Verdict,
        field_results: tuple[FieldResult, ...],
        usage: BudgetUsage,
        caps: Caps,
    ) -> float: ...


class ShapedFitness:
    version = FITNESS_VERSION

    def fitness(
        self,
        verdict: Verdict,
        field_results: tuple[FieldResult, ...],
        usage: BudgetUsage,
        caps: Caps,
    ) -> float:
        if verdict is Verdict.INFEASIBLE:
            return INFEASIBLE_FITNESS
        if verdict is Verdict.PASS:
            headroom = max(0.0, 1.0 - usage.tokens / caps.tokens)
            return 1.0 + _HEADROOM_WEIGHT * headroom
        n = len(field_results)
        if n == 0:
            return 0.0
        matched = sum(1 for r in field_results if r.matched) / n
        evidence = sum(1 for r in field_results if r.evidence_valid) / n
        return _MATCH_WEIGHT * matched + _EVIDENCE_WEIGHT * evidence
