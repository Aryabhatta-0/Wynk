"""The one interface every optimizer implements (ACO, random, halving, bandit, PBIL, ...).

Authority: an optimizer only decides HOW to execute - it proposes genomes. It never runs a
workflow, never calls a model, never calls the evaluator, and never sees ground truth:
  * input task view is ``RuntimeTask`` (no ground truth),
  * feedback arrives as ``EvaluatedRun`` objects produced elsewhere,
  * legality comes from the shared ``ConstraintChecker`` - not re-implemented here.
This base module deliberately contains no algorithm-specific (e.g. pheromone) concepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.results import EvaluatedRun
from core.task_spec import RuntimeTask


@dataclass(frozen=True)
class SearchContext:
    task: RuntimeTask
    checker: ConstraintChecker
    seed: int  # all optimizer randomness must derive from this -> reproducible searches
    round: int = 0


class InadmissibleProposal(ValueError):
    pass


class Optimizer(ABC):
    name: str

    @abstractmethod
    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        """Return up to ``k`` COMPLETE genomes that satisfy ``context.checker`` for the task."""

    @abstractmethod
    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        """Consume evaluated runs (verdict + fitness already decided by the evaluator)."""


def ensure_admissible(genomes: Sequence[Genome], context: SearchContext) -> None:
    """Guard for optimizer implementations/tests: every proposal must pass the shared checker."""
    for g in genomes:
        violations = context.checker.check(g, context.task, complete=True)
        if violations:
            raise InadmissibleProposal(f"{g.genome_hash[:12]}: {violations[0].message}")
