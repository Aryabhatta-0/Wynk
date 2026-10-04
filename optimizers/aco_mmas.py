"""MAX-MIN Ant System - intentional stub (later phase)."""

from __future__ import annotations

from collections.abc import Sequence

from core.genome import Genome
from core.results import EvaluatedRun
from optimizers.base import Optimizer, SearchContext


class MMASACO(Optimizer):
    name = "aco_mmas"

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        raise NotImplementedError("aco_mmas is not implemented yet")

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        raise NotImplementedError("aco_mmas is not implemented yet")
