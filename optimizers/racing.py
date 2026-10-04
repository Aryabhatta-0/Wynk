"""Racing - intentional stub (later phase)."""

from __future__ import annotations

from collections.abc import Sequence

from core.genome import Genome
from core.results import EvaluatedRun
from optimizers.base import Optimizer, SearchContext


class Racing(Optimizer):
    name = "racing"

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        raise NotImplementedError("racing is not implemented yet")

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        raise NotImplementedError("racing is not implemented yet")
