"""Random search baseline: uniform choice over the legal next steps (incl. END when complete)."""

from __future__ import annotations

from collections.abc import Sequence

from core.genome import Genome
from core.results import EvaluatedRun
from optimizers.base import Optimizer, SearchContext, ensure_admissible
from optimizers.construct import construct_genome, derive_rng

ATTEMPTS_PER_PROPOSAL = 20


class RandomSearch(Optimizer):
    name = "random_search"

    def __init__(self) -> None:
        self._calls = 0
        self.n_observed = 0

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        rng = derive_rng(self.name, context.seed, context.round, self._calls)
        self._calls += 1
        out: list[Genome] = []
        for _ in range(k * ATTEMPTS_PER_PROPOSAL):
            if len(out) == k:
                break
            g = construct_genome(context, rng, lambda _prev, options: [1.0] * len(options))
            if g is not None:
                out.append(g)
        ensure_admissible(out, context)
        return out

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        self.n_observed += len(results)  # random search does not learn
