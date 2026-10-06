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


class DistinctRandomSearch(RandomSearch):
    """Random search WITHOUT replacement: never proposes a workflow it already proposed until
    every admissible workflow has been proposed.

    Proposals come from the same uniform random walk as ``RandomSearch`` (so it is exactly MMAS
    ACO's construction with flat pheromone, i.e. ACO without learning); a walk that lands on an
    already-proposed workflow is rejected. If ``ATTEMPTS_PER_PROPOSAL`` walks in a row are all
    repeats, a depth-first search over the admissible space - successors shuffled by the same
    seeded rng - returns an unproposed workflow, or proves the space exhausted. Only then may a
    repeat be proposed (the harness gives a repeat fresh trials). Everything derives from
    ``context.seed``: same seed, same proposal order.
    """

    name = "random_search_distinct"
    version = "random_distinct/1"

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()
        self.exhausted = False

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        rng = derive_rng(self.name, context.seed, context.round, self._calls)
        self._calls += 1
        out: list[Genome] = []
        while len(out) < k:
            g = None if self.exhausted else self._unseen(context, rng)
            if g is None:
                self.exhausted = True
                g = construct_genome(context, rng, lambda _prev, options: [1.0] * len(options))
                if g is None:
                    break
            self._seen.add(g.genome_hash)
            out.append(g)
        ensure_admissible(out, context)
        return out

    def _unseen(self, context: SearchContext, rng) -> Genome | None:
        for _ in range(ATTEMPTS_PER_PROPOSAL):
            g = construct_genome(context, rng, lambda _prev, options: [1.0] * len(options))
            if g is not None and g.genome_hash not in self._seen:
                return g
        return self._first_unseen(context, rng)

    def _first_unseen(self, context: SearchContext, rng) -> Genome | None:
        checker, contract = context.checker, context.contract

        def dfs(partial: Genome) -> Genome | None:
            if (
                partial.stages
                and partial.genome_hash not in self._seen
                and checker.is_valid(partial, contract, complete=True)
            ):
                return partial
            successors = list(checker.admissible_successors(partial, contract))
            rng.shuffle(successors)
            for stage in successors:
                found = dfs(partial.extend(stage))
                if found is not None:
                    return found
            return None

        return dfs(Genome())
