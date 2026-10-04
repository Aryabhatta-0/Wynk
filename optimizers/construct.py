"""Shared typed-path construction used by random search and ACO.

A workflow is built stage by stage: the legal next stages come ONLY from the shared
``ConstraintChecker`` (grammar successors, hard-constraint pruning, provable budget); ``END`` is
offered only when the partial genome is a complete, valid workflow. Nothing here re-implements a
grammar or constraint rule. Randomness is derived solely from the search seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from core.canonical import canonical_hash, canonical_json
from core.genome import Genome
from core.stages import StageSpec
from optimizers.base import SearchContext

START = "START"
END = "END"

# (previous node, candidate nodes) -> non-negative selection weights, one per candidate
WeightFn = Callable[[str, Sequence[str]], Sequence[float]]

MAX_STEPS = 16  # safety net only; the grammar is finite (<= 3 verify slots, 1 filter, 1 reason)


def node_key(stage: StageSpec) -> str:
    """Stable identity of a configured stage (a node of the construction graph)."""
    return canonical_json(stage.model_dump(mode="json"))


def path_edges(genome: Genome) -> tuple[tuple[str, str], ...]:
    """The transitions a complete genome traverses: START -> s1 -> ... -> sn -> END."""
    nodes = [START, *(node_key(s) for s in genome.stages), END]
    return tuple(zip(nodes, nodes[1:], strict=False))


def derive_rng(name: str, seed: int, round_: int, call_index: int) -> random.Random:
    digest = canonical_hash([name, seed, round_, call_index])
    return random.Random(int(digest[:16], 16))


def construct_genome(
    context: SearchContext, rng: random.Random, weight_fn: WeightFn
) -> Genome | None:
    """One ant / one random walk. Returns ``None`` on a dead end (no legal continuation)."""
    checker, task = context.checker, context.task
    partial = Genome()
    prev = START
    for _ in range(MAX_STEPS):
        successors = {node_key(s): s for s in checker.admissible_successors(partial, task)}
        options = list(successors)
        if partial.stages and checker.is_valid(partial, task, complete=True):
            options.append(END)
        if not options:
            return None
        weights = weight_fn(prev, options)
        choice = rng.choices(options, weights=weights, k=1)[0]
        if choice == END:
            return partial
        partial = partial.extend(successors[choice])
        prev = choice
    return None
