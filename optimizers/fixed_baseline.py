"""Fixed workflow baseline: one deterministic workflow, chosen from the contract grammar alone.

Selection rule (``fixed_shortest/1``), a pure function of (contract, checker):

    the SHORTEST complete workflow the shared ``ConstraintChecker`` admits for the contract;
    ties broken by the grammar's canonical successor order (``StageKind`` order, then each kind's
    configuration enum order) - i.e. the first such workflow ``enumerate_admissible`` would yield.

It reads no dataset row, no evaluation and no score, so it can never be tuned on validation or
test performance. With DIRECT in the contract's vocabulary this is ``DIRECT(answer)`` - "just ask
the model" - otherwise the plainest retrieval chain (``GATHER(fetch) -> EXTRACT -> SYNTHESIZE``).

As an ``Optimizer`` it proposes that one workflow once and then nothing, so the experiment runner
drives it through exactly the same loop, ledger and split gates as random search and ACO.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.grammar import MAX_GENOME_STAGES
from core.results import EvaluatedRun
from core.task_contract import ContractError, TaskContract
from optimizers.base import Optimizer, SearchContext, ensure_admissible


def baseline_workflow(contract: TaskContract, checker: ConstraintChecker) -> Genome:
    """The fixed baseline of ``contract`` (see the module docstring for the rule).

    Breadth first over admissible prefixes, in successor order, so the first complete workflow
    found at the smallest depth is exactly the first of that length in enumeration order.
    """
    frontier = [Genome()]
    for _ in range(MAX_GENOME_STAGES):
        frontier = [
            prefix.extend(stage)
            for prefix in frontier
            for stage in checker.admissible_successors(prefix, contract)
        ]
        for genome in frontier:
            if checker.is_valid(genome, contract, complete=True):
                return genome
        if not frontier:
            break
    raise ContractError(f"contract {contract.task_id} admits no complete workflow")


class FixedBaseline(Optimizer):
    """Proposes the contract's baseline workflow once; never learns."""

    name = "fixed_baseline"
    version = "fixed_shortest/1"  # bump if the selection rule changes

    def __init__(self) -> None:
        self._proposed = False
        self.n_observed = 0

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        if self._proposed or k < 1:
            return []
        self._proposed = True
        out = [baseline_workflow(context.contract, context.checker)]
        ensure_admissible(out, context)
        return out

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        self.n_observed += len(results)  # a fixed workflow does not learn
