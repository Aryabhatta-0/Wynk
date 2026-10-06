"""Fixed workflow baselines: one deterministic workflow, chosen from the contract alone.

Two pre-registered selection rules exist. Both are pure functions of (contract, checker): they
read no dataset row, no target, no evaluation and no score, so neither can be tuned on
validation or test performance.

``fixed_shortest/1`` (``FixedBaseline``, MuSiQue protocol-v1)
    the SHORTEST complete workflow the shared ``ConstraintChecker`` admits for the contract;
    ties broken by the grammar's canonical successor order (``StageKind`` order, then each kind's
    configuration enum order) - i.e. the first such workflow ``enumerate_admissible`` would yield.
    With DIRECT in the vocabulary this is ``DIRECT(answer)``, which never sees context columns:
    on a context-dependent dataset it is not a meaningful comparator (protocol-v1 recorded this).

``fixed_context/1`` (``ContextAwareBaseline``, protocol-v2)
    decided by the dataset SHAPE only:
      * the contract's dataset declares context columns
            -> ``GATHER(fetch, sequential) -> EXTRACT(direct) -> SYNTHESIZE(direct)``
      * it declares none
            -> ``DIRECT(answer)``
    No VERIFY, REASON, FILTER or CONFIDENCE_GATE. If the chosen workflow is not admissible for
    the contract, it fails closed (``ContractError``) - it never falls back to anything else.

As an ``Optimizer`` a baseline proposes its one workflow once and then nothing, so the experiment
runner drives it through exactly the same loop, ledger and split gates as random search and ACO.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.grammar import MAX_GENOME_STAGES
from core.results import EvaluatedRun
from core.stages import DirectStage, ExtractStage, GatherStage, SynthesizeStage
from core.task_contract import ContractError, TaskContract
from optimizers.base import Optimizer, SearchContext, ensure_admissible

RETRIEVAL_BASELINE = Genome.of(
    GatherStage(source="fetch", mode="sequential"),
    ExtractStage(method="direct"),
    SynthesizeStage(method="direct"),
)
DIRECT_BASELINE = Genome.of(DirectStage(method="answer"))


def baseline_workflow(contract: TaskContract, checker: ConstraintChecker) -> Genome:
    """``fixed_shortest/1``: the shortest admissible workflow (see the module docstring).

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


def context_aware_workflow(contract: TaskContract, checker: ConstraintChecker) -> Genome:
    """``fixed_context/1``: retrieval chain iff the dataset has context columns, else DIRECT."""
    genome = RETRIEVAL_BASELINE if contract.dataset.context_columns else DIRECT_BASELINE
    violations = checker.check(genome, contract, complete=True)
    if violations:
        raise ContractError(
            f"fixed_context/1 baseline is not admissible for {contract.task_id}: "
            f"{violations[0].message}"
        )
    return genome


class FixedBaseline(Optimizer):
    """Proposes the contract's ``fixed_shortest/1`` workflow once; never learns."""

    name = "fixed_baseline"
    version = "fixed_shortest/1"  # bump if the selection rule changes

    def __init__(self) -> None:
        self._proposed = False
        self.n_observed = 0

    def _workflow(self, context: SearchContext) -> Genome:
        return baseline_workflow(context.contract, context.checker)

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        if self._proposed or k < 1:
            return []
        self._proposed = True
        out = [self._workflow(context)]
        ensure_admissible(out, context)
        return out

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        self.n_observed += len(results)  # a fixed workflow does not learn


class ContextAwareBaseline(FixedBaseline):
    """Proposes the contract's ``fixed_context/1`` workflow once; never learns."""

    name = "fixed_context_baseline"
    version = "fixed_context/1"

    def _workflow(self, context: SearchContext) -> Genome:
        return context_aware_workflow(context.contract, context.checker)


FIXED_RULES: dict[str, type[FixedBaseline]] = {
    FixedBaseline.version: FixedBaseline,
    ContextAwareBaseline.version: ContextAwareBaseline,
}
