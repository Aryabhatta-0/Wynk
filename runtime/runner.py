"""Public runtime entry point: ``WorkflowRunner.run(genome, task) -> ExecutionResult``.

    Task -> Genome -> compiler (pure DAG) -> MAF workflow -> executors -> ExecutionResult

The runner takes only a ``RuntimeTask`` (no ground truth) and never produces a verdict.
Structurally invalid genomes raise ``InadmissibleGenome`` (the caller should have used the
shared ``ConstraintChecker``). A genome whose best case provably exceeds the caps is not
executed: it returns an ``ExecutionResult`` with a ``BUDGET_EXCEEDED`` failure and zero usage,
which the evaluator maps to INFEASIBLE.
"""

from __future__ import annotations

import asyncio

from compiler.maf_compiler import MAFCompiler
from core.constraints import ConstraintChecker
from core.cost_model import exceeded_caps
from core.genome import Genome
from core.results import (
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
    RunVersions,
)
from core.task_spec import RuntimeTask
from core.violations import Violation, ViolationCode
from runtime.budget_guard import BudgetGuard
from runtime.executors.base import RunContext
from runtime.executors.registry import default_executors
from runtime.gemma_client import ModelClient
from runtime.maf_nodes import Envelope, StageNode
from runtime.prompts import PROMPT_TEMPLATE_VERSION
from runtime.sources import ApiSource, PageSource
from runtime.stage_runner import StageRunner


class InadmissibleGenome(ValueError):
    def __init__(self, violations: tuple[Violation, ...]) -> None:
        super().__init__("; ".join(f"{v.code.value}: {v.message}" for v in violations))
        self.violations = violations


class WorkflowRunner:
    def __init__(
        self,
        *,
        model: ModelClient | None,
        benchmark_hash: str,
        pages: PageSource | None = None,
        api: ApiSource | None = None,
        compiler: MAFCompiler | None = None,
        checker: ConstraintChecker | None = None,
    ) -> None:
        self.model = model
        self.benchmark_hash = benchmark_hash
        self.compiler = compiler or MAFCompiler()
        self.checker = checker or ConstraintChecker()
        self._executors = default_executors(pages=pages, api=api)

    def versions(self) -> RunVersions:
        return RunVersions(
            model_hash=self.model.model_hash if self.model else "no-model",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            benchmark_hash=self.benchmark_hash,
            compiler_version=self.compiler.version,
            grammar_version=self.checker.grammar.version,
        )

    async def run(
        self, genome: Genome, task: RuntimeTask, *, trial: int = 0, seed: int = 0
    ) -> ExecutionResult:
        key = RunKey(
            genome_hash=genome.genome_hash,
            task_id=task.id,
            trial=trial,
            seed=seed,
            versions=self.versions(),
        )
        violations = self.checker.check(genome, task, complete=True)
        structural = tuple(v for v in violations if v.code != ViolationCode.BUDGET_INFEASIBLE)
        if structural:
            raise InadmissibleGenome(structural)
        if violations:  # only BUDGET_INFEASIBLE: provably cannot fit the caps, do not execute
            return self._static_breach(key, genome, task)

        dag = self.compiler.to_dag(genome)
        ctx = RunContext(
            task=task,
            seed=seed,
            trial=trial,
            versions=key.versions,
            guard=BudgetGuard(task.caps),
            model=self.model,
        )
        runner = StageRunner(dag, self._executors, ctx)
        nodes = {
            n.node_id: StageNode(n, runner, is_end=n.node_id == dag.end_node) for n in dag.nodes
        }
        workflow = self.compiler.build(dag, nodes)
        await workflow.run(Envelope(task))
        return runner.result(key)

    def run_sync(
        self, genome: Genome, task: RuntimeTask, *, trial: int = 0, seed: int = 0
    ) -> ExecutionResult:
        return asyncio.run(self.run(genome, task, trial=trial, seed=seed))

    def _static_breach(self, key: RunKey, genome: Genome, task: RuntimeTask) -> ExecutionResult:
        caps = exceeded_caps(self.checker.cost_model.estimate(genome, task), task.caps)
        return ExecutionResult(
            key=key,
            failure=FailureInfo(
                kind=FailureKind.BUDGET_EXCEEDED,
                message="best-case estimate exceeds caps (not executed): "
                + ", ".join(c.value for c in caps),
                cap=caps[0] if caps else None,
            ),
        )
