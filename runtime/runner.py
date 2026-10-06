"""Public runtime entry point: ``WorkflowRunner.run(genome, task) -> ExecutionResult``.

    ExecutionTask -> Genome -> compiler (pure DAG) -> MAF workflow -> executors -> ExecutionResult

The runner takes only an ``ExecutionTask`` (a ``TaskContract`` + one row's inputs; no target
values) and never produces a verdict. The contract decides everything the run may do: admission
(``ConstraintChecker.check(genome, task.contract)``), the per-run caps of the budget guard, the
data source GATHER reads, and the run identity (``RunKey.contract_hash``). Structurally invalid
genomes raise ``InadmissibleGenome`` (the caller should have used the shared
``ConstraintChecker``). A genome whose best case provably exceeds the caps is not executed: it
returns an ``ExecutionResult`` with a ``BUDGET_EXCEEDED`` failure and zero usage, which the
evaluator maps to INFEASIBLE.

Models. The runner depends only on the provider-neutral ``ModelClient``. With a declared
``AllowedModels`` set (``core.models``) it refuses - before compiling or invoking anything - a
client outside that set, a client bound to a different registry entry than the allowed one, and
a workflow whose model stages the entry provably cannot serve (no text generation, or a context
window smaller than one stage call's output budget). The run's ``RunVersions.model_hash`` is the
client's, which a ``RegisteredModelClient`` takes from its pinned registry entry.
"""

from __future__ import annotations

import asyncio

from compiler.maf_compiler import MAFCompiler
from core.constraints import ConstraintChecker, ConstraintConfig
from core.cost_model import exceeded_caps
from core.genome import Genome
from core.models import (
    AllowedModels,
    ModelCapability,
    ModelEntry,
    ModelIdentityError,
    UnsupportedCapabilityError,
)
from core.results import (
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
    RunVersions,
)
from core.run_contract import ExecutionTask
from core.stages import GatherSource, VerifyMethod
from core.violations import Violation, ViolationCode
from runtime.budget_guard import BudgetGuard
from runtime.executors.base import RunContext
from runtime.executors.model_stages import MAX_OUTPUT_TOKENS, MODEL_STAGE_KINDS
from runtime.executors.registry import default_executors
from runtime.model_client import ModelClient
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
        benchmark_hash: str,  # identity of the external page/API stores ``pages``/``api`` read
        pages: PageSource | None = None,
        api: ApiSource | None = None,
        compiler: MAFCompiler | None = None,
        checker: ConstraintChecker | None = None,
        allowed_models: AllowedModels | None = None,
    ) -> None:
        self.model = model
        self.allowed_models = allowed_models
        self.benchmark_hash = benchmark_hash
        self.compiler = compiler or MAFCompiler()
        self.checker = checker or ConstraintChecker(
            config=ConstraintConfig(
                unavailable_sources=(GatherSource.JEV,),
                unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,),
            )
        )
        self._executors = default_executors(pages=pages, api=api)
        self.check_model()  # a disallowed client fails at construction, before any run

    def check_model(self, genome: Genome | None = None) -> ModelEntry | None:
        """Fail closed unless the client may run ``genome``: it is in the declared allowed set,
        bound (if registered) to that same entry, and provably able to serve the genome's model
        stages. ``None`` when no set is declared or there is no client (nothing is invoked)."""
        if self.allowed_models is None or self.model is None:
            return None
        entry = self.allowed_models.admit(self.model.model_hash)
        bound = getattr(self.model, "entry", None)
        if bound is not None and bound != entry:
            raise ModelIdentityError(
                f"client is bound to registry entry {bound.name!r}, the allowed set declares "
                f"{entry.name!r} for the same model_hash"
            )
        if genome is not None and any(s.kind in MODEL_STAGE_KINDS for s in genome.stages):
            if not entry.supports(ModelCapability.TEXT_GENERATION):
                raise UnsupportedCapabilityError(
                    f"workflow has model stages; model {entry.name!r} declares no text generation"
                )
            window = entry.context_window
            if window is not None and window < MAX_OUTPUT_TOKENS:
                raise UnsupportedCapabilityError(
                    f"model {entry.name!r} context window {window} < one stage call's output "
                    f"budget {MAX_OUTPUT_TOKENS}"
                )
        return entry

    def versions(self, task: ExecutionTask | None = None) -> RunVersions:
        """Run versions; the grammar version is ``task``'s contract vocabulary (``grammar/1`` for
        the legacy six stages), or the checker's contract-free grammar without a task."""
        grammar = self.checker.grammar_for(task.contract if task is not None else None)
        return RunVersions(
            model_hash=self.model.model_hash if self.model else "no-model",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            benchmark_hash=self.benchmark_hash,
            compiler_version=self.compiler.version,
            grammar_version=grammar.version,
        )

    def run_key(
        self, genome: Genome, task: ExecutionTask, *, trial: int = 0, seed: int = 0
    ) -> RunKey:
        return RunKey(
            genome_hash=genome.genome_hash,
            task_id=task.id,
            contract_hash=task.contract_hash,
            trial=trial,
            seed=seed,
            versions=self.versions(task),
        )

    async def run(
        self, genome: Genome, task: ExecutionTask, *, trial: int = 0, seed: int = 0
    ) -> ExecutionResult:
        self.check_model(genome)  # before admission, compilation or any model call
        key = self.run_key(genome, task, trial=trial, seed=seed)
        violations = self.checker.check(genome, task.contract, complete=True)
        structural = tuple(v for v in violations if v.code != ViolationCode.BUDGET_INFEASIBLE)
        if structural:
            raise InadmissibleGenome(structural)
        if violations:  # only BUDGET_INFEASIBLE: provably cannot fit the caps, do not execute
            return self._static_breach(key, genome, task)

        try:
            from runtime.maf_nodes import Envelope, StageNode
        except ImportError as exc:
            raise RuntimeError("install the optional maf extra to execute workflows") from exc

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
        self, genome: Genome, task: ExecutionTask, *, trial: int = 0, seed: int = 0
    ) -> ExecutionResult:
        return asyncio.run(self.run(genome, task, trial=trial, seed=seed))

    def _static_breach(self, key: RunKey, genome: Genome, task: ExecutionTask) -> ExecutionResult:
        estimate = self.checker.cost_model.estimate(genome, task.contract)
        caps = exceeded_caps(estimate, task.caps)
        return ExecutionResult(
            key=key,
            failure=FailureInfo(
                kind=FailureKind.BUDGET_EXCEEDED,
                message="best-case estimate exceeds caps (not executed): "
                + ", ".join(c.value for c in caps),
                cap=caps[0] if caps else None,
            ),
        )
