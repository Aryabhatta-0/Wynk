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

JEV. GATHER ``jev`` is available only when the runner is built with a ``JevSource`` (and the
``PageSource`` whose snapshot pages it browses). Without one, admission rejects every jev genome
(``RUNTIME_UNAVAILABLE``) and nothing is substituted. A jev run's ``RunVersions.evidence_source``
is the source's ``identity``, so it never shares a run identity with a frozen fetch/api run;
genomes that do not use jev keep exactly the identity they have without a ``JevSource``.

Websearch. GATHER ``websearch`` is available only when the runner is built with a
``WebSearchSource`` AND the task's contract opts in (``allowed_sources``); otherwise it is
``RUNTIME_UNAVAILABLE`` / ``SOURCE_NOT_ALLOWED`` and nothing is substituted. A websearch run's
``RunVersions.evidence_source`` is that source's ``identity`` - the provider it actually used.
``evidence_authority(contract)`` is the live-provider authority a durable job / workflow memory
binds (``core.evidence_authority``): the JEV identity as before, plus the websearch identity
only for a contract that allows websearch, so every other contract keeps its identity.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from compiler.maf_compiler import MAFCompiler
from core.constraints import ConstraintChecker, ConstraintConfig
from core.cost_model import exceeded_caps
from core.evidence_authority import compose_authority
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
from runtime.sources import ApiSource, JevSource, PageSource, WebSearchSource
from runtime.stage_runner import StageRunner

if TYPE_CHECKING:
    from core.task_contract import TaskContract


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
        jev: JevSource | None = None,  # GATHER jev: browses ``pages``' snapshots
        websearch: WebSearchSource | None = None,  # GATHER websearch: opt-in live web search
        compiler: MAFCompiler | None = None,
        checker: ConstraintChecker | None = None,
        allowed_models: AllowedModels | None = None,
    ) -> None:
        if jev is not None and pages is None:
            raise ValueError("a JevSource needs the PageSource whose snapshot pages it browses")
        self.model = model
        self.jev = jev
        self.websearch = websearch
        # Provider identities are read once, when the providers are attached: provenance and the
        # job authority use these pinned values (the executor refuses a provider that drifts).
        self._identities = {
            GatherSource.JEV: jev.identity if jev is not None else None,
            GatherSource.WEBSEARCH: websearch.identity if websearch is not None else None,
        }
        self.allowed_models = allowed_models
        self.benchmark_hash = benchmark_hash
        self.compiler = compiler or MAFCompiler()
        unavailable = tuple(
            s
            for s, provider in ((GatherSource.JEV, jev), (GatherSource.WEBSEARCH, websearch))
            if provider is None
        )
        self.checker = checker or ConstraintChecker(
            config=ConstraintConfig(
                unavailable_sources=unavailable,
                unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,),
            )
        )
        self._executors = default_executors(pages=pages, api=api, jev=jev, websearch=websearch)
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

    def versions(
        self, task: ExecutionTask | None = None, genome: Genome | None = None
    ) -> RunVersions:
        """Run versions; the grammar version is ``task``'s contract vocabulary (``grammar/1`` for
        the legacy six stages), or the checker's contract-free grammar without a task. A
        ``genome`` that gathers with jev / websearch also records that source's identity: the
        provider the run actually uses."""
        grammar = self.checker.grammar_for(task.contract if task is not None else None)
        source = next(
            (s.source for s in (genome.stages if genome else ()) if s.kind == "GATHER"), None
        )
        identity = self._identities.get(source) if source is not None else None
        return RunVersions(
            model_hash=self.model.model_hash if self.model else "no-model",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            benchmark_hash=self.benchmark_hash,
            compiler_version=self.compiler.version,
            grammar_version=grammar.version,
            evidence_source=identity,
        )

    def evidence_authority(self, contract: TaskContract) -> str | None:
        """The live evidence providers a job for ``contract`` is bound with
        (``core.evidence_authority``): the JEV identity whenever one is configured (unchanged from
        #55), and the websearch identity only when ``contract`` allows websearch."""
        allowed = GatherSource.WEBSEARCH in contract.workflow.allowed_sources
        return compose_authority(
            self._identities[GatherSource.JEV],
            self._identities[GatherSource.WEBSEARCH] if allowed else None,
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
            versions=self.versions(task, genome),
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
