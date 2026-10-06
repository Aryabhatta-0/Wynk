"""Executes the stages of one run and owns bounded recovery. Framework-neutral (no MAF).

Every executor call goes through ``GuardedExecutor`` (budget accounting). A VERIFY stage that
fails its check triggers the retry strategy written in the genome:

* ``retry-1`` / ``retry-2``: re-run the stage that produced the verified payload (with a fresh
  per-attempt model seed) and verify again, at most 1 / 2 times;
* ``regather``: re-run everything from GATHER up to the verifier, once (admission guarantees
  GATHER is stage 0 whenever ``regather`` is used).

A CONFIDENCE_GATE that abstains (``LOW_CONFIDENCE``) ends the run; it never retries.

Every retry is charged to the budget (``retries`` cap); a breach stops the run. When retries are
exhausted the run ends with the verifier's failure. Re-execution is bounded by construction.
"""

from __future__ import annotations

from collections.abc import Mapping

from compiler.dag import DagNode, WorkflowDAG
from core.payloads import Answer, Page, Pages, Payload
from core.results import (
    BudgetUsage,
    ExecutionMetrics,
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
)
from core.stages import FailureStrategy, StageKind
from runtime.executors.base import (
    ExecutorInput,
    ExecutorOutput,
    GuardedExecutor,
    RunContext,
    StageExecutor,
    WorkflowState,
)

_MAX_RETRIES = {FailureStrategy.RETRY_1: 1, FailureStrategy.RETRY_2: 2, FailureStrategy.REGATHER: 1}


class StageRunner:
    def __init__(
        self, dag: WorkflowDAG, executors: Mapping[StageKind, StageExecutor], ctx: RunContext
    ) -> None:
        self.ctx = ctx
        self._nodes = list(dag.nodes)
        self._guarded = {kind: GuardedExecutor(ex) for kind, ex in executors.items()}
        self.state = WorkflowState(payload=ctx.task)
        self.failure: FailureInfo | None = None
        self._inputs: dict[int, Payload] = {}
        self._source_pages: tuple[Page, ...] = ()
        self._metrics = ExecutionMetrics()
        self._last_answer: Answer | None = None
        self._final: Payload | None = None

    # -- called once per DAG node by the framework adapter ---------------------------------
    async def execute_node(self, node: DagNode, payload: Payload) -> Payload | None:
        """Run one stage (with recovery). Returns the output payload, or None if the run failed
        (``self.failure`` is then set and downstream nodes must not run)."""
        try:
            out = await self._run_with_recovery(node, payload)
        except Exception as exc:  # executor bug: record, never crash the framework
            out = ExecutorOutput(
                failure=FailureInfo(
                    kind=FailureKind.EXECUTOR_ERROR,
                    message=f"{type(exc).__name__}: {exc}",
                    stage_index=node.stage_index,
                )
            )
        if out.failure is not None:
            self.failure = out.failure
            return None
        self._final = out.payload
        return out.payload

    # -- recovery ---------------------------------------------------------------------------
    async def _run_with_recovery(self, node: DagNode, payload: Payload) -> ExecutorOutput:
        self._inputs[node.stage_index] = payload
        out = await self._exec(node, payload, attempt=0)
        if out.failure is None or node.stage.kind != "VERIFY":
            return out
        if out.failure.kind is not FailureKind.SCHEMA_INVALID:  # budget / executor errors: stop
            return out
        strategy = node.stage.on_failure
        first = 0 if strategy == FailureStrategy.REGATHER else node.stage_index - 1
        for attempt in range(1, _MAX_RETRIES[strategy] + 1):
            hit = self.ctx.guard.charge(BudgetUsage(retries=1))
            if hit:
                return self._budget_failure(node, hit[0].value, hit[0])
            produced = await self._rerun(first, node.stage_index - 1, attempt)
            if produced.failure is not None:
                return produced
            assert produced.payload is not None
            self._inputs[node.stage_index] = produced.payload
            out = await self._exec(node, produced.payload, attempt)
            if out.failure is None:
                self.state.failure = None  # recovered
                return out
        return out

    async def _rerun(self, first: int, last: int, attempt: int) -> ExecutorOutput:
        payload = self._inputs[first]
        out = ExecutorOutput(payload=payload)
        for j in range(first, last + 1):
            self._inputs[j] = payload
            out = await self._exec(self._nodes[j], payload, attempt)
            if out.failure is not None:
                return out
            assert out.payload is not None
            payload = out.payload
        return out

    def _budget_failure(self, node: DagNode, message: str, cap) -> ExecutorOutput:
        return ExecutorOutput(
            failure=FailureInfo(
                kind=FailureKind.BUDGET_EXCEEDED,
                message=f"cap exceeded: {message}",
                stage_index=node.stage_index,
                cap=cap,
            )
        )

    # -- one guarded execution --------------------------------------------------------------
    async def _exec(self, node: DagNode, payload: Payload, attempt: int) -> ExecutorOutput:
        inp = ExecutorInput(
            stage_index=node.stage_index,
            stage=node.stage,
            payload=payload,
            attempt=attempt,
            source_pages=self._source_pages,
        )
        out = await self._guarded[StageKind(node.stage.kind)].run(inp, self.ctx)
        self.state.apply(inp, out)
        self._metrics = _add_metrics(self._metrics, out.metrics)
        if out.failure is None:
            if isinstance(out.payload, Pages) and node.stage.kind == "GATHER":
                self._source_pages = out.payload.pages
            if isinstance(out.payload, Answer):
                self._last_answer = out.payload
        return out

    # -- result -----------------------------------------------------------------------------
    def result(self, key: RunKey) -> ExecutionResult:
        failure = self.failure
        answer = self._final if isinstance(self._final, Answer) else self._last_answer
        if failure is None and not isinstance(self._final, Answer):
            failure = FailureInfo(kind=FailureKind.NO_ANSWER, message="workflow produced no Answer")
        return ExecutionResult(
            key=key,
            answer=answer,
            metrics=self._metrics,
            stage_trace=tuple(self.state.trace),
            budget_usage=self.ctx.guard.usage,
            failure=failure,
        )


def _add_metrics(a: ExecutionMetrics, b: ExecutionMetrics) -> ExecutionMetrics:
    return ExecutionMetrics(
        model_calls=a.model_calls + b.model_calls,
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        pages_fetched=a.pages_fetched + b.pages_fetched,
        cache_hits=a.cache_hits + b.cache_hits,
        backoff_time_s=a.backoff_time_s + b.backoff_time_s,
    )
