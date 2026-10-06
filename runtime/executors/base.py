"""Executor contract shared by every stage implementation (gather_*, filter, model stages,
verifiers). Executors never decide correctness; failures describe what broke, not who is right.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from core.canonical import canonical_hash
from core.payloads import Page, Payload
from core.results import (
    BudgetUsage,
    ExecutionMetrics,
    FailureInfo,
    FailureKind,
    RunVersions,
    StageStatus,
    StageTrace,
)
from core.run_contract import ExecutionTask
from core.stages import StageKind, StageSpec
from runtime.budget_guard import BudgetGuard
from runtime.model_client import ModelClient


@dataclass(frozen=True)
class RunContext:
    """Everything an executor may know about the run: the contract-bound task (no targets)."""

    task: ExecutionTask
    seed: int
    trial: int
    versions: RunVersions
    guard: BudgetGuard
    model: ModelClient | None = None


@dataclass(frozen=True)
class ExecutorInput:
    stage_index: int
    stage: StageSpec
    payload: Payload
    attempt: int = 0  # 0 = first execution; >0 = a bounded retry of this stage
    # Original (unfiltered) gathered pages, for locating/validating evidence spans.
    source_pages: tuple[Page, ...] = ()


@dataclass(frozen=True)
class ExecutorOutput:
    """Exactly one of ``payload`` / ``failure`` is set."""

    payload: Payload | None = None
    failure: FailureInfo | None = None
    usage: BudgetUsage = BudgetUsage()
    metrics: ExecutionMetrics = ExecutionMetrics()

    def __post_init__(self) -> None:
        if (self.payload is None) == (self.failure is None):
            raise ValueError("ExecutorOutput needs exactly one of payload or failure")


def derive_seed(run_seed: int, stage_index: int, attempt: int) -> int:
    """Deterministic per-call model seed: same (seed, stage, attempt) => same value."""
    digest = hashlib.sha256(f"{run_seed}:{stage_index}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


class StageExecutor(ABC):
    kind: StageKind

    @abstractmethod
    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput: ...


class GuardedExecutor(StageExecutor):
    """Budget-accounting wrapper for ANY executor (the accounting hook)."""

    def __init__(
        self, inner: StageExecutor, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.inner = inner
        self.kind = inner.kind
        self._clock = clock

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        if ctx.guard.exceeded():  # already over budget: do not run the stage at all
            return _breach(inp, ctx, BudgetUsage(), ExecutionMetrics())
        start = self._clock()
        out = await self.inner.run(inp, ctx)
        usage = out.usage
        if usage.wall_time_s == 0.0:  # executor did not self-report: use measured time
            usage = usage.model_copy(
                update={"wall_time_s": max(0.0, self._clock() - start - out.metrics.backoff_time_s)}
            )
        breached = ctx.guard.charge(usage)
        if breached and not (out.failure and out.failure.kind is FailureKind.MODEL_ERROR):
            return _breach(inp, ctx, usage, out.metrics)
        return ExecutorOutput(
            payload=out.payload, failure=out.failure, usage=usage, metrics=out.metrics
        )


def _breach(
    inp: ExecutorInput, ctx: RunContext, usage: BudgetUsage, metrics: ExecutionMetrics
) -> ExecutorOutput:
    cap = ctx.guard.exceeded()[0]
    failure = FailureInfo(
        kind=FailureKind.BUDGET_EXCEEDED,
        message=f"cap exceeded: {cap.value}",
        stage_index=inp.stage_index,
        cap=cap,
    )
    return ExecutorOutput(failure=failure, usage=usage, metrics=metrics)


@dataclass
class WorkflowState:
    """Mutable per-run state threaded through executors by the runtime adapter."""

    payload: Payload
    trace: list[StageTrace] = field(default_factory=list)
    failure: FailureInfo | None = None

    def apply(self, inp: ExecutorInput, out: ExecutorOutput) -> None:
        output_digest = (
            None if out.payload is None else canonical_hash(out.payload.model_dump(mode="json"))
        )
        self.trace.append(
            StageTrace(
                stage_index=inp.stage_index,
                kind=StageKind(inp.stage.kind),
                status=StageStatus.FAILED if out.failure is not None else StageStatus.OK,
                usage=out.usage,
                input_digest=canonical_hash(inp.payload.model_dump(mode="json")),
                output_digest=output_digest,
                failure=out.failure,
            )
        )
        if out.failure is not None:
            self.failure = out.failure
        elif out.payload is not None:
            self.payload = out.payload
