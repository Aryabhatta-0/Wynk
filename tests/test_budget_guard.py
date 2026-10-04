import asyncio

import pytest

from core.payloads import Pages
from core.results import (
    BudgetCap,
    BudgetUsage,
    FailureKind,
    RunVersions,
    usage_exceeds,
)
from core.stages import StageKind
from runtime.budget_guard import BudgetExceeded, BudgetGuard
from runtime.executors.base import (
    ExecutorInput,
    ExecutorOutput,
    GuardedExecutor,
    RunContext,
    StageExecutor,
    WorkflowState,
)
from tests.conftest import gather, make_caps, make_runtime_task

CAPS = make_caps(tokens=100, wall_time_s=10.0, tool_calls=5, retries=2)


def test_usage_below_or_at_caps_is_accepted():
    g = BudgetGuard(CAPS)
    assert g.charge(BudgetUsage(tokens=50, wall_time_s=5.0, tool_calls=3, retries=1)) == ()
    assert (
        g.charge(BudgetUsage(tokens=50, wall_time_s=5.0, tool_calls=2, retries=1)) == ()
    )  # == cap
    g.ensure_within()
    assert g.usage == BudgetUsage(tokens=100, wall_time_s=10.0, tool_calls=5, retries=2)


@pytest.mark.parametrize(
    ("over", "cap"),
    [
        (BudgetUsage(tokens=101), BudgetCap.TOKENS),
        (BudgetUsage(wall_time_s=10.001), BudgetCap.WALL_TIME),
        (BudgetUsage(tool_calls=6), BudgetCap.TOOL_CALLS),
        (BudgetUsage(retries=3), BudgetCap.RETRIES),
    ],
)
def test_each_cap_independently_triggers_infeasibility(over, cap):
    g = BudgetGuard(CAPS)
    assert g.charge(over) == (cap,)  # and ONLY that cap
    with pytest.raises(BudgetExceeded) as err:
        g.ensure_within()
    assert err.value.caps_hit == (cap,)
    assert usage_exceeds(over, CAPS) == (cap,)


def test_usage_accumulates_across_charges():
    g = BudgetGuard(CAPS)
    assert g.charge(BudgetUsage(tokens=60)) == ()
    assert g.charge(BudgetUsage(tokens=60)) == (BudgetCap.TOKENS,)


class _Fake(StageExecutor):
    kind = StageKind.GATHER

    def __init__(self, usage):
        self.usage, self.calls = usage, 0

    async def run(self, inp, ctx):
        self.calls += 1
        return ExecutorOutput(payload=Pages(), usage=self.usage)


def _ctx(guard):
    versions = RunVersions(
        model_hash="m",
        prompt_template_version="p",
        benchmark_hash="b",
        compiler_version="c",
        grammar_version="g",
    )
    return RunContext(task=make_runtime_task(), seed=0, trial=0, versions=versions, guard=guard)


def _input(i=0):
    return ExecutorInput(stage_index=i, stage=gather(), payload=make_runtime_task())


def test_guarded_executor_converts_a_breach_into_a_budget_failure_and_stops_further_work():
    guard = BudgetGuard(CAPS)
    ctx = _ctx(guard)
    inner = _Fake(BudgetUsage(tool_calls=6, wall_time_s=1.0))
    ex = GuardedExecutor(inner)

    out = asyncio.run(ex.run(_input(0), ctx))
    assert out.payload is None
    assert out.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert out.failure.cap is BudgetCap.TOOL_CALLS
    assert out.failure.stage_index == 0

    again = asyncio.run(ex.run(_input(1), ctx))  # already over budget: inner is NOT invoked
    assert again.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert inner.calls == 1


def test_guarded_executor_passes_through_in_budget_output_and_measures_wall_time():
    ticks = iter([0.0, 2.5])
    guard = BudgetGuard(CAPS)
    ex = GuardedExecutor(_Fake(BudgetUsage(tokens=10)), clock=lambda: next(ticks))
    out = asyncio.run(ex.run(_input(), _ctx(guard)))
    assert out.failure is None and isinstance(out.payload, Pages)
    assert out.usage.wall_time_s == 2.5
    assert guard.usage.tokens == 10 and guard.usage.wall_time_s == 2.5


def test_wall_time_breach_from_measured_clock():
    ticks = iter([0.0, 11.0])
    ex = GuardedExecutor(_Fake(BudgetUsage()), clock=lambda: next(ticks))
    out = asyncio.run(ex.run(_input(), _ctx(BudgetGuard(CAPS))))
    assert out.failure.cap is BudgetCap.WALL_TIME


def test_executor_output_requires_exactly_one_of_payload_or_failure():
    with pytest.raises(ValueError):
        ExecutorOutput()
    with pytest.raises(ValueError):
        ExecutorOutput(payload=Pages(), failure=_failure())


def _failure():
    from core.results import FailureInfo

    return FailureInfo(kind=FailureKind.EXECUTOR_ERROR, message="x")


def test_workflow_state_records_trace_and_failure():
    state = WorkflowState(payload=make_runtime_task())
    inp = _input()
    state.apply(inp, ExecutorOutput(payload=Pages(), usage=BudgetUsage(tool_calls=1)))
    assert isinstance(state.payload, Pages) and state.failure is None
    state.apply(_input(1), ExecutorOutput(failure=_failure()))
    assert [t.status.value for t in state.trace] == ["ok", "failed"]
    assert state.failure is not None
