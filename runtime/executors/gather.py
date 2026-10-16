"""GATHER executor: Task -> Pages, from the data source the task's contract names.

* ``SnapshotSource`` (``wynk_snapshot`` datasets): local frozen pages (fetch), the mock API (api),
  or a JEV browsing agent over the snapshot's pages (jev).
* ``InlineSource`` (csv / jsonl rows): the row's own context values, one page per column (fetch).

Every page read / endpoint call is one tool call. ``parallel-N`` modes read up to N pages
concurrently; results keep a deterministic (sorted) order regardless of completion order.

JEV. Wynk runs the browser loop; the ``JevSource`` only chooses. Each step shows the agent the
pages it has not opened yet (``ctx.task.question`` is the goal), takes one validated decision
(``runtime.sources.checked_decision``) and opens the chosen pages. One decision is one tool call
and every opened page is one more, so JEV work is charged like every other source. ``sequential``
opens one page per decision, ``parallel-2`` up to two (read concurrently); ``parallel-4`` is
inadmissible for jev (``JEV_PARALLEL_4``) and refused here too. The loop ends when the agent says
done / blocked, when every page is open, or as soon as the run's usage crosses a cap (the guarded
executor then reports ``BUDGET_EXCEEDED``). Pages come back sorted by ``page_id`` with the
snapshot's own ``source_ref``. Without a configured ``JevSource`` the source fails explicitly:
there is never a silent fetch/api substitute.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from core.payloads import Page, Pages
from core.results import BudgetUsage, ExecutionMetrics, FailureInfo, FailureKind, usage_exceeds
from core.run_contract import InlineSource
from core.stages import GatherMode, GatherSource, StageKind
from runtime.executors.base import ExecutorInput, ExecutorOutput, RunContext, StageExecutor
from runtime.sources import (
    ApiSource,
    JevElement,
    JevObservation,
    JevSource,
    PageSource,
    SourceError,
    checked_decision,
)

_CONCURRENCY = {
    GatherMode.SEQUENTIAL: 1,
    GatherMode.PARALLEL_2: 2,
    GatherMode.PARALLEL_4: 4,
}
# Pages a JEV agent may open per decision. No parallel-4: constraint JEV_PARALLEL_4.
JEV_WIDTH = {GatherMode.SEQUENTIAL: 1, GatherMode.PARALLEL_2: 2}


class GatherExecutor(StageExecutor):
    kind = StageKind.GATHER

    def __init__(
        self,
        pages: PageSource | None = None,
        api: ApiSource | None = None,
        jev: JevSource | None = None,
    ) -> None:
        self._pages = pages
        self._api = api
        self._jev = jev

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        stage = inp.stage
        source = ctx.task.source
        if isinstance(source, InlineSource):
            return _inline(inp, ctx, source)
        snap = source.snapshot_id
        if stage.source == GatherSource.JEV and self._jev is not None and self._pages is not None:
            return await _jev(inp, ctx, self._jev, self._pages, snap)
        read: Callable[[str], Page]
        try:
            if stage.source == GatherSource.FETCH and self._pages is not None:
                pages_src = self._pages
                names = pages_src.list_page_ids(snap)

                def read(name: str) -> Page:
                    return pages_src.read_page(snap, name)

            elif stage.source == GatherSource.API and self._api is not None:
                api_src = self._api
                names = api_src.list_endpoints(snap)

                def read(name: str) -> Page:
                    return api_src.call(snap, name)

            else:
                return _fail(inp, f"gather source '{stage.source.value}' is not available")
            gate = asyncio.Semaphore(_CONCURRENCY[stage.mode])

            async def one(name: str) -> Page:
                async with gate:
                    return await asyncio.to_thread(read, name)

            results = await asyncio.gather(*(one(n) for n in names), return_exceptions=True)
        except SourceError as exc:
            return _fail(inp, str(exc))
        pages = [p for p in results if isinstance(p, Page)]
        usage = BudgetUsage(tool_calls=len(names))
        metrics = ExecutionMetrics(pages_fetched=len(pages))
        for result in results:
            if isinstance(result, SourceError):
                return _fail(inp, str(result), usage=usage, metrics=metrics)
            if isinstance(result, BaseException):
                raise result
        return ExecutorOutput(
            payload=Pages(pages=tuple(pages)),
            usage=usage,
            metrics=metrics,
        )


async def _jev(
    inp: ExecutorInput, ctx: RunContext, jev: JevSource, pages_src: PageSource, snap: str
) -> ExecutorOutput:
    width = JEV_WIDTH.get(inp.stage.mode)
    if width is None:
        return _fail(inp, f"gather source 'jev' cannot use mode {inp.stage.mode.value}")
    calls = 0
    opened: dict[str, Page] = {}

    def over_cap() -> bool:  # the run's usage, this stage's actions included, crossed a cap
        spent = ctx.guard.usage + BudgetUsage(tool_calls=calls)
        return bool(usage_exceeds(spent, ctx.guard.caps))

    def spent() -> dict[str, Any]:
        return {
            "usage": BudgetUsage(tool_calls=calls),
            "metrics": ExecutionMetrics(pages_fetched=len(opened)),
        }

    try:
        names = sorted(pages_src.list_page_ids(snap))
    except SourceError as exc:
        return _fail(inp, str(exc))
    gate = asyncio.Semaphore(width)

    async def open_page(name: str) -> Page:
        async with gate:
            return await asyncio.to_thread(pages_src.read_page, snap, name)

    stopped = "done"
    step = 0
    while not over_cap():
        unopened = [n for n in names if n not in opened]
        if not unopened:
            break  # nothing left to choose from: no decision (and no tool call) needed
        observation = JevObservation(
            goal=ctx.task.question,
            surface=f"snapshot://{snap}",
            elements=tuple(JevElement(element_id=n, label=n) for n in unopened),
            opened=tuple(opened.values()),
            width=width,
            step=step,
        )
        step += 1
        calls += 1  # a decision is a tool action, charged whether or not it is usable
        try:
            decision = checked_decision(
                await asyncio.to_thread(jev.decide, observation), observation
            )
        except (SourceError, ValidationError) as exc:
            return _fail(inp, f"jev: {_short(exc)}", **spent())
        if decision.operation != "open":
            stopped = decision.operation
            break
        if over_cap():
            break
        targets = sorted(decision.targets)
        calls += len(targets)
        results = await asyncio.gather(*(open_page(t) for t in targets), return_exceptions=True)
        for name, result in zip(targets, results, strict=True):
            if isinstance(result, SourceError):
                return _fail(inp, f"jev: {result}", **spent())
            if isinstance(result, BaseException):
                raise result
            if result.page_id != name:
                return _fail(inp, f"jev: opened {name!r}, got page {result.page_id!r}", **spent())
            opened[name] = result
    if not opened and not over_cap():
        return _fail(inp, f"jev gathered no pages (agent stopped: {stopped})", **spent())
    return ExecutorOutput(payload=Pages(pages=tuple(opened[n] for n in sorted(opened))), **spent())


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):  # never echo the provider's raw payload
        return f"invalid decision ({exc.error_count()} validation error(s))"
    return str(exc)


def _inline(inp: ExecutorInput, ctx: RunContext, source: InlineSource) -> ExecutorOutput:
    """Pages from the row itself: no I/O, so concurrency modes change nothing."""
    if inp.stage.source != GatherSource.FETCH:
        return _fail(inp, f"gather source '{inp.stage.source.value}' cannot read dataset rows")
    ds = ctx.task.contract.dataset
    pages = tuple(
        Page(
            page_id=column,
            source_ref=f"row://{ds.dataset_id}/{ds.dataset_version}/{ctx.task.id}#{column}",
            content=text,
        )
        for column, text in source.documents
    )
    return ExecutorOutput(
        payload=Pages(pages=pages),
        usage=BudgetUsage(tool_calls=len(pages)),
        metrics=ExecutionMetrics(pages_fetched=len(pages)),
    )


def _fail(inp: ExecutorInput, message: str, **kwargs) -> ExecutorOutput:
    return ExecutorOutput(
        failure=FailureInfo(
            kind=FailureKind.EXECUTOR_ERROR, message=message, stage_index=inp.stage_index
        ),
        **kwargs,
    )
