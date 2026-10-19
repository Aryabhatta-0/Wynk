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
done / blocked or when every page is open. Every action's cost is known before it runs, so the
tool-call cap is checked first: a decision runs only if one more call fits, and the chosen pages
are read only if all of them fit. An action that does not fit is never started; the stage stops
with ``BUDGET_EXCEEDED`` (cap ``tool_calls``) and usage stays at or below the cap. Other caps
(wall time) stay with the guarded executor. Pages come back sorted by ``page_id`` with the
snapshot's own ``source_ref``. Without a configured ``JevSource`` the source fails explicitly:
there is never a silent fetch/api substitute.

Websearch. One ``websearch`` GATHER is exactly ONE request to the configured
``WebSearchSource``, charged as one tool call. Its query is ``web_query(ctx.task.inputs_text)``:
this row's input values only - never the contract's instructions, context columns, other rows or
target values. The cost is preflighted before the request starts: it is sent only if one more
tool call fits the cap and wall time remains (the request's deadline is the remaining wall-time
budget, at most the provider's timeout); otherwise the stage stops with ``BUDGET_EXCEEDED`` and
nothing is sent. The validated results become the stage's pages (``search_pages``); no returned
URL is opened. A malformed response, a timeout or a provider error fails the stage (the request
was sent, so its tool call is charged) - never retried, and never replaced by fetch / api / jev.
``sequential`` is its only mode (``WEBSEARCH_PARALLEL``). It needs no dataset rows, so it runs
the same way for snapshot and tabular tasks. Without a configured ``WebSearchSource`` it fails
explicitly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from core.payloads import Page, Pages
from core.results import BudgetCap, BudgetUsage, ExecutionMetrics, FailureInfo, FailureKind
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
    WebSearchRequest,
    WebSearchSource,
    checked_decision,
    checked_response,
    search_pages,
    web_query,
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
        websearch: WebSearchSource | None = None,
    ) -> None:
        self._pages = pages
        self._api = api
        self._jev = jev
        self._websearch = websearch

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        stage = inp.stage
        if stage.source == GatherSource.WEBSEARCH:  # reads no dataset: any task format
            if self._websearch is None:
                return _fail(inp, "gather source 'websearch' is not available")
            return await _websearch(inp, ctx, self._websearch)
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

    def left() -> int:  # tool calls the run may still spend, this stage's actions included
        return ctx.guard.caps.tool_calls - ctx.guard.usage.tool_calls - calls

    def spent() -> dict[str, Any]:
        return {
            "usage": BudgetUsage(tool_calls=calls),
            "metrics": ExecutionMetrics(pages_fetched=len(opened)),
        }

    def unaffordable(action: str, cost: int) -> ExecutorOutput:  # refused before it runs
        failure = FailureInfo(
            kind=FailureKind.BUDGET_EXCEEDED,
            message=f"cap exceeded: tool_calls ({action} needs {cost}, {max(left(), 0)} left)",
            stage_index=inp.stage_index,
            cap=BudgetCap.TOOL_CALLS,
        )
        return ExecutorOutput(failure=failure, **spent())

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
    while True:
        unopened = [n for n in names if n not in opened]
        if not unopened:
            break  # nothing left to choose from: no decision (and no tool call) needed
        if left() < 1:  # a decision costs one tool call: never ask for one the cap forbids
            return unaffordable("the next jev decision", 1)
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
        targets = sorted(decision.targets)
        if left() < len(targets):  # the whole action fits or no page of it is read
            return unaffordable(f"opening {len(targets)} page(s)", len(targets))
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
    if not opened:
        return _fail(inp, f"jev gathered no pages (agent stopped: {stopped})", **spent())
    return ExecutorOutput(payload=Pages(pages=tuple(opened[n] for n in sorted(opened))), **spent())


async def _websearch(inp: ExecutorInput, ctx: RunContext, src: WebSearchSource) -> ExecutorOutput:
    if inp.stage.mode != GatherMode.SEQUENTIAL:
        return _fail(inp, f"gather source 'websearch' cannot use mode {inp.stage.mode.value}")
    caps, used = ctx.guard.caps, ctx.guard.usage
    if caps.tool_calls - used.tool_calls < 1:  # preflight: the request is never sent
        failure = FailureInfo(
            kind=FailureKind.BUDGET_EXCEEDED,
            message="cap exceeded: tool_calls (a websearch request needs 1, "
            f"{max(caps.tool_calls - used.tool_calls, 0)} left)",
            stage_index=inp.stage_index,
            cap=BudgetCap.TOOL_CALLS,
        )
        return ExecutorOutput(failure=failure)
    remaining_s = caps.wall_time_s - used.wall_time_s
    if remaining_s <= 0:
        failure = FailureInfo(
            kind=FailureKind.BUDGET_EXCEEDED,
            message="cap exceeded: wall_time_s (no time left for a websearch request)",
            stage_index=inp.stage_index,
            cap=BudgetCap.WALL_TIME,
        )
        return ExecutorOutput(failure=failure)
    query = web_query(ctx.task.inputs_text)
    if not query:
        return _fail(inp, "websearch: this row has no input text to search for")
    request = WebSearchRequest(query=query, max_results=src.max_results, timeout_s=remaining_s)
    spent = BudgetUsage(tool_calls=1)  # sent: charged whether or not the answer is usable
    try:
        response = checked_response(await asyncio.to_thread(src.search, request), request)
    except (SourceError, ValidationError) as exc:
        return _fail(inp, f"websearch: {_short(exc, 'response')}", usage=spent)
    pages = search_pages(response)
    metrics = ExecutionMetrics(pages_fetched=len(pages))
    if not pages:
        return _fail(inp, "websearch returned no results", usage=spent, metrics=metrics)
    return ExecutorOutput(payload=Pages(pages=pages), usage=spent, metrics=metrics)


def _short(exc: Exception, what: str = "decision") -> str:
    if isinstance(exc, ValidationError):  # never echo the provider's raw payload
        return f"invalid {what} ({exc.error_count()} validation error(s))"
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
