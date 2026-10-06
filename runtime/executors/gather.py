"""GATHER executor: Task -> Pages, from the data source the task's contract names.

* ``SnapshotSource`` (``wynk_snapshot`` datasets): local frozen pages (fetch) or the mock API (api).
* ``InlineSource`` (csv / jsonl rows): the row's own context values, one page per column (fetch).

Every page read / endpoint call is one tool call. ``parallel-N`` modes read up to N pages
concurrently; results keep a deterministic (sorted) order regardless of completion order.
Jev is not part of the MVP and fails explicitly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from core.payloads import Page, Pages
from core.results import BudgetUsage, ExecutionMetrics, FailureInfo, FailureKind
from core.run_contract import InlineSource
from core.stages import GatherMode, GatherSource, StageKind
from runtime.executors.base import ExecutorInput, ExecutorOutput, RunContext, StageExecutor
from runtime.sources import ApiSource, PageSource, SourceError

_CONCURRENCY = {
    GatherMode.SEQUENTIAL: 1,
    GatherMode.PARALLEL_2: 2,
    GatherMode.PARALLEL_4: 4,
}


class GatherExecutor(StageExecutor):
    kind = StageKind.GATHER

    def __init__(self, pages: PageSource | None = None, api: ApiSource | None = None) -> None:
        self._pages = pages
        self._api = api

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        stage = inp.stage
        source = ctx.task.source
        if isinstance(source, InlineSource):
            return _inline(inp, ctx, source)
        snap = source.snapshot_id
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
                return _fail(inp, f"gather source '{stage.source.value}' is not available (MVP)")
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
