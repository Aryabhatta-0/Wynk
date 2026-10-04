"""The MVP executor set, one per stage kind."""

from __future__ import annotations

from core.stages import StageKind
from runtime.executors.base import StageExecutor
from runtime.executors.filter import FilterExecutor
from runtime.executors.gather import GatherExecutor
from runtime.executors.gemma_stages import ExtractExecutor, ReasonExecutor, SynthesizeExecutor
from runtime.executors.verify import VerifyExecutor
from runtime.sources import ApiSource, PageSource


def default_executors(
    pages: PageSource | None = None, api: ApiSource | None = None
) -> dict[StageKind, StageExecutor]:
    return {
        StageKind.GATHER: GatherExecutor(pages=pages, api=api),
        StageKind.FILTER: FilterExecutor(),
        StageKind.EXTRACT: ExtractExecutor(),
        StageKind.REASON: ReasonExecutor(),
        StageKind.VERIFY: VerifyExecutor(),
        StageKind.SYNTHESIZE: SynthesizeExecutor(),
    }
