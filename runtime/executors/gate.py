"""CONFIDENCE_GATE executor: Answer -> Answer, deterministic, no model, no ground truth.

Confidence is evidence support (``verify.answer_support``): the fraction of answer fields whose
value is backed by a span that verifies against the gathered pages. Below the genome's
``min_support`` the run abstains with ``LOW_CONFIDENCE``; there is no retry. The gate never
decides correctness - an abstention is scored by the offline evaluator like any other failure.
"""

from __future__ import annotations

from core.payloads import Answer
from core.results import FailureInfo, FailureKind
from core.stages import SUPPORT_FRACTION, StageKind
from runtime.executors.base import ExecutorInput, ExecutorOutput, RunContext, StageExecutor
from runtime.executors.verify import answer_support


class ConfidenceGateExecutor(StageExecutor):
    kind = StageKind.CONFIDENCE_GATE

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        answer = inp.payload
        if not isinstance(answer, Answer):
            return _fail(inp, FailureKind.EXECUTOR_ERROR, "CONFIDENCE_GATE needs an Answer input")
        originals = {p.page_id: p for p in inp.source_pages}
        support = answer_support(answer, ctx.task.answer_schema, originals)
        needed = SUPPORT_FRACTION[inp.stage.min_support]
        if support < needed:
            return _fail(
                inp,
                FailureKind.LOW_CONFIDENCE,
                f"evidence support {support:.2f} is below {needed:.2f}; abstaining",
            )
        return ExecutorOutput(payload=answer)


def _fail(inp: ExecutorInput, kind: FailureKind, message: str) -> ExecutorOutput:
    return ExecutorOutput(
        failure=FailureInfo(kind=kind, message=message, stage_index=inp.stage_index)
    )
