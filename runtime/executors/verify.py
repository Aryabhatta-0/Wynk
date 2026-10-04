"""Runtime VERIFY: ``schema_check`` and ``evidence_span``. Deterministic, no model, no ground truth.

Runtime VERIFY only checks STRUCTURE and evidence integrity of what the runtime produced
(Facts->Facts or Answer->Answer). It does not and cannot decide whether an answer is correct:
that is the offline evaluator's job. A failed check returns ``SCHEMA_INVALID`` (the contract has
no separate evidence-failure kind); the runner decides whether to retry.
``self_consistency`` is not part of the MVP and fails explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from core.evidence import EvidenceSpan
from core.payloads import Answer, Facts, Page
from core.results import FailureInfo, FailureKind
from core.stages import StageKind, VerifyMethod
from core.task_spec import AnswerSchema, FieldType
from runtime.executors.base import ExecutorInput, ExecutorOutput, RunContext, StageExecutor
from runtime.spans import span_text, supports


def type_ok(value: Any, ftype: FieldType) -> bool:
    match ftype:
        case FieldType.STRING:
            return isinstance(value, str)
        case FieldType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        case FieldType.NUMBER:
            return isinstance(value, int | float) and not isinstance(value, bool)
        case FieldType.BOOLEAN:
            return isinstance(value, bool)
        case FieldType.DATE:
            try:
                return isinstance(value, str) and bool(date.fromisoformat(value))
            except ValueError:
                return False
        case FieldType.STRING_LIST:
            return isinstance(value, list) and all(isinstance(v, str) for v in value)
    return False


def check_schema(payload: Facts | Answer, schema: AnswerSchema) -> list[str]:
    problems: list[str] = []
    by_name = {f.name: f for f in schema.fields}
    if isinstance(payload, Facts):
        present = {f.field for f in payload.facts}
        items = [(f.field, f.value) for f in payload.facts]
    else:
        present = set(payload.values)
        items = list(payload.values.items())
    for name, value in items:
        if name not in by_name:
            problems.append(f"unknown field '{name}'")
        elif isinstance(payload, Answer) and not type_ok(value, by_name[name].type):
            problems.append(f"field '{name}' is not a valid {by_name[name].type.value}")
    problems += [
        f"missing required field '{f.name}'"
        for f in schema.fields
        if f.required and f.name not in present
    ]
    return problems


def _evidence_problems(
    label: str,
    value: Any,
    spans: tuple[EvidenceSpan, ...],
    originals: Mapping[str, Page],
) -> list[str]:
    if not spans:
        return [f"{label}: no evidence"]
    texts = [span_text(s, originals) for s in spans]
    if any(t is None for t in texts):
        return [f"{label}: evidence span does not match the source page"]
    if not any(supports(value, t) for t in texts if t is not None):
        return [f"{label}: evidence text does not contain the value"]
    return []


def check_evidence(payload: Facts | Answer, originals: Mapping[str, Page]) -> list[str]:
    if not originals:
        return ["no source pages available to validate evidence"]
    problems: list[str] = []
    if isinstance(payload, Facts):
        for f in payload.facts:
            problems += _evidence_problems(f"fact '{f.field}'", f.value, f.spans, originals)
    else:
        cited = {e.field: e.spans for e in payload.evidence}
        for name, value in payload.values.items():
            problems += _evidence_problems(
                f"answer '{name}'", value, cited.get(name, ()), originals
            )
    return problems


class VerifyExecutor(StageExecutor):
    kind = StageKind.VERIFY

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        payload = inp.payload
        method = inp.stage.method
        if not isinstance(payload, Facts | Answer):
            return _fail(inp, FailureKind.EXECUTOR_ERROR, "VERIFY needs Facts or Answer input")
        if method == VerifyMethod.SCHEMA_CHECK:
            problems = check_schema(payload, ctx.task.answer_schema)
        elif method == VerifyMethod.EVIDENCE_SPAN:
            problems = check_evidence(payload, {p.page_id: p for p in inp.source_pages})
        else:
            return _fail(
                inp, FailureKind.EXECUTOR_ERROR, f"verifier '{method.value}' is not in MVP"
            )
        if problems:
            return _fail(inp, FailureKind.SCHEMA_INVALID, f"{method.value}: " + "; ".join(problems))
        return ExecutorOutput(payload=payload)


def _fail(inp: ExecutorInput, kind: FailureKind, message: str) -> ExecutorOutput:
    return ExecutorOutput(
        failure=FailureInfo(kind=kind, message=message, stage_index=inp.stage_index)
    )
