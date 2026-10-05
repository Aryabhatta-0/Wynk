"""Gemma-backed stages: EXTRACT (Pages->Facts), REASON (Facts->Facts), SYNTHESIZE (Facts->Answer),
DIRECT (Task->Answer; Gemma synthesizes from the task input alone, so it cites no evidence).

Gemma only extracts / reasons / synthesizes. It is asked for values plus verbatim quotes; the
runtime turns quotes into structured ``EvidenceSpan``s (never trusting model-supplied offsets).
If no model client is configured or the backend fails, the stage FAILS - nothing is faked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from core.evidence import EvidenceSpan, FieldEvidence
from core.payloads import Answer, Fact, Facts, Page, Pages
from core.results import (
    BudgetUsage,
    ExecutionMetrics,
    FailureInfo,
    FailureKind,
)
from core.stages import StageKind
from core.task_spec import RuntimeTask
from runtime.executors.base import (
    ExecutorInput,
    ExecutorOutput,
    RunContext,
    StageExecutor,
    derive_seed,
)
from runtime.gemma_client import GenerationRequest, ModelError, ModelRole
from runtime.prompts import templates as T
from runtime.spans import locate_quote, span_text, supports

MAX_OUTPUT_TOKENS = 1024
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _fail(inp: ExecutorInput, kind: FailureKind, message: str, **kw: Any) -> ExecutorOutput:
    failure = FailureInfo(kind=kind, message=message, stage_index=inp.stage_index)
    return ExecutorOutput(failure=failure, **kw)


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


async def generate_json(
    inp: ExecutorInput,
    ctx: RunContext,
    role: ModelRole,
    template_id: str,
    prompt: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, ExecutorOutput | None, BudgetUsage, ExecutionMetrics]:
    """One model call. Returns (parsed_json, failure_output, usage, metrics)."""
    zero = BudgetUsage(), ExecutionMetrics()
    if ctx.model is None:
        out = _fail(inp, FailureKind.MODEL_ERROR, "no model client configured")
        return None, out, *zero
    remaining = ctx.task.caps.tokens - ctx.guard.usage.tokens
    request = GenerationRequest(
        role=role,
        prompt_template_id=template_id,
        prompt_template_version=T.PROMPT_TEMPLATE_VERSION,
        input_text=prompt,
        output_schema=schema,
        seed=derive_seed(ctx.seed, inp.stage_index, inp.attempt),
        max_tokens=max(1, min(MAX_OUTPUT_TOKENS, remaining)),
    )
    try:
        resp = await ctx.model.generate(request)
    except ModelError as exc:
        usage = BudgetUsage(retries=max(0, exc.attempts - 1))
        metrics = ExecutionMetrics(model_calls=exc.attempts, backoff_time_s=exc.backoff_time_s)
        return (
            None,
            _fail(inp, FailureKind.MODEL_ERROR, str(exc), usage=usage, metrics=metrics),
            usage,
            metrics,
        )
    usage = BudgetUsage(tokens=resp.total_tokens, retries=resp.attempts - 1)
    metrics = ExecutionMetrics(
        model_calls=resp.attempts,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        backoff_time_s=resp.backoff_time_s,
    )
    parsed = resp.parsed if resp.parsed is not None else parse_json_object(resp.text)
    if parsed is None:
        out = _fail(
            inp,
            FailureKind.SCHEMA_INVALID,
            "model output is not a JSON object",
            usage=usage,
            metrics=metrics,
        )
        return None, out, usage, metrics
    return parsed, None, usage, metrics


def _originals(inp: ExecutorInput, pages: Pages | None = None) -> dict[str, Page]:
    source = inp.source_pages or (pages.pages if pages else ())
    return {p.page_id: p for p in source}


def _facts_from(
    parsed: Mapping[str, Any],
    task: RuntimeTask,
    search_pages: tuple[Page, ...],
    originals: Mapping[str, Page],
    inherit: Facts | None = None,
) -> Facts:
    """Build Facts from model JSON. Unknown fields are dropped. Spans come from locating the
    quoted text; derived facts without a locatable quote inherit spans from input facts with the
    same field (REASON only)."""
    allowed = task.answer_schema.field_names
    out: list[Fact] = []
    for raw in parsed.get("facts") or []:
        if not isinstance(raw, dict) or raw.get("field") not in allowed or "value" not in raw:
            continue
        spans: tuple[EvidenceSpan, ...] = ()
        quote, page_id = raw.get("quote"), raw.get("page_id")
        if isinstance(quote, str):
            span = locate_quote(
                quote, search_pages, originals, page_id if isinstance(page_id, str) else None
            )
            spans = (span,) if span else ()
        if not spans and inherit is not None:
            spans = next(
                (f.spans for f in inherit.facts if f.field == raw["field"] and f.spans), ()
            )
        out.append(Fact(field=raw["field"], value=raw["value"], spans=spans))
    return Facts(facts=tuple(out))


class ExtractExecutor(StageExecutor):
    kind = StageKind.EXTRACT

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        pages = inp.payload
        assert isinstance(pages, Pages)
        method = inp.stage.method.value
        prompt = T.extract_prompt(method, ctx.task.question, ctx.task.answer_schema, pages.pages)
        parsed, failure, usage, metrics = await generate_json(
            inp, ctx, ModelRole.EXTRACT, f"extract.{method}", prompt, T.EXTRACT_FACTS_SCHEMA
        )
        if failure is not None:
            return failure
        facts = _facts_from(parsed or {}, ctx.task, pages.pages, _originals(inp, pages))
        return ExecutorOutput(payload=facts, usage=usage, metrics=metrics)


class ReasonExecutor(StageExecutor):
    kind = StageKind.REASON

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        facts = inp.payload
        assert isinstance(facts, Facts)
        originals = _originals(inp)
        method = inp.stage.method.value
        prompt = T.reason_prompt(
            method, ctx.task.question, ctx.task.answer_schema, _facts_text(facts, originals)
        )
        parsed, failure, usage, metrics = await generate_json(
            inp, ctx, ModelRole.REASON, f"reason.{method}", prompt, T.FACTS_SCHEMA
        )
        if failure is not None:
            return failure
        derived = _facts_from(
            parsed or {}, ctx.task, tuple(originals.values()), originals, inherit=facts
        )
        return ExecutorOutput(payload=derived, usage=usage, metrics=metrics)


class SynthesizeExecutor(StageExecutor):
    kind = StageKind.SYNTHESIZE

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        facts = inp.payload
        assert isinstance(facts, Facts)
        method = inp.stage.method.value
        prompt = T.synthesize_prompt(
            method,
            ctx.task.question,
            ctx.task.answer_schema,
            _facts_text(facts, _originals(inp), with_quotes=method == "cite_evidence"),
        )
        parsed, failure, usage, metrics = await generate_json(
            inp, ctx, ModelRole.SYNTHESIZE, f"synthesize.{method}", prompt, T.ANSWER_SCHEMA
        )
        if failure is not None:
            return failure
        values = (parsed or {}).get("answer")
        if not isinstance(values, dict):
            return _fail(
                inp,
                FailureKind.SCHEMA_INVALID,
                "model output has no 'answer' object",
                usage=usage,
                metrics=metrics,
            )
        allowed = ctx.task.answer_schema.field_names
        values = {k: v for k, v in values.items() if k in allowed}
        cited = (parsed or {}).get("citations") if method == "cite_evidence" else None
        evidence = _answer_evidence(values, facts, cited if isinstance(cited, dict) else {})
        return ExecutorOutput(
            payload=Answer(values=values, evidence=evidence), usage=usage, metrics=metrics
        )


class DirectExecutor(StageExecutor):
    kind = StageKind.DIRECT

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        assert isinstance(inp.payload, RuntimeTask)
        method = inp.stage.method.value
        prompt = T.direct_prompt(method, ctx.task.question, ctx.task.answer_schema)
        parsed, failure, usage, metrics = await generate_json(
            inp, ctx, ModelRole.SYNTHESIZE, f"direct.{method}", prompt, T.DIRECT_ANSWER_SCHEMA
        )
        if failure is not None:
            return failure
        values = (parsed or {}).get("answer")
        if not isinstance(values, dict):
            return _fail(
                inp,
                FailureKind.SCHEMA_INVALID,
                "model output has no 'answer' object",
                usage=usage,
                metrics=metrics,
            )
        allowed = ctx.task.answer_schema.field_names
        answer = Answer(values={k: v for k, v in values.items() if k in allowed})
        return ExecutorOutput(payload=answer, usage=usage, metrics=metrics)


def _facts_text(facts: Facts, originals: Mapping[str, Page], with_quotes: bool = True) -> str:
    quotes: dict[int, str] = {}
    if with_quotes:
        for i, f in enumerate(facts.facts):
            text = span_text(f.spans[0], originals) if f.spans else None
            if text:
                quotes[i] = text
    return T.render_facts(facts, quotes)


def _answer_evidence(
    values: Mapping[str, Any], facts: Facts, citations: Mapping[str, Any]
) -> tuple[FieldEvidence, ...]:
    """Evidence for each answer field = spans of the facts that back it: cited facts when the
    model cited them (cite_evidence), otherwise facts for that field whose value matches."""
    out: list[FieldEvidence] = []
    for name, value in values.items():
        idx = citations.get(name)
        chosen = [
            facts.facts[i]
            for i in (idx if isinstance(idx, list) else [])
            if isinstance(i, int) and 0 <= i < len(facts.facts) and facts.facts[i].field == name
        ]
        if not chosen:
            chosen = [f for f in facts.facts if f.field == name and _same(f.value, value)]
        spans: list[EvidenceSpan] = []
        for f in chosen:
            spans += [s for s in f.spans if s not in spans]
        if spans:
            out.append(FieldEvidence(field=name, spans=tuple(spans)))
    return tuple(out)


def _same(a: Any, b: Any) -> bool:
    return a == b or (
        isinstance(a, str | int | float) and supports(b, str(a)) and supports(a, str(b))
    )
