"""Versioned prompt templates for the Gemma stages (EXTRACT / REASON / SYNTHESIZE).

Bump ``PROMPT_TEMPLATE_VERSION`` whenever any text or schema here changes: it is part of every
run's identity (``RunVersions.prompt_template_version``) and of every model cache key.

Prompts never contain ground truth and never ask the model to judge correctness.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from core.payloads import Fact, Facts, Page
from core.task_spec import AnswerSchema

PROMPT_TEMPLATE_VERSION = "mvp-2"
MAX_PAGE_CHARS = 24_000  # total page text sent to the model


def _facts_schema(*, sourced: bool) -> dict[str, Any]:
    """Facts list schema. ``value`` is a scalar (an untyped value let constrained decoders fold
    page_id/quote into one string). Extracted facts must carry page_id + quote; derived facts
    (REASON) may omit them."""
    required = ["field", "value", "page_id", "quote"] if sourced else ["field", "value"]
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},  # extract.cot
            "steps": {"type": "array", "items": {"type": "string"}},  # reason.decompose
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"type": ["string", "number", "integer", "boolean"]},
                        "page_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts"],
    }


EXTRACT_FACTS_SCHEMA = _facts_schema(sourced=True)
FACTS_SCHEMA = _facts_schema(sourced=False)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "object"},
        "citations": {"type": "object"},
    },
    "required": ["answer"],
}

_FACT_FORMAT = (
    'Return ONLY JSON: {"facts": [{"field": <field name>, "value": <value>, '
    '"page_id": <page id>, "quote": <exact text copied from that page>}]}. '
    "The quote must be copied verbatim from the page so it can be located."
)


def _fields(schema: AnswerSchema) -> str:
    return "\n".join(
        f"- {f.name} ({f.type.value}{', required' if f.required else ''})" for f in schema.fields
    )


def render_pages(pages: Sequence[Page], max_chars: int = MAX_PAGE_CHARS) -> str:
    out, used = [], 0
    for p in pages:
        room = max_chars - used
        if room <= 0:
            break
        text = p.content[:room]
        out.append(f"[page {p.page_id}]\n{text}")
        used += len(text)
    return "\n\n".join(out)


def render_facts(facts: Facts, quotes: dict[int, str] | None = None) -> str:
    lines = []
    for i, f in enumerate(facts.facts):
        quote = (quotes or {}).get(i)
        tail = f'  (evidence: "{quote}")' if quote else ""
        lines.append(f"[{i}] {f.field} = {json.dumps(f.value)}{tail}")
    return "\n".join(lines)


_EXTRACT_MODE = {
    "direct": "",
    "schema_guided": "Value types must match the field types exactly. Omit a field the pages "
    "do not state.",
    "cot": 'First think step by step in an optional "reasoning" string, then give the facts.',
}


def extract_prompt(method: str, question: str, schema: AnswerSchema, pages: Sequence[Page]) -> str:
    return (
        "You extract facts from source pages. Use ONLY the pages below. Do not guess.\n"
        f"Question: {question}\nFields to extract:\n{_fields(schema)}\n"
        f"{_FACT_FORMAT}\n{_EXTRACT_MODE[method]}\n\nPages:\n{render_pages(pages)}"
    )


_REASON_MODE = {
    "single": "",
    "decompose": 'First break the question into sub-steps in an optional "steps" list, '
    "then give the final facts.",
}


def reason_prompt(method: str, question: str, schema: AnswerSchema, facts_text: str) -> str:
    return (
        "You are given facts extracted from source pages. Derive the facts needed to answer the "
        "question (combine, compare, compute). Use ONLY the given facts.\n"
        f"Question: {question}\nFields needed:\n{_fields(schema)}\nFacts:\n{facts_text}\n"
        f"{_FACT_FORMAT} Omit page_id and quote for derived values that have no single source "
        f"text.\n{_REASON_MODE[method]}"
    )


def synthesize_prompt(method: str, question: str, schema: AnswerSchema, facts_text: str) -> str:
    cite = (
        ' Also return "citations": {<field>: [<indices of the facts you used>]}.'
        if method == "cite_evidence"
        else ""
    )
    return (
        "Answer the question using ONLY the facts below.\n"
        f"Question: {question}\nAnswer fields (use these JSON types):\n{_fields(schema)}\n"
        f"Facts:\n{facts_text}\n"
        f'Return ONLY JSON: {{"answer": {{<field>: <value>}}}}.{cite}'
    )


def facts_to_json(facts: Sequence[Fact]) -> str:  # for debugging / traces
    return json.dumps([{"field": f.field, "value": f.value} for f in facts], sort_keys=True)
