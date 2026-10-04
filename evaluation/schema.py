"""Deterministic answer-schema validation (types, required fields, no unknown fields)."""

from __future__ import annotations

import math
from typing import Any

from core.task_spec import AnswerSchema, FieldType
from evaluation.matchers import parse_date


def _type_ok(field_type: FieldType, value: Any) -> bool:
    if field_type is FieldType.STRING:
        return isinstance(value, str)
    if field_type is FieldType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type is FieldType.NUMBER:
        return (
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        )
    if field_type is FieldType.BOOLEAN:
        return isinstance(value, bool)
    if field_type is FieldType.DATE:
        return parse_date(value) is not None
    if field_type is FieldType.STRING_LIST:
        return isinstance(value, list | tuple) and all(isinstance(x, str) for x in value)
    raise ValueError(f"unknown field type {field_type!r}")


def validate_answer(schema: AnswerSchema, values: dict[str, Any]) -> dict[str, str]:
    """``{field: problem}`` for every schema violation; empty dict means schema-valid.

    Unknown fields are reported under their own name.
    """
    problems: dict[str, str] = {}
    for f in schema.fields:
        if f.name not in values:
            if f.required:
                problems[f.name] = "missing"
        elif not _type_ok(f.type, values[f.name]):
            problems[f.name] = f"expected {f.type.value}"
    for name in values:
        if name not in schema.field_names:
            problems[name] = "unknown field"
    return problems
