"""Evaluator IMPLEMENTATIONS for ``core.evaluation_spec.EvaluationSpec`` (configuration).

Pure, deterministic, no LLM, no I/O. Each implementation measures one example: the expected
output values (from the dataset's target columns) against the predicted output values, under the
kind's already-validated config. It returns ``(quality in [0, 1], passed)``.

These functions are only reached through ``evaluation.dispatch``, which resolves the spec's kind
and pinned version against ``METRICS``, validates the config strictly, checks the targets and the
prediction's shape, and records the outcome. A kind absent from ``METRICS`` (``legacy_field_match``
included) is not a generic evaluator: the dispatcher refuses it.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from core.evaluation_spec import (
    ClassificationAccuracyConfig,
    EvaluatorKind,
    ExactMatchConfig,
    NumericToleranceConfig,
    TokenF1Config,
)
from core.task_spec import AnswerSchema

Values = Mapping[str, Any]
Measure = Callable[[Any, AnswerSchema, Values, Values], tuple[float, bool]]


class InvalidTarget(ValueError):
    """The expected value cannot be judged under this config (e.g. a label outside the label
    set). Messages never include the value itself."""


class UnsupportedOutput(ValueError):
    """The kind cannot judge this output schema (e.g. token_f1 over several fields)."""


@dataclass(frozen=True)
class Metric:
    version: str
    measure: Measure
    needs_target: bool = True  # False: the kind judges the prediction alone


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _norm_text(value: str, cfg: ExactMatchConfig) -> str:
    if cfg.normalize_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
    return value if cfg.case_sensitive else value.casefold()


def _fraction(flags: list[bool]) -> tuple[float, bool]:
    return sum(flags) / len(flags), all(flags)


def _single_field(schema: AnswerSchema) -> str:
    if len(schema.fields) != 1:
        raise UnsupportedOutput("this evaluator needs exactly one output field")
    return schema.fields[0].name


def _exact_match(cfg: ExactMatchConfig, schema, expected, predicted):
    def same(want: Any, got: Any) -> bool:
        if isinstance(want, str) and isinstance(got, str):
            return _norm_text(want, cfg) == _norm_text(got, cfg)
        return type(want) is type(got) and want == got

    return _fraction([same(expected[f.name], predicted.get(f.name)) for f in schema.fields])


def _classification_accuracy(cfg: ClassificationAccuracyConfig, schema, expected, predicted):
    name = _single_field(schema)

    def fold(v: str) -> str:
        return v if cfg.case_sensitive else v.casefold()

    labels = {fold(label) for label in cfg.labels}
    want, got = expected[name], predicted.get(name)
    if not isinstance(want, str) or fold(want) not in labels:
        raise InvalidTarget("target label is not one of the configured labels")
    ok = isinstance(got, str) and fold(got) in labels and fold(got) == fold(want)
    return (1.0 if ok else 0.0), ok


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def answer_tokens(text: str) -> list[str]:
    """SQuAD normalization: lowercase, drop punctuation and articles, split on whitespace."""
    return _ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split()


def token_f1(expected: str, predicted: str) -> float:
    want, got = answer_tokens(expected), answer_tokens(predicted)
    if not want or not got:
        return float(want == got)
    common = sum((Counter(want) & Counter(got)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(got), common / len(want)
    return 2 * precision * recall / (precision + recall)


def _token_f1(cfg: TokenF1Config, schema, expected, predicted):
    name = _single_field(schema)
    want, got = expected[name], predicted.get(name)
    if not isinstance(want, str):
        raise InvalidTarget("token_f1 needs a string target")
    f1 = token_f1(want, got) if isinstance(got, str) else 0.0
    return f1, f1 >= cfg.pass_threshold


def _json_schema_validity(cfg, schema, expected, predicted):
    # The dispatcher has already scored a schema-invalid prediction 0; reaching here means valid.
    return 1.0, True


def _numeric_tolerance(cfg: NumericToleranceConfig, schema, expected, predicted):
    def close(want: Any, got: Any) -> bool:
        if not _is_number(want):
            raise InvalidTarget("numeric_tolerance needs numeric targets")
        if not _is_number(got):
            return False
        bound = max(cfg.absolute_tolerance, cfg.relative_tolerance * abs(want))
        return abs(got - want) <= bound

    return _fraction([close(expected[f.name], predicted.get(f.name)) for f in schema.fields])


# kind -> the one implementation of that kind, at the version it implements.
METRICS: dict[EvaluatorKind, Metric] = {
    EvaluatorKind.EXACT_MATCH: Metric("exact_match/1", _exact_match),
    EvaluatorKind.CLASSIFICATION_ACCURACY: Metric(
        "classification_accuracy/1", _classification_accuracy
    ),
    EvaluatorKind.TOKEN_F1: Metric("token_f1/1", _token_f1),
    EvaluatorKind.JSON_SCHEMA_VALIDITY: Metric(
        "json_schema_validity/1", _json_schema_validity, needs_target=False
    ),
    EvaluatorKind.NUMERIC_TOLERANCE: Metric("numeric_tolerance/1", _numeric_tolerance),
}
