"""Evaluator IMPLEMENTATIONS for ``core.evaluation_spec.EvaluationSpec`` (configuration).

Pure, deterministic, no LLM, no I/O. Each implementation scores one example: the expected output
values (from the dataset's target columns) against the predicted output values. Lookup is by
(kind, version) and fails closed on a mismatch - an evaluator is never silently substituted.

``legacy_field_match`` is not implemented here: it needs evidence verification against snapshot
bytes and is executed by ``evaluation.gate.DeterministicEvaluator`` (same version string).
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
    EVALUATOR_VERSIONS,
    ClassificationAccuracyConfig,
    EvaluationSpec,
    EvaluatorKind,
    ExactMatchConfig,
    NumericToleranceConfig,
    TokenF1Config,
)
from core.task_spec import AnswerSchema
from evaluation.schema import validate_answer


@dataclass(frozen=True)
class MetricResult:
    score: float  # in [0, 1]
    passed: bool
    evaluator_version: str


Values = Mapping[str, Any]
MetricFn = Callable[[EvaluationSpec, AnswerSchema, Values, Values], tuple[float, bool]]


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _norm_text(value: str, cfg: ExactMatchConfig) -> str:
    if cfg.normalize_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
    return value if cfg.case_sensitive else value.casefold()


def _fraction(flags: list[bool]) -> tuple[float, bool]:
    return sum(flags) / len(flags), all(flags)


def _exact_match(spec, schema, expected, predicted):
    cfg: ExactMatchConfig = spec.typed_config()

    def same(want: Any, got: Any) -> bool:
        if isinstance(want, str) and isinstance(got, str):
            return _norm_text(want, cfg) == _norm_text(got, cfg)
        return type(want) is type(got) and want == got

    return _fraction([same(expected[f.name], predicted.get(f.name)) for f in schema.fields])


def _single_field(schema: AnswerSchema) -> str:
    if len(schema.fields) != 1:
        raise ValueError("this evaluator needs exactly one output field")
    return schema.fields[0].name


def _classification_accuracy(spec, schema, expected, predicted):
    cfg: ClassificationAccuracyConfig = spec.typed_config()
    name = _single_field(schema)

    def fold(v: str) -> str:
        return v if cfg.case_sensitive else v.casefold()

    labels = {fold(label) for label in cfg.labels}
    want, got = expected[name], predicted.get(name)
    if not isinstance(want, str) or fold(want) not in labels:
        raise ValueError(f"target label {want!r} is not one of the configured labels")
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


def _token_f1(spec, schema, expected, predicted):
    cfg: TokenF1Config = spec.typed_config()
    name = _single_field(schema)
    want, got = expected[name], predicted.get(name)
    if not isinstance(want, str):
        raise ValueError("token_f1 needs a string target")
    f1 = token_f1(want, got) if isinstance(got, str) else 0.0
    return f1, f1 >= cfg.pass_threshold


def _json_schema_validity(spec, schema, expected, predicted):
    return 1.0, True  # schema validity is checked for every evaluator in ``score``


def _numeric_tolerance(spec, schema, expected, predicted):
    cfg: NumericToleranceConfig = spec.typed_config()

    def close(want: Any, got: Any) -> bool:
        if not _is_number(want):
            raise ValueError("numeric_tolerance needs numeric targets")
        if not _is_number(got):
            return False
        bound = max(cfg.absolute_tolerance, cfg.relative_tolerance * abs(want))
        return abs(got - want) <= bound

    return _fraction([close(expected[f.name], predicted.get(f.name)) for f in schema.fields])


METRICS: dict[EvaluatorKind, tuple[str, MetricFn]] = {
    EvaluatorKind.EXACT_MATCH: ("exact_match/1", _exact_match),
    EvaluatorKind.CLASSIFICATION_ACCURACY: ("classification_accuracy/1", _classification_accuracy),
    EvaluatorKind.TOKEN_F1: ("token_f1/1", _token_f1),
    EvaluatorKind.JSON_SCHEMA_VALIDITY: ("json_schema_validity/1", _json_schema_validity),
    EvaluatorKind.NUMERIC_TOLERANCE: ("numeric_tolerance/1", _numeric_tolerance),
}


class EvaluatorUnavailable(LookupError):
    pass


def get_metric(spec: EvaluationSpec) -> MetricFn:
    entry = METRICS.get(spec.evaluator)
    if entry is None:
        raise EvaluatorUnavailable(
            f"{spec.evaluator.value} is not a per-example metric "
            "(legacy_field_match runs in evaluation.gate.DeterministicEvaluator)"
        )
    version, fn = entry
    if version != spec.evaluator_version or version != EVALUATOR_VERSIONS[spec.evaluator]:
        raise EvaluatorUnavailable(
            f"{spec.evaluator.value}: spec pins {spec.evaluator_version!r}, "
            f"implementation is {version!r}"
        )
    return fn


def score(
    spec: EvaluationSpec,
    output_schema: AnswerSchema,
    expected: Values,
    predicted: Values | None,
) -> MetricResult:
    """Score one example. A missing or schema-invalid prediction scores 0 and fails."""
    fn = get_metric(spec)
    if set(expected) != output_schema.field_names:
        raise ValueError("expected values must cover exactly the output schema fields")
    if predicted is None or validate_answer(output_schema, dict(predicted)):
        return MetricResult(0.0, False, spec.evaluator_version)
    value, passed = fn(spec, output_schema, expected, predicted)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{spec.evaluator.value} produced out-of-range score {value}")
    return MetricResult(float(value), bool(passed), spec.evaluator_version)
