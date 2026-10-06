"""The generic deterministic evaluator: run exactly what an ``EvaluationSpec`` declares.

    resolve(spec)                                           -> ResolvedEvaluator  | EvaluationFailed
    evaluate_prediction(spec, schema, expected, predicted)  -> EvaluatorRecord    (never raises)

Selection is driven only by the spec - evaluator kind, pinned version, strict config - never by
the task, the dataset or the benchmark:

  1. kind        a generic kind with an implementation in ``evaluation.metrics.METRICS``.
                 ``legacy_field_match`` is refused: it runs only behind the benchmark compatibility
                 boundary (``benchmarks.legacy_adapter.legacy_evaluator``)
  2. version     the pinned version must equal the implementation's (and the version
                 ``core.evaluation_spec`` pins for the kind); never silently substituted
  3. config      re-validated against the kind's strict schema; unknown keys are rejected
  4. targets     must exist, cover exactly the output fields, and fit the output schema
  5. measure     the implementation, always run (so a bad target is caught even when the
                 prediction is missing); its quality must be finite and in [0, 1]
  6. prediction  missing or schema-invalid -> quality 0, not passed. That is a FAIL of the
                 candidate, not a failure of the evaluator.

Steps 1-5 fail closed: the ``EvaluatorRecord`` has ``failure`` set and no quality, and
``core.results.Evaluation`` refuses to carry a verdict or fitness built from it.

This answers only "how is quality measured?". Hard limits (``ConstraintLimits``), ranking
(``ObjectiveSpec``), fitness shaping and optimizers live elsewhere and only consume the record.
Target values are read here and never copied into a record or an error message.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from core.evaluation_spec import (
    CONFIG_MODELS,
    EVALUATION_SPEC_SCHEMA_VERSION,
    EVALUATOR_VERSIONS,
    EvaluationSpec,
    EvaluatorKind,
)
from core.results import EvaluatorFailure, EvaluatorRecord
from core.task_contract import ContractError
from core.task_spec import AnswerSchema
from evaluation.metrics import METRICS, InvalidTarget, Metric, UnsupportedOutput
from evaluation.schema import validate_answer

Values = Mapping[str, Any]
SpecLike = EvaluationSpec | Mapping[str, Any]
_SPEC_KEYS = frozenset(EvaluationSpec.model_fields)


class EvaluationFailed(ContractError):
    """The evaluator could not judge: carries the failed ``EvaluatorRecord``."""

    def __init__(self, record: EvaluatorRecord) -> None:
        assert record.failure is not None
        super().__init__(
            f"evaluator {record.kind!r} ({record.version or 'no version'}) cannot run: "
            f"{record.failure.value}: {record.detail}"
        )
        self.record = record

    @classmethod
    def of(
        cls,
        kind: str,
        version: str,
        failure: EvaluatorFailure,
        detail: str,
        spec_hash: str | None = None,
    ) -> EvaluationFailed:
        return cls(
            EvaluatorRecord(
                kind=kind[:100],
                version=version[:100],
                spec_hash=spec_hash,
                failure=failure,
                detail=detail[:500],
            )
        )


@dataclass(frozen=True)
class ResolvedEvaluator:
    """A spec bound to its implementation: ready to measure examples."""

    spec: EvaluationSpec  # normalized (defaults filled in)
    config: BaseModel
    metric: Metric

    @property
    def kind(self) -> EvaluatorKind:
        return self.spec.evaluator

    @property
    def version(self) -> str:
        return self.spec.evaluator_version

    @property
    def spec_hash(self) -> str:
        return self.spec.identity_hash

    def record(
        self,
        *,
        quality: float | None = None,
        passed: bool | None = None,
        failure: EvaluatorFailure | None = None,
        detail: str | None = None,
    ) -> EvaluatorRecord:
        return EvaluatorRecord(
            kind=self.kind.value,
            version=self.version,
            spec_hash=self.spec_hash,
            quality=quality,
            passed=passed,
            failure=failure,
            detail=detail,
        )

    def evaluate(
        self, output_schema: AnswerSchema, expected: Values | None, predicted: Values | None
    ) -> EvaluatorRecord:
        """Measure one example. Never raises for evaluator, target or prediction problems."""
        try:
            return self._measure(output_schema, expected, predicted)
        except EvaluationFailed as exc:
            return exc.record

    def _measure(
        self, schema: AnswerSchema, expected: Values | None, predicted: Values | None
    ) -> EvaluatorRecord:
        if self.metric.needs_target:
            self._check_target(schema, expected)
        valid = predicted is not None and not validate_answer(schema, dict(predicted))
        try:
            value, passed = self.metric.measure(
                self.config, schema, expected or {}, predicted if valid else {}
            )
        except InvalidTarget as exc:
            raise self._failed(EvaluatorFailure.INVALID_TARGET, str(exc)) from None
        except UnsupportedOutput as exc:
            raise self._failed(EvaluatorFailure.UNSUPPORTED, str(exc)) from None
        if not (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
            and isinstance(passed, bool)
        ):
            raise self._failed(
                EvaluatorFailure.EVALUATOR_ERROR, "implementation returned an invalid measurement"
            )
        if not valid:
            return self.record(quality=0.0, passed=False)
        return self.record(quality=float(value), passed=passed)

    def _check_target(self, schema: AnswerSchema, expected: Values | None) -> None:
        if expected is None:
            raise self._failed(EvaluatorFailure.MISSING_TARGET, "no expected values")
        if set(expected) != schema.field_names:
            raise self._failed(
                EvaluatorFailure.MISSING_TARGET,
                "expected values must cover exactly the output schema fields",
            )
        problems = validate_answer(schema, {k: v for k, v in expected.items() if v is not None})
        missing = sorted(name for name, problem in problems.items() if problem == "missing")
        if missing:
            raise self._failed(
                EvaluatorFailure.MISSING_TARGET, f"no expected value for required {missing}"
            )
        if problems:
            raise self._failed(
                EvaluatorFailure.INVALID_TARGET,
                f"expected values of {sorted(problems)} do not fit the output schema",
            )

    def _failed(self, failure: EvaluatorFailure, detail: str) -> EvaluationFailed:
        return EvaluationFailed.of(self.kind.value, self.version, failure, detail, self.spec_hash)


def resolve(spec: SpecLike) -> ResolvedEvaluator:
    """Bind ``spec`` to its implementation, or raise ``EvaluationFailed`` (fail closed).

    Accepts a validated ``EvaluationSpec`` or its raw mapping form (e.g. a persisted spec), so an
    unknown kind or a stale version is reported as a record instead of a parse error."""
    raw = spec.model_dump(mode="json") if isinstance(spec, EvaluationSpec) else spec
    if not isinstance(raw, Mapping):
        raise EvaluationFailed.of(
            "", "", EvaluatorFailure.INVALID_CONFIG, "evaluation spec must be a mapping"
        )
    kind_raw, version_raw = raw.get("evaluator"), raw.get("evaluator_version")
    kind_label = kind_raw if isinstance(kind_raw, str) else ""
    version = version_raw if isinstance(version_raw, str) else ""
    try:
        kind = EvaluatorKind(kind_raw)
    except (ValueError, TypeError):
        raise EvaluationFailed.of(
            kind_label, version, EvaluatorFailure.UNKNOWN_KIND, "not a known evaluator kind"
        ) from None
    metric = METRICS.get(kind)
    if metric is None:
        raise EvaluationFailed.of(
            kind.value,
            version,
            EvaluatorFailure.UNSUPPORTED,
            f"{kind.value} is not a generic evaluator; it runs only behind the benchmark "
            "compatibility boundary (benchmarks.legacy_adapter.legacy_evaluator)",
        )
    if not version:
        raise EvaluationFailed.of(
            kind.value, "", EvaluatorFailure.VERSION_MISMATCH, "spec does not pin a version"
        )
    if version != metric.version or metric.version != EVALUATOR_VERSIONS[kind]:
        raise EvaluationFailed.of(
            kind.value,
            version,
            EvaluatorFailure.VERSION_MISMATCH,
            f"spec pins {version!r}, implementation is {metric.version!r}",
        )
    unknown = sorted(str(k) for k in set(raw) - _SPEC_KEYS)
    if unknown:
        raise EvaluationFailed.of(
            kind.value, version, EvaluatorFailure.INVALID_CONFIG, f"unknown spec keys {unknown}"
        )
    if raw.get("schema_version", EVALUATION_SPEC_SCHEMA_VERSION) != EVALUATION_SPEC_SCHEMA_VERSION:
        raise EvaluationFailed.of(
            kind.value, version, EvaluatorFailure.UNSUPPORTED, "unsupported spec schema_version"
        )
    config_raw = raw.get("config", {})
    if not isinstance(config_raw, Mapping):
        raise EvaluationFailed.of(
            kind.value, version, EvaluatorFailure.INVALID_CONFIG, "config must be a mapping"
        )
    try:
        config = CONFIG_MODELS[kind].model_validate(dict(config_raw))
        normalized = EvaluationSpec(
            evaluator=kind, evaluator_version=version, config=config.model_dump(mode="json")
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'config'}: {e['msg']}" for e in exc.errors()
        )
        raise EvaluationFailed.of(
            kind.value, version, EvaluatorFailure.INVALID_CONFIG, problems
        ) from None
    return ResolvedEvaluator(spec=normalized, config=config, metric=metric)


def evaluate_prediction(
    spec: SpecLike,
    output_schema: AnswerSchema,
    expected: Values | None,
    predicted: Values | None,
) -> EvaluatorRecord:
    """Measure one prediction against its target exactly as ``spec`` declares.

    The one entry point for callers outside the optimization pipeline (e.g. a runtime-authority
    service): pure, deterministic, never raises for evaluator problems - check ``record.ok``."""
    try:
        return resolve(spec).evaluate(output_schema, expected, predicted)
    except EvaluationFailed as exc:
        return exc.record
