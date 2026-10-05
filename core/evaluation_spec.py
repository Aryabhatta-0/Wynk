"""How outputs are evaluated: evaluator CONFIGURATION, separate from evaluator IMPLEMENTATION.

``EvaluationSpec(evaluator="numeric_tolerance", config={"absolute_tolerance": 0.01})`` names an
evaluator kind, the exact implementation version it must run under, and a config validated
against that kind's own strict schema. Implementations live in ``evaluation/metrics.py`` and are
looked up by (kind, version); a version mismatch is an error, never a silent substitution.

The evaluator is external authority: no kind here asks a model to judge its own output.

Adding an evaluator = one ``EvaluatorKind`` member + one config model + one version entry here,
and one implementation registered in ``evaluation/metrics.py`` (a test checks the two agree).
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from core.canonical import canonical_hash
from core.task_spec import MatcherConfig

EVALUATION_SPEC_SCHEMA_VERSION = "evaluationspec/1"


class EvaluatorKind(StrEnum):
    EXACT_MATCH = "exact_match"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    TOKEN_F1 = "token_f1"
    JSON_SCHEMA_VALIDITY = "json_schema_validity"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    # The frozen benchmark's per-field matchers + evidence check (evaluation.gate). It is only
    # valid on the legacy snapshot dataset format; see core.task_contract.
    LEGACY_FIELD_MATCH = "legacy_field_match"


# The implementation version each kind currently runs under. Bump when scoring semantics change.
EVALUATOR_VERSIONS: dict[EvaluatorKind, str] = {
    EvaluatorKind.EXACT_MATCH: "exact_match/1",
    EvaluatorKind.CLASSIFICATION_ACCURACY: "classification_accuracy/1",
    EvaluatorKind.TOKEN_F1: "token_f1/1",
    EvaluatorKind.JSON_SCHEMA_VALIDITY: "json_schema_validity/1",
    EvaluatorKind.NUMERIC_TOLERANCE: "numeric_tolerance/1",
    EvaluatorKind.LEGACY_FIELD_MATCH: "evaluator/mvp-2",  # == evaluation.gate.EVALUATOR_VERSION
}


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExactMatchConfig(_Config):
    """Type-strict equality. For text, optional case folding / whitespace collapsing."""

    case_sensitive: bool = True
    normalize_whitespace: bool = False


class ClassificationAccuracyConfig(_Config):
    """The prediction must be one of ``labels``; correct iff equal to the target label."""

    labels: tuple[str, ...] = Field(min_length=2)
    case_sensitive: bool = True

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not label.strip() for label in v):
            raise ValueError("labels must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("labels must be unique")
        return v


class TokenF1Config(_Config):
    """SQuAD-style token F1 over normalized text. PASS iff F1 >= ``pass_threshold``."""

    pass_threshold: float = Field(default=1.0, gt=0.0, le=1.0)


class JsonSchemaValidityConfig(_Config):
    """Valid iff the output satisfies the task's ``output_schema``. No options."""


class NumericToleranceConfig(_Config):
    """``|predicted - target| <= max(absolute_tolerance, relative_tolerance * |target|)``."""

    absolute_tolerance: NonNegativeFloat = Field(default=0.0, allow_inf_nan=False)
    relative_tolerance: NonNegativeFloat = Field(default=0.0, allow_inf_nan=False)


class LegacyFieldMatchConfig(_Config):
    """Per-field matchers of the frozen benchmark (no expected values here)."""

    matchers: dict[str, MatcherConfig] = Field(min_length=1)
    require_evidence: bool = True

    @field_validator("matchers")
    @classmethod
    def _finite(cls, v: dict[str, MatcherConfig]) -> dict[str, MatcherConfig]:
        for name, m in v.items():
            for tol in (m.abs_tol, m.rel_tol):
                if tol is not None and not math.isfinite(tol):
                    raise ValueError(f"matcher tolerance for {name!r} must be finite")
        return v


CONFIG_MODELS: dict[EvaluatorKind, type[_Config]] = {
    EvaluatorKind.EXACT_MATCH: ExactMatchConfig,
    EvaluatorKind.CLASSIFICATION_ACCURACY: ClassificationAccuracyConfig,
    EvaluatorKind.TOKEN_F1: TokenF1Config,
    EvaluatorKind.JSON_SCHEMA_VALIDITY: JsonSchemaValidityConfig,
    EvaluatorKind.NUMERIC_TOLERANCE: NumericToleranceConfig,
    EvaluatorKind.LEGACY_FIELD_MATCH: LegacyFieldMatchConfig,
}


class EvaluationSpec(BaseModel):
    """Versioned evaluator configuration. ``config`` is stored in normalized form (defaults
    filled in), so two specs that mean the same thing serialize and hash identically."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["evaluationspec/1"] = EVALUATION_SPEC_SCHEMA_VERSION
    evaluator: EvaluatorKind
    evaluator_version: str = ""  # empty -> pinned to the current version at construction
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_config(self) -> EvaluationSpec:
        current = EVALUATOR_VERSIONS[self.evaluator]
        if not self.evaluator_version:
            object.__setattr__(self, "evaluator_version", current)
        elif self.evaluator_version != current:
            raise ValueError(
                f"{self.evaluator.value} is implemented at {current!r}, "
                f"spec requires {self.evaluator_version!r}"
            )
        parsed = CONFIG_MODELS[self.evaluator].model_validate(self.config)
        object.__setattr__(self, "config", parsed.model_dump(mode="json"))
        return self

    def typed_config(self) -> _Config:
        return CONFIG_MODELS[self.evaluator].model_validate(self.config)

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
