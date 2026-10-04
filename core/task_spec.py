"""Task contracts, split by authority.

``RuntimeTask``  - everything the optimizer / compiler / runtime / Gemma may see.
``TaskSpec``     - ``RuntimeTask`` + ground truth + matcher config. OFFLINE ONLY.

Ground truth must never reach the runtime. The separation is structural, not a naming
convention: runtime-side APIs accept only ``RuntimeTask`` (composition, not inheritance, so a
``TaskSpec`` is NOT a ``RuntimeTask``), ``GroundTruth`` has a redacted repr, and
``tests/test_authority_boundaries.py`` fails if runtime-side modules import this file's offline
types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from core.canonical import canonical_hash, canonical_json
from core.stages import GatherSource

TASK_SPEC_SCHEMA_VERSION = "taskspec/1"


class TaskClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class FieldType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    STRING_LIST = "string_list"


class AnswerField(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    type: FieldType
    required: bool = True


class AnswerSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: tuple[AnswerField, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> AnswerSchema:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("answer schema field names must be unique")
        return self

    @property
    def field_names(self) -> frozenset[str]:
        return frozenset(f.name for f in self.fields)


class Caps(BaseModel):
    """Hard per-run limits. Crossing any cap makes the run INFEASIBLE (decided by code)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: PositiveInt
    wall_time_s: PositiveFloat
    tool_calls: PositiveInt
    retries: NonNegativeInt


class MatcherKind(StrEnum):
    EXACT = "exact"
    NORMALIZED_TEXT = "normalized_text"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    DATE = "date"
    SET_EQUAL = "set_equal"


class MatcherConfig(BaseModel):
    """How the offline evaluator compares one answer field. Contains no ground truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: MatcherKind
    abs_tol: NonNegativeFloat | None = None
    rel_tol: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _tolerances_only_for_numeric(self) -> MatcherConfig:
        if self.kind is not MatcherKind.NUMERIC_TOLERANCE and (
            self.abs_tol is not None or self.rel_tol is not None
        ):
            raise ValueError("tolerances are only valid for numeric_tolerance matchers")
        return self


class GroundTruth(BaseModel):
    """Expected answer values. Offline evaluation only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: dict[str, Any]

    def __repr__(self) -> str:  # never leak values into logs/tracebacks
        return "GroundTruth(<redacted>)"

    __str__ = __repr__


class RuntimeTask(BaseModel):
    """The only task view that optimizer/compiler/runtime/Gemma code may hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    task_class: TaskClass
    question: str = Field(min_length=1)
    answer_schema: AnswerSchema
    caps: Caps
    allowed_sources: tuple[GatherSource, ...] = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    interaction_required: bool = False

    @model_validator(mode="after")
    def _source_configuration(self) -> RuntimeTask:
        if len(set(self.allowed_sources)) != len(self.allowed_sources):
            raise ValueError("allowed_sources must not repeat a source")
        if self.interaction_required and GatherSource.JEV not in self.allowed_sources:
            raise ValueError("interaction_required needs 'jev' in allowed_sources")
        return self


class TaskSpec(BaseModel):
    """Full benchmark task: runtime view + offline-only evaluation data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: RuntimeTask
    ground_truth: GroundTruth
    matchers: dict[str, MatcherConfig]

    @model_validator(mode="after")
    def _fields_consistent(self) -> TaskSpec:
        names = self.runtime.answer_schema.field_names
        if set(self.ground_truth.values) != set(names):
            raise ValueError("ground_truth keys must equal answer_schema field names")
        if set(self.matchers) != set(names):
            raise ValueError("matchers keys must equal answer_schema field names")
        return self

    # flat convenience accessors
    @property
    def id(self) -> str:
        return self.runtime.id

    @property
    def caps(self) -> Caps:
        return self.runtime.caps

    def runtime_view(self) -> RuntimeTask:
        """The ONLY thing that may be handed to optimizer/compiler/runtime code."""
        return self.runtime

    def canonical_json(self) -> str:
        return canonical_json({"schema": TASK_SPEC_SCHEMA_VERSION, **self.model_dump(mode="json")})

    @property
    def content_hash(self) -> str:
        return canonical_hash({"schema": TASK_SPEC_SCHEMA_VERSION, **self.model_dump(mode="json")})
