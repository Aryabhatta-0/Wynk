"""What "better" means for a task: the optimization objective, and the measurements it reads.

``ObjectiveSpec`` is an optimization PREFERENCE - it ranks candidates that are already feasible.
Hard limits (``core.constraints.ConstraintLimits``) are a separate concept: a candidate that
violates one is infeasible, and no objective value can make it win (``core.task_contract``).

Ranking returns a tuple sort key, higher is better:

    maximize_quality   (quality,)
    minimize_cost      (-mean_cost_per_example, quality)
    minimize_latency   (-mean_latency_s, quality)
    balanced           (w_q * quality - sum_m w_m * value_m / scale_m,)

``balanced`` weights must be finite, non-negative and sum to exactly 1 (within 1e-9); quality must
keep a positive weight, and every penalized metric needs an explicit positive ``scale`` (the
value that costs its full weight), so nothing is normalized behind the caller's back. The default
objective is ``maximize_quality``. Any metric the objective reads must be measured: a missing one
raises ``MissingMetric`` instead of being treated as zero. A scalarized key is used for now; the
key is a function of ``CandidateMeasurements`` only, so a Pareto ranking can replace it later.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from core.canonical import canonical_hash

OBJECTIVE_SPEC_SCHEMA_VERSION = "objectivespec/1"
WEIGHT_SUM_TOLERANCE = 1e-9


class MissingMetric(ValueError):
    """The objective or a hard limit needs a measurement the candidate does not have."""


class CandidateMeasurements(BaseModel):
    """Aggregate measurements of one candidate over the examples it was evaluated on.

    ``None`` means "not measured" - never "zero". Means/p95 are over examples; ``max_*`` are the
    worst single example (per-run hard caps apply to every run, so the worst one decides).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quality: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mean_cost_per_example: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)
    mean_latency_s: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)
    p95_latency_s: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)
    mean_tokens_per_example: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)
    max_tokens_per_example: NonNegativeInt | None = None
    workflow_steps: NonNegativeInt | None = None
    max_model_calls_per_example: NonNegativeInt | None = None
    max_tool_calls_per_example: NonNegativeInt | None = None
    max_retries_per_example: NonNegativeInt | None = None
    max_wall_time_s_per_example: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)

    def require(self, field: str) -> float:
        value = getattr(self, field)
        if value is None:
            raise MissingMetric(f"measurement {field!r} is required but was not measured")
        return float(value)


class ObjectiveMode(StrEnum):
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCED = "balanced"


class Metric(StrEnum):
    QUALITY = "quality"
    COST = "cost"
    LATENCY = "latency"
    TOKENS = "tokens"


# objective metric -> the measurement it reads
METRIC_FIELDS: dict[Metric, str] = {
    Metric.QUALITY: "quality",
    Metric.COST: "mean_cost_per_example",
    Metric.LATENCY: "mean_latency_s",
    Metric.TOKENS: "mean_tokens_per_example",
}


class ObjectiveSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["objectivespec/1"] = OBJECTIVE_SPEC_SCHEMA_VERSION
    mode: ObjectiveMode = ObjectiveMode.MAXIMIZE_QUALITY
    weights: dict[Metric, float] = Field(default_factory=dict)  # balanced only
    scales: dict[Metric, float] = Field(default_factory=dict)  # balanced only

    @field_validator("weights", "scales")
    @classmethod
    def _finite_non_negative(cls, v: dict[Metric, float]) -> dict[Metric, float]:
        for metric, value in v.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{metric.value}: values must be finite and non-negative")
        return v

    @model_validator(mode="after")
    def _mode_rules(self) -> ObjectiveSpec:
        if self.mode is not ObjectiveMode.BALANCED:
            if self.weights or self.scales:
                raise ValueError("weights and scales are only meaningful for mode 'balanced'")
            return self
        if not self.weights:
            raise ValueError("balanced objective needs explicit weights")
        if abs(sum(self.weights.values()) - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError("balanced weights must sum to 1")
        if self.weights.get(Metric.QUALITY, 0.0) <= 0.0:
            raise ValueError("balanced objective must give quality a positive weight")
        penalized = {m for m, w in self.weights.items() if m is not Metric.QUALITY and w > 0}
        if set(self.scales) != penalized:
            raise ValueError(
                "scales must be given for exactly the penalized metrics: "
                + ", ".join(sorted(m.value for m in penalized))
            )
        if any(s <= 0 for s in self.scales.values()):
            raise ValueError("scales must be positive")
        return self

    def required_metrics(self) -> tuple[Metric, ...]:
        if self.mode is ObjectiveMode.MAXIMIZE_QUALITY:
            return (Metric.QUALITY,)
        if self.mode is ObjectiveMode.MINIMIZE_COST:
            return (Metric.COST, Metric.QUALITY)
        if self.mode is ObjectiveMode.MINIMIZE_LATENCY:
            return (Metric.LATENCY, Metric.QUALITY)
        return tuple(sorted(m for m, w in self.weights.items() if w > 0))

    def rank_key(self, m: CandidateMeasurements) -> tuple[float, ...]:
        """Higher is better. Raises ``MissingMetric`` if a required measurement is absent."""
        values = {metric: m.require(METRIC_FIELDS[metric]) for metric in self.required_metrics()}
        if self.mode is ObjectiveMode.MAXIMIZE_QUALITY:
            return (values[Metric.QUALITY],)
        if self.mode is ObjectiveMode.MINIMIZE_COST:
            return (-values[Metric.COST], values[Metric.QUALITY])
        if self.mode is ObjectiveMode.MINIMIZE_LATENCY:
            return (-values[Metric.LATENCY], values[Metric.QUALITY])
        utility = self.weights[Metric.QUALITY] * values[Metric.QUALITY]
        for metric, scale in sorted(self.scales.items()):
            utility -= self.weights[metric] * values[metric] / scale
        return (utility,)

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
