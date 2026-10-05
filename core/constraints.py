"""The single authoritative hard-constraint layer.

Reused unchanged by ACO, random search, exhaustive search, PBIL, bandit and hand-built
workflows. Optimizers must NOT re-implement any of these rules; they call ``check`` or
``admissible_successors``. Everything here is deterministic and LLM-free.

Two kinds of hard constraint live here:
  * structural / pre-execution: ``ConstraintChecker`` (grammar, stage rules, provable budget);
  * measured / post-evaluation: ``ConstraintLimits`` + ``check_limits`` (quality floor, cost,
    latency, per-example run caps). A violation makes the candidate infeasible; it is never a
    weighted penalty (preferences live in ``core.objective.ObjectiveSpec``).
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from core.canonical import canonical_hash
from core.cost_model import CostModel, StaticCostModel, exceeded_caps
from core.genome import Genome
from core.grammar import Grammar, GrammarError
from core.objective import CandidateMeasurements
from core.stages import GatherMode, GatherSource, StageSpec, VerifyMethod
from core.task_spec import Caps, RuntimeTask
from core.violations import Violation, ViolationCode

CONSTRAINTS_VERSION = "constraints/1"
CONSTRAINT_LIMITS_SCHEMA_VERSION = "constraintlimits/1"


class ConstraintConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_active_verifiers: PositiveInt = 2
    max_self_consistency: PositiveInt = 1
    unavailable_sources: tuple[GatherSource, ...] = ()
    unavailable_verifiers: tuple[VerifyMethod, ...] = ()


class ConstraintChecker:
    def __init__(
        self,
        grammar: Grammar | None = None,
        cost_model: CostModel | None = None,
        config: ConstraintConfig | None = None,
    ) -> None:
        self.grammar = grammar or Grammar()
        self.cost_model = cost_model or StaticCostModel()
        self.config = config or ConstraintConfig()

    def check(
        self, genome: Genome, task: RuntimeTask | None = None, *, complete: bool = True
    ) -> tuple[Violation, ...]:
        """All violations of ``genome``. ``complete=False`` accepts partial genomes (prefix
        checking): every rule below is monotone, so a prefix violation can never be repaired
        by appending more stages."""
        violations = list(self.grammar.validate(genome, complete=complete))
        violations += self._stage_rules(genome)
        if task is not None:
            violations += self._task_rules(genome, task)
        return tuple(violations)

    def is_valid(
        self, genome: Genome, task: RuntimeTask | None = None, *, complete: bool = True
    ) -> bool:
        return not self.check(genome, task, complete=complete)

    def admissible_successors(self, partial: Genome, task: RuntimeTask) -> tuple[StageSpec, ...]:
        """Stages that may be appended to ``partial`` such that the prefix stays valid for
        ``task``: grammar-legal AND constraint-clean AND still within provable budget."""
        try:
            candidates = self.grammar.valid_successor_specs(partial)
        except GrammarError:
            return ()
        return tuple(
            s for s in candidates if self.is_valid(partial.extend(s), task, complete=False)
        )

    # -- rules ----------------------------------------------------------------
    def _stage_rules(self, genome: Genome) -> list[Violation]:
        out: list[Violation] = []
        for i, stage in enumerate(genome.stages):
            if (stage.kind == "GATHER" and stage.source in self.config.unavailable_sources) or (
                stage.kind == "VERIFY" and stage.method in self.config.unavailable_verifiers
            ):
                out.append(
                    Violation(
                        code=ViolationCode.RUNTIME_UNAVAILABLE,
                        message="stage option is not implemented by this runtime",
                        stage_index=i,
                    )
                )
        gather = next((s for s in genome.stages if s.kind == "GATHER"), None)
        jev = gather is not None and gather.source == GatherSource.JEV
        if jev and gather.mode == GatherMode.PARALLEL_4:
            out.append(
                Violation(
                    code=ViolationCode.JEV_PARALLEL_4,
                    message="GATHER source jev cannot use mode parallel-4",
                    stage_index=0,
                )
            )
        verifiers = [(i, s) for i, s in enumerate(genome.stages) if s.kind == "VERIFY"]
        if len(verifiers) > self.config.max_active_verifiers:
            out.append(
                Violation(
                    code=ViolationCode.TOO_MANY_VERIFIERS,
                    message=f"{len(verifiers)} verifiers; at most "
                    f"{self.config.max_active_verifiers} allowed",
                    stage_index=verifiers[self.config.max_active_verifiers][0],
                )
            )
        consistency = [i for i, s in verifiers if s.method == VerifyMethod.SELF_CONSISTENCY]
        if len(consistency) > self.config.max_self_consistency:
            out.append(
                Violation(
                    code=ViolationCode.SELF_CONSISTENCY_REPEATED,
                    message="self_consistency may appear at most "
                    f"{self.config.max_self_consistency} time(s)",
                    stage_index=consistency[self.config.max_self_consistency],
                )
            )
        if jev:
            for i, s in verifiers:
                if s.on_failure.value == "regather":
                    out.append(
                        Violation(
                            code=ViolationCode.JEV_REGATHER,
                            message="regather cannot be used when GATHER source is jev",
                            stage_index=i,
                        )
                    )
        return out

    def _task_rules(self, genome: Genome, task: RuntimeTask) -> list[Violation]:
        out: list[Violation] = []
        gather = next((s for s in genome.stages if s.kind == "GATHER"), None)
        if gather is not None:
            if gather.source not in task.allowed_sources:
                out.append(
                    Violation(
                        code=ViolationCode.SOURCE_NOT_ALLOWED,
                        message=f"source {gather.source.value} not allowed for task {task.id}",
                        stage_index=0,
                    )
                )
            if task.interaction_required and gather.source != GatherSource.JEV:
                out.append(
                    Violation(
                        code=ViolationCode.INTERACTION_REQUIRES_JEV,
                        message="task requires interaction; GATHER source must be jev",
                        stage_index=0,
                    )
                )
        over = exceeded_caps(self.cost_model.estimate(genome, task), task.caps)
        if over:
            out.append(
                Violation(
                    code=ViolationCode.BUDGET_INFEASIBLE,
                    message="best-case estimate exceeds caps: " + ", ".join(c.value for c in over),
                )
            )
        return out


# -- measured hard limits ------------------------------------------------------------------------
class ConstraintLimits(BaseModel):
    """Hard limits on a candidate's MEASURED behaviour. ``None`` = no limit on that dimension.

    Means / p95 are compared with the corresponding aggregate measurement; ``maximum_tokens``,
    ``model_calls``, ``tool_calls``, ``retries`` and ``wall_time_s`` are per-example run caps and
    are compared with the worst example (``max_*_per_example``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_quality: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    maximum_cost_per_example: NonNegativeFloat | None = Field(default=None, allow_inf_nan=False)
    maximum_mean_latency_s: PositiveFloat | None = Field(default=None, allow_inf_nan=False)
    maximum_p95_latency_s: PositiveFloat | None = Field(default=None, allow_inf_nan=False)
    maximum_tokens_per_example: PositiveInt | None = None
    maximum_workflow_steps: PositiveInt | None = None
    maximum_model_calls: NonNegativeInt | None = None
    maximum_tool_calls: NonNegativeInt | None = None
    maximum_retries: NonNegativeInt | None = None
    maximum_wall_time_s: PositiveFloat | None = Field(default=None, allow_inf_nan=False)

    @classmethod
    def from_caps(cls, caps: Caps) -> ConstraintLimits:
        """The legacy per-run ``Caps`` expressed as measured limits (same numbers, same meaning:
        a run strictly above a cap breaches it)."""
        return cls(
            maximum_tokens_per_example=caps.tokens,
            maximum_wall_time_s=caps.wall_time_s,
            maximum_tool_calls=caps.tool_calls,
            maximum_retries=caps.retries,
        )

    def to_caps(self) -> Caps:
        """Runtime caps for the budget guard. Fails closed if any capped dimension is unset."""
        missing = [
            name
            for name in (
                "maximum_tokens_per_example",
                "maximum_wall_time_s",
                "maximum_tool_calls",
                "maximum_retries",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError("runtime caps need limits for: " + ", ".join(missing))
        return Caps(
            tokens=self.maximum_tokens_per_example,
            wall_time_s=self.maximum_wall_time_s,
            tool_calls=self.maximum_tool_calls,
            retries=self.maximum_retries,
        )

    @property
    def identity_hash(self) -> str:
        return canonical_hash(
            {"schema": CONSTRAINT_LIMITS_SCHEMA_VERSION, **self.model_dump(mode="json")}
        )


# limit -> (measurement it is compared with, True if the limit is a floor)
LIMIT_MEASUREMENTS: dict[str, tuple[str, bool]] = {
    "minimum_quality": ("quality", True),
    "maximum_cost_per_example": ("mean_cost_per_example", False),
    "maximum_mean_latency_s": ("mean_latency_s", False),
    "maximum_p95_latency_s": ("p95_latency_s", False),
    "maximum_tokens_per_example": ("max_tokens_per_example", False),
    "maximum_workflow_steps": ("workflow_steps", False),
    "maximum_model_calls": ("max_model_calls_per_example", False),
    "maximum_tool_calls": ("max_tool_calls_per_example", False),
    "maximum_retries": ("max_retries_per_example", False),
    "maximum_wall_time_s": ("max_wall_time_s_per_example", False),
}


def check_limits(
    limits: ConstraintLimits, measured: CandidateMeasurements
) -> tuple[Violation, ...]:
    """Every hard-limit violation. An active limit whose measurement is missing is itself a
    violation (fail closed). Equal to a maximum / minimum is allowed."""
    out: list[Violation] = []
    for limit_name, (field, is_floor) in LIMIT_MEASUREMENTS.items():
        limit = getattr(limits, limit_name)
        if limit is None:
            continue
        value = getattr(measured, field)
        if value is None:
            out.append(
                Violation(
                    code=ViolationCode.METRIC_MISSING,
                    message=f"{limit_name} is set but {field} was not measured",
                )
            )
        elif (value < limit) if is_floor else (value > limit):
            out.append(
                Violation(
                    code=ViolationCode.LIMIT_VIOLATED,
                    message=f"{field}={value} violates {limit_name}={limit}",
                )
            )
    return tuple(out)
