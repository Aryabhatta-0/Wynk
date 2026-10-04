"""Execution and evaluation result contracts.

Authority split, enforced by types:
  * ``ExecutionResult`` is produced by the runtime. It has NO verdict/fitness field.
  * ``Evaluation`` (verdict + fitness) is produced only by the offline evaluator.
  * ``EvaluatedRun`` pairs them; optimizers and the run store consume this.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from core.canonical import canonical_hash
from core.evidence import FieldEvidence
from core.payloads import Answer
from core.stages import StageKind
from core.task_spec import Caps


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INFEASIBLE = "INFEASIBLE"


class BudgetCap(StrEnum):
    TOKENS = "tokens"
    WALL_TIME = "wall_time_s"
    TOOL_CALLS = "tool_calls"
    RETRIES = "retries"


class BudgetUsage(BaseModel):
    """Counters checked against ``Caps``. Additive: ``a + b`` sums every counter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: NonNegativeInt = 0
    wall_time_s: NonNegativeFloat = 0.0
    tool_calls: NonNegativeInt = 0
    retries: NonNegativeInt = 0

    def __add__(self, other: BudgetUsage) -> BudgetUsage:
        return BudgetUsage(
            tokens=self.tokens + other.tokens,
            wall_time_s=self.wall_time_s + other.wall_time_s,
            tool_calls=self.tool_calls + other.tool_calls,
            retries=self.retries + other.retries,
        )


def usage_exceeds(usage: BudgetUsage, caps: Caps) -> tuple[BudgetCap, ...]:
    """Deterministic cap check shared by the budget guard and the evaluator.

    A counter equal to its cap is allowed; strictly greater is a breach.
    """
    out: list[BudgetCap] = []
    if usage.tokens > caps.tokens:
        out.append(BudgetCap.TOKENS)
    if usage.wall_time_s > caps.wall_time_s:
        out.append(BudgetCap.WALL_TIME)
    if usage.tool_calls > caps.tool_calls:
        out.append(BudgetCap.TOOL_CALLS)
    if usage.retries > caps.retries:
        out.append(BudgetCap.RETRIES)
    return tuple(out)


class ExecutionMetrics(BaseModel):
    """Descriptive metrics (not capped)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_calls: NonNegativeInt = 0
    prompt_tokens: NonNegativeInt = 0
    completion_tokens: NonNegativeInt = 0
    pages_fetched: NonNegativeInt = 0
    cache_hits: NonNegativeInt = 0
    backoff_time_s: NonNegativeFloat = 0.0


class FailureKind(StrEnum):
    BUDGET_EXCEEDED = "budget_exceeded"
    EXECUTOR_ERROR = "executor_error"
    MODEL_ERROR = "model_error"
    SCHEMA_INVALID = "schema_invalid"
    NO_ANSWER = "no_answer"


class FailureInfo(BaseModel):
    """A runtime failure. Describes what broke - never whether the answer was correct."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: FailureKind
    message: str
    stage_index: int | None = None
    cap: BudgetCap | None = None  # set when kind == BUDGET_EXCEEDED


class StageStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_index: int = Field(ge=0)
    kind: StageKind
    status: StageStatus
    usage: BudgetUsage = BudgetUsage()
    input_digest: str | None = None
    output_digest: str | None = None
    failure: FailureInfo | None = None


class RunVersions(BaseModel):
    """Everything besides genome/task/seed that can change a run's behaviour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_hash: str
    prompt_template_version: str
    benchmark_hash: str
    compiler_version: str
    grammar_version: str


class RunKey(BaseModel):
    """Deterministic identity of a run; also the cache key. No timestamps / uuids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    genome_hash: str
    task_id: str
    trial: NonNegativeInt
    seed: int
    versions: RunVersions

    @property
    def run_id(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ExecutionResult(BaseModel):
    """What the runtime reports. Contains no ground truth and no correctness judgement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: RunKey
    answer: Answer | None = None  # None when the run failed before producing one
    metrics: ExecutionMetrics = ExecutionMetrics()
    stage_trace: tuple[StageTrace, ...] = ()
    budget_usage: BudgetUsage = BudgetUsage()
    failure: FailureInfo | None = None

    @property
    def evidence(self) -> tuple[FieldEvidence, ...]:
        """Evidence lives on the Answer (single source of truth)."""
        return self.answer.evidence if self.answer else ()

    @property
    def run_id(self) -> str:
        return self.key.run_id

    @property
    def genome_hash(self) -> str:
        return self.key.genome_hash

    @property
    def task_id(self) -> str:
        return self.key.task_id


class FieldResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    matched: bool
    evidence_valid: bool | None = None


class Evaluation(BaseModel):
    """Produced ONLY by the deterministic offline evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    # Shaped search fitness, higher is better, always finite. Scale is owned by
    # evaluation/fitness.py (Track A) and recorded via evaluator_version.
    fitness: float = Field(allow_inf_nan=False)
    evaluator_version: str
    field_results: tuple[FieldResult, ...] = ()


class EvaluatedRun(BaseModel):
    """Execution + evaluation. The unit consumed by optimizers and persisted by the store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution: ExecutionResult
    evaluation: Evaluation

    @property
    def run_id(self) -> str:
        return self.execution.run_id
