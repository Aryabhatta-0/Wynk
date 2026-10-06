"""What search and execution may know about a task: a ``TaskContract`` bound to one example.

    ExecutionTask  = TaskContract + ExampleInput      (one dataset row's input/context values)
    ContractSuite  = ExecutionTasks + DatasetSplits   (what one search runs on, and for what)

The contract is the single authority. Every value the runtime and the search use is DERIVED from
it here and never stored a second time:

    prompt text         the row's input values + contract.instructions
    output schema       contract.output_schema
    per-run caps        contract.constraints  (budget guard; must be complete to execute)
    allowed workflows   contract.workflow + contract.constraints  (admission)
    data source         contract.dataset.format + the row's context values
    candidate ranking   contract.rank  (hard limits first, objective second)
    split authority     DatasetSplits  (optimization -> feedback, validation -> selection,
                                        test -> reporting only)

Target values never enter: they stay with the evaluator. ``check_executable`` refuses a contract
the runtime cannot honour (incomplete caps, an objective or limit on something the runtime does
not measure) - ``ExecutionTask`` runs it on construction, so this happens before any model call.
Nothing here knows about the legacy benchmark or its task classes
(``benchmarks/legacy_adapter.py`` is the only bridge).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.canonical import canonical_hash, canonical_json
from core.dataset import DatasetFormat, DatasetSplits, SplitRole, SplitUse
from core.genome import Genome
from core.objective import CandidateMeasurements, Metric
from core.results import EvaluatedRun, Verdict
from core.stages import GatherSource
from core.task_contract import CandidateRank, ContractError, TaskContract
from core.task_spec import AnswerSchema, Caps
from core.violations import Violation, ViolationCode

# What the runtime + evaluator measure today. Cost has no price model yet, so an objective or a
# hard limit on cost could never be checked and is refused up front instead of failing later.
MEASURED_METRICS = frozenset({Metric.QUALITY, Metric.LATENCY, Metric.TOKENS})
UNMEASURED_LIMITS = ("maximum_cost_per_example",)


# -- executability ------------------------------------------------------------------------------
def runtime_caps(contract: TaskContract) -> Caps:
    """The per-run caps the budget guard enforces, from ``contract.constraints``. Fail closed."""
    try:
        return contract.constraints.to_caps()
    except ValueError as exc:
        raise ContractError(f"contract {contract.task_id} cannot be executed: {exc}") from exc


def check_executable(contract: TaskContract) -> None:
    """Raise ``ContractError`` unless the runtime can execute and measure ``contract``."""
    runtime_caps(contract)
    unmeasured = set(contract.objective.required_metrics()) - MEASURED_METRICS
    if unmeasured:
        raise ContractError(
            "objective needs "
            + ", ".join(sorted(m.value for m in unmeasured))
            + ", which the runtime does not measure"
        )
    for name in UNMEASURED_LIMITS:
        if getattr(contract.constraints, name) is not None:
            raise ContractError(f"{name} cannot be enforced: the runtime does not measure it")
    if (
        contract.dataset.format is DatasetFormat.WYNK_SNAPSHOT
        and len(contract.dataset.context_columns) != 1
    ):
        raise ContractError("wynk_snapshot datasets need exactly one context column (snapshot id)")


# -- one example --------------------------------------------------------------------------------
class ExampleInput(BaseModel):
    """One dataset row as execution may see it: input and context values only, never targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row_id: str = Field(min_length=1, max_length=256)
    values: dict[str, Any]


@dataclass(frozen=True)
class SnapshotSource:
    """GATHER reads the frozen snapshot named by the row's context column."""

    snapshot_id: str


@dataclass(frozen=True)
class InlineSource:
    """GATHER reads the row's own context values: ``(column, text)`` pairs, in column order."""

    documents: tuple[tuple[str, str], ...]


DataSource = SnapshotSource | InlineSource


def _text(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value)


class ExecutionTask(BaseModel):
    """The only task view search and execution hold. Everything below is derived from
    ``contract``; nothing is a second copy that could disagree with it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: TaskContract
    example: ExampleInput

    @model_validator(mode="after")
    def _executable(self) -> ExecutionTask:
        check_executable(self.contract)
        schema = self.contract.input_schema
        unknown = set(self.example.values) - schema.field_names
        if unknown:
            raise ContractError(f"row {self.example.row_id} has unknown inputs {sorted(unknown)}")
        missing = [
            f.name for f in schema.fields if f.required and self.example.values.get(f.name) is None
        ]
        if missing:
            raise ContractError(f"row {self.example.row_id} is missing inputs {missing}")
        return self

    @property
    def id(self) -> str:
        return self.example.row_id

    @property
    def contract_hash(self) -> str:
        return self.contract.contract_hash

    @property
    def answer_schema(self) -> AnswerSchema:
        return self.contract.output_schema

    @property
    def caps(self) -> Caps:
        return runtime_caps(self.contract)

    @property
    def allowed_sources(self) -> tuple[GatherSource, ...]:
        return self.contract.workflow.allowed_sources

    @property
    def inputs_text(self) -> str:
        """This row's input values (context columns excluded: GATHER reads those). A single
        input is its bare value; several are ``name: value`` lines in column order."""
        present = [
            (name, _text(self.example.values[name]))
            for name in self.contract.dataset.input_columns
            if self.example.values.get(name) is not None
        ]
        if len(present) == 1:
            return present[0][1]
        return "\n".join(f"{name}: {text}" for name, text in present)

    @property
    def question(self) -> str:
        """The task text a model sees: this row's inputs + the contract's instructions."""
        return f"{self.inputs_text}\n\nInstructions: {self.contract.instructions}"

    @property
    def source(self) -> DataSource:
        ds = self.contract.dataset
        if ds.format is DatasetFormat.WYNK_SNAPSHOT:
            return SnapshotSource(snapshot_id=str(self.example.values[ds.context_columns[0]]))
        return InlineSource(
            documents=tuple(
                (name, _text(self.example.values[name]))
                for name in ds.context_columns
                if self.example.values.get(name) is not None
            )
        )


# -- candidate ranking --------------------------------------------------------------------------
def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]  # nearest rank


def candidate_measurements(genome: Genome, runs: Sequence[EvaluatedRun]) -> CandidateMeasurements:
    """Aggregate what was measured on ``runs`` of ``genome``. Cost is not measured (``None``)."""
    if not runs:
        raise ValueError("cannot measure a candidate without runs")
    usage = [r.execution.budget_usage for r in runs]
    wall = [u.wall_time_s for u in usage]
    return CandidateMeasurements(
        quality=sum(r.evaluation.verdict is Verdict.PASS for r in runs) / len(runs),
        mean_latency_s=statistics.fmean(wall),
        p95_latency_s=_p95(wall),
        mean_tokens_per_example=statistics.fmean(u.tokens for u in usage),
        max_tokens_per_example=max(u.tokens for u in usage),
        workflow_steps=len(genome),
        max_model_calls_per_example=max(r.execution.metrics.model_calls for r in runs),
        max_tool_calls_per_example=max(u.tool_calls for u in usage),
        max_retries_per_example=max(u.retries for u in usage),
        max_wall_time_s_per_example=max(wall),
    )


def rank_candidate(
    contract: TaskContract, genome: Genome, runs: Sequence[EvaluatedRun]
) -> CandidateRank:
    """``contract.rank`` over ``runs``. A run the evaluator found INFEASIBLE (a per-run cap was
    breached, possibly before execution) makes the whole candidate infeasible."""
    if any(r.evaluation.verdict is Verdict.INFEASIBLE for r in runs):
        violation = Violation(
            code=ViolationCode.LIMIT_VIOLATED, message="a run breached its per-run caps"
        )
        return CandidateRank(feasible=False, violations=(violation,), objective_key=())
    return contract.rank(candidate_measurements(genome, runs))


# -- the search unit ----------------------------------------------------------------------------
def suite_dataset_hash(contracts: Iterable[TaskContract]) -> str:
    """Identity of the data a suite covers: the dataset identity when every task shares one
    dataset, otherwise the hash of the sorted distinct identities."""
    distinct = sorted({c.dataset.identity_hash for c in contracts})
    return distinct[0] if len(distinct) == 1 else canonical_hash({"datasets": distinct})


def _search_policy(c: TaskContract) -> dict[str, Any]:
    """What one genome is admitted and ranked under; must be shared by every task of a suite."""
    return {
        "workflow": c.workflow.model_dump(mode="json"),
        "constraints": c.constraints.model_dump(mode="json"),
        "objective": c.objective.model_dump(mode="json"),
        "evaluator": [c.evaluation.evaluator.value, c.evaluation.evaluator_version],
    }


class ContractSuite(BaseModel):
    """What one search runs on: execution tasks plus the split contract over their rows.

    A genome is admitted once and evaluated across every task, so all tasks must share one
    search policy (workflow, constraints, objective, evaluator kind/version). The splits must
    assign exactly the suite's rows and be made for exactly the suite's data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)  # display label only; not authority
    tasks: tuple[ExecutionTask, ...] = Field(min_length=1)
    splits: DatasetSplits

    @model_validator(mode="after")
    def _consistent(self) -> ContractSuite:
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ContractError("row ids must be unique across a suite")
        contracts = [t.contract for t in self.tasks]
        policy = _search_policy(contracts[0])
        for c in contracts[1:]:
            if _search_policy(c) != policy:
                raise ContractError(
                    f"{c.task_id} does not share the suite's workflow/constraints/objective/"
                    "evaluator, so one genome cannot be admitted and ranked for both"
                )
        rows_per_dataset: dict[str, int] = {}
        for c in contracts:
            rows_per_dataset[c.dataset.identity_hash] = (
                rows_per_dataset.get(c.dataset.identity_hash, 0) + 1
            )
            if rows_per_dataset[c.dataset.identity_hash] > c.dataset.row_count:
                raise ContractError(f"more rows than dataset {c.dataset.dataset_id} has")
        if self.splits.dataset_hash != suite_dataset_hash(contracts):
            raise ContractError("splits were made for different data than the suite's contracts")
        assigned = {r for s in self.splits.splits for r in s.row_ids}
        if assigned != set(ids):
            raise ContractError("splits must assign exactly the suite's rows")
        return self

    @property
    def policy(self) -> TaskContract:
        """The contract whose workflow/constraints/objective govern admission and ranking (the
        same for every task, see the validator)."""
        return self.tasks[0].contract

    def tasks_for(self, role: SplitRole, use: SplitUse) -> tuple[ExecutionTask, ...]:
        """Tasks of ``role``'s split, in suite order - only if ``use`` is permitted for it."""
        rows = set(self.splits.rows_for(role, use))
        return tuple(t for t in self.tasks if t.id in rows)

    def check_feedback(self, runs: Iterable[EvaluatedRun]) -> None:
        """Fail closed unless every run is on an optimization row (``SplitAccessError``)."""
        self.splits.check_feedback(r.execution.task_id for r in runs)

    def rank(self, genome: Genome, runs: Sequence[EvaluatedRun]) -> CandidateRank:
        return rank_candidate(self.policy, genome, runs)

    @property
    def identity_hash(self) -> str:
        return canonical_hash(
            {
                "tasks": [[t.contract_hash, t.example.model_dump(mode="json")] for t in self.tasks],
                "splits": self.splits.identity_hash,
            }
        )
