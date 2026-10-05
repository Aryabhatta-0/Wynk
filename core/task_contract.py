"""The generic, dataset-driven task contract.

    TaskContract
      dataset      DatasetSpec       what data, which columns are inputs / targets / context
      evaluation   EvaluationSpec    how an output is judged (deterministic, external authority)
      objective    ObjectiveSpec     what "better" means among feasible candidates
      constraints  ConstraintLimits  hard limits; violating one makes a candidate infeasible
      workflow     WorkflowSpec      which workflow configurations a search may build

A contract is plain, frozen, versioned data validated deterministically by this module - it is
never authored or amended by a model. It replaces the ROLE of the hard-coded benchmark task
classes (A/B/C): the generic core can reason about any dataset through this contract, while the
frozen benchmark keeps working through ``benchmarks/legacy_adapter.py``.

Supported task types are deliberately bounded: classification, structured_extraction and
question_answering. Each type fixes which output shapes and evaluators are compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from core.canonical import canonical_hash, canonical_json
from core.constraints import ConstraintLimits, check_limits
from core.dataset import SLUG, DatasetFormat, DatasetSpec
from core.evaluation_spec import EvaluationSpec, EvaluatorKind
from core.objective import CandidateMeasurements, ObjectiveMode, ObjectiveSpec
from core.stages import GatherSource
from core.task_spec import AnswerSchema, FieldType
from core.violations import Violation

TASK_CONTRACT_SCHEMA_VERSION = "taskcontract/1"
MAX_INSTRUCTIONS_CHARS = 20_000

# Field schemas describe both sides of a task; ``AnswerSchema`` is the existing typed field list.
FieldSchema = AnswerSchema


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    STRUCTURED_EXTRACTION = "structured_extraction"
    QUESTION_ANSWERING = "question_answering"


_TEXT = frozenset({FieldType.STRING})
_NUMERIC = frozenset({FieldType.INTEGER, FieldType.NUMBER})

# task type -> output field type -> evaluators that can judge it
QA_EVALUATORS: dict[FieldType, frozenset[EvaluatorKind]] = {
    FieldType.STRING: frozenset({EvaluatorKind.EXACT_MATCH, EvaluatorKind.TOKEN_F1}),
    FieldType.INTEGER: frozenset({EvaluatorKind.EXACT_MATCH, EvaluatorKind.NUMERIC_TOLERANCE}),
    FieldType.NUMBER: frozenset({EvaluatorKind.EXACT_MATCH, EvaluatorKind.NUMERIC_TOLERANCE}),
}
CLASSIFICATION_EVALUATORS = frozenset(
    {EvaluatorKind.CLASSIFICATION_ACCURACY, EvaluatorKind.EXACT_MATCH}
)
EXTRACTION_EVALUATORS = frozenset(
    {
        EvaluatorKind.EXACT_MATCH,
        EvaluatorKind.JSON_SCHEMA_VALIDITY,
        EvaluatorKind.NUMERIC_TOLERANCE,
        EvaluatorKind.LEGACY_FIELD_MATCH,
    }
)


class ContractError(ValueError):
    pass


# dataset format -> the GATHER sources that can read it. Tabular rows carry their own context
# (``fetch`` reads the row's context columns); only the frozen snapshot format has a mock API and
# an interactive browser surface.
SUPPORTED_SOURCES: dict[DatasetFormat, frozenset[GatherSource]] = {
    DatasetFormat.CSV: frozenset({GatherSource.FETCH}),
    DatasetFormat.JSONL: frozenset({GatherSource.FETCH}),
    DatasetFormat.WYNK_SNAPSHOT: frozenset(GatherSource),
}


class WorkflowSpec(BaseModel):
    """The workflow configurations a search may build for this task (admission authority)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_sources: tuple[GatherSource, ...] = Field(default=(GatherSource.FETCH,), min_length=1)
    interaction_required: bool = False

    @model_validator(mode="after")
    def _sources(self) -> WorkflowSpec:
        if len(set(self.allowed_sources)) != len(self.allowed_sources):
            raise ValueError("allowed_sources must not repeat a source")
        if self.interaction_required and GatherSource.JEV not in self.allowed_sources:
            raise ValueError("interaction_required needs 'jev' in allowed_sources")
        return self


class TaskContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["taskcontract/1"] = TASK_CONTRACT_SCHEMA_VERSION
    task_id: str = Field(pattern=SLUG)
    contract_version: PositiveInt
    task_type: TaskType

    instructions: str = Field(min_length=1, max_length=MAX_INSTRUCTIONS_CHARS)
    input_schema: FieldSchema  # one field per dataset input + context column
    output_schema: FieldSchema  # one field per dataset target column

    dataset: DatasetSpec
    evaluation: EvaluationSpec
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    constraints: ConstraintLimits = Field(default_factory=ConstraintLimits)
    workflow: WorkflowSpec = Field(default_factory=WorkflowSpec)

    @model_validator(mode="after")
    def _consistent(self) -> TaskContract:
        _check_dataset_mapping(self)
        _check_task_type(self)
        _check_objective(self)
        _check_workflow(self)
        return self

    # -- identity -----------------------------------------------------------------------------
    def authoritative(self) -> dict[str, Any]:
        """Everything that changes what an experiment computes (dataset name/metadata excluded)."""
        data = self.model_dump(mode="json", exclude={"dataset"})
        data["dataset"] = self.dataset.authoritative()
        return data

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.authoritative())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    # -- ranking ------------------------------------------------------------------------------
    def rank(self, measured: CandidateMeasurements) -> CandidateRank:
        """Hard limits first, objective second. An infeasible candidate's objective is never
        computed, so no weighted score can lift it above a feasible one."""
        violations = check_limits(self.constraints, measured)
        if violations:
            return CandidateRank(feasible=False, violations=violations, objective_key=())
        return CandidateRank(
            feasible=True, violations=(), objective_key=self.objective.rank_key(measured)
        )


@dataclass(frozen=True)
class CandidateRank:
    feasible: bool
    violations: tuple[Violation, ...]
    objective_key: tuple[float, ...]

    @property
    def sort_key(self) -> tuple[float, ...]:
        """Higher is better; every feasible key outranks every infeasible one."""
        return (1.0, *self.objective_key) if self.feasible else (0.0,)


# -- validation rules -----------------------------------------------------------------------------
def _check_fields(
    schema: FieldSchema, columns: tuple[str, ...], ds: DatasetSpec, side: str
) -> None:
    if schema.field_names != set(columns):
        raise ContractError(
            f"{side} schema fields {sorted(schema.field_names)} must equal dataset columns "
            f"{sorted(columns)}"
        )
    for f in schema.fields:
        col = ds.column(f.name)
        if col.type.field_type() != f.type:
            raise ContractError(
                f"{side} field {f.name!r} is {f.type.value} but column is {col.type.value}"
            )
        if col.nullable and f.required:
            raise ContractError(f"{side} field {f.name!r} is required but its column is nullable")


def _check_dataset_mapping(c: TaskContract) -> None:
    ds = c.dataset
    _check_fields(c.input_schema, (*ds.input_columns, *ds.context_columns), ds, "input")
    _check_fields(c.output_schema, ds.target_columns, ds, "output")


def _check_task_type(c: TaskContract) -> None:
    kind = c.evaluation.evaluator
    fields = c.output_schema.fields
    if kind is EvaluatorKind.LEGACY_FIELD_MATCH:
        if c.dataset.format is not DatasetFormat.WYNK_SNAPSHOT:
            raise ContractError("legacy_field_match is only valid on wynk_snapshot datasets")
        if set(c.evaluation.config["matchers"]) != c.output_schema.field_names:
            raise ContractError("legacy matchers must cover exactly the output fields")
    if c.task_type is TaskType.CLASSIFICATION:
        if len(fields) != 1 or fields[0].type not in _TEXT:
            raise ContractError("classification needs exactly one string output field")
        if kind not in CLASSIFICATION_EVALUATORS:
            raise ContractError(f"{kind.value} cannot evaluate classification")
    elif c.task_type is TaskType.QUESTION_ANSWERING:
        if len(fields) != 1 or fields[0].type not in _TEXT | _NUMERIC:
            raise ContractError("question_answering needs exactly one string or numeric output")
        if kind not in QA_EVALUATORS[fields[0].type]:
            raise ContractError(
                f"{kind.value} cannot evaluate a {fields[0].type.value} question_answering output"
            )
    else:
        if kind not in EXTRACTION_EVALUATORS:
            raise ContractError(f"{kind.value} cannot evaluate structured_extraction")
        if kind is EvaluatorKind.NUMERIC_TOLERANCE and any(f.type not in _NUMERIC for f in fields):
            raise ContractError("numeric_tolerance needs every output field to be numeric")


def _check_objective(c: TaskContract) -> None:
    # Minimizing cost or latency with no quality floor is won by a workflow that does nothing.
    if (
        c.objective.mode in (ObjectiveMode.MINIMIZE_COST, ObjectiveMode.MINIMIZE_LATENCY)
        and c.constraints.minimum_quality is None
    ):
        raise ContractError(f"{c.objective.mode.value} requires constraints.minimum_quality")


def _check_workflow(c: TaskContract) -> None:
    unsupported = set(c.workflow.allowed_sources) - SUPPORTED_SOURCES[c.dataset.format]
    if unsupported:
        raise ContractError(
            f"{c.dataset.format.value} datasets cannot be gathered with "
            + ", ".join(sorted(s.value for s in unsupported))
        )
