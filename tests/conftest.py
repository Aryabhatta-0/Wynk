"""Shared test builders. Kept tiny and explicit so tests read like the architecture."""

from __future__ import annotations

from typing import Any

import pytest

from core.constraints import ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitRole,
)
from core.evaluation_spec import EvaluationSpec
from core.genome import Genome
from core.run_contract import ContractSuite, ExampleInput, ExecutionTask
from core.stages import (
    ExtractMethod,
    ExtractStage,
    FailureStrategy,
    FilterMethod,
    FilterStage,
    GatherMode,
    GatherSource,
    GatherStage,
    ReasonMethod,
    ReasonStage,
    SynthesizeMethod,
    SynthesizeStage,
    VerifyMethod,
    VerifyStage,
)
from core.task_contract import TaskContract, TaskType, WorkflowSpec
from core.task_spec import (
    AnswerField,
    AnswerSchema,
    Caps,
    FieldType,
    GroundTruth,
    MatcherConfig,
    MatcherKind,
    RuntimeTask,
    TaskClass,
    TaskSpec,
)

SECRET_ANSWER = "TOP-SECRET-GROUND-TRUTH-42"


def gather(source=GatherSource.FETCH, mode=GatherMode.SEQUENTIAL) -> GatherStage:
    return GatherStage(source=source, mode=mode)


def flt(method=FilterMethod.KEYWORD_CHUNK) -> FilterStage:
    return FilterStage(method=method)


def extract(method=ExtractMethod.DIRECT) -> ExtractStage:
    return ExtractStage(method=method)


def reason(method=ReasonMethod.SINGLE) -> ReasonStage:
    return ReasonStage(method=method)


def verify(method=VerifyMethod.SCHEMA_CHECK, on_failure=FailureStrategy.RETRY_1) -> VerifyStage:
    return VerifyStage(method=method, on_failure=on_failure)


def synth(method=SynthesizeMethod.DIRECT) -> SynthesizeStage:
    return SynthesizeStage(method=method)


def minimal_genome() -> Genome:
    return Genome.of(gather(), extract(), synth())


def make_caps(**overrides) -> Caps:
    base = {"tokens": 20_000, "wall_time_s": 120.0, "tool_calls": 20, "retries": 2}
    return Caps(**{**base, **overrides})


def make_runtime_task(**overrides) -> RuntimeTask:
    base = {
        "id": "task-001",
        "task_class": TaskClass.A,
        "question": "What is the capital of France?",
        "answer_schema": AnswerSchema(fields=(AnswerField(name="capital", type=FieldType.STRING),)),
        "caps": make_caps(),
        "allowed_sources": (GatherSource.FETCH, GatherSource.API, GatherSource.JEV),
        "snapshot_id": "snap-001",
    }
    return RuntimeTask(**{**base, **overrides})


CAPITAL_SCHEMA = AnswerSchema(fields=(AnswerField(name="capital", type=FieldType.STRING),))
TEST_INSTRUCTIONS = "Answer the question from the gathered pages."


def make_contract(
    *,
    id: str = "task-001",
    answer_schema: AnswerSchema = CAPITAL_SCHEMA,
    caps: Caps | None = None,
    allowed_sources: tuple[GatherSource, ...] = (
        GatherSource.FETCH,
        GatherSource.API,
        GatherSource.JEV,
    ),
    interaction_required: bool = False,
    row_count: int = 1,
    **overrides: Any,
) -> TaskContract:
    """A snapshot-backed extraction contract with the same knobs as ``make_runtime_task``."""
    targets = tuple(
        ColumnSpec(name=f.name, type=ColumnType(f.type.value)) for f in answer_schema.fields
    )
    dataset = DatasetSpec(
        dataset_id=id.lower(),
        dataset_version=1,
        name="test dataset",
        content_hash="c" * 64,
        format=DatasetFormat.WYNK_SNAPSHOT,
        columns=(
            ColumnSpec(name="row_id", type=ColumnType.STRING),
            ColumnSpec(name="question", type=ColumnType.STRING),
            ColumnSpec(name="snapshot_id", type=ColumnType.STRING),
            *targets,
        ),
        id_column="row_id",
        input_columns=("question",),
        context_columns=("snapshot_id",),
        target_columns=tuple(c.name for c in targets),
        row_count=row_count,
    )
    base: dict[str, Any] = {
        "task_id": id.lower(),
        "contract_version": 1,
        "task_type": TaskType.STRUCTURED_EXTRACTION,
        "instructions": TEST_INSTRUCTIONS,
        "input_schema": AnswerSchema(
            fields=(
                AnswerField(name="question", type=FieldType.STRING),
                AnswerField(name="snapshot_id", type=FieldType.STRING),
            )
        ),
        "output_schema": answer_schema,
        "dataset": dataset,
        "evaluation": EvaluationSpec(evaluator="exact_match"),
        "constraints": ConstraintLimits.from_caps(caps or make_caps()),
        "workflow": WorkflowSpec(
            allowed_sources=allowed_sources, interaction_required=interaction_required
        ),
    }
    return TaskContract(**{**base, **overrides})


def make_task(
    *,
    question: str = "What is the capital of France?",
    snapshot_id: str = "snap-001",
    id: str = "task-001",
    **contract_kw: Any,
) -> ExecutionTask:
    """One contract-bound example (what search and execution consume)."""
    return ExecutionTask(
        contract=make_contract(id=id, **contract_kw),
        example=ExampleInput(row_id=id, values={"question": question, "snapshot_id": snapshot_id}),
    )


def make_suite(
    train: tuple[str, ...] = ("train-1",),
    val: tuple[str, ...] = ("val-1",),
    test: tuple[str, ...] = (),
    *,
    name: str = "test",
    **contract_kw: Any,
) -> ContractSuite:
    """One contract over ``train + val + test`` rows, split explicitly by role."""
    rows = (*train, *val, *test)
    contract = make_contract(row_count=len(rows), **contract_kw)
    tasks = tuple(
        ExecutionTask(
            contract=contract,
            example=ExampleInput(
                row_id=r, values={"question": f"Question {r}?", "snapshot_id": "snap-001"}
            ),
        )
        for r in rows
    )
    splits = DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=tuple(
            DatasetSplit(split_id=role.value, role=role, row_ids=ids)
            for role, ids in (
                (SplitRole.OPTIMIZATION, train),
                (SplitRole.VALIDATION, val),
                (SplitRole.TEST, test),
            )
            if ids
        ),
    )
    return ContractSuite(name=name, tasks=tasks, splits=splits)


def make_task_spec(runtime: RuntimeTask | None = None) -> TaskSpec:
    return TaskSpec(
        runtime=runtime or make_runtime_task(),
        ground_truth=GroundTruth(values={"capital": SECRET_ANSWER}),
        matchers={"capital": MatcherConfig(kind=MatcherKind.NORMALIZED_TEXT)},
    )


@pytest.fixture
def task() -> TaskContract:
    """The admission authority the checker reads (a contract, not an example)."""
    return make_contract()


@pytest.fixture
def task_spec() -> TaskSpec:
    return make_task_spec()
