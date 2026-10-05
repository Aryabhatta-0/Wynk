"""Builders for the dataset/task-contract tests. Small and explicit, like ``conftest.py``."""

from __future__ import annotations

from typing import Any

from core.constraints import ConstraintLimits
from core.dataset import ColumnSpec, ColumnType, DatasetFormat, DatasetSpec
from core.evaluation_spec import EvaluationSpec
from core.objective import ObjectiveSpec
from core.task_contract import TaskContract, TaskType
from core.task_spec import AnswerField, AnswerSchema, FieldType

HASH_A = "a" * 64
HASH_B = "b" * 64


def col(name: str, type_: ColumnType = ColumnType.STRING, nullable: bool = False) -> ColumnSpec:
    return ColumnSpec(name=name, type=type_, nullable=nullable)


def schema(*fields: tuple[str, FieldType]) -> AnswerSchema:
    return AnswerSchema(fields=tuple(AnswerField(name=n, type=t) for n, t in fields))


def make_dataset(**overrides: Any) -> DatasetSpec:
    """A ticket-routing CSV: ``id`` (row id), ``text`` (input), ``label`` (target)."""
    base: dict[str, Any] = {
        "dataset_id": "support-tickets",
        "dataset_version": 1,
        "name": "Support tickets",
        "content_hash": HASH_A,
        "format": DatasetFormat.CSV,
        "columns": (col("id"), col("text"), col("label")),
        "id_column": "id",
        "input_columns": ("text",),
        "target_columns": ("label",),
        "row_count": 100,
    }
    return DatasetSpec(**{**base, **overrides})


def classification_contract(**overrides: Any) -> TaskContract:
    base: dict[str, Any] = {
        "task_id": "ticket-routing",
        "contract_version": 1,
        "task_type": TaskType.CLASSIFICATION,
        "instructions": "Route the support ticket to exactly one team.",
        "input_schema": schema(("text", FieldType.STRING)),
        "output_schema": schema(("label", FieldType.STRING)),
        "dataset": make_dataset(),
        "evaluation": EvaluationSpec(
            evaluator="classification_accuracy",
            config={"labels": ["billing", "bugs", "sales"]},
        ),
    }
    return TaskContract(**{**base, **overrides})


def qa_dataset(answer_type: ColumnType = ColumnType.STRING, **overrides: Any) -> DatasetSpec:
    """A reading-comprehension JSONL: ``question`` (input), ``passage`` (context), ``answer``."""
    base: dict[str, Any] = {
        "dataset_id": "reading-qa",
        "dataset_version": 1,
        "name": "Reading QA",
        "content_hash": HASH_B,
        "format": DatasetFormat.JSONL,
        "columns": (col("qid"), col("question"), col("passage"), col("answer", answer_type)),
        "id_column": "qid",
        "input_columns": ("question",),
        "context_columns": ("passage",),
        "target_columns": ("answer",),
        "row_count": 40,
    }
    return DatasetSpec(**{**base, **overrides})


def qa_contract(**overrides: Any) -> TaskContract:
    base: dict[str, Any] = {
        "task_id": "reading-qa",
        "contract_version": 1,
        "task_type": TaskType.QUESTION_ANSWERING,
        "instructions": "Answer the question from the passage in a short phrase.",
        "input_schema": schema(("question", FieldType.STRING), ("passage", FieldType.STRING)),
        "output_schema": schema(("answer", FieldType.STRING)),
        "dataset": qa_dataset(),
        "evaluation": EvaluationSpec(evaluator="token_f1", config={"pass_threshold": 0.8}),
        "objective": ObjectiveSpec(),
        "constraints": ConstraintLimits(maximum_tokens_per_example=4000),
    }
    return TaskContract(**{**base, **overrides})
