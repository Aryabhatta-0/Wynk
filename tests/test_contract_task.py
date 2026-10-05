"""TaskContract validation per task type, contract identity, and experiment identity."""

import json

import pytest
from pydantic import ValidationError

from core.constraints import ConstraintLimits
from core.dataset import ColumnType, SplitPlan, seeded_splits
from core.evaluation_spec import EvaluationSpec
from core.experiment import ExperimentIdentity, ModelConfiguration, experiment_identity
from core.grammar import GRAMMAR_VERSION
from core.objective import ObjectiveSpec
from core.task_contract import TaskContract, TaskType
from core.task_spec import FieldType
from tests.contract_helpers import (
    HASH_B,
    classification_contract,
    col,
    make_dataset,
    qa_contract,
    qa_dataset,
    schema,
)


def extraction_contract(**overrides):
    ds = make_dataset(
        dataset_id="invoices",
        columns=(
            col("id"),
            col("document"),
            col("vendor"),
            col("total", ColumnType.NUMBER),
            col("issued", ColumnType.DATE),
        ),
        input_columns=("document",),
        target_columns=("vendor", "total", "issued"),
    )
    base = {
        "task_id": "invoice-extraction",
        "contract_version": 1,
        "task_type": TaskType.STRUCTURED_EXTRACTION,
        "instructions": "Extract the vendor, total and issue date from the invoice text.",
        "input_schema": schema(("document", FieldType.STRING)),
        "output_schema": schema(
            ("vendor", FieldType.STRING), ("total", FieldType.NUMBER), ("issued", FieldType.DATE)
        ),
        "dataset": ds,
        "evaluation": EvaluationSpec(evaluator="exact_match"),
    }
    return TaskContract(**{**base, **overrides})


def numeric_qa_contract(**overrides):
    base = {
        "dataset": qa_dataset(ColumnType.NUMBER),
        "output_schema": schema(("answer", FieldType.NUMBER)),
        "evaluation": EvaluationSpec(
            evaluator="numeric_tolerance", config={"absolute_tolerance": 0.01}
        ),
    }
    return qa_contract(**{**base, **overrides})


# -- valid contracts for every supported task type ----------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        classification_contract,
        lambda: classification_contract(evaluation=EvaluationSpec(evaluator="exact_match")),
        qa_contract,
        lambda: qa_contract(evaluation=EvaluationSpec(evaluator="exact_match")),
        numeric_qa_contract,
        extraction_contract,
        lambda: extraction_contract(evaluation=EvaluationSpec(evaluator="json_schema_validity")),
    ],
)
def test_supported_task_types_build_and_roundtrip(build):
    c = build()
    again = TaskContract.model_validate_json(c.model_dump_json())
    assert again == c and again.contract_hash == c.contract_hash
    assert c.schema_version == "taskcontract/1"


def test_supported_task_types_are_bounded():
    assert {t.value for t in TaskType} == {
        "classification",
        "structured_extraction",
        "question_answering",
    }
    for unsupported in ("agent", "code_execution", "python", "free_form"):
        with pytest.raises(ValidationError):
            classification_contract(task_type=unsupported)


# -- incompatible or invalid contracts ----------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        # schema <-> dataset mapping
        lambda: classification_contract(input_schema=schema(("body", FieldType.STRING))),
        lambda: classification_contract(output_schema=schema(("team", FieldType.STRING))),
        lambda: classification_contract(
            input_schema=schema(("text", FieldType.STRING), ("id", FieldType.STRING))
        ),  # id column is not an input
        lambda: qa_contract(input_schema=schema(("question", FieldType.STRING))),  # no context
        lambda: classification_contract(input_schema=schema(("text", FieldType.INTEGER))),
        lambda: qa_contract(
            dataset=qa_dataset(ColumnType.NUMBER),  # column says number, schema says string
        ),
        lambda: qa_contract(
            dataset=qa_dataset(
                columns=(col("qid"), col("question"), col("passage"), col("answer", nullable=True))
            )
        ),  # required output field over a nullable target column
        lambda: classification_contract(
            dataset=make_dataset(columns=(col("id"), col("text"), col("label", ColumnType.JSON)))
        ),  # json targets cannot be typed
        # task type <-> output shape <-> evaluator
        lambda: classification_contract(
            dataset=make_dataset(
                columns=(col("id"), col("text"), col("label"), col("team")),
                target_columns=("label", "team"),
            ),
            output_schema=schema(("label", FieldType.STRING), ("team", FieldType.STRING)),
        ),
        lambda: classification_contract(evaluation=EvaluationSpec(evaluator="token_f1")),
        lambda: classification_contract(
            evaluation=EvaluationSpec(evaluator="json_schema_validity")
        ),
        lambda: qa_contract(
            evaluation=EvaluationSpec(evaluator="numeric_tolerance")
        ),  # string answer
        lambda: numeric_qa_contract(evaluation=EvaluationSpec(evaluator="token_f1")),
        lambda: extraction_contract(
            evaluation=EvaluationSpec(
                evaluator="classification_accuracy", config={"labels": ["a", "b"]}
            )
        ),
        lambda: extraction_contract(evaluation=EvaluationSpec(evaluator="token_f1")),
        lambda: extraction_contract(evaluation=EvaluationSpec(evaluator="numeric_tolerance")),
        lambda: extraction_contract(
            evaluation=EvaluationSpec(
                evaluator="legacy_field_match",
                config={"matchers": {"vendor": {"kind": "exact"}}},
            )
        ),  # legacy evaluator on a csv dataset
        # objective needs a quality floor to minimize cost / latency
        lambda: classification_contract(objective=ObjectiveSpec(mode="minimize_cost")),
        lambda: classification_contract(objective=ObjectiveSpec(mode="minimize_latency")),
        # identity / text fields
        lambda: classification_contract(instructions=""),
        lambda: classification_contract(instructions="x" * 20_001),
        lambda: classification_contract(task_id="/tasks/routing"),
        lambda: classification_contract(contract_version=0),
        lambda: classification_contract(schema_version="taskcontract/0"),
        lambda: classification_contract(owner="someone"),  # unknown field
    ],
)
def test_incompatible_contracts_fail_closed(build):
    with pytest.raises(ValidationError):
        build()


def test_missing_evaluator_fails_closed():
    c = classification_contract()
    data = c.model_dump(mode="json")
    del data["evaluation"]
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


def test_minimize_cost_is_valid_with_a_quality_floor():
    c = classification_contract(
        objective=ObjectiveSpec(mode="minimize_cost"),
        constraints=ConstraintLimits(minimum_quality=0.9),
    )
    assert c.objective.mode.value == "minimize_cost"


# -- contract identity --------------------------------------------------------------------------


def test_contract_hash_ignores_dataset_name_and_metadata():
    a = classification_contract()
    b = classification_contract(dataset=make_dataset(name="Other", metadata={"team": "x"}))
    assert a.contract_hash == b.contract_hash
    assert a.canonical_json() != b.canonical_json()


@pytest.mark.parametrize(
    "overrides",
    [
        {"instructions": "Route the ticket to one team. Prefer billing when unsure."},
        {"contract_version": 2},
        {"dataset": make_dataset(content_hash=HASH_B)},
        {
            "evaluation": EvaluationSpec(
                evaluator="classification_accuracy",
                config={"labels": ["billing", "bugs", "sales"], "case_sensitive": False},
            )
        },
        {"constraints": ConstraintLimits(minimum_quality=0.5)},
        {
            "objective": ObjectiveSpec(
                mode="balanced", weights={"quality": 0.9, "cost": 0.1}, scales={"cost": 0.01}
            )
        },
    ],
)
def test_contract_hash_changes_with_authoritative_content(overrides):
    assert (
        classification_contract(**overrides).contract_hash
        != classification_contract().contract_hash
    )


def test_contract_hash_is_independent_of_key_order():
    c = qa_contract()
    raw = json.loads(c.model_dump_json())
    reordered = json.loads(json.dumps(raw, sort_keys=True))
    assert TaskContract.model_validate(reordered).contract_hash == c.contract_hash


def test_contracts_hold_no_examples_or_labels():
    fields = set(TaskContract.model_fields) | set(
        type(classification_contract().dataset).model_fields
    )
    assert not {"rows", "examples", "values", "labels_data"} & fields


# -- experiment identity ------------------------------------------------------------------------

ROWS = tuple(f"t-{i:03d}" for i in range(100))
MODEL = ModelConfiguration(provider="local", model="gemma-4", parameters={"temperature": 0.0})


def identity(contract=None, plan=None, model=MODEL, grammar=GRAMMAR_VERSION) -> ExperimentIdentity:
    contract = contract or classification_contract()
    plan = plan or SplitPlan(seed=1, validation_bps=2000, test_bps=1000)
    splits = seeded_splits(contract.dataset.identity_hash, ROWS, plan)
    return experiment_identity(contract, splits, grammar_version=grammar, model=model)


def test_experiment_identity_is_deterministic():
    a, b = identity(), identity()
    assert a == b and a.experiment_id == b.experiment_id and len(a.experiment_id) == 64
    assert a.differences(b) == ()


def test_experiment_identity_ignores_dict_order_and_non_authoritative_metadata():
    reordered_model = ModelConfiguration.model_validate(
        {"parameters": {"temperature": 0.0}, "model": "gemma-4", "provider": "local"}
    )
    relabelled = classification_contract(dataset=make_dataset(name="x", metadata={"note": "y"}))
    assert identity(model=reordered_model).experiment_id == identity().experiment_id
    assert identity(contract=relabelled).experiment_id == identity().experiment_id


@pytest.mark.parametrize(
    ("kwargs", "changed"),
    [
        (
            {"contract": classification_contract(dataset=make_dataset(content_hash=HASH_B))},
            "dataset_hash",
        ),
        ({"plan": SplitPlan(seed=2, validation_bps=2000, test_bps=1000)}, "splits_hash"),
        ({"contract": classification_contract(contract_version=2)}, "contract_version"),
        ({"grammar": "grammar/2"}, "grammar_version"),
        (
            {
                "model": ModelConfiguration(
                    provider="local", model="gemma-4", parameters={"temperature": 0.7}
                )
            },
            "model_config_hash",
        ),
        (
            {
                "contract": classification_contract(
                    evaluation=EvaluationSpec(evaluator="exact_match")
                )
            },
            "evaluation_hash",
        ),
        (
            {
                "contract": classification_contract(
                    objective=ObjectiveSpec(
                        mode="balanced", weights={"quality": 0.9, "cost": 0.1}, scales={"cost": 1.0}
                    )
                )
            },
            "objective_hash",
        ),
        (
            {"contract": classification_contract(constraints=ConstraintLimits(maximum_retries=1))},
            "constraints_hash",
        ),
    ],
)
def test_experiment_identity_changes_with_each_meaningful_component(kwargs, changed):
    other = identity(**kwargs)
    assert other.experiment_id != identity().experiment_id
    assert changed in identity().differences(other)


def test_splits_for_another_dataset_are_rejected():
    contract = classification_contract()
    splits = seeded_splits(HASH_B, ROWS, SplitPlan(seed=1, validation_bps=0, test_bps=0))
    with pytest.raises(ValueError, match="different dataset"):
        experiment_identity(contract, splits, grammar_version=GRAMMAR_VERSION, model=MODEL)
