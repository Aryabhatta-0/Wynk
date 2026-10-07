"""DatasetSpec validation + identity, and the split contract (incl. the test-split boundary)."""

import json

import pytest
from pydantic import ValidationError

from core.dataset import (
    ALLOWED_USES,
    ColumnType,
    DatasetSplit,
    DatasetSplits,
    OptimizerSplitView,
    SplitAccessError,
    SplitMethod,
    SplitPlan,
    SplitRole,
    SplitUse,
    require_use,
    seeded_splits,
)
from tests.contract_helpers import HASH_A, HASH_B, col, make_dataset

# -- DatasetSpec --------------------------------------------------------------------------------


def test_valid_dataset_spec():
    ds = make_dataset(context_columns=(), metadata={"owner": "support", "rows_checked": 100})
    assert ds.schema_version == "datasetspec/1"
    assert ds.column("label").type is ColumnType.STRING
    assert len(ds.identity_hash) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"columns": ()},
        {"columns": (col("id"), col("text"), col("text"), col("label"))},  # duplicate column
        {"input_columns": ("missing",)},  # unknown column
        {"target_columns": ("missing",)},
        {"context_columns": ("missing",)},
        {"input_columns": ()},  # nothing to feed the workflow
        {"target_columns": ()},  # nothing to evaluate against
        {"input_columns": ("text", "text")},  # repeated within a role
        {"target_columns": ("text",)},  # same column as input and target
        {"input_columns": ("id",)},  # id column reused as input
        {"id_column": "nope"},
        {"columns": (col("id", ColumnType.NUMBER), col("text"), col("label"))},  # float ids
        {"columns": (col("id", nullable=True), col("text"), col("label"))},  # nullable ids
        {  # column names must be identifiers (they map 1:1 onto schema fields)
            "columns": ({"name": "bad name", "type": "string"}, col("text"), col("label")),
            "id_column": None,
        },
        {"content_hash": "abc"},
        {"content_hash": "A" * 64},  # not lowercase hex
        {"dataset_id": "/data/tickets.csv"},  # a path is never identity
        {"dataset_id": "C:\\data\\tickets.csv"},
        {"dataset_id": ""},
        {"dataset_version": 0},
        {"row_count": 0},
        {"format": "xlsx"},
        {"metadata": {"score": float("nan")}},
        {"metadata": {"bad key": 1}},
        {"metadata": {f"k{i}": i for i in range(33)}},
        {"schema_version": "datasetspec/2"},
        {"surprise": True},  # unknown field
    ],
)
def test_invalid_dataset_spec_fails_closed(overrides):
    with pytest.raises(ValidationError):
        make_dataset(**overrides)


def test_identity_ignores_name_and_metadata_but_serialization_keeps_them():
    a = make_dataset(name="Tickets v1", metadata={"owner": "a"})
    b = make_dataset(name="Renamed", metadata={"owner": "b", "note": "x"})
    assert a.identity_hash == b.identity_hash
    assert a.canonical_json() != b.canonical_json()


def test_identity_and_serialization_are_independent_of_key_order():
    ds = make_dataset(metadata={"b": 1, "a": 2})
    raw = json.loads(ds.model_dump_json())
    reordered = json.loads(json.dumps(raw, sort_keys=True))
    reordered["metadata"] = {"a": 2, "b": 1}
    again = type(ds).model_validate(reordered)
    assert again == ds
    assert again.identity_hash == ds.identity_hash
    assert again.canonical_json() == ds.canonical_json()


@pytest.mark.parametrize(
    "overrides",
    [
        {"content_hash": HASH_B},  # data bytes changed
        {"dataset_version": 2},  # new version of the same dataset
        {"row_count": 101},
        {"format": "jsonl"},
        {"columns": (col("id"), col("text"), col("label", nullable=True))},
        {"id_column": None},
        {"dataset_id": "support-tickets-eu"},
        {
            "columns": (col("id"), col("text"), col("lang"), col("label")),
            "context_columns": ("lang",),
        },
    ],
)
def test_identity_changes_with_any_authoritative_change(overrides):
    assert make_dataset(**overrides).identity_hash != make_dataset().identity_hash


def test_dataset_spec_is_immutable():
    ds = make_dataset()
    with pytest.raises(ValidationError):
        ds.row_count = 5


# -- deterministic splitting --------------------------------------------------------------------

ROWS = tuple(f"row-{i:03d}" for i in range(100))
PLAN = SplitPlan(seed=7, validation_bps=2000, test_bps=1000)


def test_seeded_split_sizes_disjointness_and_coverage():
    s = seeded_splits(HASH_A, ROWS, PLAN)
    sizes = {x.role: len(x.row_ids) for x in s.splits}
    assert sizes == {SplitRole.OPTIMIZATION: 70, SplitRole.VALIDATION: 20, SplitRole.TEST: 10}
    all_rows = [r for x in s.splits for r in x.row_ids]
    assert sorted(all_rows) == sorted(ROWS) and len(set(all_rows)) == len(all_rows)
    assert s.split(SplitRole.TEST).is_final
    assert not s.split(SplitRole.VALIDATION).is_final


def test_seeded_split_is_independent_of_input_order_and_reproducible():
    a = seeded_splits(HASH_A, ROWS, PLAN)
    b = seeded_splits(HASH_A, tuple(reversed(ROWS)), PLAN)
    assert a == b and a.identity_hash == b.identity_hash
    assert DatasetSplits.model_validate_json(a.model_dump_json()) == a


def test_seed_changes_the_assignment():
    a = seeded_splits(HASH_A, ROWS, PLAN)
    b = seeded_splits(HASH_A, ROWS, PLAN.model_copy(update={"seed": 8}))
    assert a.split(SplitRole.TEST).row_ids != b.split(SplitRole.TEST).row_ids
    assert a.identity_hash != b.identity_hash


def test_tiny_dataset_keeps_everything_for_optimization():
    s = seeded_splits(HASH_A, ("r1", "r2", "r3"), PLAN)  # 3 * 10% and 3 * 20% round down to 0
    assert [x.role for x in s.splits] == [SplitRole.OPTIMIZATION]


def test_tampered_seeded_split_is_rejected():
    s = seeded_splits(HASH_A, ROWS, PLAN)
    raw = json.loads(s.model_dump_json())
    by_role = {x["role"]: x for x in raw["splits"]}
    moved = by_role["test"]["row_ids"].pop()
    by_role["optimization"]["row_ids"].append(moved)  # leak a test row into optimization
    with pytest.raises(ValidationError, match="seeded plan"):
        DatasetSplits.model_validate(raw)


@pytest.mark.parametrize(
    "plan",
    [
        {"seed": 0, "validation_bps": 5000, "test_bps": 5000},  # nothing left to optimize on
        {"seed": 0, "validation_bps": -1, "test_bps": 0},
        {"seed": 0, "validation_bps": 0, "test_bps": 10_000},
        {"seed": 0, "validation_bps": 0.5, "test_bps": 0},
    ],
)
def test_invalid_split_plans_rejected(plan):
    with pytest.raises(ValidationError):
        SplitPlan(**plan)


def _split(role, rows, split_id=None):
    return DatasetSplit(split_id=split_id or role.value, role=role, row_ids=rows)


@pytest.mark.parametrize(
    "splits",
    [
        (_split(SplitRole.VALIDATION, ("a",)),),  # no optimization split
        (_split(SplitRole.OPTIMIZATION, ("a",)), _split(SplitRole.OPTIMIZATION, ("b",), "o2")),
        (_split(SplitRole.OPTIMIZATION, ("a", "b")), _split(SplitRole.TEST, ("b",))),  # overlap
        (_split(SplitRole.OPTIMIZATION, ("a",)), _split(SplitRole.TEST, ("b",), "optimization")),
        (
            _split(SplitRole.OPTIMIZATION, ("a",)),
            _split(SplitRole.TEST, ("b",)),
            _split(SplitRole.TEST, ("c",), "t2"),
        ),
    ],
)
def test_invalid_explicit_splits_rejected(splits):
    with pytest.raises(ValidationError):
        DatasetSplits(dataset_hash=HASH_A, method=SplitMethod.EXPLICIT, splits=splits)


def test_split_rows_must_be_unique_and_are_stored_sorted():
    with pytest.raises(ValidationError):
        _split(SplitRole.OPTIMIZATION, ("a", "a"))
    assert _split(SplitRole.OPTIMIZATION, ("b", "a")).row_ids == ("a", "b")


def test_explicit_splits_may_not_carry_a_plan_and_seeded_splits_need_one():
    one = (_split(SplitRole.OPTIMIZATION, ("a",)),)
    with pytest.raises(ValidationError):
        DatasetSplits(dataset_hash=HASH_A, method=SplitMethod.EXPLICIT, plan=PLAN, splits=one)
    with pytest.raises(ValidationError):
        DatasetSplits(dataset_hash=HASH_A, method=SplitMethod.SEEDED_HASH, splits=one)


# -- the optimization / validation / test authority boundary ------------------------------------


def test_use_policy_is_exactly_the_documented_table():
    assert ALLOWED_USES == {
        SplitRole.OPTIMIZATION: {
            SplitUse.OPTIMIZER_FEEDBACK,
            SplitUse.SELECTION,
            SplitUse.REPORTING,
        },
        SplitRole.VALIDATION: {SplitUse.SELECTION, SplitUse.REPORTING},
        SplitRole.TEST: {SplitUse.REPORTING, SplitUse.PROMOTION_GATE},
    }
    for use in (SplitUse.OPTIMIZER_FEEDBACK, SplitUse.SELECTION):
        with pytest.raises(SplitAccessError):
            require_use(SplitRole.TEST, use)
    with pytest.raises(SplitAccessError):
        require_use(SplitRole.VALIDATION, SplitUse.OPTIMIZER_FEEDBACK)
    # the binary held-out promotion gate belongs to the final test split alone
    for role in (SplitRole.OPTIMIZATION, SplitRole.VALIDATION):
        with pytest.raises(SplitAccessError):
            require_use(role, SplitUse.PROMOTION_GATE)
    require_use(SplitRole.TEST, SplitUse.REPORTING)  # reporting the final test is fine
    require_use(SplitRole.TEST, SplitUse.PROMOTION_GATE)


def test_optimizer_view_structurally_excludes_the_final_test_split():
    s = seeded_splits(HASH_A, ROWS, PLAN)
    view = s.optimizer_view()
    assert set(OptimizerSplitView.model_fields) == {
        "dataset_hash",
        "optimization_row_ids",
        "validation_row_ids",
    }
    visible = set(view.optimization_row_ids) | set(view.validation_row_ids)
    assert not visible & set(s.split(SplitRole.TEST).row_ids)
    assert visible | set(s.split(SplitRole.TEST).row_ids) == set(ROWS)
    with pytest.raises(SplitAccessError):
        s.rows_for(SplitRole.TEST, SplitUse.OPTIMIZER_FEEDBACK)
    assert s.rows_for(SplitRole.TEST, SplitUse.REPORTING) == s.split(SplitRole.TEST).row_ids


def test_feedback_gate_admits_only_optimization_rows():
    s = seeded_splits(HASH_A, ROWS, PLAN)
    s.check_feedback(s.split(SplitRole.OPTIMIZATION).row_ids)  # allowed
    s.check_feedback(())  # nothing fed back is trivially fine
    for role in (SplitRole.VALIDATION, SplitRole.TEST):
        row = s.split(role).row_ids[0]
        with pytest.raises(SplitAccessError):
            s.check_feedback([s.split(SplitRole.OPTIMIZATION).row_ids[0], row])
    with pytest.raises(SplitAccessError, match="no split"):
        s.check_feedback(["not-a-row"])
