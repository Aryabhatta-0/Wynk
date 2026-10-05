"""Ingestion service: DatasetSpec construction, identity, role mapping, versions and splits."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    SplitAccessError,
    SplitPlan,
    SplitRole,
    SplitUse,
    seeded_splits,
)
from core.evaluation_spec import EvaluationSpec
from core.task_contract import TaskContract
from core.task_spec import AnswerField, AnswerSchema
from ingestion.parse import ROW_ID_SCHEME, IngestError
from ingestion.service import DatasetService, NewProject, NotFound, RegisterDataset
from store.blobs import LocalBlobStore
from store.datasets import RowIdSource, SQLiteDatasetRepository

ROWS = [
    {"qid": f"q{i:03d}", "question": f"question {i}", "passage": f"p{i}", "answer": f"a{i % 3}"}
    for i in range(60)
]
JSONL = "".join(json.dumps(r) + "\n" for r in ROWS).encode()
CSV = b"id,text,label,notes\n" + b"".join(
    f"{i},text {i},{'ab'[i % 2]},{'' if i % 4 else 'n'}\n".encode() for i in range(30)
)
PLAN = SplitPlan(seed=11, validation_bps=2000, test_bps=1500)


def make_service(root, clock=lambda: "2026-10-05T00:00:00+00:00"):
    return DatasetService(
        SQLiteDatasetRepository(root / "metadata.sqlite3"),
        LocalBlobStore(root / "blobs"),
        None,
        clock,
    )


@pytest.fixture
def svc(tmp_path):
    return make_service(tmp_path)


@pytest.fixture
def project(svc):
    return svc.create_project(NewProject(name="Support"))


def qa_request(**kw) -> RegisterDataset:
    req = {
        "dataset_id": "reading-qa",
        "name": "Reading QA",
        "input_columns": ("question",),
        "context_columns": ("passage",),
        "target_columns": ("answer",),
        "row_ids": "column",
        "id_column": "qid",
    } | kw
    return RegisterDataset(**req)


def csv_request(**kw) -> RegisterDataset:
    req = {
        "dataset_id": "tickets",
        "name": "Tickets",
        "input_columns": ("text",),
        "target_columns": ("label",),
        "row_ids": "column",
        "id_column": "id",
    } | kw
    return RegisterDataset(**req)


def code_of(fn, *args) -> str:
    with pytest.raises(IngestError) as info:
        fn(*args)
    return info.value.code


# -- upload -------------------------------------------------------------------------------------
def test_upload_inspection_is_computed_from_bytes(svc, project):
    upload, created = svc.upload(project.project_id, CSV, "csv", "tickets.csv")
    assert created
    assert upload.content_hash == hashlib.sha256(CSV).hexdigest()
    assert upload.size_bytes == len(CSV) and upload.row_count == 30
    assert [(c.name, c.type, c.nullable) for c in upload.columns] == [
        ("id", ColumnType.INTEGER, False),
        ("text", ColumnType.STRING, False),
        ("label", ColumnType.STRING, False),
        ("notes", ColumnType.STRING, True),
    ]
    assert len(upload.preview) == 20 and upload.preview[0]["text"] == "text 0"


def test_malformed_upload_stores_nothing(tmp_path, svc, project):
    assert code_of(svc.upload, project.project_id, b"id,text\n1\n", "csv") == "malformed_csv"
    assert [p for p in tmp_path.rglob("*") if p.is_file() and "blobs" in p.parts] == []


def test_upload_to_unknown_project_is_refused(svc):
    with pytest.raises(NotFound) as info:
        svc.upload("p-missing", CSV, "csv")
    assert info.value.code == "project_not_found"


# -- registration -> DatasetSpec ----------------------------------------------------------------
def test_register_builds_the_real_dataset_spec(svc, project):
    upload, _ = svc.upload(project.project_id, JSONL, "jsonl", "qa.jsonl")
    record, created = svc.register(upload.upload_id, qa_request())
    assert created
    spec = record.spec
    assert isinstance(spec, DatasetSpec)
    assert spec == DatasetSpec(
        dataset_id="reading-qa",
        dataset_version=1,
        name="Reading QA",
        content_hash=hashlib.sha256(JSONL).hexdigest(),
        format=DatasetFormat.JSONL,
        columns=tuple(ColumnSpec(name=n, type=ColumnType.STRING) for n in ROWS[0]),
        id_column="qid",
        input_columns=("question",),
        target_columns=("answer",),
        context_columns=("passage",),
        row_count=60,
        metadata={"row_ids": "column"},
    )
    assert record.identity_hash == spec.identity_hash
    assert record.row_id_source is RowIdSource.COLUMN
    assert svc.repo.get_row_ids("reading-qa", 1) == tuple(r["qid"] for r in ROWS)


def test_registered_spec_feeds_a_task_contract(svc, project):
    """Task-specific validation stays in TaskContract; the registered spec plugs straight in."""
    upload, _ = svc.upload(project.project_id, JSONL, "jsonl")
    spec = svc.register(upload.upload_id, qa_request())[0].spec
    contract = TaskContract(
        task_id="reading-qa",
        contract_version=1,
        task_type="question_answering",
        instructions="Answer from the passage.",
        input_schema=AnswerSchema(
            fields=(
                AnswerField(name="question", type="string"),
                AnswerField(name="passage", type="string"),
            )
        ),
        output_schema=AnswerSchema(fields=(AnswerField(name="answer", type="string"),)),
        dataset=spec,
        evaluation=EvaluationSpec(evaluator="token_f1", config={"pass_threshold": 0.8}),
    )
    assert contract.dataset.identity_hash == spec.identity_hash


def test_generated_row_ids(svc, project):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    record, _ = svc.register(upload.upload_id, csv_request(row_ids="generated", id_column=None))
    assert record.spec.id_column is None
    assert record.row_id_source is RowIdSource.GENERATED
    assert record.row_id_scheme == ROW_ID_SCHEME
    assert record.spec.metadata == {"row_ids": "generated", "row_id_scheme": ROW_ID_SCHEME}
    ids = svc.repo.get_row_ids("tickets", 1)
    assert len(ids) == len(set(ids)) == 30


def test_generated_row_ids_are_stable_across_services(tmp_path, svc, project):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    svc.register(upload.upload_id, csv_request(row_ids="generated", id_column=None))
    other = make_service(tmp_path / "elsewhere")
    p2 = other.create_project(NewProject(name="Other"))
    u2, _ = other.upload(p2.project_id, CSV, "csv")
    other.register(u2.upload_id, csv_request(row_ids="generated", id_column=None))
    assert other.repo.get_row_ids("tickets", 1) == svc.repo.get_row_ids("tickets", 1)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"input_columns": ("missing",)}, "unknown_column"),
        ({"context_columns": ("label",)}, "invalid_mapping"),  # label is already the target
        ({"input_columns": ("text", "text")}, "invalid_mapping"),
        ({"input_columns": ("text", "id")}, "invalid_mapping"),  # id column reused as input
        ({"id_column": "notes"}, "invalid_id_column"),  # nullable
        (  # label is not unique
            {"id_column": "label", "input_columns": ("notes",), "target_columns": ("text",)},
            "duplicate_row_id",
        ),
    ],
)
def test_invalid_mappings(svc, project, overrides, code):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    assert code_of(svc.register, upload.upload_id, csv_request(**overrides)) == code
    assert svc.repo.list_versions("tickets") == []  # nothing half-registered


def test_json_columns_cannot_take_a_role(svc, project):
    data = b'{"id": 1, "doc": {"a": 1}, "y": "x"}\n{"id": 2, "doc": [1], "y": "z"}\n'
    upload, _ = svc.upload(project.project_id, data, "jsonl")
    assert upload.columns[1].type is ColumnType.JSON
    for req in (
        csv_request(input_columns=("doc",), target_columns=("y",)),
        csv_request(input_columns=("y",), target_columns=("doc",)),
        csv_request(input_columns=("y",), target_columns=("id",), context_columns=("doc",)),
    ):
        assert code_of(svc.register, upload.upload_id, req) == "json_column_role"
    # unmapped json columns are fine: they stay in the spec, typed json, with no role
    record, _ = svc.register(
        upload.upload_id,
        csv_request(
            input_columns=("y",), target_columns=("id",), id_column=None, row_ids="generated"
        ),
    )
    assert record.spec.column("doc").type is ColumnType.JSON


def test_register_request_is_strict():
    with pytest.raises(ValidationError):  # client-supplied facts about the data are refused
        RegisterDataset(**qa_request().model_dump(), content_hash="0" * 64)
    with pytest.raises(ValidationError):
        RegisterDataset(**qa_request().model_dump(), row_count=3)
    with pytest.raises(ValidationError):
        qa_request(row_ids="generated")  # an id column with generated ids is contradictory
    with pytest.raises(ValidationError):
        qa_request(id_column=None)  # column ids without a column
    with pytest.raises(ValidationError):
        qa_request(dataset_id="../escape")


# -- identity + versions ------------------------------------------------------------------------
def test_same_bytes_and_mapping_give_the_same_version(svc, project):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    first, created = svc.register(upload.upload_id, csv_request())
    again, created_again = svc.register(upload.upload_id, csv_request(name="Renamed"))
    assert created and not created_again
    assert again == first  # display name is not identity; nothing new is stored


def test_new_content_or_mapping_is_a_new_immutable_version(svc, project):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    v1, _ = svc.register(upload.upload_id, csv_request())
    v2, _ = svc.register(upload.upload_id, csv_request(context_columns=("notes",)))
    changed = CSV.replace(b"text 3,", b"text three,")
    u2, _ = svc.upload(project.project_id, changed, "csv")
    v3, _ = svc.register(u2.upload_id, csv_request())
    assert [v.dataset_version for v in (v1, v2, v3)] == [1, 2, 3]
    assert len({v1.identity_hash, v2.identity_hash, v3.identity_hash}) == 3
    assert v3.spec.content_hash != v1.spec.content_hash
    assert svc.get_version("tickets", 1) == v1  # earlier versions are unchanged


def test_identity_is_deterministic_across_fresh_stores(tmp_path):
    def run(root):
        s = make_service(root, clock=lambda: str(root))  # timestamps differ; identity must not
        p = s.create_project(NewProject(name="X"))
        u, _ = s.upload(p.project_id, CSV, "csv")
        v, _ = s.register(u.upload_id, csv_request())
        sp, _ = s.create_splits("tickets", 1, PLAN)
        return u.content_hash, v.identity_hash, sp.splits_hash, sp.splits

    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_dataset_id_belongs_to_one_project(svc, project):
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    svc.register(upload.upload_id, csv_request())
    other = svc.create_project(NewProject(name="Other"))
    u2, _ = svc.upload(other.project_id, CSV, "csv")
    assert code_of(svc.register, u2.upload_id, csv_request()) == "dataset_conflict"


# -- splits -------------------------------------------------------------------------------------
def test_splits_reproduce_core_seeded_splits(svc, project):
    upload, _ = svc.upload(project.project_id, JSONL, "jsonl")
    version, _ = svc.register(upload.upload_id, qa_request())
    record, created = svc.create_splits("reading-qa", 1, PLAN)
    assert created
    expected = seeded_splits(version.identity_hash, tuple(r["qid"] for r in ROWS), PLAN)
    assert record.splits == expected
    assert record.splits_hash == expected.identity_hash
    assert record.sizes == {"optimization": 39, "test": 9, "validation": 12}
    again, created_again = svc.create_splits("reading-qa", 1, PLAN)
    assert again == record and not created_again
    other, _ = svc.create_splits("reading-qa", 1, PLAN.model_copy(update={"seed": 12}))
    assert other.splits_hash != record.splits_hash
    assert [s.splits_hash for s in svc.list_splits("reading-qa", 1)] == sorted(
        [record.splits_hash, other.splits_hash]
    )


def test_validation_and_test_rows_stay_isolated_after_reload(tmp_path, svc, project):
    upload, _ = svc.upload(project.project_id, JSONL, "jsonl")
    svc.register(upload.upload_id, qa_request())
    record, _ = svc.create_splits("reading-qa", 1, PLAN)
    splits = make_service(tmp_path).get_splits("reading-qa", 1, record.splits_hash).splits
    test_rows = set(splits.split(SplitRole.TEST).row_ids)
    val_rows = set(splits.split(SplitRole.VALIDATION).row_ids)
    opt_rows = set(splits.split(SplitRole.OPTIMIZATION).row_ids)
    assert not (test_rows & val_rows) and not (test_rows & opt_rows) and not (val_rows & opt_rows)
    assert test_rows | val_rows | opt_rows == {r["qid"] for r in ROWS}
    view = splits.optimizer_view()
    assert "test_row_ids" not in view.model_dump()
    assert not test_rows & (set(view.optimization_row_ids) | set(view.validation_row_ids))
    splits.check_feedback(opt_rows)
    for row in (next(iter(val_rows)), next(iter(test_rows))):
        with pytest.raises(SplitAccessError):
            splits.check_feedback([row])
    with pytest.raises(SplitAccessError):
        splits.rows_for(SplitRole.TEST, SplitUse.SELECTION)


def test_tiny_datasets_always_keep_an_optimization_row(svc, project):
    upload, _ = svc.upload(project.project_id, b"id,t,y\n1,a,b\n", "csv")
    svc.register(upload.upload_id, csv_request(input_columns=("t",), target_columns=("y",)))
    plan = SplitPlan(seed=0, validation_bps=5000, test_bps=4999)
    assert svc.create_splits("tickets", 1, plan)[0].sizes == {"optimization": 1}
    upload2, _ = svc.upload(project.project_id, b"id,t,y\n1,a,b\n2,c,d\n", "csv")
    svc.register(upload2.upload_id, csv_request(input_columns=("t",), target_columns=("y",)))
    plan2 = SplitPlan(seed=0, validation_bps=5000, test_bps=4999)
    assert svc.create_splits("tickets", 2, plan2)[0].sizes == {"optimization": 1, "validation": 1}


def test_splits_for_unknown_version(svc, project):
    with pytest.raises(NotFound) as info:
        svc.create_splits("nope", 1, PLAN)
    assert info.value.code == "dataset_not_found"
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    svc.register(upload.upload_id, csv_request())
    with pytest.raises(NotFound) as info:
        svc.create_splits("tickets", 2, PLAN)
    assert info.value.code == "dataset_version_not_found"
