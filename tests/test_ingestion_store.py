"""Durable storage: content-addressed blobs and the SQLite metadata repository."""

import hashlib
import json
import sqlite3

import pytest

from core.dataset import SplitPlan
from ingestion.service import DatasetService, NewProject, RegisterDataset, StorageFailure
from store.blobs import BlobCorrupted, BlobNotFound, LocalBlobStore
from store.datasets import IntegrityViolation, SQLiteDatasetRepository

CSV = b"id,text,label\n" + b"".join(f"{i},text {i},{'ab'[i % 2]}\n".encode() for i in range(40))


def make_service(root):
    return DatasetService(
        SQLiteDatasetRepository(root / "metadata.sqlite3"), LocalBlobStore(root / "blobs")
    )


def register(svc, upload_id, dataset_id="tickets", **kw):
    req = {
        "dataset_id": dataset_id,
        "name": "Tickets",
        "input_columns": ("text",),
        "target_columns": ("label",),
        "row_ids": "column",
        "id_column": "id",
    } | kw
    return svc.register(upload_id, RegisterDataset(**req))


# -- blobs --------------------------------------------------------------------------------------
def test_blob_is_content_addressed_and_deduplicated(tmp_path):
    store = LocalBlobStore(tmp_path)
    first = store.put(b"hello")
    assert first.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert first.created and first.size_bytes == 5
    again = store.put(b"hello")
    assert again.sha256 == first.sha256 and not again.created
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].name == first.sha256  # one blob, no temp leftovers
    assert store.get(first.sha256) == b"hello"


def test_blob_paths_never_leave_the_store(tmp_path):
    store = LocalBlobStore(tmp_path)
    for bad in ("../../etc/passwd", "A" * 64, "abc"):
        with pytest.raises(ValueError):
            store.get(bad)


def test_missing_and_corrupted_blobs_are_refused(tmp_path):
    store = LocalBlobStore(tmp_path)
    with pytest.raises(BlobNotFound):
        store.get("0" * 64)
    ref = store.put(b"original")
    path = next(p for p in tmp_path.rglob(ref.sha256))
    path.write_bytes(b"tampered")
    with pytest.raises(BlobCorrupted):
        store.get(ref.sha256)


def test_verify_rehashes_instead_of_checking_existence(tmp_path):
    store = LocalBlobStore(tmp_path)
    with pytest.raises(BlobNotFound):
        store.verify("0" * 64)
    ref = store.put(b"original")
    store.verify(ref.sha256)
    next(tmp_path.rglob(ref.sha256)).write_bytes(b"tampered")  # same size
    with pytest.raises(BlobCorrupted):
        store.verify(ref.sha256)
    assert not hasattr(store, "exists")  # no existence-only check to trust by mistake


def test_same_size_corrupted_blob_is_not_a_dedup_hit(tmp_path):
    """Policy: put() replaces a corrupt blob atomically with the supplied (hash-matching) bytes."""
    store = LocalBlobStore(tmp_path)
    ref = store.put(b"original")
    path = next(tmp_path.rglob(ref.sha256))
    path.write_bytes(b"tampered")
    assert path.stat().st_size == ref.size_bytes
    again = store.put(b"original")
    assert again.sha256 == ref.sha256 and again.created  # rewritten, not deduplicated
    assert path.read_bytes() == b"original"
    store.verify(ref.sha256)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [path]  # no temp leftovers


def test_missing_blob_is_rewritten_by_put(tmp_path):
    store = LocalBlobStore(tmp_path)
    ref = store.put(b"original")
    next(tmp_path.rglob(ref.sha256)).unlink()
    assert store.put(b"original").created
    assert store.get(ref.sha256) == b"original"


def test_healthy_duplicate_is_deduplicated_without_rewriting(tmp_path):
    store = LocalBlobStore(tmp_path)
    ref = store.put(b"original")
    path = next(tmp_path.rglob(ref.sha256))
    before = path.stat()
    again = store.put(b"original")
    assert again == ref._replace(created=False)
    after = path.stat()
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)


def test_failed_blob_write_leaves_no_partial_file(tmp_path, monkeypatch):
    store = LocalBlobStore(tmp_path)

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr("store.blobs.os.replace", boom)
    with pytest.raises(OSError):
        store.put(b"data")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


# -- repository persistence ---------------------------------------------------------------------
def test_state_survives_repository_and_service_recreation(tmp_path):
    svc = make_service(tmp_path)
    project = svc.create_project(NewProject(name="Support"))
    upload, _ = svc.upload(project.project_id, CSV, "csv", "tickets.csv")
    version, _ = register(svc, upload.upload_id)
    splits, _ = svc.create_splits(
        "tickets", 1, SplitPlan(seed=7, validation_bps=2000, test_bps=2000)
    )
    del svc

    reopened = make_service(tmp_path)  # fresh repository + blob store objects, same directory
    assert reopened.get_project(project.project_id) == project
    assert reopened.get_upload(upload.upload_id) == upload
    assert reopened.get_version("tickets", 1) == version
    assert reopened.get_splits("tickets", 1, splits.splits_hash) == splits
    assert reopened.repo.get_row_ids("tickets", 1) == tuple(str(i) for i in range(40))
    assert [v.dataset_version for v in reopened.list_datasets(project.project_id)[0]] == [1]


def test_duplicate_uploads_reuse_one_blob(tmp_path):
    svc = make_service(tmp_path)
    a = svc.create_project(NewProject(name="A"))
    b = svc.create_project(NewProject(name="B"))
    u1, created1 = svc.upload(a.project_id, CSV, "csv", "one.csv")
    u2, created2 = svc.upload(a.project_id, CSV, "csv", "two.csv")
    u3, created3 = svc.upload(b.project_id, CSV, "csv")
    assert created1 and not created2 and created3
    assert u1 == u2  # same project, same bytes, same format -> same upload
    assert u3.upload_id != u1.upload_id and u3.content_hash == u1.content_hash
    blobs = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert [p.name for p in blobs] == [u1.content_hash]


def _tamper(db, sql, *params):
    conn = sqlite3.connect(db)
    with conn:
        conn.execute(sql, params)
    conn.close()


def test_tampered_spec_is_refused_on_read(tmp_path):
    svc = make_service(tmp_path)
    project = svc.create_project(NewProject(name="Support"))
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    version, _ = register(svc, upload.upload_id)
    record = json.loads(version.model_dump_json())
    record["spec"]["row_count"] = 39  # still a valid DatasetSpec, but not the stored identity
    _tamper(
        tmp_path / "metadata.sqlite3",
        "UPDATE dataset_versions SET record_json=?",
        json.dumps(record),
    )
    with pytest.raises(IntegrityViolation):
        make_service(tmp_path).get_version("tickets", 1)


def test_tampered_row_ids_and_splits_are_refused(tmp_path):
    svc = make_service(tmp_path)
    project = svc.create_project(NewProject(name="Support"))
    upload, _ = svc.upload(project.project_id, CSV, "csv")
    register(svc, upload.upload_id)
    splits, _ = svc.create_splits("tickets", 1, SplitPlan(seed=1, validation_bps=0, test_bps=2500))
    db = tmp_path / "metadata.sqlite3"

    record = json.loads(splits.model_dump_json())
    test, opt = (record["splits"]["splits"][i] for i in (1, 0))
    assert test["role"] == "test" and opt["role"] == "optimization"
    test["row_ids"], opt["row_ids"][0] = (  # move one test row into optimization
        test["row_ids"][1:],
        test["row_ids"][0],
    )
    _tamper(db, "UPDATE dataset_splits SET record_json=?", json.dumps(record))
    with pytest.raises(IntegrityViolation):  # DatasetSplits re-derives the seeded plan on load
        make_service(tmp_path).get_splits("tickets", 1, splits.splits_hash)

    _tamper(db, "UPDATE dataset_versions SET row_ids_json=?", json.dumps(["x"] * 40))
    with pytest.raises(IntegrityViolation):
        make_service(tmp_path).repo.get_row_ids("tickets", 1)


# -- service: blob integrity is checked on every upload/read path --------------------------------
def _uploaded(tmp_path):
    svc = make_service(tmp_path)
    project = svc.create_project(NewProject(name="Support"))
    upload, created = svc.upload(project.project_id, CSV, "csv")
    assert created
    return svc, project, upload, next((tmp_path / "blobs").rglob(upload.content_hash))


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_get_upload_fails_closed_on_a_damaged_blob(tmp_path, damage):
    svc, _, upload, blob = _uploaded(tmp_path)
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_bytes(bytes(len(CSV)))  # same size, wrong bytes
    with pytest.raises(StorageFailure):
        svc.get_upload(upload.upload_id)
    with pytest.raises(StorageFailure):
        make_service(tmp_path).get_upload(upload.upload_id)  # also after a restart


def test_registered_version_reads_fail_closed_on_a_corrupted_blob(tmp_path):
    svc, project, upload, blob = _uploaded(tmp_path)
    register(svc, upload.upload_id)
    blob.write_bytes(bytes(len(CSV)))
    for read in (
        lambda: svc.get_version("tickets", 1),
        lambda: svc.get_dataset("tickets"),
        lambda: svc.list_datasets(project.project_id),
        lambda: svc.list_splits("tickets", 1),
        lambda: svc.create_splits("tickets", 1, SplitPlan(seed=1, validation_bps=0, test_bps=0)),
    ):
        with pytest.raises(StorageFailure):
            read()


def test_corrupted_blob_then_identical_upload_restores_it(tmp_path):
    svc, project, upload, blob = _uploaded(tmp_path)
    blob.write_bytes(bytes(len(CSV)))
    again, created = svc.upload(project.project_id, CSV, "csv")
    assert again == upload and not created  # the existing record, not a new upload
    assert blob.read_bytes() == CSV  # replaced atomically with the exact supplied bytes
    assert svc.get_upload(upload.upload_id) == upload


def test_healthy_duplicate_upload_still_deduplicates(tmp_path):
    svc, project, upload, blob = _uploaded(tmp_path)
    before = blob.stat()
    again, created = svc.upload(project.project_id, CSV, "csv")
    assert again == upload and not created
    after = blob.stat()
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()] == [blob]
