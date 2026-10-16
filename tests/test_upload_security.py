"""#33 uploaded-data security and privacy: adversarial and regression tests.

Bounds, safe parsing, filenames, CSV formula injection, secrets / PII policy, encryption at
rest (round trip, tamper, rotation, missing key), tenant-scoped export and deletion, refusal to
serve deleted data, the audit trail (completeness, append-only, redaction), identity / split
reproducibility and the no-user-data-training boundary.
"""

from __future__ import annotations

import ast
import csv as csv_module
import hashlib
import io
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from api import tenancy
from api.product import ERROR_STATUS, build_api
from api.tenancy import KEY_PREFIX, TenantRouter, parse_key
from core.dataset import SplitPlan
from ingestion import export as exports
from ingestion.export import is_formula_like, neutralize_cell, safe_csv
from ingestion.parse import IngestError, IngestLimits, parse_dataset
from ingestion.sensitive import Category, SensitiveDataPolicy, find
from ingestion.service import (
    DatasetService,
    NewProject,
    RegisterDataset,
    SourceDeleted,
    StorageFailure,
)
from store.audit import SQLiteAuditLog
from store.blobs import (
    BlobCorrupted,
    BlobKeyUnavailable,
    BlobNotEncrypted,
    BlobNotFound,
    LocalBlobStore,
)
from store.crypto import BlobCipher, KeyConfigError, Keyring, generate_key_entry
from store.datasets import SQLiteDatasetRepository
from tests.conftest import TEST_DATA_KEYS
from tests.test_contract_runtime import DATA
from tests.test_tenancy import CAPITALS, PLAN, call, code, workspace

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
EMAIL = "jane.roe@example.org"
CARD = "4111 1111 1111 1111"
PASSAGE = b"is the capital of"  # a distinctive plaintext fragment of DATA


# -- harness --------------------------------------------------------------------------------
def service(tmp_path: Path, **kw: Any) -> DatasetService:
    limits = kw.pop("limits", None)
    svc = DatasetService(
        SQLiteDatasetRepository(tmp_path / "metadata.sqlite3"),
        LocalBlobStore(tmp_path / "blobs"),
        limits,
        **kw,
    )
    return svc


def project(svc: DatasetService) -> str:
    return svc.create_project(NewProject(name="P")).project_id


def raw(r: Any, key: str, method: str, path: str, body: bytes = b"") -> Any:
    h = {"Authorization": f"Bearer {key}", "Content-Length": str(len(body))}
    return r.handle(method, "/api/v1/" + path, h, io.BytesIO(body))


def files_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


@pytest.fixture
def two(tmp_path):
    r = build_api(tmp_path / "data")
    ws_a, key_a = workspace(r, "Alpha", "a@alpha.test")
    ws_b, key_b = workspace(r, "Beta", "b@beta.test")
    return r, ws_a, key_a, ws_b, key_b


def capitals(r: TenantRouter, key: str, body: bytes = DATA) -> dict[str, Any]:
    s, proj = call(r, key, "POST", "projects", {"name": "Capitals"})
    assert s == 201, proj
    s, up = call(r, key, "POST", f"projects/{proj['project_id']}/uploads?format=jsonl", body)
    assert s == 201, up
    s, version = call(r, key, "POST", f"uploads/{up['upload_id']}/register", CAPITALS)
    assert s == 201, version
    s, splits = call(r, key, "POST", "datasets/capitals/versions/1/splits", PLAN)
    assert s == 201, splits
    return {"project": proj, "upload": up, "version": version, "splits": splits}


# -- 1. upload boundaries -------------------------------------------------------------------
def test_oversized_upload_is_refused_before_the_body_is_read(tmp_path):
    r = build_api(tmp_path / "data", max_upload_bytes=64)
    _, key = workspace(r, "A", "a@a.test")
    _, proj = call(r, key, "POST", "projects", {"name": "P"})
    body = b"a,b\n" + b"1,2\n" * 100

    class Exploding(io.BytesIO):
        def read(self, *a: Any) -> bytes:
            raise AssertionError("the body must not be read")

    h = {"Authorization": f"Bearer {key}", "Content-Length": str(len(body))}
    res = r.handle("POST", f"/api/v1/projects/{proj['project_id']}/uploads?format=csv", h,
                   Exploding(body))  # fmt: skip
    assert res.status == 413 and res.body["error"]["code"] == "payload_too_large"
    assert res.body["error"]["details"] == {"limit_bytes": 64}


@pytest.mark.parametrize(
    ("limits", "data", "fmt", "expected"),
    [
        (IngestLimits(max_rows=2), b"a\n1\n2\n3\n", "csv", "too_many_rows"),
        (IngestLimits(max_rows=2), b'{"a":1}\n{"a":2}\n{"a":3}\n', "jsonl", "too_many_rows"),
        (IngestLimits(max_columns=3), b"a,b,c,d\n1,2,3,4\n", "csv", "too_many_columns"),
        (IngestLimits(max_columns=2), b'{"a":1,"b":2,"c":3}\n', "jsonl", "too_many_columns"),
        (IngestLimits(max_cell_chars=8), b"a\n" + b"x" * 9 + b"\n", "csv", "cell_too_large"),
        (IngestLimits(max_cell_chars=8), b'{"a":"' + b"x" * 9 + b'"}\n', "jsonl",
         "cell_too_large"),
        (IngestLimits(max_record_chars=16), b"a,b\n" + b"x" * 10 + b"," + b"y" * 10 + b"\n",
         "csv", "record_too_large"),
        (IngestLimits(max_record_chars=16), b'{"a":"0123456789abcdef"}\n', "jsonl",
         "record_too_large"),
        (IngestLimits(max_json_depth=3), b'{"a":[[[[1]]]]}\n', "jsonl", "json_too_deep"),
    ],
)  # fmt: skip
def test_rows_columns_cells_records_and_depth_are_bounded(limits, data, fmt, expected):
    with pytest.raises(IngestError) as info:
        parse_dataset(data, fmt, limits)
    assert info.value.code == expected
    assert ERROR_STATUS[expected] == 422


def test_limits_come_from_the_environment_and_a_bad_value_refuses_to_start():
    limits = IngestLimits.from_env({"WYNK_MAX_UPLOAD_ROWS": "5", "WYNK_MAX_UPLOAD_COLUMNS": "7"})
    assert (limits.max_rows, limits.max_columns) == (5, 7)
    with pytest.raises(ValueError):
        IngestLimits.from_env({"WYNK_MAX_UPLOAD_ROWS": "lots"})
    with pytest.raises(ValueError):
        IngestLimits.from_env({"WYNK_MAX_UPLOAD_ROWS": "0"})


def test_a_parser_bomb_never_overflows_the_stack():
    bomb = b'{"a":' + b"[" * 100_000 + b"]" * 100_000 + b"}\n"
    with pytest.raises(IngestError) as info:
        parse_dataset(bomb, "jsonl", IngestLimits())
    assert info.value.code in ("json_too_deep", "record_too_large")


# -- 2. safe parsing ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "fmt", "expected"),
    [
        (b'a,b\n1,"unterminated\n', "csv", "malformed_csv"),
        (b"a,b\n1,2,3\n", "csv", "malformed_csv"),
        (b'{"a":1}\n{"a":', "jsonl", "malformed_jsonl"),  # truncated
        (b'{"a":1}\n[1,2]\n', "jsonl", "malformed_jsonl"),
        (b'{"a":NaN}\n', "jsonl", "malformed_jsonl"),
        (b"a\n\xff\xfe\n", "csv", "invalid_encoding"),
        (b"a\nx\x00y\n", "csv", "invalid_encoding"),  # NUL
        (b"a\nx\x1by\n", "csv", "invalid_encoding"),  # ESC: terminal escape injection
        (b'{"a":"x\\u0007"}\n', "jsonl", "invalid_encoding"),  # BEL via a JSON escape
        (b"", "csv", "empty_file"),
        (b"a\n1\n", "xlsx", "unsupported_format"),
    ],
)
def test_malformed_truncated_and_unsafe_encodings_are_structured_refusals(data, fmt, expected):
    with pytest.raises(IngestError) as info:
        parse_dataset(data, fmt, IngestLimits())
    assert info.value.code == expected
    assert "Traceback" not in info.value.message


def test_a_malformed_upload_answers_a_structured_error_not_a_trace(two, caplog):
    r, _, key, *_ = two
    _, proj = call(r, key, "POST", "projects", {"name": "P"})
    caplog.set_level(logging.DEBUG)
    s, body = call(r, key, "POST", f"projects/{proj['project_id']}/uploads?format=jsonl",
                   b'{"a":1}\n{"a":')  # fmt: skip
    assert s == 422 and set(body) == {"error"} and body["error"]["code"] == "malformed_jsonl"
    assert "Traceback" not in json.dumps(body) and "Traceback" not in caplog.text


@pytest.mark.parametrize(
    "filename",
    [
        "../../etc/passwd.csv",
        "..\\..\\boot.csv",
        "/abs/data.csv",
        "C:data.csv",
        "a\x00b.csv",
        "line\nbreak.csv",
        "CON.csv",
        "lpt1.csv",
        "..",
        "payload.exe",
        "payload.csv.exe",
        "x" * 300 + ".csv",
        "data.jsonl",  # extension of another format
        "trailing.csv ",
        "trailing.csv.",
    ],
)
def test_unsafe_filenames_are_refused(tmp_path, filename):
    svc = service(tmp_path)
    with pytest.raises(IngestError) as info:
        svc.upload(project(svc), b"a\n1\n", "csv", filename)
    assert info.value.code == "invalid_filename"
    assert not list((tmp_path / "blobs").rglob("sha256/*/*/*"))  # nothing was stored


def test_a_safe_filename_is_display_metadata_only(tmp_path):
    svc = service(tmp_path)
    up, _ = svc.upload(project(svc), b"a\n1\n", "csv", "My data (v2).csv")
    assert up.filename == "My data (v2).csv"
    blob = svc.blobs._path(up.content_hash)  # type: ignore[attr-defined]
    assert blob.name == up.content_hash and "My data" not in str(blob)


# -- 3. CSV formula injection ---------------------------------------------------------------
FORMULAS = (
    b"qid,question,answer\n"
    b"q1,=1+1,@SUM(A1)\n"
    b"q2,\"+cmd|' /C calc'!A0\",-2\n"
    b'q3,-x," =HYPERLINK(""http://evil"")"\n'
    b"q4,\xef\xbc\x9dfullwidth,1e5\n"
)


def test_the_neutralization_rule():
    for value in ("=1+1", "+x", "-x", "@x", "\tx", "\rx", " =x", "＝x", "＋x", "－x", "＠x"):
        assert is_formula_like(value), value
        assert neutralize_cell(value) == "'" + value
    for value in ("-2", "+3.5", "-1e5", "1-2", "a=b", "", "plain", "'=already"):
        assert not is_formula_like(value), value


# FORMULAS after ingest neutralization: the deterministic canonical CSV (``serialize_rows``)
CANONICAL = (
    b"qid,question,answer\n"
    b"q1,'=1+1,'@SUM(A1)\n"
    b"q2,'+cmd|' /C calc'!A0,-2\n"
    b'q3,\'-x,"\' =HYPERLINK(""http://evil"")"\n'
    b"q4,'\xef\xbc\x9dfullwidth,1e5\n"
)
# every formula-capable cell of FORMULAS, as uploaded
ORIGINAL_CELLS = ("=1+1", "@SUM(A1)", "+cmd|' /C calc'!A0", "-x", ' =HYPERLINK("http://evil")',
                  "＝fullwidth")  # fmt: skip
REGISTER_F = {"dataset_id": "f", "name": "F", "input_columns": ["question"],
              "target_columns": ["answer"], "row_ids": "generated"}  # fmt: skip


def _csv_cells(data: bytes) -> list[str]:
    return [c for row in csv_module.reader(io.StringIO(data.decode("utf-8"))) for c in row]


def test_csv_formulas_are_neutralized_before_hashing_and_storage(two, tmp_path, caplog):
    r, ws, key, *_ = two
    caplog.set_level(logging.DEBUG)
    _, proj = call(r, key, "POST", "projects", {"name": "P"})
    s, up = call(r, key, "POST", f"projects/{proj['project_id']}/uploads?format=csv", FORMULAS)
    assert s == 201, up
    # the canonical stored representation is the neutralized one, and the hash is over it
    assert up["content_hash"] == hashlib.sha256(CANONICAL).hexdigest()
    assert up["content_hash"] != hashlib.sha256(FORMULAS).hexdigest()
    assert up["size_bytes"] == len(CANONICAL) and up["row_count"] == 4
    assert up["preview"][0] == {"qid": "q1", "question": "'=1+1", "answer": "'@SUM(A1)"}
    assert up["preview"][1]["answer"] == "-2"  # a signed number is data, not a formula
    report = up["sensitive_data"]
    assert report["formula_like_values"] == 6 and report["neutralized_values"] == 6
    assert report["findings"] == [] and report["redacted_values"] == 0
    # the decrypted blob is the canonical safe CSV; no cell of it is formula-capable
    svc = r.tenant(ws).api.service
    stored = svc.blobs.get(up["content_hash"])
    assert stored == CANONICAL
    assert not any(is_formula_like(c) for c in _csv_cells(stored))
    assert set(ORIGINAL_CELLS).isdisjoint(_csv_cells(stored))
    # no copy of the formula-capable upload is kept anywhere on disk
    disk = files_bytes(tmp_path / "data")
    assert FORMULAS not in disk and b"q1,=1+1" not in disk and b"=HYPERLINK" not in disk
    # the dataset identity, the row ids and the splits all reference the canonical bytes
    s, version = call(r, key, "POST", f"uploads/{up['upload_id']}/register", REGISTER_F)
    assert s == 201 and version["spec"]["content_hash"] == up["content_hash"]
    s, splits = call(r, key, "POST", "datasets/f/versions/1/splits", PLAN)
    assert s == 201 and splits["splits"]["dataset_hash"] == version["identity_hash"]
    rows = parse_dataset(stored, "csv", IngestLimits()).rows
    assert svc.get_version("f", 1).spec.row_count == len(rows) == 4
    # format=original is the canonical (safe) representation, never the unsafe upload
    res = raw(r, key, "GET", "datasets/f/versions/1/export?format=original")
    h = dict(res.headers)
    assert res.status == 200 and res.raw == CANONICAL and res.raw != FORMULAS
    assert hashlib.sha256(res.raw).hexdigest() == up["content_hash"]
    assert h["X-Wynk-Export-Safety"] == "canonical"
    assert h["Content-Disposition"] == 'attachment; filename="f-v1.original.csv"'
    # the default CSV export is still spreadsheet-safe; nothing is left to rewrite
    res = raw(r, key, "GET", "datasets/f/versions/1/export")
    h = dict(res.headers)
    assert res.status == 200 and h["X-Wynk-Export-Safety"] == "spreadsheet-neutralized"
    assert h["X-Wynk-Neutralized-Cells"] == "0" and h["Content-Type"].startswith("text/csv")
    assert not any(is_formula_like(c) for c in _csv_cells(res.raw))
    assert set(ORIGINAL_CELLS).isdisjoint(_csv_cells(res.raw))
    assert ",-2\n" in res.raw.decode() and ",1e5\n" in res.raw.decode()
    # JSONL export: the logical (neutralized) values, exact
    res = raw(r, key, "GET", "datasets/f/versions/1/export?format=jsonl")
    out = [json.loads(line) for line in res.raw.decode().splitlines()]
    assert out[0]["question"] == "'=1+1" and out[2]["answer"] == '\' =HYPERLINK("http://evil")'
    assert set(ORIGINAL_CELLS).isdisjoint(v for row in out for v in row.values())
    assert code(call(r, key, "GET", "datasets/f/versions/1/export?format=xlsx")) == (
        400,
        "invalid_request",
    )
    # the audit trail holds the counts, never a cell value; nor does any log line
    _, page = call(r, key, "GET", "audit?limit=1000")
    uploaded = next(
        e for e in page["events"] if e["action"] == "dataset.upload" and e["result"] == "ok"
    )
    assert uploaded["details"]["formula_like_values"] == 6
    assert uploaded["details"]["neutralized_values"] == 6
    trails = b"".join(
        p.read_bytes() for p in (tmp_path / "data").rglob("*audit.sqlite3*") if p.is_file()
    )
    for cell in ("=1+1", "@SUM(A1)", "+cmd|", "-x,", "HYPERLINK", "fullwidth", "calc"):
        assert cell.encode() not in trails, cell
        assert cell not in caplog.text, cell


def test_every_formula_prefix_is_neutralized_on_ingest_and_signed_numbers_are_not(tmp_path):
    svc = service(tmp_path)
    body = (
        "id,v\n"
        'r1,"\tx"\nr2,"\rx"\nr3,＝x\nr4,＋x\nr5,－x\nr6,＠x\nr7, =x\n'
        "r8,-3\nr9,+3.5\nr10,-.5\nr11,-1e3\nr12,1-2\nr13,'=done\nr14,\n"
    ).encode()
    up, _ = svc.upload(project(svc), body, "csv")
    rows = parse_dataset(svc.blobs.get(up.content_hash), "csv", IngestLimits()).rows
    got = {row["id"]: row["v"] for row in rows}
    for rid, cell in {"r1": "\tx", "r2": "\rx", "r3": "＝x", "r4": "＋x",
                      "r5": "－x", "r6": "＠x", "r7": " =x"}.items():  # fmt: skip
        assert got[rid] == "'" + cell, rid
    for rid, cell in {"r8": "-3", "r9": "+3.5", "r10": "-.5", "r11": "-1e3", "r12": "1-2",
                      "r13": "'=done", "r14": None}.items():  # fmt: skip
        assert got[rid] == cell, rid  # signed numbers, safe text and nulls are unchanged
    report = up.sensitive_data
    assert report.formula_like_values == 7 and report.neutralized_values == 7
    assert not any(isinstance(v, str) and is_formula_like(v) for v in got.values())


def test_identical_unsafe_csv_gives_identical_canonical_hashes_and_splits(tmp_path):
    one, other = service(tmp_path / "one"), service(tmp_path / "two")
    request = RegisterDataset(**REGISTER_F)
    plan = SplitPlan(**PLAN)
    results = []
    for svc in (one, other):
        project_id = project(svc)
        up, created = svc.upload(project_id, FORMULAS, "csv")
        again, created_again = svc.upload(project_id, FORMULAS, "csv")
        assert created and not created_again and again.upload_id == up.upload_id
        version, _ = svc.register(up.upload_id, request)
        splits, _ = svc.create_splits("f", 1, plan)
        results.append((up.content_hash, version.identity_hash, splits.splits_hash))
    assert results[0] == results[1]
    assert results[0][0] == hashlib.sha256(CANONICAL).hexdigest()
    # neutralization is idempotent: the canonical bytes themselves are stored verbatim
    up, _ = one.upload(project(one), CANONICAL, "csv")
    assert up.content_hash == results[0][0] and up.sensitive_data.neutralized_values == 0


def test_a_safe_csv_keeps_its_exact_bytes_and_identity(tmp_path):
    svc = service(tmp_path)
    body = b"qid,question,answer\r\nq1,What is 1-2?,-1\r\nq2,a=b,+3.5\r\n"  # CRLF kept verbatim
    up, _ = svc.upload(project(svc), body, "csv")
    assert up.content_hash == hashlib.sha256(body).hexdigest()
    assert svc.blobs.get(up.content_hash) == body
    report = up.sensitive_data
    assert report.formula_like_values == 0 and report.neutralized_values == 0
    svc.register(up.upload_id, RegisterDataset(**REGISTER_F))
    out = svc.export("f", 1, "original")
    assert out.data == body and out.safety == "canonical"


def test_jsonl_values_are_not_rewritten_by_the_csv_rule(tmp_path):
    svc = service(tmp_path)
    body = (
        b'{"qid":"q1","question":"=1+1","answer":"@SUM(A1)"}\n'
        b'{"qid":"q2","question":"-3","answer":"x"}\n'
    )
    up, _ = svc.upload(project(svc), body, "jsonl")
    assert up.content_hash == hashlib.sha256(body).hexdigest()
    assert svc.blobs.get(up.content_hash) == body
    report = up.sensitive_data
    assert report.formula_like_values == 2 and report.neutralized_values == 0
    svc.register(up.upload_id, RegisterDataset(**REGISTER_F))
    assert svc.export("f", 1, "original").data == body
    assert svc.export("f", 1, "jsonl").data == body
    as_csv = svc.export("f", 1, "csv")  # a JSONL version exported to a spreadsheet is still safe
    assert as_csv.neutralized_cells == 2
    assert not any(is_formula_like(c) for c in _csv_cells(as_csv.data))


def test_a_legacy_formula_csv_is_never_re_emitted(tmp_path, monkeypatch):
    svc = service(tmp_path)
    with monkeypatch.context() as m:  # admitted as before ingest neutralization existed
        m.setattr(exports, "neutralize_rows", lambda names, rows: (rows, 0))
        up, _ = svc.upload(project(svc), FORMULAS, "csv")
    assert svc.blobs.get(up.content_hash) == FORMULAS  # a pre-#33 blob, hashes kept
    svc.register(up.upload_id, RegisterDataset(**REGISTER_F))
    with pytest.raises(IngestError) as info:
        svc.export("f", 1, "original")
    assert info.value.code == "unsafe_original_export" and info.value.details == {"cells": 6}
    assert not any(cell in info.value.message for cell in ORIGINAL_CELLS)
    assert ERROR_STATUS["unsafe_original_export"] == 409
    safe = svc.export("f", 1, "csv")
    assert safe.neutralized_cells == 6 and safe.data == CANONICAL
    assert not any(is_formula_like(c) for c in _csv_cells(safe.data))


def test_safe_csv_round_trips_to_the_neutralized_values():
    body, n = safe_csv(["a"], [{"a": "=x"}, {"a": None}, {"a": 3}, {"a": "ok"}])
    assert body == b'a\n\'=x\n""\n3\nok\n' and n == 1  # an empty lone cell is written ""
    assert parse_dataset(body, "csv", IngestLimits()).rows[1] == {"a": None}


# -- 4. secrets and PII ---------------------------------------------------------------------
def test_a_secret_blocks_the_upload_without_leaking_it(two, caplog, tmp_path):
    r, _, key, *_ = two
    _, proj = call(r, key, "POST", "projects", {"name": "P"})
    caplog.set_level(logging.DEBUG)
    body = b"qid,note\nq1,hello\nq2,key " + AWS_KEY.encode() + b" here\n"
    s, err = call(r, key, "POST", f"projects/{proj['project_id']}/uploads?format=csv", body)
    assert s == 422 and err["error"]["code"] == "sensitive_data_blocked"
    assert err["error"]["details"]["category"] == "aws_access_key"
    assert err["error"]["details"]["row"] == 2 and err["error"]["details"]["column"] == "note"
    assert AWS_KEY not in json.dumps(err) and AWS_KEY not in caplog.text
    assert call(r, key, "GET", f"projects/{proj['project_id']}")[0] == 200
    data = files_bytes(tmp_path / "data")
    assert AWS_KEY.encode() not in data and b"hello" not in data  # nothing was stored


@pytest.mark.parametrize(
    ("value", "category"),
    [
        ("-----BEGIN RSA PRIVATE KEY-----", Category.PRIVATE_KEY),
        (AWS_KEY, Category.AWS_ACCESS_KEY),
        ("ghp_" + "a" * 36, Category.GITHUB_TOKEN),
        ("xoxb-1234567890-abcdef", Category.SLACK_TOKEN),
        ("sk_live_" + "a1" * 12, Category.STRIPE_SECRET_KEY),
        ("AIza" + "b" * 35, Category.GOOGLE_API_KEY),
        ("sk-ant-" + "c" * 30, Category.MODEL_API_KEY),
        (KEY_PREFIX + "0" * 16 + "_" + "A" * 43, Category.WYNK_API_KEY),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlc2ln", Category.JWT),
        (CARD, Category.PAYMENT_CARD),
        ("078-05-1121", Category.US_SSN),
        ("GB82 WEST 1234 5698 7654 32", Category.IBAN),
        (EMAIL, Category.EMAIL),
        ("+44 20 7946 0958", Category.PHONE),
    ],
)
def test_every_rule_detects_its_category(value, category):
    assert category in {c for _, _, c in find(f"text {value} text")}


@pytest.mark.parametrize(
    "value",
    [
        "4111 1111 1111 1112",  # fails Luhn
        "000-12-3456",  # invalid SSN area
        "GB00 WEST 1234 5698 7654 32",  # fails mod-97
        "call 020 7946 0958",  # no + : not E.164
        "version 1.2.3-alpha",
        "sk-short",
        "user at example dot org",
    ],
)
def test_lookalikes_are_not_findings(value):
    assert find(value) == []


def test_pii_is_redacted_before_storage_and_the_hash_covers_the_stored_bytes(tmp_path):
    svc = service(tmp_path)
    body = f"qid,note\nq1,mail {EMAIL} now\nq2,plain\n".encode()
    up, _ = svc.upload(project(svc), body, "csv")
    assert up.preview[0]["note"] == "mail [REDACTED:email] now"
    report = up.sensitive_data
    assert report is not None and report.redacted_values == 1
    finding = report.findings[0]
    assert (finding.category, finding.column, finding.first_row) == (Category.EMAIL, "note", 1)
    assert EMAIL not in report.model_dump_json()
    stored = svc.blobs.get(up.content_hash)
    assert EMAIL.encode() not in stored and up.content_hash == hashlib.sha256(stored).hexdigest()
    assert up.content_hash != hashlib.sha256(body).hexdigest()
    assert EMAIL.encode() not in files_bytes(tmp_path)
    # deterministic: the same upload is the same upload
    again, created = svc.upload(up.project_id, body, "csv")
    assert (again.upload_id, created) == (up.upload_id, False)


def test_the_policy_is_configurable_and_records_allowed_findings(tmp_path):
    policy = SensitiveDataPolicy.from_env({"WYNK_SENSITIVE_POLICY": "email=allow,phone=block"})
    svc = service(tmp_path, policy=policy)
    body = f"qid,note\nq1,{EMAIL}\n".encode()
    up, _ = svc.upload(project(svc), body, "csv")
    assert up.content_hash == hashlib.sha256(body).hexdigest()  # allowed: stored verbatim
    assert up.sensitive_data is not None and up.sensitive_data.findings[0].action == "allow"
    with pytest.raises(IngestError) as info:
        svc.upload(up.project_id, b"qid,note\nq1,+44 20 7946 0958\n", "csv")
    assert info.value.code == "sensitive_data_blocked"
    with pytest.raises(ValueError):
        SensitiveDataPolicy.from_env({"WYNK_SENSITIVE_POLICY": "aws_access_key=allow"})
    with pytest.raises(ValueError):
        SensitiveDataPolicy.from_env({"WYNK_SENSITIVE_POLICY": "nonsense"})


def test_a_sensitive_filename_and_sensitive_row_ids_are_refused(tmp_path):
    svc = service(tmp_path)
    pid = project(svc)
    with pytest.raises(IngestError) as info:
        svc.upload(pid, b"a\n1\n", "csv", f"{AWS_KEY}.csv")
    assert info.value.code == "sensitive_data_blocked"
    up, _ = svc.upload(pid, f"qid,q,a\n{EMAIL},x,y\nb,x,y\n".encode(), "csv", f"{EMAIL} .csv")
    assert up.filename == "[REDACTED:email] .csv"
    request = RegisterDataset(dataset_id="d", name="D", input_columns=["q"],
                              target_columns=["a"], row_ids="column", id_column="qid")  # fmt: skip
    with pytest.raises(IngestError) as info:
        svc.register(up.upload_id, request)
    assert info.value.code == "sensitive_row_ids"


# -- 5. encryption at rest ------------------------------------------------------------------
def keyring(spec: str = TEST_DATA_KEYS) -> Keyring:
    return Keyring.parse(spec)


def test_encryption_round_trip_hashes_plaintext_and_nonces_never_enter_the_name(tmp_path):
    data = b"qid,q\n1,a confidential passage\n"
    a = LocalBlobStore(tmp_path / "a", workspace_id="ws-alpha", keyring=keyring())
    b = LocalBlobStore(tmp_path / "b", workspace_id="ws-alpha", keyring=keyring())
    ra, rb = a.put(data), b.put(data)
    assert ra.sha256 == rb.sha256 == hashlib.sha256(data).hexdigest()  # over the plaintext
    ea, eb = a._path(ra.sha256).read_bytes(), b._path(rb.sha256).read_bytes()
    assert ea != eb  # fresh nonce per write ...
    assert ea[:8] == b"WYNKBLOB" and b"confidential" not in ea and data not in ea
    assert a.get(ra.sha256) == b.get(rb.sha256) == data  # ... same verified plaintext
    assert a.put(data).created is False  # a healthy duplicate is reused, not re-encrypted
    assert a._path(ra.sha256).read_bytes() == ea


def test_tampered_truncated_or_moved_ciphertext_is_refused(tmp_path):
    data = b"a\n1\n2\n"
    store = LocalBlobStore(tmp_path / "a", workspace_id="ws-alpha", keyring=keyring())
    name = store.put(data).sha256
    path = store._path(name)
    good = path.read_bytes()
    for i in range(len(good)):  # every single bit-flip position: header, nonce, body, tag
        bad = bytearray(good)
        bad[i] ^= 0x01
        path.write_bytes(bytes(bad))
        with pytest.raises((BlobCorrupted, BlobKeyUnavailable)):
            store.get(name)
    for cut in (0, 8, 20, len(good) - 1):
        path.write_bytes(good[:cut])
        with pytest.raises(BlobCorrupted):
            store.get(name)
    path.write_bytes(good)
    assert store.get(name) == data
    # the envelope is bound to its workspace and its name
    other = LocalBlobStore(tmp_path / "b", workspace_id="ws-beta", keyring=keyring())
    other._path(name).parent.mkdir(parents=True)
    other._path(name).write_bytes(good)
    with pytest.raises(BlobCorrupted):
        other.get(name)
    decoy = hashlib.sha256(b"other").hexdigest()
    store._path(decoy).parent.mkdir(parents=True, exist_ok=True)
    store._path(decoy).write_bytes(good)
    with pytest.raises(BlobCorrupted):
        store.get(decoy)


def test_wrong_key_fails_closed_and_rotation_rekeys(tmp_path):
    data = b"a\n1\n"
    store = LocalBlobStore(tmp_path / "s", workspace_id="ws-alpha", keyring=keyring())
    name = store.put(data).sha256
    same_id_other_key = keyring("test-only-k1:" + generate_key_entry("x").split(":")[1])
    with pytest.raises(BlobCorrupted):
        LocalBlobStore(tmp_path / "s", workspace_id="ws-alpha", keyring=same_id_other_key).get(name)
    unknown = keyring(generate_key_entry("k9"))
    with pytest.raises(BlobKeyUnavailable):
        LocalBlobStore(tmp_path / "s", workspace_id="ws-alpha", keyring=unknown).get(name)
    rotated = keyring(generate_key_entry("k2") + "," + TEST_DATA_KEYS)
    new = LocalBlobStore(tmp_path / "s", workspace_id="ws-alpha", keyring=rotated)
    assert new.get(name) == data  # old blobs stay readable under the retired key
    report = new.secure()
    assert (report.rekeyed, report.encrypted, report.refused) == (1, 0, [])
    assert BlobCipher.key_id_of(new._path(name).read_bytes()) == "k2"
    assert new.secure().healthy == 1  # idempotent
    with pytest.raises(BlobKeyUnavailable):
        store.get(name)  # the old-only keyring cannot read the rekeyed blob


def test_legacy_plaintext_blobs_are_refused_until_secured(tmp_path):
    data = b"a\nlegacy\n"
    store = LocalBlobStore(tmp_path / "s", workspace_id="ws-alpha", keyring=keyring())
    name = hashlib.sha256(data).hexdigest()
    store._path(name).parent.mkdir(parents=True)
    store._path(name).write_bytes(data)
    with pytest.raises(BlobNotEncrypted):
        store.get(name)
    assert store.secure().encrypted == 1
    assert store.get(name) == data and data not in store._path(name).read_bytes()


def test_a_missing_or_unsafe_production_key_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.delenv("WYNK_DATA_KEYS")
    with pytest.raises(KeyConfigError, match="no key is configured"):
        build_api(root)
    assert not (root / "identity.sqlite3").exists()  # refused before any store opened
    with pytest.raises(KeyConfigError):
        LocalBlobStore(tmp_path / "blobs")
    assert not (tmp_path / "blobs").exists()
    root.mkdir(exist_ok=True)
    inside = root / "keys.txt"
    inside.write_text(TEST_DATA_KEYS)
    monkeypatch.setenv("WYNK_DATA_KEYS_FILE", str(inside))
    with pytest.raises(KeyConfigError, match="inside the data directory"):
        build_api(root)
    outside = tmp_path / "keys.txt"
    outside.write_text(TEST_DATA_KEYS)
    monkeypatch.setenv("WYNK_DATA_KEYS_FILE", str(outside))
    assert isinstance(build_api(root), TenantRouter)  # a key file elsewhere is fine
    monkeypatch.setenv("WYNK_DATA_KEYS", TEST_DATA_KEYS)
    with pytest.raises(KeyConfigError, match="not both"):
        build_api(root)
    monkeypatch.delenv("WYNK_DATA_KEYS_FILE")
    for bad in ("k1:short", "k1", "BAD ID:" + "A" * 43, "k1:!!!!", TEST_DATA_KEYS * 2):
        monkeypatch.setenv("WYNK_DATA_KEYS", bad)
        with pytest.raises(KeyConfigError):
            build_api(root)


def test_the_keyring_never_prints_key_material():
    ring = keyring()
    secret = TEST_DATA_KEYS.split(":")[1]
    assert secret not in repr(ring) and secret not in str(ring)


def test_the_data_dir_never_holds_uploaded_plaintext(two, tmp_path):
    r, _, key, *_ = two
    capitals(r, key)
    assert PASSAGE not in files_bytes(tmp_path / "data")


# -- 6. export, deletion, retention ---------------------------------------------------------
def test_export_and_delete_are_tenant_scoped(two, tmp_path):
    r, ws_a, key_a, ws_b, key_b = two
    a = capitals(r, key_a)
    upload_id = a["upload"]["upload_id"]
    # B cannot tell A's ids from ids that never existed
    for path in ("datasets/capitals/versions/1/export", "datasets/capitals/versions/1/source"):
        assert code(call(r, key_b, "GET", path)) == (404, "dataset_not_found")
    assert code(call(r, key_b, "GET", "datasets/nope/versions/1/export")) == (
        404,
        "dataset_not_found",
    )
    assert code(call(r, key_b, "DELETE", f"uploads/{upload_id}")) == (404, "upload_not_found")
    assert code(call(r, key_b, "DELETE", "uploads/u-nope")) == (404, "upload_not_found")
    assert code(call(r, key_b, "DELETE", "datasets/capitals")) == (404, "dataset_not_found")
    # A's data is untouched and still exports exactly
    res = raw(r, key_a, "GET", "datasets/capitals/versions/1/export?format=original")
    assert res.status == 200 and res.raw == DATA
    # B's identical content is B's own: deleting it leaves A's
    b = capitals(r, key_b)
    assert b["upload"]["content_hash"] == a["upload"]["content_hash"]
    s, deleted = call(r, key_b, "DELETE", f"uploads/{b['upload']['upload_id']}")
    assert s == 200 and deleted["content_removed"] is True
    assert raw(r, key_a, "GET", "datasets/capitals/versions/1/export?format=original").raw == DATA


def test_export_needs_read_and_deletion_needs_admin(two):
    r, _, key_a, *_ = two
    capitals(r, key_a)
    keys = {}
    for scopes in (["read"], ["write"], ["read", "write", "invoke"]):
        _, made = call(r, key_a, "POST", "workspace/keys", {"name": "-".join(scopes),
                                                             "scopes": scopes})  # fmt: skip
        keys[tuple(scopes)] = made["secret"]
    ro, wo, rwi = keys[("read",)], keys[("write",)], keys[("read", "write", "invoke")]
    assert raw(r, ro, "GET", "datasets/capitals/versions/1/export").status == 200
    assert code(call(r, wo, "GET", "datasets/capitals/versions/1/export")) == (
        403,
        "insufficient_scope",
    )
    for path in ("datasets/capitals", "uploads/u-x"):
        assert code(call(r, rwi, "DELETE", path)) == (403, "insufficient_scope")
    assert code(call(r, rwi, "GET", "audit")) == (403, "insufficient_scope")
    assert tenancy.route_scope("delete_upload", "DELETE") == "admin"
    assert tenancy.route_scope("export_version", "GET") == "read"


def test_deleted_data_is_not_retrievable_and_the_state_is_explicit(two, tmp_path):
    r, ws, key, *_ = two
    a = capitals(r, key)
    up, splits = a["upload"], a["splits"]
    svc = r.tenant(ws).api.service
    blob = svc.blobs._path(up["content_hash"])  # type: ignore[attr-defined]
    assert blob.exists()
    s, deleted = call(r, key, "DELETE", f"uploads/{up['upload_id']}")
    assert s == 200 and deleted["content_removed"] is True
    d = deleted["deletion"]
    assert d["reason"] == "request" and d["affected_versions"] == [
        {"dataset_id": "capitals", "dataset_version": 1}
    ]
    assert not blob.exists()
    with pytest.raises(BlobNotFound):
        svc.blobs.get(up["content_hash"])
    with pytest.raises(StorageFailure):
        svc.dataset_bytes(up["content_hash"])  # what experiment / monitoring runs read
    # every content path refuses, with an explicit, identifier-only state
    s, err = call(r, key, "GET", f"uploads/{up['upload_id']}")
    assert s == 410 and err["error"]["code"] == "upload_deleted"
    expected = {"upload_id": up["upload_id"], "deleted_at": d["deleted_at"], "reason": "request"}
    assert err["error"]["details"] == expected
    assert code(call(r, key, "GET", "datasets/capitals/versions/1/export")) == (
        410,
        "dataset_source_deleted",
    )
    assert code(call(r, key, "POST", f"uploads/{up['upload_id']}/register", CAPITALS)) == (
        410,
        "upload_deleted",
    )
    assert code(call(r, key, "POST", "datasets/capitals/versions/1/splits",
                     {"seed": 8, "validation_bps": 2000, "test_bps": 2000})) == (
        410, "dataset_source_deleted")  # fmt: skip
    project_id = a["project"]["project_id"]
    assert code(call(r, key, "POST", f"projects/{project_id}/uploads?format=jsonl", DATA)) == (
        410,
        "upload_deleted",
    )
    # immutable provenance stays: version, splits, identity - marked source-deleted
    s, version = call(r, key, "GET", "datasets/capitals/versions/1")
    assert s == 200 and version["identity_hash"] == a["version"]["identity_hash"]
    s, kept = call(r, key, "GET", f"datasets/capitals/versions/1/splits/{splits['splits_hash']}")
    assert s == 200 and kept == splits
    s, source = call(r, key, "GET", "datasets/capitals/versions/1/source")
    assert s == 200 and source["status"] == "deleted" and source["deletion_reason"] == "request"
    # idempotent: the first tombstone is the answer
    s, again = call(r, key, "DELETE", f"uploads/{up['upload_id']}")
    assert s == 200 and again["deletion"] == d
    # the tombstone cannot be undone
    conn = sqlite3.connect(r.root / "workspaces" / ws / "metadata.sqlite3")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM upload_deletions")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE upload_deletions SET reason='retention'")
    conn.close()
    # the scrubbed record keeps no filename, preview or row ids
    record = svc.repo.get_upload(up["upload_id"])
    assert record is not None and record.filename is None and record.preview == ()


def test_shared_content_survives_until_its_last_upload_is_deleted(tmp_path):
    svc = service(tmp_path)
    first, second = project(svc), project(svc)
    a, _ = svc.upload(first, DATA, "jsonl")
    b, _ = svc.upload(second, DATA, "jsonl")
    assert a.content_hash == b.content_hash and a.upload_id != b.upload_id
    gone = svc.delete_upload(a.upload_id, "usr-test")
    assert (gone.content_removed, gone.content_shared_with) == (False, 1)
    assert svc.get_upload(b.upload_id).row_count == b.row_count
    with pytest.raises(SourceDeleted):
        svc.get_upload(a.upload_id)
    assert svc.delete_upload(b.upload_id, "usr-test").content_removed is True
    with pytest.raises(BlobNotFound):
        svc.blobs.get(a.content_hash)


def test_deleting_a_dataset_refuses_to_take_another_datasets_source(two):
    r, _, key, *_ = two
    a = capitals(r, key)
    other = dict(CAPITALS, dataset_id="capitals-2", name="Capitals 2")
    s, _ = call(r, key, "POST", f"uploads/{a['upload']['upload_id']}/register", other)
    assert s == 201
    assert code(call(r, key, "DELETE", "datasets/capitals")) == (409, "dataset_source_shared")
    assert raw(r, key, "GET", "datasets/capitals/versions/1/export").status == 200  # untouched
    s, out = call(r, key, "DELETE", f"uploads/{a['upload']['upload_id']}")
    assert s == 200 and len(out["deletion"]["affected_versions"]) == 2
    s, out = call(r, key, "DELETE", "datasets/capitals")  # already deleted: idempotent
    assert s == 200 and out["deletions"][0]["deletion"] == out["deletions"][0]["deletion"]


def test_new_experiments_over_deleted_content_are_refused(two):
    r, ws, key, *_ = two
    a = capitals(r, key)
    call(r, key, "DELETE", f"uploads/{a['upload']['upload_id']}")
    api = r.tenant(ws).api
    version = api.service.get_version("capitals", 1)
    with pytest.raises(SourceDeleted) as info:
        api.service.require_source(version)
    assert info.value.code == "dataset_source_deleted"


def test_retention_purge_deletes_expired_uploads_and_collects_orphans(tmp_path, monkeypatch):
    monkeypatch.setenv("WYNK_UPLOAD_RETENTION_DAYS", "30")
    r = build_api(tmp_path / "data")
    ws, key = workspace(r, "A", "a@a.test")
    a = capitals(r, key)
    svc = r.tenant(ws).api.service
    orphan = svc.blobs.put(b"x\norphan\n").sha256  # e.g. the loser of a quota race
    assert tenancy.purge_expired(r, gc_older_than_s=3600)[ws] == {
        "deleted_uploads": [],
        "collected_blobs": 0,
    }  # nothing expired, the orphan is too young
    later = datetime.now(UTC) + timedelta(days=31)
    out = tenancy.purge_expired(r, gc_older_than_s=0, now=later)
    assert out[ws] == {"deleted_uploads": [a["upload"]["upload_id"]], "collected_blobs": 1}
    with pytest.raises(BlobNotFound):
        svc.blobs.get(orphan)
    s, err = call(r, key, "GET", f"uploads/{a['upload']['upload_id']}")
    assert s == 410 and err["error"]["details"]["reason"] == "retention"
    events = [(e.action, e.result, e.actor_user_id) for e in r.tenant(ws).api.audit.events()]
    assert ("dataset.delete", "requested", "system:retention") in events
    assert ("dataset.delete", "ok", "system:retention") in events
    monkeypatch.setenv("WYNK_UPLOAD_RETENTION_DAYS", "0")
    with pytest.raises(ValueError):
        build_api(tmp_path / "data")


# -- 7. audit trail -------------------------------------------------------------------------
def test_the_audit_trail_is_complete_and_redacted(two, tmp_path, caplog):
    r, ws, key, ws_b, key_b = two
    caplog.set_level(logging.DEBUG)
    a = capitals(r, key)
    up = a["upload"]["upload_id"]
    project_id = a["project"]["project_id"]
    call(r, key, "GET", f"uploads/{up}")
    raw(r, key, "GET", "datasets/capitals/versions/1/export?format=csv")
    secret_body = b"qid,note\nq1," + AWS_KEY.encode() + b"\n"
    call(r, key, "POST", f"projects/{project_id}/uploads?format=csv&filename=bank.csv",
         secret_body)  # fmt: skip
    pii = f"qid,note\nq1,{EMAIL}\n".encode()
    call(r, key, "POST", f"projects/{project_id}/uploads?format=csv&filename=people.csv", pii)
    call(r, key_b, "GET", f"uploads/{up}")  # cross-tenant: refused in B, recorded in B
    call(r, key_b, "GET", "datasets/capitals/versions/1")
    _, ro = call(r, key, "POST", "workspace/keys", {"name": "ro", "scopes": ["read"]})
    call(r, ro["secret"], "DELETE", f"uploads/{up}")  # scope denial
    bad = KEY_PREFIX + parse_key(key)[0] + "_" + "Z" * 43
    call(r, bad, "GET", "projects")  # authentication failure
    call(r, key, "DELETE", f"uploads/{up}")
    s, page = call(r, key, "GET", "audit?limit=1000")
    assert s == 200
    events = [(e["action"], e["result"], e["code"]) for e in page["events"]]
    for expected in [
        ("dataset.upload", "requested", None),
        ("dataset.upload", "ok", None),
        ("dataset.register", "ok", None),
        ("dataset.read", "ok", None),
        ("dataset.export", "ok", None),
        ("dataset.upload", "refused", "sensitive_data_blocked"),
        ("security.scope_denied", "denied", "insufficient_scope"),
        ("dataset.delete", "requested", None),
        ("dataset.delete", "ok", None),
    ]:
        assert expected in events, expected
    every = page["events"]
    assert all(e["workspace_id"] == ws and e["at"] for e in every)
    owner_key_id = parse_key(key)[0]
    upload_ok = next(e for e in every if e["action"] == "dataset.upload" and e["result"] == "ok")
    assert upload_ok["actor_key_id"] == owner_key_id and upload_ok["actor_user_id"]
    assert upload_ok["resource_type"] == "project" and upload_ok["resource_id"] == project_id
    assert upload_ok["details"]["upload_id"] == up
    blocked = next(e for e in every if e["code"] == "sensitive_data_blocked")
    assert blocked["details"]["category"] == "aws_access_key"
    redacted = [e for e in every if e["details"].get("categories") == "email"]
    assert redacted and redacted[0]["details"]["redacted_values"] == 1
    registered = next(e for e in every if e["action"] == "dataset.register" and e["result"] == "ok")
    assert registered["details"] == {
        "dataset_version": 1, "identity_hash": a["version"]["identity_hash"], "created": True,
    }  # fmt: skip
    assert "could not derive audit details" not in caplog.text  # every outcome kept its details
    export = next(e for e in every if e["action"] == "dataset.export")
    assert export["details"]["export_format"] == "csv"
    # B saw a not-found in its own trail; A's trail holds nothing of B
    b_events = r.tenant(ws_b).api.audit.events()
    assert [(e.action, e.result, e.code) for e in b_events] == [
        ("dataset.read", "refused", "upload_not_found"),
        ("security.access_refused", "refused", "dataset_not_found"),
    ]
    assert b_events[1].resource_id == "capitals" and b_events[1].details == {"route": "get_version"}
    failures = r.security_audit.events()
    assert [(e.action, e.result, e.details) for e in failures] == [
        ("security.authentication_failed", "denied", {"key_id": owner_key_id})
    ]
    # nothing sensitive anywhere: not in the trails, not in the logs
    trails = b"".join(
        p.read_bytes() for p in (tmp_path / "data").rglob("*audit.sqlite3*") if p.is_file()
    )
    for needle in (AWS_KEY, EMAIL, "bank.csv", "people.csv", key, bad, ro["secret"]):
        assert needle.encode() not in trails, needle
        assert needle not in caplog.text, needle
    assert PASSAGE not in trails
    assert r.tenant(ws).api.audit.verify_chain() == len(every) + 1  # + this audit.read


def test_the_audit_trail_is_append_only_and_tamper_evident(tmp_path):
    log = SQLiteAuditLog(tmp_path / "audit.sqlite3", workspace_id="ws-alpha")
    for i in range(3):
        log.append(action="dataset.read", resource_type="upload", resource_id=f"u-{i}",
                   result="ok")  # fmt: skip
    assert log.verify_chain() == 3
    conn = sqlite3.connect(tmp_path / "audit.sqlite3")
    for sql in ("UPDATE audit_events SET result='refused'", "DELETE FROM audit_events"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(sql)
    # an attacker with file access who drops the trigger is still detected
    conn.execute("DROP TRIGGER audit_events_append_only_u")
    conn.execute("UPDATE audit_events SET resource_id='u-x' WHERE seq=2")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="altered"):
        log.verify_chain()


@pytest.mark.parametrize(
    "details",
    [
        {"note": "x"},  # not allowlisted
        {"categories": EMAIL},  # not an identifier
        {"content_hash": "a b"},
        {"row_count": [1]},
    ],
)
def test_the_audit_trail_refuses_free_text(tmp_path, details):
    log = SQLiteAuditLog(tmp_path / "audit.sqlite3")
    with pytest.raises(ValueError):
        log.append(action="dataset.read", resource_type="upload", resource_id="u-1",
                   result="ok", details=details)  # fmt: skip
    with pytest.raises(ValueError):
        log.append(action="dataset.read", resource_type="upload", resource_id=EMAIL,
                   result="ok")  # fmt: skip
    assert log.events() == []


def test_an_unavailable_audit_trail_fails_closed(two):
    r, ws, key, *_ = two
    a = capitals(r, key)
    api = r.tenant(ws).api

    class Broken:
        def append(self, **_: Any) -> None:
            raise sqlite3.OperationalError("disk I/O error")

    real, api.audit = api.audit, Broken()  # type: ignore[assignment]
    try:
        up = a["upload"]["upload_id"]
        assert code(call(r, key, "DELETE", f"uploads/{up}")) == (503, "audit_unavailable")
        res = raw(r, key, "GET", "datasets/capitals/versions/1/export")
        assert res.status == 503 and res.raw is None  # no data released unaudited
        assert code(call(r, key, "GET", f"uploads/{up}")) == (503, "audit_unavailable")
    finally:
        api.audit = real
    assert call(r, key, "GET", f"uploads/{up}")[0] == 200  # nothing was deleted


# -- 8. identity, hashes and splits are unchanged -------------------------------------------
def test_identity_and_splits_are_reproducible_across_workspaces_keys_and_restarts(
    tmp_path, monkeypatch
):
    r1 = build_api(tmp_path / "one")
    _, k1 = workspace(r1, "A", "a@a.test")
    one = capitals(r1, k1)
    monkeypatch.setenv("WYNK_DATA_KEYS", generate_key_entry("other"))
    r2 = build_api(tmp_path / "two")
    _, k2 = workspace(r2, "B", "b@b.test")
    two_ = capitals(r2, k2)
    assert one["upload"]["content_hash"] == two_["upload"]["content_hash"]
    assert one["upload"]["content_hash"] == hashlib.sha256(DATA).hexdigest()
    assert one["version"]["identity_hash"] == two_["version"]["identity_hash"]
    assert one["splits"]["splits_hash"] == two_["splits"]["splits_hash"]
    assert one["splits"]["splits"] == two_["splits"]["splits"]
    # and the split assignment is exactly the pre-#33 pure function of identity + row ids
    from core.dataset import seeded_splits

    svc = r1.tenant(next(iter(r1._tenants))).api.service
    row_ids = svc.repo.get_row_ids("capitals", 1)
    expected = seeded_splits(one["version"]["identity_hash"], row_ids, SplitPlan(**PLAN))
    assert one["splits"]["splits_hash"] == expected.identity_hash


# -- 9. no user-data training ---------------------------------------------------------------
PRODUCT_PACKAGES = ("api", "core", "ingestion", "store", "runtime", "experiments", "compiler")
TRAINING_MARKERS = ("fine_tun", "fine-tun", "finetun", "/files", "training_file", "/batches")


def _product_sources() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    out = []
    for package in PRODUCT_PACKAGES:
        for path in (root / package).rglob("*.py"):
            if "oss_baselines" not in path.parts:  # research harness, never in the server
                out.append(path)
    return out


def test_no_product_code_can_send_user_data_to_a_training_endpoint():
    """The enforceable half of the no-training guarantee: the only outbound model call in the
    product is the OpenAI-compatible ``/chat/completions`` POST; no file upload, fine-tuning or
    batch endpoint exists, and no request asks a provider to store it."""
    call_sites = []
    for path in _product_sources():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                for marker in TRAINING_MARKERS:
                    assert marker not in lowered, (path, node.value)
            if isinstance(node, ast.Attribute) and node.attr == "urlopen":
                call_sites.append(path.name)
    assert call_sites == ["openai_compatible.py"]


def test_model_requests_carry_no_storage_or_training_opt_in():
    from runtime.backends.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
    from runtime.model_client import GenerationRequest

    client = OpenAICompatibleClient(OpenAICompatibleConfig(base_url="http://x", model="m"))
    request = GenerationRequest(role="extract", prompt_template_id="t",
                                prompt_template_version="1", input_text="hi", seed=1,
                                max_tokens=5, output_schema={"type": "object"})  # fmt: skip
    payload = client._payload(request)
    assert set(payload) <= {"model", "messages", "temperature", "seed", "max_tokens",
                            "response_format"}  # fmt: skip
    assert "store" not in json.dumps(payload)


def test_log_records_are_scrubbed_with_the_same_rules(two, caplog):
    caplog.set_level(logging.DEBUG)
    log = logging.getLogger("wynk.test")
    log.warning("value %s and %s", AWS_KEY, EMAIL)
    try:
        raise ValueError(f"bad cell {CARD}")
    except ValueError:
        log.exception("failed")
    assert AWS_KEY not in caplog.text and EMAIL not in caplog.text and CARD not in caplog.text
    assert "[REDACTED:aws_access_key]" in caplog.text and "[REDACTED:payment_card]" in caplog.text
