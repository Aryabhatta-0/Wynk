"""Golden request/response pairs of the product API, shared with the UI's live adapter tests.

The UI (``ui/src/api/live.ts``) is tested against these exact exchanges, so its wire types and
mapping are pinned to what ``api.product`` really sends. This test regenerates every exchange from
the real ``ProductAPI`` (fixed clock, fixed project ids) and fails if the committed file differs:
a schema or error-code change on the Python side must be reflected in the UI fixture.

    WYNK_UPDATE_UI_FIXTURES=1 python -m pytest tests/test_product_api_ui_fixtures.py
"""

import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from api.product import ERROR_STATUS
from tests.tenancy_support import dev_api as build_api

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "ui" / "src" / "api" / "contract" / "fixtures" / "product-api.v1.json"
CLOCK = "2026-10-05T12:00:00+00:00"
UPLOAD_LIMIT = 8 * 1024

TICKETS = (
    "ticket_id,subject,body,customer_tier,queue\n"
    'T-1,Charged twice,"My card shows two charges, both for March.",pro,billing\n'
    "T-2,Cannot log in,SSO redirects back to the login page.,enterprise,technical\n"
    "T-3,Update company name,We rebranded; please change the account name.,,account\n"
    "T-4,Where is my order,Tracking has not moved for a week.,free,shipping\n"
    'T-5,Refund request,"Downgraded mid-cycle, expected a prorated refund.",pro,billing\n'
    "T-6,Webhook failures,Deliveries to our endpoint time out since Monday.,enterprise,technical\n"
    "T-7,Add a teammate,How do I invite a colleague to our workspace?,free,account\n"
    "T-8,Damaged parcel,The box arrived crushed and the device is cracked.,pro,shipping\n"
)
FAQ = "".join(
    json.dumps(row) + "\n"
    for row in [
        {
            "faq_id": 1,
            "question": "How long do refunds take?",
            "answer": "5-7 days",
            "tags": ["billing"],
            "doc": {"page": 3},
            "score": 0.5,
        },
        {
            "faq_id": 2,
            "question": "Is there a free plan?",
            "answer": "Yes",
            "tags": ["plans", "pricing"],
            "doc": {"page": 9},
            "score": None,
        },
        {
            "faq_id": 3,
            "question": "Where are invoices?",
            "answer": "Settings > Billing",
            "tags": [],
            "doc": None,
            "score": 2,
        },
    ]
)
DUPLICATE_IDS = "ticket_id,subject,queue\nT-1,a,billing\nT-1,b,account\n"

TICKETS_V1 = {
    "dataset_id": "support-tickets",
    "name": "Support tickets",
    "input_columns": ["subject", "body"],
    "target_columns": ["queue"],
    "context_columns": ["customer_tier"],
    "row_ids": "column",
    "id_column": "ticket_id",
}
# same upload, different roles and generated row ids: a new version of the same dataset
TICKETS_V2 = {
    **TICKETS_V1,
    "input_columns": ["body"],
    "context_columns": ["subject"],
    "row_ids": "generated",
    "id_column": None,
}
PLAN = {"seed": 7, "validation_bps": 2500, "test_bps": 2500}


class Recorder:
    def __init__(self, api: Any) -> None:
        self.api = api
        self.exchanges: dict[str, dict[str, Any]] = {}

    def __call__(
        self,
        name: str,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        text: str | None = None,
        size: int | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        body = b""
        request: dict[str, Any] = {"method": method, "path": path}
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
            request["body"] = {"json": json_body}
        elif text is not None:
            body = text.encode()
            headers["Content-Type"] = "application/octet-stream"
            request["body"] = {"text": text}
        elif size is not None:
            body = b"x" * size
            headers["Content-Type"] = "application/octet-stream"
            request["body"] = {"size": size}
        if method == "POST":
            headers["Content-Length"] = str(len(body))
        res = self.api.handle(method, path, headers, io.BytesIO(body))
        assert name not in self.exchanges
        self.exchanges[name] = {
            "request": request,
            "response": {"status": res.status, "body": res.body},
        }
        return res.body


def generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    ids = iter(range(1, 100))
    monkeypatch.setattr("ingestion.service.uuid.uuid4", lambda: uuid.UUID(int=next(ids) << 96))
    data_dir = tmp_path / "data"
    api = build_api(data_dir, max_upload_bytes=UPLOAD_LIMIT)
    api.service.clock = lambda: CLOCK
    rec = Recorder(api)

    # -- the happy path: project -> upload -> inspect -> register -> splits -------------------
    project = "Support triage"
    pid = rec(
        "create_project",
        "POST",
        "/api/v1/projects",
        json_body={"name": project, "description": "Route tickets to queues"},
    )["project_id"]
    rec("list_projects", "GET", "/api/v1/projects")
    rec("get_project", "GET", f"/api/v1/projects/{pid}")
    up = f"/api/v1/projects/{pid}/uploads?format=csv&filename=tickets.csv"
    uid = rec("upload_csv", "POST", up, text=TICKETS)["upload_id"]
    rec("upload_csv_again", "POST", up, text=TICKETS)
    rec("get_upload", "GET", f"/api/v1/uploads/{uid}")
    rec("register_v1", "POST", f"/api/v1/uploads/{uid}/register", json_body=TICKETS_V1)
    rec("register_v1_again", "POST", f"/api/v1/uploads/{uid}/register", json_body=TICKETS_V1)
    rec("register_v2", "POST", f"/api/v1/uploads/{uid}/register", json_body=TICKETS_V2)
    rec("list_datasets", "GET", f"/api/v1/projects/{pid}/datasets")
    rec("get_dataset", "GET", "/api/v1/datasets/support-tickets")
    rec("get_version", "GET", "/api/v1/datasets/support-tickets/versions/1")
    splits = "/api/v1/datasets/support-tickets/versions/1/splits"
    sh = rec("create_splits", "POST", splits, json_body=PLAN)["splits_hash"]
    rec("create_splits_again", "POST", splits, json_body=PLAN)
    rec("list_splits", "GET", splits)
    rec("get_splits", "GET", f"{splits}/{sh}")
    jl = f"/api/v1/projects/{pid}/uploads?format=jsonl&filename=faq.jsonl"
    faq = rec("upload_jsonl", "POST", jl, text=FAQ)["upload_id"]

    # -- every error a person can cause in the dataset flow ------------------------------------
    rec("error_project_not_found", "GET", "/api/v1/projects/p-missing")
    rec("error_dataset_not_found", "GET", "/api/v1/datasets/missing")
    rec("error_splits_not_found", "GET", f"{splits}/{'0' * 64}")
    bad = f"/api/v1/projects/{pid}/uploads?format=csv&filename=bad.csv"
    rec("error_malformed_csv", "POST", bad, text="ticket_id,subject\nT-1\n")
    rec(
        "error_malformed_jsonl",
        "POST",
        f"/api/v1/projects/{pid}/uploads?format=jsonl&filename=bad.jsonl",
        text='{"a": 1}\nnot json\n',
    )
    rec(
        "error_unsupported_format",
        "POST",
        f"/api/v1/projects/{pid}/uploads?format=parquet&filename=rows.parquet",
        text="PAR1",
    )
    rec("error_empty_file", "POST", bad, text="")
    rec("error_invalid_column_name", "POST", bad, text="ticket id,queue\n1,billing\n")
    rec("error_duplicate_column", "POST", bad, text="queue,queue\n1,2\n")
    rec("error_payload_too_large", "POST", bad, size=UPLOAD_LIMIT + 1)
    register = f"/api/v1/uploads/{uid}/register"
    rec(
        "error_unknown_column",
        "POST",
        register,
        json_body={**TICKETS_V1, "input_columns": ["subject", "missing"]},
    )
    rec(
        "error_invalid_mapping",
        "POST",
        register,
        json_body={**TICKETS_V1, "input_columns": ["subject", "queue"]},
    )
    rec(
        "error_invalid_id_column",
        "POST",
        register,
        json_body={**TICKETS_V1, "context_columns": [], "id_column": "customer_tier"},
    )
    rec("error_invalid_request", "POST", register, json_body={**TICKETS_V1, "dataset_id": "Bad Id"})
    rec(
        "error_json_column_role",
        "POST",
        f"/api/v1/uploads/{faq}/register",
        json_body={
            "dataset_id": "faq",
            "name": "FAQ",
            "input_columns": ["doc"],
            "target_columns": ["answer"],
            "context_columns": [],
            "row_ids": "generated",
            "id_column": None,
        },
    )
    dup = rec(
        "upload_duplicate_ids",
        "POST",
        f"/api/v1/projects/{pid}/uploads?format=csv&filename=dupes.csv",
        text=DUPLICATE_IDS,
    )
    rec(
        "error_duplicate_row_id",
        "POST",
        f"/api/v1/uploads/{dup['upload_id']}/register",
        json_body={
            "dataset_id": "dupes",
            "name": "Dupes",
            "input_columns": ["subject"],
            "target_columns": ["queue"],
            "context_columns": [],
            "row_ids": "column",
            "id_column": "ticket_id",
        },
    )
    other = rec(
        "create_other_project",
        "POST",
        "/api/v1/projects",
        json_body={"name": "Other", "description": ""},
    )["project_id"]
    other_up = rec(
        "upload_other",
        "POST",
        f"/api/v1/projects/{other}/uploads?format=csv&filename=tickets.csv",
        text=TICKETS,
    )
    rec(
        "error_dataset_conflict",
        "POST",
        f"/api/v1/uploads/{other_up['upload_id']}/register",
        json_body=TICKETS_V1,
    )
    rec(
        "error_split_plan",
        "POST",
        splits,
        json_body={"seed": 7, "validation_bps": 5000, "test_bps": 5000},
    )
    # stored bytes gone: reads fail closed with storage_error, never a stale success
    for blob in (data_dir / "workspaces" / api.workspace_id / "blobs").rglob(dup["content_hash"]):
        blob.unlink()
    rec("error_storage", "GET", f"/api/v1/uploads/{dup['upload_id']}")

    return {
        "_comment": "Generated by tests/test_product_api_ui_fixtures.py from the real ProductAPI. "
        "Do not edit by hand.",
        "error_status": ERROR_STATUS,
        "exchanges": rec.exchanges,
    }


def test_ui_fixture_matches_the_product_api(tmp_path, monkeypatch):
    generated = generate(tmp_path, monkeypatch)
    if os.environ.get("WYNK_UPDATE_UI_FIXTURES") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(generated, indent=2, ensure_ascii=False) + "\n"
        FIXTURE.write_text(text, encoding="utf-8", newline="\n")
    assert FIXTURE.exists(), f"{FIXTURE} is missing; regenerate with WYNK_UPDATE_UI_FIXTURES=1"
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(generated)), (
        "the product API's responses changed: regenerate the UI fixture with "
        "WYNK_UPDATE_UI_FIXTURES=1 and update ui/src/api/contract to match"
    )


def test_ui_fixture_covers_every_dataset_flow_error(tmp_path, monkeypatch):
    exchanges = generate(tmp_path, monkeypatch)["exchanges"]
    codes = {
        e["response"]["body"]["error"]["code"]
        for e in exchanges.values()
        if e["response"]["status"] >= 400
    }
    assert codes >= {
        "project_not_found",
        "dataset_not_found",
        "splits_not_found",
        "malformed_csv",
        "malformed_jsonl",
        "unsupported_format",
        "empty_file",
        "invalid_column_name",
        "duplicate_column",
        "payload_too_large",
        "unknown_column",
        "invalid_mapping",
        "invalid_id_column",
        "invalid_request",
        "json_column_role",
        "duplicate_row_id",
        "dataset_conflict",
        "storage_error",
    }
    assert exchanges["upload_csv"]["response"]["status"] == 201
    assert exchanges["upload_csv_again"]["response"]["status"] == 200
    assert exchanges["register_v2"]["response"]["body"]["spec"]["dataset_version"] == 2
