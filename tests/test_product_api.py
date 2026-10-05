"""Product API v1: success and error paths, strict schemas, stable error codes, real HTTP.

The integration test at the bottom runs upload -> map roles -> register -> split over a real HTTP
server, then reloads everything in a *separate Python process* from the same data directory, and
repeats the flow in a fresh directory to show the same bytes + config give the same identities.
"""

import hashlib
import io
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from api.product import ERROR_STATUS, ErrorResponse, build_api, make_handler

ROOT = Path(__file__).resolve().parent.parent
CSV = b"id,text,label,meta\n" + b"".join(
    f'{i},ticket {i},{["billing", "bugs", "sales"][i % 3]},"{{""k"": {i}}}"\n'.encode()
    for i in range(50)
)
JSONL = b"".join(
    json.dumps({"text": f"t{i}", "label": "ab"[i % 2], "doc": {"n": i}}).encode() + b"\n"
    for i in range(20)
)
MAPPING = {
    "dataset_id": "tickets",
    "name": "Support tickets",
    "input_columns": ["text"],
    "target_columns": ["label"],
    "context_columns": [],
    "row_ids": "column",
    "id_column": "id",
}
PLAN = {"seed": 3, "validation_bps": 2000, "test_bps": 2000}


def call(api, method, path, body=None, *, json_body=None, headers=None):
    h = {}
    if json_body is not None:
        body = json.dumps(json_body).encode()
        h["Content-Type"] = "application/json"
    if body is not None:
        h["Content-Length"] = str(len(body))
    h.update(headers or {})
    res = api.handle(method, path, h, io.BytesIO(body or b""))
    json.dumps(res.body, allow_nan=False)  # every body is plain JSON
    return res


def error(res, status, code):
    assert res.status == status, res.body
    ErrorResponse.model_validate(res.body)  # strict envelope, nothing else in it
    assert res.body["error"]["code"] == code
    assert ERROR_STATUS[code] == status
    return res.body["error"]


@pytest.fixture
def api(tmp_path):
    return build_api(tmp_path / "data", max_upload_bytes=64 * 1024)


@pytest.fixture
def project_id(api):
    res = call(api, "POST", "/api/v1/projects", json_body={"name": "Support"})
    assert res.status == 201
    return res.body["project_id"]


def upload(api, project_id, data=CSV, fmt="csv", **params):
    query = "&".join(f"{k}={v}" for k, v in {"format": fmt, **params}.items())
    return call(api, "POST", f"/api/v1/projects/{project_id}/uploads?{query}", data)


# -- projects -----------------------------------------------------------------------------------
def test_projects(api, project_id):
    got = call(api, "GET", f"/api/v1/projects/{project_id}")
    assert got.status == 200
    assert set(got.body) == {"project_id", "name", "description", "created_at"}
    listed = call(api, "GET", "/api/v1/projects")
    assert listed.body == {"projects": [got.body]}
    error(call(api, "GET", "/api/v1/projects/p-nope"), 404, "project_not_found")


@pytest.mark.parametrize(
    ("body", "headers", "status", "code"),
    [
        (b"{}", {"Content-Type": "application/json"}, 400, "invalid_request"),  # no name
        (b'{"name": "x", "owner": "y"}', {"Content-Type": "application/json"}, 400,
         "invalid_request"),  # unknown field
        (b'{"name": 3}', {"Content-Type": "application/json"}, 400, "invalid_request"),  # strict
        (b'{"name": "x"', {"Content-Type": "application/json"}, 400, "invalid_json"),
        (b'{"name": "a", "name": "b"}', {"Content-Type": "application/json"}, 400,
         "invalid_json"),
        (b"[]", {"Content-Type": "application/json"}, 400, "invalid_json"),
        (b'{"name": "x"}', {"Content-Type": "text/plain"}, 415, "unsupported_media_type"),
    ],
)  # fmt: skip
def test_create_project_validation(api, body, headers, status, code):
    error(call(api, "POST", "/api/v1/projects", body, headers=headers), status, code)


# -- uploads ------------------------------------------------------------------------------------
def test_upload_success_and_deduplication(api, project_id):
    res = upload(api, project_id, filename="tickets.csv")
    assert res.status == 201
    body = res.body
    assert set(body) == {
        "upload_id", "project_id", "filename", "format", "content_hash", "size_bytes",
        "row_count", "columns", "preview", "parser_version", "created_at",
    }  # fmt: skip
    assert body["content_hash"] == hashlib.sha256(CSV).hexdigest()
    assert body["size_bytes"] == len(CSV) and body["row_count"] == 50
    assert body["columns"][0] == {"name": "id", "type": "integer", "nullable": False,
                                  "null_count": 0}  # fmt: skip
    assert len(body["preview"]) == 20
    again = upload(api, project_id, filename="copy.csv")
    assert again.status == 200 and again.body == body
    assert call(api, "GET", f"/api/v1/uploads/{body['upload_id']}").body == body


@pytest.mark.parametrize(
    "param", ["content_hash=" + "0" * 64, "row_count=50", "types=string", "id=x"]
)
def test_client_cannot_supply_facts_about_the_data(api, project_id, param):
    res = call(api, "POST", f"/api/v1/projects/{project_id}/uploads?format=csv&{param}", CSV)
    error(res, 400, "unknown_parameter")


def test_upload_errors(api, project_id):
    error(upload(api, project_id, fmt="parquet"), 422, "unsupported_format")
    error(upload(api, project_id, b"id,text\n1\n"), 422, "malformed_csv")
    error(upload(api, project_id, b"a b,c\n1,2\n"), 422, "invalid_column_name")
    error(upload(api, project_id, b"a,a\n1,2\n"), 422, "duplicate_column")
    error(upload(api, project_id, b"\xff\xfe\n"), 422, "invalid_encoding")
    error(upload(api, project_id, b""), 422, "empty_file")
    error(upload(api, project_id, b'{"a": 1}\nnot json\n', fmt="jsonl"), 422, "malformed_jsonl")
    error(upload(api, "p-nope"), 404, "project_not_found")
    path = f"/api/v1/projects/{project_id}/uploads"
    error(call(api, "POST", path, CSV), 400, "invalid_request")  # format is required
    error(call(api, "POST", f"{path}?format=csv&format=jsonl", CSV), 400, "invalid_request")
    multipart = {"Content-Type": "multipart/form-data; boundary=x"}
    error(call(api, "POST", f"{path}?format=csv", CSV, headers=multipart), 415,
          "unsupported_media_type")  # fmt: skip


def test_upload_size_limit_and_length_rules(api, project_id):
    path = f"/api/v1/projects/{project_id}/uploads?format=csv"
    big = b"id,text\n" + b"1,x\n" * 20_000
    stream = io.BytesIO(big)
    res = api.handle("POST", path, {"Content-Length": str(len(big))}, stream)
    err = error(res, 413, "payload_too_large")
    assert err["details"] == {"limit_bytes": 64 * 1024}
    assert stream.tell() == 0  # refused before reading the body
    error(api.handle("POST", path, {}, io.BytesIO(CSV)), 411, "length_required")
    chunked = {"Transfer-Encoding": "chunked"}
    error(api.handle("POST", path, chunked, io.BytesIO(CSV)), 411, "length_required")
    short = {"Content-Length": str(len(CSV) + 10)}
    error(api.handle("POST", path, short, io.BytesIO(CSV)), 400, "invalid_request")
    error(api.handle("POST", path, {"Content-Length": "-1"}, io.BytesIO()), 400,
          "invalid_request")  # fmt: skip


# -- registration -------------------------------------------------------------------------------
def test_register_and_retrieve(api, project_id):
    upload_id = upload(api, project_id).body["upload_id"]
    res = call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=MAPPING)
    assert res.status == 201
    spec = res.body["spec"]
    assert spec["schema_version"] == "datasetspec/1"
    assert spec["content_hash"] == hashlib.sha256(CSV).hexdigest()
    assert spec["row_count"] == 50 and spec["id_column"] == "id"
    assert [c["type"] for c in spec["columns"]] == ["integer", "string", "string", "string"]
    again = call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=MAPPING)
    assert again.status == 200 and again.body == res.body

    version = call(api, "GET", "/api/v1/datasets/tickets/versions/1")
    assert version.body == res.body
    dataset = call(api, "GET", "/api/v1/datasets/tickets")
    assert dataset.body["latest_version"] == 1 and dataset.body["versions"] == [res.body]
    listed = call(api, "GET", f"/api/v1/projects/{project_id}/datasets")
    assert listed.body == {"datasets": [dataset.body]}
    error(call(api, "GET", "/api/v1/datasets/nope"), 404, "dataset_not_found")
    error(call(api, "GET", "/api/v1/datasets/tickets/versions/2"), 404,
          "dataset_version_not_found")  # fmt: skip


@pytest.mark.parametrize(
    ("overrides", "status", "code"),
    [
        ({"row_count": 50}, 400, "invalid_request"),  # not a mapping field
        ({"content_hash": "0" * 64}, 400, "invalid_request"),
        ({"input_columns": []}, 400, "invalid_request"),
        ({"row_ids": "generated"}, 400, "invalid_request"),  # with an id column
        ({"dataset_id": "Bad Id"}, 400, "invalid_request"),
        ({"input_columns": ["nope"]}, 422, "unknown_column"),
        ({"context_columns": ["text"]}, 422, "invalid_mapping"),  # text is already input
        ({"input_columns": ["text", "label"]}, 422, "invalid_mapping"),
        ({"id_column": "label", "input_columns": ["text"], "target_columns": ["meta"]}, 422,
         "duplicate_row_id"),
    ],
)  # fmt: skip
def test_register_errors(api, project_id, overrides, status, code):
    upload_id = upload(api, project_id).body["upload_id"]
    res = call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=MAPPING | overrides)
    error(res, status, code)
    error(call(api, "GET", "/api/v1/datasets/tickets"), 404, "dataset_not_found")


def test_register_json_column_is_refused(api, project_id):
    upload_id = upload(api, project_id, JSONL, fmt="jsonl").body["upload_id"]
    mapping = MAPPING | {"input_columns": ["doc"], "row_ids": "generated", "id_column": None}
    res = call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=mapping)
    assert error(res, 422, "json_column_role")["details"] == {"column": "doc"}
    ok = call(api, "POST", f"/api/v1/uploads/{upload_id}/register",
              json_body=mapping | {"input_columns": ["text"]})  # fmt: skip
    assert ok.status == 201 and ok.body["row_id_source"] == "generated"


def test_register_unknown_upload(api):
    res = call(api, "POST", "/api/v1/uploads/u-nope/register", json_body=MAPPING)
    error(res, 404, "upload_not_found")


def _blob(tmp_path, content_hash):
    (path,) = (tmp_path / "data" / "blobs").rglob(content_hash)
    return path


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_damaged_blob_fails_closed_on_every_read(api, project_id, tmp_path, damage):
    body = upload(api, project_id).body
    upload_id = body["upload_id"]
    assert call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=MAPPING).status
    assert call(api, "POST", "/api/v1/datasets/tickets/versions/1/splits", json_body=PLAN).status
    blob = _blob(tmp_path, body["content_hash"])
    if damage == "missing":
        blob.unlink()
    else:  # same size, different bytes: a size check alone would not notice
        blob.write_bytes(CSV.replace(b"ticket 1,", b"ticket X,"))
        assert blob.stat().st_size == len(CSV)

    error(call(api, "GET", f"/api/v1/uploads/{upload_id}"), 500, "storage_error")
    error(call(api, "POST", f"/api/v1/uploads/{upload_id}/register",
               json_body=MAPPING | {"dataset_id": "other"}), 500, "storage_error")  # fmt: skip
    for path in (
        "/api/v1/datasets/tickets",
        "/api/v1/datasets/tickets/versions/1",
        f"/api/v1/projects/{project_id}/datasets",
        "/api/v1/datasets/tickets/versions/1/splits",
    ):
        error(call(api, "GET", path), 500, "storage_error")
    error(call(api, "POST", "/api/v1/datasets/tickets/versions/1/splits", json_body=PLAN), 500,
          "storage_error")  # fmt: skip
    error(call(api, "GET", "/api/v1/datasets/other"), 404, "dataset_not_found")


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_identical_reupload_restores_a_damaged_blob(api, project_id, tmp_path, damage):
    """Policy: the re-uploaded bytes hash to the blob's name, so they atomically replace it."""
    body = upload(api, project_id).body
    blob = _blob(tmp_path, body["content_hash"])
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_bytes(b"x" * len(CSV))
    again = upload(api, project_id)
    assert again.status == 200 and again.body == body  # same upload record, now healthy again
    assert _blob(tmp_path, body["content_hash"]).read_bytes() == CSV
    assert call(api, "GET", f"/api/v1/uploads/{body['upload_id']}").body == body
    res = call(api, "POST", f"/api/v1/uploads/{body['upload_id']}/register", json_body=MAPPING)
    assert res.status == 201


# -- splits -------------------------------------------------------------------------------------
def test_splits(api, project_id):
    upload_id = upload(api, project_id).body["upload_id"]
    call(api, "POST", f"/api/v1/uploads/{upload_id}/register", json_body=MAPPING)
    base = "/api/v1/datasets/tickets/versions/1/splits"
    res = call(api, "POST", base, json_body=PLAN)
    assert res.status == 201
    assert res.body["sizes"] == {"optimization": 30, "test": 10, "validation": 10}
    assert res.body["splits"]["method"] == "seeded_hash/1"
    assert res.body["splits"]["plan"] == PLAN
    again = call(api, "POST", base, json_body=PLAN)
    assert again.status == 200 and again.body == res.body
    assert call(api, "GET", f"{base}/{res.body['splits_hash']}").body == res.body
    assert call(api, "GET", base).body == {"splits": [res.body]}
    error(call(api, "GET", f"{base}/{'0' * 64}"), 404, "splits_not_found")
    bad = PLAN | {"validation_bps": 5000, "test_bps": 5000}
    error(call(api, "POST", base, json_body=bad), 400, "invalid_request")
    error(call(api, "POST", base, json_body=PLAN | {"seed": "3"}), 400, "invalid_request")
    error(call(api, "POST", base, json_body=PLAN | {"row_ids": ["1"]}), 400, "invalid_request")
    error(call(api, "POST", "/api/v1/datasets/nope/versions/1/splits", json_body=PLAN), 404,
          "dataset_not_found")  # fmt: skip


# -- routing + failures -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("GET", "/api/v1/nothing", 404, "not_found"),
        ("GET", "/api/v1/datasets/../../etc", 404, "not_found"),
        ("GET", "/api/v1/datasets/tickets/versions/0", 404, "not_found"),
        ("GET", "/api/v1/projects?limit=5", 400, "unknown_parameter"),
        ("DELETE", "/api/v1/projects", 405, "method_not_allowed"),
        ("GET", "/api/v1/projects/x/uploads", 405, "method_not_allowed"),
        ("GET", "/elsewhere", 404, "not_found"),
    ],
)
def test_routing_errors(api, method, path, status, code):
    error(call(api, method, path), status, code)


def test_unexpected_errors_are_500_without_details(api, monkeypatch):
    def boom():
        raise RuntimeError("secret internals")

    monkeypatch.setattr(api.service, "list_projects", boom)
    err = error(call(api, "GET", "/api/v1/projects"), 500, "internal_error")
    assert "secret" not in err["message"]


# -- real HTTP + restart ------------------------------------------------------------------------
class Server:
    def __init__(self, handler) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method, path, data=None, content_type=None):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def json(self, method, path, body):
        return self.request(method, path, json.dumps(body).encode(), "application/json")


def run_flow(server: Server) -> dict:
    status, project = server.json("POST", "/api/v1/projects", {"name": "Support"})
    assert status == 201
    pid = project["project_id"]
    status, up = server.request(
        "POST", f"/api/v1/projects/{pid}/uploads?format=csv&filename=t.csv", CSV, "text/csv"
    )
    assert status == 201, up
    status, version = server.json("POST", f"/api/v1/uploads/{up['upload_id']}/register", MAPPING)
    assert status == 201, version
    status, splits = server.json("POST", "/api/v1/datasets/tickets/versions/1/splits", PLAN)
    assert status == 201, splits
    return {"project": project, "upload": up, "version": version, "splits": splits}


RELOAD = """
import io, json, sys
from api.product import build_api
api = build_api(sys.argv[1])
def get(path):
    res = api.handle("GET", path, {}, io.BytesIO())
    assert res.status == 200, res.body
    return res.body
splits_hash = sys.argv[2]
print(json.dumps({
    "version": get("/api/v1/datasets/tickets/versions/1"),
    "splits": get("/api/v1/datasets/tickets/versions/1/splits/" + splits_hash),
    "upload": get("/api/v1/uploads/" + sys.argv[3]),
}))
"""


def test_upload_register_split_reload_integration(tmp_path):
    data_dir = tmp_path / "data"
    with Server(make_handler(build_api(data_dir))) as server:
        first = run_flow(server)
        status, err = server.request("POST", "/api/v1/projects/x/uploads?format=csv", b"a\n1\n")
        assert status == 404 and err["error"]["code"] == "project_not_found"

    # restart: a new process reads the same directory back
    out = subprocess.run(
        [sys.executable, "-c", RELOAD, str(data_dir), first["splits"]["splits_hash"],
         first["upload"]["upload_id"]],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=True,
    )  # fmt: skip
    reloaded = json.loads(out.stdout)
    assert reloaded["version"] == first["version"]
    assert reloaded["splits"] == first["splits"]
    assert reloaded["upload"] == first["upload"]

    # restart the server on the same directory: same records over HTTP, nothing re-created
    with Server(make_handler(build_api(data_dir))) as server:
        status, version = server.request("GET", "/api/v1/datasets/tickets/versions/1")
        assert status == 200 and version == first["version"]
        status, again = server.json("POST", "/api/v1/datasets/tickets/versions/1/splits", PLAN)
        assert status == 200 and again == first["splits"]

    # same bytes + same config in a brand-new store: identical identities
    with Server(make_handler(build_api(tmp_path / "fresh"))) as server:
        second = run_flow(server)
    assert second["project"]["project_id"] != first["project"]["project_id"]
    assert second["upload"]["content_hash"] == first["upload"]["content_hash"]
    assert second["version"]["spec"] == first["version"]["spec"]
    assert second["version"]["identity_hash"] == first["version"]["identity_hash"]
    assert second["version"]["row_ids_hash"] == first["version"]["row_ids_hash"]
    assert second["splits"]["splits_hash"] == first["splits"]["splits_hash"]
    assert second["splits"]["splits"] == first["splits"]["splits"]


def test_chat_server_mounts_the_product_api(tmp_path):
    from api.chat import make_handler as chat_handler

    with Server(chat_handler(None, "no model", build_api(tmp_path / "data"))) as server:
        status, body = server.json("POST", "/api/v1/projects", {"name": "Via chat"})
        assert status == 201 and body["name"] == "Via chat"
        status, health = server.request("GET", "/api/health")
        assert status == 200 and health["ok"] is False  # the chat routes are untouched
