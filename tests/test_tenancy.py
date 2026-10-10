"""#32: authentication, workspaces and tenant isolation.

Every scenario goes through ``TenantRouter.handle`` (the real HTTP entry point) unless it is
testing a store directly. Two workspaces, A and B, upload the SAME capitals dataset under the
SAME human-readable ids, optimize, promote, deploy, invoke, label and re-optimize it - and
neither can read, change, invoke, monitor or even detect the other's resources.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from api import tenancy
from api.product import ERROR_STATUS, ProductAPI, build_api
from api.tenancy import (
    KEY_PREFIX,
    SCOPES,
    LegacyDataNotMigrated,
    Quotas,
    TenantRouter,
    hash_secret,
    migrate_legacy,
    parse_key,
)
from core.models import ModelRegistry
from experiments.deployment import InferenceRuntime
from experiments.jobs import JobWorker
from ingestion.service import DatasetService, NewProject, RegisterDataset
from runtime.model_client import RegisteredModelClient
from store.blobs import LocalBlobStore
from store.champions import SQLiteChampionStore
from store.datasets import SQLiteDatasetRepository
from store.deployments import SQLiteDeploymentStore
from store.identity import IdentityConflict
from store.jobs import SQLiteJobStore
from store.monitoring import SQLiteMonitoringStore
from store.tenancy import TenantBindingError, bind_sqlite, sqlite_binding
from tests.test_champion_inference import ENTRY, PassageDouble, definition, maf
from tests.test_champion_promotion import EVERYTHING
from tests.test_contract_runtime import DATA
from tests.test_production_monitoring import (
    COUNTRIES,
    SINCE,
    SMALL,
    UNTIL,
    AnyStamped,
    DataRuntime,
    inputs_for,
)

CAPITALS = {
    "dataset_id": "capitals",
    "name": "Capitals",
    "input_columns": ["question"],
    "context_columns": ["passage"],
    "target_columns": ["answer"],
    "row_ids": "column",
    "id_column": "qid",
}
PLAN = {"seed": 7, "validation_bps": 2000, "test_bps": 2000}
LINEAGE = "capitals.capitals"


# -- harness --------------------------------------------------------------------------------
def router(root: Path, **kw: Any) -> TenantRouter:
    backend = AnyStamped(EVERYTHING)
    inference = InferenceRuntime(
        registry=lambda: ModelRegistry(entries=(ENTRY,)),
        client=lambda entry: RegisteredModelClient(entry, PassageDouble()),
    )
    return build_api(
        root,
        runtime=lambda service: DataRuntime(backend, service.blobs),
        inference=inference,
        **kw,
    )


def call(r: Any, key: str | None, method: str, path: str, body: Any = None, **headers: str):
    raw = b""
    h = dict(headers)
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    if isinstance(body, bytes):
        raw = body
        h["Content-Type"] = "application/octet-stream"
    elif body is not None:
        raw = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    if method == "POST":
        h["Content-Length"] = str(len(raw))
    res = r.handle(method, "/api/v1/" + path, h, io.BytesIO(raw))
    json.dumps(res.body, allow_nan=False)
    assert ERROR_STATUS.get(res.body.get("error", {}).get("code"), res.status) == res.status
    return res.status, res.body


def code(result: tuple[int, dict]) -> tuple[int, str]:
    status, body = result
    return status, body["error"]["code"] if "error" in body else "ok"


def workspace(r: TenantRouter, name: str, email: str) -> tuple[str, str]:
    ws, _, _, key = r.create_workspace(name, email)
    return ws.workspace_id, key


def dataset(r: TenantRouter, key: str) -> dict[str, Any]:
    """Project -> upload -> register (the SAME human-readable dataset id everywhere) -> splits."""
    s, project = call(r, key, "POST", "projects", {"name": "Capitals"})
    assert s == 201, project
    s, up = call(r, key, "POST", f"projects/{project['project_id']}/uploads?format=jsonl", DATA)
    assert s == 201, up
    s, version = call(r, key, "POST", f"uploads/{up['upload_id']}/register", CAPITALS)
    assert s == 201, version
    s, splits = call(r, key, "POST", "datasets/capitals/versions/1/splits", PLAN)
    assert s == 201, splits
    return {"project": project, "upload": up, "dataset_version": version, "splits": splits}


def optimize(r: TenantRouter, ws: str, key: str, data: dict[str, Any]) -> str:
    d = definition()
    body = {
        "dataset_id": "capitals",
        "dataset_version": 1,
        "splits_hash": data["splits"]["splits_hash"],
        "contract": d.contract.model_dump(mode="json"),
        "plan": d.plan.model_dump(mode="json"),
    }
    s, job = call(r, key, "POST", "experiments", body)
    assert s == 201, job
    api = r.tenant(ws).api
    JobWorker(api.jobs.store, api.jobs.runtime, heartbeat=False).run_until_idle()
    s, got = call(r, key, "GET", f"experiments/{job['job_id']}")
    assert s == 200 and got["state"] == "COMPLETED", got
    return str(job["job_id"])


def deploy(r: TenantRouter, ws: str, key: str) -> dict[str, Any]:
    """Full #18-#31 flow for one workspace, every step through the authenticated API."""
    data = dataset(r, key)
    job = optimize(r, ws, key, data)
    s, promo = call(r, key, "POST", f"experiments/{job}/promote")
    assert s == 201 and promo["decision"] == "PROMOTED", promo
    champion = promo["record"]["champion"]["champion_id"]
    s, wv = call(r, key, "POST", "workflows", {"champion_id": champion})
    assert s == 201, wv
    version = wv["version_id"]
    assert call(r, key, "POST", f"workflows/{version}/stage", {"expected_revision": 0})[0] == 200
    assert call(r, key, "POST", f"workflows/{version}/promote", {"expected_revision": 1})[0] == 200
    s, policy = call(r, key, "POST", "monitoring/policies", SMALL.model_dump(mode="json"))
    assert s == 201, policy  # content-addressed id, yet NEW in every workspace (no oracle)
    inferences, feedback = [], []
    for country, city in COUNTRIES[:4]:
        s, inf = call(
            r, key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": inputs_for(country, city)}
        )
        assert s == 200, inf
        inferences.append(inf["inference_id"])
        s, fb = call(
            r,
            key,
            "POST",
            f"inferences/{inf['inference_id']}/feedback",
            {
                "inputs": inputs_for(country, city),
                "output": inf["output"],
                "expected": {"answer": "Nowhere"},
                "actor": "spoofed-admin",
            },
        )
        assert s == 201, fb
        feedback.append(fb["feedback_id"])
    s, trig = call(
        r,
        key,
        "POST",
        f"workflows/{version}/triggers/evaluate",
        {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id},
    )
    assert s == 201 and trig["outcome"] == "TRIGGERED", trig
    s, ch = call(r, key, "GET", f"triggers/{trig['trigger_id']}/challenger")
    assert s == 200 and ch["state"] == "JOB_CREATED", ch
    return {
        **data,
        "job": job,
        "promotion": promo["promotion_id"],
        "champion": champion,
        "version": version,
        "inferences": inferences,
        "feedback": feedback,
        "trigger": trig["trigger_id"],
        "challenger_job": ch["challenger_job_id"],
    }


# -- 1. the two-tenant end to end ------------------------------------------------------------
def _absent_like(r, key, method, path, real_id, fake_id, body=None):
    """B asking for A's id must look EXACTLY like B asking for an id that never existed."""
    real = call(r, key, method, path.format(real_id), body)
    fake = call(r, key, method, path.format(fake_id), body)
    assert real[0] == fake[0] and real[0] in (404, 422), (path, real)
    normalized = json.dumps(real[1]).replace(real_id, "ID")
    assert normalized == json.dumps(fake[1]).replace(fake_id, "ID"), (path, real, fake)
    return real


@maf
def test_two_tenants_with_identical_ids_are_isolated_at_every_boundary(tmp_path):
    r = router(tmp_path / "data")
    ws_a, key_a = workspace(r, "Alpha", "a@alpha.test")
    ws_b, key_b = workspace(r, "Beta", "b@beta.test")
    a = deploy(r, ws_a, key_a)
    b = deploy(r, ws_b, key_b)  # SAME bytes, dataset id, lineage id, policy id

    # the overlapping human-readable ids resolve to each workspace's OWN resource
    assert a["dataset_version"]["spec"] == b["dataset_version"]["spec"]  # identical datasets
    assert a["project"]["project_id"] != b["project"]["project_id"]
    assert call(r, key_b, "GET", "datasets/capitals")[1]["project_id"] == b["project"]["project_id"]
    assert call(r, key_a, "GET", "datasets/capitals")[1]["project_id"] == a["project"]["project_id"]
    assert call(r, key_b, "GET", f"champions/{LINEAGE}")[1]["promotion_id"] == b["promotion"]
    assert call(r, key_a, "GET", f"champions/{LINEAGE}")[1]["promotion_id"] == a["promotion"]
    assert [p["project_id"] for p in call(r, key_b, "GET", "projects")[1]["projects"]] == [
        b["project"]["project_id"]
    ]
    listed = {j["job_id"] for j in call(r, key_b, "GET", "experiments")[1]["jobs"]}
    assert listed == {b["job"], b["challenger_job"]}
    assert {w["version_id"] for w in call(r, key_b, "GET", "workflows")[1]["versions"]} == {
        b["version"]
    }

    # every A identifier, asked for with B's key: indistinguishable from a random id
    hexid = lambda prefix, n: prefix + "0" * n  # noqa: E731
    checks = [
        ("GET", "projects/{}", a["project"]["project_id"], "p-" + "0" * 20, None),
        ("GET", "projects/{}/datasets", a["project"]["project_id"], "p-" + "0" * 20, None),
        ("POST", "projects/{}/uploads?format=jsonl", a["project"]["project_id"], "p-" + "0" * 20,
         DATA),
        ("GET", "uploads/{}", a["upload"]["upload_id"], "u-" + "0" * 20, None),
        ("POST", "uploads/{}/register", a["upload"]["upload_id"], "u-" + "0" * 20, CAPITALS),
        ("GET", "datasets/capitals/versions/1/splits/{}", a["splits"]["splits_hash"],
         "1" * 64, None),
        ("GET", "experiments/{}", a["job"], "j-" + "0" * 24, None),
        ("GET", "experiments/{}/artifact", a["job"], "j-" + "0" * 24, None),
        ("GET", "experiments/{}/provenance", a["job"], "j-" + "0" * 24, None),
        ("GET", "experiments/{}/verify", a["job"], "j-" + "0" * 24, None),
        ("POST", "experiments/{}/cancel", a["challenger_job"], "j-" + "0" * 24, None),
        ("POST", "experiments/{}/promote", a["job"], "j-" + "0" * 24, None),
        ("GET", "promotions/{}", a["promotion"], "pr-" + "0" * 24, None),
        ("POST", "workflows", a["champion"], hexid("c-", 24), "CHAMPION"),
        ("GET", "workflows/{}", a["version"], hexid("wv-", 24), None),
        ("POST", "workflows/{}/stage", a["version"], hexid("wv-", 24), {"expected_revision": 2}),
        ("POST", "workflows/{}/invoke", a["version"], hexid("wv-", 24),
         {"inputs": inputs_for("Peru", "Lima")}),
        ("GET", "workflows/{}/monitoring", a["version"], hexid("wv-", 24), None),
        ("GET", "workflows/{}/drift", a["version"], hexid("wv-", 24), None),
        ("GET", "workflows/{}/triggers", a["version"], hexid("wv-", 24), None),
        ("POST", "workflows/{}/triggers/evaluate", a["version"], hexid("wv-", 24),
         {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id}),
        ("GET", "inferences/{}", a["inferences"][0], hexid("inf-", 24), None),
        ("POST", "inferences/{}/feedback", a["inferences"][0], hexid("inf-", 24),
         {"inputs": inputs_for("Peru", "Lima"), "expected": {"answer": "Lima"}}),
        ("GET", "feedback/{}", a["feedback"][0], hexid("fb-", 24), None),
        ("GET", "triggers/{}", a["trigger"], hexid("tr-", 24), None),
        ("GET", "triggers/{}/challenger", a["trigger"], hexid("tr-", 24), None),
        ("POST", "triggers/{}/reoptimize", a["trigger"], hexid("tr-", 24), None),
    ]  # fmt: skip
    before = _fingerprint(r, ws_a)
    for method, path, real_id, fake_id, body in checks:
        if body == "CHAMPION":
            real = call(r, key_b, "POST", "workflows", {"champion_id": real_id})
            fake = call(r, key_b, "POST", "workflows", {"champion_id": fake_id})
            assert code(real) == code(fake) and real[0] == 404, real
            continue
        if path == "datasets/capitals/versions/1/splits/{}":
            # B has its OWN capitals v1 (same bytes, same plan -> same splits hash): visible
            assert call(r, key_b, "GET", path.format(real_id))[0] == 200
            continue
        _absent_like(r, key_b, method, path, real_id, fake_id, body)
    assert _fingerprint(r, ws_a) == before  # nothing B did changed anything of A's

    # B invoking "production" of the shared lineage name runs B's OWN version, recorded in B
    s, inf = call(r, key_b, "POST", f"deployments/{LINEAGE}/invoke",
                  {"inputs": inputs_for("Peru", "Lima")})  # fmt: skip
    assert s == 200 and inf["workflow_version"] == b["version"]
    assert inf["authenticated_principal"]["workspace_id"] == ws_b
    assert call(r, key_a, "GET", f"inferences/{inf['inference_id']}")[0] == 404

    # the authenticated principal is recorded separately from the untrusted actor
    s, fb = call(r, key_a, "GET", f"feedback/{a['feedback'][0]}")
    assert fb["untrusted_metadata"]["actor"] == "spoofed-admin" and fb["metadata_trusted"] is False
    assert fb["authenticated_principal"]["workspace_id"] == ws_a
    s, inf_a = call(r, key_a, "GET", f"inferences/{a['inferences'][0]}")
    assert inf_a["authenticated_principal"] == {
        "workspace_id": ws_a,
        "user_id": r.identity.user_by_email("a@alpha.test").user_id,
        "key_id": parse_key(key_a)[0],
        "authenticated_by": "api_key",
    }

    # physically: each partition's stores are bound to their own workspace
    for ws in (ws_a, ws_b):
        for name in tenancy.STORE_FILES:
            assert sqlite_binding(tmp_path / "data" / "workspaces" / ws / name) == ws


def _fingerprint(r: TenantRouter, ws: str) -> dict[str, list]:
    out = {}
    for name in tenancy.STORE_FILES:
        conn = sqlite3.connect(r.root / "workspaces" / ws / name)
        try:
            tables = [
                t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            out[name] = [
                (t, sorted(map(repr, conn.execute(f"SELECT * FROM {t}").fetchall())))
                for t in sorted(tables)
            ]
        finally:
            conn.close()
    return out


# -- 2. authentication ----------------------------------------------------------------------
@pytest.fixture
def two(tmp_path):
    r = build_api(tmp_path / "data")
    ws_a, key_a = workspace(r, "Alpha", "a@alpha.test")
    ws_b, key_b = workspace(r, "Beta", "b@beta.test")
    return r, ws_a, key_a, ws_b, key_b


def test_every_tenant_route_requires_a_key_and_health_is_public(two):
    r, *_ = two
    assert call(r, None, "GET", "health") == (200, {"status": "ok"})
    from api.product import _ROUTES

    for method, pattern, name in _ROUTES:
        path = re.sub(r"\(.*?\)", "x", pattern.pattern.strip("^$"))
        path = path.replace("\\", "")
        for key in (None, "nope", KEY_PREFIX + "0" * 16 + "_" + "A" * 43):
            assert code(call(r, key, method, path, {})) == (401, "unauthenticated"), name
    for path in ("workspace", "workspace/members", "workspace/keys", "workspace/quota"):
        assert code(call(r, None, "GET", path)) == (401, "unauthenticated")


def test_unknown_wrong_revoked_and_expired_keys_all_fail_the_same_way(two):
    r, ws_a, key_a, *_ = two
    key_id, secret = parse_key(key_a)
    wrong = f"{KEY_PREFIX}{key_id}_{'B' * 43}"
    other = f"{KEY_PREFIX}{'f' * 16}_{secret}"
    s, made = call(r, key_a, "POST", "workspace/keys", {"name": "short", "scopes": ["read"],
                                                         "expires_in_s": 60})  # fmt: skip
    assert s == 201, made
    short = made["secret"]
    assert call(r, short, "GET", "projects")[0] == 200
    s, made = call(r, key_a, "POST", "workspace/keys", {"name": "tmp", "scopes": ["read"]})
    revocable = made["secret"]
    assert call(r, revocable, "GET", "projects")[0] == 200
    s, revoked = call(r, key_a, "POST", f"workspace/keys/{made['key']['key_id']}/revoke")
    assert s == 200 and revoked["active"] is False and revoked["revoked_at"]
    real_now = r.now
    r.now = lambda: real_now() + __import__("datetime").timedelta(seconds=61)
    denied = []
    for key in (wrong, other, revocable, short, "Bearer", key_a + "x"):
        denied.append(call(r, key, "GET", "projects"))
    r.now = real_now
    assert all(d == denied[0] for d in denied) and code(denied[0]) == (401, "unauthenticated")
    # scheme / header shape
    assert code(call(r, None, "GET", "projects", Authorization=f"Basic {key_a}")) == (
        401,
        "unauthenticated",
    )
    assert call(r, key_a, "GET", "projects")[0] == 200  # the owner key is untouched


def test_missing_scope_is_refused_before_anything_is_written(two):
    r, ws_a, key_a, *_ = two
    keys = {}
    for scopes in (["read"], ["write"], ["invoke"], ["read", "write"]):
        s, made = call(r, key_a, "POST", "workspace/keys", {"name": "-".join(scopes),
                                                             "scopes": scopes})  # fmt: skip
        assert s == 201
        keys[tuple(scopes)] = made["secret"]
    ro, wo, io_ = keys[("read",)], keys[("write",)], keys[("invoke",)]
    assert code(call(r, ro, "POST", "projects", {"name": "x"})) == (403, "insufficient_scope")
    assert code(call(r, io_, "POST", "projects", {"name": "x"})) == (403, "insufficient_scope")
    assert code(call(r, wo, "GET", "projects")) == (403, "insufficient_scope")
    assert call(r, ro, "GET", "projects") == (200, {"projects": []})  # nothing was created
    assert code(call(r, wo, "POST", "deployments/x/invoke", {"inputs": {}})) == (
        403,
        "insufficient_scope",
    )
    assert code(call(r, ro, "POST", f"inferences/inf-{'0' * 24}/feedback", {})) == (
        403,
        "insufficient_scope",
    )
    assert code(call(r, wo, "POST", "workspace/keys", {"name": "x", "scopes": ["read"]})) == (
        403,
        "insufficient_scope",
    )
    # a key can never mint more than it holds
    rw = keys[("read", "write")]
    assert code(call(r, rw, "GET", "workspace/keys")) == (403, "insufficient_scope")


def test_plaintext_keys_are_returned_once_and_never_persisted(two, tmp_path, caplog):
    r, ws_a, key_a, ws_b, key_b = two
    caplog.set_level("DEBUG")
    s, made = call(r, key_a, "POST", "workspace/keys", {"name": "ci", "scopes": ["read"]})
    assert s == 201 and made["secret"].startswith(f"{KEY_PREFIX}{made['key']['key_id']}_")
    listed = call(r, key_a, "GET", "workspace/keys")[1]["keys"]
    assert all("secret" not in k and "secret_hash" not in k for k in listed)
    call(r, made["secret"], "GET", "projects")
    secrets_ = [key_a, key_b, made["secret"]] + [parse_key(k)[1] for k in (key_a, key_b)]
    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            for s_ in secrets_:
                assert s_.encode() not in blob, path
    assert not any(s_ in caplog.text for s_ in secrets_)
    stored = r.identity.key(made["key"]["key_id"])
    assert stored.secret_hash == hash_secret(*parse_key(made["secret"]))
    # immutable: a key's hash / scopes / workspace cannot be rewritten in place
    conn = sqlite3.connect(tmp_path / "data" / "identity.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE api_keys SET scopes='read,write,invoke,admin'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE api_keys SET workspace_id=?", (ws_b,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM api_keys")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE workspaces SET name='x'")
    conn.close()


def test_a_caller_cannot_choose_its_workspace(two):
    r, ws_a, key_a, ws_b, key_b = two
    s, project = call(r, key_b, "POST", "projects", {"name": "B only"})
    pid = project["project_id"]
    for attempt in (
        call(r, key_a, "GET", f"projects/{pid}", **{"X-Wynk-Workspace": ws_b}),
        call(r, key_a, "GET", f"projects/{pid}", **{"X-Workspace-Id": ws_b}),
        call(r, key_a, "GET", f"projects/{pid}?workspace_id={ws_b}"),
    ):
        assert attempt[0] in (400, 404) and attempt[1]["error"]["code"] in (
            "project_not_found",
            "unknown_parameter",
        )
    assert call(r, key_a, "GET", "projects", **{"X-Wynk-Workspace": ws_b}) == (
        200,
        {"projects": []},
    )
    assert code(call(r, key_a, "GET", f"workspaces/{ws_b}")) == (404, "workspace_not_found")
    assert code(call(r, key_a, "GET", "workspaces/ws-nonexistent")) == (404, "workspace_not_found")
    assert call(r, key_a, "GET", f"workspaces/{ws_a}")[1]["workspace"]["workspace_id"] == ws_a
    # nor another workspace's keys / members
    other_key = parse_key(key_b)[0]
    assert code(call(r, key_a, "POST", f"workspace/keys/{other_key}/revoke")) == (
        404,
        "api_key_not_found",
    )
    assert code(call(r, key_a, "POST", f"workspace/keys/{'e' * 16}/revoke")) == (
        404,
        "api_key_not_found",
    )
    assert call(r, key_b, "GET", "projects")[0] == 200  # B's key still works
    b_owner = r.identity.user_by_email("b@beta.test").user_id
    assert code(call(r, key_a, "POST", f"workspace/members/{b_owner}/remove")) == (
        404,
        "member_not_found",
    )


def test_cross_tenant_project_ids_are_never_reused(two):
    r, ws_a, key_a, ws_b, key_b = two
    s, pa = call(r, key_a, "POST", "projects", {"name": "A"})
    pid = pa["project_id"]
    assert code(call(r, key_b, "GET", f"projects/{pid}")) == (404, "project_not_found")
    assert code(call(r, key_b, "POST", f"projects/{pid}/uploads?format=jsonl", DATA)) == (
        404,
        "project_not_found",
    )
    assert code(call(r, key_b, "GET", f"projects/{pid}/datasets")) == (404, "project_not_found")
    # and A's project gained nothing
    assert call(r, key_a, "GET", f"projects/{pid}/datasets") == (200, {"datasets": []})
    assert r.tenant(ws_a).usage()["uploads"] == 0 and r.tenant(ws_b).usage()["uploads"] == 0


# -- 3. memberships and roles ---------------------------------------------------------------
def test_membership_roles(two):
    r, ws_a, key_a, *_ = two
    s, member = call(r, key_a, "POST", "workspace/members", {"email": "m@alpha.test"})
    assert s == 201 and member["role"] == "MEMBER"
    uid = member["user_id"]
    assert code(call(r, key_a, "POST", "workspace/members", {"email": "m@alpha.test"})) == (
        409,
        "member_conflict",
    )
    # an OWNER can give a MEMBER keys, but never the admin scope
    assert code(
        call(r, key_a, "POST", "workspace/keys", {"name": "x", "scopes": ["admin"], "user_id": uid})
    ) == (422, "invalid_scopes")
    s, mk = call(r, key_a, "POST", "workspace/keys",
                 {"name": "m", "scopes": ["read", "write"], "user_id": uid})  # fmt: skip
    assert s == 201 and mk["key"]["user_id"] == uid
    mkey = mk["secret"]
    assert call(r, mkey, "GET", "workspace")[1]["principal"]["role"] == "MEMBER"
    assert call(r, mkey, "POST", "projects", {"name": "by member"})[0] == 201
    for path, body in (
        ("workspace/members", {"email": "x@alpha.test"}),
        (f"workspace/members/{uid}/remove", None),
        ("workspace/keys", {"name": "y", "scopes": ["read"]}),
        ("workspaces", {"name": "mine"}),
    ):
        assert code(call(r, mkey, "POST", path, body)) == (403, "insufficient_scope"), path
    # the last OWNER cannot be removed
    owner = call(r, key_a, "GET", "workspace")[1]["principal"]["user_id"]
    assert code(call(r, key_a, "POST", f"workspace/members/{owner}/remove")) == (
        409,
        "member_conflict",
    )
    # removing a member kills their keys immediately (no key revocation needed)
    assert call(r, key_a, "POST", f"workspace/members/{uid}/remove")[0] == 200
    assert code(call(r, mkey, "GET", "projects")) == (401, "unauthenticated")
    members = call(r, key_a, "GET", "workspace/members")[1]["members"]
    assert [m["user_id"] for m in members] == [owner]
    conn = sqlite3.connect(r.root / "identity.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):  # role / workspace never change in place
        conn.execute("UPDATE memberships SET role='OWNER'")
    conn.close()


def test_a_new_workspace_gets_its_owner_and_a_one_time_key(two):
    r, ws_a, key_a, *_ = two
    s, made = call(r, key_a, "POST", "workspaces", {"name": "Gamma"})
    assert s == 201 and made["membership"]["role"] == "OWNER"
    ws_c, key_c = made["workspace"]["workspace_id"], made["api_key"]["secret"]
    assert ws_c not in (ws_a,) and call(r, key_c, "GET", "projects") == (200, {"projects": []})
    assert call(r, key_c, "GET", "workspace")[1]["workspace"]["workspace_id"] == ws_c
    assert call(r, key_a, "GET", "workspace")[1]["workspace"]["workspace_id"] == ws_a


# -- 4. quotas ------------------------------------------------------------------------------
def test_quotas_refuse_before_writing_anything(tmp_path):
    q = Quotas(max_projects=1, max_uploads=1, max_stored_bytes=len(DATA) + 10, max_api_keys=2)
    r = build_api(tmp_path / "data", quotas=q)
    ws, key = workspace(r, "Q", "q@q.test")
    s, p = call(r, key, "POST", "projects", {"name": "one"})
    assert s == 201
    before = _fingerprint(r, ws)
    err = call(r, key, "POST", "projects", {"name": "two"})
    assert code(err) == (429, "quota_exceeded")
    assert err[1]["error"]["details"] == {"resource": "projects", "limit": 1, "used": 1}
    assert _fingerprint(r, ws) == before
    pid = p["project_id"]
    big = b"a,b\n" + b"10,20\n" * (len(DATA) // 6 + 4)  # valid, but over the byte quota
    assert len(big) > len(DATA) + 10
    blobs = sorted((r.root / "workspaces" / ws / "blobs").rglob("*"))
    assert code(call(r, key, "POST", f"projects/{pid}/uploads?format=csv", big)) == (
        429,
        "quota_exceeded",
    )
    assert _fingerprint(r, ws) == before
    assert sorted((r.root / "workspaces" / ws / "blobs").rglob("*")) == blobs
    assert call(r, key, "POST", f"projects/{pid}/uploads?format=jsonl", DATA)[0] == 201
    assert code(call(r, key, "POST", f"projects/{pid}/uploads?format=csv", b"a,b\n1,2\n")) == (
        429,
        "quota_exceeded",
    )
    # API keys: the owner key + one more
    assert call(r, key, "POST", "workspace/keys", {"name": "k", "scopes": ["read"]})[0] == 201
    keys_before = r.identity.keys(ws)
    assert code(call(r, key, "POST", "workspace/keys", {"name": "k2", "scopes": ["read"]})) == (
        429,
        "quota_exceeded",
    )
    assert r.identity.keys(ws) == keys_before
    s, quota = call(r, key, "GET", "workspace/quota")
    assert quota["usage"]["projects"] == 1 and quota["usage"]["api_keys"] == 2
    assert quota["limits"]["projects"] == 1
    # another workspace's usage is its own
    ws2, key2 = workspace(r, "Q2", "q2@q.test")
    assert call(r, key2, "POST", "projects", {"name": "one"})[0] == 201
    assert Quotas.from_env({"WYNK_QUOTA_PROJECTS": "7"}).max_projects == 7


@maf
def test_the_active_job_quota_counts_running_experiments(tmp_path):
    r = router(tmp_path / "data", quotas=Quotas(max_active_jobs=1))
    ws, key = workspace(r, "J", "j@j.test")
    data = dataset(r, key)
    d = definition()
    body = {
        "dataset_id": "capitals",
        "dataset_version": 1,
        "splits_hash": data["splits"]["splits_hash"],
        "contract": d.contract.model_dump(mode="json"),
        "plan": d.plan.model_dump(mode="json"),
    }
    assert call(r, key, "POST", "experiments", body)[0] == 201
    jobs_before = _fingerprint(r, ws)["jobs.sqlite3"]
    assert code(call(r, key, "POST", "experiments", body)) == (429, "quota_exceeded")
    assert _fingerprint(r, ws)["jobs.sqlite3"] == jobs_before
    api = r.tenant(ws).api
    JobWorker(api.jobs.store, api.jobs.runtime, heartbeat=False).run_until_idle()
    assert call(r, key, "POST", "experiments", body)[0] == 201  # the first one finished


# -- 5. store-level binding -----------------------------------------------------------------
def test_stores_refuse_another_workspace_or_an_unscoped_open(two, tmp_path):
    r, ws_a, key_a, ws_b, key_b = two
    call(r, key_a, "POST", "projects", {"name": "A"})
    part = tmp_path / "data" / "workspaces" / ws_a
    opens = [
        lambda ws: SQLiteDatasetRepository(part / "metadata.sqlite3", workspace_id=ws),
        lambda ws: SQLiteJobStore(part / "jobs.sqlite3", workspace_id=ws),
        lambda ws: SQLiteChampionStore(part / "champions.sqlite3", workspace_id=ws),
        lambda ws: SQLiteDeploymentStore(part / "deployments.sqlite3", workspace_id=ws),
        lambda ws: SQLiteMonitoringStore(part / "monitoring.sqlite3", workspace_id=ws),
        lambda ws: LocalBlobStore(part / "blobs", workspace_id=ws),
    ]
    for open_ in opens:
        with pytest.raises(TenantBindingError):
            open_(ws_b)
        with pytest.raises(TenantBindingError):
            open_(None)
        open_(ws_a)  # its own workspace: fine
    conn = sqlite3.connect(part / "metadata.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE tenant_binding SET workspace_id=?", (ws_b,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM tenant_binding")
    conn.close()


def test_a_tampered_partition_binding_fails_closed(two, tmp_path):
    r, ws_a, key_a, ws_b, key_b = two
    call(r, key_a, "POST", "projects", {"name": "A"})
    path = tmp_path / "data" / "workspaces" / ws_a / "jobs.sqlite3"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER tenant_binding_no_update")
    conn.execute("UPDATE tenant_binding SET workspace_id=?", (ws_b,))
    conn.close()
    fresh = build_api(tmp_path / "data")  # a restarted process re-opens every store
    assert code(call(fresh, key_a, "GET", "projects")) == (500, "tenant_integrity_error")
    assert call(fresh, key_b, "GET", "projects")[0] == 200  # B is unaffected
    with pytest.raises(TenantBindingError):
        bind_sqlite(path, ws_a)


# -- 6. restart and legacy migration --------------------------------------------------------
def test_a_restart_preserves_workspaces_keys_and_isolation(two, tmp_path):
    r, ws_a, key_a, ws_b, key_b = two
    s, pa = call(r, key_a, "POST", "projects", {"name": "A"})
    again = build_api(tmp_path / "data")
    assert call(again, key_a, "GET", f"projects/{pa['project_id']}") == (200, pa)
    assert code(call(again, key_b, "GET", f"projects/{pa['project_id']}")) == (
        404,
        "project_not_found",
    )
    assert call(again, key_b, "GET", "workspace")[1]["workspace"]["workspace_id"] == ws_b


def _legacy(root: Path) -> str:
    """A pre-#32 single-tenant data dir: the stores directly at the root, unbound."""
    service = DatasetService(
        SQLiteDatasetRepository(root / "metadata.sqlite3"), LocalBlobStore(root / "blobs")
    )
    project = service.create_project(NewProject(name="old"))
    upload, _ = service.upload(project.project_id, DATA, "jsonl")
    service.register(upload.upload_id, RegisterDataset.model_validate(CAPITALS))
    api = ProductAPI(service)
    assert api.handle("GET", "/api/v1/projects", {}, io.BytesIO()).status == 200
    for name, cls in (
        ("jobs.sqlite3", SQLiteJobStore),
        ("champions.sqlite3", SQLiteChampionStore),
        ("deployments.sqlite3", SQLiteDeploymentStore),
        ("monitoring.sqlite3", SQLiteMonitoringStore),
    ):
        cls(root / name)
    return project.project_id


def test_legacy_data_is_assigned_explicitly_and_idempotently(tmp_path):
    root = tmp_path / "data"
    pid = _legacy(root)
    with pytest.raises(LegacyDataNotMigrated):  # never served, never guessed
        build_api(root)
    out = migrate_legacy(root, "ops@example.test", workspace_id="ws-legacy")
    assert out["status"] == "migrated" and set(out["moved"]) == {
        *tenancy.STORE_FILES,
        *tenancy.STORE_DIRS,
    }
    assert migrate_legacy(root, "ops@example.test", workspace_id="ws-legacy")["status"] == (
        "already_migrated"
    )
    with pytest.raises(IdentityConflict):  # a different workspace for the same data: refused
        migrate_legacy(root, "ops@example.test", workspace_id="ws-other")
    r = build_api(root)
    key = r.issue_key(
        "ws-legacy",
        r.identity.user_by_email("ops@example.test").user_id,
        "ops",
        list(SCOPES),
        created_by="cli",
    )[1]
    assert call(r, key, "GET", f"projects/{pid}")[1]["project_id"] == pid
    assert call(r, key, "GET", "datasets/capitals")[0] == 200
    for name in tenancy.STORE_FILES:
        assert sqlite_binding(root / "workspaces" / "ws-legacy" / name) == "ws-legacy"
    with pytest.raises(TenantBindingError):  # the old unscoped handle no longer opens it
        SQLiteDatasetRepository(root / "workspaces" / "ws-legacy" / "metadata.sqlite3")
    # a second workspace never sees the migrated data
    ws2, key2 = workspace(r, "New", "new@example.test")
    assert code(call(r, key2, "GET", f"projects/{pid}")) == (404, "project_not_found")


def test_an_interrupted_legacy_migration_resumes(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _legacy(root)
    real = Path.rename
    calls = {"n": 0}

    def crash(self, target):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("power cut")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(OSError):
        migrate_legacy(root, "ops@example.test")
    monkeypatch.setattr(Path, "rename", real)
    with pytest.raises(LegacyDataNotMigrated):
        build_api(root)
    assert migrate_legacy(root, "ops@example.test")["status"] == "migrated"
    assert migrate_legacy(root, "ops@example.test")["status"] == "already_migrated"
    build_api(root)


def test_the_bootstrap_key_is_idempotent_and_hash_only(tmp_path):
    key = f"{KEY_PREFIX}{'1' * 16}_{'Z' * 43}"
    r = build_api(tmp_path / "data")
    assert r.bootstrap("dev@x.test", key_plaintext=key) == ("ws-default", key)
    assert r.bootstrap("dev@x.test", key_plaintext=key) == ("ws-default", None)
    assert build_api(tmp_path / "data").bootstrap("dev@x.test", key_plaintext=key)[1] is None
    assert call(r, key, "GET", "projects")[0] == 200
    with pytest.raises(IdentityConflict):
        r.bootstrap("dev@x.test", key_plaintext=f"{KEY_PREFIX}{'1' * 16}_{'Y' * 43}")
    assert key.encode() not in (tmp_path / "data" / "identity.sqlite3").read_bytes()


# -- 7. review fixes: one job-admission authority, isolated worker start, idempotent retries --
def _active_jobs(r: TenantRouter, ws: str) -> int:
    return r.tenant(ws).usage()["active_jobs"]


@maf
def test_trigger_evaluation_cannot_bypass_the_active_job_quota(tmp_path):
    """max_active_jobs = 1, one experiment already active, enough production feedback to
    TRIGGER: evaluating the trigger records the decision but admits no hidden challenger;
    once capacity frees, the SAME trigger creates exactly one challenger."""
    r = router(tmp_path / "data", quotas=Quotas(max_active_jobs=1))
    ws, key = workspace(r, "Q", "q@q.test")
    data = dataset(r, key)
    job = optimize(r, ws, key, data)  # the champion's experiment (now COMPLETED)
    s, promo = call(r, key, "POST", f"experiments/{job}/promote")
    champion = promo["record"]["champion"]["champion_id"]
    version = call(r, key, "POST", "workflows", {"champion_id": champion})[1]["version_id"]
    call(r, key, "POST", f"workflows/{version}/stage", {"expected_revision": 0})
    call(r, key, "POST", f"workflows/{version}/promote", {"expected_revision": 1})
    call(r, key, "POST", "monitoring/policies", SMALL.model_dump(mode="json"))
    for country, city in COUNTRIES[:4]:
        inf = call(r, key, "POST", f"deployments/{LINEAGE}/invoke",
                   {"inputs": inputs_for(country, city)})[1]  # fmt: skip
        assert call(r, key, "POST", f"inferences/{inf['inference_id']}/feedback",
                    {"inputs": inputs_for(country, city), "output": inf["output"],
                     "expected": {"answer": "Nowhere"}})[0] == 201  # fmt: skip
    # one experiment active: the workspace's single slot is taken
    d = definition()
    body = {
        "dataset_id": "capitals",
        "dataset_version": 1,
        "splits_hash": data["splits"]["splits_hash"],
        "contract": d.contract.model_dump(mode="json"),
        "plan": d.plan.model_dump(mode="json"),
    }
    s, blocker = call(r, key, "POST", "experiments", body)
    assert s == 201 and _active_jobs(r, ws) == 1
    datasets_before = r.tenant(ws).usage()["dataset_versions"]
    window = {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id}

    s, trig = call(r, key, "POST", f"workflows/{version}/triggers/evaluate", window)
    assert s == 201 and trig["outcome"] == "TRIGGERED", trig  # the decision is usable
    assert _active_jobs(r, ws) == 1  # ...but no hidden challenger was admitted
    jobs = {j["job_id"] for j in call(r, key, "GET", "experiments")[1]["jobs"]}
    assert jobs == {job, blocker["job_id"]}
    assert r.tenant(ws).usage()["dataset_versions"] == datasets_before  # nothing half-built
    s, ch = call(r, key, "GET", f"triggers/{trig['trigger_id']}/challenger")
    assert s == 200 and ch["challenger_job_id"] is None and ch["state"] is None
    # idempotent re-evaluation: same decision, still usable, still no job
    s, again = call(r, key, "POST", f"workflows/{version}/triggers/evaluate", window)
    assert s == 200 and again["trigger_id"] == trig["trigger_id"]
    assert _active_jobs(r, ws) == 1
    # the explicit route shares the SAME authority and says why
    err = call(r, key, "POST", f"triggers/{trig['trigger_id']}/reoptimize")
    assert code(err) == (429, "quota_exceeded")
    assert err[1]["error"]["details"]["resource"] == "active_jobs"
    assert err[1]["error"]["details"]["trigger_id"] == trig["trigger_id"]
    assert _active_jobs(r, ws) == 1

    # free the slot: the same trigger resumes and creates exactly one challenger
    assert call(r, key, "POST", f"experiments/{blocker['job_id']}/cancel")[0] == 200
    api = r.tenant(ws).api
    JobWorker(api.jobs.store, api.jobs.runtime, heartbeat=False).run_until_idle()
    assert _active_jobs(r, ws) == 0
    s, ch = call(r, key, "POST", f"triggers/{trig['trigger_id']}/reoptimize")
    assert s == 200 and ch["state"] == "JOB_CREATED" and ch["challenger_job_id"], ch
    assert _active_jobs(r, ws) == 1
    again = call(r, key, "POST", f"triggers/{trig['trigger_id']}/reoptimize")[1]
    assert again["challenger_job_id"] == ch["challenger_job_id"]  # never duplicated
    s, _ = call(r, key, "POST", f"workflows/{version}/triggers/evaluate", window)
    assert s == 200 and _active_jobs(r, ws) == 1
    jobs = {j["job_id"] for j in call(r, key, "GET", "experiments")[1]["jobs"]}
    assert jobs == {job, blocker["job_id"], ch["challenger_job_id"]}


def test_the_store_is_the_job_admission_authority(tmp_path):
    """No caller - not the API router - gets past ``create_job``'s in-transaction count."""
    from store.tenancy import QuotaExceeded, StoreQuota

    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.quota = StoreQuota(max_active_jobs=1)
    store.create_job("j-1", "h", "{}", "{}", [("fixed", 0, "r1")])
    with pytest.raises(QuotaExceeded):
        store.create_job("j-2", "h", "{}", "{}", [("fixed", 0, "r2")])
    with pytest.raises(QuotaExceeded):
        store.admit_job()
    assert store.list_job_ids() == ["j-1"]
    from store.datasets import Conflict

    with pytest.raises(Conflict):  # an existing id is a conflict, never a quota hit
        store.create_job("j-1", "h", "{}", "{}", [("fixed", 0, "r1")])


@maf
def test_one_corrupt_workspace_cannot_stop_worker_start(tmp_path, caplog):
    root = tmp_path / "data"
    r = router(root)
    ws_a, key_a = workspace(r, "A", "a@a.test")
    ws_b, key_b = workspace(r, "B", "b@b.test")
    pa = call(r, key_a, "POST", "projects", {"name": "A"})[1]["project_id"]
    path = root / "workspaces" / ws_a / "jobs.sqlite3"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DROP TRIGGER tenant_binding_no_update")
    conn.execute("UPDATE tenant_binding SET workspace_id=?", (ws_b,))
    conn.close()

    restarted = router(root)
    caplog.set_level("ERROR")
    group = restarted.start_workers()  # must not fail globally
    try:
        assert group is not None
        assert ws_a not in restarted._tenants  # A: not opened, no worker
        assert ws_a in caplog.text and "integrity" in caplog.text
        assert restarted.tenant(ws_b).worker is not None  # B: started
        for path_ in ("projects", f"projects/{pa}", "experiments"):
            assert code(call(restarted, key_a, "GET", path_)) == (500, "tenant_integrity_error")
        data = dataset(restarted, key_b)  # B: fully usable, its worker runs its jobs
        d = definition()
        s, job = call(restarted, key_b, "POST", "experiments", {
            "dataset_id": "capitals", "dataset_version": 1,
            "splits_hash": data["splits"]["splits_hash"],
            "contract": d.contract.model_dump(mode="json"),
            "plan": d.plan.model_dump(mode="json"),
        })  # fmt: skip
        assert s == 201
        import time

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = call(restarted, key_b, "GET", f"experiments/{job['job_id']}")[1]["state"]
            if state == "COMPLETED":
                break
            time.sleep(0.05)
        assert state == "COMPLETED"
        # a workspace created later still starts normally
        ws_c, key_c = workspace(restarted, "C", "c@c.test")
        assert restarted.tenant(ws_c).worker is not None
        assert call(restarted, key_c, "GET", "projects") == (200, {"projects": []})
        assert code(call(restarted, key_a, "GET", "projects")) == (500, "tenant_integrity_error")
    finally:
        group.stop()


def test_no_runtime_means_no_partition_is_opened_at_start(two, tmp_path):
    r, ws_a, key_a, ws_b, key_b = two
    fresh = build_api(tmp_path / "data")
    assert fresh.start_workers() is None
    assert fresh._tenants == {}


def test_retries_that_create_nothing_pass_a_full_quota(tmp_path):
    q = Quotas(max_uploads=1, max_stored_bytes=len(DATA), max_dataset_versions=1)
    r = build_api(tmp_path / "data", quotas=q)
    ws, key = workspace(r, "Q", "q@q.test")
    pid = call(r, key, "POST", "projects", {"name": "p"})[1]["project_id"]
    s, up = call(r, key, "POST", f"projects/{pid}/uploads?format=jsonl", DATA)
    assert s == 201
    s, again = call(r, key, "POST", f"projects/{pid}/uploads?format=jsonl", DATA)
    assert s == 200 and again == up  # identical upload at a full quota: not refused
    s, v = call(r, key, "POST", f"uploads/{up['upload_id']}/register", CAPITALS)
    assert s == 201
    s, v2 = call(r, key, "POST", f"uploads/{up['upload_id']}/register", CAPITALS)
    assert s == 200 and v2 == v  # identical registration: not refused
    other = {**CAPITALS, "dataset_id": "capitals-2"}
    assert code(call(r, key, "POST", f"uploads/{up['upload_id']}/register", other)) == (
        429,
        "quota_exceeded",
    )
    assert code(call(r, key, "POST", f"projects/{pid}/uploads?format=csv", b"a,b\n1,2\n")) == (
        429,
        "quota_exceeded",
    )


def test_republishing_an_existing_version_passes_a_full_quota(tmp_path):
    from store.tenancy import QuotaExceeded, StoreQuota

    store = SQLiteDeploymentStore(tmp_path / "deployments.sqlite3")
    store.quota = StoreQuota(max_workflow_versions=1)
    first, created = store.publish("wv-" + "1" * 24, "l", "c-" + "1" * 24, "{}", "0" * 64, "t")
    assert created
    again, created = store.publish("wv-" + "1" * 24, "l", "c-" + "1" * 24, "{}", "0" * 64, "t")
    assert not created and again == first  # the same champion: nothing new, not refused
    with pytest.raises(QuotaExceeded):
        store.publish("wv-" + "2" * 24, "l", "c-" + "2" * 24, "{}", "0" * 64, "t")
    assert [v.version_id for v in store.versions()] == [first.version_id]
