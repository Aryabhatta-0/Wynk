"""Experiment-creation idempotency (dogfood audit R4): ``POST /api/v1/experiments`` with an
``Idempotency-Key`` creates at most one job per (workspace, key); a retry gets that job back.

Every test drives the real multi-tenant router (``tests.tenancy_support.dev_api``) over the
TEST DOUBLE passage runner of ``tests.test_durable_jobs``; ``Counting`` additionally counts each
runtime binding (``DatasetRuntime.backend``) and each model call, so "nothing was sent again"
is measured, not assumed."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from api.product import ApiResponse, build_api
from api.tenancy import Quotas
from core.constraints import ConstraintChecker
from experiments.jobs import DatasetRuntime, JobWorker, WorkflowBackend
from experiments.optimization_experiment import without_timing
from store.jobs import IdempotencyClaim, SQLiteJobStore
from tests.tenancy_support import DEV_KEY, DEV_WORKSPACE, DevAPI, dev_api
from tests.test_durable_jobs import MODEL_HASH, _create_body, _passage_runner, _register
from tests.test_product_api import call, error

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = "/api/v1/experiments"


class Counting:
    """The test runtime, plus counters: bindings (``backend``) and model calls."""

    def __init__(self) -> None:
        self.binds = 0
        self.calls = 0
        self.fail_binds = 0  # the next N bindings raise (a crash before the job exists)

    def __call__(self, service: Any) -> DatasetRuntime:
        def backend(definition: Any) -> WorkflowBackend:
            self.binds += 1
            if self.fail_binds:
                self.fail_binds -= 1
                raise RuntimeError("model backend unreachable")
            run = _passage_runner(MODEL_HASH)

            def counted(*args: Any) -> Any:
                self.calls += 1
                return run(*args)

            return WorkflowBackend(counted, ConstraintChecker())

        return DatasetRuntime(data=service.blobs.get, backend=backend, synthetic=True)


def create(api: Any, body: dict | bytes, key: str | None = None, **headers: str) -> ApiResponse:
    h = dict(headers)
    if key is not None:
        h["Idempotency-Key"] = key
    if isinstance(body, bytes):
        h["Content-Type"] = "application/json"
        return call(api, "POST", EXPERIMENTS, body, headers=h)
    return call(api, "POST", EXPERIMENTS, json_body=body, headers=h)


def replayed(res: ApiResponse) -> bool:
    return dict(res.headers).get("Idempotent-Replayed") == "true"


def rows(api: Any, table: str) -> int:
    conn = sqlite3.connect(api.jobs.store.path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def counts(api: Any) -> dict[str, int]:
    return {
        t: rows(api, t)
        for t in (
            "experiment_jobs",
            "experiment_units",
            "experiment_attempts",
            "experiment_idempotency",
        )
    }


def run_jobs(api: Any) -> None:
    JobWorker(api.jobs.store, api.jobs.runtime, heartbeat=False).run_until_idle()


@pytest.fixture
def counting() -> Counting:
    return Counting()


@pytest.fixture
def setup(tmp_path, counting):
    api = dev_api(tmp_path / "data", runtime=counting)
    version, splits = _register(api)
    return api, _create_body(version, splits)


# 1 -------------------------------------------------------------------------------------------
def test_without_a_key_every_create_is_a_new_job_as_before(setup):
    api, body = setup
    a, b = create(api, body), create(api, body)
    assert a.status == b.status == 201
    assert a.body["job_id"] != b.body["job_id"]
    assert not replayed(a) and not replayed(b) and a.headers == b.headers == ()
    assert a.body["experiment_id"] == b.body["experiment_id"]  # same science, two jobs
    assert counts(api)["experiment_jobs"] == 2 and counts(api)["experiment_idempotency"] == 0


# 2 + 7 + 12 ----------------------------------------------------------------------------------
def test_a_retry_after_a_lost_response_gets_the_same_job_and_spends_nothing(setup, counting):
    api, body = setup
    first = create(api, body, "retry-1")  # the client never reads this response
    assert first.status == 201 and not replayed(first)
    assert counting.binds == 1
    run_jobs(api)  # the job runs and spends before the client retries
    spent = counts(api)
    calls, binds = counting.calls, counting.binds
    events = len(api.audit.events(limit=1000))
    assert calls > 0 and spent["experiment_attempts"] > 0

    again = create(api, body, "retry-1")
    assert again.status == 201 and replayed(again)
    assert again.body["job_id"] == first.body["job_id"]
    assert again.body["state"] == "COMPLETED"  # the job's current view, the same job
    assert again.body["experiment_id"] == first.body["experiment_id"]
    run_jobs(api)
    assert counts(api) == spent  # no job, unit, attempt or binding row was added
    assert (counting.calls, counting.binds) == (calls, binds)  # nothing bound, nothing sent
    assert len(api.audit.events(limit=1000)) == events  # no further audit event

    pending = create(api, body, "retry-2")  # replayed before anything ran: the same body
    assert create(api, body, "retry-2").body == pending.body


# 3 -------------------------------------------------------------------------------------------
def test_canonically_equivalent_bodies_replay_the_same_job(setup):
    api, body = setup
    first = create(api, body, "canon-1")
    reordered = json.dumps(dict(reversed(list({**body, "workers": 1}.items()))), indent=3)
    again = create(api, reordered.encode(), "canon-1")  # explicit default, other order, spaces
    assert again.status == 201 and replayed(again)
    assert again.body["job_id"] == first.body["job_id"]
    assert counts(api)["experiment_jobs"] == 1


# 4 -------------------------------------------------------------------------------------------
def test_the_same_key_with_another_request_fails_closed(setup, counting):
    api, body = setup
    first = create(api, body, "conflict-1")
    binds = counting.binds
    changed = {**body, "workers": 2}
    err = error(create(api, changed, "conflict-1"), 409, "idempotency_conflict")
    assert first.body["job_id"] not in json.dumps(err) and "conflict-1" not in json.dumps(err)
    assert counting.binds == binds and counts(api)["experiment_jobs"] == 1
    other_plan = json.loads(json.dumps(body))
    other_plan["plan"]["seeds"] = [0]
    error(create(api, other_plan, "conflict-1"), 409, "idempotency_conflict")
    # the mapping is unchanged: the original request still replays
    again = create(api, body, "conflict-1")
    assert replayed(again) and again.body["job_id"] == first.body["job_id"]
    assert create(api, changed).status == 201  # the changed request itself is fine without it


def test_a_malformed_key_is_refused_before_anything_is_stored(setup, counting):
    api, body = setup
    for bad in ("", " ", "a b", "-lead", "x" * 129, "kéy", "a,b"):
        err = error(create(api, body, bad), 400, "invalid_request")
        assert err["details"]["field"] == "Idempotency-Key"
        if bad.strip():
            assert bad not in err["message"]  # the key is never echoed
    assert counting.binds == 0 and counts(api)["experiment_jobs"] == 0
    assert create(api, body, "x" * 128).status == 201  # the bound itself is accepted


# 5 -------------------------------------------------------------------------------------------
def test_one_literal_key_is_independent_in_two_workspaces(setup):
    api, body = setup
    a = create(api, body, "shared-key")
    ws_b, _, _, key_b = api.router.create_workspace("Other", "other@example.test")
    b_api = DevAPI(api.router, ws_b.workspace_id, key_b)
    version_b, splits_b = _register(b_api)
    body_b = _create_body(version_b, splits_b)
    b = create(b_api, body_b, "shared-key")
    assert b.status == 201 and not replayed(b)  # nothing of A's is visible or reused
    assert b.body["job_id"] != a.body["job_id"]
    # B reusing the key with A's exact body is B's own replay, never A's job
    assert create(b_api, body_b, "shared-key").body["job_id"] == b.body["job_id"]
    assert create(api, body, "shared-key").body["job_id"] == a.body["job_id"]
    error(call(b_api, "GET", f"{EXPERIMENTS}/{a.body['job_id']}"), 404, "job_not_found")
    err = error(create(b_api, {**body_b, "workers": 3}, "shared-key"), 409, "idempotency_conflict")
    assert a.body["job_id"] not in json.dumps(err) and b.body["job_id"] not in json.dumps(err)
    assert counts(api)["experiment_jobs"] == 1 and counts(b_api)["experiment_jobs"] == 1
    assert [j["job_id"] for j in call(b_api, "GET", EXPERIMENTS).body["jobs"]] == [b.body["job_id"]]


# 6 -------------------------------------------------------------------------------------------
def test_concurrent_requests_with_one_key_create_exactly_one_job(setup):
    api, body = setup
    routers = [DevAPI(build_api(api.router.root, runtime=Counting()), DEV_WORKSPACE, DEV_KEY)]
    routers += [routers[0]] * 3 + [api] * 4  # separate routers / stores / connections, shared
    barrier = threading.Barrier(len(routers))
    out: list[ApiResponse] = []

    def go(r: Any) -> None:
        barrier.wait()
        out.append(create(r, body, "thread-race"))

    threads = [threading.Thread(target=go, args=(r,)) for r in routers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(out) == len(routers) and all(r.status == 201 for r in out)
    assert len({r.body["job_id"] for r in out}) == 1
    assert sum(not replayed(r) for r in out) == 1  # exactly one request created it
    assert counts(api)["experiment_jobs"] == 1 and counts(api)["experiment_idempotency"] == 1


CHILD = """
import json, pathlib, sys, time
from api.product import build_api
from tests.tenancy_support import DEV_KEY, DEV_WORKSPACE, DevAPI
from tests.test_durable_jobs import _runtime
from tests.test_product_api import call
data, body, ready, go = sys.argv[1:5]
api = DevAPI(build_api(data, runtime=_runtime), DEV_WORKSPACE, DEV_KEY)
api.router.tenant(DEV_WORKSPACE)  # open the partition before the race
payload = json.loads(pathlib.Path(body).read_text())
pathlib.Path(ready).write_text("ready")
while not pathlib.Path(go).exists():
    time.sleep(0.001)
res = call(api, "POST", "/api/v1/experiments", json_body=payload,
           headers={"Idempotency-Key": "process-race"})
print(json.dumps({"status": res.status, "job_id": res.body.get("job_id"),
                  "replayed": dict(res.headers).get("Idempotent-Replayed")}))
"""


def test_separate_processes_with_one_key_create_exactly_one_job(setup, tmp_path):
    api, body = setup
    (tmp_path / "body.json").write_text(json.dumps(body))
    go = tmp_path / "go"
    n = 4
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                CHILD,
                str(api.router.root),
                str(tmp_path / "body.json"),
                str(tmp_path / f"ready{i}"),
                str(go),
            ],
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )
        for i in range(n)
    ]
    try:
        deadline = time.monotonic() + 120
        while not all((tmp_path / f"ready{i}").exists() for i in range(n)):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        go.write_text("go")  # every process sends at the same instant
        out = [json.loads(p.communicate(timeout=120)[0].strip().splitlines()[-1]) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert all(p.returncode == 0 for p in procs)
    assert all(o["status"] == 201 for o in out), out
    assert len({o["job_id"] for o in out}) == 1
    assert sum(o["replayed"] is None for o in out) == 1, out  # exactly one creator
    assert counts(api)["experiment_jobs"] == 1 and counts(api)["experiment_idempotency"] == 1


def test_racing_store_writers_bind_a_key_once(tmp_path):
    """The store alone: the check + insert + bind is one ``BEGIN IMMEDIATE`` transaction."""
    path = tmp_path / "jobs.sqlite3"
    SQLiteJobStore(path, workspace_id="ws-race")
    claim = IdempotencyClaim("experiments.create", "k" * 64, "r" * 64)
    barrier = threading.Barrier(8)
    bound: list[Any] = []

    def go(i: int) -> None:
        store = SQLiteJobStore(path, workspace_id="ws-race")  # its own connections
        barrier.wait()
        bound.append(store.create_job(f"j-{i}", "d" * 64, "{}", "{}", [("fixed", 0, "r")], claim))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    winners = [b for b in bound if b is None]
    assert len(bound) == 8 and len(winners) == 1
    assert len({b.job_id for b in bound if b is not None}) == 1
    store = SQLiteJobStore(path, workspace_id="ws-race")
    assert len(store.list_job_ids()) == 1
    assert store.idempotent(claim.operation, claim.key_hash).job_id == store.list_job_ids()[0]  # type: ignore[union-attr]
    with pytest.raises(sqlite3.IntegrityError):  # bound once: never overwritten or removed
        conn = sqlite3.connect(path)
        try:
            conn.execute("UPDATE experiment_idempotency SET job_id='j-x'")
        finally:
            conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        conn = sqlite3.connect(path)
        try:
            conn.execute("DELETE FROM experiment_idempotency")
        finally:
            conn.close()


# 8 -------------------------------------------------------------------------------------------
def test_a_restarted_server_keeps_the_mapping(setup, tmp_path, counting):
    api, body = setup
    first = create(api, body, "restart-1")
    restarted = DevAPI(build_api(api.router.root, runtime=Counting()), DEV_WORKSPACE, DEV_KEY)
    again = create(restarted, body, "restart-1")
    assert again.status == 201 and replayed(again)
    assert again.body["job_id"] == first.body["job_id"]
    error(create(restarted, {**body, "workers": 2}, "restart-1"), 409, "idempotency_conflict")
    assert counts(restarted)["experiment_jobs"] == 1


# 9 -------------------------------------------------------------------------------------------
def test_a_crash_before_the_job_exists_leaves_the_key_free(setup, counting):
    api, body = setup
    counting.fail_binds = 1
    error(create(api, body, "crash-before"), 500, "internal_error")
    assert counts(api)["experiment_jobs"] == 0 and counts(api)["experiment_idempotency"] == 0
    retry = create(api, body, "crash-before")
    assert retry.status == 201 and not replayed(retry)
    assert counts(api)["experiment_jobs"] == 1


def test_a_failure_inside_the_creation_transaction_binds_nothing(setup):
    api, body = setup
    conn = sqlite3.connect(api.jobs.store.path)
    conn.execute(
        "CREATE TRIGGER fail_units BEFORE INSERT ON experiment_units "
        "BEGIN SELECT RAISE(ABORT, 'disk gone'); END"
    )
    conn.close()
    res = create(api, body, "crash-during")
    assert res.status >= 500
    assert counts(api) == {
        "experiment_jobs": 0,
        "experiment_units": 0,
        "experiment_attempts": 0,
        "experiment_idempotency": 0,
    }  # the job row was rolled back with the key: no key names a missing / partial job
    conn = sqlite3.connect(api.jobs.store.path)
    conn.execute("DROP TRIGGER fail_units")
    conn.close()
    retry = create(api, body, "crash-during")
    assert retry.status == 201 and not replayed(retry)
    assert counts(api)["experiment_jobs"] == 1 and counts(api)["experiment_idempotency"] == 1


def test_a_quota_refusal_binds_nothing_and_a_replay_needs_no_quota(tmp_path):
    api = dev_api(tmp_path / "data", runtime=Counting(), quotas=Quotas(max_active_jobs=1))
    version, splits = _register(api)
    body = _create_body(version, splits)
    first = create(api, body, "quota-1")
    assert first.status == 201
    # the workspace is at its active-job limit: a replay is still answered (admits nothing) ...
    again = create(api, body, "quota-1")
    assert replayed(again) and again.body["job_id"] == first.body["job_id"]
    # ... while a new key is refused, in the creation transaction, without binding the key
    error(create(api, body, "quota-2"), 429, "quota_exceeded")
    assert counts(api)["experiment_idempotency"] == 1
    assert call(api, "POST", f"{EXPERIMENTS}/{first.body['job_id']}/cancel").status == 200
    retry = create(api, body, "quota-2")
    assert retry.status == 201 and not replayed(retry)
    assert retry.body["job_id"] != first.body["job_id"]


# 10 ------------------------------------------------------------------------------------------
def test_read_only_and_unauthenticated_callers_reserve_nothing(setup, counting):
    api, body = setup
    made = call(api, "POST", "/api/v1/workspace/keys", json_body={"name": "ro", "scopes": ["read"]})
    assert made.status == 201, made.body
    read_only = {"Authorization": f"Bearer {made.body['secret']}"}
    error(create(api, body, "guarded-1", **read_only), 403, "insufficient_scope")
    bogus = {"Authorization": "Bearer wynk_sk_0000000000000000_" + "Z" * 43}
    error(create(api, body, "guarded-1", **bogus), 401, "unauthenticated")
    error(create(api, body, "guarded-1", Authorization=""), 401, "unauthenticated")
    assert counting.binds == 0
    assert counts(api)["experiment_jobs"] == 0 and counts(api)["experiment_idempotency"] == 0
    first = create(api, body, "guarded-1")  # the key was never reserved by those requests
    assert first.status == 201 and not replayed(first)


# 11 ------------------------------------------------------------------------------------------
def test_the_key_never_changes_the_jobs_science(setup):
    api, body = setup
    plain = create(api, body)
    keyed = create(api, body, "science-1")
    p, k = (api.jobs.store.get_job(r.body["job_id"]) for r in (plain, keyed))
    assert p.definition_json == k.definition_json  # type: ignore[union-attr]
    assert p.definition_hash == k.definition_hash  # type: ignore[union-attr]
    assert p.identity_json == k.identity_json  # type: ignore[union-attr]  # ids, seeds, budget
    assert "science-1" not in k.definition_json + k.identity_json  # type: ignore[union-attr]
    assert [(u.strategy, u.seed, u.position, u.run_id) for u in p.units] == [  # type: ignore[union-attr]
        (u.strategy, u.seed, u.position, u.run_id)
        for u in k.units  # type: ignore[union-attr]
    ]
    for field in ("experiment_id", "problem_id", "contract_hash", "protocol_hash", "budget"):
        assert plain.body[field] == keyed.body[field]
    run_jobs(api)
    arts = [
        call(api, "GET", f"{EXPERIMENTS}/{r.body['job_id']}/artifact").body for r in (plain, keyed)
    ]
    assert arts[0]["experiment_id"] == arts[1]["experiment_id"]
    assert without_timing(arts[0]["artifact"]) == without_timing(arts[1]["artifact"])
    provenance = [
        call(api, "GET", f"{EXPERIMENTS}/{r.body['job_id']}/provenance").body
        for r in (plain, keyed)
    ]
    assert provenance[0]["provenance_id"] == provenance[1]["provenance_id"]


def test_a_replay_survives_the_dataset_source_being_deleted(setup, counting):
    """A retry recovers the stored job without re-checking (or re-reading) its inputs."""
    api, body = setup
    first = create(api, body, "deleted-1")
    res = call(api, "DELETE", "/api/v1/datasets/capitals-upload")
    assert res.status == 200, res.body
    binds = counting.binds
    again = create(api, body, "deleted-1")
    assert replayed(again) and again.body["job_id"] == first.body["job_id"]
    assert counting.binds == binds
    error(create(api, body, "deleted-2"), 410, "dataset_source_deleted")  # a new one is refused
