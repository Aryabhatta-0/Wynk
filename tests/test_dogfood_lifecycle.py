"""v0.1 production dogfood: a fresh user's order-desk CSV through Wynk's whole lifecycle.

Every step goes through ``TenantRouter.handle`` (the HTTP entry point) of an API wired exactly
like ``python -m api.product --model-registry ...``: ``registry_runtime`` executes experiments
with the real MAF ``WorkflowRunner`` and the real ``OpenAICompatibleClient``, and
``registry_inference`` re-reads the registry FILE on every invocation. The only fake is the far
end of the model's HTTP connection (``tests.dogfood_support.FakeProvider``), so no data leaves
the machine. See docs/v0_1_release_readiness.md for the matrix these tests back.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.provenance import differences, scientific
from runtime.prompts import PROMPT_TEMPLATE_VERSION
from store.audit import SQLiteAuditLog
from tests.dogfood_support import (
    DATASET_ID,
    FIXTURE,
    INSTRUCTIONS,
    LINEAGE,
    PROVIDER_KEY,
    REGISTER,
    SPLIT_PLAN,
    TASK_ID,
    FakeProvider,
    call,
    code,
    entry_for,
    files_under,
    product,
    write_registry,
)

pytest.importorskip("agent_framework")  # the real runtime executes workflows with MAF

NEW_ORDER = {
    "question": "Which city does order ORD-2001 ship to?",
    "order_record": "Order ORD-2001 (1 x tarp) was packed at the Reno warehouse and ships to "
    "Lincoln by Ground on 2026-04-01.",
}


# -- harness ----------------------------------------------------------------------------------
class World:
    """One data directory, one registry file, one fake provider; ``restart`` = a new process."""

    def __init__(self, root: Path, provider: FakeProvider, *, key: str | None = PROVIDER_KEY):
        self.root, self.provider, self.key = root, provider, key
        root.mkdir(parents=True, exist_ok=True)
        self.registry = root / "registry.json"
        self.entry = entry_for(provider.endpoint)
        write_registry(self.registry, self.entry)
        self.data = root / "data"
        self.responses: list[Any] = []
        self.r = product(self.data, self.registry, key=key)

    def restart(self) -> None:
        self.r = product(self.data, self.registry, key=self.key)

    def call(self, key: str | None, method: str, path: str, body: Any = None, **h: str):
        res = call(self.r, key, method, path, body, **h)
        self.responses.append(res)
        return res

    def ok(self, key: str, method: str, path: str, body: Any = None, status: int = 200):
        s, b = self.call(key, method, path, body)
        assert s == status, (path, s, b)
        return b

    def workspace(self, name: str, email: str) -> tuple[str, str]:
        ws, _, _, key = self.r.create_workspace(name, email)
        return ws.workspace_id, key

    def api(self, ws: str) -> Any:
        return self.r.tenant(ws).api

    # -- the user's steps --
    def dataset(self, key: str, raw: bytes | None = None) -> dict[str, Any]:
        project = self.ok(key, "POST", "projects", {"name": "Order desk"}, 201)
        up = self.ok(
            key,
            "POST",
            f"projects/{project['project_id']}/uploads?format=csv",
            FIXTURE.read_bytes() if raw is None else raw,
            201,
        )
        version = self.ok(key, "POST", f"uploads/{up['upload_id']}/register", REGISTER, 201)
        splits = self.ok(key, "POST", f"datasets/{DATASET_ID}/versions/1/splits", SPLIT_PLAN, 201)
        return {"project": project, "upload": up, "version": version, "splits": splits}

    def experiment_body(
        self, data: dict[str, Any], strategies=("fixed", "random", "aco"), **contract: Any
    ) -> dict[str, Any]:
        return {
            "dataset_id": DATASET_ID,
            "dataset_version": data["version"]["spec"]["dataset_version"],
            "splits_hash": data["splits"]["splits_hash"],
            "contract": contract_for(data["version"]["spec"], **contract),
            "plan": plan_for(self.entry, strategies),
        }

    def run(self, ws: str, job_id: str, *, clock=None) -> None:
        """A worker exactly as ``api.product.start_worker`` builds it, run to idle."""
        from experiments.jobs import JobWorker

        api = self.api(ws)
        kw = {"clock": clock} if clock is not None else {}
        worker = JobWorker(
            api.jobs.store, api.jobs.runtime, priors=api.jobs.priors, heartbeat=False, **kw
        )
        worker.recover()
        worker.run_until_idle(job_id)

    def deploy(self, key: str, version_id: str) -> int:
        rev = self.ok(key, "GET", f"deployments/{LINEAGE}")["revision"]
        self.ok(key, "POST", f"workflows/{version_id}/stage", {"expected_revision": rev})
        self.ok(key, "POST", f"workflows/{version_id}/promote", {"expected_revision": rev + 1})
        return rev + 2


def contract_for(spec: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """The TaskContract a user writes for the registered dataset version."""
    return {
        "task_id": TASK_ID,
        "contract_version": 1,
        "task_type": "question_answering",
        "instructions": INSTRUCTIONS,
        "input_schema": {
            "fields": [
                {"name": "question", "type": "string"},
                {"name": "order_record", "type": "string"},
            ]
        },
        "output_schema": {"fields": [{"name": "answer", "type": "string"}]},
        "dataset": spec,
        "evaluation": {"evaluator": "exact_match", "config": {"case_sensitive": False}},
        "constraints": {
            "maximum_tokens_per_example": 20_000,
            "maximum_wall_time_s": 120.0,
            "maximum_tool_calls": 20,
            "maximum_retries": 2,
            "maximum_model_calls": 4,
        },
        **overrides,
    }


def plan_for(entry: Any, strategies=("fixed", "random", "aco")) -> dict[str, Any]:
    return {
        "model": {"provider": entry.provider, "model": entry.model_id},
        "expected_model_hash": entry.model_hash,
        "expected_prompt_version": PROMPT_TEMPLATE_VERSION,
        "budget": {"max_candidate_evaluations": 4},
        "seeds": [0],
        "strategies": list(strategies),
    }


def rows_of(role: str, splits: dict[str, Any]) -> list[str]:
    return [s for x in splits["splits"]["splits"] if x["role"] == role for s in x["row_ids"]]


def order_ids(row_ids: list[str]) -> set[str]:
    """Order ids of fixture rows (T-0007 -> ORD-1007): how a prompt names its row."""
    return {f"ORD-1{r[-3:]}" for r in row_ids}


def prompts_about(provider: FakeProvider, orders: set[str], since: int = 0) -> list[str]:
    return [p for p in provider.prompts()[since:] if any(o in p for o in orders)]


@pytest.fixture
def provider():
    p = FakeProvider().start()
    yield p
    p.stop()


@pytest.fixture
def world(tmp_path, provider):
    return World(tmp_path, provider)


# == 3. the happy path ===========================================================================
def test_a_fresh_user_takes_a_real_dataset_through_the_whole_lifecycle(world):
    w, p = world, world.provider
    # 1. workspace + auth context: an owner's one-time key; nothing works without it
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    assert code(w.call(None, "GET", "projects")) == (401, "unauthenticated")

    # 2. upload: parsed, screened, encrypted; the server derives every fact about the data
    data = w.dataset(key)
    up = data["upload"]
    assert up["row_count"] == 18 and up["format"] == "csv"
    assert [c["name"] for c in up["columns"]] == ["ticket_id", "question", "order_record", "answer"]
    assert w.ok(key, "GET", f"uploads/{up['upload_id']}")["upload_id"] == up["upload_id"]

    # 3. registered contract: the dataset version the TaskContract pins
    spec = data["version"]["spec"]
    assert spec["content_hash"] == up["content_hash"] and spec["row_count"] == 18
    assert w.ok(key, "GET", f"datasets/{DATASET_ID}")["versions"][0]["spec"] == spec
    heldout = rows_of("test", data["splits"])
    validation = rows_of("validation", data["splits"])
    optimization = rows_of("optimization", data["splits"])
    assert heldout and validation and optimization
    assert len(heldout + validation + optimization) == 18

    # 4.-5. one optimization job (fixed / random / ACO), run by the production worker loop
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    assert job["state"] == "PENDING" and p.requests == []  # creating runs nothing
    group = w.r.start_workers()
    try:
        deadline = time.monotonic() + 120
        while (state := w.ok(key, "GET", f"experiments/{job['job_id']}")["state"]) not in (
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        ):
            assert time.monotonic() < deadline, state
            time.sleep(0.1)
    finally:
        group.stop()
    assert state == "COMPLETED"
    assert p.requests and all(a == f"Bearer {PROVIDER_KEY}" for a in p.authorizations)
    # the search never opened a held-out row
    assert prompts_about(p, order_ids(heldout)) == []
    search_requests = len(p.requests)

    # 6. canonical provenance
    verify = w.ok(key, "GET", f"experiments/{job['job_id']}/verify")
    assert verify["verified"] is True
    art = w.ok(key, "GET", f"experiments/{job['job_id']}/artifact")["artifact"]
    assert art["synthetic"] is False and art["test_runs"] == 0
    assert art["model_hashes"] == [w.entry.model_hash]
    prov = w.ok(key, "GET", f"experiments/{job['job_id']}/provenance")["provenance"]
    assert prov["dataset"]["content_hash"] == up["content_hash"]
    assert prov["synthetic"] is False
    assert {s["strategy"] for s in art["memory"]["runs"]} == {"fixed", "random", "aco"}
    assert all(r["mode"] == "cold" for r in art["memory"]["runs"])

    # 7. memory learned from the verified search - search + validation evidence only
    api = w.api(ws)
    learned = api.jobs.priors.store.learned(job["job_id"])
    assert learned is not None and learned.outcome == "learned", learned
    active = api.jobs.priors.store.active()
    assert {row.record.kind.value for row in active} == {"dataset_champion", "task_family"}

    # 8. champion promotion opens the held-out rows exactly now, once each
    promo = w.ok(key, "POST", f"experiments/{job['job_id']}/promote", status=201)
    assert promo["decision"] == "PROMOTED", promo
    held_prompts = prompts_about(p, order_ids(heldout), search_requests)
    assert held_prompts and len(p.requests) > search_requests
    # re-derived from stored evidence alone (fails closed on any mismatch)
    assert w.ok(key, "GET", f"promotions/{promo['promotion_id']}/verify")["decision"] == "PROMOTED"
    champion = promo["record"]["champion"]["champion_id"]
    assert w.ok(key, "GET", f"champions/{LINEAGE}")["champion_id"] == champion
    # held-out evidence never taught memory: the priors are unchanged by promotion
    assert {r.prior_id for r in api.jobs.priors.store.active()} == {r.prior_id for r in active}

    # 9. immutable workflow version, pinned to the exact canonical identities
    wv = w.ok(key, "POST", "workflows", {"champion_id": champion}, 201)
    v1 = wv["version_id"]
    assert wv["state"] == "CREATED"
    doc = wv["document"]
    assert doc["versions"]["model_hash"] == w.entry.model_hash
    assert doc["provenance"]["artifact_id"] == verify["artifact_id"]
    again = w.call(key, "POST", "workflows", {"champion_id": champion})
    assert again[0] == 200 and again[1]["version_id"] == v1  # publishing is idempotent
    assert w.ok(key, "GET", f"deployments/{LINEAGE}")["production_version_id"] is None

    # 10.-11. staging is not production; promotion is explicit and revision-fenced
    staged = w.ok(key, "POST", f"workflows/{v1}/stage", {"expected_revision": 0})
    assert staged["staging_version_id"] == v1 and staged["production_version_id"] is None
    assert code(w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})) == (
        409,
        "no_production_version",
    )
    prod = w.ok(key, "POST", f"workflows/{v1}/promote", {"expected_revision": 1})
    assert prod["production_version_id"] == v1 and prod["revision"] == 2

    # 12.-13. invoke by production lineage; provenance and usage of the exact version
    inf = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert inf["status"] == "SUCCEEDED" and inf["output"] == {"answer": "Lincoln"}
    assert inf["workflow_version"] == v1 and inf["addressed_by"] == "production"
    record = w.ok(key, "GET", f"inferences/{inf['inference_id']}")
    assert record["versions"] == doc["versions"] == inf["versions"]
    assert record["champion_id"] == champion and record["deployment_revision"] == 2
    assert record["provenance"]["artifact_id"] == verify["artifact_id"]
    assert record["output"] is None and record["output_sha256"]  # stored by hash only
    usage = record["usage"]
    assert usage["model_calls"] > 0 and usage["cost_authoritative"] is True and usage["cost"] > 0

    # a second release: a warm-started search over the same dataset/runtime (memory reuse)
    warm = w.ok(key, "POST", "experiments", w.experiment_body(data, ("aco_warm",)), 201)
    w.run(ws, warm["job_id"])
    assert w.ok(key, "GET", f"experiments/{warm['job_id']}")["state"] == "COMPLETED"
    warm_art = w.ok(key, "GET", f"experiments/{warm['job_id']}/artifact")["artifact"]
    (start,) = warm_art["memory"]["runs"]
    assert start["mode"] == "warm_dataset_champion" and start["prior_id"] in {
        r.prior_id for r in active
    }, start
    assert warm_art["memory"]["prior_loading_model_calls"] == 0
    promo2 = w.ok(key, "POST", f"experiments/{warm['job_id']}/promote", status=201)
    assert promo2["decision"] == "PROMOTED", promo2
    v2 = w.ok(
        key, "POST", "workflows", {"champion_id": promo2["record"]["champion"]["champion_id"]}, 201
    )["version_id"]
    assert v2 != v1
    assert w.deploy(key, v2) == 4
    assert (
        w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})[
            "workflow_version"
        ]
        == v2
    )

    # 14.-15. rollback restores the exact previous immutable version, which then serves
    back = w.ok(key, "POST", f"deployments/{LINEAGE}/rollback", {"expected_revision": 4})
    assert back["production_version_id"] == v1 and back["revision"] == 5
    inf3 = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert inf3["workflow_version"] == v1 and inf3["output"] == {"answer": "Lincoln"}
    assert inf3["versions"] == doc["versions"]
    assert w.ok(key, "GET", f"workflows/{v1}")["document"] == doc  # unchanged, byte for byte
    history = w.ok(key, "GET", f"deployments/{LINEAGE}/history")
    assert history and "rollback" in str(history).lower()

    # 16. export returns the stored canonical rows
    exported = w.call(key, "GET", f"datasets/{DATASET_ID}/versions/1/export")
    assert exported[0] == 200
    rows = list(csv.DictReader(io.StringIO(_text(exported[1]))))
    assert len(rows) == 18 and rows[0]["answer"] == "Denver"

    # 17. deletion (owner, admin scope)
    w.ok(key, "DELETE", f"datasets/{DATASET_ID}")

    # 18. dependent state: content gone, new work refused, memory invalidated, history kept
    source = w.ok(key, "GET", f"datasets/{DATASET_ID}/versions/1/source")
    assert source["status"] == "deleted"
    gone = (410, "dataset_source_deleted")
    assert code(w.call(key, "GET", f"datasets/{DATASET_ID}/versions/1/export")) == gone
    assert code(w.call(key, "POST", "experiments", w.experiment_body(data))) == gone
    assert code(w.call(key, "POST", f"datasets/{DATASET_ID}/versions/1/splits", SPLIT_PLAN)) == gone
    assert code(w.call(key, "GET", f"uploads/{up['upload_id']}")) == (410, "upload_deleted")
    assert api.jobs.priors.store.active() == []
    retired = api.jobs.priors.store.history()
    assert retired and all(
        r.invalidated and r.invalidated_reason.value == "dataset_deleted" for r in retired
    )
    # immutable provenance is kept (documented): artifacts verify, the deployed version serves
    assert w.ok(key, "GET", f"experiments/{job['job_id']}/verify")["verified"] is True
    assert w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})["status"] == (
        "SUCCEEDED"
    )

    # the audit chain recorded every data action and still verifies
    audit = w.ok(key, "GET", "audit")
    actions = {e["action"] for e in audit["events"]}
    assert {"dataset.upload", "dataset.register", "dataset.export", "dataset.delete"} <= actions
    log = SQLiteAuditLog(w.data / "workspaces" / ws / "audit.sqlite3", workspace_id=ws)
    log.verify_chain()

    # the provider credential never reached a stored file or a response
    secret = PROVIDER_KEY.encode()
    assert not [
        f for f in files_under(w.root) if f.name != "registry.json" and secret in f.read_bytes()
    ]
    assert PROVIDER_KEY not in repr(w.responses)


def _text(body: Any) -> str:
    if isinstance(body, bytes | bytearray):
        return body.decode()
    if isinstance(body, str):
        return body
    raise AssertionError(f"unexpected export body {type(body)}")


# == 4. crash / resume through the real runtime =================================================
class Crash(BaseException):
    """Simulated process death: nothing in the product catches it."""


def die_after(monkeypatch, cls: type, method: str, n: int, when=lambda *a, **k: True) -> list[int]:
    """Kill the "process" right AFTER the ``n``-th committed ``cls.method`` call ``when``
    selects."""
    count = [0]
    real = getattr(cls, method)

    def wrapped(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        if when(*args, **kwargs):
            count[0] += 1
            if count[0] == n:
                raise Crash
        return out

    monkeypatch.setattr(cls, method, wrapped)
    return count


def later() -> float:
    """The restarted process starts after every lease of the dead one expired."""
    return time.time() + 3600


def reference_run(tmp_path, provider) -> dict[str, Any]:
    """The same experiment in a separate world, uninterrupted."""
    ref = World(tmp_path / "reference", provider)
    ws, key = ref.workspace("Reference", "ref@example.test")
    data = ref.dataset(key)
    before = len(provider.requests)
    job = ref.ok(key, "POST", "experiments", ref.experiment_body(data), 201)
    ref.run(ws, job["job_id"])
    return {
        "requests": len(provider.requests) - before,
        "artifact": ref.ok(key, "GET", f"experiments/{job['job_id']}/artifact")["artifact"],
        "job": ref.ok(key, "GET", f"experiments/{job['job_id']}"),
    }


@pytest.mark.parametrize(
    "kind, n",
    [
        ("reserved", 2),  # a candidate reserved, nothing run for it yet
        ("attempt", 7),  # mid-candidate: some of its rows completed
        ("settled", 3),  # right after a candidate checkpoint
    ],
)
def test_a_crash_resumes_exactly_without_duplicate_calls_or_spend(
    tmp_path, provider, monkeypatch, kind, n
):
    from store.jobs import SQLiteJobStore

    ref = reference_run(tmp_path, provider)
    w = World(tmp_path / "crash", provider)
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    start = len(provider.requests)
    with monkeypatch.context() as m:
        if kind == "attempt":
            die_after(m, SQLiteJobStore, "complete_attempt", n)
        else:
            die_after(
                m, SQLiteJobStore, "append_checkpoint", n, lambda claim, k, *a, **kw: k == kind
            )
        with pytest.raises(Crash):
            w.run(ws, job["job_id"])
    before_restart = len(provider.requests) - start
    assert 0 < before_restart < ref["requests"]
    assert w.ok(key, "GET", f"experiments/{job['job_id']}")["state"] == "RUNNING"

    w.restart()  # a new process over the same data directory
    w.run(ws, job["job_id"], clock=later)
    after = w.ok(key, "GET", f"experiments/{job['job_id']}")
    assert after["state"] == "COMPLETED", after
    # same job identity; nothing re-sent (no restart from scratch, no duplicate evaluation)
    for k in ("job_id", "experiment_id", "definition_hash", "problem_id"):
        assert after[k] == job[k]
    assert len(provider.requests) - start == ref["requests"]
    assert w.ok(key, "GET", f"experiments/{job['job_id']}/verify")["verified"] is True
    # the optimizer saw exactly the uninterrupted feedback (proposals, scores, pheromones,
    # champions, ledger): the scientific content equals the clean run's, except the upload id
    # (each world uploaded the fixture to its own project)
    art = w.ok(key, "GET", f"experiments/{job['job_id']}/artifact")["artifact"]
    assert differences(scientific(art), scientific(ref["artifact"])) == [
        ("provenance", "upload_id")
    ]
    # ... and was charged exactly once: same calls, tokens and cost as the clean run
    for unit, clean in zip(after["units"], ref["job"]["units"], strict=True):
        assert unit["candidate_evaluations"] == clean["candidate_evaluations"]
        assert unit["attempts"] == clean["attempts"]
        for k in clean["usage"]:
            if "time" not in k:  # measured wall time is never reproduced
                assert unit["usage"][k] == clean["usage"][k], (unit["strategy"], k)


def test_a_crash_with_a_call_in_flight_is_never_resent_or_hidden(world, monkeypatch):
    """The provider may have charged for an in-flight call: the unit is reported ambiguous and
    is never silently resumed or re-sent. The user cancels it and starts a new job."""
    from runtime.backends.openai_compatible import OpenAICompatibleClient

    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    one_unit = w.experiment_body(data, ("fixed",))
    job = w.ok(key, "POST", "experiments", one_unit, 201)
    real = OpenAICompatibleClient._post_once
    calls = [0]

    def dies_mid_call(self, payload):
        calls[0] += 1
        out = real(self, payload)  # the provider received (and may bill) this call
        if calls[0] == 5:
            raise Crash
        return out

    with monkeypatch.context() as m:
        m.setattr(OpenAICompatibleClient, "_post_once", dies_mid_call)
        with pytest.raises((Crash, BaseExceptionGroup)):  # surfaces from the run's thread
            w.run(ws, job["job_id"])
    sent = len(w.provider.requests)

    w.restart()
    w.run(ws, job["job_id"], clock=later)
    view = w.ok(key, "GET", f"experiments/{job['job_id']}")
    assert view["state"] == "INTERRUPTED" and view["reason"] == "ambiguous_attempt", view
    assert code(w.call(key, "POST", f"experiments/{job['job_id']}/resume")) == (
        409,
        "ambiguous_attempt",
    )
    assert len(w.provider.requests) == sent  # nothing was re-sent
    assert w.ok(key, "POST", f"experiments/{job['job_id']}/cancel")["state"] == "CANCELLED"
    again = w.ok(key, "POST", "experiments", one_unit, 201)
    assert again["job_id"] != job["job_id"]
    w.run(ws, again["job_id"])
    assert w.ok(key, "GET", f"experiments/{again['job_id']}")["state"] == "COMPLETED"


@pytest.mark.parametrize("point", ["before_finalize", "before_learning"])
def test_a_crash_around_artifact_finalization_recovers_once(world, monkeypatch, point):
    from memory.production import PriorService
    from store.jobs import SQLiteJobStore

    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)

    def dies(self, *a, **k):
        raise Crash

    with monkeypatch.context() as m:
        if point == "before_finalize":  # every unit done, the artifact never stored
            m.setattr(SQLiteJobStore, "put_artifact", dies)
        else:  # the artifact committed; the process died before memory learned from it
            m.setattr(PriorService, "learn", dies)
        with pytest.raises(Crash):
            w.run(ws, job["job_id"])
    sent = len(w.provider.requests)
    finalized = w.call(key, "GET", f"experiments/{job['job_id']}/verify")
    assert (finalized[0] == 200) is (point == "before_learning"), finalized

    w.restart()
    w.run(ws, job["job_id"], clock=later)  # recover(): assemble / learn what is pending
    assert len(w.provider.requests) == sent  # recovery calls no model
    verify = w.ok(key, "GET", f"experiments/{job['job_id']}/verify")
    if point == "before_learning":  # the immutable artifact is unchanged by recovery
        assert verify == finalized[1]
    learned = w.api(ws).jobs.priors.store.learned(job["job_id"])
    assert learned is not None and learned.outcome == "learned"
    w.restart()
    w.run(ws, job["job_id"], clock=later)
    assert w.ok(key, "GET", f"experiments/{job['job_id']}/verify") == verify


def test_a_crash_during_promotion_resumes_without_resending_held_out_calls(world, monkeypatch):
    from store.champions import SQLiteChampionStore

    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    w.run(ws, job["job_id"])
    heldout = order_ids(rows_of("test", data["splits"]))
    start = len(w.provider.requests)
    with monkeypatch.context() as m:
        die_after(m, SQLiteChampionStore, "complete_attempt", 1)
        with pytest.raises(Crash):
            w.call(key, "POST", f"experiments/{job['job_id']}/promote")
    assert prompts_about(w.provider, heldout, start)

    w.restart()
    promotions = w.api(ws).promotions
    promotions.clock = later
    assert [reason.value for _, reason in promotions.recover()] == ["process_lost"]
    promo = w.ok(key, "POST", f"experiments/{job['job_id']}/promote")  # continues: 200
    assert promo["decision"] == "PROMOTED", promo
    sent = prompts_about(w.provider, heldout, start)
    # each completed held-out row was reused from the store, never sent twice
    assert len(sent) == len(set(sent))
    assert w.ok(key, "GET", f"promotions/{promo['promotion_id']}/verify")["decision"] == "PROMOTED"


def test_a_staged_deployment_survives_a_restart_and_stays_fenced(world):
    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    w.run(ws, job["job_id"])
    promo = w.ok(key, "POST", f"experiments/{job['job_id']}/promote", status=201)
    champion = promo["record"]["champion"]["champion_id"]
    v1 = w.ok(key, "POST", "workflows", {"champion_id": champion}, 201)["version_id"]
    staged = w.ok(key, "POST", f"workflows/{v1}/stage", {"expected_revision": 0})
    doc = w.ok(key, "GET", f"workflows/{v1}")["document"]

    w.restart()
    assert w.ok(key, "GET", f"deployments/{LINEAGE}") == staged
    assert w.ok(key, "GET", f"workflows/{v1}")["document"] == doc
    # a write against the pre-stage revision 0 is stale and changes nothing
    assert code(w.call(key, "POST", f"workflows/{v1}/promote", {"expected_revision": 0})) == (
        409,
        "stale_deployment",
    )
    assert w.ok(key, "GET", f"deployments/{LINEAGE}") == staged
    prod = w.ok(key, "POST", f"workflows/{v1}/promote", {"expected_revision": 1})
    assert prod["production_version_id"] == v1
    out = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})["output"]
    assert out == {"answer": "Lincoln"}


# -- shared: one released (published) champion --------------------------------------------------
def release(w: World, ws: str, key: str, data: dict[str, Any], strategies=("fixed",)) -> dict:
    job = w.ok(key, "POST", "experiments", w.experiment_body(data, strategies), 201)
    w.run(ws, job["job_id"])
    promo = w.ok(key, "POST", f"experiments/{job['job_id']}/promote", status=201)
    assert promo["decision"] == "PROMOTED", promo
    champion = promo["record"]["champion"]["champion_id"]
    wv = w.ok(key, "POST", "workflows", {"champion_id": champion}, 201)
    return {"job": job, "promotion": promo, "champion": champion, "version": wv["version_id"]}


def sqlite(path: Path, statement: str, params: tuple = ()) -> None:
    import sqlite3

    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute(statement, params)
    finally:
        conn.close()


# == 5. tenant isolation =========================================================================
def test_workspace_b_can_never_reach_workspace_a(world):
    w = world
    ws_a, key_a = w.workspace("Dogfood A", "owner-a@example.test")
    ws_b, key_b = w.workspace("Dogfood B", "owner-b@example.test")
    data = w.dataset(key_a)
    rel = release(w, ws_a, key_a, data)
    w.deploy(key_a, rel["version"])
    a_before = w.ok(key_a, "GET", f"deployments/{LINEAGE}")
    job_a, v_a = rel["job"]["job_id"], rel["version"]
    upload_a = data["upload"]["upload_id"]

    def like_absent(method, path, real, fake, body=None):
        """B naming A's id answers EXACTLY like B naming an id that never existed."""
        got = w.call(key_b, method, path.format(real), body)
        never = w.call(key_b, method, path.format(fake), body)
        assert got[0] == never[0] and got[0] in (404, 409, 422), (path, got)
        assert repr(got[1]).replace(real, "ID") == repr(never[1]).replace(fake, "ID")

    fake_job, fake_v = "j-" + "0" * 24, "wv-" + "0" * 24
    for method, path, real, fake, body in (
        ("GET", "uploads/{}", upload_a, "u-" + "0" * 32, None),
        ("GET", "datasets/{}", DATASET_ID, "never-registered", None),
        ("GET", "datasets/{}/versions/1/export", DATASET_ID, "never-registered", None),
        ("DELETE", "datasets/{}", DATASET_ID, "never-registered", None),
        ("DELETE", "uploads/{}", upload_a, "u-" + "0" * 32, None),
        ("GET", "experiments/{}", job_a, fake_job, None),
        ("GET", "experiments/{}/artifact", job_a, fake_job, None),
        ("GET", "experiments/{}/provenance", job_a, fake_job, None),
        ("POST", "experiments/{}/promote", job_a, fake_job, None),
        ("GET", "workflows/{}", v_a, fake_v, None),
        ("POST", "workflows/{}/stage", v_a, fake_v, {"expected_revision": 0}),
        ("POST", "workflows/{}/invoke", v_a, fake_v, {"inputs": NEW_ORDER}),
        ("GET", "deployments/{}", LINEAGE, "never.deployed", None),
        ("POST", "deployments/{}/invoke", LINEAGE, "never.deployed", {"inputs": NEW_ORDER}),
        ("POST", "deployments/{}/rollback", LINEAGE, "never.deployed", {"expected_revision": 2}),
        ("GET", "champions/{}", LINEAGE, "never.deployed", None),
    ):
        like_absent(method, path, real, fake, body)
    # publishing A's champion id from B
    assert code(w.call(key_b, "POST", "workflows", {"champion_id": rel["champion"]})) == code(
        w.call(key_b, "POST", "workflows", {"champion_id": "c-" + "0" * 24})
    )
    # B uploads the IDENTICAL dataset (same content-addressed identity as A's) ...
    data_b = w.dataset(key_b)
    assert data_b["version"]["identity_hash"] == data["version"]["identity_hash"]
    # ... yet A's learned prior is invisible to B: B's warm start is cold, from B's own memory
    warm = w.ok(key_b, "POST", "experiments", w.experiment_body(data_b, ("aco_warm",)), 201)
    w.run(ws_b, warm["job_id"])
    (start,) = w.ok(key_b, "GET", f"experiments/{warm['job_id']}/artifact")["artifact"]["memory"][
        "runs"
    ]
    assert start["mode"] == "cold" and start["reason"] == "cold_no_prior", start
    assert w.api(ws_a).jobs.priors.store.active()  # A's prior exists, in A's partition only
    # A is untouched by everything B tried
    assert w.ok(key_a, "GET", f"deployments/{LINEAGE}") == a_before
    assert w.ok(key_a, "GET", f"datasets/{DATASET_ID}/versions/1/source")["status"] == "available"


def test_read_only_and_member_keys_cannot_change_or_delete(world):
    w = world
    ws, owner = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(owner)
    rel = release(w, ws, owner, data)
    w.deploy(owner, rel["version"])
    member = w.ok(owner, "POST", "workspace/members", {"email": "analyst@example.test"}, 201)
    keys = {
        scopes: w.ok(
            owner,
            "POST",
            "workspace/keys",
            {"name": "-".join(scopes), "scopes": list(scopes), "user_id": member["user_id"]},
            201,
        )["secret"]
        for scopes in (("read",), ("read", "write", "invoke"))
    }
    viewer, writer = keys[("read",)], keys[("read", "write", "invoke")]
    job, version = rel["job"]["job_id"], rel["version"]
    # a read-only (viewer) key reads everything of its workspace ...
    for path in (f"experiments/{job}", f"experiments/{job}/artifact", f"deployments/{LINEAGE}"):
        assert w.call(viewer, "GET", path)[0] == 200, path
    # ... and changes nothing: no upload, run, promotion, publication, deployment or invocation
    denied = (403, "insufficient_scope")
    project = data["project"]["project_id"]
    for method, path, body in (
        ("POST", f"projects/{project}/uploads?format=csv", FIXTURE.read_bytes()),
        ("POST", "experiments", w.experiment_body(data)),
        ("POST", f"experiments/{job}/promote", None),
        ("POST", "workflows", {"champion_id": rel["champion"]}),
        ("POST", f"workflows/{version}/stage", {"expected_revision": 2}),
        ("POST", f"deployments/{LINEAGE}/rollback", {"expected_revision": 2}),
        ("POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER}),
        ("DELETE", f"datasets/{DATASET_ID}", None),
        ("GET", "audit", None),
    ):
        assert code(w.call(viewer, method, path, body)) == denied, path
    # a MEMBER's write key can invoke, but deletion and the audit trail stay OWNER-only (admin)
    assert w.call(writer, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})[0] == 200
    for method, path in (("DELETE", f"datasets/{DATASET_ID}"), ("GET", "audit")):
        assert code(w.call(writer, method, path)) == denied, path
    assert w.ok(owner, "GET", f"datasets/{DATASET_ID}/versions/1/source")["status"] == "available"
    # documented (#33): export needs only `read`, so a viewer key can download the rows
    assert w.call(viewer, "GET", f"datasets/{DATASET_ID}/versions/1/export")[0] == 200


# == 6. provider failure modes ===================================================================
def test_an_unavailable_model_provider_fails_closed_everywhere(world):
    w, p = world, world.provider
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    rel = release(w, ws, key, data)
    w.deploy(key, rel["version"])

    p.status = 503  # the provider is down
    job = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    w.run(ws, job["job_id"])
    view = w.ok(key, "GET", f"experiments/{job['job_id']}")
    assert view["state"] == "FAILED" and view["reason"] == "model_unavailable", view
    assert code(w.call(key, "GET", f"experiments/{job['job_id']}/artifact"))[0] in (404, 409)
    assert code(w.call(key, "POST", f"experiments/{job['job_id']}/promote")) == (
        409,
        "experiment_not_promotable",
    )
    s, body = w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert (s, body["error"]["code"]) == (502, "inference_failed")
    stored = w.ok(key, "GET", f"inferences/{body['error']['details']['inference_id']}")
    assert stored["status"] == "FAILED" and stored["output_sha256"] is None
    # one endpoint, never another: no fallback model was asked
    assert {r["model"] for r in p.requests} == {w.entry.model_id}
    assert PROVIDER_KEY not in repr(w.responses)


def test_a_changed_or_missing_model_identity_fails_closed_without_requests(world):
    w, p = world, world.provider
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    rel = release(w, ws, key, data)
    w.deploy(key, rel["version"])
    candidate = release(w, ws, key, data, ("random",))  # published, not yet staged
    pending = w.ok(key, "POST", "experiments", w.experiment_body(data), 201)
    sent = len(p.requests)

    # the operator re-pins the same model name to another revision (another model_hash)
    write_registry(w.registry, entry_for(p.endpoint, revision="2026-11-01"))
    s, body = w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert (s, body["error"]["code"]) == (503, "model_binding_failed")
    assert body["error"]["details"]["reason"] == "registry_entry_missing"
    w.restart()
    w.run(ws, pending["job_id"])
    view = w.ok(key, "GET", f"experiments/{pending['job_id']}")
    assert (view["state"], view["reason"]) == ("INTERRUPTED", "runtime_unavailable"), view
    # a disabled entry, or a published version whose model left the registry: nothing stages
    write_registry(w.registry, entry_for(p.endpoint, enabled=False))
    s, body = w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert body["error"]["details"]["reason"] == "registry_entry_disabled"
    rev = w.ok(key, "GET", f"deployments/{LINEAGE}")["revision"]
    stage = w.call(
        key, "POST", f"workflows/{candidate['version']}/stage", {"expected_revision": rev}
    )
    assert code(stage) == (503, "model_binding_failed")
    assert w.ok(key, "GET", f"deployments/{LINEAGE}")["revision"] == rev
    assert len(p.requests) == sent  # every refusal happened before a single request

    # the pinned entry is back: the same job resumes and the same version serves again
    write_registry(w.registry, w.entry)
    w.restart()
    w.ok(key, "POST", f"experiments/{pending['job_id']}/resume")
    w.run(ws, pending["job_id"])
    assert w.ok(key, "GET", f"experiments/{pending['job_id']}")["state"] == "COMPLETED"
    inf = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert inf["versions"]["model_hash"] == w.entry.model_hash
    assert PROVIDER_KEY not in repr(w.responses)


def test_the_product_never_runs_jev_on_an_uploaded_dataset(world):
    """The product API configures no JEV (``registry_runtime`` binds none) and an uploaded
    CSV cannot be gathered with JEV: the contract is refused before anything runs. JEV identity
    changes are runtime-level (tests/test_runtime_jev.py, test_warm_start_evidence_provider.py)."""
    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    body = w.experiment_body(data, workflow={"allowed_sources": ["jev"]})
    s, err = w.call(key, "POST", "experiments", body)
    assert (s, err["error"]["code"]) == (400, "invalid_request")
    assert "jev" in err["error"]["message"]
    job = w.ok(key, "POST", "experiments", w.experiment_body(data, ("fixed",)), 201)
    w.run(ws, job["job_id"])
    versions = w.ok(key, "GET", f"experiments/{job['job_id']}/provenance")["provenance"]["versions"]
    assert "evidence_provider" not in versions  # bound with no live provider
    assert w.provider.requests


def test_websearch_without_a_configured_provider_is_never_substituted(world):
    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    body = w.experiment_body(data, workflow={"allowed_sources": ["websearch"]})
    job = w.ok(key, "POST", "experiments", body, 201)
    w.run(ws, job["job_id"])
    art = w.ok(key, "GET", f"experiments/{job['job_id']}/artifact")["artifact"]
    stages = [s for run in art["runs"] for c in run["candidates"] for s in c["genome"]["stages"]]
    # websearch is unavailable, so no candidate gathers at all: nothing (fetch, api, jev) is
    # substituted, and no run reports an evidence source
    assert not [s for s in stages if s["kind"] == "GATHER"]
    assert all(v.get("evidence_source") is None for v in art["run_versions"])
    # without the order record the direct answers are wrong: the search reports it honestly
    assert all(
        u["champion_validation_score"] == 0.0
        for u in w.ok(key, "GET", f"experiments/{job['job_id']}")["units"]
    )


def websearch_world(tmp_path, provider, transport, endpoint=None) -> World:
    from tests.dogfood_support import WEBSEARCH_A, websearch

    w = World.__new__(World)
    w.root, w.provider, w.key = tmp_path, provider, PROVIDER_KEY
    tmp_path.mkdir(parents=True, exist_ok=True)
    w.registry = tmp_path / "registry.json"
    w.entry = entry_for(provider.endpoint)
    write_registry(w.registry, w.entry)
    w.data, w.responses = tmp_path / "data", []
    w.source = websearch(transport, endpoint or WEBSEARCH_A)
    w.r = product(w.data, w.registry, websearch=w.source)
    w.restart = lambda: None  # noqa: E731 - replaced per test when the provider changes
    return w


def test_a_websearch_champion_is_pinned_to_its_provider_and_never_teaches_memory(
    tmp_path, provider
):
    from tests.dogfood_support import WEBSEARCH_B, WEBSEARCH_KEY, FakeBrave, websearch

    brave = FakeBrave(extra={NEW_ORDER["question"]: NEW_ORDER["order_record"]})
    w = websearch_world(tmp_path, provider, brave)
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    contract = {"workflow": {"allowed_sources": ["websearch"]}}
    job = w.ok(
        key, "POST", "experiments", w.experiment_body(data, ("fixed", "aco"), **contract), 201
    )
    w.run(ws, job["job_id"])
    assert w.ok(key, "GET", f"experiments/{job['job_id']}")["state"] == "COMPLETED"
    assert brave.requests
    # outbound minimization: the query is the row's input only (never its record or target)
    for url, headers in brave.requests:
        assert "ships" not in url and "warehouse" not in url and WEBSEARCH_KEY not in url
        assert headers["X-Subscription-Token"] == WEBSEARCH_KEY
    # memory never learns from live evidence
    learned = w.api(ws).jobs.priors.store.learned(job["job_id"])
    assert learned is not None and learned.outcome == "refused", learned
    assert "websearch" in (learned.detail or "")
    assert w.api(ws).jobs.priors.store.active() == []
    promo = w.ok(key, "POST", f"experiments/{job['job_id']}/promote", status=201)
    assert promo["decision"] == "PROMOTED", promo
    wv = w.ok(
        key, "POST", "workflows", {"champion_id": promo["record"]["champion"]["champion_id"]}, 201
    )
    pinned = wv["document"]["versions"].get("evidence_source")
    assert pinned == w.source.identity
    w.deploy(key, wv["version_id"])
    n = len(brave.requests)
    inf = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert inf["status"] == "SUCCEEDED" and len(brave.requests) == n + 1  # live: one search

    # the operator swaps the websearch provider: production refuses before any request
    w.r = product(w.data, w.registry, websearch=websearch(brave, WEBSEARCH_B))
    sent, model_sent = len(brave.requests), len(provider.requests)
    s, body = w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert (s, body["error"]["code"]) == (503, "model_binding_failed")
    assert body["error"]["details"]["reason"] == "websearch_identity_mismatch"
    # ... and with no provider at all
    w.r = product(w.data, w.registry)
    s, body = w.call(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER})
    assert body["error"]["details"]["reason"] == "websearch_unavailable"
    assert (len(brave.requests), len(provider.requests)) == (sent, model_sent)
    assert WEBSEARCH_KEY not in repr(w.responses) and PROVIDER_KEY not in repr(w.responses)
    secrets = (WEBSEARCH_KEY.encode(), PROVIDER_KEY.encode())
    leaked = [f for f in files_under(w.data) if any(s in f.read_bytes() for s in secrets)]
    assert leaked == []


def test_a_websearch_job_never_resumes_under_another_or_a_drifted_provider(tmp_path, provider):
    from runtime.backends.brave_search import BraveSearchConfig
    from tests.dogfood_support import WEBSEARCH_B, FakeBrave, websearch

    brave = FakeBrave()
    w = websearch_world(tmp_path, provider, brave)
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    contract = {"workflow": {"allowed_sources": ["websearch"]}}
    body = w.experiment_body(data, ("aco",), **contract)  # ACO reaches the websearch GATHER

    # live drift DURING execution: the provider object starts answering as another identity
    job = w.ok(key, "POST", "experiments", body, 201)
    real_search = type(w.source).search
    drifted = [False]

    def drifting(self, request):
        response = real_search(self, request)
        if len(brave.requests) == 3 and not drifted[0]:  # reconfigured after the 3rd search
            drifted[0] = True
            self.config = BraveSearchConfig(api_key=self.config.api_key, endpoint=WEBSEARCH_B)
        return response

    type(w.source).search = drifting
    try:
        w.run(ws, job["job_id"])
    finally:
        type(w.source).search = real_search
    assert drifted[0] and len(brave.requests) == 3  # nothing was sent once it drifted
    # the next search is refused before its run is recorded: a refusal measures nothing, so the
    # job is INTERRUPTED, never completed with refused rows scored (formerly residual risk R3)
    view = w.ok(key, "GET", f"experiments/{job['job_id']}")
    assert (view["state"], view["reason"]) == ("INTERRUPTED", "runtime_unavailable"), view
    (unit,) = view["units"]
    assert unit["usage"]["tool_calls"] == 3
    assert "search-b.example.test" not in repr(view)

    # a NEW process bound to another provider never runs a job bound to the first
    w.r = product(w.data, w.registry, websearch=websearch(brave))  # a fresh provider-A process
    job2 = w.ok(key, "POST", "experiments", body, 201)
    w.r = product(w.data, w.registry, websearch=websearch(brave, WEBSEARCH_B))
    sent = len(brave.requests)
    w.run(ws, job2["job_id"])
    view = w.ok(key, "GET", f"experiments/{job2['job_id']}")
    assert (view["state"], view["reason"]) == ("INTERRUPTED", "runtime_unavailable"), view
    assert len(brave.requests) == sent


def test_a_websearch_drift_during_a_job_interrupts_it_and_resume_matches_an_undisturbed_job(
    tmp_path, provider
):
    """A drifted search is refused, never sent: it measures nothing, so it must never be scored
    (residual risk R3). The job is interrupted before the refused run is recorded, and once the
    provider is back it resumes to exactly the result of a job the drift never touched: no
    refusal reaches the optimizer or the champion, and no search or model call is repeated."""
    from runtime.backends.brave_search import BraveSearchConfig
    from tests.dogfood_support import WEBSEARCH_B, FakeBrave

    contract = {"workflow": {"allowed_sources": ["fetch", "websearch"]}}

    def result(w, key, job_id):
        art = w.ok(key, "GET", f"experiments/{job_id}/artifact")["artifact"]
        return [
            (
                r["strategy"],
                r["champion"]["genome_hash"],
                {k: r["usage"][k] for k in ("model_calls", "tool_calls", "workflow_runs")},
                [
                    (
                        c["genome_hash"],
                        c["validation"]["score_mean"],
                        [x["failure"] for x in c["runs"]],
                    )
                    for c in r["candidates"]
                ],
            )
            for r in art["runs"]
        ]

    # the reference: the same job, never disturbed
    ref_brave = FakeBrave()
    ref = websearch_world(tmp_path / "ref", provider, ref_brave)
    ref_ws, ref_key = ref.workspace("Dogfood A", "owner-a@example.test")
    body = ref.experiment_body(ref.dataset(ref_key), ("random", "aco"), **contract)
    calls = len(provider.requests)
    ref_job = ref.ok(ref_key, "POST", "experiments", body, 201)
    ref.run(ref_ws, ref_job["job_id"])
    expected, ref_calls = result(ref, ref_key, ref_job["job_id"]), len(provider.requests) - calls
    assert sum(u["tool_calls"] for _, _, u, _ in expected) > 3  # the search reaches websearch
    assert not any(f == "executor_error" for *_, cands in expected for c in cands for f in c[2])

    brave = FakeBrave()
    w = websearch_world(tmp_path / "drift", provider, brave)
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    body = w.experiment_body(w.dataset(key), ("random", "aco"), **contract)
    calls = len(provider.requests)
    job = w.ok(key, "POST", "experiments", body, 201)
    original = w.source.config
    real_search = type(w.source).search

    def drifting(self, request):
        response = real_search(self, request)
        if len(brave.requests) == 3:  # the operator's provider changes after the 3rd search
            self.config = BraveSearchConfig(api_key=original.api_key, endpoint=WEBSEARCH_B)
        return response

    type(w.source).search = drifting
    try:
        w.run(ws, job["job_id"])
    finally:
        type(w.source).search = real_search
    view = w.ok(key, "GET", f"experiments/{job['job_id']}")
    assert (view["state"], view["reason"]) == ("INTERRUPTED", "runtime_unavailable"), view
    assert len(brave.requests) == 3  # nothing was sent once it drifted
    store = w.api(ws).jobs.store
    attempts = [
        a
        for u in store.get_job(job["job_id"]).units
        for a in store.attempts(u.job_id, u.strategy, u.seed)
    ]
    assert all(a.state.value == "COMPLETED" for a in attempts)  # nothing ambiguous, nothing lost
    assert not any("identity changed" in (a.result_json or "") for a in attempts)  # not scored

    # the provider is back: the job resumes and finishes exactly as if it had never drifted
    w.source.config = original
    w.ok(key, "POST", f"experiments/{job['job_id']}/resume")
    w.run(ws, job["job_id"])
    assert w.ok(key, "GET", f"experiments/{job['job_id']}")["state"] == "COMPLETED"
    assert result(w, key, job["job_id"]) == expected
    assert len(brave.requests) == len(ref_brave.requests)  # no search was sent twice
    assert len(provider.requests) - calls == ref_calls  # no model call was sent twice


# == 7. memory lifecycle =========================================================================
def test_memory_is_reused_only_for_the_exact_dataset_model_and_runtime(world):
    from memory.priors import RetireReason, row_hash

    w, p = world, world.provider
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    job = w.ok(key, "POST", "experiments", w.experiment_body(data, ("fixed", "aco")), 201)
    w.run(ws, job["job_id"])
    store = w.api(ws).jobs.priors.store
    active = store.active()
    assert {r.record.kind.value for r in active} == {"dataset_champion", "task_family"}
    # held-out rows are never evidence of any prior
    heldout = {row_hash(DATASET_ID, r) for r in rows_of("test", data["splits"])}
    assert all(not heldout & set(r.record.evidence.rows) for r in active)

    def start(body) -> dict[str, Any]:
        job = w.ok(key, "POST", "experiments", body, 201)
        w.run(ws, job["job_id"])
        art = w.ok(key, "GET", f"experiments/{job['job_id']}/artifact")["artifact"]
        (run,) = art["memory"]["runs"]
        return run

    assert start(w.experiment_body(data, ("aco_warm",)))["reason"] == "warm_dataset_champion"

    # a mutated dataset (a new version: one more labelled ticket) never reuses the champion
    rows = FIXTURE.read_text(encoding="utf-8").rstrip("\n") + (
        "\nT-0019,Which city does order ORD-1019 ship to?,Order ORD-1019 (1 x lantern) was "
        "packed at the Reno warehouse and ships to Helena by Ground on 2026-03-13.,Helena\n"
    )
    project = data["project"]["project_id"]
    up2 = w.ok(key, "POST", f"projects/{project}/uploads?format=csv", rows.encode(), 201)
    v2 = w.ok(key, "POST", f"uploads/{up2['upload_id']}/register", REGISTER, 201)
    assert v2["spec"]["dataset_version"] == 2
    sp2 = w.ok(key, "POST", f"datasets/{DATASET_ID}/versions/2/splits", SPLIT_PLAN, 201)
    mutated = {"project": data["project"], "upload": up2, "version": v2, "splits": sp2}
    run = start(w.experiment_body(mutated, ("aco_warm",)))
    assert run["prior_kind"] != "dataset_champion", run

    # another model (a second registry entry): cold, never a stale pin
    other = entry_for(p.endpoint, model_id="dogfood-desk-model-2", name="dogfood-desk-2")
    write_registry(w.registry, w.entry, other)
    w.restart()
    body = w.experiment_body(data, ("aco_warm",))
    body["plan"] = plan_for(other, ("aco_warm",))
    assert start(body)["reason"] == "cold_model_changed"

    # a prior invalidated after a job pinned it: the job fails closed, never starts another way
    pinned = w.ok(key, "POST", "experiments", w.experiment_body(data, ("aco_warm",)), 201)
    for row in store.active():
        store.invalidate(row.prior_id, RetireReason.PROVENANCE_INVALID, "audit")
    sent = len(p.requests)
    w.run(ws, pinned["job_id"])
    view = w.ok(key, "GET", f"experiments/{pinned['job_id']}")
    assert view["state"] in ("FAILED", "INTERRUPTED"), view
    assert len(p.requests) == sent

    # deletion: every prior of the dataset is invalidated and new runs are refused
    w.ok(key, "DELETE", f"datasets/{DATASET_ID}")
    assert store.active() == []
    assert code(w.call(key, "POST", "experiments", w.experiment_body(data, ("aco_warm",)))) == (
        410,
        "dataset_source_deleted",
    )


# == 8. deployment lifecycle =====================================================================
def test_only_the_current_champion_reaches_production_and_tampering_fails_closed(world):
    w, p = world, world.provider
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    first = release(w, ws, key, data, ("fixed",))
    second = release(w, ws, key, data, ("random",))  # champion 2 replaces champion 1
    assert second["version"] != first["version"]
    # champion 1's version may be staged, but never promoted: it is no longer the champion
    w.ok(key, "POST", f"workflows/{first['version']}/stage", {"expected_revision": 0})
    assert code(
        w.call(key, "POST", f"workflows/{first['version']}/promote", {"expected_revision": 1})
    ) == (409, "champion_not_current")
    assert w.ok(key, "GET", f"deployments/{LINEAGE}")["production_version_id"] is None
    assert w.deploy(key, second["version"]) == 3

    # a version document altered behind the database's back no longer verifies
    db = w.data / "workspaces" / ws / "deployments.sqlite3"
    sqlite(db, "DROP TRIGGER workflow_versions_are_immutable")
    doc = w.ok(key, "GET", f"workflows/{second['version']}")["document"]
    doc["versions"]["model_hash"] = "0" * 64
    import json

    sqlite(
        db,
        "UPDATE workflow_versions SET document_json=? WHERE version_id=?",
        (json.dumps(doc), second["version"]),
    )
    sent = len(p.requests)
    for method, path, body in (
        ("GET", f"workflows/{second['version']}", None),
        ("POST", f"deployments/{LINEAGE}/invoke", {"inputs": NEW_ORDER}),
        ("POST", f"workflows/{second['version']}/invoke", {"inputs": NEW_ORDER}),
    ):
        s, err = w.call(key, method, path, body)
        assert s >= 409 and "integrity" in err["error"]["code"], (path, s, err)
    assert len(p.requests) == sent


# == 9. security / privacy =======================================================================
def test_upload_boundaries_malformed_input_and_sensitive_data(world):
    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    project = w.ok(key, "POST", "projects", {"name": "Edge cases"}, 201)["project_id"]

    def upload(raw: bytes, fmt="csv"):
        return w.call(key, "POST", f"projects/{project}/uploads?format={fmt}", raw)

    limit = 1024 * 1024  # the product under test was built with a 1 MiB limit
    header = b"ticket_id,question,order_record,answer\n"

    def line(i: int, n: int) -> bytes:
        return b"T-%06d,q,%s,a\n" % (i, b"r" * n)

    rows = (limit - len(header)) // len(line(0, 1000)) - 1
    exact = header + b"".join(line(i, 1000) for i in range(rows))
    exact += line(rows, limit - len(exact) - len(line(rows, 0)))
    assert len(exact) == limit
    assert upload(exact)[0] == 201
    assert code(upload(exact + b"x"))[1] == "payload_too_large"
    for raw, fmt, expected in (
        (b"a,b\n1,2,3\n", "csv", "malformed_csv"),
        (b'{"a": 1}\n{"a": \n', "jsonl", "malformed_jsonl"),
        (b"a,b\n1,\x00\n", "csv", None),
        (b"a,b\n\xff\xfe,1\n", "csv", "invalid_encoding"),
    ):
        s, err = upload(raw, fmt)
        assert s == 422 or s == 400, (raw, s, err)
        if expected:
            assert err["error"]["code"] == expected, err
    # secrets are blocked (nothing stored); PII is redacted before storage
    s, err = upload(header + b"T-1,q," + PROVIDER_KEY.encode() + b",a\n")
    assert (s, err["error"]["code"]) == (422, "sensitive_data_blocked")
    assert PROVIDER_KEY not in repr(err)
    s, up = upload(header + b"T-1,q,contact jane.doe@example.com,a\n")
    assert s == 201
    assert "jane.doe@example.com" not in repr(up)
    # CSV formula injection: neutralized on ingest, so no workflow or export sees a formula
    s, up = upload(header + b'T-1,q,"=HYPERLINK(""http://evil.example"")",a\n')
    assert s == 201 and up["sensitive_data"]["neutralized_values"] == 1
    assert up["preview"][0]["order_record"].startswith("'=")


def test_outbound_model_requests_carry_one_rows_inputs_and_no_identifiers(world):
    w, p = world, world.provider
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    release(w, ws, key, data, ("fixed", "random"))
    assert p.requests
    for prompt in p.prompts():
        assert len(set(SHIPS_ORDER.findall(prompt))) <= 1  # never another row's data
        assert "T-00" not in prompt  # row ids never leave
        assert "Order desk" not in prompt and DATASET_ID not in prompt
    # the key goes only in the Authorization header, never in a body
    assert all(PROVIDER_KEY not in repr(r) for r in p.requests)


SHIPS_ORDER = __import__("re").compile(r"ORD-\d{4}")


# == monitoring -> re-optimization ===============================================================
def test_production_feedback_triggers_a_challenger_that_never_replaces_production_by_itself(world):
    from tests.test_production_monitoring import SINCE, SMALL, UNTIL

    w = world
    ws, key = w.workspace("Dogfood A", "owner-a@example.test")
    data = w.dataset(key)
    rel = release(w, ws, key, data)
    w.deploy(key, rel["version"])
    prod = w.ok(key, "GET", f"deployments/{LINEAGE}")
    w.ok(key, "POST", "monitoring/policies", SMALL.model_dump(mode="json"), 201)
    # the desk now books orders by their ORIGIN warehouse: reviewers label every answer wrong
    for i, (warehouse, city) in enumerate(
        (("Reno", "Tulsa"), ("Fresno", "Akron"), ("Columbus", "Provo"), ("Reno", "Ogden"))
    ):
        inputs = {
            "question": f"Which city does order ORD-30{i:02d} ship to?",
            "order_record": f"Order ORD-30{i:02d} (1 x crate) was packed at the {warehouse} "
            f"warehouse and ships to {city} by Ground on 2026-05-0{i + 1}.",
        }
        inf = w.ok(key, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": inputs})
        assert inf["status"] == "SUCCEEDED"
        w.ok(
            key,
            "POST",
            f"inferences/{inf['inference_id']}/feedback",
            {"inputs": inputs, "output": inf["output"], "expected": {"answer": warehouse}},
            201,
        )
    trig = w.ok(
        key,
        "POST",
        f"workflows/{rel['version']}/triggers/evaluate",
        {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id},
        201,
    )
    assert trig["outcome"] == "TRIGGERED", trig
    ch = w.ok(key, "GET", f"triggers/{trig['trigger_id']}/challenger")
    assert ch["state"] == "JOB_CREATED", ch
    w.run(ws, ch["challenger_job_id"])
    job = w.ok(key, "GET", f"experiments/{ch['challenger_job_id']}")
    assert job["state"] == "COMPLETED", job
    ch = w.ok(key, "GET", f"triggers/{trig['trigger_id']}/challenger")
    assert ch["dataset"] is not None and ch["deployed"] is False, ch  # a NEW, labelled version
    # the challenger is only a candidate: production is unchanged until a human promotes it
    assert w.ok(key, "GET", f"deployments/{LINEAGE}") == prod
