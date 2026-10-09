"""Versioned champion inference (Issue #30).

    #25 CHAMPION -> immutable workflow version -> staging -> production -> inference -> rollback

Every model here is a TEST DOUBLE. Experiments and their promotions run on the #25 ``Rigged``
backend (a run passes iff a rule over (genome, row) says so), stamped with the runtime versions the
real ``WorkflowRunner`` reports for ``PassageDouble`` - so the champion's #26 provenance pins
exactly the runtime that later serves it. Inference runs the REAL path: compiler ->
``WorkflowRunner`` (MAF) -> ``RegisteredModelClient`` -> ``PassageDouble``, which reads
"<City> is the capital of" from the request's passage. Nothing here is a benchmark result.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from api.product import ProductAPI
from core.genome import Genome
from core.models import (
    AllowedModels,
    ModelCapability,
    ModelEntry,
    ModelPricing,
    ModelRegistry,
)
from core.results import Answer, ExecutionResult
from core.task_contract import TaskContract
from experiments.contract_run import contract_suite
from experiments.deployment import (
    ChampionDeployments,
    ChampionNotCurrent,
    InferenceBackendUnavailable,
    InferenceRuntime,
    InvalidDeploymentTransition,
    InvalidInferenceRequest,
    ModelBindingFailed,
    NoProductionVersion,
    NotDeployed,
    NotPublishable,
    OutputSchemaViolation,
    StaleDeploymentRevision,
    VersionIntegrityError,
    WorkflowVersionNotFound,
    check_document,
    workflow_version_document,
)
from experiments.jobs import ExperimentJobDefinition, ExperimentJobs, JobWorker, load_canonical
from experiments.optimization_experiment import Strategy
from experiments.promotion import ChampionNotFound, ChampionPromotions, Decision
from runtime.model_client import RegisteredModelClient
from runtime.prompts import PROMPT_TEMPLATE_VERSION
from store.champions import SQLiteChampionStore
from store.deployments import InvalidTransition, SQLiteDeploymentStore, VersionState
from store.jobs import JobState, SQLiteJobStore
from tests.test_champion_promotion import (
    EVERYTHING,
    TEST_ROW,
    GateRuntime,
    Rigged,
    World,
    winners,
)
from tests.test_champion_promotion import definition as promotion_definition
from tests.test_contract_runtime import DATA, PassageModel
from tests.test_durable_jobs import LEASE, Clock

try:
    import agent_framework  # noqa: F401

    HAS_MAF = True
except ImportError:  # the maf extra executes workflows; everything else runs without it
    HAS_MAF = False
maf = pytest.mark.skipif(not HAS_MAF, reason="install the maf extra to execute workflows")

LINEAGE = "capitals.capitals"
PRICES = ModelPricing(
    version="test-prices/1", currency="USD", prompt_per_million=1.0, completion_per_million=2.0
)
ENTRY = ModelEntry(
    name="passage-double",
    provider="test-double",
    adapter="test-double",
    model_id="scripted",
    model_hash=PassageModel.model_hash,
    capabilities=(ModelCapability.TEXT_GENERATION,),
    pricing=PRICES,
)
OTHER = ENTRY.model_copy(update={"name": "other-double", "model_id": "other", "model_hash": "m-2"})
SHAPE = ("schema", "lineage_id", "synthetic", "champion", "workflow", "contract", "grammar")
SHAPE += ("model", "versions")
INPUTS = {
    "question": "Which city is the capital of Peru?",
    "passage": "Lima is the capital of Peru. It is a large city.",
}


# -- test doubles -------------------------------------------------------------------------------
class PassageDouble(PassageModel):
    """TEST DOUBLE model (``PassageModel``) that counts every request it receives."""


class OtherBackend(PassageModel):
    """TEST DOUBLE: a different model (another model_hash) behind the same protocol."""

    model_hash = OTHER.model_hash


def real_versions(contract: TaskContract) -> dict[str, Any]:
    """The RunVersions the real runner reports for ``ENTRY`` on ``contract``."""
    from runtime.runner import WorkflowRunner

    runner = WorkflowRunner(
        model=RegisteredModelClient(ENTRY, PassageDouble()), benchmark_hash="inline"
    )
    suite, _ = contract_suite(contract, promotion_definition().splits, DATA)
    return runner.versions(suite.tasks[0]).model_dump(mode="json")


class Stamped(Rigged):
    """``Rigged``, with every run stamped with the real runtime's versions for ``ENTRY``."""

    def __init__(self, passes, **kw) -> None:
        super().__init__(passes, **kw)
        inner = self._evaluate

        def evaluate(genome, task, trial, seed):
            run = inner(genome, task, trial, seed)
            versions = run.execution.key.versions.model_validate(real_versions(task.contract))
            key = run.execution.key.model_copy(update={"versions": versions})
            execution = run.execution.model_copy(update={"key": key})
            return run.model_copy(update={"execution": execution})

        self._evaluate = evaluate


class ModelRuntime(GateRuntime):
    """TEST DOUBLE runtime that binds the experiment's model from the registry entry (#27)."""

    def bind(self, definition: ExperimentJobDefinition):
        return replace(super().bind(definition), models=AllowedModels((ENTRY,)))


def definition(strategies=(Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)):
    d = promotion_definition(strategies=strategies)
    plan = d.plan.model_copy(
        update={
            "expected_model_hash": ENTRY.model_hash,
            "expected_prompt_version": PROMPT_TEMPLATE_VERSION,
        }
    )
    return ExperimentJobDefinition.model_validate(d.model_dump() | {"plan": plan.model_dump()})


class Env:
    """One process: jobs (#24), promotions (#25), deployments (#30) over one data directory."""

    def __init__(self, root: Path, backend: Rigged | None = None, *, name: str = "p") -> None:
        self.root = root
        self.backend = backend or Stamped(EVERYTHING)
        self.clock = Clock()
        self.runtime = ModelRuntime(self.backend)
        self.jobs = ExperimentJobs(
            SQLiteJobStore(root / "jobs.sqlite3"), self.runtime, clock=self.clock
        )
        self.worker = JobWorker(
            self.jobs.store,
            self.runtime,
            worker_id=name,
            lease_s=LEASE,
            clock=self.clock,
            heartbeat=False,
        )
        self.promotions = ChampionPromotions(
            SQLiteChampionStore(root / "champions.sqlite3"),
            self.jobs.store,
            self.runtime,
            clock=self.clock,
            owner=name,
            lease_s=LEASE,
            heartbeat=False,
        )
        self.registry = ModelRegistry(entries=(ENTRY,))
        self.model = PassageDouble()
        self.clients = 0
        self.inference_runtime = InferenceRuntime(
            registry=lambda: self.registry, client=self.client
        )
        self.deployments = ChampionDeployments(
            SQLiteDeploymentStore(root / "deployments.sqlite3"),
            self.promotions,
            self.inference_runtime,
        )

    def client(self, entry: ModelEntry):
        self.clients += 1
        return RegisteredModelClient(entry, self.model)

    def experiment(self, d: ExperimentJobDefinition) -> str:
        job_id = self.jobs.create(d).job_id
        self.worker.run_until_idle(job_id)
        assert self.jobs.get(job_id).state is JobState.COMPLETED
        return job_id

    def champion(self, d: ExperimentJobDefinition | None = None) -> str:
        view = self.promotions.promote(self.experiment(d or definition()))
        assert view.decision is Decision.PROMOTED, view.record
        assert view.record is not None
        return view.record["champion"]["champion_id"]

    def publish(self, champion_id: str) -> str:
        view, _ = self.deployments.publish(champion_id)
        return view.version_id

    def revision(self, lineage: str = LINEAGE) -> int:
        return self.deployments.deployment(lineage).revision

    def deploy(self, version_id: str) -> None:
        self.deployments.stage(version_id, self.revision())
        self.deployments.promote(version_id, self.revision())

    def db(self, statement: str, params: tuple = ()) -> int:
        conn = sqlite3.connect(self.root / "deployments.sqlite3", isolation_level=None)
        try:
            return (
                conn.execute(statement, params).fetchone()[0]
                if statement.startswith("SELECT")
                else conn.execute(statement, params).rowcount
            )
        finally:
            conn.close()

    def inference_records(self) -> int:
        return self.db("SELECT COUNT(*) FROM inference_records")


@cache
def win() -> dict[str, str]:
    """The workflow only fixed / random / ACO reaches first (``winners``), per strategy. Search
    order does not depend on the model, so the #25 synthetic definition finds the same ones."""
    return winners(promotion_definition())


FIRST = (Strategy.FIXED,)
SECOND = (Strategy.RANDOM,)


def lineage(tmp_path: Path) -> Env:
    """Lineage capitals.capitals where the fixed and the random winner pass every row: an
    experiment of FIRST makes the fixed winner champion 1, a later one of SECOND makes the
    random winner champion 2 (a tie on held-out is no regression)."""
    return Env(tmp_path, Stamped(lambda g, row: g in (win()["fixed"], win()["random"])))


def release(env: Env, strategies, deploy: bool = False) -> str:
    """The lineage's next champion, published (and staged + promoted when ``deploy``)."""
    version = env.publish(env.champion(definition(strategies)))
    if deploy:
        env.deploy(version)
    return version


# -- 1. only a champion is published, as an immutable pinned version ----------------------------
def test_only_a_promoted_champion_can_be_published(tmp_path):
    def passes(g: str, row: str) -> bool:  # the random winner regresses on the test row
        return g == win()["fixed"] or (g == win()["random"] and row != TEST_ROW)

    env = Env(tmp_path, Stamped(passes))
    incumbent = env.champion(definition(FIRST))
    view = env.promotions.promote(env.experiment(definition(SECOND)))
    assert view.decision is Decision.REJECTED and view.challenger is not None
    assert env.promotions.store.current(LINEAGE).champion_id == incumbent

    for not_a_champion in ("c-" + "0" * 24, view.promotion_id, view.challenger["genome_hash"]):
        with pytest.raises(ChampionNotFound):
            env.deployments.publish(not_a_champion)
    assert env.deployments.list().versions == []
    assert env.deployments.publish(incumbent)[0].champion_id == incumbent  # the champion can


def test_publishing_creates_a_version_but_deploys_nothing(tmp_path):
    env = Env(tmp_path)
    version = env.publish(env.champion())

    view = env.deployments.get(version)
    deployment = env.deployments.deployment(LINEAGE)
    assert view.state is VersionState.CREATED
    assert (deployment.revision, deployment.staging_version_id) == (0, None)
    assert deployment.production_version_id is None
    with pytest.raises(NotDeployed):
        env.deployments.invoke(version, INPUTS)
    with pytest.raises(NoProductionVersion):
        env.deployments.invoke_production(LINEAGE, INPUTS)
    assert env.model.requests == [] and env.clients == 0


def test_the_version_pins_every_identity_from_the_canonical_provenance(tmp_path):
    env = Env(tmp_path)
    champion_id = env.champion()
    view, created = env.deployments.publish(champion_id)
    doc = view.document

    champ = env.promotions.store.champion(champion_id)
    record = json.loads(champ.record_json)
    _, art = load_canonical(env.jobs.store, record["provenance"]["job_id"])
    prov = art.provenance
    assert created and view.version_id.startswith("wv-")
    assert doc["champion"] | {} == {
        **doc["champion"],
        "champion_id": champion_id,
        "champion_version": 1,
        "promotion_id": champ.promotion_id,
        "compat_hash": champ.compat_hash,
    }
    assert doc["workflow"] == {"genome_hash": record["genome_hash"], "genome": record["genome"]}
    assert (
        doc["provenance"]["artifact_id"] == art.artifact_id == record["provenance"]["artifact_id"]
    )
    assert doc["provenance"]["provenance_id"] == prov.provenance_id
    assert doc["provenance"]["experiment_sha256"] == art.experiment_sha256
    assert doc["contract"]["task_contract"] == prov.contract.contract
    assert doc["contract"]["contract_hash"] == prov.contract.contract_hash
    contract = TaskContract.model_validate(prov.contract.contract)
    assert doc["contract"]["input_schema"] == contract.input_schema.model_dump(mode="json")
    assert doc["contract"]["output_schema"] == contract.output_schema.model_dump(mode="json")
    assert doc["grammar"]["version"] == prov.grammar.version
    assert doc["grammar"]["grammar_hash"] == prov.grammar.grammar_hash
    assert doc["model"]["registry_entry"] == ENTRY.model_dump(mode="json")
    assert doc["model"]["registry_entry_hash"] == ENTRY.identity_hash
    assert doc["model"]["model_hash"] == ENTRY.model_hash == prov.model.model_hash
    assert doc["versions"] == prov.versions.run_versions[0] == real_versions(contract)
    assert doc["versions"]["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    # the same champion always publishes the same, content-addressed version
    again, created_again = env.deployments.publish(champion_id)
    assert (again.version_id, again.document, created_again) == (view.version_id, doc, False)


def test_publishing_cross_checks_the_champion_against_its_provenance(tmp_path):
    env = Env(tmp_path)
    champion_id = env.champion()
    champ = env.promotions.store.champion(champion_id)
    verified = env.promotions.verify(champ.promotion_id)
    job, art = load_canonical(env.jobs.store, verified.job_id)
    record = json.loads(champ.record_json)
    kw = dict(
        champion_id=champion_id,
        champion_version=champ.version,
        lineage_id=champ.lineage_id,
        compat_hash=champ.compat_hash,
        record_json=champ.record_json,
        decision=verified.record,
        definition=ExperimentJobDefinition.model_validate_json(job.definition_json),
        artifact_ref=art.ref(),
        body=art.body,
        provenance=art.provenance,
    )
    assert workflow_version_document(record=record, **kw)["champion"]["champion_id"] == champion_id

    def tampered(path: tuple[str, ...], value: Any) -> dict[str, Any]:
        out = json.loads(champ.record_json)
        node = out
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
        return out

    for path, value in [
        (("compatibility", "model_hash"), "another-model"),
        (("provenance", "artifact_id"), "f" * 64),
        (("provenance", "strategy"), "aco"),
        (("genome_hash",), "0" * 64),
        (("compatibility", "grammar_version"), "grammar/0"),
    ]:
        with pytest.raises(NotPublishable):
            workflow_version_document(record=tampered(tuple(path), value), **kw)
    # a provenance record that pinned no registry entry has no exact model to serve
    other = art.provenance.model_copy(
        update={"model": art.provenance.model.model_copy(update={"registry_entry": None})}
    )
    with pytest.raises(NotPublishable, match="registry entry"):
        workflow_version_document(record=record, **(kw | {"provenance": other}))


def test_a_champion_without_a_pinned_registry_entry_is_not_publishable(tmp_path):
    """A #25 champion whose experiment bound no registry entry has nothing exact to serve."""
    w = World(tmp_path, Rigged(EVERYTHING))
    w.promote(w.experiment(promotion_definition(strategies=(Strategy.FIXED,))))
    champion = w.promotions.current(LINEAGE)
    deployments = ChampionDeployments(SQLiteDeploymentStore(tmp_path / "d.sqlite3"), w.promotions)
    with pytest.raises(NotPublishable, match="registry entry"):
        deployments.publish(champion.champion_id)
    assert deployments.list().versions == []


def test_a_workflow_version_is_immutable(tmp_path):
    env = Env(tmp_path)
    version = env.publish(env.champion())
    for statement in (
        "UPDATE workflow_versions SET document_json = '{}'",
        "UPDATE workflow_versions SET lineage_id = 'other'",
        "DELETE FROM workflow_versions",
        "UPDATE deployment_events SET actor = 'mallory'",
        "DELETE FROM deployment_events",
        "UPDATE version_states SET state = 'PRODUCTION'",  # CREATED -> PRODUCTION skips staging
    ):
        with pytest.raises(sqlite3.IntegrityError):
            env.db(statement)
    # a stored document altered behind the database's back no longer verifies: fail closed
    env.deploy(version)
    env.db("DROP TRIGGER workflow_versions_are_immutable")
    row = env.deployments.store.version(version)
    doc = json.loads(row.document_json)
    doc["versions"]["model_hash"] = "swapped"
    env.db(
        "UPDATE workflow_versions SET document_json=? WHERE version_id=?",
        (json.dumps(doc), version),
    )
    with pytest.raises(VersionIntegrityError):
        env.deployments.get(version)
    with pytest.raises(VersionIntegrityError):
        env.deployments.invoke_production(LINEAGE, INPUTS)
    assert env.model.requests == []


# -- 2. lifecycle: CREATED -> STAGING -> PRODUCTION -> RETIRED ----------------------------------
def test_staging_is_not_production_and_promotion_is_explicit(tmp_path):
    env = Env(tmp_path)
    version = env.publish(env.champion())
    env.clients = 0
    with pytest.raises(InvalidDeploymentTransition):  # CREATED cannot skip staging
        env.deployments.promote(version, env.revision())
    assert env.clients == 0  # refused before any model was bound
    with pytest.raises(InvalidTransition):  # and the store refuses it on its own
        env.deployments.store.promote(version, env.revision(), "api")

    staged = env.deployments.stage(version, 0)
    assert (staged.revision, staged.staging_version_id, staged.production_version_id) == (
        1,
        version,
        None,
    )
    assert env.deployments.get(version).state is VersionState.STAGING
    with pytest.raises(NoProductionVersion):
        env.deployments.invoke_production(LINEAGE, INPUTS)
    with pytest.raises(InvalidDeploymentTransition):  # a version is staged once
        env.deployments.stage(version, 1)

    promoted = env.deployments.promote(version, 1)
    assert (promoted.revision, promoted.staging_version_id) == (2, None)
    assert promoted.production_version_id == version
    assert env.deployments.get(version).state is VersionState.PRODUCTION
    history = env.deployments.history(LINEAGE)
    assert [
        (e.action, e.version_id, e.revision_before, e.revision_after) for e in history.events
    ] == [
        ("publish", version, 0, 0),
        ("stage", version, 0, 1),
        ("promote", version, 1, 2),
    ]


def test_a_stale_deployment_write_fails_closed_and_writes_nothing(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)  # revision 2
    v2 = release(env, SECOND)
    env.deployments.stage(v2, 2)  # revision 3
    before = env.deployments.history(LINEAGE)

    for stale in (0, 1, 2, 4):  # the deployment is at revision 3
        with pytest.raises(StaleDeploymentRevision):
            env.deployments.promote(v2, stale)
        with pytest.raises(StaleDeploymentRevision):
            env.deployments.rollback(LINEAGE, stale, v1)
    assert env.deployments.history(LINEAGE) == before
    assert env.deployments.get(v1).state is VersionState.PRODUCTION
    assert env.deployments.get(v2).state is VersionState.STAGING


def test_concurrent_production_promotions_never_both_win(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)
    v2 = release(env, SECOND)
    env.deployments.stage(v2, env.revision())
    revision = env.revision()
    barrier = threading.Barrier(4)
    outcomes: list[str] = []
    lock = threading.Lock()

    def promote() -> None:
        deployments = ChampionDeployments(  # one process each: its own store connection
            SQLiteDeploymentStore(tmp_path / "deployments.sqlite3"),
            env.promotions,
            env.inference_runtime,
        )
        barrier.wait()
        try:
            deployments.promote(v2, revision)
            result = "won"
        except (StaleDeploymentRevision, InvalidDeploymentTransition):
            result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=promote) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["refused", "refused", "refused", "won"]
    deployment = env.deployments.deployment(LINEAGE)
    assert (deployment.production_version_id, deployment.revision) == (v2, revision + 1)
    promotes = [e for e in env.deployments.history(LINEAGE).events if e.action == "promote"]
    assert [(e.version_id, e.replaced_version_id) for e in promotes] == [(v1, None), (v2, v1)]


def test_the_database_holds_at_most_one_production_version_per_lineage(tmp_path):
    env = lineage(tmp_path)
    release(env, FIRST, deploy=True)
    v2 = release(env, SECOND)
    env.db("DROP TRIGGER version_states_follow_the_lifecycle")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        env.db("UPDATE version_states SET state='PRODUCTION' WHERE version_id=?", (v2,))


def test_only_the_current_champion_can_be_promoted(tmp_path):
    env = lineage(tmp_path)
    old = release(env, FIRST)
    env.champion(definition(SECOND))  # champion 1 is no longer the lineage's current champion
    env.deployments.stage(old, env.revision())
    with pytest.raises(ChampionNotCurrent):
        env.deployments.promote(old, env.revision())
    assert env.deployments.deployment(LINEAGE).production_version_id is None


# -- 3. rollback --------------------------------------------------------------------------------
def test_rollback_restores_the_exact_previous_immutable_version(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)
    v1_stored = env.deployments.store.version(v1).document_json
    v2 = release(env, SECOND, deploy=True)
    second = env.promotions.store.current(LINEAGE).champion_id
    assert env.deployments.get(v1).state is VersionState.RETIRED

    rolled = env.deployments.rollback(LINEAGE, env.revision(), actor="oncall")

    assert rolled.production_version_id == v1
    assert env.deployments.get(v1).state is VersionState.PRODUCTION
    assert env.deployments.get(v2).state is VersionState.RETIRED
    assert env.deployments.store.version(v1).document_json == v1_stored  # never rewritten
    assert env.deployments.get(v1).document["workflow"]["genome_hash"] == win()["fixed"]
    events = env.deployments.history(LINEAGE).events
    assert [(e.action, e.version_id, e.replaced_version_id, e.actor) for e in events] == [
        ("publish", v1, None, "api"),
        ("stage", v1, None, "api"),
        ("promote", v1, None, "api"),
        ("publish", v2, None, "api"),
        ("stage", v2, None, "api"),
        ("promote", v2, v1, "api"),
        ("rollback", v1, v2, "oncall"),
    ]
    # rolling back never reran promotion, evaluation or optimization: the champions are unchanged
    assert env.promotions.store.current(LINEAGE).champion_id == second
    # forward again: v2 was in production before, so it can be restored the same way
    again = env.deployments.rollback(LINEAGE, env.revision(), v2)
    assert again.production_version_id == v2


def test_rollback_only_restores_a_version_that_was_in_production(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST)
    with pytest.raises(NoProductionVersion):
        env.deployments.rollback(LINEAGE, env.revision())
    v2 = release(env, SECOND, deploy=True)
    with pytest.raises(InvalidDeploymentTransition):  # nothing came before v2
        env.deployments.rollback(LINEAGE, env.revision())
    with pytest.raises(InvalidDeploymentTransition):  # v1 never served production
        env.deployments.rollback(LINEAGE, env.revision(), v1)
    with pytest.raises(InvalidDeploymentTransition):
        env.deployments.rollback(LINEAGE, env.revision(), v2)
    assert env.deployments.deployment(LINEAGE).production_version_id == v2


def test_a_displaced_staging_version_is_never_rolled_back_to(tmp_path):
    """RETIRED is not enough: rollback restores only a version that served production."""
    env = lineage(tmp_path)
    v1 = release(env, FIRST)
    env.deployments.stage(v1, env.revision())
    v2 = release(env, SECOND)
    env.deployments.stage(v2, env.revision())  # displaces v1: RETIRED, never in production
    assert env.deployments.get(v1).state is VersionState.RETIRED
    env.deployments.promote(v2, env.revision())
    with pytest.raises(InvalidDeploymentTransition, match="never in production"):
        env.deployments.rollback(LINEAGE, env.revision(), v1)
    assert env.deployments.deployment(LINEAGE).production_version_id == v2


def test_a_restart_preserves_the_active_production_version(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)
    release(env, SECOND, deploy=True)
    env.deployments.rollback(LINEAGE, env.revision())
    before = env.deployments.history(LINEAGE)

    restarted = Env(tmp_path, name="after-restart")  # new process over the same data directory

    assert restarted.deployments.history(LINEAGE) == before
    assert restarted.deployments.deployment(LINEAGE).production_version_id == v1
    assert restarted.deployments.get(v1).document == env.deployments.get(v1).document


# -- 4. model / runtime binding fails closed before any model call ------------------------------
def _deployed(tmp_path) -> tuple[Env, str]:
    """A deployed champion found by random search (it answers from the passage)."""
    env = Env(tmp_path, Stamped(lambda g, row: g == win()["random"]))
    version = env.publish(env.champion())
    env.deploy(version)
    env.clients = 0
    return env, version


def _refused(env: Env, reason: str) -> None:
    with pytest.raises(ModelBindingFailed) as exc:
        env.deployments.invoke_production(LINEAGE, INPUTS)
    assert exc.value.details["reason"] == reason
    assert env.model.requests == [] and env.inference_records() == 0


@pytest.mark.parametrize(
    "registry, reason",
    [
        (lambda: ModelRegistry(entries=(OTHER,)), "registry_entry_missing"),
        (lambda: ModelRegistry(entries=()), "registry_entry_missing"),
        (
            lambda: ModelRegistry(entries=(ENTRY.model_copy(update={"enabled": False}),)),
            "registry_entry_disabled",
        ),
        (
            lambda: ModelRegistry(entries=(ENTRY.model_copy(update={"revision": "2026-10"}),)),
            "registry_entry_changed",
        ),
        (  # a capability the workflow needs is no longer declared
            lambda: ModelRegistry(entries=(ENTRY.model_copy(update={"capabilities": ()}),)),
            "registry_entry_changed",
        ),
    ],
)
def test_a_missing_disabled_or_changed_registry_entry_fails_closed(tmp_path, registry, reason):
    env, _ = _deployed(tmp_path)
    env.registry = registry()
    _refused(env, reason)


def test_a_client_with_another_model_hash_is_never_substituted(tmp_path):
    env, _ = _deployed(tmp_path)
    env.deployments.runtime = InferenceRuntime(
        registry=lambda: env.registry,
        client=lambda entry: RegisteredModelClient(OTHER, OtherBackend()),
    )
    _refused(env, "model_hash_mismatch")
    assert env.deployments.store.history(LINEAGE)[-1].action == "promote"


def test_a_client_bound_to_another_registry_entry_is_refused(tmp_path):
    env, _ = _deployed(tmp_path)
    twin = ENTRY.model_copy(update={"name": "twin", "endpoint": "http://elsewhere"})
    env.deployments.runtime = InferenceRuntime(
        registry=lambda: env.registry, client=lambda entry: RegisteredModelClient(twin, env.model)
    )
    _refused(env, "model_identity_mismatch")


@pytest.mark.parametrize(
    "target, value",
    [
        ("runtime.runner.PROMPT_TEMPLATE_VERSION", "mvp-999"),
        ("compiler.maf_compiler.MAFCompiler.version", "maf/compiler/999"),
    ],
)
def test_an_incompatible_prompt_or_compiler_version_fails_closed(
    tmp_path, monkeypatch, target, value
):
    env, _ = _deployed(tmp_path)
    monkeypatch.setattr(target, value)
    _refused(env, "runtime_incompatible")


def test_an_unavailable_runtime_cannot_stage_promote_or_serve(tmp_path):
    env = Env(tmp_path)
    version = env.publish(env.champion())
    offline = ChampionDeployments(env.deployments.store, env.promotions)  # no model runtime
    with pytest.raises(InferenceBackendUnavailable):
        offline.stage(version, 0)
    env.deploy(version)
    with pytest.raises(InferenceBackendUnavailable):
        offline.invoke_production(LINEAGE, INPUTS)
    assert offline.get(version).state is VersionState.PRODUCTION  # still inspectable


def test_a_version_whose_model_cannot_bind_is_never_staged(tmp_path):
    env = Env(tmp_path)
    version = env.publish(env.champion())
    env.registry = ModelRegistry(entries=())
    with pytest.raises(ModelBindingFailed):
        env.deployments.stage(version, 0)
    assert env.deployments.get(version).state is VersionState.CREATED


# -- 5. request / response schemas --------------------------------------------------------------
@pytest.mark.parametrize(
    "inputs, field",
    [
        ({"question": "Which city?"}, "inputs.passage"),  # missing
        ({**INPUTS, "answer": "Lima"}, "inputs.answer"),  # unknown (a target column)
        ({**INPUTS, "passage": 42}, "inputs.passage"),  # wrong type
        ({**INPUTS, "question": None}, "inputs.question"),  # required
    ],
)
def test_a_bad_request_is_rejected_before_any_model_is_bound(tmp_path, inputs, field):
    env, version = _deployed(tmp_path)
    with pytest.raises(InvalidInferenceRequest) as exc:
        env.deployments.invoke(version, inputs)
    assert exc.value.details["field"] == field
    with pytest.raises(InvalidInferenceRequest) as again:
        env.deployments.invoke_production(LINEAGE, inputs)
    # nothing is bound or invoked; each rejection is ONE FAILED #30 telemetry record
    assert env.clients == 0 and env.model.requests == [] and env.inference_records() == 2
    for err in (exc.value, again.value):
        assert err.details["failure_kind"] == "input_schema_invalid"
        record = env.deployments.inference(err.details["inference_id"])
        assert record.status == "FAILED" and record.workflow_version == version
        assert record.failure["kind"] == "input_schema_invalid" and record.run_id is None
        assert record.usage.model_calls == 0 and record.output_sha256 is None
        assert record.failure["violations"]  # value-free: names / codes / JSON types only
        for v in record.failure["violations"]:
            assert set(v) <= {"field", "field_sha256", "code", "observed_type"}


def test_an_answer_that_breaks_the_output_schema_is_never_returned(tmp_path, monkeypatch):
    env, version = _deployed(tmp_path)

    def run_sync(self, genome, task, *, trial=0, seed=0):  # the workflow answers a number
        key = self.run_key(genome, task, trial=trial, seed=seed)
        return ExecutionResult(key=key, answer=Answer(values={"answer": 42}))

    monkeypatch.setattr("runtime.runner.WorkflowRunner.run_sync", run_sync)
    with pytest.raises(OutputSchemaViolation) as exc:
        env.deployments.invoke(version, INPUTS)
    record = env.deployments.inference(exc.value.details["inference_id"])
    assert record.status == "FAILED" and record.failure["kind"] == "output_schema_invalid"
    assert record.output is None and record.output_sha256 is None


# -- 6. inference: the exact frozen champion, nothing else --------------------------------------
@maf
@pytest.mark.parametrize("strategy", ["fixed", "random", "aco"])
def test_fixed_random_and_aco_champions_are_equally_deployable(tmp_path, strategy):
    genome_hash = win()[strategy]
    env = Env(tmp_path, Stamped(lambda g, row: g == genome_hash))
    champion_id = env.champion()
    version = env.publish(champion_id)
    env.deploy(version)

    out = env.deployments.invoke_production(LINEAGE, INPUTS)

    assert out.status == "SUCCEEDED" and out.workflow_version == version
    assert out.genome_hash == genome_hash and out.provenance["strategy"] == strategy
    assert set(out.output) == {"answer"} and isinstance(out.output["answer"], str)
    assert out.usage.model_calls == len(env.model.requests) > 0
    # the version document carries no optimizer-specific behaviour: only provenance differs
    doc = env.deployments.get(version).document
    assert set(doc) - {"provenance"} == set(SHAPE)


@maf
def test_inference_runs_the_exact_champion_genome_and_no_optimizer_or_evaluator(
    tmp_path, monkeypatch
):
    env, version = _deployed(tmp_path)
    champion = json.loads(env.promotions.store.current(LINEAGE).record_json)

    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not search, evaluate or promote")

    for target in (
        "optimizers.base.Optimizer.propose",
        "optimizers.base.Optimizer.observe",
        "experiments.optimization_experiment.run_strategy",
        "experiments.optimization_experiment.TimedEvaluate.__call__",
        "evaluation.contract_eval.ContractEvaluator.evaluate",
        "evaluation.contract_eval.ContractEvaluator.evaluate_run",
        "experiments.promotion.ChampionPromotions.promote",
        "experiments.promotion.ChampionPromotions.verify",
        "tests.test_durable_jobs.Backend.__call__",
    ):
        monkeypatch.setattr(target, forbidden)
    executed: list[Genome] = []
    from runtime.runner import WorkflowRunner

    run = WorkflowRunner.run

    async def spy(self, genome, task, **kw):
        executed.append(genome)
        return await run(self, genome, task, **kw)

    monkeypatch.setattr(WorkflowRunner, "run", spy)

    out = env.deployments.invoke(version, INPUTS)

    assert executed == [Genome.from_canonical(champion["genome"])]
    assert out.genome_hash == champion["genome_hash"] == executed[0].genome_hash
    assert out.versions == env.deployments.get(version).document["versions"]
    assert all(INPUTS["question"] in r.input_text for r in env.model.requests)


@maf
def test_every_inference_record_traces_to_the_champion_and_its_provenance(tmp_path):
    env, version = _deployed(tmp_path)
    doc = env.deployments.get(version).document

    out = env.deployments.invoke_production(LINEAGE, INPUTS)

    assert out.output == {"answer": "Lima"}
    record = env.deployments.inference(out.inference_id)
    _, art = load_canonical(env.jobs.store, doc["provenance"]["job_id"])
    assert record.output is None  # records keep only the output's hash
    assert record.output_sha256 is not None and record.request_sha256
    assert record.model_dump(exclude={"output"}) == out.model_dump(exclude={"output"})
    assert (record.workflow_version, record.champion_id) == (
        version,
        doc["champion"]["champion_id"],
    )
    assert record.addressed_by == "production" and record.deployment_revision == env.revision()
    assert record.provenance["artifact_id"] == art.artifact_id
    assert record.provenance["provenance_id"] == art.provenance_id
    assert record.provenance["promotion_id"] == doc["champion"]["promotion_id"]
    assert record.model == {
        "model_hash": ENTRY.model_hash,
        "registry_entry_hash": ENTRY.identity_hash,
        "name": ENTRY.name,
    }
    assert record.versions["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    usage = record.usage
    requests = env.model.requests
    assert usage.model_calls == len(requests)
    assert (usage.prompt_tokens, usage.completion_tokens) == (
        50 * len(requests),
        10 * len(requests),
    )
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.cost == pytest.approx(PRICES.cost(usage.prompt_tokens, usage.completion_tokens))
    assert usage.cost_authoritative and usage.latency_s >= 0.0


@maf
def test_a_staging_version_is_served_by_exact_version_only(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)
    v2 = release(env, SECOND)
    env.deployments.stage(v2, env.revision())

    staged = env.deployments.invoke(v2, INPUTS)
    production = env.deployments.invoke_production(LINEAGE, INPUTS)

    assert (staged.workflow_version, staged.addressed_by) == (v2, "version")
    assert (production.workflow_version, production.addressed_by) == (v1, "production")


@maf
def test_after_rollback_production_serves_the_old_version(tmp_path):
    env = lineage(tmp_path)
    v1 = release(env, FIRST, deploy=True)
    v2 = release(env, SECOND, deploy=True)
    assert env.deployments.invoke_production(LINEAGE, INPUTS).genome_hash == win()["random"]

    env.deployments.rollback(LINEAGE, env.revision())

    out = env.deployments.invoke_production(LINEAGE, INPUTS)
    assert (out.workflow_version, out.genome_hash) == (v1, win()["fixed"])
    with pytest.raises(NotDeployed):  # the retired version no longer serves
        env.deployments.invoke(v2, INPUTS)


# -- 7. the API -----------------------------------------------------------------------------------
def call(api: ProductAPI, method: str, path: str, body: Any = None):
    raw = b"" if body is None else json.dumps(body).encode()
    headers = {"Content-Length": str(len(raw))}
    if body is not None:
        headers["Content-Type"] = "application/json"
    res = api.handle(method, "/api/v1/" + path, headers, io.BytesIO(raw))
    return res.status, res.body


@maf
def test_the_api_publishes_stages_promotes_serves_and_rolls_back(tmp_path):
    env = lineage(tmp_path)
    api = ProductAPI(None, env.jobs, env.promotions, env.deployments)  # type: ignore[arg-type]
    first = env.champion(definition(FIRST))

    status, v1 = call(api, "POST", "workflows", {"champion_id": first})
    assert status == 201 and v1["state"] == "CREATED"
    assert call(api, "POST", "workflows", {"champion_id": first})[0] == 200
    assert call(api, "POST", "workflows", {"champion_id": "not-a-champion"})[0] == 400
    assert (
        call(api, "POST", "workflows", {"champion_id": "c-" + "0" * 24})[1]["error"]["code"]
        == "champion_not_found"
    )
    wv1 = v1["version_id"]
    assert call(api, "GET", f"workflows/{wv1}")[1]["document"] == v1["document"]
    status, d = call(api, "POST", f"workflows/{wv1}/stage", {"expected_revision": 0})
    assert status == 200 and d["staging_version_id"] == wv1
    status, err = call(api, "POST", f"workflows/{wv1}/promote", {"expected_revision": 0})
    assert (status, err["error"]["code"]) == (409, "stale_deployment")
    status, d = call(api, "POST", f"workflows/{wv1}/promote", {"expected_revision": 1})
    assert status == 200 and d["production_version_id"] == wv1

    status, out = call(api, "POST", f"deployments/{LINEAGE}/invoke", {"inputs": INPUTS})
    assert status == 200 and set(out["output"]) == {"answer"}
    assert out["workflow_version"] == wv1 and out["genome_hash"] == win()["fixed"]
    assert call(api, "GET", f"inferences/{out['inference_id']}")[1]["output"] is None
    status, err = call(api, "POST", f"workflows/{wv1}/invoke", {"inputs": {"question": "x"}})
    assert (status, err["error"]["code"]) == (422, "invalid_inference_request")
    assert err["error"]["details"]["field"] == "inputs.passage"

    second = env.champion(definition(SECOND))
    wv2 = call(api, "POST", "workflows", {"champion_id": second})[1]["version_id"]
    call(api, "POST", f"workflows/{wv2}/stage", {"expected_revision": 2})
    call(api, "POST", f"workflows/{wv2}/promote", {"expected_revision": 3, "actor": "release"})
    status, d = call(
        api, "POST", f"deployments/{LINEAGE}/rollback", {"expected_revision": 4, "actor": "oncall"}
    )
    assert status == 200 and d["production_version_id"] == wv1
    status, history = call(api, "GET", f"deployments/{LINEAGE}/history")
    assert [(e["action"], e["actor"]) for e in history["events"]][-2:] == [
        ("promote", "release"),
        ("rollback", "oncall"),
    ]
    listed = call(api, "GET", f"workflows?lineage={LINEAGE}")[1]["versions"]
    assert {v["version_id"]: v["state"] for v in listed} == {wv1: "PRODUCTION", wv2: "RETIRED"}
    assert call(api, "GET", "workflows/wv-" + "0" * 24)[1]["error"]["code"] == (
        "workflow_version_not_found"
    )
    assert call(api, "GET", "deployments/nope")[1]["error"]["code"] == "deployment_not_found"


def test_unknown_versions_and_lineages_are_not_found(tmp_path):
    env = Env(tmp_path)
    with pytest.raises(WorkflowVersionNotFound):
        env.deployments.get("wv-" + "1" * 24)
    with pytest.raises(WorkflowVersionNotFound):
        env.deployments.invoke("wv-" + "1" * 24, INPUTS)
    with pytest.raises(VersionIntegrityError):  # a row that is not its content address
        version = env.publish(env.champion())
        row = env.deployments.store.version(version)
        check_document(replace(row, version_id="wv-" + "2" * 24))
