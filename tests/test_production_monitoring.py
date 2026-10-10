"""Production monitoring + re-optimization triggers (Issue #31).

    deployed workflow version -> #30 inference records + quality feedback -> aggregates, drift,
    trigger rules -> NEW dataset version -> NEW challenger job (Fixed vs Random vs ACO)

Every model here is a TEST DOUBLE (see ``tests.test_champion_inference``): experiments and their
promotions run on the #25 ``Rigged`` backend; inference runs the REAL compiler -> MAF ->
``RegisteredModelClient`` path against ``PassageDouble``. Nothing here is a benchmark result.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from api.product import ProductAPI
from core.canonical import canonical_hash
from core.constraints import ConstraintChecker
from core.dataset import SplitRole, SplitUse
from core.evaluation_spec import EvaluationSpec
from core.models import AllowedModels
from core.objective import Metric, ObjectiveMode, ObjectiveSpec
from core.task_spec import AnswerSchema
from experiments.contract_run import contract_suite
from experiments.deployment import (
    INPUT_SCHEMA_REJECTED,
    ChampionDeployments,
    InferenceFailed,
    InvalidInferenceRequest,
)
from experiments.jobs import (
    ExperimentJobDefinition,
    ExperimentJobs,
    HeldoutBinding,
    JobBinding,
    JobWorker,
)
from experiments.monitoring import (
    FAIR_STRATEGIES,
    DriftConfig,
    FeedbackAlreadyRecorded,
    FeedbackMismatch,
    FeedbackRequest,
    InvalidFeedback,
    MonitoringIntegrityError,
    MonitoringPolicy,
    ProductionMonitor,
    TriggerRules,
    _summarize,
    assign_new_rows,
    ks_statistic,
    percentile,
    total_variation,
)
from experiments.optimization_experiment import Strategy
from experiments.promotion import (
    ChampionNotFound,
    ChampionPromotions,
    Decision,
    lineage_identity,
)
from ingestion.service import DatasetService, NewProject, RegisterDataset
from store.blobs import LocalBlobStore
from store.champions import SQLiteChampionStore
from store.datasets import SQLiteDatasetRepository
from store.deployments import SQLiteDeploymentStore, VersionState
from store.jobs import JobState, SQLiteJobStore
from store.monitoring import ReoptState, SQLiteMonitoringStore, TriggerOutcome
from tests.test_champion_inference import (
    ENTRY,
    FIRST,
    INPUTS,
    LINEAGE,
    SECOND,
    Env,
    ModelRuntime,
    PassageDouble,
    definition,
    maf,
    real_versions,
    win,
)
from tests.test_champion_promotion import EVERYTHING, SYNTHETIC_VERSION, Rigged
from tests.test_contract_runtime import DATA
from tests.test_durable_jobs import LEASE

pytestmark = maf  # every scenario serves real inferences through the compiler / MAF path

ALWAYS = (0.0, "2000-01-01T00:00:00+00:00", "2100-01-01T00:00:00+00:00")
SINCE, UNTIL = ALWAYS[1], ALWAYS[2]
SMALL = MonitoringPolicy(
    name="test-small",
    drift=DriftConfig(min_samples=4, min_labelled=4),
    trigger=TriggerRules(
        min_inferences=4,
        min_labelled=4,
        min_quality=0.7,
        max_failure_rate=0.5,
        on_material_drift=False,
        on_quality_drift=False,
        min_new_labelled_examples=2,
    ),
)
COUNTRIES = [
    ("Peru", "Lima"),
    ("Chile", "Santiago"),
    ("Kenya", "Nairobi"),
    ("Egypt", "Cairo"),
    ("Norway", "Oslo"),
    ("Greece", "Athens"),
    ("Cuba", "Havana"),
    ("Nepal", "Kathmandu"),
]


def inputs_for(country: str, city: str) -> dict[str, str]:
    return {
        "question": f"Which city is the capital of {country}?",
        "passage": f"{city} is the capital of {country}. It is a large city.",
    }


# -- test doubles -------------------------------------------------------------------------------
class FlakyDouble(PassageDouble):
    """TEST DOUBLE: ``PassageModel`` that raises for a passage mentioning Atlantis."""

    async def generate(self, request):
        if "Atlantis" in request.input_text:
            raise RuntimeError("test double outage")
        return await super().generate(request)


class AnyStamped(Rigged):
    """``Stamped`` for any dataset version of the capitals contract: every run carries the
    runtime versions the real ``WorkflowRunner`` reports for ``ENTRY`` (they do not depend on
    which rows the dataset version holds)."""

    def __init__(self, passes, **kw) -> None:
        super().__init__(passes, **kw)
        inner = self._evaluate

        def evaluate(genome, task, trial, seed):
            run = inner(genome, task, trial, seed)
            versions = run.execution.key.versions.model_validate(_real_versions())
            key = run.execution.key.model_copy(update={"versions": versions})
            execution = run.execution.model_copy(update={"key": key})
            return run.model_copy(update={"execution": execution})

        self._evaluate = evaluate


@cache
def _real_versions() -> dict[str, Any]:
    return real_versions(definition().contract)


class DataRuntime(ModelRuntime):
    """TEST DOUBLE runtime binding ANY registered dataset version by its content hash, so a
    re-optimization's new dataset version runs through the same #24 / #25 machinery."""

    def __init__(self, backend, blobs: LocalBlobStore) -> None:
        super().__init__(backend)
        self.blobs = blobs

    def _data(self, definition: ExperimentJobDefinition) -> bytes:
        return self.blobs.get(definition.contract.dataset.content_hash)

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        suite, _ = contract_suite(definition.contract, definition.splits, self._data(definition))
        return JobBinding(
            suite=suite,
            evaluate=self.backend,
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
            pricing=self.pricing,
            models=AllowedModels((ENTRY,)),
            replay_proof=self.proof,
        )

    def bind_heldout(self, definition: ExperimentJobDefinition) -> HeldoutBinding:
        suite, _ = contract_suite(definition.contract, definition.splits, self._data(definition))
        tasks = suite.tasks_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)
        return HeldoutBinding(tasks, self.backend, SYNTHETIC_VERSION, self.proof)


class World(Env):
    """``Env`` (#24 jobs, #25 promotions, #30 deployments) + the dataset store holding the
    champion's dataset version + the #31 monitor, all over one data directory."""

    def __init__(self, root: Path, backend=None, *, name: str = "p", owner: str = "m-1") -> None:
        super().__init__(root, backend or AnyStamped(EVERYTHING), name=name)
        self.model = FlakyDouble()
        self.blobs = LocalBlobStore(root / "blobs")
        self.datasets = DatasetService(
            SQLiteDatasetRepository(root / "metadata.sqlite3"), self.blobs
        )
        self.runtime = DataRuntime(self.backend, self.blobs)
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
        self.deployments = ChampionDeployments(
            SQLiteDeploymentStore(root / "deployments.sqlite3"),
            self.promotions,
            self.inference_runtime,
        )
        self.register_capitals()
        self.monitor = self.make_monitor(owner)

    def make_monitor(self, owner: str) -> ProductionMonitor:
        return ProductionMonitor(
            self.deployments,
            SQLiteMonitoringStore(self.root / "monitoring.sqlite3"),
            self.jobs,
            self.datasets,
            owner=owner,
        )

    def register_capitals(self) -> None:
        repo = self.datasets.repo
        if repo.list_versions("capitals"):
            return
        project = self.datasets.create_project(NewProject(name="capitals"))
        upload, _ = self.datasets.upload(project.project_id, DATA, "jsonl")
        record, _ = self.datasets.register(
            upload.upload_id,
            RegisterDataset(
                dataset_id="capitals",
                name="Capitals",
                input_columns=("question",),
                context_columns=("passage",),
                target_columns=("answer",),
                row_ids="column",
                id_column="qid",
            ),
        )
        assert record.identity_hash == definition().contract.dataset.identity_hash

    def serve(
        self, version: str | None = None, strategies=(Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
    ) -> str:
        if version is None:
            version = self.publish(self.champion(definition(strategies)))
            self.deploy(version)
        return version

    def ask(self, version: str, country: str, city: str) -> tuple[str, dict | None]:
        """One real inference of exactly ``version``: ``(inference_id, output)``."""
        try:
            view = self.deployments.invoke(version, inputs_for(country, city))
        except InferenceFailed as exc:
            return str(exc.details["inference_id"]), None
        return view.inference_id, view.output

    def label(self, version: str, n: int, *, wrong: bool, actor: str = "labeler") -> list[str]:
        """``n`` served + labelled inferences (``wrong``: the label disagrees with the
        answer, so the pinned exact-match evaluator fails it)."""
        out = []
        for country, city in COUNTRIES[:n]:
            inference_id, output = self.ask(version, country, city)
            expected = {"answer": "Nowhere"} if wrong else dict(output or {"answer": city})
            view, _ = self.monitor.submit_feedback(
                inference_id,
                FeedbackRequest(
                    inputs=inputs_for(country, city),
                    output=output,
                    expected=expected,
                    actor=actor,
                    source={"channel": "test"},
                ),
            )
            out.append(view.feedback_id)
        return out

    def label_rest(self, version: str, n: int, *, actor: str) -> list[str]:
        out = []
        for country, city in COUNTRIES[len(COUNTRIES) - n :]:
            inference_id, output = self.ask(version, country, city)
            view, _ = self.monitor.submit_feedback(
                inference_id,
                FeedbackRequest(
                    inputs=inputs_for(country, city),
                    output=output,
                    expected={"answer": "Nowhere"},
                    actor=actor,
                ),
            )
            out.append(view.feedback_id)
        return out

    def policy(self, policy: MonitoringPolicy = SMALL) -> str:
        return self.monitor.register_policy(policy)[1]

    def evaluate(self, version: str, policy_id: str, monitor: ProductionMonitor | None = None):
        return (monitor or self.monitor).evaluate(version, SINCE, UNTIL, policy_id)

    def rows(self, db: str, sql: str) -> list[tuple]:
        conn = sqlite3.connect(self.root / db)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()


def snapshot(world: World) -> dict[str, Any]:
    """Everything a trigger must never change."""
    return {
        "datasets_v1": world.rows(
            "metadata.sqlite3", "SELECT * FROM dataset_versions WHERE dataset_version=1"
        ),
        "splits_v1": world.rows(
            "metadata.sqlite3", "SELECT * FROM dataset_splits WHERE dataset_version=1"
        ),
        "blob_v1": hashlib.sha256(
            world.blobs.get(definition().contract.dataset.content_hash)
        ).hexdigest(),
        "jobs": world.rows(
            "jobs.sqlite3",
            "SELECT job_id, state, definition_hash, definition_json, identity_json, artifact_json "
            "FROM experiment_jobs ORDER BY job_id",
        ),
        "attempts": world.rows("jobs.sqlite3", "SELECT * FROM experiment_attempts ORDER BY 1"),
        "checkpoints": world.rows(
            "jobs.sqlite3", "SELECT * FROM experiment_checkpoints ORDER BY 1"
        ),
        "champions": world.rows("champions.sqlite3", "SELECT * FROM champions ORDER BY 1"),
        "promotions": world.rows("champions.sqlite3", "SELECT * FROM promotions ORDER BY 1"),
        "versions": world.rows("deployments.sqlite3", "SELECT * FROM workflow_versions"),
        "states": world.rows("deployments.sqlite3", "SELECT * FROM version_states ORDER BY 1"),
        "deployments": world.rows("deployments.sqlite3", "SELECT * FROM deployments"),
        "inferences": world.rows(
            "deployments.sqlite3", "SELECT * FROM inference_records ORDER BY 1"
        ),
    }


def _drop(snap: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in snap.items() if k not in keys}


# -- 0. statistics are transparent and deterministic -------------------------------------------
def test_statistics_are_plain_deterministic_formulas():
    assert percentile([5, 1, 3, 2, 4], 0.5) == 3 and percentile([5, 1, 3, 2, 4], 0.95) == 5
    assert percentile([], 0.5) is None
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    assert ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert total_variation(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 0.0
    assert total_variation(["a", "a"], ["c", "c"]) == 1.0  # an unseen category is all the mass
    assert total_variation(["a", "b"], ["a", "a"]) == pytest.approx(0.5)
    assert MonitoringPolicy().policy_id == MonitoringPolicy().policy_id
    assert MonitoringPolicy().policy_id != SMALL.policy_id
    with pytest.raises(ValueError):  # a single sample can never be "sufficient evidence"
        TriggerRules(min_inferences=1)
    with pytest.raises(ValueError):
        TriggerRules(min_labelled=1)
    with pytest.raises(ValueError):
        DriftConfig(min_samples=1)


# -- 1. telemetry is the #30 record, tied to the exact version + provenance ---------------------
def test_monitoring_reads_inference_records_tied_to_the_exact_version(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    ids = [world.ask(version, c, city)[0] for c, city in COUNTRIES[:3]]
    doc = world.deployments.get(version).document
    s = world.monitor.summary(version, SINCE, UNTIL)

    assert s.evidence["inference_ids"] == sorted(ids)
    assert s.pins["artifact_id"] == doc["provenance"]["artifact_id"]
    assert s.pins["provenance_id"] == doc["provenance"]["provenance_id"]
    assert s.pins["model_hash"] == doc["model"]["model_hash"] == ENTRY.model_hash
    assert s.pins["champion_id"] == doc["champion"]["champion_id"]
    records = [world.deployments.inference(i) for i in ids]
    for r in records:
        assert r.workflow_version == version
        assert r.provenance["artifact_id"] == s.pins["artifact_id"]
    assert s.inferences["count"] == 3 and s.inferences["failure_rate"] == 0.0
    assert s.tokens["prompt_tokens"]["total"] == sum(r.usage.prompt_tokens for r in records)
    assert s.tokens["total_tokens"]["total"] == sum(r.usage.total_tokens for r in records)
    assert s.model_calls["total"] == sum(r.usage.model_calls for r in records)
    assert s.cost["authoritative"] is True  # ENTRY prices the model
    assert s.cost["total"] == pytest.approx(sum(r.usage.cost for r in records), abs=1e-6)
    lat = sorted(r.usage.latency_s for r in records)
    assert s.latency_s["p50"] == pytest.approx(lat[1], abs=1e-6)
    assert s.latency_s["p95"] == pytest.approx(lat[2], abs=1e-6)
    # no competing telemetry: the monitoring store has no usage / latency table of its own
    tables = {r[0] for r in world.rows("monitoring.sqlite3", "SELECT name FROM sqlite_master")}
    assert not any("inference" in t or "telemetry" in t for t in tables)


def test_failures_come_from_the_inference_records(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    world.ask(version, "Peru", "Lima")
    failed, output = world.ask(version, "Atlantis", "Poseidonis")
    assert output is None
    s = world.monitor.summary(version, SINCE, UNTIL)
    assert world.deployments.inference(failed).status == "FAILED"
    assert (s.inferences["count"], s.inferences["failed"]) == (2, 1)
    assert s.inferences["failure_rate"] == 0.5
    assert sum(s.inferences["failure_kinds"].values()) == 1


# -- 2. feedback binds to one inference record and never alters it ------------------------------
def test_feedback_is_bound_to_its_record_and_cannot_alter_it(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    inference_id, output = world.ask(version, "Peru", "Lima")
    before = world.rows("deployments.sqlite3", "SELECT * FROM inference_records")
    good = FeedbackRequest(inputs=inputs_for("Peru", "Lima"), output=output, expected=dict(output))

    with pytest.raises(FeedbackMismatch):  # not the request that was served
        world.monitor.submit_feedback(
            inference_id, good.model_copy(update={"inputs": inputs_for("Chile", "Santiago")})
        )
    with pytest.raises(FeedbackMismatch):  # not the answer that was returned
        world.monitor.submit_feedback(
            inference_id, good.model_copy(update={"output": {"answer": "Cusco"}})
        )
    view, created = world.monitor.submit_feedback(inference_id, good)
    assert created and view.labelled and view.evaluation["passed"] is True
    assert view.workflow_version == version
    assert (
        view.lineage["artifact_id"]
        == world.deployments.get(version).document["provenance"]["artifact_id"]
    )
    assert view.lineage["dataset_id"] == "capitals" and view.lineage["dataset_version"] == 1
    again, created = world.monitor.submit_feedback(inference_id, good)
    assert not created and again == view
    with pytest.raises(FeedbackAlreadyRecorded):  # a different label is a conflict, never an edit
        world.monitor.submit_feedback(
            inference_id, good.model_copy(update={"expected": {"answer": "Cusco"}})
        )
    assert world.rows("deployments.sqlite3", "SELECT * FROM inference_records") == before
    conn = sqlite3.connect(tmp_path / "monitoring.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE feedback SET record_json='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM feedback")
    conn.close()
    conn = sqlite3.connect(tmp_path / "deployments.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE inference_records SET status='FAILED'")
    conn.close()


def test_actor_metadata_is_untrusted_and_decides_nothing(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    fids = world.label(version, 2, wrong=True, actor="alice@example.com")
    fids += world.label_rest(version, 2, actor="root")
    view = world.monitor.feedback(fids[0])
    assert view.metadata_trusted is False
    assert view.untrusted_metadata == {"actor": "alice@example.com", "source": {"channel": "test"}}
    # the feedback identity is (inference, label): no actor, so a claimed identity cannot fork it
    inference_id = view.inference_id
    req = FeedbackRequest(
        inputs=inputs_for(*COUNTRIES[0]),
        output=json.loads(world.monitor.store.feedback(fids[0]).record_json)["output"],
        expected={"answer": "Nowhere"},
        actor="mallory",
        source={"channel": "spoofed"},
    )
    again, created = world.monitor.submit_feedback(inference_id, req)
    assert not created and again.feedback_id == fids[0]
    assert again.untrusted_metadata["actor"] == "alice@example.com"  # never overwritten
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.TRIGGERED
    text = json.dumps(d.model_dump(mode="json"))
    for claimed in ("alice@example.com", "root", "mallory", "spoofed", "untrusted_metadata"):
        assert claimed not in text  # no decision, aggregate or evidence reads it
    summary = world.monitor.summary(version, SINCE, UNTIL, pid).model_dump_json()
    assert "alice" not in summary and "root" not in summary


def _ids(world: World):
    counter = iter(range(10_000))
    return lambda: f"inf-{next(counter):024x}"


def _strip(evidence: dict) -> dict:
    out = json.loads(json.dumps(evidence))
    out["summary"].pop("window", None)
    return out


# -- 3. aggregates never mix versions ------------------------------------------------------------
def test_aggregates_never_mix_workflow_versions(tmp_path):
    world = World(tmp_path, AnyStamped(lambda g, row: g in (win()["fixed"], win()["random"])))
    v1 = world.serve(strategies=FIRST)
    v2 = world.publish(world.champion(definition(SECOND)))
    world.deployments.stage(v2, world.revision())  # v1 serves production, v2 staging
    assert world.deployments.get(v2).state is VersionState.STAGING
    one = [world.ask(v1, c, city)[0] for c, city in COUNTRIES[:3]]
    two = [world.ask(v2, c, city)[0] for c, city in COUNTRIES[3:5]]
    s1 = world.monitor.summary(v1, SINCE, UNTIL)
    s2 = world.monitor.summary(v2, SINCE, UNTIL)
    assert s1.evidence["inference_ids"] == sorted(one)
    assert s2.evidence["inference_ids"] == sorted(two)
    assert s1.inferences["count"] == 3 and s2.inferences["count"] == 2
    assert s1.pins["champion_id"] != s2.pins["champion_id"]
    # an aggregate handed another version's record refuses to compute (never averages across)
    ctx = world.monitor._context(v1)
    foreign = json.loads(world.deployments.store.inference(two[0]).record_json)
    with pytest.raises(MonitoringIntegrityError):
        _summarize(ctx, *_bounds(), [foreign], [])


def _bounds():
    from experiments.monitoring import _when

    return _when(SINCE), _when(UNTIL)


# -- 4. thresholds + minimum sample gate ---------------------------------------------------------
def test_one_bad_request_never_triggers_and_the_minimum_gate_holds(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 1, wrong=True)  # one terrible, labelled request
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.NOT_TRIGGERED and d.reasons == []
    quality = next(c for c in d.checks if c["rule"] == "quality_below_threshold")
    assert quality["status"] == "INSUFFICIENT_EVIDENCE" and quality["samples"] == 1
    failure = next(c for c in d.checks if c["rule"] == "failure_rate_above_threshold")
    assert failure["status"] == "INSUFFICIENT_EVIDENCE"
    assert world.monitor.store.reoptimization(d.trigger_id) is None
    assert len(world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 1
    # the default (production) policy needs 20 labels / 30 inferences
    d = world.monitor.evaluate(version, SINCE, UNTIL)
    assert d.outcome is TriggerOutcome.NOT_TRIGGERED


def test_quality_threshold_is_deterministic_and_fires_on_enough_evidence(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 3, wrong=True)
    assert world.evaluate(version, pid).outcome is TriggerOutcome.NOT_TRIGGERED  # 3 < 4
    world.label_more = world.label  # noqa: B010
    fid = _label_one(world, version, COUNTRIES[3], wrong=True)
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.TRIGGERED, d.checks
    (reason,) = [r for r in d.reasons if r["code"] == "quality_below_threshold"]
    assert reason["observed"] == 0.0 and reason["threshold"] == 0.7 and reason["samples"] == 4
    assert fid in d.frozen_feedback_ids and len(d.frozen_feedback_ids) == 4
    # the same evidence, decided again: the same bytes (deterministic), and nothing new
    drift_a = world.monitor.drift(version, SINCE, UNTIL, pid)
    drift_b = world.make_monitor("m-2").drift(version, SINCE, UNTIL, pid)
    assert drift_a.model_dump_json() == drift_b.model_dump_json()


def _label_one(world: World, version: str, pair, *, wrong: bool) -> str:
    country, city = pair
    inference_id, output = world.ask(version, country, city)
    view, _ = world.monitor.submit_feedback(
        inference_id,
        FeedbackRequest(
            inputs=inputs_for(country, city),
            output=output,
            expected={"answer": "Nowhere"} if wrong else dict(output or {"answer": city}),
        ),
    )
    return view.feedback_id


def test_failure_rate_rule_needs_its_minimum_too(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy(
        SMALL.model_copy(update={"trigger": SMALL.trigger.model_copy(update={"min_quality": None})})
    )
    for _ in range(3):
        world.ask(version, "Atlantis", "Poseidonis")
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.NOT_TRIGGERED  # 3 failures < 4 inferences
    world.ask(version, "Atlantis", "Poseidonis")
    d = world.evaluate(version, pid)
    # the rule fires, but there are no new labelled examples to re-optimize on: SUPPRESSED
    assert d.outcome is TriggerOutcome.SUPPRESSED
    codes = [r["code"] for r in d.reasons]
    assert "failure_rate_above_threshold" in codes
    assert "insufficient_new_labelled_examples" in codes


def test_drift_statistics_against_the_champions_dataset(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=False)
    report = world.monitor.drift(version, SINCE, UNTIL, pid)
    assert report.reference["dataset_version"] == 1
    assert report.reference["splits"] == ["optimization", "validation"]
    assert report.reference["rows"] == 5  # q1..q5: the test row q6 stays sealed
    assert report.reference["champion_runs"] > 0
    by_name = {f["feature"]: f for f in report.features}
    assert by_name["prompt_tokens"]["status"] in ("STABLE", "DRIFTED")
    assert by_name["prompt_tokens"]["value"] is not None
    assert by_name["question.type"]["status"] == "STABLE"
    assert by_name["input.length"]["observed_n"] == 4
    assert report.quality["observed_quality"] == 1.0
    assert report.material == (len(report.drifted_features) >= 1)


# -- 5. re-optimization: a NEW dataset version + a NEW challenger; nothing old mutated ------------
def test_a_trigger_creates_a_new_version_and_challenger_and_mutates_nothing(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 3, wrong=True)
    before = snapshot(world)
    _label_one(world, version, COUNTRIES[3], wrong=True)
    mid = snapshot(world)
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.TRIGGERED
    after = snapshot(world)

    # the parent dataset version, its splits and bytes, the old experiment (definition,
    # identity, artifact, attempts, checkpoints), champion, promotion, workflow version,
    # deployment and every inference record are byte-for-byte what they were
    assert _drop(after, "jobs") == _drop(mid, "jobs")
    assert after["jobs"][: len(mid["jobs"])] == mid["jobs"] or set(mid["jobs"]) <= set(
        after["jobs"]
    )
    assert _drop(before, "inferences") == _drop(mid, "inferences")
    view = world.monitor.challenger(d.trigger_id)
    assert view.state is ReoptState.JOB_CREATED
    assert view.dataset["parent"]["dataset_version"] == 1
    assert view.dataset["new"]["dataset_version"] == 2
    versions = world.datasets.get_dataset("capitals")
    assert [v.dataset_version for v in versions] == [1, 2]
    new_ids = world.datasets.repo.get_row_ids("capitals", 2)
    assert set(world.datasets.repo.get_row_ids("capitals", 1)) <= set(new_ids)
    assert sorted(set(new_ids) - {f"q{i}" for i in range(1, 7)}) == sorted(d.frozen_feedback_ids)
    job = world.jobs.get(view.challenger_job_id)
    assert job.state is JobState.PENDING  # created, not run, not promoted, not deployed
    job_row = world.jobs.store.get_job(view.challenger_job_id)
    challenger = ExperimentJobDefinition.model_validate_json(job_row.definition_json)
    link = challenger.provenance["reoptimization"]
    assert link["trigger_id"] == d.trigger_id and link["workflow_version"] == version
    assert link["artifact_id"] == d.links["artifact_id"]
    assert link["provenance_id"] == d.links["provenance_id"]
    assert link["source_job_id"] == d.links["source_job_id"]
    assert challenger.contract.dataset.dataset_version == 2
    source = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(d.links["source_job_id"]).definition_json
    )
    # same TaskContract / evaluator / model authority, re-bound to the new version only
    strip = {"dataset"}
    assert challenger.contract.model_dump(exclude=strip) == source.contract.model_dump(
        exclude=strip
    )
    assert challenger.plan.model == source.plan.model
    assert challenger.plan.expected_model_hash == source.plan.expected_model_hash


def test_new_splits_keep_every_parent_row_in_its_role_and_tests_isolated(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    # one label duplicates a PARENT row's inputs (q1, France): it must not enter the new version
    inference_id, output = world.ask(version, "France", "Paris")
    world.monitor.submit_feedback(
        inference_id,
        FeedbackRequest(
            inputs=inputs_for("France", "Paris"), output=output, expected={"answer": "Nowhere"}
        ),
    )
    world.label(version, 5, wrong=True)
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.TRIGGERED
    view = world.monitor.challenger(d.trigger_id)
    excluded = view.dataset["excluded"]
    assert list(excluded.values()) == ["duplicate_of_parent_row"]

    job = world.jobs.store.get_job(view.challenger_job_id)
    new = ExperimentJobDefinition.model_validate_json(job.definition_json).splits
    old = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(d.links["source_job_id"]).definition_json
    ).splits
    for split in old.splits:  # every parent row keeps its parent role
        for row in split.row_ids:
            assert new.role_of(row) is split.role
    parent_test = set(old.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE))
    new_test = set(new.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE))
    view_opt = new.optimizer_view()
    assert parent_test <= new_test
    assert not new_test & set(view_opt.optimization_row_ids)
    assert not new_test & set(view_opt.validation_row_ids)
    added = sorted(set(view.dataset["kept_feedback_ids"]))
    expected = assign_new_rows(added, SMALL.new_row_splits)
    assert view.dataset["new_rows"] == {r.value: list(ids) for r, ids in sorted(expected.items())}
    for role, ids in expected.items():
        assert set(ids) <= set(new.split(role).row_ids)
    # the new splits are stored with the new version, never with the parent
    stored = world.datasets.list_splits("capitals", 2)
    assert [s.splits_hash for s in stored] == [view.dataset["new"]["splits_hash"]]


# -- 6. idempotency + concurrency ----------------------------------------------------------------
def test_the_same_evidence_never_triggers_twice(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    first = world.evaluate(version, pid)
    again = world.evaluate(version, pid)
    other_window = world.monitor.evaluate(
        version, "1999-01-01T00:00:00+00:00", "2200-01-01T00:00:00+00:00", pid
    )
    assert first.created and not again.created and not other_window.created
    assert first.trigger_id == again.trigger_id == other_window.trigger_id
    assert world.monitor.reoptimize(first.trigger_id).challenger_job_id == (
        world.monitor.challenger(first.trigger_id).challenger_job_id
    )
    # new evidence inside the cooldown: decided, but SUPPRESSED (one open trigger per version)
    world.label_extra = None
    _label_one(world, version, COUNTRIES[5], wrong=True)
    _label_one(world, version, COUNTRIES[6], wrong=True)
    later = world.evaluate(version, pid)
    assert later.trigger_id != first.trigger_id
    assert later.outcome is TriggerOutcome.SUPPRESSED
    assert "cooldown_open_trigger" in [r["code"] for r in later.reasons]
    jobs = world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")
    assert len(jobs) == 2  # the champion's experiment + exactly one challenger
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == [1, 2]
    triggered = [
        h for h in world.monitor.history(version).decisions if h.outcome is TriggerOutcome.TRIGGERED
    ]
    assert [h.trigger_id for h in triggered] == [first.trigger_id]


def test_concurrent_monitors_create_one_trigger_one_dataset_one_job(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    monitors = [world.make_monitor(f"m-{i}") for i in range(6)]
    barrier = threading.Barrier(len(monitors))
    results: list[Any] = []
    errors: list[BaseException] = []

    def run(m: ProductionMonitor) -> None:
        try:
            barrier.wait()
            results.append(world.evaluate(version, pid, m))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(m,)) for m in monitors]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len({r.trigger_id for r in results}) == 1
    assert sum(r.created for r in results) == 1
    tid = results[0].trigger_id
    for m in monitors:  # stragglers finish (or find finished) the same re-optimization
        m.reoptimize(tid)
    jobs = world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")
    assert len(jobs) == 2
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == [1, 2]
    assert len(world.datasets.list_splits("capitals", 2)) == 1
    assert world.monitor.challenger(tid).state is ReoptState.JOB_CREATED


def test_a_crashed_reoptimization_resumes_without_a_second_dataset_or_job(tmp_path, monkeypatch):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)

    class Crash(BaseException):
        pass

    def boom(self, definition):
        raise Crash()

    monkeypatch.setattr(ExperimentJobs, "create", boom)
    with pytest.raises(Crash):
        world.evaluate(version, pid)  # dies after the dataset version, before the job
    monkeypatch.undo()
    (decision,) = world.monitor.history(version).decisions
    row = world.monitor.store.reoptimization(decision.trigger_id)
    assert row.state is ReoptState.DATASET_CREATED and row.job_id is None
    # the dead monitor's lease still holds: another monitor does not take over early
    other = world.make_monitor("m-other")
    assert other.reoptimize(decision.trigger_id).state is ReoptState.DATASET_CREATED
    other.clock = lambda: 10**12  # ... until it expires
    view = other.reoptimize(decision.trigger_id)
    assert view.state is ReoptState.JOB_CREATED
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == [1, 2]
    assert len(world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 2
    conn = sqlite3.connect(tmp_path / "monitoring.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):  # progress never moves backwards
        conn.execute("UPDATE reoptimizations SET state='CLAIMED'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE trigger_decisions SET outcome='NOT_TRIGGERED'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM frozen_examples")
    conn.close()


def test_a_crash_after_the_job_was_created_never_creates_a_second_job(tmp_path, monkeypatch):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    real = SQLiteMonitoringStore.advance_reoptimization

    class Crash(BaseException):
        pass

    def crash_on_job(self, trigger_id, owner, fence, state, **fields):
        if state is ReoptState.JOB_CREATED:
            raise Crash()  # the job exists, the claim does not know it yet
        return real(self, trigger_id, owner, fence, state, **fields)

    monkeypatch.setattr(SQLiteMonitoringStore, "advance_reoptimization", crash_on_job)
    with pytest.raises(Crash):
        world.evaluate(version, pid)
    monkeypatch.undo()
    assert len(world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 2
    (decision,) = world.monitor.history(version).decisions
    other = world.make_monitor("m-other")
    other.clock = lambda: 10**12  # the crashed monitor's lease expired
    view = other.reoptimize(decision.trigger_id)
    assert view.state is ReoptState.JOB_CREATED
    jobs = world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")
    assert len(jobs) == 2 and (view.challenger_job_id,) in jobs  # the same job, adopted
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == [1, 2]


# -- 7. fair challenger, no deployment, #25 still required ---------------------------------------
def test_the_challenger_runs_a_fair_fixed_random_aco_comparison(tmp_path):
    world = World(tmp_path)
    version = world.serve(strategies=FIRST)  # even a champion found by Fixed alone ...
    pid = world.policy()
    world.label(version, 4, wrong=True)
    d = world.evaluate(version, pid)
    view = world.monitor.challenger(d.trigger_id)
    assert view.strategies == [s.value for s in FAIR_STRATEGIES]  # ... is challenged by all 3
    job_id = view.challenger_job_id
    challenger = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(job_id).definition_json
    )
    source = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(d.links["source_job_id"]).definition_json
    )
    same = ("model", "budget", "seeds", "trials", "batch_size", "lcb_z", "fixed_rule")
    for name in same:  # one protocol for every strategy: no strategy gets more budget
        assert getattr(challenger.plan, name) == getattr(source.plan, name), name
    world.worker.run_until_idle(job_id)
    job = world.jobs.get(job_id)
    assert job.state is JobState.COMPLETED
    assert sorted({u.strategy for u in job.units}) == sorted(s.value for s in FAIR_STRATEGIES)
    budgets = {u.strategy: u.usage for u in job.units}
    assert len(budgets) == 3


def test_a_trigger_deploys_nothing_and_the_challenger_still_needs_promotion(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    deployment = world.deployments.deployment(LINEAGE)
    champion = world.promotions.store.current(LINEAGE)
    d = world.evaluate(version, pid)
    job_id = world.monitor.challenger(d.trigger_id).challenger_job_id
    world.worker.run_until_idle(job_id)  # even a COMPLETED challenger ...

    assert world.deployments.deployment(LINEAGE) == deployment  # ... changes no deployment
    assert world.promotions.store.current(LINEAGE) == champion  # ... and no champion
    assert world.monitor.challenger(d.trigger_id).deployed is False
    assert world.promotions.store.for_job(job_id) is None  # nothing promoted it
    assert len(world.deployments.list().versions) == 1
    with pytest.raises(ChampionNotFound):  # #30 publishes only a #25 champion
        world.deployments.publish("c-" + "0" * 24)
    # the explicit #25 gate is the only way forward
    promoted = world.promotions.promote(job_id, "capitals.reopt")
    assert promoted.decision is Decision.PROMOTED
    new_version, _ = world.deployments.publish(promoted.record["champion"]["champion_id"])
    assert new_version.state is VersionState.CREATED  # and #30 stage / promote stay explicit
    assert world.deployments.deployment(LINEAGE).production_version_id == version


# -- 8. restart ---------------------------------------------------------------------------------
def test_restart_reproduces_monitoring_and_trigger_state(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    d = world.evaluate(version, pid)
    summary = world.monitor.summary(version, SINCE, UNTIL, pid)
    drift = world.monitor.drift(version, SINCE, UNTIL, pid)
    history = world.monitor.history(version)
    challenger = world.monitor.challenger(d.trigger_id)

    restarted = World(tmp_path, owner="m-restarted")  # a fresh process over the same files
    m = restarted.monitor
    assert m.summary(version, SINCE, UNTIL, pid) == summary
    assert m.drift(version, SINCE, UNTIL, pid).model_dump_json() == drift.model_dump_json()
    assert m.history(version) == history
    assert m.challenger(d.trigger_id) == challenger
    again = restarted.evaluate(version, pid)
    assert again.trigger_id == d.trigger_id and not again.created
    assert len(restarted.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 2


# -- 9. API --------------------------------------------------------------------------------------
def call(api: ProductAPI, method: str, path: str, body: Any = None):
    raw = b"" if body is None else json.dumps(body).encode()
    headers = {"content-type": "application/json", "content-length": str(len(raw))}
    response = api.handle(method, "/api/v1/" + path, headers, io.BytesIO(raw))
    return response.status, response.body


def test_the_api_exposes_feedback_summary_drift_triggers_and_challenger(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    api = ProductAPI(world.datasets, world.jobs, world.promotions, world.deployments, world.monitor)
    status, body = call(api, "POST", "monitoring/policies", SMALL.model_dump(mode="json"))
    assert status == 201 and body["policy_id"] == SMALL.policy_id
    assert call(api, "GET", f"monitoring/policies/{SMALL.policy_id}")[0] == 200
    for country, city in COUNTRIES[:4]:
        inference_id, output = world.ask(version, country, city)
        status, fb = call(
            api,
            "POST",
            f"inferences/{inference_id}/feedback",
            {
                "inputs": inputs_for(country, city),
                "output": output,
                "expected": {"answer": "Nowhere"},
                "actor": "ops",
            },
        )
        assert status == 201 and fb["metadata_trusted"] is False
        assert call(api, "GET", f"feedback/{fb['feedback_id']}")[1] == fb
    status, fb = call(
        api,
        "POST",
        f"inferences/{inference_id}/feedback",
        {"inputs": INPUTS, "output": output, "expected": {"answer": "x"}},
    )
    assert status == 422 and fb["error"]["code"] == "feedback_mismatch"
    q = f"since={SINCE.replace('+', '%2B')}&until={UNTIL.replace('+', '%2B')}"
    status, summary = call(api, "GET", f"workflows/{version}/monitoring?{q}")
    assert status == 200 and summary["inferences"]["count"] == 4
    status, drift = call(api, "GET", f"workflows/{version}/drift?{q}&policy_id={SMALL.policy_id}")
    assert status == 200 and drift["record_schema"] == "wynk-drift-report/1"
    status, d = call(
        api,
        "POST",
        f"workflows/{version}/triggers/evaluate",
        {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id},
    )
    assert status == 201 and d["outcome"] == "TRIGGERED"
    status, again = call(
        api,
        "POST",
        f"workflows/{version}/triggers/evaluate",
        {"since": SINCE, "until": UNTIL, "policy_id": SMALL.policy_id},
    )
    assert status == 200 and again["trigger_id"] == d["trigger_id"]
    status, history = call(api, "GET", f"workflows/{version}/triggers")
    assert status == 200 and [h["trigger_id"] for h in history["decisions"]] == [d["trigger_id"]]
    assert call(api, "GET", f"triggers/{d['trigger_id']}")[1]["evidence"] == d["evidence"]
    status, ch = call(api, "GET", f"triggers/{d['trigger_id']}/challenger")
    assert status == 200 and ch["state"] == "JOB_CREATED" and ch["deployed"] is False
    assert call(api, "GET", f"experiments/{ch['challenger_job_id']}")[0] == 200
    assert call(api, "POST", f"triggers/{d['trigger_id']}/reoptimize")[1] == ch
    assert call(api, "GET", "triggers/tr-" + "0" * 24)[0] == 404
    assert call(api, "GET", f"workflows/{version}/drift?bogus=1")[0] == 400


# -- 10. end to end: #31 trigger -> #25 cross-version promotion (same lineage) -> #30 explicit ---
PARENT_ROWS = frozenset(f"q{i}" for i in range(1, 7))  # dataset v1 (q6 is its test row)


def _optional_last(schema: AnswerSchema) -> AnswerSchema:
    *head, last = schema.fields
    return AnswerSchema(fields=(*head, last.model_copy(update={"required": False})))


CHANGED_TASK = {
    "input_schema": lambda c: _optional_last(c.input_schema),
    "output_schema": lambda c: _optional_last(c.output_schema),
    "evaluation": lambda c: EvaluationSpec(
        evaluator="exact_match", config={"case_sensitive": True}
    ),
    "objective": lambda c: ObjectiveSpec(
        mode=ObjectiveMode.BALANCED,
        weights={Metric.QUALITY: 0.8, Metric.LATENCY: 0.2},
        scales={Metric.LATENCY: 1.0},
    ),
    "constraints": lambda c: c.constraints.model_copy(update={"maximum_retries": 1}),
    "instructions": lambda c: "Name the capital city.",
}
CHANGED_RUNTIME = {
    "evaluator_run_version": "evaluator-2",
    "grammar_version": "grammar-2",
    "model_hash": "model-2",
    "model_config_hash": "model-config-2",
    "prompt_template_version": "prompt-2",
}


def _legacy_artifact(d: ExperimentJobDefinition, compat: dict[str, Any], **problem) -> dict:
    """The identity block a (pre-provenance) artifact of ``d`` carries, built from a stored
    champion's ``compatibility`` pins: ``lineage_identity`` then reads the same components the
    provenance path read, so the real stored lineage hash can be reproduced and perturbed."""
    c = d.contract
    base = {
        "dataset_id": c.dataset.dataset_id,
        "dataset_version": c.dataset.dataset_version,
        "dataset_hash": c.dataset.identity_hash,
        "dataset_content_hash": c.dataset.content_hash,
        "splits_hash": d.splits.identity_hash,
        "task_id": c.task_id,
        "contract_version": c.contract_version,
        "task_contract_hash": c.contract_hash,
        "objective_hash": compat["objective_hash"],
        "constraints_hash": compat["constraints_hash"],
        "evaluation_hash": compat["evaluation_hash"],
        "evaluator_run_version": compat["evaluator_version"],
        "grammar_version": compat["grammar_version"],
        "model": compat["model"],
        "model_config_hash": compat["model_config_hash"],
        "model_hash": compat["model_hash"],
        "prompt_template_version": compat["prompt_template_version"],
    }
    return {
        "identity": {"problem": base | problem},
        "run_versions": compat["run_versions"],
        "synthetic": compat["synthetic"],
    }


def test_end_to_end_reoptimization_promotes_in_the_same_lineage_on_v2_heldout(tmp_path):
    """champion v1 -> production inference -> labelled feedback -> trigger -> dataset v2 ->
    challenger job (Fixed vs Random vs ACO) -> ONE validation challenger -> #25 promotion in
    the SAME lineage: the v1 incumbent re-runs on the v2 test rows / trials / seeds -> new
    champion -> deployed only through explicit #30 stage / promote (and rollback still works).
    Nothing between the trigger and the deployment is mocked: the real ``ProductionMonitor``,
    ``ExperimentJobs`` / ``JobWorker``, ``ChampionPromotions`` and ``ChampionDeployments``."""
    box: dict[str, str] = {}

    def passes(genome: str, row: str) -> bool:
        # the v1 champion never learned the production rows; every other workflow answers all
        return row in PARENT_ROWS or genome != box.get("incumbent")

    backend = AnyStamped(passes)
    world = World(tmp_path, backend)

    # 1. production champion v1, served by #30 (real compiler -> MAF -> registered model)
    version = world.serve()
    v1 = world.promotions.current(LINEAGE)
    assert v1.version == 1 and v1.record["evaluation_context"]["dataset_version"] == 1
    box["incumbent"] = v1.record["genome_hash"]
    v1_heldout = v1.record["heldout"]
    assert v1_heldout["row_ids"] == ["q6"] and v1_heldout["pass_rate"] == 1.0
    v1_job = v1.record["provenance"]["job_id"]
    deployment = world.deployments.deployment(LINEAGE)
    assert deployment.production_version_id == version

    # 2. production inference + labelled feedback -> 3. the monitoring trigger
    pid = world.policy()
    world.label(version, len(COUNTRIES), wrong=True)
    before = snapshot(world)
    v1_champion_row = world.promotions.store.champion(v1.champion_id)
    d = world.evaluate(version, pid)
    assert d.outcome is TriggerOutcome.TRIGGERED and d.created
    again = world.evaluate(version, pid)  # idempotent: the same evidence, the same trigger
    assert again.trigger_id == d.trigger_id and not again.created

    # 4. an immutable dataset v2 + ONE challenger job; the old version / job are untouched
    view = world.monitor.challenger(d.trigger_id)
    assert view.dataset["new"]["dataset_version"] == 2 and view.deployed is False
    assert view.strategies == [s.value for s in FAIR_STRATEGIES]
    job_id = view.challenger_job_id
    assert world.monitor.reoptimize(d.trigger_id).challenger_job_id == job_id
    challenger_def = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(job_id).definition_json
    )
    splits = challenger_def.splits
    v2_test = set(splits.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE))
    opt_view = splits.optimizer_view()
    v2_opt, v2_val = set(opt_view.optimization_row_ids), set(opt_view.validation_row_ids)
    assert "q6" in v2_test and v2_test - PARENT_ROWS  # parent test row + new production rows
    assert v2_val - PARENT_ROWS  # new rows also reach validation (selection must see them)
    assert not v2_test & (v2_opt | v2_val)

    # 5. Fixed vs Random vs ACO on v2: the v2 test rows never reach optimization or validation
    calls = len(backend.calls)
    world.worker.run_until_idle(job_id)
    job = world.jobs.get(job_id)
    assert job.state is JobState.COMPLETED
    assert sorted({u.strategy for u in job.units}) == sorted(s.value for s in FAIR_STRATEGIES)
    experiment_rows = {k[1] for k in backend.calls[calls:]}
    assert experiment_rows <= v2_opt | v2_val and not experiment_rows & v2_test
    assert world.promotions.store.for_job(job_id) is None  # nothing promoted it by itself
    assert world.promotions.current(LINEAGE).champion_id == v1.champion_id

    # 6. #25 promotion into the SAME (default) lineage: validation picks ONE challenger, the
    # v2 test split opens once, the v1 incumbent is RE-EVALUATED on exactly those rows
    calls = len(backend.calls)
    promoted = world.promotions.promote(job_id)
    rec = promoted.record
    assert promoted.lineage_id == LINEAGE, rec
    assert rec["lineage_identity"]["lineage_hash"] == v1.record["lineage_identity"]["lineage_hash"]
    ctx = rec["evaluation_context"]
    assert (ctx["dataset_id"], ctx["dataset_version"]) == ("capitals", 2)
    assert set(ctx["test_row_ids"]) == v2_test
    assert rec["incumbent"]["champion_id"] == v1.champion_id

    candidates = rec["selection"]["candidates"]
    validated = [c["candidate_id"] for c in candidates if c["status"] == "VALIDATED"]
    assert validated == [rec["selection"]["challenger"]]  # exactly one validation challenger
    challenger_genome = promoted.challenger["genome_hash"]
    assert challenger_genome != box["incumbent"]
    heldout_calls = backend.calls[calls:]
    assert {k[1] for k in heldout_calls} == v2_test  # the gate ran ONLY the v2 test rows
    assert {k[0] for k in heldout_calls} == {challenger_genome, box["incumbent"]}
    runners_up = {c["genome_hash"] for c in candidates} - {challenger_genome, box["incumbent"]}
    assert not runners_up & {k[0] for k in heldout_calls}

    inc, ch = rec["heldout"]["incumbent"], rec["heldout"]["challenger"]
    ran: dict[str, list[tuple]] = {"challenger": [], "incumbent": []}
    for subject, attempt in world.promotions.store.attempts(promoted.promotion_id):
        ran[subject].append((attempt.task_id, attempt.trial, attempt.run_seed))
    assert sorted(ran["incumbent"]) == sorted(ran["challenger"])  # same rows / trials / seeds
    assert sorted(ran["incumbent"]) == sorted(tuple(x) for x in ctx["seeds"])
    assert inc["row_ids"] == ch["row_ids"] and set(inc["row_ids"]) == v2_test
    assert inc["genome_hash"] == box["incumbent"]
    # the v1 held-out score (1.0 on q6) is NOT reused: re-run on v2 it fails the new rows
    new_test = len(v2_test - PARENT_ROWS)
    assert new_test >= 1 and inc["pass_rate"] == pytest.approx(1 - new_test / len(v2_test))
    assert inc != v1_heldout and inc["pass_rate"] < v1_heldout["pass_rate"]
    assert not set(inc["attempt_ids"]) & set(v1_heldout["attempt_ids"])
    assert ch["pass_rate"] == 1.0 and rec["comparison"]["relation"] == "better"

    # 7. the challenger becomes the v2 champion of the same lineage ...
    assert promoted.decision is Decision.PROMOTED
    current = world.promotions.current(LINEAGE)
    assert current.version == 2 and current.record["previous_champion_id"] == v1.champion_id
    assert current.record["compatibility"]["dataset_version"] == 2
    assert current.record["provenance"]["job_id"] == job_id
    world.promotions.verify(promoted.promotion_id)

    # ... the old dataset version, splits, bytes, job, artifact, champion record, workflow
    # version and inference records are unchanged ...
    after = snapshot(world)
    for key in ("datasets_v1", "splits_v1", "blob_v1", "versions", "inferences"):
        assert after[key] == before[key], key
    old_jobs = {row[0]: row for row in before["jobs"]}
    assert {row[0]: row for row in after["jobs"]}[v1_job] == old_jobs[v1_job]
    assert world.promotions.store.champion(v1.champion_id).record_json == (
        v1_champion_row.record_json
    )
    assert world.evaluate(version, pid).trigger_id == d.trigger_id  # still idempotent

    # ... and NOTHING was deployed: production still serves the v1 version
    assert world.deployments.deployment(LINEAGE) == deployment
    assert len(world.deployments.list().versions) == 1
    assert world.monitor.challenger(d.trigger_id).deployed is False

    # 8. #30 stays explicit: publish -> CREATED; stage + promote are required to serve v2
    new_version, _ = world.deployments.publish(current.champion_id)
    assert new_version.state is VersionState.CREATED
    assert world.deployments.deployment(LINEAGE).production_version_id == version
    world.deploy(new_version.version_id)
    assert world.deployments.deployment(LINEAGE).production_version_id == new_version.version_id
    served, output = world.ask(new_version.version_id, *COUNTRIES[0])
    assert output is not None
    world.deployments.rollback(LINEAGE, world.revision())
    assert world.deployments.deployment(LINEAGE).production_version_id == version

    # 9. lineage: v1 and v2 of the unchanged task share it; any semantic change fails closed
    compat = current.record["compatibility"]
    v1_def = ExperimentJobDefinition.model_validate_json(
        world.jobs.store.get_job(v1_job).definition_json
    )
    v1_compat = v1.record["compatibility"]
    same_v1 = lineage_identity(v1_def, _legacy_artifact(v1_def, v1_compat))
    same_v2 = lineage_identity(challenger_def, _legacy_artifact(challenger_def, compat))
    assert (
        same_v1["lineage_hash"]
        == same_v2["lineage_hash"]
        == rec["lineage_identity"]["lineage_hash"]
    )
    for key in ("dataset_version", "dataset_hash", "splits_hash", "test_row_ids"):
        assert key not in same_v2
    stored = rec["lineage_identity"]["lineage_hash"]
    for name, change in CHANGED_TASK.items():
        contract = challenger_def.contract
        changed = contract.model_validate(contract.model_dump() | {name: change(contract)})
        d2 = challenger_def.model_copy(update={"contract": changed})
        problem = {
            "objective_hash": canonical_hash(changed.objective.model_dump(mode="json")),
            "constraints_hash": canonical_hash(changed.constraints.model_dump(mode="json")),
            "evaluation_hash": changed.evaluation.identity_hash,
        }
        assert (
            lineage_identity(d2, _legacy_artifact(d2, compat, **problem))["lineage_hash"] != stored
        ), name
    for name, value in CHANGED_RUNTIME.items():
        artifact = _legacy_artifact(challenger_def, compat, **{name: value})
        assert lineage_identity(challenger_def, artifact)["lineage_hash"] != stored, name
    runtime = _legacy_artifact(challenger_def, compat) | {"run_versions": ["runtime-2"]}
    assert lineage_identity(challenger_def, runtime)["lineage_hash"] != stored


# -- 11. review: a source artifact that is HERE but does not verify fails closed ----------------
def _drop_triggers(conn: sqlite3.Connection) -> None:
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")


def _tamper(world: World, job_id: str, how: str) -> None:
    """Corrupt the champion's stored experiment in place (bypassing its immutability triggers,
    as an attacker or a bad disk would)."""
    conn = sqlite3.connect(world.root / "jobs.sqlite3", isolation_level=None)
    try:
        _drop_triggers(conn)
        statements = {
            "envelope": (
                "UPDATE experiment_artifacts SET envelope_json = "
                "replace(envelope_json, '\"parent\"', '\"parent_\"') WHERE job_id=?"
            ),
            "body": (
                "UPDATE experiment_jobs SET artifact_json = "
                "replace(artifact_json, '\"identity\"', '\"identity_\"') WHERE job_id=?"
            ),
            "identity": (
                "UPDATE experiment_jobs SET identity_json = "
                'replace(identity_json, \'"model_hash":"\', \'"model_hash":"x\') WHERE job_id=?'
            ),
            "not_finalized": "DELETE FROM experiment_artifacts WHERE job_id=?",
            "not_completed": "UPDATE experiment_jobs SET state='RUNNING' WHERE job_id=?",
        }
        assert conn.execute(statements[how], (job_id,)).rowcount == 1
    finally:
        conn.close()


@pytest.mark.parametrize("how", ["envelope", "body", "identity", "not_finalized", "not_completed"])
def test_a_tampered_source_experiment_fails_closed_and_is_never_reported_absent(tmp_path, how):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    source_job = world.deployments.get(version).document["provenance"]["job_id"]
    datasets_before = [v.dataset_version for v in world.datasets.get_dataset("capitals")]
    jobs_before = world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")
    _tamper(world, source_job, how)

    for read in (
        lambda: world.monitor.summary(version, SINCE, UNTIL, pid),
        lambda: world.monitor.drift(version, SINCE, UNTIL, pid),
        lambda: world.evaluate(version, pid),
    ):
        with pytest.raises(MonitoringIntegrityError):
            read()
    # no decision, no dataset version, no challenger job was derived from unverified evidence
    assert world.monitor.history(version).decisions == []
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == datasets_before
    assert world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs") == jobs_before


def test_tampering_after_a_trigger_blocks_its_resumed_reoptimization(tmp_path, monkeypatch):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)

    class Crash(BaseException):
        pass

    def boom(self, trigger_id, owner, fence, state, **fields):
        raise Crash()  # dies right after claiming, before any dataset version

    real = SQLiteMonitoringStore.advance_reoptimization
    monkeypatch.setattr(SQLiteMonitoringStore, "advance_reoptimization", boom)
    with pytest.raises(Crash):
        world.evaluate(version, pid)
    monkeypatch.setattr(SQLiteMonitoringStore, "advance_reoptimization", real)
    (decision,) = world.monitor.history(version).decisions
    versions_before = [v.dataset_version for v in world.datasets.get_dataset("capitals")]
    _tamper(world, world.deployments.get(version).document["provenance"]["job_id"], "envelope")

    other = world.make_monitor("m-other")
    other.clock = lambda: 10**12  # the crashed monitor's lease expired
    with pytest.raises(MonitoringIntegrityError):
        other.reoptimize(decision.trigger_id)
    row = world.monitor.store.reoptimization(decision.trigger_id)
    # never recorded as a terminal "source experiment not found": it stays resumable / visible
    assert row.state is not ReoptState.FAILED and row.job_id is None
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == versions_before
    assert len(world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 1


def test_only_a_source_job_absent_from_this_store_is_reported_unavailable(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    elsewhere = tmp_path / "other-process"
    elsewhere.mkdir()
    monitor = ProductionMonitor(  # a process whose job store does not hold the champion's job
        world.deployments,
        SQLiteMonitoringStore(tmp_path / "monitoring.sqlite3"),
        ExperimentJobs(SQLiteJobStore(elsewhere / "jobs.sqlite3"), world.runtime),
        world.datasets,
        owner="m-elsewhere",
    )
    report = monitor.drift(version, SINCE, UNTIL, pid)
    assert report.reference["unavailable"] == ["source_experiment_not_in_this_store"]


# -- 12. review: schema drift is observed in #30's records of REJECTED requests -----------------
SCHEMA_POLICY = MonitoringPolicy(
    name="test-schema",
    drift=DriftConfig(min_samples=4, min_labelled=4, schema_violation_rate=0.1),
    trigger=SMALL.trigger,
)


def test_rejected_requests_are_failed_telemetry_and_drive_schema_drift(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.monitor.register_policy(SCHEMA_POLICY)[1]
    served = [world.ask(version, c, city)[0] for c, city in COUNTRIES[:4]]
    model_calls = len(world.model.requests)
    bad = [
        {**inputs_for("Peru", "Lima"), "passage": 42},  # a pinned field, wrong type
        {"question": "Which city is the capital of Peru?"},  # a required field missing
        {**inputs_for("Peru", "Lima"), "api_key": "sk-SECRET-123"},  # a new, unknown field
    ]
    rejected = []
    for inputs in bad:
        with pytest.raises(InvalidInferenceRequest) as exc:
            world.deployments.invoke(version, inputs)
        assert exc.value.details["failure_kind"] == INPUT_SCHEMA_REJECTED
        rejected.append(exc.value.details["inference_id"])
    assert len(world.model.requests) == model_calls  # no model call for any rejection

    pins = world.deployments.inference(served[0])
    for inference_id in rejected:
        record = world.deployments.inference(inference_id)
        assert record.status == "FAILED" and record.failure["kind"] == INPUT_SCHEMA_REJECTED
        assert record.usage.model_calls == 0 and record.run_id is None
        for key in ("workflow_version", "champion_id", "genome_hash", "provenance", "versions"):
            assert getattr(record, key) == getattr(pins, key), key
        assert record.model == pins.model
    raw = " ".join(
        r[0] for r in world.rows("deployments.sqlite3", "SELECT record_json FROM inference_records")
    )
    assert "sk-SECRET-123" not in raw and "api_key" not in raw  # never stored verbatim
    with pytest.raises(InvalidFeedback):  # a rejected request cannot carry feedback either
        world.monitor.submit_feedback(rejected[0], FeedbackRequest(inputs=bad[0]))

    summary = world.monitor.summary(version, SINCE, UNTIL, pid)
    assert summary.inferences["count"] == 4 and summary.inferences["rejected_inputs"] == 3
    assert summary.inferences["failure_rate"] == 0.0  # a client error is not a workflow failure
    report = world.monitor.drift(version, SINCE, UNTIL, pid)
    by = {f["feature"]: f for f in report.features if f["kind"] == "schema"}
    assert by["passage.type"]["violations"] == 1 and by["passage.type"]["observed_n"] == 7
    assert (
        by["passage.type"]["value"] == round(1 / 7, 6) and by["passage.type"]["status"] == "DRIFTED"
    )
    assert by["passage.missing"]["violations"] == 1
    assert by["input.unknown_fields"]["violations"] == 1
    assert by["question.type"]["violations"] == 0 and by["question.type"]["status"] == "STABLE"
    assert {"passage.type", "passage.missing", "input.unknown_fields"} <= set(
        report.drifted_features
    )
    assert set(rejected) <= set(report.evidence["inference_ids"])
    again = world.monitor.drift(version, SINCE, UNTIL, pid)  # deterministic
    assert again.model_dump_json() == report.model_dump_json()


def test_schema_drift_needs_its_minimum_sample(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    pid = world.monitor.register_policy(SCHEMA_POLICY)[1]
    world.ask(version, *COUNTRIES[0])
    with pytest.raises(InvalidInferenceRequest):
        world.deployments.invoke(version, {"question": "x"})
    report = world.monitor.drift(version, SINCE, UNTIL, pid)
    by = {f["feature"]: f for f in report.features if f["kind"] == "schema"}
    assert by["passage.missing"]["status"] == "INSUFFICIENT_EVIDENCE"  # 2 < min_samples
    assert "passage.missing" not in report.drifted_features


# -- 13. review: a delayed label after an unlabelled observation (documented limitation) --------
def test_an_unlabelled_observation_blocks_a_later_label_for_that_inference(tmp_path):
    world = World(tmp_path)
    version = world.serve()
    inference_id, output = world.ask(version, *COUNTRIES[0])
    inputs = inputs_for(*COUNTRIES[0])
    first, _ = world.monitor.submit_feedback(
        inference_id, FeedbackRequest(inputs=inputs, output=output)
    )
    assert first.labelled is False
    before = world.rows("monitoring.sqlite3", "SELECT * FROM feedback")
    with pytest.raises(FeedbackAlreadyRecorded):  # UNIQUE(inference_id): no second record
        world.monitor.submit_feedback(
            inference_id, FeedbackRequest(inputs=inputs, output=output, expected=output)
        )
    assert world.rows("monitoring.sqlite3", "SELECT * FROM feedback") == before  # never edited
    assert world.monitor.feedback(first.feedback_id) == first


def test_a_quota_refusal_releases_only_the_claim_that_monitor_acquired(tmp_path, monkeypatch):
    """#32 review: A claims, A's lease expires, B re-claims (larger fence), THEN A hits a quota
    refusal. A's release must not touch B's lease; B finishes; A can neither advance nor
    release B's work."""
    from experiments.monitoring import ReoptimizationQuotaExceeded
    from store.monitoring import ReoptClaimLost
    from store.tenancy import QuotaExceeded

    world = World(tmp_path)
    version = world.serve()
    pid = world.policy()
    world.label(version, 4, wrong=True)
    tid = world.monitor.evaluate(version, SINCE, UNTIL, pid, reoptimize=False).trigger_id
    a = world.make_monitor("m-a")
    store = world.monitor.store
    seen: dict[str, Any] = {}

    def refused_after_takeover(self, definition):
        seen["a"] = store.reoptimization(tid)  # A's claim, as A acquired it
        b = store.claim_reoptimization(tid, "m-b", 10**12, 600.0)  # A's lease long expired
        assert b is not None and b.fence > seen["a"].fence
        seen["b"] = b
        raise QuotaExceeded("active_jobs", 1, 1)  # ...and only now A's admission is refused

    monkeypatch.setattr(ExperimentJobs, "create", refused_after_takeover)
    with pytest.raises(ReoptimizationQuotaExceeded):
        a.reoptimize(tid)
    monkeypatch.undo()
    a_claim, b_claim = seen["a"], seen["b"]
    assert a_claim.owner == "m-a"
    after = store.reoptimization(tid)
    assert (after.owner, after.fence, after.lease_until) == (
        b_claim.owner,
        b_claim.fence,
        b_claim.lease_until,
    )  # B's claim is untouched
    # A's stale token can neither release nor advance B's work
    assert store.release_reoptimization(tid, a_claim.owner, a_claim.fence) is False
    with pytest.raises(ReoptClaimLost):
        store.advance_reoptimization(tid, a_claim.owner, a_claim.fence, ReoptState.JOB_CREATED)
    assert store.reoptimization(tid).lease_until == b_claim.lease_until
    # B still advances: it finishes the SAME re-optimization with exactly one challenger
    b = world.make_monitor("m-b")
    b.clock = lambda: 10**12
    view = b.reoptimize(tid)
    assert view.state is ReoptState.JOB_CREATED
    assert len(world.rows("jobs.sqlite3", "SELECT job_id FROM experiment_jobs")) == 2
    assert [v.dataset_version for v in world.datasets.get_dataset("capitals")] == [1, 2]
    # the owner's own release still works while it holds the claim (and only then)
    assert store.release_reoptimization(tid, "m-b", store.reoptimization(tid).fence) is False
