"""One search over JEV and fetch genomes: one RunVersions per distinct identity.

A JEV genome's runs report the pinned ``evidence_source`` (#52) and a fetch genome's none, so one
canonical strategy run records several RunVersions. Provenance stays strict: every candidate
records the exact RunVersions its runs reported, every genome reports exactly one, and all of
them share one runtime identity (model, prompts, benchmark, compiler, grammar). Per-task
champions and the dataset-champion prior keep the evidence source of their OWN runs.

Model backends here are TEST DOUBLES (the synthetic objective, relabelled with the RunVersions
``WorkflowRunner.versions`` would report for each genome - see ``tests.test_memory_priors``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core.canonical import canonical_json
from core.constraints import ConstraintChecker
from core.genome import Genome
from core.models import AllowedModels
from core.results import EvaluatedRun, RunVersions, Verdict, runtime_identity
from core.stages import GatherSource
from experiments import task_champions as tc
from experiments.jobs import ExperimentJobDefinition, JobBinding, load_canonical
from experiments.optimization_experiment import Strategy
from experiments.provenance import ArtifactIntegrityError, recompute_declared
from memory.priors import EliteReason, EliteRole, PriorKind, row_hash
from store.jobs import JobState
from tests.conftest import make_suite
from tests.test_durable_jobs import Backend
from tests.test_memory_priors import (
    BUDGET,
    DECLARED,
    ENTRY,
    EVALUATOR,
    WARM,
    RealRuntime,
    World,
    relabel,
)
from tests.test_optimization_experiment import ALL, plan

JEV_ID = "jev:scripted/test@1"
# One snapshot-backed contract whose rows may be gathered with fetch OR jev (a jsonl dataset
# cannot be gathered with jev), split q1-q3 optimization / q4-q5 validation / q6 test.
SUITE = make_suite(
    ("q1", "q2", "q3"),
    ("q4", "q5"),
    ("q6",),
    allowed_sources=(GatherSource.FETCH, GatherSource.JEV),
)
CONTRACT = SUITE.policy
SEARCH = (Strategy.RANDOM, Strategy.ACO)
JEV_ROW, FETCH_ROW = "q5", "q4"  # the validation row each kind of genome wins


def uses_jev(genome: Genome | dict[str, Any]) -> bool:
    g = genome if isinstance(genome, Genome) else Genome.from_canonical(genome)
    return any(s.kind == "GATHER" and s.source == GatherSource.JEV for s in g.stages)


def source_of(genome: Genome | dict[str, Any]) -> str | None:
    return JEV_ID if uses_jev(genome) else None


class MixedBackend(Backend):
    """TEST DOUBLE: reports what ``WorkflowRunner.versions`` reports - the pinned JEV identity
    for a genome that gathers with jev, none otherwise - and makes jev genomes the better ones
    on ``JEV_ROW`` and fetch genomes the better ones on ``FETCH_ROW``. ``versions`` may override
    the RunVersions a run reports (``(genome, task, trial) -> RunVersions``)."""

    def __init__(self, versions=None) -> None:
        super().__init__()
        inner = self._evaluate

        def evaluate(g, t, tr, s):
            v = versions(g, t, tr) if versions else None
            v = v or DECLARED.model_copy(update={"evidence_source": source_of(g)})
            run = relabel(inner(g, t, tr, s), v, EVALUATOR)
            if t.id == (FETCH_ROW if uses_jev(g) else JEV_ROW):
                worse = {"verdict": Verdict.FAIL, "fitness": 0.0, "evaluator": None}
                run = EvaluatedRun(
                    execution=run.execution, evaluation=run.evaluation.model_copy(update=worse)
                )
            return run

        self._evaluate = evaluate


class SuiteRuntime(RealRuntime):
    """TEST DOUBLE runtime binding every job to ``SUITE`` (the real-labelled production rules),
    with ``JEV_ID`` as its live evidence provider."""

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        return JobBinding(
            suite=SUITE,
            evaluate=self.backend,
            checker=ConstraintChecker(),
            evaluator_version=self.evaluator_version,
            models=AllowedModels((ENTRY,)),
            run_versions=self.versions.model_dump(mode="json"),
            evidence_provider=JEV_ID,  # the pinned JEV its jev genomes gather with
        )


def definition(strategies=ALL, **plan_kw: Any) -> ExperimentJobDefinition:
    return ExperimentJobDefinition(
        contract=CONTRACT,
        splits=SUITE.splits,
        plan=plan(BUDGET, seeds=(0,), strategies=strategies, **plan_kw),
        synthetic=False,
    )


def mixed_world(root: Path, backend: Backend | None = None) -> World:
    world = World(root, SuiteRuntime(backend or MixedBackend()))
    world.priors.source_live = lambda ref: True  # the snapshot dataset is held
    return world


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """A workspace that learned from one random / aco job over jev AND fetch genomes (the fixed
    baseline evaluates one fetch genome only)."""
    root = tmp_path_factory.mktemp("mixed")
    world = mixed_world(root)
    job_id = world.run(definition(SEARCH))
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    learned = world.prior_store.learned(job_id)
    assert learned is not None and learned.outcome == "learned", learned
    return root


@pytest.fixture
def world(seeded, tmp_path) -> World:
    shutil.copytree(seeded, tmp_path / "ws")
    return mixed_world(tmp_path / "ws")


def body_of(world: World) -> dict[str, Any]:
    (job_id,) = world.store.list_job_ids()
    return world.body(job_id)


# == the search ==================================================================================
def test_one_search_records_one_run_versions_per_identity(world):
    body = body_of(world)
    jev = DECLARED.model_copy(update={"evidence_source": JEV_ID}).model_dump(mode="json")
    fetch = DECLARED.model_dump(mode="json")
    assert jev in body["run_versions"] and fetch in body["run_versions"]
    assert len(body["run_versions"]) == 2
    assert runtime_identity(body["run_versions"]) == fetch  # one model / prompts / grammar ...
    mixed = 0
    for r in body["runs"]:
        for c in r["candidates"]:  # each candidate: the exact identity its runs ran under
            assert c["run_versions"].get("evidence_source") == source_of(c["genome"])
        assert len(r["run_versions"]) == len(
            {canonical_json(c["run_versions"]) for c in r["candidates"]}
        )
        mixed += len(r["run_versions"]) == 2
    assert mixed  # at least one strategy run evaluated both kinds of genome


def test_a_mixed_artifact_verifies(world):
    (job_id,) = world.store.list_job_ids()
    _, art = load_canonical(world.store, job_id)  # verifies hashes and the provenance record
    assert recompute_declared(art.body, CONTRACT) > 0
    for r in art.body["runs"]:
        assert tc.rederive(r, CONTRACT) == [] and tc.rederive(r, None) == []


def test_each_task_champion_carries_its_own_runs_evidence_source(world):
    body = body_of(world)
    sources = set()
    for r in body["runs"]:
        by_path = {tr.path: tr for tr in tc.artifact_task_runs(r)}
        for t in r["task_champions"]:
            want = source_of(t["genome"])
            assert t["evidence_sources"] == ([want] if want else [])
            assert [v.get("evidence_source") for v in t["run_versions"]] == [want]
            assert {by_path[p].evidence_source for p in t["run_paths"]} == {want}
            sources.add((t["task_id"], want))
    assert (JEV_ROW, JEV_ID) in sources and (FETCH_ROW, None) in sources


# == production memory ===========================================================================
def test_the_prior_keeps_every_elites_own_evidence_source(world):
    row = world.active(PriorKind.DATASET_CHAMPION)
    champ = row.record.champion
    assert champ.evidence_source == source_of(champ.genome)
    assert champ.tasks
    for t in champ.tasks:
        assert t.evidence_source == source_of(t.genome)
    assert {t.evidence_source for t in champ.tasks} == {JEV_ID, None}
    assert world.priors.verify_provenance(row.record) is None


def test_a_fetch_task_champion_stores_exactly_what_it_stored_before(world):
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    for t in champ.tasks:
        dumped = t.model_dump(mode="json")
        assert ("evidence_source" in dumped) is (t.evidence_source is not None)


@pytest.mark.parametrize("live", [None, JEV_ID, "jev:scripted/test@2"])
def test_routing_offers_only_elites_measured_with_the_live_provider(world, live):
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    d = definition(WARM)
    binding = world.runtime.bind(d)
    for task_id in (FETCH_ROW, JEV_ROW):
        best = champ.task(row_hash(CONTRACT.dataset.dataset_id, task_id))
        assert best is not None
        pick = world.priors.elite(d, binding, task_id, evidence_source=live)
        if best.evidence_source in (None, live):
            assert pick.reason is EliteReason.TASK_ELITE
            assert pick.elite is not None and pick.elite.genome_hash == best.genome_hash
        elif champ.evidence_source in (None, live):
            assert pick.reason is EliteReason.TASK_EVIDENCE_SOURCE_CHANGED
            assert pick.elite is not None and pick.elite.role is EliteRole.DATASET
        else:
            assert pick.reason is EliteReason.EVIDENCE_SOURCE_CHANGED and pick.elite is None


# == strictness ==================================================================================
def _failed(tmp_path: Path, versions) -> str:
    world = mixed_world(tmp_path, MixedBackend(versions))
    job_id = world.run(definition())
    view = world.jobs.get(job_id)
    assert view.state is JobState.FAILED
    return json.dumps(view.model_dump(mode="json"))


def test_runs_that_differ_beyond_the_evidence_source_fail_closed(tmp_path):
    drift = DECLARED.model_copy(update={"evidence_source": JEV_ID, "compiler_version": "other"})
    reason = _failed(tmp_path, lambda g, t, tr: drift if uses_jev(g) else None)
    assert "besides their evidence source" in reason


def test_one_genome_reporting_two_identities_fails_closed(tmp_path):
    other = DECLARED.model_copy(update={"evidence_source": "jev:scripted/test@2"})
    reason = _failed(tmp_path, lambda g, t, tr: other if t.id == "q2" else None)
    # #12: the job (bound with test@1) now refuses the test@2 run as soon as it is reported
    assert "reported different versions" in reason or "bound with" in reason


def _tampered(world: World) -> dict[str, Any]:
    body = json.loads(json.dumps(body_of(world)))
    run = next(r for r in body["runs"] if len(r["run_versions"]) == 2)
    return body | {"_run": run}


def test_a_candidate_relabelled_to_another_identity_fails_closed(world):
    body = _tampered(world)
    run = body.pop("_run")
    jev = next(t for t in run["task_champions"] if t["evidence_sources"])
    evaluation = int(jev["run_paths"][0].split(".")[1])
    cand = next(c for c in run["candidates"] if c["evaluation"] == evaluation)
    cand["run_versions"].pop("evidence_source")  # the jev task champion claims it ran on fetch
    with pytest.raises(ArtifactIntegrityError, match="do not re-derive"):
        recompute_declared(body, CONTRACT)


def test_a_genome_reporting_two_identities_never_verifies(world):
    body = _tampered(world)
    run = body.pop("_run")
    by_genome: dict[str, list[dict[str, Any]]] = {}
    for c in run["candidates"]:
        by_genome.setdefault(c["genome_hash"], []).append(c)
    repeated = next((cs for cs in by_genome.values() if len(cs) > 1), None)
    if repeated is None:  # no repeat evaluation: make one candidate a repeat of another
        a, b = run["candidates"][:2]
        b["genome_hash"], repeated = a["genome_hash"], [a, b]
    a, b = repeated[:2]
    other = {**a["run_versions"], "evidence_source": "jev:x/y@z"}
    b["run_versions"] = other
    run["run_versions"] = [
        json.loads(v) for v in sorted({canonical_json(v) for v in (*run["run_versions"], other)})
    ]
    with pytest.raises(ArtifactIntegrityError, match="two RunVersions"):
        recompute_declared(body, CONTRACT)


def test_an_unreferenced_or_missing_identity_fails_closed(world):
    body = _tampered(world)
    run = body.pop("_run")
    extra = RunVersions.model_validate(run["run_versions"][0])
    run["run_versions"].append(
        extra.model_copy(update={"evidence_source": "jev:x/y@z"}).model_dump(mode="json")
    )
    with pytest.raises(ArtifactIntegrityError, match="run_versions"):
        recompute_declared(body, CONTRACT)
    body = _tampered(world)
    body.pop("_run")
    body["run_versions"] = body["run_versions"][:1]
    with pytest.raises(ArtifactIntegrityError, match="run_versions"):
        recompute_declared(body, CONTRACT)


def test_runs_of_another_runtime_identity_never_verify(world):
    body = _tampered(world)
    body.pop("_run")
    body["run_versions"][1]["compiler_version"] = "other"
    with pytest.raises(ArtifactIntegrityError, match="evidence source"):
        recompute_declared(body, CONTRACT)


def test_a_legacy_candidate_needs_a_single_identity(world):
    run = json.loads(json.dumps(_tampered(world)["_run"]))
    for c in run["candidates"]:
        c.pop("run_versions")  # recorded before per-candidate RunVersions
    with pytest.raises(ValueError, match="reports several"):
        tc.artifact_task_runs(run)
    run["run_versions"] = run["run_versions"][:1]
    runs = tc.artifact_task_runs(run)  # one identity: every run ran under it
    assert {r.run_versions for r in runs} == {canonical_json(run["run_versions"][0])}


def test_runtime_identity_ignores_only_the_evidence_source():
    fetch = DECLARED.model_dump(mode="json")
    jev = {**fetch, "evidence_source": JEV_ID}
    assert runtime_identity([fetch, jev]) == fetch
    with pytest.raises(ValueError, match="beyond their evidence source"):
        runtime_identity([fetch, {**jev, "grammar_version": "grammar/9"}])
    with pytest.raises(ValueError, match="no runtime identity"):
        runtime_identity([])
