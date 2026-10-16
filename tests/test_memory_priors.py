"""Provenance-gated workflow memory priors (Issue #28).

Two levels of production memory - a dataset champion and a task-family MMAS pheromone state -
learned only from verified, real-labelled, finalized jobs, keyed by the #26 provenance, scoped to
one workspace, and used only by the opt-in ``aco_warm`` strategy, exactly or not at all.

Every model backend here is a TEST DOUBLE: the synthetic objective with injected usage,
relabelled with declared (non-synthetic) runtime versions and the contract evaluator's version so
that the production learning rules can be exercised. ``Sources`` stands in for #33's
``DatasetService.source_status``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from core.canonical import canonical_hash, canonical_json
from core.constraints import ConstraintChecker
from core.dataset import DatasetSplit, DatasetSplits, SplitMethod, SplitRole
from core.genome import Genome
from core.models import AllowedModels, ModelConfiguration
from core.results import EvaluatedRun, RunVersions
from core.stages import LEGACY_STAGE_KINDS
from core.task_contract import ConstraintLimits, WorkflowSpec
from experiments.budget_ledger import ExperimentBudget
from experiments.contract_run import contract_suite
from experiments.jobs import (
    ExperimentJobDefinition,
    ExperimentJobs,
    InvalidExperiment,
    JobBinding,
    JobWorker,
)
from experiments.optimization_experiment import Strategy, without_timing
from ingestion.parse import sha256_bytes
from memory.priors import (
    PriorKind,
    RetireReason,
    StartMode,
    StartReason,
    WarmStart,
    aco_config,
    champion_state,
    converged_epoch,
    heldout_rows,
    row_hash,
)
from memory.production import PriorService
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.checkpoint import restore
from store.jobs import JobState, SQLiteJobStore, UnitState
from store.priors import SQLitePriorStore
from store.tenancy import TenantBindingError
from tests.test_contract_runtime import DATA, qa_contract, splits_for
from tests.test_durable_jobs import LEASE, Backend, Clock, Runtime
from tests.test_experiment_provenance import ENTRY, tamper
from tests.test_optimization_experiment import ALL, LIMITS, plan

DATA2 = DATA.replace(b"It is a large city.", b"It is a big city.")  # same task, other dataset
CONTRACT = qa_contract(constraints=ConstraintLimits(**LIMITS))
EVALUATOR = f"{CONTRACT.evaluation.evaluator_version}+test-fitness/1"
# What the TEST DOUBLE runtime declares and reports (model_hash = ENTRY's).
DECLARED = RunVersions(
    model_hash=ENTRY.model_hash,
    prompt_template_version="prompts/1",
    benchmark_hash="inline",
    compiler_version="compiler/1",
    grammar_version="runtime-grammar/1",
)
BUDGET = ExperimentBudget(max_candidate_evaluations=4)
WARM = (Strategy.ACO, Strategy.ACO_WARM)


def relabel(run: EvaluatedRun, versions: RunVersions, evaluator_version: str) -> EvaluatedRun:
    key = run.execution.key.model_copy(update={"versions": versions})
    return EvaluatedRun(
        execution=run.execution.model_copy(update={"key": key}),
        evaluation=run.evaluation.model_copy(update={"evaluator_version": evaluator_version}),
    )


class RealBackend(Backend):
    """TEST DOUBLE labelled as a real runtime (declared versions, contract evaluator)."""

    def __init__(self, versions: RunVersions = DECLARED, evaluator_version: str = EVALUATOR):
        super().__init__()
        inner = self._evaluate
        self._evaluate = lambda g, t, tr, s: relabel(
            inner(g, t, tr, s), versions, evaluator_version
        )


class RealRuntime:
    """TEST DOUBLE runtime with ``synthetic = False``: the registry entry, the declared versions
    and the contract evaluator's version - what the production learning rules require."""

    synthetic = False

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        versions: RunVersions = DECLARED,
        evaluator_version: str = EVALUATOR,
        declare: bool = True,
    ) -> None:
        self.versions, self.evaluator_version, self.declare = versions, evaluator_version, declare
        self.backend = backend or RealBackend(versions, evaluator_version)

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        data = {sha256_bytes(DATA): DATA, sha256_bytes(DATA2): DATA2}
        suite, _ = contract_suite(
            definition.contract, definition.splits, data[definition.contract.dataset.content_hash]
        )
        return JobBinding(
            suite=suite,
            evaluate=self.backend,
            checker=ConstraintChecker(),
            evaluator_version=self.evaluator_version,
            models=AllowedModels((ENTRY,)),
            run_versions=self.versions.model_dump(mode="json") if self.declare else None,
        )


class Sources:
    """TEST DOUBLE for #33's ``source_status``: which dataset versions are still held."""

    def __init__(self) -> None:
        self.deleted: set[tuple[str, int]] = set()

    def __call__(self, ref: Any) -> bool:
        known = ref.content_hash in {sha256_bytes(DATA), sha256_bytes(DATA2)}
        return known and (ref.dataset_id, ref.dataset_version) not in self.deleted


class World:
    """One workspace's server process: jobs, priors, the job service and its worker."""

    def __init__(self, root: Path, runtime: Any = None, *, workspace: str | None = None) -> None:
        self.root, self.clock, self.sources = root, Clock(), Sources()
        self.runtime = runtime or RealRuntime()
        self.store = SQLiteJobStore(root / "jobs.sqlite3", workspace_id=workspace)
        self.prior_store = SQLitePriorStore(root / "priors.sqlite3", workspace_id=workspace)
        self.priors = PriorService(self.prior_store, self.store, source_live=self.sources)
        self.jobs = ExperimentJobs(self.store, self.runtime, clock=self.clock, priors=self.priors)
        self.worker = self.worker_for(self.runtime)

    def worker_for(self, runtime: Any, priors: Any = "self") -> JobWorker:
        return JobWorker(
            self.store,
            runtime,
            worker_id="w",
            lease_s=LEASE,
            clock=self.clock,
            heartbeat=False,
            priors=self.priors if priors == "self" else priors,
        )

    def run(self, d: ExperimentJobDefinition) -> str:
        job_id = self.jobs.create(d).job_id
        self.worker.run_until_idle()
        return job_id

    def definition(self, job_id: str) -> ExperimentJobDefinition:
        job = self.store.get_job(job_id)
        assert job is not None
        return ExperimentJobDefinition.model_validate_json(job.definition_json)

    def pin(self, job_id: str) -> WarmStart:
        pin = self.definition(job_id).plan.warm_start
        assert pin is not None
        return pin

    def body(self, job_id: str) -> dict[str, Any]:
        return self.jobs.artifact(job_id).artifact

    def active(self, kind: PriorKind):
        (row,) = self.prior_store.active(kind)
        return row


def definition(
    strategies=ALL,
    *,
    contract=CONTRACT,
    splits=None,
    synthetic=False,
    seeds=(0,),
    **plan_kw: Any,
) -> ExperimentJobDefinition:
    return ExperimentJobDefinition(
        contract=contract,
        splits=splits or splits_for(contract),
        plan=plan(BUDGET, seeds=seeds, strategies=strategies, **plan_kw),
        synthetic=synthetic,
    )


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """A workspace that learned from one real-labelled fixed / random / aco job."""
    root = tmp_path_factory.mktemp("seeded")
    world = World(root)
    job_id = world.run(definition())
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    learned = world.prior_store.learned(job_id)
    assert learned is not None and learned.outcome == "learned", learned
    return root


@pytest.fixture
def world(seeded, tmp_path) -> World:
    shutil.copytree(seeded, tmp_path / "ws")
    return World(tmp_path / "ws")


def source_job(world: World) -> str:
    (job_id,) = world.store.list_job_ids()
    return job_id


def drop_triggers(path: Path, *names: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    for name in names:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    return conn


# == what is learned ============================================================================
def test_a_finalized_real_job_teaches_one_champion_and_one_pheromone_prior(world):
    job_id = source_job(world)
    body = world.body(job_id)
    champ = world.active(PriorKind.DATASET_CHAMPION).record
    family = world.active(PriorKind.TASK_FAMILY).record
    best = max(
        (r for r in body["runs"] if r["champion"] and r["champion"]["feasible"]),
        key=lambda r: tuple(r["champion"]["rank_key"]),
    )
    assert champ.champion is not None and champ.pheromones is None
    assert champ.champion.genome_hash == best["champion"]["genome_hash"]
    assert champ.source.job_id == job_id and champ.source.strategy == best["strategy"]
    assert champ.source.artifact_id == world.jobs.artifact(job_id).artifact_id
    assert family.pheromones is not None and family.champion is None
    assert family.source.strategy == Strategy.ACO.value  # only an MMAS run teaches pheromones
    assert champ.key["workspace_id"] == family.key["workspace_id"] == "unscoped"
    assert champ.key["dataset_content_hash"] == CONTRACT.dataset.content_hash
    assert "dataset_id" not in family.key  # a task family is reusable across datasets
    for record in (champ, family):
        assert world.priors.verify_provenance(record) is None
        assert record.synthetic is False


def test_learning_is_idempotent_per_job(world):
    job_id = source_job(world)
    before = [r.prior_id for r in world.prior_store.history()]
    assert world.priors.learn(job_id) == world.prior_store.learned(job_id)
    assert world.priors.learn_pending() == []
    assert [r.prior_id for r in world.prior_store.history()] == before


# == warm starts ================================================================================
def test_a_compatible_dataset_warm_starts_from_its_champion(world):
    champ = world.active(PriorKind.DATASET_CHAMPION)
    job_id = world.run(definition(WARM))
    pin = world.pin(job_id)
    assert (pin.mode, pin.reason) == (
        StartMode.WARM_DATASET_CHAMPION,
        StartReason.WARM_DATASET_CHAMPION,
    )
    assert pin.prior_id == champ.prior_id
    config = ACOConfig(lcb_z=definition(WARM).plan.lcb_z)
    genome = Genome.from_canonical(champ.record.champion.genome)
    assert pin.state == champion_state(genome, config)
    assert pin.state.epoch == converged_epoch(config)
    assert all(t == config.tau_max for _, _, t in pin.state.edges)
    memory = world.body(job_id)["memory"]
    assert memory["prior_loading_model_calls"] == 0
    starts = {r["strategy"]: r for r in memory["runs"]}
    assert starts["aco"] == {
        "strategy": "aco",
        "seed": 0,
        "mode": "cold",
        "reason": "cold_baseline",
        "prior_kind": None,
        "prior_id": None,
        "state_hash": None,
    }
    assert starts["aco_warm"]["mode"] == "warm_dataset_champion"
    assert starts["aco_warm"]["prior_id"] == champ.prior_id
    assert starts["aco_warm"]["state_hash"] == pin.state.state_hash
    assert world.jobs.get(job_id).state is JobState.COMPLETED


def test_a_compatible_task_family_warm_starts_from_its_exact_pheromones(world):
    family = world.active(PriorKind.TASK_FAMILY)
    other = qa_contract(DATA2, constraints=ConstraintLimits(**LIMITS))
    job_id = world.run(definition(WARM, contract=other))
    pin = world.pin(job_id)
    assert (pin.mode, pin.reason) == (StartMode.WARM_TASK_FAMILY, StartReason.WARM_TASK_FAMILY)
    assert pin.prior_id == family.prior_id
    assert pin.state == family.record.pheromones
    diag = {d.kind: d for d in pin.diagnostics}
    assert diag[PriorKind.DATASET_CHAMPION].reason is StartReason.COLD_DATASET_CHANGED
    assert world.body(job_id)["memory"]["runs"][1]["mode"] == "warm_task_family"


def test_the_aco_epoch_and_pheromones_are_reconstructed_exactly(world):
    job_id = source_job(world)
    family = world.active(PriorKind.TASK_FAMILY).record
    state = None
    for row in world.store.checkpoints(job_id, "aco", 0):
        state = json.loads(row.payload_json).get("optimizer_state") or state
    final = restore(state)
    assert isinstance(final, MMASACO)
    assert family.pheromones.epoch == final.epoch
    assert family.pheromones.as_map() == final.explored_pheromones()
    warm = WarmStart(
        mode=StartMode.WARM_TASK_FAMILY,
        reason=StartReason.WARM_TASK_FAMILY,
        prior_kind=PriorKind.TASK_FAMILY,
        prior_id=family.prior_id,
        state=family.pheromones,
        run_versions={},
    ).optimizer(final.config)
    assert warm.epoch == final.epoch
    assert warm.explored_pheromones() == final.explored_pheromones()


def test_a_warm_start_is_deterministic(seeded, tmp_path):
    outs = []
    for name in ("a", "b"):
        shutil.copytree(seeded, tmp_path / name)
        w = World(tmp_path / name)
        job_id = w.run(definition(WARM))
        body = w.body(job_id)
        outs.append((w.pin(job_id).dump(), without_timing(body)))
    assert outs[0] == outs[1]


# == cold starts ================================================================================
def test_no_prior_is_a_cold_start_identical_to_plain_aco(tmp_path):
    w = World(tmp_path)
    job_id = w.run(definition(WARM))
    pin = w.pin(job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_NO_PRIOR)
    assert pin.state is None and pin.prior_id is None
    runs = {r["strategy"]: r for r in w.body(job_id)["runs"]}
    assert runs["aco_warm"]["evaluated_genome_hashes"] == runs["aco"]["evaluated_genome_hashes"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {
                "contract": qa_contract(
                    constraints=ConstraintLimits(**LIMITS), instructions="Name the capital."
                )
            },
            StartReason.COLD_CONTRACT_CHANGED,
        ),
        (
            {
                "contract": qa_contract(
                    constraints=ConstraintLimits(**LIMITS),
                    workflow=WorkflowSpec(stages=LEGACY_STAGE_KINDS),
                )
            },
            StartReason.COLD_GRAMMAR_CHANGED,
        ),
        (
            {
                "model": ModelConfiguration(
                    provider="test-double", model="scripted", parameters={"temperature": 0.3}
                )
            },
            StartReason.COLD_MODEL_CHANGED,
        ),
        ({"lcb_z": 0.5}, StartReason.COLD_OPTIMIZER_CHANGED),
    ],
    ids=["contract", "grammar", "model", "optimizer-config"],
)
def test_an_incompatible_context_is_a_true_cold_start(world, change, reason):
    d = definition(WARM, **change)
    job_id = world.run(d)
    pin = world.pin(job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, reason)
    assert pin.state is None
    # a true cold start: exactly what a workspace without memory runs
    cold = World(world.root.parent / "cold")
    cold_id = cold.run(d)
    runs = world.body(job_id)["runs"]
    assert [r["evaluated_genome_hashes"] for r in runs] == [
        r["evaluated_genome_hashes"] for r in cold.body(cold_id)["runs"]
    ]


@pytest.mark.parametrize(
    ("runtime", "reason"),
    [
        (
            lambda: RealRuntime(evaluator_version=EVALUATOR.replace("/1", "/2")),
            StartReason.COLD_EVALUATOR_CHANGED,
        ),
        (
            lambda: RealRuntime(
                versions=DECLARED.model_copy(update={"prompt_template_version": "prompts/2"})
            ),
            StartReason.COLD_RUNTIME_CHANGED,
        ),
        (lambda: RealRuntime(declare=False), StartReason.COLD_RUNTIME_UNDECLARED),
    ],
    ids=["evaluator", "prompts", "undeclared"],
)
def test_a_changed_runtime_is_a_cold_start(seeded, tmp_path, runtime, reason):
    shutil.copytree(seeded, tmp_path / "ws")
    w = World(tmp_path / "ws", runtime())
    pin = w.pin(w.run(definition(WARM)))
    assert (pin.mode, pin.reason) == (StartMode.COLD, reason)


def test_a_synthetic_experiment_never_consults_memory(seeded, tmp_path):
    shutil.copytree(seeded, tmp_path / "ws")
    w = World(tmp_path / "ws", Runtime(Backend()))
    d = definition(WARM, synthetic=True, contract=CONTRACT)
    pin = w.pin(w.jobs.create(d).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_SYNTHETIC)


def test_without_memory_an_aco_warm_run_is_cold(tmp_path):
    w = World(tmp_path)
    w.jobs.priors = None
    pin = w.pin(w.jobs.create(definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_MEMORY_UNAVAILABLE)


def test_a_client_can_never_supply_a_warm_start(world):
    d = definition(WARM)
    data = d.model_dump(mode="json")
    data["plan"]["warm_start"] = WarmStart(
        mode=StartMode.COLD, reason=StartReason.COLD_NO_PRIOR
    ).dump()
    with pytest.raises(InvalidExperiment, match="never supplied"):
        world.jobs.create(ExperimentJobDefinition.model_validate(data))


# == baselines stay cold ========================================================================
def test_an_explicit_cold_run_with_memory_equals_one_without_memory(world, tmp_path):
    d = definition()  # fixed / random / aco: the cold comparison
    job_id = world.run(d)
    assert world.definition(job_id).plan.warm_start is None
    assert world.definition(job_id).definition_hash == d.definition_hash
    fresh = World(tmp_path / "fresh")
    fresh_id = fresh.run(d)
    a, b = world.body(job_id), fresh.body(fresh_id)
    assert a["experiment_id"] == b["experiment_id"]
    assert without_timing(a) == without_timing(b)  # the whole body, minus the clock
    assert {r["mode"] for r in a["memory"]["runs"]} == {"cold"}


def test_a_plan_without_aco_warm_serializes_as_before():
    d = definition()
    assert "warm_start" not in d.plan.model_dump(mode="json")
    assert "warm_start" not in d.model_dump_json()
    assert "warm_start" not in d.plan.protocol()


def test_equal_budget_accounting_is_unchanged_by_a_warm_start(world):
    backend = world.runtime.backend
    calls = len(backend.calls)
    job_id = world.jobs.create(definition(WARM)).job_id
    assert len(backend.calls) == calls  # resolving + loading a prior costs no model call
    world.worker.run_until_idle()
    runs = {r["strategy"]: r for r in world.body(job_id)["runs"]}
    assert world.pin(job_id).mode is StartMode.WARM_DATASET_CHAMPION
    assert runs["aco"]["budget"] == runs["aco_warm"]["budget"]
    assert runs["aco"]["reservation_per_run"] == runs["aco_warm"]["reservation_per_run"]
    for r in runs.values():
        assert len(r["evaluated_genome_hashes"]) <= BUDGET.max_candidate_evaluations
        assert r["usage"]["candidate_evaluations"] <= BUDGET.max_candidate_evaluations


# == provenance, staleness, invalidation ========================================================
def test_a_corrupted_source_artifact_excludes_its_priors(world):
    job_id = source_job(world)
    env = json.loads(world.store.artifact(job_id).envelope_json)
    env["provenance"]["model"]["prompt_template_version"] = "prompts/edited"
    tamper(
        world.store.path,
        "UPDATE experiment_artifacts SET envelope_json=? WHERE job_id=?",
        canonical_json(env),
        job_id,
    )
    pin = world.pin(world.jobs.create(definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_PROVENANCE_INVALID)
    rows = world.prior_store.history()
    assert rows and all(r.invalidated_reason is RetireReason.PROVENANCE_INVALID for r in rows)
    assert world.prior_store.active() == []


def test_a_tampered_prior_record_fails_closed(world):
    prior_id = world.active(PriorKind.DATASET_CHAMPION).prior_id
    conn = drop_triggers(world.prior_store.path, "prior_content_is_immutable")
    (text,) = conn.execute(
        "SELECT record_json FROM priors WHERE prior_id=?", (prior_id,)
    ).fetchone()
    record = json.loads(text)
    record["champion"]["validation"]["score_mean"] = 1.0
    conn.execute("UPDATE priors SET record_json=? WHERE prior_id=?", (json.dumps(record), prior_id))
    conn.commit()
    conn.close()
    pin = world.pin(world.jobs.create(definition(WARM)).job_id)
    diag = {d.kind: d for d in pin.diagnostics}
    assert diag[PriorKind.DATASET_CHAMPION].reason is StartReason.COLD_PROVENANCE_INVALID
    assert pin.prior_id != prior_id


def test_a_newer_family_prior_supersedes_and_is_never_blended(world):
    old = world.active(PriorKind.TASK_FAMILY)
    world.run(definition(WARM, seeds=(1,)))  # a compatible job learns a newer family state
    new = world.active(PriorKind.TASK_FAMILY)
    assert new.prior_id != old.prior_id
    history = {r.prior_id: r for r in world.prior_store.history(PriorKind.TASK_FAMILY)}
    assert history[old.prior_id].retired_reason is RetireReason.SUPERSEDED
    assert history[old.prior_id].superseded_by == new.prior_id
    other = qa_contract(DATA2, constraints=ConstraintLimits(**LIMITS))
    pin = world.pin(world.jobs.create(definition(WARM, contract=other)).job_id)
    assert pin.prior_id == new.prior_id
    assert pin.state == new.record.pheromones  # exactly the newest state, nothing else
    stale_only = set(old.record.pheromones.as_map()) - set(new.record.pheromones.as_map())
    assert not stale_only & set(pin.state.as_map())  # no stale edge was blended in


def test_an_incompatible_pheromone_map_is_never_merged(world):
    world.run(definition(lcb_z=0.5))  # another MMAS configuration: its own family prior
    family = {
        r.record.key["aco_config_hash"]: r for r in world.prior_store.active(PriorKind.TASK_FAMILY)
    }
    assert len(family) == 2
    other = qa_contract(DATA2, constraints=ConstraintLimits(**LIMITS))
    used = set()
    for lcb_z in (definition().plan.lcb_z, 0.5):
        d = definition(WARM, contract=other, lcb_z=lcb_z)
        pin = world.pin(world.jobs.create(d).job_id)
        match = family[canonical_hash(aco_config(lcb_z))]
        assert (pin.prior_id, pin.state) == (match.prior_id, match.record.pheromones)
        used.add(pin.prior_id)
    assert len(used) == 2


def test_deleting_a_dataset_invalidates_every_prior_that_depends_on_it(world):
    ids = sorted(r.prior_id for r in world.prior_store.active())
    world.sources.deleted.add((CONTRACT.dataset.dataset_id, CONTRACT.dataset.dataset_version))
    pin = world.pin(world.jobs.create(definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_DATASET_DELETED)
    assert world.priors.invalidate_deleted() == []  # already invalidated on use, idempotent
    history = world.prior_store.history()
    assert sorted(r.prior_id for r in history) == ids  # history kept
    assert all(r.invalidated_reason is RetireReason.DATASET_DELETED for r in history)
    assert all(r.invalidated_at and r.retired_at for r in history)


def test_invalidate_deleted_retires_eagerly(world):
    world.sources.deleted.add((CONTRACT.dataset.dataset_id, CONTRACT.dataset.dataset_version))
    assert len(world.priors.invalidate_deleted()) == 2
    assert world.prior_store.active() == []


def test_a_pinned_prior_deleted_before_execution_fails_the_unit_closed(world):
    job_id = world.jobs.create(definition(WARM)).job_id
    assert world.pin(job_id).mode is StartMode.WARM_DATASET_CHAMPION
    world.sources.deleted.add((CONTRACT.dataset.dataset_id, CONTRACT.dataset.dataset_version))
    calls = len(world.runtime.backend.calls)
    world.worker.run_until_idle()
    unit = next(u for u in world.store.get_job(job_id).units if u.strategy == "aco_warm")
    assert unit.state is UnitState.FAILED
    assert len(world.runtime.backend.calls) > calls  # aco ran ...
    assert world.jobs.get(job_id).state is not JobState.COMPLETED  # ... aco_warm never did


def test_a_worker_without_memory_never_runs_a_warm_unit(world):
    job_id = world.jobs.create(definition(WARM)).job_id
    worker = world.worker_for(world.runtime, priors=None)
    worker.run_until_idle()
    unit = next(u for u in world.store.get_job(job_id).units if u.strategy == "aco_warm")
    assert unit.state is UnitState.INTERRUPTED


def test_retired_history_stays_queryable_and_cannot_be_deleted(world):
    world.run(definition(WARM, seeds=(1,)))
    retired = [r for r in world.prior_store.history() if not r.active]
    assert retired
    for r in retired:
        assert r.retired_at and r.retired_reason is not None
        assert world.prior_store.get(r.prior_id) == r
    conn = sqlite3.connect(world.prior_store.path)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM priors WHERE prior_id=?", (retired[0].prior_id,))
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE priors SET state='ACTIVE' WHERE prior_id=?", (retired[0].prior_id,))
    conn.close()


# == isolation ==================================================================================
def test_priors_never_cross_workspaces(tmp_path):
    a = World(tmp_path / "a", workspace="ws-alpha")
    a.run(definition())
    assert a.prior_store.active()
    b = World(tmp_path / "b", workspace="ws-beta")
    pin = b.pin(b.jobs.create(definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_NO_PRIOR)
    with pytest.raises(TenantBindingError):
        SQLitePriorStore(tmp_path / "a" / "priors.sqlite3", workspace_id="ws-beta")
    with pytest.raises(ValueError, match="different workspaces"):
        PriorService(a.prior_store, b.store, source_live=a.sources)
    assert {r.record.key["workspace_id"] for r in a.prior_store.history()} == {"ws-alpha"}


def test_synthetic_jobs_never_teach_production_memory(tmp_path):
    w = World(tmp_path, Runtime(Backend()))
    job_id = w.run(definition(synthetic=True))
    assert w.jobs.get(job_id).state is JobState.COMPLETED
    learned = w.prior_store.learned(job_id)
    assert learned is not None and (learned.outcome, learned.detail) == ("refused", "synthetic")
    assert w.prior_store.history() == []


def test_a_job_without_a_registry_entry_never_teaches(tmp_path):
    class Unregistered(RealRuntime):
        def bind(self, definition):
            return dataclasses.replace(super().bind(definition), models=None)

    w = World(tmp_path, Unregistered())
    job_id = w.run(definition())
    learned = w.prior_store.learned(job_id)
    # a real job without its registry entry is never finalized as evidence: nothing is learned
    assert learned is None or (learned.outcome, learned.detail) == ("refused", "no_registry_entry")
    assert w.prior_store.history() == []


# == held-out isolation =========================================================================
def test_held_out_rows_never_reach_memory(world):
    job_id = source_job(world)
    test_rows = heldout_rows(CONTRACT, splits_for(CONTRACT))
    assert test_rows == {row_hash("capitals", "q6")}
    for row in world.prior_store.history():
        assert not set(row.record.evidence.rows) & test_rows
        assert set(row.record.evidence.rows) == {row_hash("capitals", f"q{i}") for i in range(1, 6)}
    # the experiment that taught them never executed a test row
    body = world.body(job_id)
    assert body["test_runs"] == 0
    assert all(r["split_usage"]["test_runs"] == 0 for r in body["runs"])


def test_a_prior_whose_evidence_is_held_out_here_is_refused(world):
    other = qa_contract(DATA2, constraints=ConstraintLimits(**LIMITS))
    splits = DatasetSplits(
        dataset_hash=other.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("q2", "q3", "q6")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("q4", "q5")),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=("q1",)),
        ),
    )
    pin = world.pin(world.jobs.create(definition(WARM, contract=other, splits=splits)).job_id)
    assert pin.mode is StartMode.COLD and pin.state is None
    diag = {d.kind: d for d in pin.diagnostics}
    # the family prior matches exactly, but it learned from q1, a test row here: refused
    assert diag[PriorKind.TASK_FAMILY].reason is StartReason.COLD_HELDOUT_OVERLAP
    assert diag[PriorKind.TASK_FAMILY].prior_id == world.active(PriorKind.TASK_FAMILY).prior_id
