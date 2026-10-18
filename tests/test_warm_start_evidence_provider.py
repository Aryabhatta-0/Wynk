"""Production warm starts are keyed by the live external evidence provider (#28 after #52/#54).

GATHER jev means whatever the pinned JEV provider/model returns, so an ACO prior is only
compatible with a runtime that has exactly the JEV identity its source job was bound with. The
authority is the live provider AVAILABLE to the bound runtime (``JobBinding.evidence_provider``,
pinned in the durable job identity and the artifact's provenance), never the candidate mix:
candidate ``evidence_source`` values are only cross-checked against it. The provider is NOT part
of the shared ``runtime_identity`` (one search may run JEV and fetch genomes): it is a
validity-key field of its own (``evidence_provider``), left out of the key, the job identity and
the provenance while ``None`` so a fetch-only job and prior keep their ids and compatibility.

Model backends here are TEST DOUBLES (see ``tests.test_mixed_run_versions``).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from core.canonical import canonical_hash
from experiments.jobs import ExperimentJobDefinition, JobBinding, Reason, load_canonical
from memory import production
from memory.priors import (
    EVIDENCE_PROVIDER,
    DatasetRef,
    EliteReason,
    PriorKind,
    PriorRecord,
    PriorUnavailable,
    StartMode,
    StartReason,
    row_hash,
    source_provider,
)
from store.jobs import JobState, UnitState
from tests import test_memory_priors as fetch_only
from tests.test_memory_priors import DECLARED, WARM, RealRuntime
from tests.test_mixed_run_versions import (
    CONTRACT,
    FETCH_ROW,
    JEV_ID,
    JEV_ROW,
    SEARCH,
    MixedBackend,
    SuiteRuntime,
    definition,
    uses_jev,
)
from tests.test_mixed_run_versions import World as _World

JEV_B = "jev:scripted/test@2"
COLD_EVIDENCE = StartReason.COLD_EVIDENCE_SOURCE_CHANGED


def backend_for(provider: str | None) -> MixedBackend:
    """A mixed JEV/fetch TEST DOUBLE whose jev genomes report ``provider`` (``None``: every
    candidate is a fetch run)."""
    return MixedBackend(
        lambda g, t, tr: DECLARED.model_copy(
            update={"evidence_source": provider if uses_jev(g) else None}
        )
    )


class ProviderRuntime(SuiteRuntime):
    """``SuiteRuntime`` that declares a live evidence provider (``None``: no JEV available)."""

    def __init__(self, provider: str | None, backend: MixedBackend | None = None) -> None:
        super().__init__(backend or backend_for(provider))
        self.provider = provider

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        return dataclasses.replace(super().bind(definition), evidence_provider=self.provider)


class FetchRuntime(RealRuntime):
    """The fetch-only ``tests.test_memory_priors`` runtime, optionally declaring a provider."""

    def __init__(self, provider: str | None = None) -> None:
        super().__init__()
        self.provider = provider

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        return dataclasses.replace(super().bind(definition), evidence_provider=self.provider)


def world_at(root: Path, provider: str | None, backend: MixedBackend | None = None) -> _World:
    world = _World(root, ProviderRuntime(provider, backend))
    world.priors.source_live = lambda ref: True  # the snapshot dataset is held
    return world


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """A workspace that learned from one random / aco job over jev AND fetch genomes, run by a
    runtime whose live JEV provider is ``JEV_ID``."""
    root = tmp_path_factory.mktemp("jev-provider")
    world = world_at(root, JEV_ID)
    job_id = world.run(definition(SEARCH))
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    assert world.prior_store.learned(job_id).outcome == "learned"
    return root


@pytest.fixture
def copy(seeded, tmp_path):
    def make(provider: str | None, backend: MixedBackend | None = None) -> _World:
        root = tmp_path / f"ws{len(list(tmp_path.iterdir()))}"
        shutil.copytree(seeded, root)
        return world_at(root, provider, backend)

    return make


def rekey(record: PriorRecord, key: dict) -> PriorRecord:
    """``record`` with another (consistently hashed) validity key."""
    return PriorRecord.model_validate(record.dump() | {"key": key, "key_hash": canonical_hash(key)})


def warm_unit(world: _World, job_id: str):
    return next(u for u in world.store.get_job(job_id).units if u.strategy == "aco_warm")


def identity_of(world: _World, job_id: str) -> dict:
    return json.loads(world.store.get_job(job_id).identity_json)


# == the key ======================================================================================
def test_a_mixed_search_teaches_priors_keyed_by_its_jev_identity(copy):
    world = copy(JEV_ID)
    for kind in PriorKind:
        record = world.active(kind).record
        assert record.key[EVIDENCE_PROVIDER] == JEV_ID
        assert world.priors.verify_provenance(record) is None  # full re-verification


def test_a_fetch_only_prior_keeps_the_key_it_had_before(tmp_path):
    world = fetch_only.World(tmp_path)
    world.run(fetch_only.definition())
    for kind in PriorKind:
        record = world.active(kind).record
        assert EVIDENCE_PROVIDER not in record.key  # same key, key hash and prior id as before
        assert world.priors.verify_provenance(record) is None


def test_source_provider_is_the_job_authority_cross_checked_with_its_candidates():
    fetch = DECLARED.model_dump(mode="json")
    jev = {**fetch, "evidence_source": JEV_ID}
    assert source_provider(None, [fetch]) is None
    assert source_provider(JEV_ID, [fetch]) == JEV_ID  # JEV available, only fetch genomes ran
    assert source_provider(JEV_ID, [fetch, jev]) == JEV_ID  # mixed: valid
    with pytest.raises(ValueError, match="bound with 'jev"):  # a second JEV provider
        source_provider(JEV_ID, [jev, {**fetch, "evidence_source": JEV_B}])
    with pytest.raises(ValueError, match="bound with None"):  # JEV used, no JEV authority
        source_provider(None, [jev])
    # a warm run carries its parent prior forward: the parent must share the job's provider
    parent = {"mode": "warm_dataset_champion", "prior_id": "p", EVIDENCE_PROVIDER: JEV_ID}
    assert source_provider(JEV_ID, [fetch], parent) == JEV_ID
    with pytest.raises(ValueError, match="carried forward"):
        source_provider(JEV_B, [fetch], parent)
    with pytest.raises(ValueError, match="carried forward"):
        source_provider(None, [fetch], parent)
    with pytest.raises(ValueError, match="carried forward"):  # a parent without a provider
        source_provider(JEV_ID, [fetch], {"mode": "warm_task_family", "prior_id": "p"})
    assert source_provider(JEV_ID, [fetch], {"mode": "cold", "prior_id": None}) == JEV_ID


# == resolution ===================================================================================
def test_the_same_jev_provider_warm_starts(copy):
    world = copy(JEV_ID)
    champ = world.active(PriorKind.DATASET_CHAMPION)
    job_id = world.run(definition(WARM))
    pin = world.pin(job_id)
    assert pin.mode is StartMode.WARM_DATASET_CHAMPION and pin.prior_id == champ.prior_id
    assert pin.evidence_provider == JEV_ID
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    starts = {r["strategy"]: r for r in world.body(job_id)["memory"]["runs"]}
    assert starts["aco_warm"]["evidence_provider"] == JEV_ID
    assert "evidence_provider" not in starts["aco"]  # a cold baseline's summary is unchanged
    # the key is the job's authority, warm or cold, whatever the run's candidates used
    _, art = load_canonical(world.store, job_id)
    fetch = [DECLARED.model_dump(mode="json")]
    for kind in PriorKind:
        for strategy in ("aco_warm", "aco"):
            key = world.priors.key_from_provenance(kind, art.provenance, strategy, 0, fetch)
            assert key[EVIDENCE_PROVIDER] == JEV_ID
    # the warm job teaches priors that carry the provider forward, and they verify
    for kind in PriorKind:
        record = world.active(kind).record
        assert record.key[EVIDENCE_PROVIDER] == JEV_ID
        assert world.priors.verify_provenance(record) is None


@pytest.mark.parametrize("now", [JEV_B, None], ids=["other-jev", "jev-unavailable"])
def test_another_or_no_jev_provider_is_a_true_cold_start(copy, now):
    world = copy(now)
    job_id = world.run(definition(WARM))
    pin = world.pin(job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, COLD_EVIDENCE)
    assert pin.state is None and pin.evidence_provider is None
    for diag in pin.diagnostics:  # both levels: only the provider differs
        assert diag.reason is COLD_EVIDENCE and diag.mismatched == (EVIDENCE_PROVIDER,)
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    runs = {r["strategy"]: r for r in world.body(job_id)["runs"]}
    assert runs["aco_warm"]["evaluated_genome_hashes"] == runs["aco"]["evaluated_genome_hashes"]


def test_no_jev_then_no_jev_stays_compatible(tmp_path):
    world = fetch_only.World(tmp_path / "ws", FetchRuntime(None))
    world.run(fetch_only.definition())
    pin = world.pin(world.run(fetch_only.definition(WARM)))
    assert pin.mode is StartMode.WARM_DATASET_CHAMPION and pin.evidence_provider is None
    assert "evidence_provider" not in pin.dump()  # the plan hashes exactly as before


def test_a_fetch_only_prior_does_not_warm_start_a_jev_runtime(tmp_path):
    world = fetch_only.World(tmp_path / "ws", FetchRuntime(None))
    world.run(fetch_only.definition())
    now = fetch_only.World(tmp_path / "ws", FetchRuntime(JEV_ID))
    pin = now.pin(now.jobs.create(fetch_only.definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, COLD_EVIDENCE)


# == old priors ===================================================================================
def test_an_old_jev_prior_without_provider_authority_fails_closed(tmp_path, monkeypatch):
    """A job and prior from before the provider authority existed: the job's identity and
    provenance claim no provider, yet its jev candidates used one. Learning it now refuses,
    and a prior old code learned from it (keyed with no provider, so it would look compatible
    with a fetch-only runtime) is invalidated on re-verification - never used."""
    world = world_at(tmp_path / "ws", None, backend_for(JEV_ID))  # old: no authority bound
    world.worker.priors = None  # finalize without learning
    job_id = world.run(definition(SEARCH))
    assert "evidence_provider" not in identity_of(world, job_id)
    _, art = load_canonical(world.store, job_id)
    assert art.provenance.versions.evidence_provider is None
    ds = art.provenance.dataset
    own = DatasetRef(
        dataset_id=ds.dataset_id, dataset_version=ds.dataset_version, content_hash=ds.content_hash
    )
    with pytest.raises(ValueError, match="bound with None"):
        world.priors._records(job_id, world.definition(job_id), art, own)
    with monkeypatch.context() as m:  # what old code learned: a key with no provider
        m.setattr(production, "source_provider", lambda authority, *_: None)
        old = world.priors._records(job_id, world.definition(job_id), art, own)
    assert old and all(EVIDENCE_PROVIDER not in r.key for r in old)
    world.prior_store.learn(job_id, old, None)
    for provider in (None, JEV_ID):
        now = world_at(world.root, provider)
        pin = now.pin(now.jobs.create(definition(WARM)).job_id)
        assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_PROVENANCE_INVALID)
    for record in old:
        assert world.prior_store.state_of(record.prior_id)[1] == "provenance_invalid"


def test_a_job_whose_candidate_used_another_provider_teaches_nothing(tmp_path):
    world = world_at(tmp_path / "ws", JEV_ID, backend_for(JEV_B))  # bound JEV_ID, runs report B
    job_id = world.run(definition(SEARCH))
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    learned = world.prior_store.learned(job_id)
    assert learned.outcome == "refused" and learned.detail.startswith("evidence_invalid")
    assert "bound with" in learned.detail
    assert not world.prior_store.active(PriorKind.DATASET_CHAMPION)


def test_a_tampered_provider_never_verifies(copy):
    world = copy(JEV_ID)
    record = world.active(PriorKind.TASK_FAMILY).record
    key = record.key | {EVIDENCE_PROVIDER: JEV_B}
    assert world.priors.verify_provenance(rekey(record, key)) == (
        "the validity key does not derive from the provenance"
    )
    no_provider = {k: v for k, v in record.key.items() if k != EVIDENCE_PROVIDER}
    assert world.priors.verify_provenance(rekey(record, no_provider)) == (
        "the validity key does not derive from the provenance"
    )


def test_a_tampered_job_identity_provider_never_verifies(copy):
    """The provider is in both the job identity and the immutable provenance: rewriting the
    job's identity to another (or no) provider breaks artifact verification, so nothing the job
    taught verifies."""
    world = copy(JEV_ID)
    record = world.active(PriorKind.DATASET_CHAMPION).record
    job_id = record.source.job_id
    identity = identity_of(world, job_id)
    without = {k: v for k, v in identity.items() if k != "evidence_provider"}
    for forged in ({**identity, "evidence_provider": JEV_B}, without):
        conn = sqlite3.connect(world.root / "jobs.sqlite3")
        with conn:
            conn.execute(
                "UPDATE experiment_jobs SET identity_json = ? WHERE job_id = ?",
                (json.dumps(forged), job_id),
            )
        conn.close()
        problem = world.priors.verify_provenance(record)
        assert problem is not None and "evidence_provider" in problem


def test_an_old_pin_without_the_provider_fails_closed(copy):
    world = copy(JEV_ID)
    pin = world.pin(world.jobs.create(definition(WARM)).job_id)
    old = pin.model_copy(update={"evidence_provider": None})
    with pytest.raises(PriorUnavailable, match="evidence provider"):
        world.priors.verify_pin(old, definition(WARM).plan.lcb_z)


# == the durable job identity ====================================================================
def test_the_provider_is_pinned_in_the_job_identity_and_its_provenance(copy, tmp_path):
    world = copy(JEV_ID)
    (job_id,) = world.store.list_job_ids()
    assert identity_of(world, job_id)["evidence_provider"] == JEV_ID
    _, art = load_canonical(world.store, job_id)  # survives full artifact re-verification
    assert art.provenance.versions.evidence_provider == JEV_ID
    assert art.envelope["provenance"]["versions"]["evidence_provider"] == JEV_ID
    # a fetch-only (no provider) job keeps the identity and provenance it always had
    fetch = fetch_only.World(tmp_path / "fetch")
    fetch_job = fetch.run(fetch_only.definition())
    assert "evidence_provider" not in identity_of(fetch, fetch_job)
    _, art = load_canonical(fetch.store, fetch_job)
    assert "evidence_provider" not in art.envelope["provenance"]["versions"]


@pytest.mark.parametrize(
    ("bound", "now"),
    [(JEV_ID, JEV_B), (JEV_ID, None), (None, JEV_ID)],
    ids=["jev-a-to-jev-b", "jev-a-to-none", "none-to-jev-a"],
)
def test_a_job_never_runs_or_resumes_under_another_provider(tmp_path, bound, now):
    world = world_at(tmp_path / "ws", bound)
    job_id = world.jobs.create(definition(SEARCH)).job_id
    world.worker_for(ProviderRuntime(now)).run_until_idle()  # the deployment changed JEV
    for unit in world.store.get_job(job_id).units:  # rejected before any workflow ran
        assert (unit.state, unit.reason) == (UnitState.INTERRUPTED, Reason.RUNTIME_UNAVAILABLE)
        assert "identity" in (unit.detail or "")
        assert world.store.attempts(job_id, unit.strategy, unit.seed) == []
    world.jobs.resume(job_id)  # the bound provider is back: the job runs
    world.worker.run_until_idle()
    assert world.jobs.get(job_id).state is JobState.COMPLETED


def test_jev_available_but_only_fetch_candidates_still_keys_the_jev_provider(tmp_path):
    """The authority is the provider the runtime HAS, not what a candidate happened to use."""
    world = world_at(tmp_path / "ws", JEV_ID, backend_for(None))  # every candidate is fetch
    job_id = world.run(definition(SEARCH))
    assert all(rv.get("evidence_source") is None for rv in world.body(job_id)["run_versions"])
    for kind in PriorKind:
        record = world.active(kind).record
        assert record.key[EVIDENCE_PROVIDER] == JEV_ID
        assert world.priors.verify_provenance(record) is None
    pin = world.pin(world.run(definition(WARM)))  # the same JEV later: warm
    assert pin.mode is StartMode.WARM_DATASET_CHAMPION and pin.evidence_provider == JEV_ID
    now = world_at(world.root, None)  # no JEV now: cold
    pin = now.pin(now.jobs.create(definition(WARM)).job_id)
    assert (pin.mode, pin.reason) == (StartMode.COLD, COLD_EVIDENCE)


def test_a_warm_run_reporting_another_provider_fails_closed(copy):
    world = copy(JEV_ID, backend_for(JEV_B))  # declares JEV_ID, its jev runs report JEV_B
    job_id = world.run(definition(WARM))
    unit = warm_unit(world, job_id)
    assert unit.state is UnitState.FAILED
    assert "evidence provider" in (unit.detail or "")


# == elites (#54) are judged per elite, unchanged ================================================
@pytest.mark.parametrize("now", [None, JEV_ID, JEV_B])
def test_elite_routing_ignores_the_runtime_provider_in_the_key(copy, now):
    world = copy(now)
    d = definition(WARM)
    with_provider = world.runtime.bind(d)
    without = dataclasses.replace(with_provider, evidence_provider=None)
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    for task_id in (FETCH_ROW, JEV_ROW):
        pick = world.priors.elite(d, with_provider, task_id, evidence_source=now)
        assert pick == world.priors.elite(d, without, task_id, evidence_source=now)
        best = champ.task(row_hash(CONTRACT.dataset.dataset_id, task_id))
        if best.evidence_source in (None, now):
            assert pick.reason is EliteReason.TASK_ELITE
        else:
            assert pick.reason is not EliteReason.NO_COMPATIBLE_PRIOR
