"""Production warm starts are keyed by the live external evidence provider (#28 after #52/#54).

GATHER jev means whatever the pinned JEV provider/model returns, so an ACO prior whose learned
state reflects JEV candidate outcomes is only compatible with a runtime that would use exactly
that JEV identity. The provider is NOT part of the shared ``runtime_identity`` (one search may
run JEV and fetch genomes): it is a validity-key field of its own (``evidence_provider``), left
out of the key while ``None`` so a fetch-only prior keeps its key, its id and its compatibility.

Model backends here are TEST DOUBLES (see ``tests.test_mixed_run_versions``).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from core.canonical import canonical_hash
from experiments.jobs import ExperimentJobDefinition, JobBinding, load_canonical
from memory.priors import (
    EVIDENCE_PROVIDER,
    DatasetRef,
    EliteReason,
    PriorKind,
    PriorRecord,
    PriorUnavailable,
    StartMode,
    StartReason,
    learned_provider,
    row_hash,
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
    """A mixed JEV/fetch TEST DOUBLE whose jev genomes report ``provider``."""
    return MixedBackend(
        lambda g, t, tr: (
            DECLARED.model_copy(update={"evidence_source": provider})
            if uses_jev(g) and provider is not None
            else None
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


def test_learned_provider():
    fetch = DECLARED.model_dump(mode="json")
    jev = {**fetch, "evidence_source": JEV_ID}
    assert learned_provider([fetch]) is None
    assert learned_provider([fetch, jev]) == JEV_ID  # mixed: the JEV identity
    assert learned_provider([fetch], JEV_ID) == JEV_ID  # a warm start's state is carried forward
    with pytest.raises(ValueError, match="several"):
        learned_provider([jev, {**fetch, "evidence_source": JEV_B}])
    with pytest.raises(ValueError, match="several"):
        learned_provider([jev], JEV_B)


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
    # a warm run's state is carried forward: even a fetch-only warm run keys the inherited JEV
    _, art = load_canonical(world.store, job_id)
    fetch = [DECLARED.model_dump(mode="json")]
    for kind in PriorKind:
        key = world.priors.key_from_provenance(kind, art.provenance, "aco_warm", 0, fetch)
        assert key[EVIDENCE_PROVIDER] == JEV_ID
        cold = world.priors.key_from_provenance(kind, art.provenance, "aco", 0, fetch)
        assert EVIDENCE_PROVIDER not in cold
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
def test_an_old_prior_without_a_provider_fails_closed(tmp_path):
    """A prior learned from JEV evidence before the key recorded the provider: its key claims
    none, so it would look compatible with a fetch-only runtime. Re-verification re-derives the
    key from the provenance (with the provider), so it is invalidated - never used."""
    world = world_at(tmp_path / "ws", JEV_ID)
    world.worker.priors = None  # finalize without learning
    job_id = world.run(definition(SEARCH))
    assert world.prior_store.learned(job_id) is None
    _, art = load_canonical(world.store, job_id)
    ds = art.provenance.dataset
    own = DatasetRef(
        dataset_id=ds.dataset_id, dataset_version=ds.dataset_version, content_hash=ds.content_hash
    )
    old = []
    for record in world.priors._records(job_id, world.definition(job_id), art, own):
        old.append(rekey(record, {k: v for k, v in record.key.items() if k != EVIDENCE_PROVIDER}))
    world.prior_store.learn(job_id, old, None)
    for provider in (None, JEV_ID):
        now = world_at(world.root, provider)
        pin = now.pin(now.jobs.create(definition(WARM)).job_id)
        assert (pin.mode, pin.reason) == (StartMode.COLD, StartReason.COLD_PROVENANCE_INVALID)
    for record in old:
        assert world.prior_store.state_of(record.prior_id)[1] == "provenance_invalid"


def test_a_tampered_provider_never_verifies(copy):
    world = copy(JEV_ID)
    record = world.active(PriorKind.TASK_FAMILY).record
    key = record.key | {EVIDENCE_PROVIDER: JEV_B}
    assert world.priors.verify_provenance(rekey(record, key)) == (
        "the validity key does not derive from the provenance"
    )


def test_an_old_pin_without_the_provider_fails_closed(copy):
    world = copy(JEV_ID)
    pin = world.pin(world.jobs.create(definition(WARM)).job_id)
    old = pin.model_copy(update={"evidence_provider": None})
    with pytest.raises(PriorUnavailable, match="evidence provider"):
        world.priors.verify_pin(old, definition(WARM).plan.lcb_z)


# == execution ====================================================================================
def test_a_warm_unit_never_runs_under_another_provider(copy):
    world = copy(JEV_ID)
    job_id = world.jobs.create(definition(WARM)).job_id
    assert world.pin(job_id).evidence_provider == JEV_ID
    world.worker_for(ProviderRuntime(JEV_B)).run_until_idle()  # the deployment changed JEV
    unit = warm_unit(world, job_id)
    assert unit.state is UnitState.INTERRUPTED
    assert "evidence provider" in (unit.detail or "")
    world.jobs.resume(job_id)  # the pinned provider is back: the unit runs warm
    world.worker.run_until_idle()
    assert world.jobs.get(job_id).state is JobState.COMPLETED


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
