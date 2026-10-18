"""Deterministic per-task best-genome selection (Issue #10).

Every validation task gets the validated genome its contract ranks best on that task's own
validation runs; the mapping is stored in the search result, re-derives from the stored runs,
persists in the #28 dataset-champion prior and routes the chat's elite ant. The suite champion,
optimizer feedback and equal-budget accounting are unchanged.

Model backends here are TEST DOUBLES (the synthetic objective, relabelled where a production rule
needs a real-looking runtime - see ``tests.test_memory_priors``).
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest

from benchmarks.legacy_adapter import legacy_execution_task, legacy_suite
from core.canonical import canonical_hash
from core.dataset import SplitRole
from core.genome import Genome
from core.results import Verdict
from core.run_contract import RunMeasure
from experiments import task_champions as tc
from experiments.learning_curves import ExperimentConfig, make_optimizer, run_search, search_tasks
from experiments.provenance import ArtifactIntegrityError, recompute_declared, trace
from experiments.synthetic import synthetic_evaluate
from memory.priors import (
    Champion,
    EliteReason,
    EliteRole,
    PriorKind,
    PriorRecord,
    StartReason,
    TaskChampion,
    elite_for,
    row_hash,
)
from memory.production import _champion
from optimizers.aco_mmas import MMASACO
from tests.conftest import extract, gather, synth
from tests.test_contract_runtime import splits_for
from tests.test_memory_priors import (
    CONTRACT,
    DECLARED,
    WARM,
    RealRuntime,
    World,
    definition,
    drop_triggers,
    source_job,
)

JEV_ID = "jev:scripted/test@1"


class JevRuntime(RealRuntime):
    """TEST DOUBLE runtime that gathers with a pinned JEV: every run reports ``JEV_ID`` and the
    binding declares it as the live evidence provider (as ``WorkflowBackend`` does)."""

    def __init__(self) -> None:
        super().__init__(versions=DECLARED.model_copy(update={"evidence_source": JEV_ID}))

    def bind(self, definition: Any) -> Any:
        return dataclasses.replace(super().bind(definition), evidence_provider=JEV_ID)


CONFIG = ExperimentConfig(budget=60)


# == helpers ====================================================================================
def measure(verdict=Verdict.PASS, tokens=100, wall=1.0) -> RunMeasure:
    return RunMeasure(
        verdict=verdict, tokens=tokens, model_calls=2, tool_calls=1, retries=0, wall_time_s=wall
    )


def task_run(h, task, trial=0, *, fitness=1.0, verdict=Verdict.PASS, tokens=100) -> tc.TaskRun:
    return tc.TaskRun(
        genome_hash=h,
        task_id=task,
        trial=trial,
        measure=measure(verdict, tokens),
        fitness=fitness,
        score=fitness,
    )


def genomes_for(*hashes: str) -> dict[str, dict[str, Any]]:
    return {h: {"stages": [{}, {}, {}]} for h in hashes}


SUITE_A = legacy_suite("A")
POLICY = SUITE_A.policy
G1, G2, G3 = "1" * 64, "2" * 64, "3" * 64


def by_task(records):
    return {t["task_id"]: t for t in records}


# == pure selection =============================================================================
def test_two_tasks_select_different_genomes():
    runs = [
        task_run(G1, "t1", fitness=1.0),
        task_run(G1, "t2", fitness=0.0, verdict=Verdict.FAIL),
        task_run(G2, "t1", fitness=0.0, verdict=Verdict.FAIL),
        task_run(G2, "t2", fitness=1.0),
    ]
    got = by_task(tc.select(POLICY, genomes_for(G1, G2), runs))
    assert got["t1"]["genome_hash"] == G1 and got["t2"]["genome_hash"] == G2
    assert got["t1"]["compared_genomes"] == got["t2"]["compared_genomes"] == 2
    assert got["t1"]["validation"] == {
        "runs": 1,
        "pass_rate": 1.0,
        "fitness_mean": 1.0,
        "score_mean": 1.0,
    }


def test_an_exact_tie_is_resolved_by_the_genome_hash_not_by_execution_order():
    runs = [task_run(h, "t1", trial=i) for h in (G1, G3, G2) for i in range(2)]
    want = max(G1, G2, G3)
    for _ in range(5):
        random.shuffle(runs)
        (rec,) = tc.select(POLICY, genomes_for(G1, G2, G3), runs)
        assert rec["genome_hash"] == want


def test_selection_order_is_feasible_then_rank_then_fitness_then_hash():
    # G3 has the highest fitness but breached a cap (INFEASIBLE): every feasible one outranks it
    runs = [
        task_run(G3, "t1", fitness=1.0, verdict=Verdict.INFEASIBLE),
        task_run(G1, "t1", fitness=0.4),
        task_run(G2, "t1", fitness=0.6),
    ]
    (rec,) = tc.select(POLICY, genomes_for(G1, G2, G3), runs)
    assert rec["feasible"] is True and rec["genome_hash"] == G2  # equal rank, higher fitness


def test_an_infeasible_winner_is_diagnostic_only():
    runs = [task_run(G1, "t1", verdict=Verdict.INFEASIBLE)]
    (rec,) = tc.select(POLICY, genomes_for(G1), runs)
    assert rec["feasible"] is False and rec["rank_key"] == [0.0]
    assert rec["violations"] == ["a run breached its per-run caps"]
    assert not tc.usable(rec)


def test_selection_refuses_a_run_that_is_not_on_a_validation_task():
    runs = [task_run(G1, "train-row")]
    with pytest.raises(ValueError, match="not a validation task"):
        tc.select(POLICY, genomes_for(G1), runs, validation_tasks=["val-row"])


def test_two_final_runs_of_one_trial_are_refused():
    runs = [task_run(G1, "t1", trial=0), task_run(G1, "t1", trial=0)]
    with pytest.raises(ValueError, match="two final runs"):
        tc.select(POLICY, genomes_for(G1), runs)


# == the learning-curves search (Class A and Class B) ===========================================
@pytest.mark.parametrize("task_class", ["A", "B"])
def test_a_search_selects_a_deterministic_champion_per_validation_task(task_class):
    suite = legacy_suite(task_class)
    _, val = search_tasks(suite)

    def search():
        return run_search(
            make_optimizer(MMASACO.name, CONFIG), synthetic_evaluate, suite, CONFIG, 0
        )

    a, b = search(), search()
    assert a["task_champions"] == b["task_champions"]  # reproducible
    assert [t["task_id"] for t in a["task_champions"]] == sorted(t.id for t in val)
    validated = {w["genome_hash"] for w in a["workflows"]}
    for t in a["task_champions"]:
        assert t["genome_hash"] in validated
        assert canonical_hash(t["genome"]) == t["genome_hash"]
        assert t["validation"]["runs"] == CONFIG.trials
        assert len(t["run_ids"]) == CONFIG.trials
        assert t["contract_hashes"] == [next(x for x in val if x.id == t["task_id"]).contract_hash]
    # the suite champion is still selected over ALL validation rows, as before
    assert a["champion"]["genome_hash"] in validated
    assert set(a["champion"]) == {
        "genome_hash",
        "feasible",
        "rank_key",
        "violations",
        "validation_fitness",
        "validation_pass_rate",
    }


def test_the_learning_curves_mapping_survives_a_json_round_trip(tmp_path):
    r = run_search(make_optimizer(MMASACO.name, CONFIG), synthetic_evaluate, SUITE_A, CONFIG, 1)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(r, sort_keys=True), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["task_champions"] == r["task_champions"]


# == the canonical artifact =====================================================================
@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """A workspace that learned from one real-labelled fixed / random / aco job."""
    root = tmp_path_factory.mktemp("seeded")
    World(root).run(definition())
    return root


@pytest.fixture
def world(seeded, tmp_path) -> World:
    shutil.copytree(seeded, tmp_path / "ws")
    return World(tmp_path / "ws")


@pytest.fixture(scope="module")
def jev_seeded(tmp_path_factory) -> Path:
    """A workspace that learned from a job whose runtime gathered with a pinned JEV."""
    root = tmp_path_factory.mktemp("jev")
    world = World(root, JevRuntime())
    world.run(definition())
    return root


def _artifact(world: World):
    from experiments.jobs import load_canonical

    return load_canonical(world.store, source_job(world))[1]


def test_every_strategy_run_records_task_champions_from_validation_rows_only(world):
    body = world.body(source_job(world))
    splits = splits_for(CONTRACT)
    validation = set(splits.split(SplitRole.VALIDATION).row_ids)
    for r in body["runs"]:
        champs = r["task_champions"]
        assert [t["task_id"] for t in champs] == sorted(validation)
        for t in champs:
            assert t["task_id"] in validation  # never an optimization or test row
            assert t["run_paths"] and len(t["run_ids"]) == t["validation"]["runs"]
            assert t["genome_hash"] in r["evaluated_genome_hashes"]
    assert body["test_runs"] == 0


def test_task_champions_re_derive_from_the_stored_validation_runs(world):
    art = _artifact(world)
    assert recompute_declared(art.body, CONTRACT) > 0
    for r in art.body["runs"]:
        assert tc.rederive(r, CONTRACT) == []
        assert tc.rederive(r, None) == []  # rank-free parts, without a contract


def test_the_mapping_survives_an_artifact_round_trip(world):
    art = _artifact(world)
    again = json.loads(json.dumps(art.body, sort_keys=True))
    assert [r["task_champions"] for r in again["runs"]] == [
        r["task_champions"] for r in art.body["runs"]
    ]
    assert all(tc.rederive(r, CONTRACT) == [] for r in again["runs"])


def test_a_task_champion_run_reference_traces_to_its_validation_run(world):
    art = _artifact(world)
    run = art.body["runs"][0]
    t = run["task_champions"][0]
    hit = trace(art, f"strategy.{run['strategy']}.seed.{run['seed']}.{t['run_paths'][0]}")
    assert hit["evidence"]["row_id"] == t["task_id"]
    assert hit["evidence"]["split"] == SplitRole.VALIDATION.value
    assert hit["evidence"]["genome_hash"] == t["genome_hash"]
    assert hit["evidence"]["attempt_id"]  # a durable job's write-ahead attempt


def test_a_tampered_task_champion_fails_closed(world):
    art = _artifact(world)
    body = json.loads(json.dumps(art.body))
    t = body["runs"][0]["task_champions"][0]
    others = [h for h in body["runs"][0]["evaluated_genome_hashes"] if h != t["genome_hash"]]
    if others:
        t["genome_hash"] = others[0]
    else:
        t["validation"]["fitness_mean"] += 0.5
    with pytest.raises(ArtifactIntegrityError, match="do not re-derive"):
        recompute_declared(body, CONTRACT)


def test_the_suite_champion_is_unchanged_by_per_task_selection(world):
    body = world.body(source_job(world))
    for r in body["runs"]:
        champ = r["champion"]
        assert set(champ) == {
            "genome_hash",
            "genome",
            "feasible",
            "rank_key",
            "violations",
            "first_evaluation",
            "validation",
        }
        assert champ["genome_hash"] == r["curve"][-1]["best_so_far_genome_hash"]
    prior = world.active(PriorKind.DATASET_CHAMPION).record.champion
    best = max(
        (r for r in body["runs"] if r["champion"]["feasible"]),
        key=lambda r: tuple(r["champion"]["rank_key"]),
    )
    assert prior.genome_hash == best["champion"]["genome_hash"]
    assert list(prior.rank_key) == best["champion"]["rank_key"]


# == production memory ==========================================================================
def test_the_dataset_champion_prior_persists_the_feasible_task_champions(world):
    row = world.active(PriorKind.DATASET_CHAMPION)
    champ = row.record.champion
    body = world.body(source_job(world))
    run = next(
        r
        for r in body["runs"]
        if (r["strategy"], r["seed"]) == (row.record.source.strategy, row.record.source.seed)
    )
    want = sorted(
        (row_hash(CONTRACT.dataset.dataset_id, t["task_id"]), t["genome_hash"])
        for t in run["task_champions"]
        if t["feasible"]
    )
    assert [(t.task, t.genome_hash) for t in champ.tasks] == want and want
    assert champ.evidence_source is None  # fetch/inline runs: no live provider
    assert "q4" not in row.record.model_dump_json()  # row ids never kept in clear
    assert world.priors.verify_provenance(row.record) is None


GENOME = Genome.of(gather(), extract(), synth())


def _entry(row: str, split: str, *, verdict="PASS", score=1.0, trial=0) -> dict[str, Any]:
    """One stored workflow-run entry (``candidates[].runs[]``) of an experiment artifact."""
    return {
        "row_id": row,
        "split": split,
        "trial": trial,
        "attempt": 0,
        "verdict": verdict,
        "score": score,
        "fitness": score,
        "tokens": 100,
        "model_calls": 2,
        "tool_calls": 1,
        "retries": 0,
        "wall_time_s": 1.0,
    }


def test_an_infeasible_task_winner_is_never_stored():
    run = {
        "champion": {
            "genome": GENOME.canonical(),
            "genome_hash": GENOME.genome_hash,
            "rank_key": [1.0, 0.5],
            "validation": {"runs": 2, "score_mean": 1.0, "pass_rate": 1.0, "fitness_mean": 1.0},
        },
        "run_versions": [DECLARED.model_dump(mode="json")],
        "candidates": [
            {
                "evaluation": 1,
                "genome_hash": GENOME.genome_hash,
                "run_versions": DECLARED.model_dump(mode="json"),
                "runs": [
                    _entry("q1", "optimization"),
                    _entry("q2", "optimization"),
                    _entry("q3", "validation"),
                    _entry("q4", "validation", verdict="FAIL", score=0.0),
                ],
            }
        ],
        "task_champions": [
            {
                "task_id": "q4",
                "genome": GENOME.canonical(),
                "genome_hash": GENOME.genome_hash,
                "feasible": False,
                "rank_key": [0.0],
                "validation": {"runs": 1, "score_mean": 0, "pass_rate": 0, "fitness_mean": 0},
                "evidence_sources": [],
                "run_paths": ["candidate.1.runs.3"],
            }
        ],
    }
    assert _champion(run, "capitals").tasks == ()


def test_old_priors_remain_readable_with_the_same_content_address(world):
    record = world.active(PriorKind.DATASET_CHAMPION).record
    legacy = record.dump()
    legacy["champion"].pop("tasks", None)
    legacy["champion"].pop("evidence_source", None)  # exactly what #28 stored
    old = PriorRecord.model_validate_json(json.dumps(legacy))
    assert old.champion is not None and old.champion.tasks == ()
    assert old.dump() == legacy  # the absent fields are not serialized back
    assert old.prior_id == "prior-" + canonical_hash(legacy)
    for row in world.prior_store.history():
        assert row.record.prior_id == row.prior_id  # stored priors still hash to their ids


def test_an_old_prior_without_task_champions_falls_back_to_the_dataset_elite(world):
    record = world.active(PriorKind.DATASET_CHAMPION).record
    old = record.model_copy(update={"champion": record.champion.model_copy(update={"tasks": ()})})
    pick = elite_for(old, row_hash("capitals", "q4"), None)
    assert pick.reason is EliteReason.NO_TASK_CHAMPION
    assert pick.elite.role is EliteRole.DATASET
    assert pick.elite.genome_hash == record.champion.genome_hash


def test_the_elite_of_a_validation_row_is_its_task_champion(world):
    d = definition(WARM)
    binding = world.runtime.bind(d)
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    for t in ("q4", "q5"):
        pick = world.priors.elite(d, binding, t)
        stored = champ.task(row_hash("capitals", t))
        assert pick.reason is EliteReason.TASK_ELITE and stored is not None
        assert pick.elite.role is EliteRole.TASK
        assert pick.elite.genome_hash == stored.genome_hash
        assert pick.diagnostic.reason is StartReason.WARM_DATASET_CHAMPION
    # an optimization row has no task champion: the dataset (suite) elite, deterministically
    pick = world.priors.elite(d, binding, "q1")
    assert pick.reason is EliteReason.NO_TASK_CHAMPION
    assert pick.elite.role is EliteRole.DATASET
    assert pick.elite.genome_hash == champ.genome_hash
    assert world.priors.elite(d, binding, "q1") == pick


def test_a_stale_or_incompatible_prior_supplies_no_elite(seeded, tmp_path):
    shutil.copytree(seeded, tmp_path / "ws")
    runtime = RealRuntime(versions=DECLARED.model_copy(update={"prompt_template_version": "p/2"}))
    w = World(tmp_path / "ws", runtime)
    d = definition(WARM)
    pick = w.priors.elite(d, runtime.bind(d), "q4")
    assert pick.reason is EliteReason.NO_COMPATIBLE_PRIOR and pick.elite is None
    assert pick.diagnostic.reason is StartReason.COLD_RUNTIME_CHANGED


def test_a_tampered_task_champion_in_a_prior_is_excluded(world):
    prior_id = world.active(PriorKind.DATASET_CHAMPION).prior_id
    conn = drop_triggers(world.prior_store.path, "prior_content_is_immutable")
    (text,) = conn.execute(
        "SELECT record_json FROM priors WHERE prior_id=?", (prior_id,)
    ).fetchone()
    record = json.loads(text)
    record["champion"]["tasks"][0]["validation"]["fitness_mean"] = 0.123
    assert world.priors.verify_provenance(PriorRecord.model_validate(record)) is not None
    conn.execute("UPDATE priors SET record_json=? WHERE prior_id=?", (json.dumps(record), prior_id))
    conn.commit()
    conn.close()
    d = definition(WARM)
    pick = world.priors.elite(d, world.runtime.bind(d), "q4")
    assert pick.elite is None
    assert pick.diagnostic.reason is StartReason.COLD_PROVENANCE_INVALID


def test_deleting_the_dataset_invalidates_its_task_elites(world):
    world.sources.deleted.add((CONTRACT.dataset.dataset_id, CONTRACT.dataset.dataset_version))
    d = definition(WARM)
    pick = world.priors.elite(d, world.runtime.bind(d), "q4")
    assert pick.elite is None and pick.diagnostic.reason is StartReason.COLD_DATASET_DELETED


def test_task_elites_never_cross_workspaces(tmp_path):
    a = World(tmp_path / "a", workspace="ws-alpha")
    a.run(definition())
    d = definition(WARM)
    assert a.priors.elite(d, a.runtime.bind(d), "q4").elite is not None
    b = World(tmp_path / "b", workspace="ws-beta")
    pick = b.priors.elite(d, b.runtime.bind(d), "q4")
    assert pick.elite is None and pick.diagnostic.reason is StartReason.COLD_NO_PRIOR


def test_a_synthetic_job_never_gets_an_elite(world):
    d = definition(WARM, synthetic=True)
    pick = world.priors.elite(d, world.runtime.bind(d), "q4")
    assert pick.elite is None and pick.diagnostic.reason is StartReason.COLD_SYNTHETIC


def test_no_held_out_row_ever_has_a_task_elite(world):
    d = definition(WARM)
    pick = world.priors.elite(d, world.runtime.bind(d), "q6")  # q6 is the test row
    assert pick.reason is EliteReason.NO_TASK_CHAMPION  # never a task elite for a test row
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    assert row_hash("capitals", "q6") not in {t.task for t in champ.tasks}
    for t in ("q1", "q2", "q3"):  # optimization rows never select either
        assert champ.task(row_hash("capitals", t)) is None


def test_per_task_selection_never_reaches_the_optimizer_or_the_budget(monkeypatch):
    """Selection runs after the search: with it disabled, every evaluation (order, split, genome)
    and every other result field is identical - nothing feeds back into the optimizer."""
    import experiments.learning_curves as lc

    def search():
        calls = []

        def evaluate(genome, task, trial, seed):
            calls.append((genome.genome_hash, task.id, trial, seed))
            return synthetic_evaluate(genome, task, trial, seed)

        out = run_search(make_optimizer(MMASACO.name, CONFIG), evaluate, SUITE_A, CONFIG, 2)
        return calls, out

    calls, with_sel = search()
    monkeypatch.setattr(lc, "select_task_champions", lambda *a, **k: [])
    calls_off, without = search()
    assert calls == calls_off
    assert with_sel["task_champions"] and without["task_champions"] == []
    strip = lambda r: {k: v for k, v in r.items() if k != "task_champions"}  # noqa: E731
    assert strip(with_sel) == strip(without)


# == JEV provenance =============================================================================
def test_a_jev_task_champion_keeps_its_pinned_evidence_source(jev_seeded, tmp_path):
    shutil.copytree(jev_seeded, tmp_path / "ws")
    runtime = JevRuntime()
    w = World(tmp_path / "ws", runtime)
    body = w.body(source_job(w))
    for r in body["runs"]:
        assert r["run_versions"][0]["evidence_source"] == JEV_ID
        for t in r["task_champions"]:
            assert t["evidence_sources"] == [JEV_ID]
            assert t["run_versions"][0]["evidence_source"] == JEV_ID
    champ = w.active(PriorKind.DATASET_CHAMPION).record.champion
    assert champ.evidence_source == JEV_ID and champ.tasks
    d = definition(WARM)
    pick = w.priors.elite(d, runtime.bind(d), "q4", evidence_source=JEV_ID)
    assert pick.elite is not None and pick.elite.role is EliteRole.TASK


@pytest.mark.parametrize("now", ["jev:scripted/test@2", None])
def test_another_jev_model_never_reuses_an_incompatible_elite(jev_seeded, tmp_path, now):
    shutil.copytree(jev_seeded, tmp_path / "ws")
    runtime = JevRuntime()
    w = World(tmp_path / "ws", runtime)
    d = definition(WARM)
    pick = w.priors.elite(d, runtime.bind(d), "q4", evidence_source=now)
    assert pick.reason is EliteReason.EVIDENCE_SOURCE_CHANGED and pick.elite is None


def test_a_non_jev_elite_ignores_the_configured_jev(world):
    d = definition(WARM)
    pick = world.priors.elite(d, world.runtime.bind(d), "q4", evidence_source=JEV_ID)
    assert pick.elite is not None and pick.elite.role is EliteRole.TASK


def test_a_prior_record_rejects_an_unsorted_or_mismatched_task_champion(world):
    champ = world.active(PriorKind.DATASET_CHAMPION).record.champion
    t = champ.tasks[0]
    with pytest.raises(ValueError, match="sorted and distinct"):
        Champion.model_validate(
            {**champ.model_dump(mode="json"), "tasks": [t.model_dump(), t.model_dump()]}
        )
    with pytest.raises(ValueError, match="does not match its hash"):
        TaskChampion.model_validate({**t.model_dump(mode="json"), "genome_hash": "0" * 64})


# == product chat ===============================================================================
class _Model:
    model_hash = "scripted-model/1"


def _chat_runner(jev=None):
    pytest.importorskip("agent_framework")
    from experiments.real_runtime import build_runner

    return build_runner(_Model(), jev=jev)


VALIDATION_A = [t.id for t in search_tasks(SUITE_A)[1]]


def _results(tmp_path: Path, records: list[dict], optimizer: str = MMASACO.name) -> Path:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"runs": [{"optimizer": optimizer, "task_champions": records}]}),
        encoding="utf-8",
    )
    return path


def _record(task_id: str, genome: Genome, runner, *, feasible=True, rank=(1.0, 0.9), fit=1.0):
    template = _template(task_id)
    return {
        "task_id": task_id,
        "genome": genome.canonical(),
        "genome_hash": genome.genome_hash,
        "feasible": feasible,
        "rank_key": list(rank) if feasible else [0.0],
        "validation": {"runs": 2, "pass_rate": 1.0, "fitness_mean": fit, "score_mean": fit},
        "run_versions": [runner.versions(template, genome).model_dump(mode="json")],
        "contract_hashes": [template.contract_hash],
    }


def _template(task_id: str):
    from benchmarks.loader import load_task_specs
    from benchmarks.snapshot_store import SnapshotStore

    return legacy_execution_task(load_task_specs()[task_id], SnapshotStore())


def _genomes(n: int) -> list[Genome]:
    from core.constraints import ConstraintChecker
    from optimizers.base import SearchContext

    ctx = SearchContext(contract=SUITE_A.policy, checker=ConstraintChecker(), seed=7, round=0)
    out = MMASACO().propose(n + 4, ctx)
    assert len({g.genome_hash for g in out}) >= n
    return list({g.genome_hash: g for g in out}.values())[:n]


def _versions(runner, template):
    return lambda g: runner.versions(template, g).model_dump(mode="json")


def test_chat_takes_the_compatible_task_elite_then_the_suite_elite(tmp_path):
    from api.chat import Elites

    runner = _chat_runner()
    task_g, other_g, suite_g = _genomes(3)
    tid = VALIDATION_A[0]
    records = [
        _record(tid, other_g, runner, rank=(1.0, 0.2)),
        _record(tid, task_g, runner, rank=(1.0, 0.8)),  # another seed's, better ranked
    ]
    genome = tmp_path / "best.json"
    genome.write_text(suite_g.canonical_json(), encoding="utf-8")
    elites = Elites.load(genome, _results(tmp_path, records))
    template = _template(tid)
    got = elites.candidates(tid, template, _versions(runner, template))
    assert [(k, g.genome_hash) for k, g in got] == [
        ("task", task_g.genome_hash),
        ("suite", suite_g.genome_hash),
    ]


def test_chat_falls_back_to_the_suite_elite_without_a_task_elite(tmp_path):
    from api.chat import Elites

    runner = _chat_runner()
    task_g, suite_g = _genomes(2)
    genome = tmp_path / "best.json"
    genome.write_text(suite_g.canonical_json(), encoding="utf-8")
    train_task = search_tasks(SUITE_A)[0][0].id  # a train row: never a task champion
    elites = Elites.load(genome, _results(tmp_path, [_record(VALIDATION_A[0], task_g, runner)]))
    template = _template(train_task)
    got = elites.candidates(train_task, template, _versions(runner, template))
    assert [(k, g.genome_hash) for k, g in got] == [("suite", suite_g.genome_hash)]
    # an older search result (no task_champions) and no suite genome: no elite at all
    (tmp_path / "old.json").write_text(json.dumps({"runs": [{"optimizer": "aco_mmas"}]}))
    empty = Elites.load(tmp_path / "missing.json", tmp_path / "old.json")
    assert empty.candidates(train_task, template, _versions(runner, template)) == []


def test_chat_never_uses_an_infeasible_or_incompatible_task_elite(tmp_path):
    from api.chat import Elites

    runner = _chat_runner()
    g1, g2, g3 = _genomes(3)
    tid = VALIDATION_A[1]
    other = _record(VALIDATION_A[0], g3, runner)  # recorded for another template task
    stale = _record(tid, g2, runner)
    stale["run_versions"][0]["model_hash"] = "another-model"
    random_run = _record(tid, g3, runner)
    records = [_record(tid, g1, runner, feasible=False), stale, other]
    elites = Elites.load(tmp_path / "none.json", _results(tmp_path, records))
    template = _template(tid)
    assert elites.candidates(tid, template, _versions(runner, template)) == []
    # a random-search run's champion is not the recorded ACO search's
    rs = Elites.load(tmp_path / "none.json", _results(tmp_path, [random_run], "random_search"))
    assert rs.candidates(tid, template, _versions(runner, template)) == []


def test_chat_never_reuses_a_task_elite_measured_with_another_jev_model(tmp_path):
    from api.chat import Elites
    from tests.test_runtime_jev import JEV_GENOME, hosted

    tid = VALIDATION_A[0]
    template = _template(tid)
    sep, oct_ = _chat_runner(hosted("jev-2026-09")), _chat_runner(hosted("jev-2026-10"))
    genome = Genome.from_canonical(JEV_GENOME.canonical())
    elites = Elites.load(tmp_path / "none.json", _results(tmp_path, [_record(tid, genome, sep)]))
    assert [k for k, _ in elites.candidates(tid, template, _versions(sep, template))] == ["task"]
    assert elites.candidates(tid, template, _versions(oct_, template)) == []
    assert elites.candidates(tid, template, _versions(_chat_runner(), template)) == []


def test_chat_routes_by_the_stable_template_task_id_and_labels_the_elite(tmp_path):
    from api.chat import Colony, Engine, Library

    runner = _chat_runner()
    task_g, suite_g = _genomes(2)
    seen: list[str] = []

    class Recording:
        suite = suite_g

        def candidates(self, task_id, template, versions_of):
            seen.append(task_id)
            assert (
                template.id == task_id and versions_of(task_g)["model_hash"] == "scripted-model/1"
            )
            return [("task", task_g), ("suite", suite_g)]

    engine = Engine.__new__(Engine)
    engine.library, engine.runner, engine.elites = Library(), runner, Recording()
    engine.colony, engine.count = Colony(suite_g), 0
    events: list[dict] = []

    class Stop(Exception):
        pass

    def emit(event):
        events.append(event)
        if event["type"] == "ants":
            raise Stop

    sid, name, template_id = next(s for s in engine.library.sources if s[2] in VALIDATION_A)
    for _ in range(2):  # two questions: chat-1, chat-2 - the same template task both times
        with pytest.raises(Stop):
            engine.answer(f"Who is the CEO of {name}?", emit)
    assert seen == [template_id, template_id]
    ants = [e for e in events if e["type"] == "ants"][-1]["ants"]
    elite = ants[0]
    assert (elite["role"], elite["elite"], elite["elite_task"]) == ("elite", "task", template_id)
    assert all(a["role"] == "explorer" and "elite" not in a for a in ants[1:])
    assert ants[0]["stages"] and sid
