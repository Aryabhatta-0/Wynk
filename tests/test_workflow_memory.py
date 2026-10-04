"""Persistent workflow memory + ACO warm start.

Data here comes from the SYNTHETIC fake objective or from hand-made TEST/FIXTURE memories;
none of it is a learned benchmark result.
"""

import ast
from pathlib import Path

import pytest

from benchmarks.loader import runtime_tasks
from core.canonical import canonical_hash
from core.constraints import ConstraintChecker
from core.genome import Genome
from core.grammar import GRAMMAR_VERSION
from core.stages import GatherMode, GatherSource
from core.task_spec import TaskClass
from experiments.learning_curves import ExperimentConfig, run_experiment, run_search
from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate
from memory import summary
from memory.models import (
    EdgePheromone,
    MemoryKey,
    WorkflowMemory,
    WorkflowRecord,
    genome_path,
    node_label,
)
from memory.store import WorkflowMemoryStore, dumps, loads
from memory.warm_start import (
    IncompatibleMemory,
    aco_from_memory,
    run_aco,
    synthetic_key,
    update_from_experiment,
)
from optimizers.aco_mmas import MMASACO
from optimizers.base import SearchContext
from optimizers.construct import END, START, node_key
from tests.conftest import extract, gather, synth

TRAIN = runtime_tasks("train", TaskClass.B)
VAL = runtime_tasks("validation", TaskClass.B)
CONFIG = ExperimentConfig(budget=60)
LEARN_CONFIG = ExperimentConfig(budget=200)  # long enough for >3 distinct incumbents
STAMP = "2026-10-04T00:00:00+00:00"
FIXTURE_KEY = MemoryKey(
    task_class="B",
    grammar_version=GRAMMAR_VERSION,
    optimizer_version=MMASACO.version,
    benchmark_hash="TEST/FIXTURE",
    model_hash="TEST/FIXTURE",
    evaluator_version="TEST/FIXTURE",
)


def fixture_record(source: GatherSource, fitness: float) -> WorkflowRecord:
    g = Genome.of(gather(source, GatherMode.SEQUENTIAL), extract(), synth())
    return WorkflowRecord(
        genome_hash=g.genome_hash,
        path=genome_path(g),
        genome=g,
        validation_fitness=fitness,
        pass_rate=0.5,
        mean_tokens=1000.0,
        mean_wall_time_s=1.5,
        validation_runs=4,
    )


def fixture_memory(**kw) -> WorkflowMemory:
    """TEST/FIXTURE memory - hand-made, not learned."""
    a = node_key(gather(GatherSource.API, GatherMode.SEQUENTIAL))
    b = node_key(extract())
    fields = dict(
        key=FIXTURE_KEY,
        source="TEST/FIXTURE",
        updated_at=STAMP,
        runs_merged=1,
        workflows=(fixture_record(GatherSource.API, 0.9), fixture_record(GatherSource.FETCH, 0.5)),
        pheromones=(
            EdgePheromone(src=START, dst=a, tau=0.8, label=f"START -> {node_label(a)}"),
            EdgePheromone(src=a, dst=b, tau=0.3, label=f"{node_label(a)} -> {node_label(b)}"),
        ),
        aco_epoch=0,
    )
    return WorkflowMemory(**{**fields, **kw})


@pytest.fixture(scope="module")
def learned():
    """One cold synthetic ACO run and the memory learned from it."""
    opt = MMASACO()
    result = run_search(opt, synthetic_evaluate, TRAIN, VAL, LEARN_CONFIG, seed=0)
    return result, opt, update_from_experiment(result, opt, updated_at=STAMP)


def proposals(opt, seed=3, n=8):
    ctx = SearchContext(task=TRAIN[0], checker=ConstraintChecker(), seed=seed)
    return [g.genome_hash for g in opt.propose(n, ctx)]


# -- persistence ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path, learned):
    store = WorkflowMemoryStore(tmp_path)
    for memory in (fixture_memory(), learned[2]):
        store.save(memory)
        assert store.load("B") == memory


def test_file_bytes_are_deterministic(tmp_path, learned):
    store = WorkflowMemoryStore(tmp_path)
    p = store.save(learned[2])
    first = p.read_bytes()
    store.save(store.load("B"))
    assert p.read_bytes() == first
    rebuilt = update_from_experiment(*learned[:2], updated_at=STAMP)
    assert dumps(rebuilt) == first == dumps(loads(first))
    assert b"\r\n" not in first


def test_load_missing_and_clear(tmp_path):
    store = WorkflowMemoryStore(tmp_path)
    assert store.load("B") is None
    store.save(fixture_memory())
    assert store.clear("B") is True
    assert store.load("B") is None and store.clear("B") is False


def test_file_for_another_class_is_rejected(tmp_path):
    store = WorkflowMemoryStore(tmp_path)
    p = store.save(fixture_memory())
    p.rename(tmp_path / "A.json")
    with pytest.raises(ValueError):
        store.load("A")


# -- learning ------------------------------------------------------------------------------
def test_memory_comes_from_the_experiment_result(learned):
    result, opt, memory = learned
    assert memory.source == "synthetic" and memory.key == synthetic_key("B")
    assert (
        memory.workflows[0].genome_hash
        == max(result["workflows"], key=lambda w: w["validation_fitness"])["genome_hash"]
    )
    assert memory.pheromone_map() == opt.explored_pheromones()
    assert memory.aco_epoch == opt.epoch > 0


def test_keeps_only_top_three(learned):
    result, opt, memory = learned
    assert len(result["workflows"]) >= 3
    fits = [w.validation_fitness for w in memory.workflows]
    assert len(fits) == 3 and fits == sorted(fits, reverse=True)
    assert fits[0] == max(w["validation_fitness"] for w in result["workflows"])
    # 2 remembered + >=3 new candidates -> still 3: the best old one survives, the worst goes
    old = (fixture_record(GatherSource.FETCH, 99.0), fixture_record(GatherSource.JEV, -5.0))
    prev = memory.model_copy(update={"workflows": old, "runs_merged": 4})
    merged = update_from_experiment(result, opt, previous=prev, updated_at=STAMP)
    assert len(merged.workflows) == 3 and merged.workflows[0].validation_fitness == 99.0
    assert -5.0 not in [w.validation_fitness for w in merged.workflows]
    assert merged.runs_merged == 5
    with pytest.raises(ValueError):
        fixture_memory(workflows=(fixture_record(GatherSource.API, 1.0),) * 4)


def test_merge_into_incompatible_memory_is_refused(learned):
    with pytest.raises(IncompatibleMemory):
        update_from_experiment(*learned[:2], previous=fixture_memory())


def test_mixed_versions_in_results_are_refused(learned):
    result, opt, _ = learned
    bad = dict(result, versions=[*result["versions"], {**result["versions"][0], "x": 1}])
    with pytest.raises(ValueError):
        update_from_experiment(bad, opt)


def test_only_aco_runs_can_be_learned_from(learned):
    result, opt, _ = learned
    with pytest.raises(ValueError):
        update_from_experiment(dict(result, optimizer="random_search"), opt)


# -- compatibility -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("task_class", "A"),
        ("grammar_version", "grammar/0"),
        ("optimizer_version", "aco_mmas/0"),
        ("benchmark_hash", "other-bench"),
        ("model_hash", "other-model"),
        ("evaluator_version", "evaluator/other"),
    ],
)
def test_incompatible_memory_is_rejected_with_a_reason(field, value):
    expected = FIXTURE_KEY.model_copy(update={field: value})
    opt, report = aco_from_memory(fixture_memory(), expected)
    assert report.mode == "cold" and report.edges_loaded == 0
    assert any(r.startswith(f"{field}:") for r in report.reasons)
    assert proposals(opt) == proposals(MMASACO())


def test_synthetic_memory_never_warm_starts_a_real_search(learned):
    real = synthetic_key("B").model_copy(
        update={"benchmark_hash": "bench", "model_hash": "gemma", "evaluator_version": "eval/1"}
    )
    assert aco_from_memory(learned[2], real)[1].mode == "cold"


def test_missing_memory_cold_starts_and_says_so():
    _, report = aco_from_memory(None, FIXTURE_KEY)
    assert report.mode == "cold" and "no memory" in report.reasons[0]


# -- warm start ----------------------------------------------------------------------------
def test_pheromone_is_restored_exactly(learned):
    _, opt, memory = learned
    warm, report = aco_from_memory(memory, memory.key)
    assert report.mode == "warm" and report.edges_loaded == len(memory.pheromones)
    for (a, b), tau in memory.pheromone_map().items():
        assert warm.pheromone((a, b)) == pytest.approx(tau)
    # the whole learned state comes back, unknown edges included
    unseen = ("never", "seen")
    assert warm.pheromone(unseen) == pytest.approx(opt.pheromone(unseen))


def test_unknown_edges_keep_the_normal_default():
    memory = fixture_memory()
    warm, _ = aco_from_memory(memory, FIXTURE_KEY)
    cold = MMASACO()
    unknown = (node_key(extract()), node_key(synth()))
    assert unknown not in memory.pheromone_map()
    assert warm.pheromone(unknown) == cold.pheromone(unknown) == cold.config.tau_max
    assert warm.pheromone((START, END)) == cold.config.tau_max
    # with remembered epochs, unknown edges follow MMAS's own evaporation of tau_max
    aged, _ = aco_from_memory(fixture_memory(aco_epoch=3), FIXTURE_KEY)
    cfg = cold.config
    assert aged.pheromone(unknown) == pytest.approx(cfg.tau_max * (1 - cfg.rho) ** 3)


def test_same_memory_and_seed_give_same_proposals(learned):
    memory = learned[2]
    a = proposals(aco_from_memory(memory, memory.key)[0])
    b = proposals(aco_from_memory(loads(dumps(memory)), memory.key)[0])
    assert a == b
    assert a != proposals(MMASACO())  # and the memory actually changes the search


def test_cold_start_behaviour_is_unchanged(tmp_path, learned):
    # empty warm start == cold
    assert proposals(MMASACO.from_pheromones({})) == proposals(MMASACO())
    # run_aco(start="cold") ignores stored memory
    store = WorkflowMemoryStore(tmp_path)
    store.save(learned[2])
    cold, _, report = run_aco(
        synthetic_evaluate,
        TRAIN,
        VAL,
        start="cold",
        expected=learned[2].key,
        store=store,
        seed=0,
        config=LEARN_CONFIG,
    )
    assert report.mode == "cold" and cold == learned[0]
    # the headline ACO-vs-random experiment is byte-for-byte what it was before memory existed
    r = run_experiment(
        synthetic_evaluate,
        TRAIN,
        VAL,
        seeds=(0, 1),
        config=CONFIG,
        synthetic=True,
        evaluator_version=SYNTHETIC_VERSION,
    )
    keys = [
        "optimizer",
        "seed",
        "workflow_evaluations",
        "best_so_far_fitness",
        "validation_fitness",
        "pass_rate",
        "train_pass_rate",
        "best_genome_hash",
        "curve",
    ]
    golden = "acc5d8659dd059844b28eaf8ab0a535c7953045746a07ffd8f538f14e20954b2"
    assert canonical_hash([{k: run[k] for k in keys} for run in r["runs"]]) == golden


def test_warm_run_uses_stored_memory(tmp_path, learned):
    store = WorkflowMemoryStore(tmp_path)
    store.save(learned[2])
    _, opt, report = run_aco(
        synthetic_evaluate,
        TRAIN,
        VAL,
        start="warm",
        expected=learned[2].key,
        store=store,
        seed=1,
        config=CONFIG,
    )
    assert report.mode == "warm" and opt.epoch > learned[2].aco_epoch


# -- summary -------------------------------------------------------------------------------
def test_summary_is_deterministic_and_labelled(tmp_path, learned, capsys):
    text = summary.render(learned[2])
    assert text == summary.render(learned[2])
    assert text.startswith("Class B memory:") and "SYNTHETIC" in text
    assert "Best workflow: GATHER(" in text and "Strong edges:" in text
    assert "TEST/FIXTURE" in summary.render(fixture_memory())
    WorkflowMemoryStore(tmp_path).save(learned[2])
    assert summary.main(["B", "--dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == text
    assert summary.main(["A", "--dir", str(tmp_path)]) == 1


def test_memory_package_never_touches_ground_truth_or_the_evaluator():
    for f in sorted((Path(__file__).resolve().parent.parent / "memory").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        assert "ground_truth" not in src.lower()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "evaluation", f.name
                assert not {"TaskSpec", "GroundTruth"} & {a.name for a in node.names}, f.name
