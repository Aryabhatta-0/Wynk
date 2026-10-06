"""Experiment harness. All numbers here come from the SYNTHETIC fake objective or a test-only
oracle runner; none of it is a benchmark result."""

import json

import pytest

from benchmarks.legacy_adapter import legacy_evaluator, legacy_references, legacy_suite
from benchmarks.snapshot_store import SnapshotStore
from core.evidence import FieldEvidence
from core.payloads import Answer
from core.results import BudgetUsage, ExecutionResult, RunKey, RunVersions
from core.run_contract import ContractSuite
from evaluation.gate import EVALUATOR_VERSION
from experiments.learning_curves import (
    ExperimentConfig,
    make_evaluate_fn,
    run_experiment,
    run_search,
    search_tasks,
    write_results,
)
from experiments.report import aggregate_curves, plot_learning_curves
from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate

SUITE = legacy_suite("A")
TRAIN, VAL = search_tasks(SUITE)
CONFIG = ExperimentConfig(budget=60, batch_size=2, trials=2)


def run(seeds=(0, 1), **kw):
    return run_experiment(
        synthetic_evaluate,
        SUITE,
        seeds=seeds,
        config=CONFIG,
        synthetic=True,
        evaluator_version=SYNTHETIC_VERSION,
        **kw,
    )


def test_harness_runs_both_optimizers_and_reports_the_required_fields():
    results = run()
    assert results["synthetic"] is True
    assert {r["optimizer"] for r in results["runs"]} == {"random_search", "aco_mmas"}
    per_genome = len(TRAIN) * CONFIG.trials
    for r in results["runs"]:
        assert r["workflow_evaluations"] == (CONFIG.budget // per_genome) * per_genome
        assert r["workflow_evaluations"] <= CONFIG.budget
        assert isinstance(r["best_so_far_fitness"], float)
        assert 0.0 <= r["pass_rate"] <= 1.0
        assert len(r["best_genome_hash"]) == 64
        xs = [p["evaluations"] for p in r["curve"]]
        assert xs == sorted(xs) and xs[-1] == r["workflow_evaluations"]
        # best-so-far (train, lcb) is monotone non-decreasing by construction of the incumbent
        best = [p["best_so_far_train_fitness"] for p in r["curve"]]
        assert best == sorted(best)


def test_harness_is_deterministic_and_json_serialisable(tmp_path):
    a, b = run(), run()
    assert a == b
    write_results(a, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(a)
    )


def test_parallel_workers_give_identical_results():
    from optimizers.aco_mmas import MMASACO

    a = run_search(MMASACO(), synthetic_evaluate, SUITE, CONFIG, seed=3)
    b = run_search(MMASACO(), synthetic_evaluate, SUITE, CONFIG, seed=3, workers=4)
    assert a == b


def test_different_seeds_give_different_searches():
    results = run(seeds=(0, 1), optimizers=("random_search",))
    h0, h1 = (r["best_genome_hash"] for r in results["runs"])
    assert h0 != h1


def test_unlabelled_synthetic_results_are_refused(tmp_path):
    bad = dict(run(seeds=(0,)), evaluator_version="evaluator/mvp-1")
    with pytest.raises(ValueError):
        write_results(bad, tmp_path / "x.json")


def test_tasks_with_different_search_policies_cannot_share_a_suite():
    # A and B differ in caps and allowed sources: one genome cannot be admitted for both.
    b = legacy_suite("B")
    with pytest.raises(ValueError, match="does not share"):
        ContractSuite(name="mixed", tasks=(*SUITE.tasks, *b.tasks), splits=SUITE.splits)


def test_aggregate_and_plot_learning_curve(tmp_path):
    pytest.importorskip("matplotlib")
    results = run()
    agg = aggregate_curves(results)
    assert set(agg) == {"random_search", "aco_mmas"}
    assert all(c["x"] == sorted(c["x"]) and len(c["x"]) == len(c["mean"]) for c in agg.values())
    out = tmp_path / "curve.png"
    plot_learning_curves(results, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# -- wiring for the real runtime: oracle runner is TEST-ONLY (it peeks at ground truth) --------
def test_make_evaluate_fn_wires_a_runner_and_the_real_evaluator_end_to_end():
    refs = legacy_references()
    store = SnapshotStore()
    versions = RunVersions(
        model_hash="m",
        prompt_template_version="p",
        benchmark_hash="b",
        compiler_version="c",
        grammar_version="g",
    )

    def oracle_runner(genome, task, trial, seed):
        expected = refs[task.id]  # test-only: a real runtime never has this
        page = store.pages(task.source.snapshot_id).pages[0]
        evidence = tuple(FieldEvidence(field=f, spans=(page.span(0, 5),)) for f in expected)
        return ExecutionResult(
            key=RunKey(
                genome_hash=genome.genome_hash,
                task_id=task.id,
                contract_hash=task.contract_hash,
                trial=trial,
                seed=seed,
                versions=versions,
            ),
            answer=Answer(values=dict(expected), evidence=evidence),
            budget_usage=BudgetUsage(tokens=800),
        )

    evaluate = make_evaluate_fn(oracle_runner, legacy_evaluator(), (*TRAIN, *VAL))
    results = run_experiment(
        evaluate,
        SUITE,
        seeds=(0,),
        config=ExperimentConfig(budget=20),
        synthetic=True,  # oracle runner is not a real runtime result
        evaluator_version="synthetic-oracle+" + EVALUATOR_VERSION,
    )
    assert all(r["pass_rate"] == 1.0 and r["train_pass_rate"] == 1.0 for r in results["runs"])
