"""Behavior regressions for the 2026-10-04 main review; no external models are called."""

import asyncio
import json
import shutil
from dataclasses import replace

import pytest

from benchmarks.loader import BENCH_DIR, benchmark_hash, load_task_specs
from core.constraints import ConstraintChecker
from core.genome import Genome
from core.payloads import Page
from core.results import (
    BudgetCap,
    EvaluatedRun,
    Evaluation,
    FailureInfo,
    FailureKind,
    Verdict,
)
from core.stages import GatherMode, GatherSource, VerifyMethod
from evaluation.gate import DeterministicEvaluator
from experiments.learning_curves import ExperimentConfig, make_evaluate_fn, run_experiment
from experiments.real_runtime import RunCache
from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate
from optimizers.aco_mmas import ACOConfig
from runtime.executors.base import ExecutorInput
from runtime.executors.gather import GatherExecutor
from runtime.gemma_client import GemmaConfig, OpenAICompatibleClient
from runtime.sources import DirectoryApiSource, DirectorySnapshotSource, SourceError
from store.runs import InMemoryRunStore
from tests.conftest import (
    extract,
    gather,
    make_caps,
    make_runtime_task,
    minimal_genome,
    synth,
    verify,
)
from tests.runtime_helpers import make_ctx
from tests.test_evaluation_gate import result_for


@pytest.mark.parametrize("kind", [FailureKind.EXECUTOR_ERROR, FailureKind.MODEL_ERROR])
def test_terminal_failure_cannot_pass_with_a_correct_answer(kind):
    spec = load_task_specs()["A-001"]
    result = result_for(spec, failure=FailureInfo(kind=kind, message="trailing stage failed"))
    evaluation = DeterministicEvaluator().evaluate(spec, result)
    assert evaluation.verdict is Verdict.FAIL
    assert evaluation.fitness == 0


def test_omitted_optional_answer_field_fails_without_keyerror():
    spec = load_task_specs()["A-001"]
    schema = spec.runtime.answer_schema
    fields = tuple(f.model_copy(update={"required": False}) for f in schema.fields)
    runtime = spec.runtime.model_copy(
        update={"answer_schema": schema.model_copy(update={"fields": fields})}
    )
    spec = spec.model_copy(update={"runtime": runtime})
    assert (
        DeterministicEvaluator().evaluate(spec, result_for(spec, values={})).verdict is Verdict.FAIL
    )


def test_custom_benchmark_hash_includes_its_own_snapshot_bytes(tmp_path):
    shutil.copytree(BENCH_DIR, tmp_path / "benchmark")
    root = tmp_path / "benchmark"
    before = benchmark_hash(root)
    page = root / "snapshots" / "A-001" / "pages" / "overview.txt"
    page.write_bytes(page.read_bytes() + b"\nnew snapshot content\n")
    assert benchmark_hash(root) != before


def test_run_store_replaces_a_stale_evaluation_for_the_same_execution():
    spec = load_task_specs()["A-001"]
    execution = result_for(spec)
    old = EvaluatedRun(
        execution=execution,
        evaluation=Evaluation(verdict=Verdict.FAIL, fitness=0, evaluator_version="old"),
    )
    new = DeterministicEvaluator().evaluate_run(spec, execution)
    store = InMemoryRunStore()
    store.save_run(old)
    store.save_run(new)
    assert store.get_run(new.run_id) == new


@pytest.mark.parametrize("source", [DirectorySnapshotSource, DirectoryApiSource])
@pytest.mark.parametrize("snapshot", ["../outside", "absolute"])
def test_directory_sources_reject_snapshot_paths_outside_root(tmp_path, source, snapshot):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "api").mkdir(parents=True)
    snapshot = str(outside) if snapshot == "absolute" else snapshot
    with pytest.raises(SourceError):
        src = source(root)
        if source is DirectorySnapshotSource:
            src.list_page_ids(snapshot)
        else:
            src.list_endpoints(snapshot)


def test_cache_recovers_truncated_tail_and_future_appends(tmp_path):
    result = result_for(load_task_specs()["A-001"])
    path = tmp_path / "runs.jsonl"
    path.write_text(result.model_dump_json() + '\n{"key":', encoding="utf-8")
    cache = RunCache(path)
    assert cache.get(result.run_id) == result
    other = result.model_copy(update={"key": result.key.model_copy(update={"seed": 42})})
    cache.put(other)
    assert RunCache(path).get(other.run_id) == other


def test_cache_does_not_reuse_wall_time_breaches(tmp_path):
    result = result_for(
        load_task_specs()["A-001"],
        failure=FailureInfo(
            kind=FailureKind.BUDGET_EXCEEDED, cap=BudgetCap.WALL_TIME, message="slow endpoint"
        ),
    )
    cache = RunCache(tmp_path / "cache.jsonl")
    cache.put(result)
    assert cache.get(result.run_id) is None


@pytest.mark.parametrize("period", [0, -1])
def test_aco_rejects_invalid_global_best_period(period):
    with pytest.raises(ValueError):
        ACOConfig(global_best_period=period)


def test_model_identity_distinguishes_endpoint_and_structured_mode():
    config = GemmaConfig(base_url="https://one.example/v1", model="gemma", revision="r1")
    hashes = {
        OpenAICompatibleClient(c).model_hash
        for c in (
            config,
            replace(config, base_url="https://two.example/v1"),
            replace(config, structured=False),
        )
    }
    assert len(hashes) == 3


def test_same_id_modified_task_is_rejected_before_execution():
    specs = load_task_specs()

    def run(*args):
        raise AssertionError("mismatched task reached runtime")

    evaluate = make_evaluate_fn(run, DeterministicEvaluator(), specs)
    task = specs["A-001"].runtime.model_copy(update={"question": "a different question"})
    with pytest.raises(ValueError, match="task"):
        evaluate(minimal_genome(), task, 0, 0)


def test_uncalibrated_estimates_do_not_reject_feasible_workflows():
    task = make_runtime_task(caps=make_caps(tokens=100, tool_calls=1, wall_time_s=1))
    genome = Genome.of(gather(GatherSource.API), extract(), synth())
    assert ConstraintChecker().is_valid(genome, task)


def test_partial_gather_failure_still_charges_all_attempted_reads():
    class Source:
        def list_page_ids(self, snapshot_id):
            return ["good", "bad"]

        def read_page(self, snapshot_id, page_id):
            if page_id == "bad":
                raise SourceError("read failed")
            return Page(page_id=page_id, source_ref="test", content="ok")

    task = make_runtime_task()
    inp = ExecutorInput(stage_index=0, stage=gather(mode=GatherMode.PARALLEL_2), payload=task)
    output = asyncio.run(GatherExecutor(pages=Source()).run(inp, make_ctx(task)))
    assert output.failure.kind is FailureKind.EXECUTOR_ERROR
    assert output.usage.tool_calls == 2
    assert output.metrics.pages_fetched == 1


def test_experiment_propagates_lcb_z_to_aco(monkeypatch):
    from benchmarks.loader import runtime_tasks
    from core.task_spec import TaskClass
    from experiments import learning_curves

    real_run_search = learning_curves.run_search

    def checked_search(optimizer, *args, **kwargs):
        if optimizer.name == "aco_mmas":
            assert optimizer.config.lcb_z == 2.5
        return real_run_search(optimizer, *args, **kwargs)

    monkeypatch.setattr(learning_curves, "run_search", checked_search)
    run_experiment(
        synthetic_evaluate,
        runtime_tasks("train", TaskClass.A),
        runtime_tasks("validation", TaskClass.A),
        optimizers=("aco_mmas",),
        seeds=(0,),
        config=ExperimentConfig(budget=10, lcb_z=2.5),
        synthetic=True,
        evaluator_version=SYNTHETIC_VERSION,
    )


def test_transient_model_failure_is_retried_without_penalizing_genome():
    from experiments.learning_curves import run_search
    from optimizers.aco_mmas import MMASACO

    task = make_runtime_task()
    seen = set()

    def evaluate(genome, task, trial, seed):
        run = synthetic_evaluate(genome, task, trial, seed)
        if run.run_id not in seen:
            seen.add(run.run_id)
            return run.model_copy(
                update={
                    "execution": run.execution.model_copy(
                        update={
                            "failure": FailureInfo(
                                kind=FailureKind.MODEL_ERROR, message="temporary"
                            )
                        }
                    ),
                    "evaluation": run.evaluation.model_copy(
                        update={"verdict": Verdict.FAIL, "fitness": 0}
                    ),
                }
            )
        return run

    config = ExperimentConfig(budget=2, batch_size=1, trials=1)
    actual = run_search(MMASACO(), evaluate, [task], [task], config, 0)
    expected = run_search(MMASACO(), synthetic_evaluate, [task], [task], config, 0)
    assert actual == expected


def test_persistent_model_outage_stops_search_without_pheromone_update():
    from experiments.learning_curves import run_search
    from optimizers.aco_mmas import MMASACO

    task = make_runtime_task()

    def evaluate(genome, task, trial, seed):
        run = synthetic_evaluate(genome, task, trial, seed)
        return run.model_copy(
            update={
                "execution": run.execution.model_copy(
                    update={"failure": FailureInfo(kind=FailureKind.MODEL_ERROR, message="down")}
                )
            }
        )

    optimizer = MMASACO()
    with pytest.raises(RuntimeError, match="model"):
        run_search(
            optimizer,
            evaluate,
            [task],
            [task],
            ExperimentConfig(budget=1, batch_size=1, trials=1),
            0,
        )
    assert optimizer.epoch == 0


def test_runtime_checker_prunes_unimplemented_options_for_both_optimizers():
    from optimizers.aco_mmas import MMASACO
    from optimizers.base import SearchContext
    from optimizers.random_search import RandomSearch
    from runtime.runner import WorkflowRunner

    task = make_runtime_task()
    checker = WorkflowRunner(model=None, benchmark_hash="test").checker
    assert not checker.is_valid(Genome.of(gather(GatherSource.JEV), extract(), synth()), task)
    assert not checker.is_valid(
        Genome.of(gather(), extract(), synth(), verify(VerifyMethod.SELF_CONSISTENCY)), task
    )
    for optimizer in (MMASACO(), RandomSearch()):
        genomes = optimizer.propose(30, SearchContext(task=task, checker=checker, seed=0))
        assert genomes
        for genome in genomes:
            assert all(
                getattr(stage, "source", None) is not GatherSource.JEV
                and getattr(stage, "method", None) is not VerifyMethod.SELF_CONSISTENCY
                for stage in genome.stages
            )


def test_warm_start_rejects_changed_aco_parameters():
    from memory.warm_start import aco_from_memory
    from tests.test_workflow_memory import fixture_memory

    memory = fixture_memory()
    _, report = aco_from_memory(memory, memory.key, ACOConfig(rho=0.7))
    assert report.mode == "cold"


def test_runtime_import_does_not_require_optional_maf():
    import subprocess
    import sys

    script = """
import sys
sys.modules["agent_framework"] = None
from runtime.runner import WorkflowRunner
from experiments.real_runtime import build_runner
assert WorkflowRunner and build_runner
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_evaluation_log_resets_and_labels_single_run_max(tmp_path):
    from experiments.run_mvp import EvalLog

    path = tmp_path / "evaluations.jsonl"
    path.write_text('{"old": true}\n', encoding="utf-8")
    log = EvalLog(path, set())
    logged = log.wrap(synthetic_evaluate, "aco_mmas", 0)
    logged(minimal_genome(), make_runtime_task(), 0, 0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert "best_so_far_fitness" not in rows[0]
    assert rows[0]["best_single_run_fitness"] == rows[0]["fitness"]


def test_heldout_split_is_disjoint_and_hash_pinned():
    from benchmarks.loader import load_splits

    heldout_dir = BENCH_DIR / "heldout"
    specs = load_task_specs(heldout_dir)
    splits = load_splits(heldout_dir)
    assert set(splits) == {"test"}
    assert set(splits["test"]) == set(specs)
    assert len(specs) == 16
    assert not set(specs) & set(load_task_specs())
    assert (
        benchmark_hash(heldout_dir)
        == "f88b7fc4649b49fd581f6ae0f1e95cd0691209b144301576a800b270442f296d"
    )


def test_final_uses_heldout_tasks_for_saved_task_class(tmp_path, monkeypatch):
    from argparse import Namespace

    from experiments import run_mvp

    (tmp_path / "best_aco_genome.json").write_text(
        minimal_genome().canonical_json(), encoding="utf-8"
    )
    (tmp_path / "results.json").write_text(json.dumps({"task_class": "B"}), encoding="utf-8")
    monkeypatch.setattr(run_mvp, "client_from_env", lambda: None)
    monkeypatch.setattr(run_mvp, "real_evaluate_fn", lambda *a, **kw: synthetic_evaluate)
    run_mvp.cmd_final(Namespace(out=tmp_path, task=None, seed=0))
    result = json.loads((tmp_path / "final_test.json").read_text(encoding="utf-8"))
    assert result["split"] == "test"
    assert len(result["runs"]) == 8
    assert all(run["task"].startswith("TB-") for run in result["runs"])


def test_final_refuses_validation_tasks(tmp_path, monkeypatch):
    from argparse import Namespace

    from experiments import run_mvp

    (tmp_path / "best_aco_genome.json").write_text(
        minimal_genome().canonical_json(), encoding="utf-8"
    )
    (tmp_path / "results.json").write_text(json.dumps({"task_class": "A"}), encoding="utf-8")
    monkeypatch.setattr(run_mvp, "client_from_env", lambda: None)
    with pytest.raises(ValueError, match="held-out"):
        run_mvp.cmd_final(Namespace(out=tmp_path, task="A-001", seed=0))


def test_evaluation_log_marks_infrastructure_failures_unscored(tmp_path):
    from experiments.run_mvp import EvalLog

    def evaluate(*args):
        run = synthetic_evaluate(*args)
        return run.model_copy(
            update={
                "execution": run.execution.model_copy(
                    update={"failure": FailureInfo(kind=FailureKind.MODEL_ERROR, message="down")}
                )
            }
        )

    path = tmp_path / "log.jsonl"
    logged = EvalLog(path, set()).wrap(evaluate, "aco_mmas", 0)
    logged(minimal_genome(), make_runtime_task(), 0, 0)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["evaluation"] == 0
    assert row["scored"] is False
    assert row["best_single_run_fitness"] is None


def test_final_does_not_publish_model_outages_as_heldout_failures(tmp_path, monkeypatch):
    from argparse import Namespace

    from experiments import run_mvp

    (tmp_path / "best_aco_genome.json").write_text(
        minimal_genome().canonical_json(), encoding="utf-8"
    )
    (tmp_path / "results.json").write_text(json.dumps({"task_class": "A"}), encoding="utf-8")
    monkeypatch.setattr(run_mvp, "client_from_env", lambda: None)

    def evaluate(*args):
        run = synthetic_evaluate(*args)
        return run.model_copy(
            update={
                "execution": run.execution.model_copy(
                    update={"failure": FailureInfo(kind=FailureKind.MODEL_ERROR, message="down")}
                )
            }
        )

    monkeypatch.setattr(run_mvp, "real_evaluate_fn", lambda *a, **kw: evaluate)
    with pytest.raises(RuntimeError, match="model"):
        run_mvp.cmd_final(Namespace(out=tmp_path, task=None, seed=0))
    assert not (tmp_path / "final_test.json").exists()
