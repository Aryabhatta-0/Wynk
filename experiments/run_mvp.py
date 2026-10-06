"""Real MVP driver: Gemma (via ``GEMMA_*`` env) + MAF runtime + deterministic evaluator.

    python -m experiments.run_mvp smoke                       # one hand-built genome, one task
    python -m experiments.run_mvp experiment --budget 60 --seeds 1 --workers 4
    python -m experiments.run_mvp final                       # held-out test tasks

Every evaluation (train and validation) is appended to ``<out>/evaluations.jsonl``.

The frozen benchmark enters only through ``benchmarks.legacy_adapter.legacy_suite``: search,
execution and evaluation run on the resulting contracts, and the suite's splits decide which
tasks are searched (optimization), selected on (validation) and reported (test, ``final``).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.legacy_adapter import legacy_evaluator, legacy_suite
from benchmarks.loader import BENCH_DIR, HELDOUT_DIR, benchmark_hash
from benchmarks.snapshot_store import SnapshotStore
from core.dataset import SplitRole, SplitUse
from core.genome import Genome
from core.results import EvaluatedRun, FailureKind
from core.run_contract import ExecutionTask, SnapshotSource
from evaluation.contract_eval import ContractEvaluator
from experiments.learning_curves import (
    OPTIMIZER_FACTORIES,
    RESULTS_SCHEMA,
    EvaluateFn,
    ExperimentConfig,
    evaluate_with_retries,
    make_optimizer,
    run_search,
    search_tasks,
    write_results,
)
from experiments.real_runtime import RunCache, build_runner, dumps, real_evaluate_fn, run_summary
from experiments.report import plot_learning_curves
from runtime.gemma_client import client_from_env
from runtime.mvp_genomes import GENOME_A

OUT = Path("experiments/results/real")


def stage_path(genome: Genome) -> str:
    def one(s) -> str:
        d = s.model_dump(mode="json")
        kind = d.pop("kind")
        return f"{kind.upper()}({', '.join(str(v) for v in d.values())})"

    return " -> ".join(one(s) for s in genome.stages)


def with_span_text(summary: dict, snapshot_id: str, store: SnapshotStore) -> dict:
    for fe in summary["evidence"]:
        for s in fe["spans"]:
            page = store.page(snapshot_id, s["page_id"])
            s["text"] = page.content[s["char_start"] : s["char_end"]] if page else None
    return summary


def evaluator_for(bench_dir: Path) -> ContractEvaluator:
    return legacy_evaluator(bench_dir)


def print_run(task: ExecutionTask, genome: Genome, run: EvaluatedRun, store: SnapshotStore) -> dict:
    assert isinstance(task.source, SnapshotSource)
    s = with_span_text(run_summary(run), task.source.snapshot_id, store)
    print(f"task      : {task.id}  {task.example.values['question']}")
    print(f"workflow  : {stage_path(genome)}")
    print(f"answer    : {s['answer']}")
    for fe in s["evidence"]:
        for sp in fe["spans"]:
            print(
                f"evidence  : {fe['field']} <- {sp['page_id']}[{sp['char_start']}:{sp['char_end']}]"
                f" {sp['text']!r}"
            )
    print(f"verdict   : {s['verdict']}   fitness {s['fitness']:.3f}   fields {s['field_results']}")
    print(
        f"tokens {s['tokens']}  wall {s['wall_time_s']}s  tool calls {s['tool_calls']}"
        f"  model calls {s['model_calls']}  retries {s['retries']}"
    )
    print(f"stages    : {[(t['kind'], t['status']) for t in s['stages']]}")
    if s["failure"]:
        print(f"failure   : {s['failure']}")
    return s


def cmd_smoke(args: argparse.Namespace) -> None:
    store = SnapshotStore()
    runner = build_runner(client_from_env(), store)
    train, _ = search_tasks(legacy_suite("A"))
    task = next(t for t in train if t.id == args.task)
    evaluate = real_evaluate_fn(runner, evaluator_for(BENCH_DIR), [task])
    run = evaluate(GENOME_A, task, 0, 0)
    print_run(task, GENOME_A, run, store)


class EvalLog:
    """Appends one line per evaluation: optimizer, n, genome hash, fitness, verdict, best-so-far."""

    def __init__(self, path: Path, validation_ids: set[str]) -> None:
        self.path = path
        self.validation_ids = validation_ids  # validation runs: measured, not budgeted
        self._lock = threading.Lock()
        self.genomes: dict[str, Genome] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def wrap(self, evaluate: EvaluateFn, optimizer: str, seed: int) -> EvaluateFn:
        state = {"n": 0, "best": None}

        def logged(genome: Genome, task: ExecutionTask, trial: int, s: int) -> EvaluatedRun:
            run = evaluate(genome, task, trial, s)
            ev = run.evaluation
            validation = task.id in self.validation_ids
            failure = run.execution.failure
            scored = failure is None or failure.kind is not FailureKind.MODEL_ERROR
            # runs of one search complete concurrently: counter, best and the write share a lock
            with self._lock:
                self.genomes[genome.genome_hash] = genome
                if not validation and scored:
                    state["n"] += 1
                    state["best"] = (
                        ev.fitness if state["best"] is None else max(state["best"], ev.fitness)
                    )
                row = {
                    "optimizer": optimizer,
                    "seed": seed,
                    "split": "validation" if validation else "train",
                    "evaluation": state["n"],  # completion order when --workers > 1
                    "task_id": task.id,
                    "genome_hash": genome.genome_hash,
                    "fitness": ev.fitness,
                    "verdict": ev.verdict.value,
                    "scored": scored,
                    "failure": failure.model_dump(mode="json") if failure else None,
                    "best_single_run_fitness": state["best"],
                }
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            return run

        return logged


def cmd_experiment(args: argparse.Namespace) -> None:
    out: Path = args.out
    suite = legacy_suite(args.task_class)
    train, val = search_tasks(suite)
    store = SnapshotStore()
    runner = build_runner(client_from_env(), store)
    evaluator = evaluator_for(BENCH_DIR)
    evaluate = real_evaluate_fn(
        runner, evaluator, (*train, *val), RunCache(out / "run_cache.jsonl")
    )
    config = ExperimentConfig(budget=args.budget, batch_size=args.batch, trials=args.trials)
    log = EvalLog(out / "evaluations.jsonl", {t.id for t in val})
    jobs = [(name, seed) for name in OPTIMIZER_FACTORIES for seed in range(args.seeds)]
    t0 = time.time()
    print(
        f"suite {suite.name}: {len(jobs)} searches x {args.workers} workers "
        f"(up to {len(jobs) * args.workers} concurrent workflow runs)",
        flush=True,
    )

    def job(name: str, seed: int) -> dict:
        r = run_search(
            make_optimizer(name, config),
            log.wrap(evaluate, name, seed),
            suite,
            config,
            seed,
            checker=runner.checker,
            workers=args.workers,
        )
        print(
            f"[{time.time() - t0:6.0f}s] {name} seed {seed}: val fitness "
            f"{r['validation_fitness']:.3f} val pass {r['pass_rate']:.2f}",
            flush=True,
        )
        return r

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        runs = list(pool.map(lambda j: job(*j), jobs))

    results = {
        "schema": RESULTS_SCHEMA,
        "synthetic": False,
        "evaluator_version": (
            f"{suite.policy.evaluation.evaluator_version}+{evaluator.fitness_fn.version}"
        ),
        "benchmark_hash": benchmark_hash(store=store),
        "model_hash": runner.versions().model_hash,
        "suite": suite.name,
        "suite_hash": suite.identity_hash,
        "train_tasks": [t.id for t in train],
        "validation_tasks": [t.id for t in val],
        "config": vars(config) if hasattr(config, "__dict__") else str(config),
        "runs": runs,
    }
    write_results(results, out / "results.json")
    plot_learning_curves(results, out / "learning_curve.png")

    # Select across seeds by the suite contract's ranking of each seed's validation champion.
    champions = [r["champion"] for r in runs if r["optimizer"] == "aco_mmas" and r["champion"]]
    champion = max(
        champions, key=lambda c: (c["rank_key"], c["validation_fitness"], c["genome_hash"])
    )
    if not champion["feasible"]:
        print(f"WARNING: no ACO workflow met the contract's limits: {champion['violations']}")
    best = log.genomes[champion["genome_hash"]]
    (out / "best_aco_genome.json").write_text(best.canonical_json() + "\n", encoding="utf-8")
    (out / "best_aco_workflow.txt").write_text(stage_path(best) + "\n", encoding="utf-8")
    print(f"\nwall time {time.time() - t0:.0f}s")
    summarize(results)
    print(f"best ACO workflow: {stage_path(best)}")
    print(
        f"wrote {out}/results.json, learning_curve.png, best_aco_genome.json, "
        f"best_aco_workflow.txt, evaluations.jsonl"
    )


def summarize(results: dict) -> None:
    by: dict[str, list[dict]] = {}
    for r in results["runs"]:
        by.setdefault(r["optimizer"], []).append(r)
    for name, rs in sorted(by.items()):
        vf = [r["validation_fitness"] for r in rs]
        vp = [r["pass_rate"] for r in rs]
        print(
            f"{name:14s} final val fitness mean {sum(vf) / len(vf):.3f} {[round(x, 3) for x in vf]}"
            f"  val pass rate mean {sum(vp) / len(vp):.2f}"
        )


def cmd_final(args: argparse.Namespace) -> None:
    saved = json.loads((args.out / "best_aco_genome.json").read_text(encoding="utf-8"))
    best = Genome.from_stages(saved["stages"])
    results = json.loads((args.out / "results.json").read_text(encoding="utf-8"))
    suite = legacy_suite(results["suite"])
    # The held-out split is for REPORTING only: these runs never reach an optimizer.
    tasks = list(suite.tasks_for(SplitRole.TEST, SplitUse.REPORTING))
    if args.task is not None:
        tasks = [task for task in tasks if task.id == args.task]
        if not tasks:
            raise ValueError(f"task {args.task} is not a held-out test task of suite {suite.name}")
    store = SnapshotStore(HELDOUT_DIR / "snapshots")
    runner = build_runner(client_from_env(), store, bench_dir=HELDOUT_DIR)
    evaluate = real_evaluate_fn(runner, evaluator_for(HELDOUT_DIR), tasks)
    summaries = []
    for task in tasks:
        run = evaluate_with_retries(evaluate, (best, task, 0, args.seed))
        s = print_run(task, best, run, store)
        summaries.append({"task": task.id, "question": task.example.values["question"], **s})
    (args.out / "final_test.json").write_text(
        dumps(
            {
                "split": "test",
                "suite": suite.name,
                "benchmark_hash": benchmark_hash(HELDOUT_DIR),
                "seed": args.seed,
                "workflow": stage_path(best),
                "runs": summaries,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--task", default="A-002")
    e = sub.add_parser("experiment")
    e.add_argument("--task-class", choices=["A", "B"], default="A")
    e.add_argument("--budget", type=int, default=60)
    e.add_argument("--batch", type=int, default=2)
    e.add_argument("--trials", type=int, default=2)
    e.add_argument("--seeds", type=int, default=1)
    e.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrent workflow runs inside each search; seeded submission order is preserved",
    )
    e.add_argument("--out", type=Path, default=OUT)
    f = sub.add_parser("final")
    f.add_argument(
        "--task", help="one held-out task; defaults to every test task in the saved class"
    )
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    {"smoke": cmd_smoke, "experiment": cmd_experiment, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
