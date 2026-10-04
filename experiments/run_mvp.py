"""Real MVP driver: Gemma (via ``GEMMA_*`` env) + MAF runtime + deterministic evaluator.

    python -m experiments.run_mvp smoke                       # one hand-built genome, one task
    python -m experiments.run_mvp experiment --budget 60 --seeds 1 --workers 4
    python -m experiments.run_mvp final                       # best ACO genome on a validation task

Every evaluation (train and validation) is appended to ``<out>/evaluations.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.loader import benchmark_hash, load_task_specs, runtime_tasks
from benchmarks.snapshot_store import SnapshotStore
from core.genome import Genome
from core.results import EvaluatedRun
from core.task_spec import RuntimeTask, TaskClass
from evaluation.gate import DeterministicEvaluator
from experiments.learning_curves import (
    OPTIMIZER_FACTORIES,
    RESULTS_SCHEMA,
    EvaluateFn,
    ExperimentConfig,
    run_search,
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


def print_run(task: RuntimeTask, genome: Genome, run: EvaluatedRun, store: SnapshotStore) -> dict:
    s = with_span_text(run_summary(run), task.snapshot_id, store)
    print(f"task      : {task.id}  {task.question}")
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
    evaluate = real_evaluate_fn(runner)
    task = next(t for t in runtime_tasks("train", TaskClass.A) if t.id == args.task)
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

    def wrap(self, evaluate: EvaluateFn, optimizer: str, seed: int) -> EvaluateFn:
        state = {"n": 0, "best": None}

        def logged(genome: Genome, task: RuntimeTask, trial: int, s: int) -> EvaluatedRun:
            run = evaluate(genome, task, trial, s)
            ev = run.evaluation
            validation = task.id in self.validation_ids
            # runs of one search complete concurrently: counter, best and the write share a lock
            with self._lock:
                self.genomes[genome.genome_hash] = genome
                if not validation:
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
                    "fitness": round(ev.fitness, 4),
                    "verdict": ev.verdict.value,
                    "best_so_far_fitness": state["best"],
                }
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            return run

        return logged


def cmd_experiment(args: argparse.Namespace) -> None:
    out: Path = args.out
    cls = TaskClass(args.task_class)
    train, val = runtime_tasks("train", cls), runtime_tasks("validation", cls)
    store = SnapshotStore()
    runner = build_runner(client_from_env(), store)
    evaluator = DeterministicEvaluator()
    evaluate = real_evaluate_fn(runner, evaluator, RunCache(out / "run_cache.jsonl"))
    config = ExperimentConfig(budget=args.budget, batch_size=args.batch, trials=args.trials)
    log = EvalLog(out / "evaluations.jsonl", {t.id for t in val})
    jobs = [(name, seed) for name in OPTIMIZER_FACTORIES for seed in range(args.seeds)]
    t0 = time.time()
    print(
        f"class {cls.value}: {len(jobs)} searches x {args.workers} workers "
        f"(up to {len(jobs) * args.workers} concurrent workflow runs)",
        flush=True,
    )

    def job(name: str, seed: int) -> dict:
        r = run_search(
            OPTIMIZER_FACTORIES[name](),
            log.wrap(evaluate, name, seed),
            train,
            val,
            config,
            seed,
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
        "evaluator_version": f"{evaluator.version}+{evaluator.fitness_fn.version}",
        "benchmark_hash": benchmark_hash(store=store),
        "model_hash": runner.versions().model_hash,
        "task_class": cls.value,
        "train_tasks": [t.id for t in train],
        "validation_tasks": [t.id for t in val],
        "config": vars(config) if hasattr(config, "__dict__") else str(config),
        "runs": runs,
    }
    write_results(results, out / "results.json")
    plot_learning_curves(results, out / "learning_curve.png")

    # best ACO genome = incumbent of the ACO seed with the best final validation fitness
    aco = max(
        (r for r in runs if r["optimizer"] == "aco_mmas"), key=lambda r: r["validation_fitness"]
    )
    best = log.genomes[aco["best_genome_hash"]]
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
    store = SnapshotStore()
    saved = json.loads((args.out / "best_aco_genome.json").read_text(encoding="utf-8"))
    best = Genome.from_stages(saved["stages"])
    runner = build_runner(client_from_env(), store)
    evaluate = real_evaluate_fn(runner)
    specs = load_task_specs()
    task = specs[args.task].runtime_view()
    run = evaluate(best, task, 0, args.seed)
    s = print_run(task, best, run, store)
    (args.out / "final_test.json").write_text(
        dumps({"task": task.id, "question": task.question, "workflow": stage_path(best), **s})
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
        help="concurrent workflow runs inside each search (results are identical to 1)",
    )
    e.add_argument("--out", type=Path, default=OUT)
    f = sub.add_parser("final")
    f.add_argument("--task", default="A-001")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    {"smoke": cmd_smoke, "experiment": cmd_experiment, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
