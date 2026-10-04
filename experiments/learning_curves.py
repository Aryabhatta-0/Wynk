"""Optimizer comparison harness: ACO vs random search learning curves (MVP).

The single entry point is ``run_experiment``. It takes an ``EvaluateFn`` -
``(genome, task, trial, seed) -> EvaluatedRun`` - so it is agnostic to where results come from.
For the real thing, build one with ``make_evaluate_fn(run_workflow, evaluator, specs)`` where
``run_workflow(genome, runtime_task, trial, seed) -> ExecutionResult`` is Track A's runtime;
for tests/demos use ``experiments.synthetic.synthetic_evaluate`` (labelled synthetic).

Protocol (per optimizer, per seed, one task class at a time):
  * each proposed genome is run on every TRAIN task x ``trials`` trials; each run is one
    "workflow evaluation" charged against ``budget``;
  * the incumbent is the genome with the best train lower-confidence-bound score so far;
  * after every genome the incumbent's VALIDATION fitness is recorded (validation runs are
    measurement only - never shown to the optimizer, not charged to the budget).
So the curve's y is the validation fitness of the best-so-far workflow *as selected on train*.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.canonical import canonical_hash, canonical_json
from core.constraints import ConstraintChecker
from core.genome import Genome
from core.results import EvaluatedRun, ExecutionResult, Verdict
from core.task_spec import RuntimeTask, TaskSpec
from optimizers.aco_mmas import MMASACO
from optimizers.base import Optimizer, SearchContext
from optimizers.random_search import RandomSearch
from optimizers.scoring import DEFAULT_Z, ScoreBoard

RESULTS_SCHEMA = "wynk-experiment/1"

EvaluateFn = Callable[[Genome, RuntimeTask, int, int], EvaluatedRun]
RunWorkflowFn = Callable[[Genome, RuntimeTask, int, int], ExecutionResult]


def make_evaluate_fn(
    run_workflow: RunWorkflowFn, evaluator, specs: Mapping[str, TaskSpec]
) -> EvaluateFn:
    """Compose the real runtime (Track A) with the deterministic evaluator.

    Ground truth stays inside ``evaluator``/``specs``; the harness and optimizers only ever
    see the resulting ``EvaluatedRun``.
    """

    def evaluate(genome: Genome, task: RuntimeTask, trial: int, seed: int) -> EvaluatedRun:
        result = run_workflow(genome, task, trial, seed)
        return EvaluatedRun(execution=result, evaluation=evaluator.evaluate(specs[task.id], result))

    return evaluate


@dataclass(frozen=True)
class ExperimentConfig:
    budget: int = 200  # train workflow evaluations per run
    batch_size: int = 2  # genomes proposed per round (= ants per pheromone update)
    trials: int = 2  # trials per (genome, task)
    lcb_z: float = DEFAULT_Z


OPTIMIZER_FACTORIES: dict[str, Callable[[], Optimizer]] = {
    RandomSearch.name: RandomSearch,
    MMASACO.name: MMASACO,
}


def _exec_seed(base_seed: int, genome_hash: str, task_id: str, trial: int) -> int:
    return int(canonical_hash([base_seed, genome_hash, task_id, trial])[:8], 16)


def _check_homogeneous(tasks: Sequence[RuntimeTask]) -> None:
    """One genome is evaluated across many tasks, so they must share the hard-constraint view."""
    ref = tasks[0]
    for t in tasks[1:]:
        same = (t.caps, t.allowed_sources, t.interaction_required, t.task_class) == (
            ref.caps,
            ref.allowed_sources,
            ref.interaction_required,
            ref.task_class,
        )
        if not same:
            raise ValueError(f"task {t.id} differs from {ref.id} in caps/sources/class")


def run_search(
    optimizer: Optimizer,
    evaluate: EvaluateFn,
    train_tasks: Sequence[RuntimeTask],
    val_tasks: Sequence[RuntimeTask],
    config: ExperimentConfig,
    seed: int,
    checker: ConstraintChecker | None = None,
) -> dict[str, Any]:
    checker = checker or ConstraintChecker()
    _check_homogeneous([*train_tasks, *val_tasks])
    board = ScoreBoard(config.lcb_z)
    genomes: dict[str, Genome] = {}
    repeats: dict[str, int] = {}
    val_cache: dict[str, tuple[float, float]] = {}  # hash -> (validation fitness, pass rate)
    val_stats: dict[str, dict[str, Any]] = {}  # hash -> validated workflow record (for memory)
    versions: set[str] = set()  # canonical RunVersions + evaluator version seen in validation
    curve: list[dict[str, Any]] = []
    evaluations = 0
    train_passes = 0
    per_genome = len(train_tasks) * config.trials
    best_hash: str | None = None

    def validate(g: Genome) -> tuple[float, float]:
        h = g.genome_hash
        if h not in val_cache:
            runs = [
                evaluate(g, t, trial, _exec_seed(seed, h, t.id, 10_000 + trial))
                for t in val_tasks
                for trial in range(config.trials)
            ]
            val_cache[h] = (
                statistics.fmean(r.evaluation.fitness for r in runs),
                sum(r.evaluation.verdict is Verdict.PASS for r in runs) / len(runs),
            )
            val_stats[h] = {
                "genome_hash": h,
                "genome": g.canonical(),
                "validation_fitness": val_cache[h][0],
                "validation_pass_rate": val_cache[h][1],
                "mean_tokens": statistics.fmean(r.execution.budget_usage.tokens for r in runs),
                "mean_wall_time_s": statistics.fmean(
                    r.execution.budget_usage.wall_time_s for r in runs
                ),
                "validation_runs": len(runs),
            }
            versions.update(
                canonical_json(
                    {
                        "run_versions": r.execution.key.versions.model_dump(mode="json"),
                        "evaluator_version": r.evaluation.evaluator_version,
                    }
                )
                for r in runs
            )
        return val_cache[h]

    rnd = 0
    exhausted = False
    while not exhausted:
        context = SearchContext(task=train_tasks[0], checker=checker, seed=seed, round=rnd)
        proposals = optimizer.propose(config.batch_size, context)
        if not proposals:
            break
        batch: list[EvaluatedRun] = []
        for g in proposals:
            if evaluations + per_genome > config.budget:
                exhausted = True
                break
            h = g.genome_hash
            genomes[h] = g
            first_trial = repeats.get(h, 0) * config.trials  # fresh trials on re-evaluation
            repeats[h] = repeats.get(h, 0) + 1
            runs = [
                evaluate(g, t, first_trial + i, _exec_seed(seed, h, t.id, first_trial + i))
                for t in train_tasks
                for i in range(config.trials)
            ]
            batch += runs
            board.add(runs)
            evaluations += len(runs)
            train_passes += sum(r.evaluation.verdict is Verdict.PASS for r in runs)
            incumbent = board.best()
            best_hash = incumbent.genome_hash
            val_fit, val_pass = validate(genomes[best_hash])
            curve.append(
                {
                    "evaluations": evaluations,
                    "best_so_far_train_fitness": incumbent.lcb,
                    "validation_fitness": val_fit,
                    "validation_pass_rate": val_pass,
                    "best_genome_hash": best_hash,
                }
            )
        if batch:
            optimizer.observe(batch)
        rnd += 1

    last = curve[-1] if curve else None
    return {
        "optimizer": optimizer.name,
        "seed": seed,
        "workflow_evaluations": evaluations,
        "best_so_far_fitness": last["best_so_far_train_fitness"] if last else None,
        "validation_fitness": last["validation_fitness"] if last else None,
        "pass_rate": last["validation_pass_rate"] if last else None,
        "train_pass_rate": train_passes / evaluations if evaluations else None,
        "best_genome_hash": best_hash,
        "curve": curve,
        # Additive, for persistent workflow memory: every workflow that was ever the incumbent,
        # with its measured validation stats, and the run/evaluator versions it was measured on.
        "task_class": train_tasks[0].task_class.value,
        "workflows": [val_stats[h] for h in sorted(val_stats)],
        "versions": [json.loads(v) for v in sorted(versions)],
    }


def run_experiment(
    evaluate: EvaluateFn,
    train_tasks: Sequence[RuntimeTask],
    val_tasks: Sequence[RuntimeTask],
    *,
    optimizers: Sequence[str] = (RandomSearch.name, MMASACO.name),
    seeds: Sequence[int] = (0, 1, 2),
    config: ExperimentConfig | None = None,
    synthetic: bool,
    evaluator_version: str,
    benchmark_hash: str | None = None,
    checker: ConstraintChecker | None = None,
) -> dict[str, Any]:
    """Run every optimizer for every seed and return one machine-readable results document.

    ``synthetic`` is mandatory so a fake objective can never be mistaken for a real result.
    """
    config = config or ExperimentConfig()
    runs = [
        run_search(
            OPTIMIZER_FACTORIES[name](), evaluate, train_tasks, val_tasks, config, seed, checker
        )
        for name in optimizers
        for seed in seeds
    ]
    return {
        "schema": RESULTS_SCHEMA,
        "synthetic": synthetic,
        "evaluator_version": evaluator_version,
        "benchmark_hash": benchmark_hash,
        "task_class": train_tasks[0].task_class.value,
        "train_tasks": [t.id for t in train_tasks],
        "validation_tasks": [t.id for t in val_tasks],
        "config": asdict(config),
        "runs": runs,
    }


def write_results(results: Mapping[str, Any], path: Path) -> None:
    if results["synthetic"] and "synthetic" not in results["evaluator_version"]:
        raise ValueError("synthetic results must carry a synthetic evaluator_version")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fake-objective demo: ``python -m experiments.learning_curves --synthetic``.

    The real run needs Track A's runtime: call ``run_experiment`` with ``make_evaluate_fn``.
    """
    import argparse

    from benchmarks.loader import runtime_tasks
    from core.task_spec import TaskClass
    from experiments.report import plot_learning_curves
    from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--synthetic",
        action="store_true",
        required=True,
        help="required: only the labelled fake objective is available before merge",
    )
    p.add_argument("--task-class", choices=["A", "B"], default="A")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("experiments/results/synthetic"))
    args = p.parse_args(argv)

    cls = TaskClass(args.task_class)
    results = run_experiment(
        synthetic_evaluate,
        runtime_tasks("train", cls),
        runtime_tasks("validation", cls),
        seeds=range(args.seeds),
        config=ExperimentConfig(budget=args.budget),
        synthetic=True,
        evaluator_version=SYNTHETIC_VERSION,
    )
    write_results(results, args.out / "results.json")
    plot_learning_curves(results, args.out / "learning_curve.png")
    print(f"wrote {args.out / 'results.json'} and {args.out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
