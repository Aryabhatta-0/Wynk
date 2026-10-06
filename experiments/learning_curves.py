"""Optimizer comparison harness: ACO vs random search learning curves (MVP).

The single entry point is ``run_experiment``. It takes an ``EvaluateFn`` -
``(genome, task, trial, seed) -> EvaluatedRun`` - so it is agnostic to where results come from.
For the real thing, build one with ``make_evaluate_fn(run_workflow, evaluator, tasks)`` where
``run_workflow(genome, execution_task, trial, seed) -> ExecutionResult`` is Track A's runtime and
``evaluator`` is a ``ContractEvaluator``; for tests/demos use
``experiments.synthetic.synthetic_evaluate`` (labelled synthetic).

A search runs on one ``ContractSuite``; its ``DatasetSplits`` decide what each row may be used
for, and the harness enforces it:
  * optimization rows -> optimizer feedback: each proposed genome is run on every one of them x
    ``trials`` trials; each run is one "workflow evaluation" charged against ``budget``; every
    batch passes ``suite.check_feedback`` before ``optimizer.observe`` (fail closed);
  * validation rows -> selection only: after every genome the incumbent's validation fitness is
    recorded (never shown to the optimizer, not charged to the budget), and the CHAMPION is the
    validated genome ranked best by the suite's contract (``TaskContract.rank``: hard limits
    first, objective second);
  * test rows -> reporting only: never run here.
The incumbent is the genome with the best optimization-row lower-confidence-bound score so far,
so the curve's y is the validation fitness of the best-so-far workflow *as selected on train*.
Admission comes from the suite's contract (``SearchContext.contract``), never from a task class.

``workers > 1`` runs the workflow evaluations of a round (and of a validation pass) concurrently.
Every run has its own deterministic seed and results are consumed in submission order, so the
search order is reproducible for identical evaluations. Backend responses and hard wall-time
breaches can still vary with concurrency.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.canonical import canonical_hash, canonical_json
from core.constraints import ConstraintChecker
from core.dataset import SplitRole, SplitUse
from core.genome import Genome
from core.results import EvaluatedRun, ExecutionResult, FailureKind, Verdict
from core.run_contract import ContractSuite, ExecutionTask
from core.task_contract import ContractError
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import Optimizer, SearchContext
from optimizers.random_search import RandomSearch
from optimizers.scoring import DEFAULT_Z, ScoreBoard

RESULTS_SCHEMA = "wynk-experiment/1"

EvaluateFn = Callable[[Genome, ExecutionTask, int, int], EvaluatedRun]
RunWorkflowFn = Callable[[Genome, ExecutionTask, int, int], ExecutionResult]


def make_evaluate_fn(
    run_workflow: RunWorkflowFn, evaluator, tasks: Iterable[ExecutionTask]
) -> EvaluateFn:
    """Compose the real runtime (Track A) with the contract evaluator.

    Every task is checked up front (``evaluator.check_task``: expected values present, evaluator
    implemented at the pinned version), so a task that cannot be judged fails before ANY model
    call. Expected values stay inside ``evaluator``; the harness and optimizers only ever see the
    resulting ``EvaluatedRun``.
    """
    bound = {t.id: t for t in tasks}
    evaluator.check_tasks(bound.values())

    def evaluate(genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
        if bound.get(task.id) != task:
            raise ContractError(f"task {task.id} is not one this evaluator was bound to")
        result = run_workflow(genome, task, trial, seed)
        return evaluator.evaluate_run(task, result)

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


def make_optimizer(name: str, config: ExperimentConfig) -> Optimizer:
    if name == MMASACO.name:
        return MMASACO(ACOConfig(lcb_z=config.lcb_z))
    return OPTIMIZER_FACTORIES[name]()


def _exec_seed(base_seed: int, genome_hash: str, task_id: str, trial: int) -> int:
    return int(canonical_hash([base_seed, genome_hash, task_id, trial])[:8], 16)


Job = tuple[Genome, ExecutionTask, int, int]


def _evaluate_all(
    evaluate: EvaluateFn, jobs: Iterable[Job], pool: ThreadPoolExecutor | None
) -> list[EvaluatedRun]:
    """Order-preserving map of ``evaluate`` over ``jobs``, concurrent when ``pool`` is given."""

    if pool is None:
        return [evaluate_with_retries(evaluate, j) for j in jobs]
    return list(pool.map(lambda j: evaluate_with_retries(evaluate, j), jobs))


def evaluate_with_retries(evaluate: EvaluateFn, job: Job) -> EvaluatedRun:
    """Backend outages are retried, then surfaced instead of becoming genome observations."""
    for _ in range(3):
        run = evaluate(*job)
        failure = run.execution.failure
        if failure is None or failure.kind is not FailureKind.MODEL_ERROR:
            return run
    raise RuntimeError("model unavailable after 3 workflow attempts; stopped without scoring")


def search_tasks(suite: ContractSuite) -> tuple[tuple[ExecutionTask, ...], ...]:
    """(optimization tasks, validation tasks) of ``suite``, each fetched for its permitted use."""
    train = suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK)
    val = suite.tasks_for(SplitRole.VALIDATION, SplitUse.SELECTION)
    if not train or not val:
        raise ContractError("a search needs optimization rows (feedback) and validation rows")
    return train, val


def run_search(
    optimizer: Optimizer,
    evaluate: EvaluateFn,
    suite: ContractSuite,
    config: ExperimentConfig,
    seed: int,
    checker: ConstraintChecker | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return _run_search(optimizer, evaluate, suite, config, seed, checker, pool)
    return _run_search(optimizer, evaluate, suite, config, seed, checker, None)


def _run_search(
    optimizer: Optimizer,
    evaluate: EvaluateFn,
    suite: ContractSuite,
    config: ExperimentConfig,
    seed: int,
    checker: ConstraintChecker | None,
    pool: ThreadPoolExecutor | None,
) -> dict[str, Any]:
    checker = checker or ConstraintChecker()
    train_tasks, val_tasks = search_tasks(suite)
    board = ScoreBoard(config.lcb_z)
    genomes: dict[str, Genome] = {}
    repeats: dict[str, int] = {}
    val_cache: dict[str, tuple[float, float]] = {}  # hash -> (validation fitness, pass rate)
    val_runs: dict[str, list[EvaluatedRun]] = {}  # hash -> validation runs (selection only)
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
            runs = _evaluate_all(
                evaluate,
                (
                    (g, t, trial, _exec_seed(seed, h, t.id, 10_000 + trial))
                    for t in val_tasks
                    for trial in range(config.trials)
                ),
                pool,
            )
            val_runs[h] = runs
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
        context = SearchContext(contract=suite.policy, checker=checker, seed=seed, round=rnd)
        proposals = optimizer.propose(config.batch_size, context)
        if not proposals:
            break
        # decide which proposals fit the budget, then run all their train runs in one go
        accepted: list[tuple[Genome, int]] = []
        for g in proposals:
            if evaluations + (len(accepted) + 1) * per_genome > config.budget:
                exhausted = True
                break
            h = g.genome_hash
            genomes[h] = g
            accepted.append((g, repeats.get(h, 0) * config.trials))  # fresh trials on repeats
            repeats[h] = repeats.get(h, 0) + 1
        all_runs = _evaluate_all(
            evaluate,
            (
                (g, t, first + i, _exec_seed(seed, g.genome_hash, t.id, first + i))
                for g, first in accepted
                for t in train_tasks
                for i in range(config.trials)
            ),
            pool,
        )
        batch: list[EvaluatedRun] = []
        for k in range(len(accepted)):
            runs = all_runs[k * per_genome : (k + 1) * per_genome]
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
            suite.check_feedback(batch)  # only optimization rows may reach optimizer state
            optimizer.observe(batch)
        rnd += 1

    last = curve[-1] if curve else None
    champion = _champion(suite, genomes, val_runs, val_cache)
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
        # Selection under the suite's contract (validation rows only): see ``_champion``.
        "champion": champion,
        "suite": suite.name,
        "suite_hash": suite.identity_hash,
        # Additive, for persistent workflow memory: every workflow that was ever the incumbent,
        # with its measured validation stats, and the run/evaluator versions it was measured on.
        "workflows": [val_stats[h] for h in sorted(val_stats)],
        "versions": [json.loads(v) for v in sorted(versions)],
    }


def _champion(
    suite: ContractSuite,
    genomes: Mapping[str, Genome],
    val_runs: Mapping[str, Sequence[EvaluatedRun]],
    val_cache: Mapping[str, tuple[float, float]],
) -> dict[str, Any] | None:
    """The validated genome the suite's contract ranks best (feasible first, then objective;
    ties by validation fitness, then hash). ``feasible`` is False when no validated genome meets
    the contract's hard limits - such a genome is reported, never silently promoted."""
    if not val_runs:
        return None
    ranks = {h: suite.rank(genomes[h], runs) for h, runs in val_runs.items()}
    h = max(ranks, key=lambda k: (ranks[k].sort_key, val_cache[k][0], k))
    rank = ranks[h]
    return {
        "genome_hash": h,
        "feasible": rank.feasible,
        "rank_key": list(rank.sort_key),
        "violations": [v.message for v in rank.violations],
        "validation_fitness": val_cache[h][0],
        "validation_pass_rate": val_cache[h][1],
    }


def run_experiment(
    evaluate: EvaluateFn,
    suite: ContractSuite,
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
    train_tasks, val_tasks = search_tasks(suite)
    runs = [
        run_search(make_optimizer(name, config), evaluate, suite, config, seed, checker)
        for name in optimizers
        for seed in seeds
    ]
    return {
        "schema": RESULTS_SCHEMA,
        "synthetic": synthetic,
        "evaluator_version": evaluator_version,
        "benchmark_hash": benchmark_hash,
        "suite": suite.name,
        "suite_hash": suite.identity_hash,
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

    from benchmarks.legacy_adapter import legacy_suite
    from experiments.report import plot_learning_curves
    from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--synthetic",
        action="store_true",
        required=True,
        help="required: only the labelled fake objective is available before merge",
    )
    # the demo runs on a frozen benchmark suite, adapted at benchmarks.legacy_adapter
    p.add_argument("--task-class", dest="suite", choices=["A", "B"], default="A")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("experiments/results/synthetic"))
    args = p.parse_args(argv)

    results = run_experiment(
        synthetic_evaluate,
        legacy_suite(args.suite),
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
