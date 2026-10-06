"""Optimize an uploaded dataset: fixed baseline vs random search vs MMAS ACO under equal budgets.

    optimize_uploaded_dataset(contract, splits, data, run_workflow, plan, ...) -> artifact
      = contract_suite -> ContractEvaluator (EvaluationSpec dispatcher)
        -> run_optimization_experiment

One runner, ``run_strategy``, drives every strategy through the same loop. For one experiment all
strategies share, by construction:

    dataset bytes + DatasetSpec    the ``ContractSuite`` (rows + ``DatasetSplits``)
    TaskContract                   ``suite.policy``: admission, per-run caps, ranking
    workflow grammar / checker     one ``ConstraintChecker``, the contract's vocabulary
    model                          ``plan.model`` (declared) - every run must report one model
    evaluator                      one ``EvaluateFn`` whose evaluator_version must match
    seed policy                    ``plan.seeds``; run seeds = f(seed, genome, row, trial)
    experiment budget              ``plan.budget``, one fresh ``BudgetLedger`` per strategy run

Fairness invariant: the same candidate-evaluation budget gives every strategy the same
opportunity to spend it. A *candidate evaluation* is one proposed workflow run on every
optimization row x ``trials`` and every validation row x ``trials``; the ledger
(``experiments.budget_ledger``) counts candidates and charges MEASURED usage under the same rules
for every strategy. The same genome under the same experiment seed gets the same run seeds in
every strategy (common random numbers), so strategies differ only in WHICH workflows they try.

Strategies (``Strategy``):
    fixed   ``optimizers.fixed_baseline``: the shortest admissible workflow in canonical grammar
            order. Chosen without data; it uses one candidate evaluation and then stops.
    random  ``optimizers.random_search.DistinctRandomSearch``: the uniform random walk, without
            replacement until the admissible space is exhausted.
    aco     ``optimizers.aco_mmas.MMASACO``, unchanged. A re-proposed workflow is evaluated again
            with fresh trials and costs one candidate evaluation, exactly like any other.

Data isolation:
    optimization rows   the only runs an optimizer ever observes (``suite.check_feedback`` gate)
    validation rows     candidate scores and champion selection (``TaskContract.rank``); never
                        shown to an optimizer
    test rows           never executed here. The runner never asks for them; the uploaded-dataset
                        entry point does not even bind their expected values to the evaluator.
                        The final test result belongs to later promotion logic.

Reproducibility: every strategy run carries an identity (dataset content + split + contract
hashes, grammar version, evaluator identity/version, model, pricing, strategy + version, budget,
seed) and its ``run_id``. With a deterministic backend, the same identity gives the same candidate
order and learning curve. The artifact holds no timestamps, paths or uuids.

Durable execution (#24) can call ``run_strategy`` per (strategy, seed) and ``assemble`` the
records afterwards; ``run_optimization_experiment`` is just that loop.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from core.canonical import canonical_hash
from core.constraints import ConstraintChecker
from core.dataset import DatasetSplits, SplitRole
from core.experiment import ModelConfiguration, experiment_identity
from core.genome import Genome
from core.results import EvaluatedRun, FailureKind, Verdict
from core.run_contract import ContractSuite, ExecutionTask
from core.task_contract import TaskContract, workflow_grammar
from evaluation.contract_eval import ContractEvaluator, References
from evaluation.fitness import FitnessFunction
from experiments.budget_ledger import (
    LEDGER_VERSION,
    BudgetLedger,
    ExperimentBudget,
    PricingPolicy,
    Resource,
    StopReason,
    check_budget,
)
from experiments.contract_run import contract_suite
from experiments.learning_curves import EvaluateFn, RunWorkflowFn, make_evaluate_fn, search_tasks
from ingestion.parse import IngestLimits
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import Optimizer, SearchContext
from optimizers.fixed_baseline import FixedBaseline
from optimizers.random_search import DistinctRandomSearch
from optimizers.scoring import DEFAULT_Z, ScoreBoard

ARTIFACT_SCHEMA = "wynk-optimization-experiment/1"
RUNNER_VERSION = "optimization-runner/1"
MODEL_ATTEMPTS = 3  # a MODEL_ERROR run is retried (each attempt charged), then surfaced
VALIDATION_TRIAL_OFFSET = 10_000  # validation runs never share a trial (hence seed) with feedback


class Strategy(StrEnum):
    FIXED = "fixed"
    RANDOM = "random"
    ACO = "aco"


class ExperimentError(ValueError):
    """The experiment broke one of its own invariants (fail closed, no artifact)."""


class ModelUnavailable(RuntimeError):
    """The backend kept failing; stopped without turning outages into scores."""


class ExperimentPlan(BaseModel):
    """Everything about HOW strategies are compared. Serializable, hashed into run identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelConfiguration
    budget: ExperimentBudget
    seeds: tuple[int, ...] = Field(min_length=1)
    strategies: tuple[Strategy, ...] = (Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
    trials: PositiveInt = 1  # trials per (candidate, row)
    batch_size: PositiveInt = 2  # proposals per round (= ants per pheromone update)
    lcb_z: float = Field(default=DEFAULT_Z, ge=0.0, allow_inf_nan=False)

    @field_validator("seeds")
    @classmethod
    def _distinct_seeds(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(v)) != len(v):
            raise ValueError("seeds must be distinct")
        return v

    @field_validator("strategies")
    @classmethod
    def _distinct_strategies(cls, v: tuple[Strategy, ...]) -> tuple[Strategy, ...]:
        if not v or len(set(v)) != len(v):
            raise ValueError("strategies must be non-empty and distinct")
        return v

    def protocol(self) -> dict[str, Any]:
        """What every strategy run of this plan is held to (seed and strategy excluded)."""
        return {
            "budget": self.budget.model_dump(mode="json"),
            "trials": self.trials,
            "batch_size": self.batch_size,
            "lcb_z": self.lcb_z,
        }


OptimizerFactory = Callable[[Strategy, ExperimentPlan], Optimizer]


def make_strategy(strategy: Strategy, plan: ExperimentPlan) -> Optimizer:
    if strategy is Strategy.FIXED:
        return FixedBaseline()
    if strategy is Strategy.RANDOM:
        return DistinctRandomSearch()
    return MMASACO(ACOConfig(lcb_z=plan.lcb_z))


def optimizer_version(optimizer: Optimizer) -> str:
    return str(getattr(optimizer, "version", optimizer.name))


def run_seed(seed: int, genome_hash: str, task_id: str, trial: int) -> int:
    """Execution seed of one run (same derivation as ``experiments.learning_curves``)."""
    return int(canonical_hash([seed, genome_hash, task_id, trial])[:8], 16)


# -- identity -----------------------------------------------------------------------------------
def problem_identity(
    suite: ContractSuite,
    model: ModelConfiguration,
    *,
    evaluator_version: str,
    pricing: PricingPolicy | None,
) -> dict[str, Any]:
    """WHAT is being optimized and how it is measured - shared by every strategy and seed."""
    contract = suite.policy
    core = experiment_identity(
        contract, suite.splits, grammar_version=workflow_grammar(contract).version, model=model
    )
    out = {
        **core.model_dump(mode="json"),
        "experiment_identity_id": core.experiment_id,
        "dataset_id": contract.dataset.dataset_id,
        "dataset_version": contract.dataset.dataset_version,
        "dataset_content_hash": contract.dataset.content_hash,
        "suite_hash": suite.identity_hash,
        "evaluator_kind": contract.evaluation.evaluator.value,
        "evaluator_run_version": evaluator_version,  # what every EvaluatedRun must report
        "model": model.model_dump(mode="json"),
        "pricing": pricing.identity if pricing is not None else None,
        "runner_version": RUNNER_VERSION,
        "ledger_version": LEDGER_VERSION,
    }
    out["problem_id"] = canonical_hash(out)
    return out


def run_identity(
    problem: Mapping[str, Any],
    plan: ExperimentPlan,
    strategy: Strategy,
    optimizer: Optimizer,
    seed: int,
) -> dict[str, Any]:
    out = {
        "problem_id": problem["problem_id"],
        "protocol": plan.protocol(),
        "budget_hash": plan.budget.identity_hash,
        "strategy": strategy.value,
        "optimizer": optimizer.name,
        "optimizer_version": optimizer_version(optimizer),
        "seed": seed,
    }
    out["run_id"] = canonical_hash(out)
    return out


# -- one strategy x one seed --------------------------------------------------------------------
def _stats_of(runs: Sequence[EvaluatedRun]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "fitness_mean": statistics.fmean(r.evaluation.fitness for r in runs),
        "pass_rate": sum(r.evaluation.verdict is Verdict.PASS for r in runs) / len(runs),
    }


_RESOURCE_STOP = {
    Resource.MODEL_CALLS: StopReason.MODEL_CALLS,
    Resource.TOKENS: StopReason.TOKENS,
    Resource.WALL_TIME: StopReason.WALL_TIME,
    Resource.COST: StopReason.COST,
}


def run_strategy(
    plan: ExperimentPlan,
    strategy: Strategy,
    seed: int,
    suite: ContractSuite,
    evaluate: EvaluateFn,
    *,
    checker: ConstraintChecker,
    evaluator_version: str,
    pricing: PricingPolicy | None = None,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Run ONE strategy for ONE seed under ``plan.budget``; returns its self-contained record."""
    check_budget(plan.budget, pricing)
    if seed not in plan.seeds or strategy not in plan.strategies:
        raise ExperimentError(f"{strategy.value}/seed {seed} is not part of this plan")
    train, val = search_tasks(suite)
    problem = problem_identity(
        suite, plan.model, evaluator_version=evaluator_version, pricing=pricing
    )
    optimizer = optimizer or make_strategy(strategy, plan)
    identity = run_identity(problem, plan, strategy, optimizer, seed)

    ledger = BudgetLedger(plan.budget, pricing)
    board = ScoreBoard(plan.lcb_z)  # optimization-row score per genome (what the optimizer sees)
    repeats: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    genomes: dict[str, Genome] = {}
    val_runs: dict[str, list[EvaluatedRun]] = {}
    ranks: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    executed = {role.value: 0 for role in SplitRole}
    model_hashes: set[str] = set()
    champion: str | None = None

    def execute(
        genome: Genome, task: ExecutionTask, trial: int, entries: list[dict[str, Any]]
    ) -> tuple[EvaluatedRun | None, StopReason | None]:
        role = suite.splits.role_of(task.id)
        if role not in (SplitRole.OPTIMIZATION, SplitRole.VALIDATION):
            raise ExperimentError(f"row {task.id} is not an optimization or validation row")
        rs = run_seed(seed, genome.genome_hash, task.id, trial)
        for attempt in range(MODEL_ATTEMPTS):
            stop = ledger.run_stop()
            if stop is not None:
                return None, stop
            run = evaluate(genome, task, trial, rs)
            key = run.execution.key
            if (key.genome_hash, key.task_id, key.trial, key.seed) != (
                genome.genome_hash,
                task.id,
                trial,
                rs,
            ):
                raise ExperimentError("the evaluator returned a run for a different job")
            if run.evaluation.evaluator_version != evaluator_version:
                raise ExperimentError(
                    f"run judged by {run.evaluation.evaluator_version!r}, experiment declares "
                    f"{evaluator_version!r}"
                )
            model_hashes.add(key.versions.model_hash)
            if len(model_hashes) > 1:
                raise ExperimentError(f"runs used more than one model: {sorted(model_hashes)}")
            executed[role.value] += 1
            charged = ledger.charge(run)
            failure = run.execution.failure
            model_error = failure is not None and failure.kind is FailureKind.MODEL_ERROR
            entries.append(
                {
                    "row_id": task.id,
                    "split": role.value,
                    "trial": trial,
                    "attempt": attempt,
                    "verdict": None if model_error else run.evaluation.verdict.value,
                    "fitness": None if model_error else run.evaluation.fitness,
                    "failure": failure.kind.value if failure else None,
                    **charged,
                }
            )
            over = ledger.exceeded()
            if over:
                return None, _RESOURCE_STOP[over[0]]
            if not model_error:
                return run, None
        raise ModelUnavailable(f"model unavailable after {MODEL_ATTEMPTS} attempts on {task.id}")

    def evaluate_candidate(
        genome: Genome, round_: int
    ) -> tuple[list[EvaluatedRun], StopReason | None]:
        nonlocal champion
        h = genome.genome_hash
        rep = repeats.get(h, 0)
        first = rep * plan.trials
        entries: list[dict[str, Any]] = []
        record: dict[str, Any] = {
            "evaluation": None,
            "round": round_,
            "genome_hash": h,
            "genome": genome.canonical(),
            "repeat": rep > 0,
            "runs": entries,
        }
        opt: list[EvaluatedRun] = []
        sel: list[EvaluatedRun] = []
        plan_runs = [(t, first + i, opt) for t in train for i in range(plan.trials)] + [
            (t, VALIDATION_TRIAL_OFFSET + first + i, sel) for t in val for i in range(plan.trials)
        ]
        for task, trial, sink in plan_runs:
            run, stop = execute(genome, task, trial, entries)
            if stop is not None:  # aborted: real spend, but no feedback, no selection, no curve
                ledger.aborted_candidates += 1
                record.update(status="over_budget", stop_reason=stop.value)
                candidates.append(record)
                return [], stop
            sink.append(run)

        ledger.candidate_evaluations += 1
        n = ledger.candidate_evaluations
        repeats[h] = rep + 1
        first_seen.setdefault(h, n)
        genomes[h] = genome
        board.add(opt)
        val_runs.setdefault(h, []).extend(sel)
        ranks[h] = suite.rank(genome, val_runs[h])

        def selection_key(k: str) -> tuple:
            fit = statistics.fmean(r.evaluation.fitness for r in val_runs[k])
            return (ranks[k].sort_key, fit, -first_seen[k])

        champion = max(ranks, key=selection_key)
        cand_val = _stats_of(sel)
        champ = _stats_of(val_runs[champion])
        totals = ledger.totals()
        record.update(
            evaluation=n,
            status="admitted",
            optimization={**_stats_of(opt), "genome_lcb": board.score(h).lcb},
            validation=cand_val,
            selection={
                "feasible": ranks[h].feasible,
                "rank_key": list(ranks[h].sort_key),
                "violations": [v.message for v in ranks[h].violations],
            },
        )
        candidates.append(record)
        curve.append(
            {
                "evaluation": n,
                "genome_hash": h,
                "score": cand_val["fitness_mean"],
                "validation_pass_rate": cand_val["pass_rate"],
                "optimization_fitness": record["optimization"]["fitness_mean"],
                "best_so_far_score": champ["fitness_mean"],
                "best_so_far_pass_rate": champ["pass_rate"],
                "best_so_far_genome_hash": champion,
                "cumulative_model_calls": totals["model_calls"],
                "cumulative_tokens": totals["tokens"],
                "cumulative_wall_time_s": totals["wall_time_s"],
                "cumulative_cost": totals["cost"],
            }
        )
        return opt, None

    stop: StopReason | None = None
    rnd = 0
    while stop is None:
        stop = ledger.candidate_stop()
        if stop is not None:
            break
        context = SearchContext(contract=suite.policy, checker=checker, seed=seed, round=rnd)
        proposals = optimizer.propose(plan.batch_size, context)
        if not proposals:
            stop = StopReason.STRATEGY_EXHAUSTED
            break
        feedback: list[EvaluatedRun] = []
        for genome in proposals:
            stop = ledger.candidate_stop()
            if stop is not None:
                break
            opt_runs, stop = evaluate_candidate(genome, rnd)
            if stop is not None:
                break
            feedback += opt_runs
        if feedback:
            suite.check_feedback(feedback)  # only optimization rows may reach optimizer state
            optimizer.observe(feedback)
        rnd += 1

    champion_record = None
    if champion is not None:
        rank = ranks[champion]
        champion_record = {
            "genome_hash": champion,
            "genome": genomes[champion].canonical(),
            "feasible": rank.feasible,
            "rank_key": list(rank.sort_key),
            "violations": [v.message for v in rank.violations],
            "first_evaluation": first_seen[champion],
            "validation": _stats_of(val_runs[champion]),
        }
    admitted = [c for c in candidates if c["status"] == "admitted"]
    return {
        "strategy": strategy.value,
        "optimizer": optimizer.name,
        "optimizer_version": optimizer_version(optimizer),
        "seed": seed,
        "run_id": identity["run_id"],
        "identity": identity,
        "budget": plan.budget.model_dump(mode="json"),
        "usage": ledger.totals(),
        "stop_reason": stop.value,
        "rounds": rnd,
        "evaluated_genome_hashes": [c["genome_hash"] for c in admitted],
        "distinct_genomes": len(genomes),
        "candidates": candidates,
        "curve": curve,
        "champion": champion_record,  # validation-selected under the contract's ranking
        "split_usage": {
            "optimization_runs": executed[SplitRole.OPTIMIZATION.value],
            "validation_runs": executed[SplitRole.VALIDATION.value],
            "test_runs": executed[SplitRole.TEST.value],
        },
        "model_hashes": sorted(model_hashes),
    }


# -- many runs -> one artifact ------------------------------------------------------------------
def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,  # sample std
        "min": min(values),
        "max": max(values),
    }


SUMMARY_METRICS = (
    "champion_validation_fitness",
    "champion_validation_pass_rate",
    "candidate_evaluations",
    "model_calls",
    "tokens",
    "wall_time_s",
    "cost",
)


def _per_seed(run: Mapping[str, Any]) -> dict[str, Any]:
    champ = run["champion"]
    usage = run["usage"]
    return {
        "seed": run["seed"],
        "run_id": run["run_id"],
        "stop_reason": run["stop_reason"],
        "champion_genome_hash": champ["genome_hash"] if champ else None,
        "champion_feasible": champ["feasible"] if champ else None,
        "champion_validation_fitness": champ["validation"]["fitness_mean"] if champ else None,
        "champion_validation_pass_rate": champ["validation"]["pass_rate"] if champ else None,
        **{k: usage[k] for k in ("candidate_evaluations", "model_calls", "tokens")},
        "wall_time_s": usage["wall_time_s"],
        "cost": usage["cost"],
    }


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-strategy mean/std/min/max over seeds. Every seed's own values stay listed."""
    out: dict[str, Any] = {}
    for strategy in dict.fromkeys(r["strategy"] for r in runs):
        mine = sorted((r for r in runs if r["strategy"] == strategy), key=lambda r: r["seed"])
        per_seed = [_per_seed(r) for r in mine]
        metrics = {
            name: _stats([p[name] for p in per_seed if p[name] is not None])
            for name in SUMMARY_METRICS
        }
        longest = max((len(r["curve"]) for r in mine), default=0)
        curve = [
            {
                "evaluation": x,
                "best_so_far_score": _stats(
                    [r["curve"][x - 1]["best_so_far_score"] for r in mine if len(r["curve"]) >= x]
                ),
            }
            for x in range(1, longest + 1)
        ]
        out[strategy] = {
            "seeds": [r["seed"] for r in mine],
            "metrics": metrics,
            "feasible_champions": sum(bool(p["champion_feasible"]) for p in per_seed),
            "per_seed": per_seed,
            "best_so_far_curve": curve,
        }
    means = {
        s: v["metrics"]["champion_validation_fitness"]["mean"]
        for s, v in out.items()
        if v["metrics"]["champion_validation_fitness"]["mean"] is not None
    }
    top = max(means.values(), default=None)
    return {
        "by_strategy": out,
        # Measured on VALIDATION rows (selection split); no test-split result exists here.
        "highest_mean_champion_validation_fitness": sorted(s for s, m in means.items() if m == top),
    }


def assemble(
    plan: ExperimentPlan,
    suite: ContractSuite,
    runs: Sequence[Mapping[str, Any]],
    *,
    evaluator_version: str,
    synthetic: bool,
    pricing: PricingPolicy | None = None,
) -> dict[str, Any]:
    """The canonical experiment artifact from independently produced strategy-run records."""
    problem = problem_identity(
        suite, plan.model, evaluator_version=evaluator_version, pricing=pricing
    )
    expected = {(s.value, seed) for s in plan.strategies for seed in plan.seeds}
    got = [(r["strategy"], r["seed"]) for r in runs]
    if sorted(got) != sorted(expected):
        raise ExperimentError("runs must cover every (strategy, seed) of the plan exactly once")
    for r in runs:
        if r["identity"]["problem_id"] != problem["problem_id"]:
            raise ExperimentError(f"run {r['run_id']} belongs to a different problem")
        if r["identity"]["protocol"] != plan.protocol():
            raise ExperimentError(f"run {r['run_id']} ran under a different budget/protocol")
    models = sorted({h for r in runs for h in r["model_hashes"]})
    if len(models) > 1:
        raise ExperimentError(f"strategies ran on different models: {models}")
    splits: DatasetSplits = suite.splits
    identity = {
        "problem": problem,
        "plan": plan.model_dump(mode="json"),
    }
    return {
        "schema": ARTIFACT_SCHEMA,
        "synthetic": synthetic,
        "experiment_id": canonical_hash(identity),
        "identity": identity,
        "fairness": {
            "unit": "candidate_evaluation",
            "candidate_evaluation": (
                "one proposed workflow run on every optimization row x trials (feedback) and "
                "every validation row x trials (selection)"
            ),
            "budget": plan.budget.model_dump(mode="json"),
            "fixed_baseline_rule": FixedBaseline.version,
        },
        "splits": {
            role.value: len(s.row_ids) if (s := splits.split(role)) else 0 for role in SplitRole
        },
        "test_runs": sum(r["split_usage"]["test_runs"] for r in runs),
        "model_hashes": models,
        "runs": list(runs),
        "summary": summarize(runs),
    }


def run_optimization_experiment(
    plan: ExperimentPlan,
    suite: ContractSuite,
    evaluate: EvaluateFn,
    *,
    checker: ConstraintChecker,
    evaluator_version: str,
    synthetic: bool,
    pricing: PricingPolicy | None = None,
    optimizer_factory: OptimizerFactory | None = None,
) -> dict[str, Any]:
    """Every strategy of ``plan`` x every seed, under one budget; returns the artifact.

    ``synthetic`` is mandatory: True whenever the model or objective is a stand-in, so such an
    artifact can never be mistaken for a real result.
    """
    check_budget(plan.budget, pricing)  # fail closed before ANY run
    search_tasks(suite)
    factory = optimizer_factory or make_strategy
    runs = [
        run_strategy(
            plan,
            strategy,
            seed,
            suite,
            evaluate,
            checker=checker,
            evaluator_version=evaluator_version,
            pricing=pricing,
            optimizer=factory(strategy, plan),
        )
        for strategy in plan.strategies
        for seed in plan.seeds
    ]
    return assemble(
        plan,
        suite,
        runs,
        evaluator_version=evaluator_version,
        synthetic=synthetic,
        pricing=pricing,
    )


def contract_evaluator_version(contract: TaskContract, evaluator: ContractEvaluator) -> str:
    """The ``evaluator_version`` a ``ContractEvaluator`` stamps on every generic evaluation."""
    return f"{contract.evaluation.evaluator_version}+{evaluator.fitness_fn.version}"


def optimize_uploaded_dataset(
    contract: TaskContract,
    splits: DatasetSplits,
    data: bytes,
    run_workflow: RunWorkflowFn,
    plan: ExperimentPlan,
    *,
    checker: ConstraintChecker,
    synthetic: bool,
    pricing: PricingPolicy | None = None,
    fitness: FitnessFunction | None = None,
    limits: IngestLimits | None = None,
) -> dict[str, Any]:
    """Uploaded dataset bytes + contract + splits -> fixed / random / ACO artifact.

    Only optimization + validation rows are bound to the evaluator, with only their expected
    values: a test row could not even be judged here.
    """
    check_budget(plan.budget, pricing)
    suite, references = contract_suite(contract, splits, data, limits=limits)
    train, val = search_tasks(suite)
    searchable = (*train, *val)
    evaluator = ContractEvaluator(
        References({t.id: references.expected(t.id) for t in searchable}), fitness=fitness
    )
    evaluate = make_evaluate_fn(run_workflow, evaluator, searchable)
    return run_optimization_experiment(
        plan,
        suite,
        evaluate,
        checker=checker,
        evaluator_version=contract_evaluator_version(contract, evaluator),
        synthetic=synthetic,
        pricing=pricing,
    )


def write_experiment(artifact: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """``experiment.json`` (canonical, everything) + one file per strategy run."""
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("not an optimization experiment artifact")
    paths = [out_dir / "experiment.json"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(_dumps(artifact), encoding="utf-8")
    for run in artifact["runs"]:
        path = out_dir / "runs" / run["strategy"] / f"seed-{run['seed']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _dumps({"experiment_id": artifact["experiment_id"], **run}), encoding="utf-8"
        )
        paths.append(path)
    return paths


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"
