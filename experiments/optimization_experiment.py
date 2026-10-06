"""Optimize an uploaded dataset: fixed baseline vs random search vs MMAS ACO under equal budgets.

    optimize_uploaded_dataset(contract, splits, data, run_workflow, plan, ...) -> artifact
      = contract_suite -> ContractEvaluator (EvaluationSpec dispatcher)
        -> run_optimization_experiment

One runner, ``run_strategy``, drives every strategy through the same loop. For one experiment all
strategies share, by construction:

    dataset bytes + DatasetSpec    the ``ContractSuite`` (rows + ``DatasetSplits``)
    TaskContract                   ``suite.policy``: admission, per-run limits, ranking
    workflow grammar / checker     one ``ConstraintChecker``, the contract's vocabulary
    model                          ``plan.expected_model_hash``: every run must report exactly it
                                   (``bind_model``: pinned in the registry, described by
                                   ``plan.model``, reported by the client - before any run)
    prompts                        ``plan.expected_prompt_version`` (when set) on every run
    evaluator                      one ``EvaluateFn`` whose evaluator_version must match
    seed policy                    ``plan.seeds``; run seeds = f(seed, genome, row, trial)
    experiment budget              ``plan.budget``, one fresh ``BudgetLedger`` per strategy run

Fairness invariant: the same candidate-evaluation budget gives every strategy the same
opportunity to spend it. A *candidate evaluation* is one proposed workflow run on every
optimization row x ``trials`` and every validation row x ``trials``. Before a candidate starts,
the ledger (``experiments.budget_ledger``) RESERVES its complete worst case from the contract's
authoritative per-run limits; if it does not fit, zero rows run. Afterwards the measured usage
is settled and the rest released, so final usage never exceeds a cap. The same genome under the
same experiment seed gets the same run seeds in every strategy (common random numbers), so
strategies differ only in WHICH workflows they try.

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
    test rows           never executed here. The runner refuses them; the uploaded-dataset entry
                        point does not even bind their expected values to the evaluator. The final
                        test result belongs to later promotion logic.

Score. A run's score is the evaluator's measured quality (e.g. token F1 for ``token_f1``; an
INFEASIBLE run scores 0); evaluators without a record (the synthetic objective) use fitness.
Curves plot the validation score against every resource axis: candidate evaluations, model
calls, tokens, execution time, end-to-end wall time and cost (when priced).

Time. ``latency`` is the runtime-measured wall time of one workflow run (model + stage
execution). Optimizer overhead (``propose`` / ``observe``), evaluator overhead and end-to-end
wall time are measured separately with ``clock`` and never mixed into latency. Clock-derived
values are reported under ``timing`` / ``*_e2e_*`` keys; they are not part of any identity and
are not expected to reproduce byte for byte.

Reproducibility: every strategy run carries an identity (dataset content + manifest + split +
contract hashes, grammar version, evaluator identity/version, model config + exact model hash,
prompt version, pricing, strategy + version, budget, seed) and its ``run_id``. With a
deterministic backend, the same identity gives the same candidate order and learning curve.

Durable execution (#24) can call ``run_strategy`` per (strategy, seed) and ``assemble`` the
records afterwards; ``run_optimization_experiment`` is just that loop.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from core.canonical import canonical_hash, canonical_json
from core.constraints import ConstraintChecker
from core.dataset import DatasetSplits, SplitRole
from core.experiment import ModelConfiguration, experiment_identity
from core.genome import Genome
from core.models import (
    AllowedModels,
    ModelEntry,
    ModelIdentityError,
    ModelRegistry,
    ModelRequirements,
)
from core.results import EvaluatedRun, ExecutionResult, FailureKind, Verdict
from core.run_contract import ContractSuite, ExecutionTask
from core.task_contract import ContractError, TaskContract, workflow_grammar
from evaluation.contract_eval import ContractEvaluator, References
from evaluation.fitness import FitnessFunction
from experiments.budget_ledger import (
    LEDGER_VERSION,
    BudgetLedger,
    ExperimentBudget,
    PricingPolicy,
    StopReason,
    check_budget,
    run_reservation,
)
from experiments.contract_run import contract_suite
from experiments.learning_curves import EvaluateFn, RunWorkflowFn, search_tasks
from ingestion.parse import IngestLimits
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import Optimizer, SearchContext
from optimizers.fixed_baseline import FIXED_RULES, FixedBaseline
from optimizers.random_search import DistinctRandomSearch
from optimizers.scoring import DEFAULT_Z, ScoreBoard

ARTIFACT_SCHEMA = "wynk-optimization-experiment/2"
RUNNER_VERSION = "optimization-runner/2"
MODEL_ATTEMPTS = 3  # a MODEL_ERROR run is retried inside its reservation, then surfaced
VALIDATION_TRIAL_OFFSET = 10_000  # validation runs never share a trial (hence seed) with feedback

Clock = Callable[[], float]


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
    # The exact ``ModelClient.model_hash`` every run must report (RunVersions.model_hash).
    expected_model_hash: str = Field(min_length=1)
    # The runtime prompt template version every run must report, when pinned.
    expected_prompt_version: str | None = None
    # Hash of the frozen dataset manifest (provenance + selected rows), when there is one.
    dataset_manifest_hash: str | None = None
    budget: ExperimentBudget
    seeds: tuple[int, ...] = Field(min_length=1)
    strategies: tuple[Strategy, ...] = (Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
    trials: PositiveInt = 1  # trials per (candidate, row)
    batch_size: PositiveInt = 2  # proposals per round (= ants per pheromone update)
    lcb_z: float = Field(default=DEFAULT_Z, ge=0.0, allow_inf_nan=False)
    # Pre-registered fixed-baseline rule; ``None`` = ``fixed_shortest/1`` (the original rule).
    fixed_rule: str | None = None
    # Hash of the frozen protocol document this plan was built from, when there is one.
    protocol_id: str | None = None

    @field_validator("fixed_rule")
    @classmethod
    def _known_fixed_rule(cls, v: str | None) -> str | None:
        if v is not None and v not in FIXED_RULES:
            raise ValueError(f"unknown fixed-baseline rule {v!r}; known: {sorted(FIXED_RULES)}")
        return v

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
        out: dict[str, Any] = {
            "budget": self.budget.model_dump(mode="json"),
            "trials": self.trials,
            "batch_size": self.batch_size,
            "lcb_z": self.lcb_z,
        }
        return out | self._optional()

    def _optional(self) -> dict[str, Any]:
        # Added after protocol-v1 ran; omitted while unset so v1 identities stay byte-identical.
        return {
            k: v
            for k, v in (("fixed_rule", self.fixed_rule), ("protocol_id", self.protocol_id))
            if v is not None
        }

    def identity_dump(self) -> dict[str, Any]:
        """The plan as hashed into the experiment identity (unset optional fields omitted)."""
        out = self.model_dump(mode="json", exclude={"fixed_rule", "protocol_id"})
        return out | self._optional()

    @property
    def fixed_baseline_rule(self) -> str:
        return self.fixed_rule or FixedBaseline.version


OptimizerFactory = Callable[[Strategy, ExperimentPlan], Optimizer]


# -- model binding ------------------------------------------------------------------------------
def check_models(plan: ExperimentPlan, models: AllowedModels) -> ModelEntry:
    """Fail closed unless the plan's model is in the declared allowed set and ``plan.model``
    describes that same registry entry (one identity: entry -> model_hash -> plan)."""
    entry = models.admit(plan.expected_model_hash)
    entry.check_configuration(plan.model)
    return entry


def bind_model(
    plan: ExperimentPlan,
    registry: ModelRegistry,
    client: Any | None = None,
    requirements: ModelRequirements | None = None,
) -> AllowedModels:
    """The experiment's allowed model set - exactly its expected model - resolved from the
    registry (unknown or disabled: fail closed), checked against ``plan.model``, ``requirements``
    and, given the ``ModelClient`` that will execute, its exact runtime ``model_hash``. Pass the
    result to the ``WorkflowRunner`` (``allowed_models``) and to the experiment (``models``)."""
    entry = registry.by_hash(plan.expected_model_hash)
    models = AllowedModels((entry,), requirements or ModelRequirements())
    check_models(plan, models)
    if client is not None and client.model_hash != entry.model_hash:
        raise ModelIdentityError(
            f"client reports model_hash {client.model_hash!r}, the experiment declares "
            f"{entry.model_hash!r} ({entry.name!r})"
        )
    return models


def make_strategy(strategy: Strategy, plan: ExperimentPlan) -> Optimizer:
    if strategy is Strategy.FIXED:
        return FIXED_RULES[plan.fixed_baseline_rule]()
    if strategy is Strategy.RANDOM:
        return DistinctRandomSearch()
    return MMASACO(ACOConfig(lcb_z=plan.lcb_z))


def optimizer_version(optimizer: Optimizer) -> str:
    return str(getattr(optimizer, "version", optimizer.name))


def run_seed(seed: int, genome_hash: str, task_id: str, trial: int) -> int:
    """Execution seed of one run (same derivation as ``experiments.learning_curves``)."""
    return int(canonical_hash([seed, genome_hash, task_id, trial])[:8], 16)


# -- evaluation with separate workflow / evaluator timing ---------------------------------------
Job = tuple[Genome, ExecutionTask, int, int]


class TimedEvaluate:
    """``EvaluateFn`` = runtime then contract evaluator, timing the two separately.

    Every task is checked up front (``evaluator.check_tasks``), so a task that cannot be judged
    fails before any model call; only the bound tasks can be evaluated at all.
    """

    def __init__(
        self,
        run_workflow: RunWorkflowFn,
        evaluator: ContractEvaluator,
        tasks: Iterable[ExecutionTask],
        clock: Clock = time.perf_counter,
    ) -> None:
        self._bound = {t.id: t for t in tasks}
        evaluator.check_tasks(self._bound.values())
        self._run, self._evaluator, self._clock = run_workflow, evaluator, clock
        self._lock = threading.Lock()
        self._timings: dict[tuple[str, str, int, int], tuple[float, float]] = {}

    def __call__(self, genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
        if self._bound.get(task.id) != task:
            raise ContractError(f"task {task.id} is not one this evaluator was bound to")
        t0 = self._clock()
        result: ExecutionResult = self._run(genome, task, trial, seed)
        t1 = self._clock()
        run = self._evaluator.evaluate_run(task, result)
        t2 = self._clock()
        with self._lock:
            self._timings[(genome.genome_hash, task.id, trial, seed)] = (t1 - t0, t2 - t1)
        return run

    def timing(self, genome: Genome, task: ExecutionTask, trial: int, seed: int):
        """``(workflow seconds, evaluator seconds)`` of the last call for this job."""
        with self._lock:
            return self._timings.pop((genome.genome_hash, task.id, trial, seed), None)


# -- identity -----------------------------------------------------------------------------------
def problem_identity(
    suite: ContractSuite,
    plan: ExperimentPlan,
    *,
    evaluator_version: str,
    pricing: PricingPolicy | None,
) -> dict[str, Any]:
    """WHAT is being optimized and how it is measured - shared by every strategy and seed."""
    contract = suite.policy
    core = experiment_identity(
        contract, suite.splits, grammar_version=workflow_grammar(contract).version, model=plan.model
    )
    out = {
        **core.model_dump(mode="json"),
        "experiment_identity_id": core.experiment_id,
        "dataset_id": contract.dataset.dataset_id,
        "dataset_version": contract.dataset.dataset_version,
        "dataset_content_hash": contract.dataset.content_hash,
        "dataset_manifest_hash": plan.dataset_manifest_hash,
        "suite_hash": suite.identity_hash,
        "evaluator_kind": contract.evaluation.evaluator.value,
        "evaluator_config": contract.evaluation.config,
        "evaluator_run_version": evaluator_version,  # what every EvaluatedRun must report
        "model": plan.model.model_dump(mode="json"),
        "model_hash": plan.expected_model_hash,
        "prompt_template_version": plan.expected_prompt_version,
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


# -- statistics ---------------------------------------------------------------------------------
def run_score(run: EvaluatedRun) -> float:
    """The evaluator's measured quality (INFEASIBLE: 0); fitness when there is no record."""
    record = run.evaluation.evaluator
    if record is None:
        return run.evaluation.fitness
    return record.quality if record.quality is not None else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in (0, 1]); the same rule as the contract's p95."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _split_stats(runs: Sequence[EvaluatedRun], entries: Sequence[Mapping[str, Any]]):
    tokens = sum(e["tokens"] for e in entries)
    return {
        "runs": len(runs),
        "score_mean": statistics.fmean(run_score(r) for r in runs),
        "pass_rate": sum(r.evaluation.verdict is Verdict.PASS for r in runs) / len(runs),
        "fitness_mean": statistics.fmean(r.evaluation.fitness for r in runs),
        "model_calls": sum(e["model_calls"] for e in entries),
        "prompt_tokens": sum(e["prompt_tokens"] for e in entries),
        "completion_tokens": sum(e["completion_tokens"] for e in entries),
        "tokens": tokens,
        "tokens_per_example": tokens / len(runs),
        "latency_s": distribution([r.execution.budget_usage.wall_time_s for r in runs]),
    }


# -- one strategy x one seed --------------------------------------------------------------------
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
    workers: int = 1,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Run ONE strategy for ONE seed under ``plan.budget``; returns its self-contained record.

    ``workers`` runs a candidate's workflow runs concurrently (results are consumed in
    submission order); it changes end-to-end time only, never what is reserved or selected.
    """
    t_start = clock()
    check_budget(plan.budget, pricing)
    if seed not in plan.seeds or strategy not in plan.strategies:
        raise ExperimentError(f"{strategy.value}/seed {seed} is not part of this plan")
    train, val = search_tasks(suite)
    problem = problem_identity(suite, plan, evaluator_version=evaluator_version, pricing=pricing)
    optimizer = optimizer or make_strategy(strategy, plan)
    identity = run_identity(problem, plan, strategy, optimizer, seed)
    per_run = run_reservation(plan.budget, suite.policy, pricing, plan.expected_model_hash)
    ledger = BudgetLedger(plan.budget, pricing, per_run)
    runs_per_candidate = (len(train) + len(val)) * plan.trials
    timing_of = getattr(evaluate, "timing", None)

    board = ScoreBoard(plan.lcb_z)  # optimization-row score per genome (what the optimizer sees)
    repeats: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    genomes: dict[str, Genome] = {}
    val_runs: dict[str, list[EvaluatedRun]] = {}
    ranks: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    all_latency: list[float] = []
    executed = {role.value: 0 for role in SplitRole}
    run_versions: set[str] = set()
    timing = {"optimizer_s": 0.0, "evaluate_calls_s": 0.0, "evaluator_s": 0.0}
    evaluator_timed = timing_of is not None
    champion: str | None = None
    max_score = -math.inf

    def call(job: Job) -> tuple[EvaluatedRun, float, tuple[float, float] | None]:
        t0 = clock()
        run = evaluate(*job)
        dt = clock() - t0
        return run, dt, timing_of(*job) if timing_of is not None else None

    def check(run: EvaluatedRun, job: Job) -> SplitRole:
        genome, task, trial, rs = job
        role = suite.splits.role_of(task.id)
        if role not in (SplitRole.OPTIMIZATION, SplitRole.VALIDATION):
            raise ExperimentError(f"row {task.id} is not an optimization or validation row")
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
        if key.versions.model_hash != plan.expected_model_hash:
            raise ExperimentError(
                f"run executed by model {key.versions.model_hash!r}, experiment is bound to "
                f"{plan.expected_model_hash!r}"
            )
        prompts = plan.expected_prompt_version
        if prompts is not None and key.versions.prompt_template_version != prompts:
            raise ExperimentError(
                f"run used prompts {key.versions.prompt_template_version!r}, experiment is "
                f"bound to {prompts!r}"
            )
        run_versions.add(canonical_json(key.versions.model_dump(mode="json")))
        if len(run_versions) > 1:
            raise ExperimentError("runs of one strategy run reported different run versions")
        executed[role.value] += 1
        return role

    def entry(run, job, role, attempt, call_s, ev) -> dict[str, Any]:
        failure = run.execution.failure
        model_error = failure is not None and failure.kind is FailureKind.MODEL_ERROR
        answer = run.execution.answer
        return {
            "row_id": job[1].id,
            "split": role.value,
            "trial": job[2],
            "attempt": attempt,
            "verdict": None if model_error else run.evaluation.verdict.value,
            "score": None if model_error else run_score(run),
            "fitness": None if model_error else run.evaluation.fitness,
            "failure": failure.kind.value if failure else None,
            "prediction": answer.values if answer is not None else None,
            **ledger.measure(run),
            "timing": {
                "call_s": call_s,
                "workflow_s": ev[0] if ev else None,
                "evaluator_s": ev[1] if ev else None,
            },
        }

    def is_model_error(run: EvaluatedRun) -> bool:
        f = run.execution.failure
        return f is not None and f.kind is FailureKind.MODEL_ERROR

    def evaluate_candidate(
        genome: Genome, round_: int, pool: ThreadPoolExecutor | None
    ) -> tuple[list[EvaluatedRun], StopReason | None]:
        nonlocal champion, max_score
        stop = ledger.reserve(runs_per_candidate)
        if stop is not None:  # the complete candidate cannot be paid for: run ZERO rows
            return [], stop
        t0 = clock()
        h = genome.genome_hash
        rep = repeats.get(h, 0)
        first = rep * plan.trials
        layout = [(t, first + i) for t in train for i in range(plan.trials)] + [
            (t, VALIDATION_TRIAL_OFFSET + first + i) for t in val for i in range(plan.trials)
        ]
        jobs: list[Job] = [(genome, t, tr, run_seed(seed, h, t.id, tr)) for t, tr in layout]
        results = list(pool.map(call, jobs)) if pool is not None else [call(j) for j in jobs]

        entries: list[dict[str, Any]] = []
        final: list[EvaluatedRun] = []
        try:
            for job, (run, call_s, ev) in zip(jobs, results, strict=True):
                role = check(run, job)
                entries.append(entry(run, job, role, 0, call_s, ev))
                attempt = 1
                while is_model_error(run):  # retries are paid from the open reservation
                    spent = {k: sum((e[k] or 0.0) for e in entries) for k in per_run.as_dict()}
                    if attempt >= MODEL_ATTEMPTS or not ledger.slack(spent):
                        raise ModelUnavailable(
                            f"model unavailable on {job[1].id} after {attempt} attempt(s)"
                        )
                    run, call_s, ev = call(job)
                    role = check(run, job)
                    entries.append(entry(run, job, role, attempt, call_s, ev))
                    attempt += 1
                final.append(run)
        except BaseException:
            ledger.settle(entries, admitted=False)  # commit real spend, then fail closed
            raise
        ledger.settle(entries, admitted=True)  # raises ReservationOverflow if limits broke

        for e in entries:
            timing["evaluate_calls_s"] += e["timing"]["call_s"]
            if e["timing"]["evaluator_s"] is not None:
                timing["evaluator_s"] += e["timing"]["evaluator_s"]
        n_opt = len(train) * plan.trials
        opt, sel = final[:n_opt], final[n_opt:]
        opt_entries = [e for e in entries if e["split"] == SplitRole.OPTIMIZATION.value]
        sel_entries = [e for e in entries if e["split"] == SplitRole.VALIDATION.value]
        all_latency.extend(r.execution.budget_usage.wall_time_s for r in final)

        n = ledger.candidate_evaluations
        repeats[h] = rep + 1
        first_seen.setdefault(h, n)
        genomes[h] = genome
        board.add(opt)
        val_runs.setdefault(h, []).extend(sel)
        ranks[h] = suite.rank(genome, val_runs[h])

        def selection_key(k: str) -> tuple:
            score = statistics.fmean(run_score(r) for r in val_runs[k])
            return (ranks[k].sort_key, score, -first_seen[k])

        champion = max(ranks, key=selection_key)
        cand_val = _split_stats(sel, sel_entries)
        max_score = max(max_score, cand_val["score_mean"])
        champ_runs = val_runs[champion]
        totals = ledger.totals()
        cost = None if ledger.cost is None else sum(e["cost"] for e in entries)
        e2e = clock() - t0
        candidates.append(
            {
                "evaluation": n,
                "round": round_,
                "genome_hash": h,
                "genome": genome.canonical(),
                "repeat": rep > 0,
                "optimization": {
                    **_split_stats(opt, opt_entries),
                    "genome_lcb": board.score(h).lcb,
                },
                "validation": cand_val,
                "selection": {
                    "feasible": ranks[h].feasible,
                    "rank_key": list(ranks[h].sort_key),
                    "violations": [v.message for v in ranks[h].violations],
                },
                "usage": {
                    k: sum(e[k] for e in entries)
                    for k in ("model_calls", "prompt_tokens", "completion_tokens", "tokens")
                }
                | {
                    "workflow_runs": len(entries),
                    "execution_s": sum(e["wall_time_s"] for e in entries),
                    "cost": cost,
                },
                "timing": {
                    "e2e_s": e2e,
                    "evaluator_s": sum(e["timing"]["evaluator_s"] or 0.0 for e in entries)
                    if evaluator_timed
                    else None,
                },
                "runs": entries,
            }
        )
        curve.append(
            {
                "evaluation": n,
                "genome_hash": h,
                "score": cand_val["score_mean"],
                "validation_pass_rate": cand_val["pass_rate"],
                "optimization_score": statistics.fmean(run_score(r) for r in opt),
                "best_so_far_score": statistics.fmean(run_score(r) for r in champ_runs),
                "best_so_far_pass_rate": (
                    sum(r.evaluation.verdict is Verdict.PASS for r in champ_runs) / len(champ_runs)
                ),
                "best_so_far_genome_hash": champion,
                "max_score_so_far": max_score,
                "cumulative_model_calls": totals["model_calls"],
                "cumulative_prompt_tokens": totals["prompt_tokens"],
                "cumulative_completion_tokens": totals["completion_tokens"],
                "cumulative_tokens": totals["tokens"],
                "cumulative_execution_s": totals["wall_time_s"],
                "cumulative_cost": totals["cost"],
                "cumulative_e2e_wall_s": clock() - t_start,
            }
        )
        return opt, None

    def timed(fn, *args):
        t0 = clock()
        try:
            return fn(*args)
        finally:
            timing["optimizer_s"] += clock() - t0

    stop: StopReason | None = None
    rnd = 0
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while stop is None:
            stop = ledger.can_reserve(runs_per_candidate)  # can a complete candidate start?
            if stop is not None:
                break
            context = SearchContext(contract=suite.policy, checker=checker, seed=seed, round=rnd)
            proposals = timed(optimizer.propose, plan.batch_size, context)
            if not proposals:
                stop = StopReason.STRATEGY_EXHAUSTED
                break
            feedback: list[EvaluatedRun] = []
            for genome in proposals:
                opt_runs, stop = evaluate_candidate(genome, rnd, pool)
                if stop is not None:
                    break
                feedback += opt_runs
            if feedback:
                suite.check_feedback(feedback)  # only optimization rows may reach optimizer state
                timed(optimizer.observe, feedback)
            rnd += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    champion_record = None
    if champion is not None:
        rank = ranks[champion]
        champ_runs = val_runs[champion]
        champion_record = {
            "genome_hash": champion,
            "genome": genomes[champion].canonical(),
            "feasible": rank.feasible,
            "rank_key": list(rank.sort_key),
            "violations": [v.message for v in rank.violations],
            "first_evaluation": first_seen[champion],
            "validation": {
                "runs": len(champ_runs),
                "score_mean": statistics.fmean(run_score(r) for r in champ_runs),
                "pass_rate": sum(r.evaluation.verdict is Verdict.PASS for r in champ_runs)
                / len(champ_runs),
                "fitness_mean": statistics.fmean(r.evaluation.fitness for r in champ_runs),
            },
        }
    usage = ledger.totals()
    e2e = clock() - t_start
    n_cand = usage["candidate_evaluations"]
    return {
        "strategy": strategy.value,
        "optimizer": optimizer.name,
        "optimizer_version": optimizer_version(optimizer),
        "seed": seed,
        "run_id": identity["run_id"],
        "identity": identity,
        "budget": plan.budget.model_dump(mode="json"),
        "reservation_per_run": per_run.as_dict(),
        "usage": usage,
        "efficiency": {
            "tokens_per_example": usage["tokens"] / usage["workflow_runs"]
            if usage["workflow_runs"]
            else None,
            "tokens_per_candidate": usage["tokens"] / n_cand if n_cand else None,
            "cost_per_candidate": (usage["cost"] / n_cand)
            if (n_cand and usage["cost"] is not None)
            else None,
        },
        "latency_s": distribution(all_latency),
        "timing": {
            "e2e_wall_s": e2e,
            "execution_s": usage["wall_time_s"],
            "optimizer_overhead_s": timing["optimizer_s"],
            "evaluator_overhead_s": timing["evaluator_s"] if evaluator_timed else None,
            "evaluate_calls_s": timing["evaluate_calls_s"],
            "examples_per_s": usage["workflow_runs"] / e2e if e2e > 0 else None,
            "candidates_per_min": 60.0 * n_cand / e2e if e2e > 0 else None,
            "candidate_e2e_s": distribution([c["timing"]["e2e_s"] for c in candidates]),
            "workers": workers,
        },
        "stop_reason": stop.value,
        "rounds": rnd,
        "evaluated_genome_hashes": [c["genome_hash"] for c in candidates],
        "distinct_genomes": len(genomes),
        "candidates": candidates,
        "curve": curve,
        "champion": champion_record,  # validation-selected under the contract's ranking
        "split_usage": {
            "optimization_runs": executed[SplitRole.OPTIMIZATION.value],
            "validation_runs": executed[SplitRole.VALIDATION.value],
            "test_runs": executed[SplitRole.TEST.value],
        },
        "model_hashes": sorted({json.loads(v)["model_hash"] for v in run_versions}),
        "run_versions": [json.loads(v) for v in sorted(run_versions)],
    }


# -- many runs -> one artifact ------------------------------------------------------------------
def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,  # sample std
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


SUMMARY_METRICS = (
    "champion_validation_score",
    "champion_validation_pass_rate",
    "candidate_evaluations",
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "tokens_per_example",
    "tokens_per_candidate",
    "mean_latency_s",
    "p50_latency_s",
    "p95_latency_s",
    "max_latency_s",
    "execution_s",
    "mean_candidate_e2e_s",
    "e2e_wall_s",
    "optimizer_overhead_s",
    "evaluator_overhead_s",
    "examples_per_s",
    "candidates_per_min",
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
        "champion_validation_score": champ["validation"]["score_mean"] if champ else None,
        "champion_validation_pass_rate": champ["validation"]["pass_rate"] if champ else None,
        **{
            k: usage[k]
            for k in (
                "candidate_evaluations",
                "model_calls",
                "prompt_tokens",
                "completion_tokens",
                "tokens",
                "cost",
            )
        },
        "tokens_per_example": run["efficiency"].get("tokens_per_example"),
        "tokens_per_candidate": run["efficiency"]["tokens_per_candidate"],
        "mean_latency_s": run["latency_s"]["mean"],
        "p50_latency_s": run["latency_s"].get("p50"),
        "p95_latency_s": run["latency_s"]["p95"],
        "max_latency_s": run["latency_s"].get("max"),
        "execution_s": run["timing"]["execution_s"],
        "mean_candidate_e2e_s": (run["timing"].get("candidate_e2e_s") or {}).get("mean"),
        "e2e_wall_s": run["timing"]["e2e_wall_s"],
        "optimizer_overhead_s": run["timing"]["optimizer_overhead_s"],
        "evaluator_overhead_s": run["timing"]["evaluator_overhead_s"],
        "examples_per_s": run["timing"].get("examples_per_s"),
        "candidates_per_min": run["timing"].get("candidates_per_min"),
    }


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-strategy mean/std/median/min/max over seeds. Every seed's own values stay listed."""
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
        s: v["metrics"]["champion_validation_score"]["mean"]
        for s, v in out.items()
        if v["metrics"]["champion_validation_score"]["mean"] is not None
    }
    top = max(means.values(), default=None)
    return {
        "by_strategy": out,
        # Measured on VALIDATION rows (selection split); no test-split result exists here.
        "highest_mean_champion_validation_score": sorted(s for s, m in means.items() if m == top),
    }


CURVE_AXES = (
    "evaluation",
    "cumulative_model_calls",
    "cumulative_tokens",
    "cumulative_execution_s",
    "cumulative_e2e_wall_s",
    "cumulative_cost",
)


def resource_curves(run: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Validation best-so-far score against every resource axis (``None`` axes are omitted)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for axis in CURVE_AXES:
        points = [(p[axis], p["best_so_far_score"]) for p in run["curve"]]
        if all(x is not None for x, _ in points):
            out[axis] = points
    return out


def assemble(
    plan: ExperimentPlan,
    suite: ContractSuite,
    runs: Sequence[Mapping[str, Any]],
    *,
    evaluator_version: str,
    synthetic: bool,
    pricing: PricingPolicy | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The canonical experiment artifact from independently produced strategy-run records."""
    problem = problem_identity(suite, plan, evaluator_version=evaluator_version, pricing=pricing)
    expected = {(s.value, seed) for s in plan.strategies for seed in plan.seeds}
    got = [(r["strategy"], r["seed"]) for r in runs]
    if sorted(got) != sorted(expected):
        raise ExperimentError("runs must cover every (strategy, seed) of the plan exactly once")
    for r in runs:
        if r["identity"]["problem_id"] != problem["problem_id"]:
            raise ExperimentError(f"run {r['run_id']} belongs to a different problem")
        if r["identity"]["protocol"] != plan.protocol():
            raise ExperimentError(f"run {r['run_id']} ran under a different budget/protocol")
        if any(h != plan.expected_model_hash for h in r["model_hashes"]):
            raise ExperimentError(
                f"run {r['run_id']} executed on {r['model_hashes']}, experiment is bound to "
                f"{plan.expected_model_hash!r}"
            )
    versions = {canonical_json(v) for r in runs for v in r["run_versions"]}
    if len(versions) > 1:
        raise ExperimentError("strategies ran under different model/prompt/compiler/grammar")
    splits: DatasetSplits = suite.splits
    identity = {"problem": problem, "plan": plan.identity_dump()}
    return {
        "schema": ARTIFACT_SCHEMA,
        "synthetic": synthetic,
        "experiment_id": canonical_hash(identity),
        "identity": identity,
        "provenance": dict(provenance) if provenance is not None else None,
        "fairness": {
            "unit": "candidate_evaluation",
            "candidate_evaluation": (
                "one proposed workflow run on every optimization row x trials (feedback) and "
                "every validation row x trials (selection)"
            ),
            "budget": plan.budget.model_dump(mode="json"),
            "enforcement": "reserve complete candidate from contract limits -> run -> settle",
            "fixed_baseline_rule": plan.fixed_baseline_rule,
        },
        "splits": {
            role.value: len(s.row_ids) if (s := splits.split(role)) else 0 for role in SplitRole
        },
        "test_runs": sum(r["split_usage"]["test_runs"] for r in runs),
        "model_hashes": sorted({h for r in runs for h in r["model_hashes"]}),
        "run_versions": [json.loads(v) for v in sorted(versions)],
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
    provenance: Mapping[str, Any] | None = None,
    workers: int = 1,
    clock: Clock = time.perf_counter,
    on_run: Callable[[Mapping[str, Any]], None] | None = None,
    models: AllowedModels | None = None,
) -> dict[str, Any]:
    """Every strategy of ``plan`` x every seed, under one budget; returns the artifact.

    ``synthetic`` is mandatory: True whenever the model or objective is a stand-in, so such an
    artifact can never be mistaken for a real result. ``on_run`` receives each strategy-run
    record as soon as it completes (e.g. to persist it). ``models`` is the declared allowed
    model set (``bind_model``); the plan must fit it before any run.
    """
    if models is not None:
        check_models(plan, models)
    check_budget(plan.budget, pricing)  # fail closed before ANY run
    run_reservation(plan.budget, suite.policy, pricing, plan.expected_model_hash)
    search_tasks(suite)
    factory = optimizer_factory or make_strategy
    runs = []
    for seed in plan.seeds:
        for strategy in plan.strategies:
            record = run_strategy(
                plan,
                strategy,
                seed,
                suite,
                evaluate,
                checker=checker,
                evaluator_version=evaluator_version,
                pricing=pricing,
                optimizer=factory(strategy, plan),
                workers=workers,
                clock=clock,
            )
            if on_run is not None:
                on_run(record)
            runs.append(record)
    return assemble(
        plan,
        suite,
        runs,
        evaluator_version=evaluator_version,
        synthetic=synthetic,
        pricing=pricing,
        provenance=provenance,
    )


def contract_evaluator_version(contract: TaskContract, evaluator: ContractEvaluator) -> str:
    """The ``evaluator_version`` a ``ContractEvaluator`` stamps on every generic evaluation."""
    return f"{contract.evaluation.evaluator_version}+{evaluator.fitness_fn.version}"


def searchable_evaluator(
    contract: TaskContract,
    splits: DatasetSplits,
    data: bytes,
    run_workflow: RunWorkflowFn,
    *,
    fitness: FitnessFunction | None = None,
    limits: IngestLimits | None = None,
) -> tuple[ContractSuite, TimedEvaluate, str]:
    """``(suite, evaluate, evaluator_version)`` with ONLY optimization + validation rows (and
    only their expected values) bound to the evaluator: a test row cannot even be judged."""
    suite, references = contract_suite(contract, splits, data, limits=limits)
    train, val = search_tasks(suite)
    searchable = (*train, *val)
    evaluator = ContractEvaluator(
        References({t.id: references.expected(t.id) for t in searchable}), fitness=fitness
    )
    evaluate = TimedEvaluate(run_workflow, evaluator, searchable)
    return suite, evaluate, contract_evaluator_version(contract, evaluator)


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
    provenance: Mapping[str, Any] | None = None,
    workers: int = 1,
    on_run: Callable[[Mapping[str, Any]], None] | None = None,
    models: AllowedModels | None = None,
) -> dict[str, Any]:
    """Uploaded dataset bytes + contract + splits -> fixed / random / ACO artifact.

    ``models`` (``bind_model``) declares the allowed model set; registry prices, when every
    allowed model has them, come from ``core.models.registry_pricing(models)``."""
    if models is not None:
        check_models(plan, models)
    check_budget(plan.budget, pricing)
    run_reservation(plan.budget, contract, pricing, plan.expected_model_hash)
    suite, evaluate, version = searchable_evaluator(
        contract, splits, data, run_workflow, fitness=fitness, limits=limits
    )
    return run_optimization_experiment(
        plan,
        suite,
        evaluate,
        checker=checker,
        evaluator_version=version,
        synthetic=synthetic,
        pricing=pricing,
        provenance=provenance,
        workers=workers,
        on_run=on_run,
        models=models,
    )


def write_experiment(artifact: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """``experiment.json`` (canonical, everything) + one file per strategy run."""
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("not an optimization experiment artifact")
    paths = [out_dir / "experiment.json"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(dumps(artifact), encoding="utf-8")
    for run in artifact["runs"]:
        path = out_dir / "runs" / run["strategy"] / f"seed-{run['seed']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            dumps({"experiment_id": artifact["experiment_id"], **run}), encoding="utf-8"
        )
        paths.append(path)
    return paths


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


# -- compact persistence: one compressed canonical artifact + a small reviewable summary --------
ARTIFACT_GZ = "experiment.json.gz"
SUMMARY_SCHEMA = "wynk-optimization-summary/1"
_RUN_SUMMARY_KEYS = (
    "strategy",
    "seed",
    "run_id",
    "identity",
    "optimizer",
    "optimizer_version",
    "budget",
    "reservation_per_run",
    "usage",
    "efficiency",
    "latency_s",
    "timing",
    "stop_reason",
    "rounds",
    "evaluated_genome_hashes",
    "distinct_genomes",
    "champion",
    "curve",
    "split_usage",
    "model_hashes",
)


def compact_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Everything in ``artifact`` except per-candidate run entries (those stay in the .gz)."""
    provenance = artifact.get("provenance") or {}
    return {
        "schema": SUMMARY_SCHEMA,
        "artifact_schema": artifact["schema"],
        "synthetic": artifact["synthetic"],
        "experiment_id": artifact["experiment_id"],
        "identity": artifact["identity"],
        "manifest_hash": provenance.get("manifest_hash"),
        "fairness": artifact["fairness"],
        "splits": artifact["splits"],
        "test_runs": artifact["test_runs"],
        "model_hashes": artifact["model_hashes"],
        "run_versions": artifact["run_versions"],
        "runs": [{k: r.get(k) for k in _RUN_SUMMARY_KEYS} for r in artifact["runs"]],
        "summary": artifact["summary"],
    }


def write_compact(artifact: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write ``experiment.json.gz`` (deterministic bytes) + ``summary.json``; returns the summary.

    The summary records the SHA-256 of both the canonical JSON and the gzip file, so the raw
    artifact can be verified (``load_compact``) without committing it uncompressed.
    """
    import gzip
    import hashlib

    raw = dumps(artifact).encode("utf-8")
    gz = gzip.compress(raw, compresslevel=9, mtime=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ARTIFACT_GZ).write_bytes(gz)
    summary = compact_summary(artifact) | {
        "artifact": {
            "file": ARTIFACT_GZ,
            "json_sha256": hashlib.sha256(raw).hexdigest(),
            "gz_sha256": hashlib.sha256(gz).hexdigest(),
            "json_bytes": len(raw),
        }
    }
    (out_dir / "summary.json").write_text(dumps(summary), encoding="utf-8")
    return summary


def load_compact(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(summary, artifact)`` after verifying both recorded SHA-256 digests (fail closed)."""
    import gzip
    import hashlib

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    meta = summary["artifact"]
    gz = (out_dir / meta["file"]).read_bytes()
    if hashlib.sha256(gz).hexdigest() != meta["gz_sha256"]:
        raise ExperimentError(f"{meta['file']} does not match its recorded gzip sha256")
    raw = gzip.decompress(gz)
    if hashlib.sha256(raw).hexdigest() != meta["json_sha256"]:
        raise ExperimentError(f"{meta['file']} does not match its recorded json sha256")
    return summary, json.loads(raw)


CLOCK_KEYS = frozenset(
    {
        "timing",
        "cumulative_e2e_wall_s",
        "e2e_wall_s",
        "optimizer_overhead_s",
        "evaluator_overhead_s",
        "mean_candidate_e2e_s",
        "examples_per_s",
        "candidates_per_min",
    }
)


def without_timing(obj: Any) -> Any:
    """``obj`` minus every clock-derived value (the part that must reproduce exactly)."""
    if isinstance(obj, dict):
        return {k: without_timing(v) for k, v in obj.items() if k not in CLOCK_KEYS}
    if isinstance(obj, list):
        return [without_timing(v) for v in obj]
    return obj
