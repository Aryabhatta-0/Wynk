"""Fixed vs random vs ACO on an uploaded dataset under one experiment budget (Issue #23).

Every model backend and objective here is a TEST DOUBLE (``synthetic_evaluate`` with injected,
deterministic usage, or a scripted model behind the real MAF runtime). Nothing here is a
benchmark result, and no test asserts which strategy wins.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from core.constraints import ConstraintChecker, ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitRole,
    SplitUse,
)
from core.evaluation_spec import EvaluationSpec
from core.experiment import ModelConfiguration
from core.genome import Genome
from core.results import BudgetUsage, EvaluatedRun, ExecutionMetrics
from core.run_contract import ExecutionTask
from core.stages import StageKind
from core.task_contract import TaskContract, TaskType, WorkflowSpec
from experiments.budget_ledger import (
    BudgetLedger,
    ExperimentBudget,
    ExperimentBudgetError,
    MeasuredUsage,
    PricingError,
)
from experiments.contract_run import contract_suite
from experiments.optimization_experiment import (
    ARTIFACT_SCHEMA,
    ExperimentError,
    ExperimentPlan,
    Strategy,
    make_strategy,
    optimize_uploaded_dataset,
    optimizer_version,
    run_optimization_experiment,
    run_seed,
    run_strategy,
    summarize,
    write_experiment,
)
from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate
from ingestion.parse import sha256_bytes
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import Optimizer, SearchContext
from optimizers.construct import path_edges
from optimizers.fixed_baseline import FixedBaseline, baseline_workflow
from optimizers.random_search import DistinctRandomSearch
from tests.contract_helpers import classification_contract
from tests.test_contract_runtime import DATA, FULL_CAPS, qa_contract, splits_for

MODEL = ModelConfiguration(provider="test-double", model="scripted")
ALL = (Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
RUNS_PER_CANDIDATE = 5  # capitals suite: 3 optimization + 2 validation rows, 1 trial


def suite_and_contract():
    contract = qa_contract()
    suite, _ = contract_suite(contract, splits_for(contract), DATA)
    return suite, contract


def plan(budget: ExperimentBudget | None = None, **overrides: Any) -> ExperimentPlan:
    base: dict[str, Any] = {
        "model": MODEL,
        "budget": budget or ExperimentBudget(max_candidate_evaluations=6),
        "seeds": (0, 1, 2),
    }
    return ExperimentPlan(**{**base, **overrides})


Usage = dict[str, float]
FLAT: Usage = {"model_calls": 2, "tokens": 300, "wall_time_s": 1.5}


def metered(usage: Usage | Callable[[Genome, ExecutionTask], Usage] = FLAT, calls=None):
    """Deterministic EvaluateFn: the synthetic objective + injected MEASURED usage."""

    def evaluate(genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
        run = synthetic_evaluate(genome, task, trial, seed)
        u = usage(genome, task) if callable(usage) else usage
        execution = run.execution.model_copy(
            update={
                "metrics": ExecutionMetrics(
                    model_calls=int(u["model_calls"]),
                    prompt_tokens=int(u["tokens"]) // 2,
                    completion_tokens=int(u["tokens"]) - int(u["tokens"]) // 2,
                ),
                "budget_usage": BudgetUsage(tokens=int(u["tokens"]), wall_time_s=u["wall_time_s"]),
            }
        )
        if calls is not None:
            calls.append(task.id)
        return EvaluatedRun(execution=execution, evaluation=run.evaluation)

    return evaluate


class LinearPricing:
    """TEST DOUBLE pricing policy: deterministic, provider-neutral, invented numbers."""

    identity = "test-linear-pricing/1"

    def __init__(self, per_call: float = 0.01, per_token: float = 0.0001) -> None:
        self.per_call, self.per_token = per_call, per_token

    def cost(self, usage: MeasuredUsage) -> float:
        return usage.model_calls * self.per_call + usage.tokens * self.per_token


FLAT_COST = 2 * 0.01 + 300 * 0.0001  # LinearPricing of one FLAT run


def run_all(budget: ExperimentBudget, evaluate=None, pricing=None, **plan_overrides):
    suite, _ = suite_and_contract()
    return run_optimization_experiment(
        plan(budget, **plan_overrides),
        suite,
        evaluate or metered(),
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        synthetic=True,
        pricing=pricing,
    )


def by_strategy(artifact) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in artifact["runs"]:
        out.setdefault(r["strategy"], []).append(r)
    return out


# -- 1. one budget rule set for every strategy --------------------------------------------------
def test_all_strategies_run_under_the_same_budget_rules_and_problem():
    budget = ExperimentBudget(max_candidate_evaluations=4, max_tokens=50_000)
    artifact = run_all(budget)
    assert artifact["schema"] == ARTIFACT_SCHEMA and artifact["synthetic"] is True
    runs = artifact["runs"]
    assert {(r["strategy"], r["seed"]) for r in runs} == {(s, d) for s in ALL for d in (0, 1, 2)}
    # identical budget, protocol and problem (dataset, splits, contract, grammar, evaluator,
    # model, ledger rules) for every strategy run; only strategy/optimizer/seed differ
    assert {json.dumps(r["budget"], sort_keys=True) for r in runs} == {
        json.dumps(budget.model_dump(mode="json"), sort_keys=True)
    }
    assert len({json.dumps(r["identity"]["protocol"], sort_keys=True) for r in runs}) == 1
    assert len({r["identity"]["problem_id"] for r in runs}) == 1
    assert artifact["fairness"]["unit"] == "candidate_evaluation"
    assert artifact["model_hashes"] == ["synthetic"]


# -- 2. equal candidate-evaluation caps ---------------------------------------------------------
@pytest.mark.parametrize("cap", [1, 5])
def test_every_strategy_gets_the_same_candidate_evaluation_cap(cap):
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=cap))
    for r in artifact["runs"]:
        used = r["usage"]["candidate_evaluations"]
        assert used <= cap and len(r["curve"]) == used
        assert r["usage"]["workflow_runs"] == used * RUNS_PER_CANDIDATE
        if r["strategy"] == "fixed":  # one workflow: done after one candidate evaluation
            assert used == 1
            assert r["stop_reason"] == (
                "candidate_evaluations" if cap == 1 else "strategy_exhausted"
            )
        else:  # random and ACO get exactly the whole cap, no more
            assert used == cap and r["stop_reason"] == "candidate_evaluations"
        assert [p["evaluation"] for p in r["curve"]] == list(range(1, used + 1))


# -- 3 + 4. equal model-call / token / wall-time / cost enforcement ----------------------------
RESOURCES = {
    "model_calls": ("max_model_calls", FLAT["model_calls"]),
    "tokens": ("max_tokens", FLAT["tokens"]),
    "wall_time_s": ("max_wall_time_s", FLAT["wall_time_s"]),
    "cost": ("max_cost", FLAT_COST),
}


@pytest.mark.parametrize("resource", sorted(RESOURCES))
@pytest.mark.parametrize("candidates_worth", [2.5, 0.5])
def test_resource_caps_are_enforced_identically_on_measured_usage(resource, candidates_worth):
    field, per_run = RESOURCES[resource]
    cap = candidates_worth * RUNS_PER_CANDIDATE * per_run
    budget = ExperimentBudget(max_candidate_evaluations=100, **{field: cap})
    pricing = LinearPricing() if resource == "cost" else None
    artifact = run_all(budget, pricing=pricing)
    full = math.floor(candidates_worth)  # candidates that fit completely
    for r in artifact["runs"]:
        used = r["usage"]
        # the ledger charged measured usage, run by run
        assert used["workflow_runs"] == len([e for c in r["candidates"] for e in c["runs"]])
        assert used[resource] == pytest.approx(per_run * used["workflow_runs"])
        # nothing on the curve was paid for beyond the cap; overshoot is at most one run
        if r["curve"]:
            assert r["curve"][-1][f"cumulative_{resource}"] <= cap
        assert used[resource] - per_run <= cap
        if r["strategy"] == "fixed" and full >= 1:
            assert used["candidate_evaluations"] == 1 and r["stop_reason"] == "strategy_exhausted"
        else:  # same rule, same stopping point for every strategy
            assert used["candidate_evaluations"] == full
            assert used["aborted_candidates"] == 1
            assert r["stop_reason"] == resource
            aborted = r["candidates"][-1]
            assert aborted["status"] == "over_budget" and aborted["evaluation"] is None
            assert "optimization" not in aborted and len(r["curve"]) == full
    if candidates_worth < 1:  # not even the baseline fits: nobody gets a champion
        assert all(r["champion"] is None and r["curve"] == [] for r in artifact["runs"])


def test_cost_is_priced_from_measured_usage_and_reported_on_the_curve():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=3), pricing=LinearPricing())
    assert artifact["identity"]["problem"]["pricing"] == LinearPricing.identity
    for r in artifact["runs"]:
        for point in r["curve"]:
            runs_so_far = point["evaluation"] * RUNS_PER_CANDIDATE
            assert point["cumulative_cost"] == pytest.approx(runs_so_far * FLAT_COST)
        entries = [e for c in r["candidates"] for e in c["runs"]]
        assert all(e["cost"] == pytest.approx(FLAT_COST) for e in entries)


# -- 5. cost cap without pricing fails closed ---------------------------------------------------
def test_a_cost_cap_without_a_pricing_policy_fails_closed_before_any_run():
    calls: list[str] = []
    budget = ExperimentBudget(max_candidate_evaluations=3, max_cost=1.0)
    with pytest.raises(ExperimentBudgetError, match="pricing policy"):
        run_all(budget, evaluate=metered(calls=calls))
    with pytest.raises(ExperimentBudgetError):
        BudgetLedger(budget)
    contract = qa_contract()

    def run_workflow(*_args):
        raise AssertionError("the runtime must not be reached")

    with pytest.raises(ExperimentBudgetError):
        optimize_uploaded_dataset(
            contract,
            splits_for(contract),
            DATA,
            run_workflow,
            plan(budget),
            checker=ConstraintChecker(),
            synthetic=True,
        )
    assert calls == []


def test_without_pricing_cost_is_unknown_not_zero():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=2))
    for r in artifact["runs"]:
        assert r["usage"]["cost"] is None
        assert all(p["cumulative_cost"] is None for p in r["curve"])
    assert artifact["identity"]["problem"]["pricing"] is None


def test_a_pricing_policy_that_cannot_price_fails_closed():
    class Refuses(LinearPricing):
        identity = "refuses/1"

        def cost(self, usage):
            raise KeyError(usage.model_hash)  # e.g. a model the table does not know

    class Negative(LinearPricing):
        identity = "negative/1"

        def cost(self, usage):
            return -1.0

    for pricing in (Refuses(), Negative()):
        with pytest.raises(PricingError):
            run_all(ExperimentBudget(max_candidate_evaluations=2), pricing=pricing)


# -- 6. random and ACO can never spend past the shared budget -----------------------------------
def _uneven(genome: Genome, task: ExecutionTask) -> Usage:
    k = len(genome) + int(task.id[1:])  # varies by workflow and row: strategies spend unevenly
    return {"model_calls": k, "tokens": 97 * k, "wall_time_s": 0.25 * k}


def test_no_strategy_spends_past_the_shared_budget_with_uneven_measured_usage():
    budget = ExperimentBudget(
        max_candidate_evaluations=8,
        max_model_calls=170,
        max_tokens=15_000,
        max_wall_time_s=40.0,
        max_cost=2.0,
    )
    pricing = LinearPricing()
    artifact = run_all(budget, evaluate=metered(_uneven), pricing=pricing)
    caps = {
        "model_calls": budget.max_model_calls,
        "tokens": budget.max_tokens,
        "wall_time_s": budget.max_wall_time_s,
        "cost": budget.max_cost,
    }
    for r in artifact["runs"]:
        assert r["usage"]["candidate_evaluations"] <= budget.max_candidate_evaluations
        # replay the ledger from the recorded MEASURED charges
        totals = dict.fromkeys(caps, 0.0)
        for cand in r["candidates"]:
            for i, entry in enumerate(cand["runs"]):
                # a run only starts while every resource has headroom
                assert all(totals[k] < caps[k] for k in caps)
                for k in caps:
                    totals[k] += entry[k]
                crossed = any(totals[k] > caps[k] for k in caps)
                if cand["status"] == "admitted":
                    assert not crossed
                else:  # aborted exactly at the run that crossed a cap, and nothing after it
                    assert crossed == (i == len(cand["runs"]) - 1)
        assert all(r["usage"][k] == pytest.approx(totals[k]) for k in caps)
        if r["candidates"] and r["candidates"][-1]["status"] == "over_budget":
            assert r["stop_reason"] in caps
        if r["curve"]:
            assert all(r["curve"][-1][f"cumulative_{k}"] <= caps[k] for k in caps)


# -- 7. optimizer feedback comes from optimization rows only ------------------------------------
class Spy(Optimizer):
    def __init__(self, inner: Optimizer) -> None:
        self.inner = inner
        self.name = inner.name
        self.version = optimizer_version(inner)
        self.observed: list[str] = []
        self.proposed: list[str] = []

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        out = self.inner.propose(k, context)
        self.proposed += [g.genome_hash for g in out]
        return out

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        self.observed += [r.execution.task_id for r in results]
        self.inner.observe(results)


def test_optimizers_only_ever_observe_optimization_rows():
    suite, contract = suite_and_contract()
    spies: dict[Strategy, list[Spy]] = {}

    def factory(strategy, p):
        spy = Spy(make_strategy(strategy, p))
        spies.setdefault(strategy, []).append(spy)
        return spy

    calls: list[str] = []
    artifact = run_optimization_experiment(
        plan(ExperimentBudget(max_candidate_evaluations=5)),
        suite,
        metered(calls=calls),
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        synthetic=True,
        optimizer_factory=factory,
    )
    splits = splits_for(contract)
    opt_rows = set(splits.split(SplitRole.OPTIMIZATION).row_ids)
    val_rows = set(splits.split(SplitRole.VALIDATION).row_ids)
    assert val_rows <= set(calls)  # validation rows were executed (for selection) ...
    for strategy in ALL:
        for spy in spies[strategy]:
            assert spy.observed and set(spy.observed) <= opt_rows  # ... but never observed
    for r in artifact["runs"]:
        assert r["split_usage"]["validation_runs"] > 0
        assert r["champion"]["validation"]["runs"] > 0  # selection used validation rows


# -- 8. the test split is never touched ---------------------------------------------------------
def test_the_test_split_is_never_executed_or_bound_to_the_evaluator():
    calls: list[str] = []
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=6), evaluate=metered(calls=calls))
    contract = qa_contract()
    test_rows = set(splits_for(contract).split(SplitRole.TEST).row_ids)
    assert test_rows and not test_rows & set(calls)
    assert artifact["test_runs"] == 0 and artifact["splits"]["test"] == len(test_rows)
    assert all(r["split_usage"]["test_runs"] == 0 for r in artifact["runs"])
    rows_in_artifact = {
        e["row_id"] for r in artifact["runs"] for c in r["candidates"] for e in c["runs"]
    }
    assert not rows_in_artifact & test_rows


def test_an_evaluator_returning_a_foreign_run_or_version_fails_closed():
    suite, _ = suite_and_contract()
    p = plan(ExperimentBudget(max_candidate_evaluations=2), strategies=(Strategy.FIXED,))

    def wrong_version(g, t, trial, seed):
        run = synthetic_evaluate(g, t, trial, seed)
        return run.model_copy(
            update={"evaluation": run.evaluation.model_copy(update={"evaluator_version": "x/9"})}
        )

    def wrong_job(g, t, trial, seed):
        return synthetic_evaluate(g, t, trial, seed + 1)

    for evaluate in (wrong_version, wrong_job):
        with pytest.raises(ExperimentError):
            run_strategy(
                p,
                Strategy.FIXED,
                0,
                suite,
                evaluate,
                checker=ConstraintChecker(),
                evaluator_version=SYNTHETIC_VERSION,
            )


# -- 9. same seed + contract -> identical curves ------------------------------------------------
def test_same_identity_reproduces_identical_candidate_order_curves_and_artifact():
    budget = ExperimentBudget(max_candidate_evaluations=7, max_tokens=40_000)
    a = run_all(budget, evaluate=metered(_uneven), pricing=LinearPricing())
    b = run_all(budget, evaluate=metered(_uneven), pricing=LinearPricing())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for ra, rb in zip(a["runs"], b["runs"], strict=True):
        assert ra["run_id"] == rb["run_id"]
        assert ra["evaluated_genome_hashes"] == rb["evaluated_genome_hashes"]
        assert ra["curve"] == rb["curve"]


def test_identity_covers_data_split_contract_grammar_evaluator_model_strategy_budget_seed():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=2))
    problem = artifact["identity"]["problem"]
    contract = qa_contract()
    assert problem["dataset_content_hash"] == contract.dataset.content_hash
    assert problem["dataset_hash"] == contract.dataset.identity_hash
    assert problem["splits_hash"] == splits_for(contract).identity_hash
    assert problem["task_contract_hash"] == contract.contract_hash
    assert problem["grammar_version"].startswith("grammar/")
    assert problem["evaluator_run_version"] == SYNTHETIC_VERSION
    assert problem["evaluation_hash"] == contract.evaluation.identity_hash
    assert problem["model"] == MODEL.model_dump(mode="json")
    for r in artifact["runs"]:
        ident = r["identity"]
        assert ident["strategy"] == r["strategy"] and ident["seed"] == r["seed"]
        assert ident["budget_hash"] == ExperimentBudget(max_candidate_evaluations=2).identity_hash
    # every component is load-bearing: change one, get a different run identity
    base = {r["run_id"] for r in artifact["runs"]}
    other_budget = run_all(ExperimentBudget(max_candidate_evaluations=3))
    other_model = run_all(
        ExperimentBudget(max_candidate_evaluations=2),
        model=ModelConfiguration(provider="test-double", model="other"),
    )
    assert base.isdisjoint(r["run_id"] for r in other_budget["runs"])
    assert base.isdisjoint(r["run_id"] for r in other_model["runs"])


# -- 10. different seeds are recorded separately ------------------------------------------------
def test_each_seed_is_recorded_and_persisted_independently(tmp_path):
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=4))
    for strategy, runs in by_strategy(artifact).items():
        assert [r["seed"] for r in runs] == [0, 1, 2]
        assert len({r["run_id"] for r in runs}) == 3
        per_seed = artifact["summary"]["by_strategy"][strategy]["per_seed"]
        assert [p["seed"] for p in per_seed] == [0, 1, 2]
        assert [p["run_id"] for p in per_seed] == [r["run_id"] for r in runs]
    # random's seeds are not one run copied three times
    random_orders = {tuple(r["evaluated_genome_hashes"]) for r in by_strategy(artifact)["random"]}
    assert len(random_orders) == 3
    paths = write_experiment(artifact, tmp_path)
    assert paths[0] == tmp_path / "experiment.json"
    assert json.loads(paths[0].read_text(encoding="utf-8")) == artifact
    for r in artifact["runs"]:
        path = tmp_path / "runs" / r["strategy"] / f"seed-{r['seed']}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["experiment_id"] == artifact["experiment_id"]
        assert {k: v for k, v in saved.items() if k != "experiment_id"} == r


# -- 11. aggregate statistics -------------------------------------------------------------------
def _fake_run(strategy: str, seed: int, fitness: float | None, curve: list[float]) -> dict:
    champ = (
        None
        if fitness is None
        else {
            "genome_hash": f"g{seed}",
            "feasible": True,
            "validation": {"fitness_mean": fitness, "pass_rate": fitness / 2},
        }
    )
    return {
        "strategy": strategy,
        "seed": seed,
        "run_id": f"{strategy}-{seed}",
        "stop_reason": "candidate_evaluations",
        "champion": champ,
        "usage": {
            "candidate_evaluations": len(curve),
            "model_calls": 10 * (seed + 1),
            "tokens": 100 * (seed + 1),
            "wall_time_s": 1.0,
            "cost": None,
        },
        "curve": [{"best_so_far_score": v} for v in curve],
    }


def test_aggregate_statistics_are_exact_and_keep_every_seed():
    runs = [
        _fake_run("random", 0, 0.2, [0.1, 0.2]),
        _fake_run("random", 1, 0.4, [0.4, 0.4, 0.4]),
        _fake_run("random", 2, 0.9, [0.9]),
        _fake_run("fixed", 0, 0.3, [0.3]),
        _fake_run("fixed", 1, None, []),
    ]
    summary = summarize(runs)
    random = summary["by_strategy"]["random"]
    fit = random["metrics"]["champion_validation_fitness"]
    assert fit["n"] == 3
    assert fit["mean"] == pytest.approx(0.5)
    assert fit["std"] == pytest.approx(statistics.stdev([0.2, 0.4, 0.9]))
    assert (fit["min"], fit["max"]) == (0.2, 0.9)
    calls = random["metrics"]["model_calls"]
    assert (calls["mean"], calls["min"], calls["max"]) == (20, 10, 30)
    assert random["metrics"]["cost"]["n"] == 0 and random["metrics"]["cost"]["mean"] is None
    assert [p["champion_validation_fitness"] for p in random["per_seed"]] == [0.2, 0.4, 0.9]
    # learning-curve aggregate, with how many seeds reached each evaluation
    curve = random["best_so_far_curve"]
    assert [c["best_so_far_score"]["n"] for c in curve] == [3, 2, 1]
    assert curve[0]["best_so_far_score"]["mean"] == pytest.approx((0.1 + 0.4 + 0.9) / 3)
    assert curve[2]["best_so_far_score"]["std"] is None  # one value: no sample std
    fixed = summary["by_strategy"]["fixed"]["metrics"]["champion_validation_fitness"]
    assert fixed["n"] == 1 and fixed["mean"] == 0.3  # the seed without a champion is excluded
    assert summary["by_strategy"]["fixed"]["per_seed"][1]["champion_validation_fitness"] is None
    assert summary["highest_mean_champion_validation_fitness"] == ["random"]


def test_artifact_summary_matches_its_own_per_seed_records():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=5))
    for strategy, runs in by_strategy(artifact).items():
        values = [r["champion"]["validation"]["fitness_mean"] for r in runs]
        got = artifact["summary"]["by_strategy"][strategy]["metrics"]["champion_validation_fitness"]
        assert got["mean"] == pytest.approx(statistics.fmean(values))
        assert got["std"] == pytest.approx(statistics.stdev(values))
        assert (got["min"], got["max"]) == (min(values), max(values))


# -- 12. ACO maths is unchanged -----------------------------------------------------------------
def _cls_task() -> ExecutionTask:
    contract = classification_contract(constraints=ConstraintLimits(**FULL_CAPS))
    return ExecutionTask(contract=contract, example={"row_id": "r1", "values": {"text": "x"}})


def _scored(genome: Genome, task: ExecutionTask, fitness: float) -> EvaluatedRun:
    run = synthetic_evaluate(genome, task, 0, 0)
    return run.model_copy(
        update={"evaluation": run.evaluation.model_copy(update={"fitness": fitness})}
    )


def test_mmas_pheromone_update_matches_its_closed_form():
    task = _cls_task()
    aco = MMASACO()
    cfg = aco.config
    assert (cfg.alpha, cfg.rho, cfg.tau_max, cfg.tau_min, cfg.global_best_period) == (
        1.0,
        0.30,
        1.0,
        0.05,
        5,
    )
    ctx = SearchContext(contract=task.contract, checker=ConstraintChecker(), seed=0)
    hi, lo, mid = aco.propose(3, ctx)
    aco.observe([_scored(hi, task, 1.0), _scored(lo, task, 0.0)])  # epoch 1: iteration-best = hi
    hi_edges, lo_edges, mid_edges = (set(path_edges(g)) for g in (hi, lo, mid))
    for e in hi_edges:  # evaporate 1.0 -> 0.7, deposit quality 1.0 * rho * tau_max, clamp to 1
        assert aco.pheromone(e) == pytest.approx(1.0)
    for e in lo_edges - hi_edges:
        assert aco.pheromone(e) == pytest.approx(0.7)
    aco.observe([_scored(mid, task, 0.5)])  # epoch 2: quality (0.5 - 0) / (1 - 0) = 0.5
    for e in mid_edges:
        expected = (1.0 * 0.7 if e in hi_edges else 0.49) + 0.5 * 0.30
        assert aco.pheromone(e) == pytest.approx(expected)
    for e in hi_edges - mid_edges:
        assert aco.pheromone(e) == pytest.approx(0.7)
    for e in lo_edges - hi_edges - mid_edges:
        assert aco.pheromone(e) == pytest.approx(0.49)


def test_the_runner_drives_mmas_exactly_like_a_plain_mmas_loop():
    suite, _ = suite_and_contract()
    p = plan(ExperimentBudget(max_candidate_evaluations=9), seeds=(3,), strategies=(Strategy.ACO,))
    captured: list[MMASACO] = []

    def factory(strategy, pl):
        captured.append(MMASACO(ACOConfig(lcb_z=pl.lcb_z)))
        return captured[0]

    evaluate = metered()
    artifact = run_optimization_experiment(
        p,
        suite,
        evaluate,
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        synthetic=True,
        optimizer_factory=factory,
    )
    # replay by hand: propose a batch, run each genome on the optimization rows, observe
    ref = MMASACO(ACOConfig(lcb_z=p.lcb_z))
    train = suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK)
    order: list[str] = []
    repeats: dict[str, int] = {}
    rnd = 0
    while len(order) < 9:
        ctx = SearchContext(contract=suite.policy, checker=ConstraintChecker(), seed=3, round=rnd)
        batch = []
        for g in ref.propose(p.batch_size, ctx)[: 9 - len(order)]:
            first = repeats.get(g.genome_hash, 0) * p.trials
            repeats[g.genome_hash] = repeats.get(g.genome_hash, 0) + 1
            order.append(g.genome_hash)
            batch += [
                evaluate(g, t, first + i, run_seed(3, g.genome_hash, t.id, first + i))
                for t in train
                for i in range(p.trials)
            ]
        ref.observe(batch)
        rnd += 1
    assert artifact["runs"][0]["evaluated_genome_hashes"] == order
    assert captured[0].epoch == ref.epoch
    assert captured[0].explored_pheromones() == ref.explored_pheromones()


# -- strategies ---------------------------------------------------------------------------------
def test_fixed_baseline_is_the_shortest_admissible_workflow_chosen_without_data():
    checker = ConstraintChecker()
    cls = classification_contract(constraints=ConstraintLimits(**FULL_CAPS))
    admissible = list(checker.enumerate_admissible(cls))
    shortest = min(len(g) for g in admissible)
    assert baseline_workflow(cls, checker) == next(g for g in admissible if len(g) == shortest)
    assert [s.kind for s in baseline_workflow(cls, checker).stages] == ["DIRECT"]
    retrieval_only = qa_contract(
        workflow=WorkflowSpec(
            stages=(StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE, StageKind.VERIFY)
        )
    )
    g = baseline_workflow(retrieval_only, checker)
    assert [s.kind for s in g.stages] == ["GATHER", "EXTRACT", "SYNTHESIZE"]
    assert checker.is_valid(g, retrieval_only)
    # proposed once, then nothing; it never learns from what it observes
    opt = FixedBaseline()
    ctx = SearchContext(contract=cls, checker=checker, seed=0)
    assert opt.propose(4, ctx) == [baseline_workflow(cls, checker)] and opt.propose(4, ctx) == []
    # the same contract over different data -> the same baseline (it reads no rows or scores)
    other = qa_contract(data=DATA.replace(b"Paris", b"Lyon "))
    assert baseline_workflow(other, checker) == baseline_workflow(qa_contract(), checker)


def test_random_search_never_repeats_a_workflow_until_the_space_is_exhausted():
    checker = ConstraintChecker()
    cls = classification_contract(constraints=ConstraintLimits(**FULL_CAPS))
    space = {g.genome_hash for g in checker.enumerate_admissible(cls)}
    opt = DistinctRandomSearch()
    seen: list[str] = []
    for rnd in range(len(space)):
        out = opt.propose(1, SearchContext(contract=cls, checker=checker, seed=0, round=rnd))
        seen += [g.genome_hash for g in out]
    assert len(seen) == len(set(seen)) == len(space) and set(seen) == space
    assert not opt.exhausted
    more = opt.propose(2, SearchContext(contract=cls, checker=checker, seed=0, round=99))
    assert opt.exhausted and len(more) == 2 and {g.genome_hash for g in more} <= space

    big = qa_contract()

    def order(seed: int) -> list[str]:
        o = DistinctRandomSearch()
        return [
            g.genome_hash
            for r in range(10)
            for g in o.propose(3, SearchContext(contract=big, checker=checker, seed=seed, round=r))
        ]

    assert len(set(order(0))) == 30  # no duplicates in a large space
    assert order(0) == order(0) and order(0) != order(1)  # the seed controls proposal order


def test_plan_rejects_ambiguous_seed_or_strategy_lists():
    with pytest.raises(ValueError):
        plan(seeds=(1, 1))
    with pytest.raises(ValueError):
        plan(strategies=(Strategy.ACO, Strategy.ACO))
    with pytest.raises(ValueError):
        plan(seeds=())


# -- the real thing: uploaded bytes -> contract_suite -> 3 strategies -> dispatcher -> curves --
CITIES = [
    ("c1", "France", "Paris"),
    ("c2", "Germany", "Berlin"),
    ("c3", "Italy", "Rome"),
    ("c4", "Spain", "Madrid"),
    ("c5", "Japan", "Tokyo"),
    ("c6", "Canada", "Ottawa"),
    ("c7", "Kenya", "Nairobi"),
    ("c8", "Peru", "Lima"),
]
UPLOAD = (
    "id,question,passage,answer\n"
    + "".join(
        f"{rid},Which city is the capital of {country}?,"
        f"{city} is the capital of {country}. It is a large city.,{city}\n"
        for rid, country, city in CITIES
    )
).encode()


def uploaded_contract() -> TaskContract:
    return TaskContract(
        task_id="capitals-upload",
        contract_version=1,
        task_type=TaskType.QUESTION_ANSWERING,
        instructions="Name the capital city, using only the passage.",
        input_schema={
            "fields": [
                {"name": "question", "type": "string"},
                {"name": "passage", "type": "string"},
            ]
        },
        output_schema={"fields": [{"name": "answer", "type": "string"}]},
        dataset=DatasetSpec(
            dataset_id="capitals-upload",
            dataset_version=1,
            name="Capitals (uploaded CSV)",
            content_hash=sha256_bytes(UPLOAD),
            format=DatasetFormat.CSV,
            columns=tuple(
                ColumnSpec(name=n, type=ColumnType.STRING)
                for n in ("id", "question", "passage", "answer")
            ),
            id_column="id",
            input_columns=("question",),
            context_columns=("passage",),
            target_columns=("answer",),
            row_count=len(CITIES),
        ),
        evaluation=EvaluationSpec(evaluator="exact_match", config={"case_sensitive": False}),
        constraints=ConstraintLimits(**FULL_CAPS),
    )


def uploaded_splits(contract: TaskContract) -> DatasetSplits:
    return DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("c1", "c2", "c3")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("c4", "c5")),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=("c6", "c7", "c8")),
        ),
    )


def _without_wall_time(obj: Any) -> Any:
    """The real runtime measures wall-clock time; everything else must reproduce exactly."""
    if isinstance(obj, dict):
        return {k: _without_wall_time(v) for k, v in obj.items() if "wall_time" not in k}
    if isinstance(obj, list):
        return [_without_wall_time(v) for v in obj]
    return obj


def test_uploaded_dataset_runs_fixed_random_and_aco_through_the_real_pipeline():
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner
    from tests.test_contract_runtime import PassageModel

    contract = uploaded_contract()
    splits = uploaded_splits(contract)
    test_rows = set(splits.split(SplitRole.TEST).row_ids)
    p = plan(ExperimentBudget(max_candidate_evaluations=4, max_tokens=200_000), seeds=(0, 1))

    def experiment() -> tuple[dict, list[str], PassageModel]:
        model = PassageModel()
        runner = WorkflowRunner(model=model, benchmark_hash="uploaded-inline")
        executed: list[str] = []

        def run_workflow(genome, task, trial, seed):
            executed.append(task.id)
            return runner.run_sync(genome, task, trial=trial, seed=seed)

        artifact = optimize_uploaded_dataset(
            contract, splits, UPLOAD, run_workflow, p, checker=runner.checker, synthetic=True
        )
        return artifact, executed, model

    artifact, executed, model = experiment()

    # the contract's own rows ran, by split role; the test rows never reached the runtime
    assert set(executed) == {"c1", "c2", "c3", "c4", "c5"} and not test_rows & set(executed)
    assert artifact["test_runs"] == 0
    assert model.requests and all(contract.instructions in r.input_text for r in model.requests)
    # judged by the EvaluationSpec dispatcher (exact_match), at its pinned version
    problem = artifact["identity"]["problem"]
    assert problem["evaluator_kind"] == "exact_match"
    assert problem["evaluator_run_version"].startswith(contract.evaluation.evaluator_version + "+")
    assert problem["dataset_content_hash"] == sha256_bytes(UPLOAD)
    assert artifact["model_hashes"] == [PassageModel.model_hash]

    runs = by_strategy(artifact)
    assert set(runs) == {"fixed", "random", "aco"}
    for strategy, mine in runs.items():
        for r in mine:
            used = r["usage"]["candidate_evaluations"]
            assert used == (1 if strategy == "fixed" else 4)
            assert len(r["curve"]) == used and r["champion"] is not None
            for point in r["curve"]:
                assert point["cumulative_model_calls"] > 0 and point["cumulative_tokens"] > 0
                assert 0.0 <= point["validation_pass_rate"] <= 1.0
            calls = [p["cumulative_model_calls"] for p in r["curve"]]
            assert calls == sorted(calls)
    # the fixed baseline is DIRECT: it cannot see the passage, and the report says so honestly
    for r in runs["fixed"]:
        assert [s["kind"] for s in r["champion"]["genome"]["stages"]] == ["DIRECT"]
    assert set(artifact["summary"]["by_strategy"]) == {"fixed", "random", "aco"}

    # same identity + deterministic backend -> same candidate order and curves
    again, _, _ = experiment()
    for ra, rb in zip(artifact["runs"], again["runs"], strict=True):
        assert ra["run_id"] == rb["run_id"]
        assert ra["evaluated_genome_hashes"] == rb["evaluated_genome_hashes"]
        assert _without_wall_time(ra["curve"]) == _without_wall_time(rb["curve"])
