"""Fixed vs random vs ACO on an uploaded dataset under one experiment budget (Issue #23).

Every model backend and objective here is a TEST DOUBLE (``synthetic_evaluate`` with injected,
deterministic usage, or a scripted model behind the real MAF runtime). Nothing here is a
benchmark result, and no test asserts which strategy wins.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

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
    ReservationOverflow,
)
from experiments.contract_run import contract_suite
from experiments.optimization_experiment import (
    ARTIFACT_SCHEMA,
    CURVE_AXES,
    ExperimentError,
    ExperimentPlan,
    Strategy,
    TimedEvaluate,
    assemble,
    distribution,
    make_strategy,
    optimize_uploaded_dataset,
    optimizer_version,
    percentile,
    resource_curves,
    run_optimization_experiment,
    run_seed,
    run_strategy,
    summarize,
    without_timing,
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
SYNTHETIC_MODEL_HASH = "synthetic"  # what ``synthetic_evaluate`` reports as RunVersions.model_hash
ALL = (Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
RUNS_PER_CANDIDATE = 5  # capitals suite: 3 optimization + 2 validation rows, 1 trial
# authoritative per-run limits of the test contract (what a candidate reservation is made of)
LIMITS = {**FULL_CAPS, "maximum_model_calls": 4}
PER_RUN = {
    "model_calls": 4,
    "tokens": FULL_CAPS["maximum_tokens_per_example"],
    "wall_time_s": FULL_CAPS["maximum_wall_time_s"],
}


def limited_contract() -> TaskContract:
    return qa_contract(constraints=ConstraintLimits(**LIMITS))


def suite_and_contract():
    contract = limited_contract()
    suite, _ = contract_suite(contract, splits_for(contract), DATA)
    return suite, contract


def plan(budget: ExperimentBudget | None = None, **overrides: Any) -> ExperimentPlan:
    base: dict[str, Any] = {
        "model": MODEL,
        "expected_model_hash": SYNTHETIC_MODEL_HASH,
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

    def max_cost(self, model_hash: str, model_calls: int, tokens: int) -> float:
        # linear in calls and total tokens: the bound is exact for the limits it is given
        return model_calls * self.per_call + tokens * self.per_token


FLAT_COST = 2 * 0.01 + 300 * 0.0001  # LinearPricing of one FLAT run
PER_RUN_COST = LinearPricing().max_cost("", PER_RUN["model_calls"], PER_RUN["tokens"])


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


# -- one budget rule set for every strategy -----------------------------------------------------
def test_all_strategies_run_under_the_same_budget_rules_and_problem():
    budget = ExperimentBudget(max_candidate_evaluations=4, max_tokens=500_000)
    artifact = run_all(budget)
    assert artifact["schema"] == ARTIFACT_SCHEMA and artifact["synthetic"] is True
    runs = artifact["runs"]
    assert {(r["strategy"], r["seed"]) for r in runs} == {(s, d) for s in ALL for d in (0, 1, 2)}
    assert {json.dumps(r["budget"], sort_keys=True) for r in runs} == {
        json.dumps(budget.model_dump(mode="json"), sort_keys=True)
    }
    assert all(
        r["reservation_per_run"]
        == {"model_calls": None, "tokens": 20000.0, "wall_time_s": None, "cost": None}
        for r in runs
    )
    assert len({json.dumps(r["identity"]["protocol"], sort_keys=True) for r in runs}) == 1
    assert len({r["identity"]["problem_id"] for r in runs}) == 1
    assert artifact["fairness"]["unit"] == "candidate_evaluation"
    assert artifact["model_hashes"] == [SYNTHETIC_MODEL_HASH]


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


# -- hard caps by reservation -------------------------------------------------------------------
RESOURCES = {
    "model_calls": ("max_model_calls", FLAT["model_calls"], PER_RUN["model_calls"]),
    "tokens": ("max_tokens", FLAT["tokens"], PER_RUN["tokens"]),
    "wall_time_s": ("max_wall_time_s", FLAT["wall_time_s"], PER_RUN["wall_time_s"]),
    "cost": ("max_cost", FLAT_COST, PER_RUN_COST),
}


@pytest.mark.parametrize("resource", sorted(RESOURCES))
@pytest.mark.parametrize("fits", ["two_candidates", "nothing"])
def test_a_candidate_runs_only_if_its_complete_reservation_fits(resource, fits):
    field, measured, reserved = RESOURCES[resource]
    reservation = RUNS_PER_CANDIDATE * reserved  # one complete candidate, worst case
    actual = RUNS_PER_CANDIDATE * measured  # what a candidate really spends here
    # room for the 2nd candidate's full reservation after the 1st, but not for the 3rd's
    cap = reservation + 1.5 * actual if fits == "two_candidates" else 0.5 * reservation
    budget = ExperimentBudget(max_candidate_evaluations=100, **{field: cap})
    calls: list[str] = []
    pricing = LinearPricing() if resource == "cost" else None
    artifact = run_all(budget, evaluate=metered(calls=calls), pricing=pricing)
    total_runs = 0
    for r in artifact["runs"]:
        used = r["usage"]
        assert used[resource] <= cap  # final usage NEVER exceeds the cap
        assert all(len(c["runs"]) == RUNS_PER_CANDIDATE for c in r["candidates"])  # no partials
        total_runs += used["workflow_runs"]
        if fits == "nothing":  # not even the baseline's reservation fits: zero rows for everyone
            assert used["workflow_runs"] == 0 and r["champion"] is None
            assert r["stop_reason"] == resource
        elif r["strategy"] == "fixed":
            assert used["candidate_evaluations"] == 1 and r["stop_reason"] == "strategy_exhausted"
        else:  # same rule, same stopping point for random and ACO
            assert used["candidate_evaluations"] == 2 and r["stop_reason"] == resource
            assert used[resource] + reservation > cap  # the 3rd could not be paid for in full
    assert len(calls) == total_runs  # nothing ran that is not on the ledger


def _uneven(genome: Genome, task: ExecutionTask) -> Usage:
    k = len(genome) + int(task.id[1:])  # varies by workflow and row: strategies spend unevenly
    return {"model_calls": 1 + k % 4, "tokens": 97 * k, "wall_time_s": 0.25 * k}


def test_final_usage_never_exceeds_any_cap_with_uneven_measured_usage():
    budget = ExperimentBudget(
        max_candidate_evaluations=20,
        max_model_calls=60,
        max_tokens=120_000,
        max_wall_time_s=900.0,
        max_cost=30.0,
    )
    artifact = run_all(budget, evaluate=metered(_uneven), pricing=LinearPricing())
    caps = {
        "model_calls": budget.max_model_calls,
        "tokens": budget.max_tokens,
        "wall_time_s": budget.max_wall_time_s,
        "cost": budget.max_cost,
    }
    per_candidate = {
        "model_calls": 5 * 4,
        "tokens": 5 * 20_000,
        "wall_time_s": 5 * 120.0,
        "cost": 5 * PER_RUN_COST,
    }
    for r in artifact["runs"]:
        totals = dict.fromkeys(caps, 0.0)
        for cand in r["candidates"]:
            assert len(cand["runs"]) == RUNS_PER_CANDIDATE
            # it started only because its full reservation fitted on top of what was committed
            assert all(totals[k] + per_candidate[k] <= caps[k] for k in caps)
            for k in caps:
                totals[k] += sum(e[k] for e in cand["runs"])
        assert all(r["usage"][k] == pytest.approx(totals[k]) for k in caps)
        assert all(r["usage"][k] <= caps[k] for k in caps)
        if r["stop_reason"] in caps:  # it stopped because the next reservation did not fit
            k = r["stop_reason"]
            assert totals[k] + per_candidate[k] > caps[k]


def test_measured_usage_above_its_reservation_fails_closed():
    over = {"model_calls": 1, "tokens": 25_000, "wall_time_s": 1.0}  # > 20 000 tokens per run
    with pytest.raises(ReservationOverflow, match="tokens"):
        run_all(
            ExperimentBudget(max_candidate_evaluations=3, max_tokens=10_000_000),
            evaluate=metered(over),
        )
    # one run above its own limit is absorbed while the CANDIDATE stays inside its reservation
    one_over = metered(lambda g, t: over if t.id == "q1" else FLAT)
    artifact = run_all(
        ExperimentBudget(max_candidate_evaluations=2, max_tokens=10_000_000), evaluate=one_over
    )
    assert all(r["usage"]["candidate_evaluations"] >= 1 for r in artifact["runs"])


def test_a_ledger_settlement_above_the_reservation_raises():
    from experiments.budget_ledger import Reservation

    ledger = BudgetLedger(
        ExperimentBudget(max_candidate_evaluations=5, max_tokens=1000),
        per_run=Reservation(tokens=100),
    )
    assert ledger.reserve(2) is None
    entry = {
        "model_calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens": 150,
        "tool_calls": 0,
        "retries": 0,
        "wall_time_s": 0.0,
        "cost": None,
    }
    with pytest.raises(ReservationOverflow):
        ledger.settle([entry, entry], admitted=True)  # 300 measured > 200 reserved


def test_caps_without_an_authoritative_limit_to_reserve_against_fail_before_any_run():
    calls: list[str] = []
    contract = qa_contract()  # sets no maximum_model_calls
    suite, _ = contract_suite(contract, splits_for(contract), DATA)
    with pytest.raises(ExperimentBudgetError, match="model calls"):
        run_optimization_experiment(
            plan(ExperimentBudget(max_candidate_evaluations=3, max_model_calls=100)),
            suite,
            metered(calls=calls),
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
            synthetic=True,
        )
    assert calls == []


# -- cost ---------------------------------------------------------------------------------------
def test_a_cost_cap_without_a_pricing_policy_fails_closed_before_any_run():
    calls: list[str] = []
    budget = ExperimentBudget(max_candidate_evaluations=3, max_cost=1.0)
    with pytest.raises(ExperimentBudgetError, match="pricing policy"):
        run_all(budget, evaluate=metered(calls=calls))
    with pytest.raises(ExperimentBudgetError):
        BudgetLedger(budget)
    contract = limited_contract()

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


def test_a_cost_cap_needs_an_upper_bound_quote_from_the_pricing_policy():
    class NoQuote:
        identity = "no-quote/1"

        def cost(self, usage):
            return 0.01

    calls: list[str] = []
    with pytest.raises(ExperimentBudgetError, match="upper-bound quote"):
        run_all(
            ExperimentBudget(max_candidate_evaluations=3, max_cost=5.0),
            evaluate=metered(calls=calls),
            pricing=NoQuote(),
        )
    assert calls == []


def test_cost_is_priced_from_measured_usage_and_reported_on_the_curve():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=3), pricing=LinearPricing())
    assert artifact["identity"]["problem"]["pricing"] == LinearPricing.identity
    for r in artifact["runs"]:
        for point in r["curve"]:
            runs_so_far = point["evaluation"] * RUNS_PER_CANDIDATE
            assert point["cumulative_cost"] == pytest.approx(runs_so_far * FLAT_COST)
        n = r["usage"]["candidate_evaluations"]
        assert r["efficiency"]["cost_per_candidate"] == pytest.approx(
            RUNS_PER_CANDIDATE * FLAT_COST
        )
        assert r["usage"]["cost"] == pytest.approx(n * RUNS_PER_CANDIDATE * FLAT_COST)


def test_without_pricing_cost_is_unknown_not_zero():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=2))
    for r in artifact["runs"]:
        assert r["usage"]["cost"] is None and r["efficiency"]["cost_per_candidate"] is None
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


# -- the exact runtime model --------------------------------------------------------------------
def test_expected_model_hash_is_required():
    with pytest.raises(ValidationError):
        ExperimentPlan(
            model=MODEL, budget=ExperimentBudget(max_candidate_evaluations=1), seeds=(0,)
        )


def test_a_consistently_wrong_model_hash_fails_closed():
    calls: list[str] = []
    with pytest.raises(ExperimentError, match="bound to 'the-frozen-model'"):
        run_all(
            ExperimentBudget(max_candidate_evaluations=2),
            evaluate=metered(calls=calls),
            expected_model_hash="the-frozen-model",
        )
    assert len(calls) <= RUNS_PER_CANDIDATE  # stopped inside the very first candidate


def test_a_wrong_prompt_version_fails_closed():
    with pytest.raises(ExperimentError, match="prompts"):
        run_all(ExperimentBudget(max_candidate_evaluations=2), expected_prompt_version="mvp-3")


def test_assemble_rechecks_the_model_hash_of_every_record():
    suite, _ = suite_and_contract()
    p = plan(ExperimentBudget(max_candidate_evaluations=2), seeds=(0,))
    records = [
        run_strategy(
            p,
            s,
            0,
            suite,
            metered(),
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
        )
        for s in p.strategies
    ]
    kwargs = {"evaluator_version": SYNTHETIC_VERSION, "synthetic": True}
    assert assemble(p, suite, records, **kwargs)["model_hashes"] == [SYNTHETIC_MODEL_HASH]
    tampered = [dict(records[0], model_hashes=["someone-else"]), *records[1:]]
    with pytest.raises(ExperimentError, match="bound to"):
        assemble(p, suite, tampered, **kwargs)
    rebound = plan(
        ExperimentBudget(max_candidate_evaluations=2), seeds=(0,), expected_model_hash="x"
    )
    with pytest.raises(ExperimentError):
        assemble(rebound, suite, records, **kwargs)


# -- split isolation ----------------------------------------------------------------------------
class Spy(Optimizer):
    def __init__(self, inner: Optimizer, delay_s: float = 0.0) -> None:
        self.inner = inner
        self.name = inner.name
        self.version = optimizer_version(inner)
        self.delay_s = delay_s
        self.observed: list[str] = []
        self.proposals = 0

    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        time.sleep(self.delay_s)
        self.proposals += 1
        return self.inner.propose(k, context)

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
        assert r["champion"]["validation"]["runs"] > 0


def test_the_test_split_is_never_executed():
    calls: list[str] = []
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=6), evaluate=metered(calls=calls))
    test_rows = set(splits_for(limited_contract()).split(SplitRole.TEST).row_ids)
    assert test_rows and not test_rows & set(calls)
    assert artifact["test_runs"] == 0 and artifact["splits"]["test"] == len(test_rows)
    rows = {e["row_id"] for r in artifact["runs"] for c in r["candidates"] for e in c["runs"]}
    assert not rows & test_rows


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


# -- measurement: tokens, latency, overheads, curves --------------------------------------------
LATENCY = {"q1": 1.0, "q2": 2.0, "q3": 3.0, "q4": 4.0, "q5": 10.0}


def _by_row(genome: Genome, task: ExecutionTask) -> Usage:
    return {"model_calls": 1, "tokens": 100 * int(task.id[1:]), "wall_time_s": LATENCY[task.id]}


def test_token_and_latency_statistics_are_exact():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=3), evaluate=metered(_by_row))
    for r in artifact["runs"]:
        entries = [e for c in r["candidates"] for e in c["runs"]]
        lat = [e["wall_time_s"] for e in entries]
        assert r["latency_s"] == {
            "n": len(lat),
            "mean": statistics.fmean(lat),
            "p50": statistics.median(lat),
            "p95": percentile(lat, 0.95),
            "max": 10.0,
        }
        assert r["usage"]["tokens"] == sum(e["tokens"] for e in entries)
        assert r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] == r["usage"]["tokens"]
        n = r["usage"]["candidate_evaluations"]
        assert r["efficiency"]["tokens_per_candidate"] == r["usage"]["tokens"] / n
        assert r["efficiency"]["tokens_per_example"] == r["usage"]["tokens"] / len(entries)
        for c in r["candidates"]:  # validation rows q4, q5
            assert c["validation"]["latency_s"] == {
                "n": 2,
                "mean": 7.0,
                "p50": 7.0,
                "p95": 10.0,
                "max": 10.0,
            }
            assert c["validation"]["tokens"] == 900 and c["validation"]["tokens_per_example"] == 450
            assert c["usage"]["tokens"] == 1500 and c["usage"]["execution_s"] == 20.0
    assert distribution([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.95) == 5.0


def test_optimizer_overhead_is_measured_apart_from_model_latency():
    suite, _ = suite_and_contract()
    p = plan(ExperimentBudget(max_candidate_evaluations=4), seeds=(0,), strategies=(Strategy.ACO,))
    spy = Spy(make_strategy(Strategy.ACO, p), delay_s=0.05)
    record = run_strategy(
        p,
        Strategy.ACO,
        0,
        suite,
        metered(),
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        optimizer=spy,
    )
    timing = record["timing"]
    assert timing["optimizer_overhead_s"] >= 0.05 * spy.proposals
    # model/workflow latency is the runtime's measurement, untouched by the optimizer's time
    runs = record["usage"]["workflow_runs"]
    assert timing["execution_s"] == pytest.approx(FLAT["wall_time_s"] * runs)
    assert record["latency_s"]["max"] == FLAT["wall_time_s"]
    assert timing["e2e_wall_s"] >= timing["optimizer_overhead_s"]
    assert timing["evaluator_overhead_s"] is None  # a bare EvaluateFn cannot separate it


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_evaluator_overhead_is_separated_from_workflow_time():
    suite, _ = suite_and_contract()
    clock = FakeClock()
    produced: dict[str, EvaluatedRun] = {}

    def run_workflow(genome, task, trial, seed):
        clock.now += 2.0  # the workflow takes 2 s of wall clock
        run = metered()(genome, task, trial, seed)
        produced[run.run_id] = run
        return run.execution

    class Evaluator:
        def check_tasks(self, tasks):
            list(tasks)

        def evaluate_run(self, task, result):
            clock.now += 0.5  # judging takes 0.5 s
            return produced[result.run_id]

    tasks = (
        *suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK),
        *suite.tasks_for(SplitRole.VALIDATION, SplitUse.SELECTION),
    )
    evaluate = TimedEvaluate(run_workflow, Evaluator(), tasks, clock=clock)
    p = plan(
        ExperimentBudget(max_candidate_evaluations=1), seeds=(0,), strategies=(Strategy.FIXED,)
    )
    record = run_strategy(
        p,
        Strategy.FIXED,
        0,
        suite,
        evaluate,
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        clock=clock,
    )
    entries = record["candidates"][0]["runs"]
    assert all(e["timing"]["workflow_s"] == 2.0 for e in entries)
    assert all(e["timing"]["evaluator_s"] == 0.5 for e in entries)
    assert record["timing"]["evaluator_overhead_s"] == 0.5 * RUNS_PER_CANDIDATE
    assert record["timing"]["evaluate_calls_s"] == 2.5 * RUNS_PER_CANDIDATE
    assert record["timing"]["e2e_wall_s"] == 2.5 * RUNS_PER_CANDIDATE
    assert record["timing"]["optimizer_overhead_s"] == 0.0
    assert record["timing"]["execution_s"] == FLAT["wall_time_s"] * RUNS_PER_CANDIDATE


def test_learning_curves_exist_on_every_resource_axis():
    artifact = run_all(
        ExperimentBudget(max_candidate_evaluations=4),
        evaluate=metered(_uneven),
        pricing=LinearPricing(),
    )
    for r in artifact["runs"]:
        curves = resource_curves(r)
        assert set(curves) == set(CURVE_AXES)
        for points in curves.values():
            xs = [x for x, _ in points]
            assert xs == sorted(xs) and len(points) == r["usage"]["candidate_evaluations"]
            assert [y for _, y in points] == [p["best_so_far_score"] for p in r["curve"]]
        last = r["curve"][-1]
        assert last["cumulative_tokens"] == r["usage"]["tokens"]
        assert last["cumulative_model_calls"] == r["usage"]["model_calls"]
        assert last["cumulative_execution_s"] == pytest.approx(r["usage"]["wall_time_s"])
    unpriced = run_all(ExperimentBudget(max_candidate_evaluations=2))
    assert "cumulative_cost" not in resource_curves(unpriced["runs"][0])


# -- reproducibility ----------------------------------------------------------------------------
def test_same_identity_reproduces_identical_candidate_order_curves_and_artifact():
    budget = ExperimentBudget(max_candidate_evaluations=7, max_tokens=400_000)
    a = run_all(budget, evaluate=metered(_uneven), pricing=LinearPricing())
    b = run_all(budget, evaluate=metered(_uneven), pricing=LinearPricing())
    assert json.dumps(without_timing(a), sort_keys=True) == json.dumps(
        without_timing(b), sort_keys=True
    )
    for ra, rb in zip(a["runs"], b["runs"], strict=True):
        assert ra["run_id"] == rb["run_id"]
        assert ra["evaluated_genome_hashes"] == rb["evaluated_genome_hashes"]
        assert without_timing(ra["curve"]) == without_timing(rb["curve"])


def test_identity_covers_data_manifest_split_contract_grammar_evaluator_model_prompt_budget():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=2), dataset_manifest_hash="m1")
    problem = artifact["identity"]["problem"]
    contract = limited_contract()
    assert problem["dataset_content_hash"] == contract.dataset.content_hash
    assert problem["dataset_hash"] == contract.dataset.identity_hash
    assert problem["dataset_manifest_hash"] == "m1"
    assert problem["splits_hash"] == splits_for(contract).identity_hash
    assert problem["task_contract_hash"] == contract.contract_hash
    assert problem["grammar_version"].startswith("grammar/")
    assert problem["evaluator_run_version"] == SYNTHETIC_VERSION
    assert problem["evaluation_hash"] == contract.evaluation.identity_hash
    assert problem["model"] == MODEL.model_dump(mode="json")
    assert problem["model_hash"] == SYNTHETIC_MODEL_HASH
    for r in artifact["runs"]:
        ident = r["identity"]
        assert ident["strategy"] == r["strategy"] and ident["seed"] == r["seed"]
        assert ident["budget_hash"] == ExperimentBudget(max_candidate_evaluations=2).identity_hash
    base = {r["run_id"] for r in artifact["runs"]}
    for change in (
        {"budget": ExperimentBudget(max_candidate_evaluations=3)},
        {"model": ModelConfiguration(provider="test-double", model="other")},
        {"dataset_manifest_hash": "m2"},
    ):
        budget = change.pop("budget", ExperimentBudget(max_candidate_evaluations=2))
        other = run_all(budget, **{"dataset_manifest_hash": "m1", **change})
        assert base.isdisjoint(r["run_id"] for r in other["runs"])


# -- seeds and aggregates -----------------------------------------------------------------------
def test_each_seed_is_recorded_and_persisted_independently(tmp_path):
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=4))
    for strategy, runs in by_strategy(artifact).items():
        assert sorted(r["seed"] for r in runs) == [0, 1, 2]
        assert len({r["run_id"] for r in runs}) == 3
        per_seed = artifact["summary"]["by_strategy"][strategy]["per_seed"]
        assert [p["seed"] for p in per_seed] == [0, 1, 2]
    random_orders = {tuple(r["evaluated_genome_hashes"]) for r in by_strategy(artifact)["random"]}
    assert len(random_orders) == 3
    paths = write_experiment(artifact, tmp_path)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == artifact
    for r in artifact["runs"]:
        path = tmp_path / "runs" / r["strategy"] / f"seed-{r['seed']}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["experiment_id"] == artifact["experiment_id"]
        assert {k: v for k, v in saved.items() if k != "experiment_id"} == r


def _fake_run(strategy: str, seed: int, score: float | None, curve: list[float]) -> dict:
    champ = (
        None
        if score is None
        else {
            "genome_hash": f"g{seed}",
            "feasible": True,
            "validation": {"score_mean": score, "pass_rate": score / 2},
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
            "prompt_tokens": 60 * (seed + 1),
            "completion_tokens": 40 * (seed + 1),
            "tokens": 100 * (seed + 1),
            "cost": None,
        },
        "efficiency": {"tokens_per_candidate": 50.0},
        "latency_s": {"mean": 1.0 + seed, "p95": 2.0 + seed},
        "timing": {
            "execution_s": 3.0,
            "e2e_wall_s": 4.0,
            "optimizer_overhead_s": 0.1,
            "evaluator_overhead_s": None,
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
    f1 = random["metrics"]["champion_validation_score"]
    assert f1["n"] == 3 and f1["mean"] == pytest.approx(0.5) and f1["median"] == 0.4
    assert f1["std"] == pytest.approx(statistics.stdev([0.2, 0.4, 0.9]))
    assert (f1["min"], f1["max"]) == (0.2, 0.9)
    calls = random["metrics"]["model_calls"]
    assert (calls["mean"], calls["median"], calls["min"], calls["max"]) == (20, 20, 10, 30)
    p95 = random["metrics"]["p95_latency_s"]
    assert (p95["mean"], p95["min"], p95["max"]) == (3.0, 2.0, 4.0)
    assert random["metrics"]["cost"]["n"] == 0 and random["metrics"]["cost"]["mean"] is None
    assert random["metrics"]["evaluator_overhead_s"]["n"] == 0
    assert [p["champion_validation_score"] for p in random["per_seed"]] == [0.2, 0.4, 0.9]
    curve = random["best_so_far_curve"]
    assert [c["best_so_far_score"]["n"] for c in curve] == [3, 2, 1]
    assert curve[0]["best_so_far_score"]["mean"] == pytest.approx((0.1 + 0.4 + 0.9) / 3)
    assert curve[2]["best_so_far_score"]["std"] is None
    fixed = summary["by_strategy"]["fixed"]["metrics"]["champion_validation_score"]
    assert fixed["n"] == 1 and fixed["mean"] == 0.3
    assert summary["highest_mean_champion_validation_score"] == ["random"]


def test_artifact_summary_matches_its_own_per_seed_records():
    artifact = run_all(ExperimentBudget(max_candidate_evaluations=5))
    for strategy, runs in by_strategy(artifact).items():
        values = [r["champion"]["validation"]["score_mean"] for r in runs]
        got = artifact["summary"]["by_strategy"][strategy]["metrics"]["champion_validation_score"]
        assert got["mean"] == pytest.approx(statistics.fmean(values))
        assert got["std"] == pytest.approx(statistics.stdev(values))
        assert got["median"] == pytest.approx(statistics.median(values))
        assert (got["min"], got["max"]) == (min(values), max(values))


# -- ACO maths is unchanged ---------------------------------------------------------------------
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
    opt = FixedBaseline()
    ctx = SearchContext(contract=cls, checker=checker, seed=0)
    assert opt.propose(4, ctx) == [baseline_workflow(cls, checker)] and opt.propose(4, ctx) == []
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

    assert len(set(order(0))) == 30
    assert order(0) == order(0) and order(0) != order(1)


def test_plan_rejects_ambiguous_seed_or_strategy_lists():
    with pytest.raises(ValueError):
        plan(seeds=(1, 1))
    with pytest.raises(ValueError):
        plan(strategies=(Strategy.ACO, Strategy.ACO))
    with pytest.raises(ValueError):
        plan(seeds=())


# -- the real pipeline: uploaded bytes -> contract_suite -> 3 strategies -> dispatcher -> curves
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


def _deterministic(obj: Any) -> Any:
    """The real runtime measures wall-clock time; everything else must reproduce exactly."""
    if isinstance(obj, dict):
        return {
            k: _deterministic(v)
            for k, v in obj.items()
            if "wall_time" not in k and "execution_s" not in k and "latency" not in k
        }
    if isinstance(obj, list):
        return [_deterministic(v) for v in obj]
    return obj


def test_uploaded_dataset_runs_fixed_random_and_aco_through_the_real_pipeline():
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner
    from tests.test_contract_runtime import PassageModel

    contract = uploaded_contract()
    splits = uploaded_splits(contract)
    test_rows = set(splits.split(SplitRole.TEST).row_ids)
    p = plan(
        ExperimentBudget(max_candidate_evaluations=4, max_tokens=200_000),
        seeds=(0, 1),
        expected_model_hash=PassageModel.model_hash,
        expected_prompt_version="mvp-3",
    )

    def experiment() -> tuple[dict, list[str], PassageModel]:
        model = PassageModel()
        runner = WorkflowRunner(model=model, benchmark_hash="uploaded-inline")
        executed: list[str] = []

        def run_workflow(genome, task, trial, seed):
            executed.append(task.id)
            return runner.run_sync(genome, task, trial=trial, seed=seed)

        artifact = optimize_uploaded_dataset(
            contract,
            splits,
            UPLOAD,
            run_workflow,
            p,
            checker=runner.checker,
            synthetic=True,
            workers=3,
        )
        return artifact, executed, model

    artifact, executed, model = experiment()

    assert set(executed) == {"c1", "c2", "c3", "c4", "c5"} and not test_rows & set(executed)
    assert artifact["test_runs"] == 0
    assert model.requests and all(contract.instructions in r.input_text for r in model.requests)
    problem = artifact["identity"]["problem"]
    assert problem["evaluator_kind"] == "exact_match"
    assert problem["evaluator_run_version"].startswith(contract.evaluation.evaluator_version + "+")
    assert problem["dataset_content_hash"] == sha256_bytes(UPLOAD)
    assert artifact["model_hashes"] == [PassageModel.model_hash]
    assert artifact["run_versions"][0]["prompt_template_version"] == "mvp-3"

    runs = by_strategy(artifact)
    assert set(runs) == {"fixed", "random", "aco"}
    for strategy, mine in runs.items():
        for r in mine:
            used = r["usage"]["candidate_evaluations"]
            assert used == (1 if strategy == "fixed" else 4)
            assert len(r["curve"]) == used and r["champion"] is not None
            assert r["timing"]["evaluator_overhead_s"] is not None  # TimedEvaluate separates it
            for point in r["curve"]:
                assert point["cumulative_model_calls"] > 0 and point["cumulative_tokens"] > 0
            assert set(resource_curves(r)) == set(CURVE_AXES) - {"cumulative_cost"}
    for r in runs["fixed"]:
        assert [s["kind"] for s in r["champion"]["genome"]["stages"]] == ["DIRECT"]

    again, _, _ = experiment()
    for ra, rb in zip(artifact["runs"], again["runs"], strict=True):
        assert ra["run_id"] == rb["run_id"]
        assert ra["evaluated_genome_hashes"] == rb["evaluated_genome_hashes"]
        assert _deterministic(without_timing(ra["curve"])) == _deterministic(
            without_timing(rb["curve"])
        )
