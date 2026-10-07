"""Validation-based champion selection + held-out promotion gate (Issue #25).

    COMPLETED experiment -> ONE validation-selected challenger -> held-out gate -> promote/reject

Every model here is a TEST DOUBLE (``Rigged``): a run passes iff a rule over (genome, row) says
so, with injected deterministic usage. Rules are written per test to make one specific workflow
win validation, pass or fail the held-out row, or regress - nothing here is a benchmark result,
and no test assumes that search beats the fixed baseline.

Capitals dataset splits (``splits_for``): optimization q1-q3, validation q4-q5, test q6.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from api.product import ProductAPI
from core.constraints import ConstraintChecker, ConstraintLimits
from core.dataset import SplitRole, SplitUse
from core.experiment import ModelConfiguration
from core.genome import Genome
from core.objective import CandidateMeasurements, Metric, ObjectiveMode, ObjectiveSpec
from core.results import EvaluatedRun, FailureInfo, FailureKind, Verdict
from experiments.budget_ledger import ExperimentBudget
from experiments.contract_run import contract_suite
from experiments.jobs import (
    ExperimentJobDefinition,
    ExperimentJobs,
    HeldoutBinding,
    JobWorker,
    NotResumable,
)
from experiments.optimization_experiment import (
    MODEL_ATTEMPTS,
    Strategy,
    run_strategy,
    without_timing,
)
from experiments.promotion import (
    CandidateStatus,
    ChampionNotFound,
    ChampionPromotions,
    Decision,
    EvidenceMismatch,
    IncompatibleIncumbent,
    NotPromotable,
    PromotionAmbiguous,
    PromotionFailed,
    ReasonCode,
    gate,
)
from experiments.synthetic import SYNTHETIC_VERSION
from store.champions import (
    NewChampion,
    PromotionReason,
    PromotionState,
    SQLiteChampionStore,
    StaleIncumbent,
)
from store.datasets import Conflict
from store.jobs import JobState, SQLiteJobStore
from tests.test_contract_runtime import DATA, qa_contract, splits_for
from tests.test_durable_jobs import LEASE, Backend, Clock, Crash, Runtime
from tests.test_optimization_experiment import ALL, FLAT, LIMITS, MODEL, metered, plan

ROOT = Path(__file__).resolve().parent.parent
TEST_ROW = "q6"
VALIDATION_ROWS = ("q4", "q5")
EVERYTHING = lambda g, row: True  # noqa: E731


# -- test doubles -------------------------------------------------------------------------------
class Rigged(Backend):
    """TEST DOUBLE model: a run PASSES (fitness 1) iff ``passes(genome_hash, row)``; its measured
    wall time is ``latency(genome_hash, row)``. Deterministic; records every call it receives."""

    def __init__(
        self,
        passes: Callable[[str, str], bool],
        latency: Callable[[str, str], float] | None = None,
        hook=None,
        outage: Callable[[str, str], bool] | None = None,
    ) -> None:
        super().__init__(FLAT, hook)
        self.passes = passes
        self.latency = latency or (lambda g, row: FLAT["wall_time_s"])
        self.outage = outage
        base = metered(lambda g, t: {**FLAT, "wall_time_s": self.latency(g.genome_hash, t.id)})

        def evaluate(genome, task, trial, seed) -> EvaluatedRun:
            run = base(genome, task, trial, seed)
            ok = self.passes(genome.genome_hash, task.id)
            evaluation = run.evaluation.model_copy(
                update={"verdict": Verdict.PASS if ok else Verdict.FAIL, "fitness": float(ok)}
            )
            execution = run.execution
            if self.outage is not None and self.outage(genome.genome_hash, task.id):
                execution = execution.model_copy(
                    update={"failure": FailureInfo(kind=FailureKind.MODEL_ERROR, message="down")}
                )
            return EvaluatedRun(execution=execution, evaluation=evaluation)

        self._evaluate = evaluate

    def rows(self) -> list[str]:
        return [k[1] for k in self.calls]

    def on_test(self) -> list[tuple]:
        return [k for k in self.calls if k[1] == TEST_ROW]


class GateRuntime(Runtime):
    """TEST DOUBLE runtime: the durable-jobs runtime plus the held-out (test-row) binding."""

    def __init__(self, backend: Backend, proof=None, on_heldout=None) -> None:
        super().__init__(backend, None, proof)
        self.on_heldout = on_heldout
        self.heldout_binds = 0

    def bind_heldout(self, definition: ExperimentJobDefinition) -> HeldoutBinding:
        self.heldout_binds += 1
        if self.on_heldout is not None:
            self.on_heldout(definition)
        suite, _ = contract_suite(definition.contract, definition.splits, DATA)
        tasks = suite.tasks_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)
        return HeldoutBinding(tasks, self.backend, SYNTHETIC_VERSION, self.proof)


class World:
    """One process: a job store + worker (#24) and a champion store + promotion service."""

    def __init__(
        self,
        root: Path,
        backend: Backend,
        *,
        clock: Clock | None = None,
        name: str = "p",
        champion_store=SQLiteChampionStore,
        proof=None,
        on_heldout=None,
        ids: list[str] | None = None,
    ) -> None:
        self.backend = backend
        self.clock = clock or Clock()
        self.runtime = GateRuntime(backend, proof, on_heldout)
        queue = list(ids or [])
        counter = [0]

        def next_id() -> str:
            counter[0] += 1
            return queue.pop(0) if queue else f"j-{name}-{counter[0]}"

        self.jobs = ExperimentJobs(
            SQLiteJobStore(root / "jobs.sqlite3"), self.runtime, clock=self.clock, _ids=next_id
        )
        self.worker = JobWorker(
            self.jobs.store,
            self.runtime,
            worker_id=name,
            lease_s=LEASE,
            clock=self.clock,
            heartbeat=False,
        )
        self.store = champion_store(root / "champions.sqlite3")
        self.promotions = ChampionPromotions(
            self.store,
            self.jobs.store,
            self.runtime,
            clock=self.clock,
            owner=name,
            lease_s=LEASE,
            heartbeat=False,
        )

    def experiment(self, d: ExperimentJobDefinition) -> str:
        job_id = self.jobs.create(d).job_id
        self.worker.run_until_idle(job_id)
        assert self.jobs.get(job_id).state is JobState.COMPLETED
        return job_id

    def promote(self, job_id: str, lineage: str | None = None):
        return self.promotions.promote(job_id, lineage)


def definition(
    strategies=ALL,
    seeds=(0,),
    *,
    budget: int = 4,
    trials: int = 1,
    constraints: ConstraintLimits | None = None,
    objective: ObjectiveSpec | None = None,
    model: ModelConfiguration = MODEL,
) -> ExperimentJobDefinition:
    extra = {"objective": objective} if objective is not None else {}
    contract = qa_contract(constraints=constraints or ConstraintLimits(**LIMITS), **extra)
    return ExperimentJobDefinition(
        contract=contract,
        splits=splits_for(contract),
        plan=plan(
            ExperimentBudget(max_candidate_evaluations=budget),
            seeds=seeds,
            strategies=strategies,
            trials=trials,
            model=model,
        ),
        synthetic=True,
    )


def evaluated(d: ExperimentJobDefinition, strategy: Strategy, seed: int = 0) -> list[str]:
    """Genome hashes ``strategy`` evaluates, in order, under a neutral backend. Fixed and random
    proposals never depend on feedback; ACO's first round (``batch_size``) does not either."""
    suite, _ = contract_suite(d.contract, d.splits, DATA)
    record = run_strategy(
        d.plan,
        strategy,
        seed,
        suite,
        metered(),
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
    )
    return record["evaluated_genome_hashes"]


def winners(d: ExperimentJobDefinition) -> dict[str, str]:
    """One workflow per strategy that ONLY that strategy reaches at its earliest evaluation."""
    d = d.model_copy(update={"plan": d.plan.model_copy(update={"strategies": ALL})})
    fixed = evaluated(d, Strategy.FIXED)[0]
    rand = evaluated(d, Strategy.RANDOM)
    aco_first_round = evaluated(d, Strategy.ACO)[: d.plan.batch_size]
    random_only = rand[0]
    aco_only = next(g for g in aco_first_round if g not in rand and g != fixed)
    # preconditions of the scenario (fail loudly if the test double's search ever changes)
    assert random_only != fixed and random_only not in aco_first_round
    return {"fixed": fixed, "random": random_only, "aco": aco_only}


def record(view) -> dict[str, Any]:
    assert view.record is not None
    return view.record


def statuses(rec: dict[str, Any]) -> dict[str, str]:
    return {c["candidate_id"]: c["status"] for c in rec["selection"]["candidates"]}


# -- lifecycle: whichever strategy wins validation may become champion --------------------------
@pytest.mark.parametrize("strategy", ["fixed", "random", "aco"])
def test_fixed_random_or_aco_can_become_champion(tmp_path, strategy):
    d = definition()
    win = winners(d)[strategy]
    w = World(tmp_path, Rigged(lambda g, row: g == win))
    job = w.experiment(d)

    view = w.promote(job)

    rec = record(view)
    assert view.state is PromotionState.DECIDED and view.decision is Decision.PROMOTED
    assert rec["challenger"]["strategy"] == strategy and rec["challenger"]["genome_hash"] == win
    assert rec["challenger"]["status"] == CandidateStatus.CHAMPION
    assert rec["reason_codes"] == [
        ReasonCode.HELDOUT_CONSTRAINTS_PASSED,
        ReasonCode.INITIAL_CHAMPION,
    ]
    champion = w.promotions.current("capitals.capitals")
    assert champion.version == 1 and champion.record["genome_hash"] == win
    assert champion.record["provenance"]["strategy"] == strategy
    assert champion.record["provenance"]["job_id"] == job
    # lifecycle: CANDIDATE -> VALIDATED -> CHAMPION; every other candidate is CANDIDATE
    selection_status = statuses(rec)
    assert selection_status[rec["challenger"]["candidate_id"]] == CandidateStatus.VALIDATED
    assert sorted(set(selection_status.values())) == ["CANDIDATE", "VALIDATED"]


def test_champion_identity_pins_workflow_dataset_contract_model_and_provenance(tmp_path):
    d = definition(strategies=(Strategy.FIXED,))
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(d)
    w.promote(job)

    rec = w.promotions.current("capitals.capitals").record
    compat = rec["compatibility"]
    artifact = w.jobs.artifact(job).artifact
    problem = artifact["identity"]["problem"]
    assert Genome.from_canonical(rec["genome"]).genome_hash == rec["genome_hash"]
    assert compat["dataset_version"] == d.contract.dataset.dataset_version
    assert compat["dataset_content_hash"] == d.contract.dataset.content_hash
    assert compat["task_contract_hash"] == d.contract.contract_hash
    assert compat["grammar_version"] == problem["grammar_version"]
    assert compat["evaluator_version"] == SYNTHETIC_VERSION
    assert compat["model_hash"] == d.plan.expected_model_hash == "synthetic"
    assert compat["model"] == d.plan.model.model_dump(mode="json")
    assert compat["prompt_template_version"] == d.plan.expected_prompt_version
    assert compat["objective"] == d.contract.objective.model_dump(mode="json")
    assert compat["constraints"] == d.contract.constraints.model_dump(mode="json")
    assert compat["run_versions"] == artifact["run_versions"]
    assert compat["test_row_ids"] == [TEST_ROW]
    assert rec["provenance"] | {} == {
        **rec["provenance"],
        "job_id": job,
        "experiment_id": artifact["experiment_id"],
        "run_id": artifact["runs"][0]["run_id"],
        "strategy": "fixed",
        "seed": 0,
    }


# -- validation selection -----------------------------------------------------------------------
def test_an_infeasible_candidate_never_becomes_the_challenger(tmp_path):
    """The fixed workflow is the best on validation quality but breaks a hard latency limit on
    validation: it is REJECTED, and a feasible (worse-scoring) candidate is the challenger."""
    d = definition(constraints=ConstraintLimits(**LIMITS, maximum_mean_latency_s=10.0))
    fixed = winners(d)["fixed"]
    rigged = Rigged(
        lambda g, row: g == fixed,
        latency=lambda g, row: 50.0 if g == fixed and row in VALIDATION_ROWS else 1.5,
    )
    w = World(tmp_path, rigged)
    view = w.promote(w.experiment(d))

    rec = record(view)
    by_genome = {c["genome_hash"]: c for c in rec["selection"]["candidates"]}
    assert by_genome[fixed]["status"] == CandidateStatus.REJECTED
    assert by_genome[fixed]["reason_codes"] == [ReasonCode.VALIDATION_INFEASIBLE]
    assert by_genome[fixed]["rank"]["feasible"] is False
    assert by_genome[fixed]["validation"]["pass_rate"] == 1.0  # better score, still rejected
    assert rec["challenger"]["genome_hash"] != fixed
    assert by_genome[rec["challenger"]["genome_hash"]]["rank"]["feasible"] is True
    assert fixed not in {k[0] for k in rigged.on_test()}  # it never reached the test split


def test_with_no_feasible_candidate_there_is_no_challenger_and_test_stays_closed(tmp_path):
    d = definition(constraints=ConstraintLimits(**LIMITS, minimum_quality=0.5))
    rigged = Rigged(lambda g, row: False)
    w = World(tmp_path, rigged)
    job = w.experiment(d)

    view = w.promote(job)

    rec = record(view)
    assert view.decision is Decision.REJECTED and view.test_opened_at is None
    assert rec["reason_codes"] == [ReasonCode.NO_FEASIBLE_CHALLENGER]
    assert rec["challenger"] is None and rec["heldout"] is None and rec["test_opened"] is False
    assert {c["status"] for c in rec["selection"]["candidates"]} == {CandidateStatus.REJECTED}
    assert rigged.on_test() == [] and w.runtime.heldout_binds == 0
    with pytest.raises(ChampionNotFound):
        w.promotions.current("capitals.capitals")
    assert w.promote(job) == view  # final: promoting again changes nothing
    w.promotions.verify(view.promotion_id)


def test_validation_alone_chooses_the_challenger(tmp_path):
    """The random workflow wins validation but fails the test row; the fixed workflow fails
    validation but would pass the test row. The challenger is the validation winner, and the
    workflow that is better on test is never even run on it."""
    d = definition(strategies=(Strategy.FIXED, Strategy.RANDOM))
    win = winners(d)
    val_best, test_best = win["random"], win["fixed"]
    rigged = Rigged(
        lambda g, row: (g == val_best) if row != TEST_ROW else (g == test_best),
    )
    w = World(tmp_path, rigged)
    view = w.promote(w.experiment(d))

    rec = record(view)
    assert rec["challenger"]["genome_hash"] == val_best
    assert rec["selection"]["ranking"][0] == rec["selection"]["challenger"]
    assert rec["heldout"]["challenger"]["pass_rate"] == 0.0  # disappointing on test ...
    assert {k[0] for k in rigged.on_test()} == {val_best}  # ... and nobody else is tried
    assert view.decision is Decision.PROMOTED  # no constraint forbids it, no incumbent


def test_the_test_split_is_opened_durably_before_any_test_target_is_bound(tmp_path):
    seen: list[dict[str, Any]] = []
    box: dict[str, World] = {}

    def on_heldout(definition: ExperimentJobDefinition) -> None:
        w = box["w"]
        (row,) = w.store.promotions()
        seen.append(
            {
                "state": row.state,
                "test_opened": row.test_opened_at is not None,
                "challenger": json.loads(row.selection_json)["selection"]["challenger"],
                "test_calls": len(w.backend.on_test()),
            }
        )

    d = definition()
    rigged = Rigged(EVERYTHING)
    w = box["w"] = World(tmp_path, rigged, on_heldout=on_heldout)
    job = w.experiment(d)
    assert rigged.on_test() == []  # the experiment never ran a test row
    view = w.promote(job)

    assert seen == [
        {
            "state": PromotionState.OPEN,
            "test_opened": True,
            "challenger": view.challenger["candidate_id"],
            "test_calls": 0,
        }
    ]
    assert record(view)["selection"]["challenger"] == view.challenger["candidate_id"]


def test_test_rows_never_reach_an_optimizer_or_the_experiment(tmp_path, monkeypatch):
    from optimizers.aco_mmas import MMASACO
    from optimizers.random_search import RandomSearch

    observed: list[str] = []
    for cls in (MMASACO, RandomSearch):
        original = cls.observe

        def spy(self, results, _original=original):
            observed.extend(r.execution.task_id for r in results)
            return _original(self, results)

        monkeypatch.setattr(cls, "observe", spy)

    d = definition()
    rigged = Rigged(EVERYTHING)
    w = World(tmp_path, rigged)
    job = w.experiment(d)
    store = w.jobs.store
    before = {
        "job": store.get_job(job),
        "checkpoints": [
            store.checkpoints(job, u.strategy, u.seed) for u in store.get_job(job).units
        ],
        "attempts": [store.attempts(job, u.strategy, u.seed) for u in store.get_job(job).units],
    }
    assert w.jobs.artifact(job).artifact["test_runs"] == 0
    assert TEST_ROW not in rigged.rows()

    w.promote(job)

    assert observed and set(observed) <= {"q1", "q2", "q3"}  # optimization rows only
    assert TEST_ROW in rigged.rows()  # the gate ran ...
    job_row = store.get_job(job)
    after = {
        "job": job_row,
        "checkpoints": [store.checkpoints(job, u.strategy, u.seed) for u in job_row.units],
        "attempts": [store.attempts(job, u.strategy, u.seed) for u in job_row.units],
    }
    assert after == before  # ... but nothing of it was fed back into the experiment
    assert all(a.task_id != TEST_ROW for unit in after["attempts"] for a in unit)
    with pytest.raises(NotResumable):
        w.jobs.resume(job)


# -- the held-out gate --------------------------------------------------------------------------
def two_experiments(tmp_path, rule, latency=None, objective=None, constraints=None):
    """Experiment 1 (fixed only) makes the fixed workflow champion; experiment 2 (random only)
    brings the random workflow as challenger against it."""
    d1 = definition(strategies=(Strategy.FIXED,), objective=objective, constraints=constraints)
    d2 = definition(strategies=(Strategy.RANDOM,), objective=objective, constraints=constraints)
    win = winners(definition())
    rigged = Rigged(
        lambda g, row: rule(win, g, row),
        latency=(lambda g, r: latency(win, g, r)) if latency else None,
    )
    w = World(tmp_path, rigged)
    first = w.promote(w.experiment(d1))
    assert first.decision is Decision.PROMOTED
    return w, w.experiment(d2), win, rigged


def test_a_challenger_that_regresses_against_the_incumbent_is_rejected(tmp_path):
    def rule(win, g, row):  # the challenger wins validation, the incumbent is better on test
        if row == TEST_ROW:
            return g == win["fixed"]
        return True

    w, job, win, rigged = two_experiments(tmp_path, rule)
    calls = len(rigged.on_test())
    view = w.promote(job)

    rec = record(view)
    assert view.decision is Decision.REJECTED
    assert ReasonCode.REGRESSION in rec["reason_codes"]
    assert rec["comparison"]["relation"] == "worse"
    assert rec["challenger"]["genome_hash"] == win["random"]
    assert rec["incumbent"]["genome_hash"] == win["fixed"]
    assert rec["challenger"]["status"] == CandidateStatus.REJECTED
    champion = w.promotions.current("capitals.capitals")
    assert champion.version == 1 and champion.record["genome_hash"] == win["fixed"]
    # the incumbent was re-run on exactly the challenger's held-out rows / trials / seeds
    new = rigged.on_test()[calls:]
    by_genome = {g: sorted(k[1:] for k in new if k[0] == g) for g in (win["random"], win["fixed"])}
    assert by_genome[win["random"]] == by_genome[win["fixed"]] and by_genome[win["fixed"]]
    assert {k[0] for k in new} == {win["random"], win["fixed"]}  # only these two touch test


def test_no_regression_promotes_and_equal_is_not_a_regression(tmp_path):
    w, job, win, _ = two_experiments(tmp_path, lambda win, g, row: True)
    view = w.promote(job)

    rec = record(view)
    assert view.decision is Decision.PROMOTED
    assert rec["comparison"]["relation"] == "equal"
    assert rec["reason_codes"] == [
        ReasonCode.HELDOUT_CONSTRAINTS_PASSED,
        ReasonCode.NO_REGRESSION,
        ReasonCode.EQUAL_TO_INCUMBENT,
    ]
    history = w.promotions.history("capitals.capitals").champions
    assert [c.version for c in history] == [1, 2]
    assert [c.record["genome_hash"] for c in history] == [win["fixed"], win["random"]]
    assert history[1].record["previous_champion_id"] == history[0].champion_id


@pytest.mark.parametrize(
    "challenger_latency, decision, relation",
    [
        (1.0, Decision.PROMOTED, "better"),
        (2.0, Decision.PROMOTED, "equal"),
        (3.0, Decision.REJECTED, "worse"),
    ],
)
def test_minimize_objectives_promote_only_at_or_below_the_incumbent(
    tmp_path, challenger_latency, decision, relation
):
    """minimize_latency: lower held-out latency is better; equal is no regression."""

    def latency(win, g, row):
        if row != TEST_ROW:
            return 1.5
        return challenger_latency if g == win["random"] else 2.0

    w, job, win, _ = two_experiments(
        tmp_path,
        lambda win, g, row: True,
        latency=latency,
        objective=ObjectiveSpec(mode=ObjectiveMode.MINIMIZE_LATENCY),
        constraints=ConstraintLimits(**LIMITS, minimum_quality=0.5),
    )
    view = w.promote(job)

    rec = record(view)
    assert view.decision is decision and rec["comparison"]["relation"] == relation
    assert rec["heldout"]["challenger"]["measurements"]["mean_latency_s"] == challenger_latency


def _rank(contract, **measured) -> dict[str, Any]:
    r = contract.rank(CandidateMeasurements(**measured))
    return {"feasible": r.feasible, "rank_key": list(r.sort_key)}


@pytest.mark.parametrize(
    "objective, incumbent, better, equal, worse",
    [
        (
            ObjectiveSpec(),
            {"quality": 0.6},
            {"quality": 0.7},
            {"quality": 0.6},
            {"quality": 0.5},
        ),
        (
            ObjectiveSpec(mode=ObjectiveMode.MINIMIZE_LATENCY),
            {"quality": 0.8, "mean_latency_s": 2.0},
            {"quality": 0.8, "mean_latency_s": 1.0},
            {"quality": 0.8, "mean_latency_s": 2.0},
            {"quality": 0.8, "mean_latency_s": 3.0},
        ),
        (  # minimize: the contract ranks equal latency by quality next
            ObjectiveSpec(mode=ObjectiveMode.MINIMIZE_LATENCY),
            {"quality": 0.8, "mean_latency_s": 2.0},
            {"quality": 0.9, "mean_latency_s": 2.0},
            {"quality": 0.8, "mean_latency_s": 2.0},
            {"quality": 0.7, "mean_latency_s": 2.0},
        ),
        (
            ObjectiveSpec(
                mode=ObjectiveMode.BALANCED,
                weights={Metric.QUALITY: 0.5, Metric.LATENCY: 0.5},
                scales={Metric.LATENCY: 10.0},
            ),
            {"quality": 0.8, "mean_latency_s": 4.0},
            {"quality": 0.8, "mean_latency_s": 2.0},
            {"quality": 0.8, "mean_latency_s": 4.0},
            {"quality": 0.6, "mean_latency_s": 4.0},
        ),
    ],
    ids=["maximize_quality", "minimize_latency", "minimize_latency_tie", "balanced"],
)
def test_the_gate_uses_the_contracts_own_ranking(objective, incumbent, better, equal, worse):
    contract = qa_contract(constraints=ConstraintLimits(minimum_quality=0.5), objective=objective)
    inc = _rank(contract, **incumbent)
    for measured, want in ((better, "better"), (equal, "equal"), (worse, "worse")):
        decision, reasons, comparison = gate(_rank(contract, **measured), inc)
        assert comparison["relation"] == want
        assert decision is (Decision.REJECTED if want == "worse" else Decision.PROMOTED)
        assert (ReasonCode.REGRESSION in reasons) is (want == "worse")
    # a feasible challenger always beats an incumbent that breaks a hard limit on held-out
    broken = _rank(contract, **{**incumbent, "quality": 0.1})
    decision, reasons, _ = gate(_rank(contract, **worse), broken)
    assert decision is Decision.PROMOTED and ReasonCode.INCUMBENT_HELDOUT_INFEASIBLE in reasons
    # ... and an infeasible challenger never passes, with or without an incumbent
    for other in (None, broken):
        decision, reasons, _ = gate(broken, other)
        assert (decision, reasons) == (
            Decision.REJECTED,
            [ReasonCode.HELDOUT_CONSTRAINT_VIOLATION],
        )


def test_an_initial_champion_still_needs_the_held_out_constraints(tmp_path):
    d = definition(
        strategies=(Strategy.FIXED,), constraints=ConstraintLimits(**LIMITS, minimum_quality=0.5)
    )
    rigged = Rigged(lambda g, row: row != TEST_ROW)  # passes validation, fails held-out
    w = World(tmp_path, rigged)
    view = w.promote(w.experiment(d))

    rec = record(view)
    assert view.decision is Decision.REJECTED and view.test_opened_at is not None
    assert rec["reason_codes"] == [ReasonCode.HELDOUT_CONSTRAINT_VIOLATION]
    assert rec["incumbent"] is None
    assert rec["constraints"]["challenger"]["feasible"] is False
    assert rec["constraints"]["challenger"]["violation_codes"] == ["limit_violated"]
    assert rigged.on_test()  # the test split was used even though there is no incumbent
    with pytest.raises(ChampionNotFound):
        w.promotions.current("capitals.capitals")


def test_a_rejected_challenger_cannot_trigger_another_test(tmp_path):
    d = definition(constraints=ConstraintLimits(**LIMITS, minimum_quality=0.5))
    win = winners(d)
    # two workflows pass validation; only the runner-up would pass the test row
    rigged = Rigged(
        lambda g, row: g in (win["fixed"], win["aco"]) and (row != TEST_ROW or g == win["aco"])
    )
    w = World(tmp_path, rigged)
    job = w.experiment(d)
    view = w.promote(job)
    assert view.decision is Decision.REJECTED
    assert view.challenger["genome_hash"] == win["fixed"]  # validation tie-break: earliest
    calls, binds = list(rigged.calls), w.runtime.heldout_binds

    assert w.promote(job) == view  # the decision is final and simply returned
    with pytest.raises(NotPromotable):
        w.promote(job, lineage="another-lineage")  # no second look under another name
    with pytest.raises(Conflict):
        w.store.open("p-other", job, "capitals.capitals", "{}", None, 0, "x", 0.0, LEASE)
    with sqlite3.connect(w.store.path) as conn, pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE promotions SET selection_json='{}', state='OPEN'")
    assert rigged.calls == calls and w.runtime.heldout_binds == binds
    assert {k[0] for k in rigged.on_test()} == {win["fixed"]}  # the runner-up never ran there


# -- incumbent identity -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change",
    [
        {"constraints": ConstraintLimits(**{**LIMITS, "maximum_retries": 1})},
        {"model": ModelConfiguration(provider="test-double", model="other")},
    ],
    ids=["contract", "model_config"],
)
def test_an_incompatible_incumbent_fails_closed(tmp_path, change):
    rigged = Rigged(EVERYTHING)
    w = World(tmp_path, rigged)
    w.promote(w.experiment(definition(strategies=(Strategy.FIXED,))))
    other = w.experiment(definition(strategies=(Strategy.RANDOM,), **change))
    calls = len(rigged.on_test())

    with pytest.raises(IncompatibleIncumbent):
        w.promote(other)

    assert w.store.for_job(other) is None  # nothing recorded: its test split is still closed
    assert len(rigged.on_test()) == calls
    assert w.promotions.current("capitals.capitals").version == 1
    # naming a new lineage starts one, never comparing the two
    view = w.promote(other, lineage="capitals.capitals.v2")
    assert view.decision is Decision.PROMOTED and record(view)["incumbent"] is None
    assert w.promotions.current("capitals.capitals.v2").version == 1


# -- concurrency --------------------------------------------------------------------------------
def test_a_promotion_never_overwrites_a_newer_champion(tmp_path):
    """A opens its test split against an empty lineage; while it is evaluating, B is promoted.
    A's gate passes, but compare-and-promote fails: A is REJECTED / incumbent_superseded."""
    box: dict[str, Any] = {}

    def hook(n, key):
        if key[1] == TEST_ROW and not box.get("fired"):
            box["fired"] = True
            box["b"] = box["other"].promote(box["job_b"])

    rigged = Rigged(EVERYTHING, hook=hook)
    w = World(tmp_path, rigged, name="a")
    job_a = w.experiment(definition(strategies=(Strategy.FIXED,)))
    box["job_b"] = w.experiment(definition(strategies=(Strategy.RANDOM,)))
    box["other"] = World(tmp_path, rigged, name="b").promotions

    view_a = w.promote(job_a)

    rec = record(view_a)
    assert box["b"].decision is Decision.PROMOTED
    assert view_a.decision is Decision.REJECTED
    assert rec["gate"]["decision"] == Decision.PROMOTED  # it passed against what it was judged on
    assert rec["reason_codes"][-1] == ReasonCode.INCUMBENT_SUPERSEDED
    assert rec["champion"] is None
    current = w.promotions.current("capitals.capitals")
    assert current.promotion_id == box["b"].promotion_id and current.version == 1
    assert len(w.promotions.history("capitals.capitals").champions) == 1
    w.promotions.verify(view_a.promotion_id)


def test_racing_promotions_promote_exactly_one(tmp_path):
    barrier = threading.Barrier(2, timeout=30)

    def hook(n, key):
        if key[1] == TEST_ROW:
            barrier.wait()  # both are evaluating held-out, both pinned the same incumbent

    rigged = Rigged(EVERYTHING, hook=hook)
    setup = World(tmp_path, Rigged(EVERYTHING), name="setup")
    jobs = [
        setup.experiment(definition(strategies=(Strategy.FIXED,))),
        setup.experiment(definition(strategies=(Strategy.RANDOM,))),
    ]
    results: dict[str, Any] = {}

    def run(i: int) -> None:
        results[jobs[i]] = World(tmp_path, rigged, name=f"t{i}").promote(jobs[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    decisions = sorted(v.decision.value for v in results.values())
    assert decisions == ["PROMOTED", "REJECTED"]
    loser = next(v for v in results.values() if v.decision is Decision.REJECTED)
    assert record(loser)["reason_codes"][-1] == ReasonCode.INCUMBENT_SUPERSEDED
    history = setup.promotions.history("capitals.capitals").champions
    assert [c.version for c in history] == [1]


def test_compare_and_promote_refuses_a_stale_version_and_writes_nothing(tmp_path):
    store = SQLiteChampionStore(tmp_path / "c.sqlite3")
    a = store.open("pa", "ja", "lin", "{}", None, 0, "a", 0.0, 1e9)
    b = store.open("pb", "jb", "lin", "{}", None, 0, "b", 0.0, 1e9)
    store.decide(a, "PROMOTED", "{}", NewChampion("c1", "compat", "{}"))

    with pytest.raises(StaleIncumbent):
        store.decide(b, "PROMOTED", "{}", NewChampion("c2", "compat", "{}"))

    assert store.get("pb").state is PromotionState.OPEN  # nothing of the failed attempt stuck
    assert [c.champion_id for c in store.history("lin")] == ["c1"]
    assert store.current("lin").version == 1
    with pytest.raises(StaleIncumbent):  # nor can a promotion open against a stale incumbent
        store.open("pc", "jc", "lin", "{}", None, 0, "c", 0.0, 1e9)
    store.decide(b, "REJECTED", "{}")
    assert store.get("pb").state is PromotionState.DECIDED


def test_champion_records_and_decisions_are_immutable(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    w.promote(w.experiment(definition(strategies=(Strategy.FIXED,))))
    with sqlite3.connect(w.store.path) as conn:
        for statement in (
            "UPDATE champions SET record_json='{}'",
            "DELETE FROM champions",
            "UPDATE promotions SET decision='REJECTED'",
            "DELETE FROM promotions",
            "UPDATE promotion_attempts SET result_json='{}'",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)


# -- evidence -----------------------------------------------------------------------------------
def test_promotion_evidence_reproduces_every_decision(tmp_path):
    def rule(win, g, row):
        return row != TEST_ROW or g == win["fixed"]

    w, job, win, _ = two_experiments(tmp_path, rule)
    rejected = w.promote(job)
    promoted = w.promotions.for_job(w.store.promotions()[0].job_id)
    assert (promoted.decision, rejected.decision) == (Decision.PROMOTED, Decision.REJECTED)

    for view in (promoted, rejected):
        rec = record(view)
        assert w.promotions.verify(view.promotion_id) == view
        # the gate is a pure function of the recorded held-out evidence
        decision, reasons, comparison = gate(
            rec["heldout"]["challenger"],
            rec["heldout"]["incumbent"],
        )
        assert (decision, reasons, comparison) == (
            rec["gate"]["decision"],
            rec["gate"]["reason_codes"],
            rec["comparison"],
        )
        assert rec["identities"]["evaluator_version"] == SYNTHETIC_VERSION
        assert rec["identities"]["model_hash"] == "synthetic"
        assert rec["identities"]["heldout_protocol"] == "heldout-gate/1"
        assert rec["experiment"]["job_id"] == view.job_id
        assert len(rec["heldout"]["challenger"]["attempt_ids"]) == 1

    # tampering with a stored held-out run is detected
    with sqlite3.connect(w.store.path) as conn:
        conn.execute("DROP TRIGGER completed_attempts_are_immutable")
        result = json.loads(
            conn.execute(
                "SELECT result_json FROM promotion_attempts WHERE promotion_id=? AND "
                "subject='incumbent'",
                (rejected.promotion_id,),
            ).fetchone()[0]
        )
        result["evaluation"]["verdict"] = "FAIL"
        conn.execute(
            "UPDATE promotion_attempts SET result_json=? WHERE promotion_id=? AND "
            "subject='incumbent'",
            (json.dumps(result), rejected.promotion_id),
        )
    with pytest.raises(EvidenceMismatch):
        w.promotions.verify(rejected.promotion_id)


# -- durable integration ------------------------------------------------------------------------
def test_only_a_completed_experiment_may_enter_promotion(tmp_path):
    clock = Clock()

    def explode(n, key):
        if key[1] == "q2":
            raise RuntimeError("runner broke")

    w = World(tmp_path, Rigged(EVERYTHING), clock=clock)
    pending = w.jobs.create(definition(strategies=(Strategy.FIXED,))).job_id
    cancelled = w.jobs.create(definition(strategies=(Strategy.FIXED,))).job_id
    w.jobs.cancel(cancelled)

    failing = World(tmp_path, Rigged(EVERYTHING, hook=explode), clock=clock, name="f")
    failed = failing.jobs.create(definition(strategies=(Strategy.FIXED,))).job_id
    failing.worker.run_until_idle(failed)

    def crash(n, key):
        raise Crash

    crashing = World(tmp_path, Rigged(EVERYTHING, hook=crash), clock=clock, name="c")
    interrupted = crashing.jobs.create(definition(strategies=(Strategy.FIXED,))).job_id
    with pytest.raises(Crash):
        crashing.worker.run_until_idle(interrupted)
    clock.restart()
    w.worker.recover()

    expected = {
        pending: JobState.PENDING,
        cancelled: JobState.CANCELLED,
        failed: JobState.FAILED,
        interrupted: JobState.INTERRUPTED,
    }
    for job, state in expected.items():
        assert w.jobs.get(job).state is state
        with pytest.raises(NotPromotable):
            w.promote(job)
        assert w.store.for_job(job) is None
    assert w.runtime.heldout_binds == 0


def _crash_after_completed_heldout_attempts(n: int):
    count = [0]

    class Store(SQLiteChampionStore):
        def complete_attempt(self, *args, **kwargs):
            super().complete_attempt(*args, **kwargs)
            count[0] += 1
            if count[0] == n:
                raise Crash

    return Store


def test_a_crash_during_held_out_resumes_without_double_spend(tmp_path):
    d = definition(strategies=(Strategy.FIXED,), trials=2)
    reference = World(tmp_path / "ref", Rigged(EVERYTHING), ids=["j-1"])
    expected = reference.promote(reference.experiment(d))

    clock = Clock()
    rigged = Rigged(EVERYTHING)
    first = World(
        tmp_path / "run",
        rigged,
        clock=clock,
        name="first",
        champion_store=_crash_after_completed_heldout_attempts(1),
        ids=["j-1"],
    )
    job = first.experiment(d)
    with pytest.raises(Crash):
        first.promote(job)
    assert len(rigged.on_test()) == 1

    clock.restart()
    second = World(tmp_path / "run", rigged, clock=clock, name="second")
    assert second.promotions.recover() == [(expected.promotion_id, PromotionReason.PROCESS_LOST)]
    view = second.promote(job)

    assert len(rigged.on_test()) == 2  # the completed held-out run was reused, never re-sent
    assert view.record == expected.record  # and the decision is the uninterrupted one
    assert view.heldout_attempts.completed == 2
    second.promotions.verify(view.promotion_id)


def test_an_ambiguous_held_out_call_is_never_resent(tmp_path):
    d = definition(strategies=(Strategy.FIXED,))
    clock = Clock()

    def crash_on_test(n, key):
        if key[1] == TEST_ROW:
            raise Crash  # the call is in flight: the provider may have charged for it

    crashing = Rigged(EVERYTHING, hook=crash_on_test)
    first = World(tmp_path, crashing, clock=clock, name="first")
    job = first.experiment(d)
    with pytest.raises(Crash):
        first.promote(job)

    clock.restart()
    honest = Rigged(EVERYTHING)
    second = World(tmp_path, honest, clock=clock, name="second")
    (found,) = second.promotions.recover()
    assert found[1] is PromotionReason.AMBIGUOUS_ATTEMPT
    with pytest.raises(PromotionAmbiguous):
        second.promote(job)
    assert honest.calls == []  # nothing re-sent

    suite, _ = contract_suite(d.contract, d.splits, DATA)
    tasks = {t.id: t for t in suite.tasks}

    def proof(attempt):  # e.g. a deterministic response cache: proves without a call
        genome = Genome.from_canonical(json.loads(attempt.genome_json))
        return honest.result(genome, tasks[attempt.task_id], attempt.trial, attempt.run_seed)

    third = World(tmp_path, honest, clock=clock, name="third", proof=proof)
    view = third.promote(job)
    assert view.decision is Decision.PROMOTED and honest.calls == []
    assert view.heldout_attempts.reused_by_proof == 1
    third.promotions.verify(view.promotion_id)


def test_a_model_outage_on_held_out_fails_the_promotion_without_a_decision(tmp_path):
    rigged = Rigged(EVERYTHING, outage=lambda g, row: row == TEST_ROW)
    w = World(tmp_path, rigged)
    job = w.experiment(definition(strategies=(Strategy.FIXED,)))

    with pytest.raises(PromotionFailed):
        w.promote(job)

    view = w.promotions.for_job(job)
    assert view.state is PromotionState.FAILED and view.decision is None
    assert view.reason == PromotionReason.MODEL_UNAVAILABLE
    assert len(rigged.on_test()) == MODEL_ATTEMPTS
    assert w.promote(job) == view  # terminal: the experiment's test split stays spent
    assert len(rigged.on_test()) == MODEL_ATTEMPTS
    with pytest.raises(ChampionNotFound):
        w.promotions.current("capitals.capitals")


def test_promotion_leaves_aco_and_every_strategy_record_exactly_as_the_runner_made_them(tmp_path):
    """The durable job's ACO (and fixed / random) records equal a fresh synchronous #23 run -
    promotion reads them, it never recomputes or alters them."""
    d = definition(seeds=(0, 1))
    rigged = Rigged(lambda g, row: row in ("q1", "q4"))
    w = World(tmp_path, rigged)
    job = w.experiment(d)
    w.promote(job)

    suite, _ = contract_suite(d.contract, d.splits, DATA)
    artifact = w.jobs.artifact(job).artifact
    for run in artifact["runs"]:
        fresh = run_strategy(
            d.plan,
            Strategy(run["strategy"]),
            run["seed"],
            suite,
            Rigged(rigged.passes),
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
        )
        assert without_timing(json.loads(json.dumps(fresh))) == without_timing(run)


def test_promotion_code_never_imports_an_optimizer():
    import ast

    tree = ast.parse((ROOT / "experiments" / "promotion.py").read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(m.split(".")[0] == "optimizers" for m in modules)


# -- product API --------------------------------------------------------------------------------
def _call(api: ProductAPI, method: str, path: str):
    res = api.handle(method, f"/api/v1/{path}", {}, io.BytesIO(b""))
    return res.status, res.body


def test_product_api_promotes_and_exposes_champions(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    api = ProductAPI(None, w.jobs, w.promotions)  # type: ignore[arg-type]
    job = w.experiment(definition(strategies=(Strategy.FIXED,)))
    cancelled = w.jobs.create(definition(strategies=(Strategy.FIXED,))).job_id
    w.jobs.cancel(cancelled)

    assert _call(api, "GET", "champions/capitals.capitals")[0] == 404
    status, body = _call(api, "POST", f"experiments/{job}/promote")
    assert status == 201 and body["decision"] == "PROMOTED"
    pid = body["promotion_id"]
    assert _call(api, "POST", f"experiments/{job}/promote") == (200, body)
    assert _call(api, "GET", f"experiments/{job}/promotion") == (200, body)
    assert _call(api, "GET", f"promotions/{pid}") == (200, body)
    assert _call(api, "GET", f"promotions/{pid}/verify") == (200, body)
    assert _call(api, "GET", "promotions?lineage=capitals.capitals")[1]["promotions"] == [body]
    status, champion = _call(api, "GET", "champions/capitals.capitals")
    assert status == 200 and champion["promotion_id"] == pid and champion["version"] == 1
    status, history = _call(api, "GET", "champions/capitals.capitals/history")
    assert status == 200 and history["champions"] == [champion]

    status, err = _call(api, "POST", f"experiments/{cancelled}/promote")
    assert status == 409 and err["error"]["code"] == "experiment_not_promotable"
    status, err = _call(api, "POST", f"experiments/{job}/promote?bogus=1")
    assert status == 400 and err["error"]["code"] == "unknown_parameter"
    status, err = _call(api, "GET", "promotions/p-missing")
    assert status == 404 and err["error"]["code"] == "promotion_not_found"
