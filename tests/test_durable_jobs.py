"""Durable experiment jobs (Issue #24): restart, exact resume, no double spend, cancellation.

Every model backend here is a TEST DOUBLE (the synthetic objective with injected, deterministic
usage). A "process crash" is ``Crash`` - a ``BaseException`` nothing in the engine catches -
raised at a chosen point; the "restarted process" is a fresh store, runtime and worker on the
same database file with the clock moved past every lease.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.results import (
    BudgetUsage,
    EvaluatedRun,
    ExecutionMetrics,
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
)
from core.run_contract import ExecutionTask
from core.task_contract import TaskContract
from experiments.budget_ledger import ExperimentBudget, MeasuredUsage
from experiments.contract_run import contract_suite
from experiments.jobs import (
    AmbiguousAttemptError,
    ArtifactUnavailable,
    BackendUnavailable,
    DatasetRuntime,
    ExperimentJobDefinition,
    ExperimentJobs,
    InvalidExperiment,
    JobBinding,
    JobWorker,
    NotCancellable,
    NotResumable,
    WorkflowBackend,
    settle_partial,
    zero_totals,
)
from experiments.learning_curves import search_tasks
from experiments.optimization_experiment import (
    ARTIFACT_SCHEMA,
    MODEL_ATTEMPTS,
    Strategy,
    make_strategy,
    optimize_uploaded_dataset,
    run_optimization_experiment,
    run_seed,
    run_strategy,
    without_timing,
)
from experiments.synthetic import SYNTHETIC_VERSION
from optimizers.aco_mmas import MMASACO
from optimizers.base import SearchContext
from optimizers.checkpoint import OptimizerStateError, checkpoint_state, restore
from optimizers.random_search import DistinctRandomSearch
from store.jobs import (
    Claim,
    JobState,
    JobStopped,
    LeaseLost,
    NewAttempt,
    Reason,
    SQLiteJobStore,
    UnitState,
)
from tests.runtime_helpers import versions
from tests.test_contract_runtime import DATA, splits_for
from tests.test_optimization_experiment import (
    ALL,
    FLAT,
    PER_RUN,
    RUNS_PER_CANDIDATE,
    UPLOAD,
    LinearPricing,
    limited_contract,
    metered,
    plan,
    suite_and_contract,
    uploaded_contract,
)

ROOT = Path(__file__).resolve().parent.parent
LEASE = 30.0


class Crash(BaseException):
    """Simulated process death: nothing in the engine catches it."""


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def restart(self) -> None:  # the restarted process starts after every lease expired
        self.now += 10 * LEASE


class Backend:
    """TEST DOUBLE model backend: the synthetic objective + injected usage. Records every call
    it receives (``calls``) and the measured usage of every call it completed (``spent``)."""

    def __init__(self, usage=FLAT, hook: Callable[[int, tuple], None] | None = None) -> None:
        self._evaluate = metered(usage)
        self.hook = hook
        self.calls: list[tuple] = []
        self.spent: list[MeasuredUsage] = []
        self._lock = threading.Lock()

    def __call__(self, genome: Genome, task: ExecutionTask, trial: int, seed: int):
        key = (genome.genome_hash, task.id, trial, seed)
        with self._lock:
            self.calls.append(key)
            n = len(self.calls)
        if self.hook is not None:
            self.hook(n, key)  # may raise Crash: the call is in flight, its result is lost
        run = self._evaluate(genome, task, trial, seed)
        with self._lock:
            self.spent.append(MeasuredUsage.of(run))
        return run

    def result(self, genome: Genome, task: ExecutionTask, trial: int, seed: int):
        """What the backend deterministically answers (without counting a call)."""
        return self._evaluate(genome, task, trial, seed)


class Runtime:
    """TEST DOUBLE runtime: binds a definition to ``Backend`` over the capitals dataset."""

    synthetic = True

    def __init__(self, backend: Backend, pricing=None, proof=None) -> None:
        self.backend, self.pricing, self.proof = backend, pricing, proof

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        suite, _ = contract_suite(definition.contract, definition.splits, DATA)
        return JobBinding(
            suite=suite,
            evaluate=self.backend,
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
            pricing=self.pricing,
            replay_proof=self.proof,
        )


def definition(
    budget: ExperimentBudget | None = None,
    strategies=ALL,
    seeds=(0, 1),
    workers: int = 1,
    **plan_kw: Any,
) -> ExperimentJobDefinition:
    contract = limited_contract()
    return ExperimentJobDefinition(
        contract=contract,
        splits=splits_for(contract),
        plan=plan(
            budget or ExperimentBudget(max_candidate_evaluations=4),
            seeds=seeds,
            strategies=strategies,
            **plan_kw,
        ),
        synthetic=True,
        workers=workers,
    )


class Process:
    """One server process: its own store connection, job service and worker."""

    def __init__(self, path, runtime, clock, store_cls=SQLiteJobStore, name="p") -> None:
        self.store = store_cls(path)
        self.jobs = ExperimentJobs(self.store, runtime, clock=clock)
        self.worker = JobWorker(
            self.store, runtime, worker_id=name, lease_s=LEASE, clock=clock, heartbeat=False
        )

    def run(self):
        try:
            return self.worker.run_until_idle()
        except Crash:
            return "crashed"


def dies_after(kind: str, n: int, then: Callable[[], None] | None = None):
    """A store class whose process dies right AFTER its ``n``-th commit of ``kind``: a
    checkpoint kind ('settled', 'reserved', ...) or 'attempt' (a completed workflow run).
    With ``then``, it calls ``then()`` at that point instead of dying."""
    count = [0]

    def hit() -> None:
        count[0] += 1
        if count[0] == n:
            if then is None:
                raise Crash
            then()

    class Store(SQLiteJobStore):
        def append_checkpoint(self, claim, kind_, *args, **kwargs):
            out = super().append_checkpoint(claim, kind_, *args, **kwargs)
            if kind_ == kind:
                hit()
            return out

        def complete_attempt(self, *args, **kwargs):
            super().complete_attempt(*args, **kwargs)
            if kind == "attempt":
                hit()

    return Store


def reference(tmp_path, d, usage=FLAT, pricing=None):
    """The same job, uninterrupted."""
    backend = Backend(usage)
    proc = Process(tmp_path / "reference.sqlite3", Runtime(backend, pricing), Clock())
    job_id = proc.jobs.create(d).job_id
    proc.run()
    assert proc.jobs.get(job_id).state is JobState.COMPLETED
    return proc, job_id, backend


def crash_and_resume(tmp_path, d, kind, n, usage=FLAT, pricing=None):
    backend, clock, path = Backend(usage), Clock(), tmp_path / "jobs.sqlite3"
    first = Process(path, Runtime(backend, pricing), clock, dies_after(kind, n), "first")
    job_id = first.jobs.create(d).job_id
    assert first.run() == "crashed"
    calls_before_restart = len(backend.calls)
    clock.restart()
    second = Process(path, Runtime(backend, pricing), clock, name="second")
    found = second.worker.recover()
    outcomes = second.run()
    return second, job_id, backend, found, outcomes, calls_before_restart


def artifact(proc: Process, job_id: str) -> dict:
    return proc.jobs.artifact(job_id).artifact


def log(proc: Process, job_id: str, strategy: str, seed: int) -> list[tuple[str, Any]]:
    """A unit's checkpoint log, minus clock-derived values."""
    return [
        (c.kind, without_timing(json.loads(c.payload_json)))
        for c in proc.store.checkpoints(job_id, strategy, seed)
    ]


def spend(usages) -> dict[str, float]:
    return {
        "model_calls": sum(u.model_calls for u in usages),
        "tokens": sum(u.tokens for u in usages),
        "wall_time_s": sum(u.wall_time_s for u in usages),
    }


# == equivalence with the synchronous runner =====================================================
def test_an_uninterrupted_job_reproduces_the_synchronous_experiment(tmp_path):
    pricing = LinearPricing()
    budget = ExperimentBudget(max_candidate_evaluations=4, max_tokens=500_000, max_cost=50.0)
    d = definition(budget, workers=3)
    proc, job_id, backend = reference(tmp_path, d, pricing=pricing)

    suite, _ = suite_and_contract()
    sync_calls: list[str] = []
    sync = run_optimization_experiment(
        d.plan,
        suite,
        metered(calls=sync_calls),
        checker=ConstraintChecker(),
        evaluator_version=SYNTHETIC_VERSION,
        synthetic=True,
        pricing=pricing,
        workers=3,
    )
    durable = artifact(proc, job_id)
    assert durable["schema"] == ARTIFACT_SCHEMA
    assert without_timing(durable) == without_timing(sync)  # the existing assemble(), same science
    assert len(backend.calls) == len(sync_calls)
    view = proc.jobs.get(job_id)
    assert view.experiment_id == durable["experiment_id"]
    assert [(u.strategy, u.seed) for u in view.units] == [
        (r["strategy"], r["seed"]) for r in sync["runs"]
    ]


def test_the_checkpoint_hook_never_changes_what_runs():
    suite, _ = suite_and_contract()
    p = plan(ExperimentBudget(max_candidate_evaluations=6), seeds=(0,))
    for strategy in ALL:
        events: list[tuple[str, dict]] = []
        kwargs = dict(checker=ConstraintChecker(), evaluator_version=SYNTHETIC_VERSION)
        plain = run_strategy(p, strategy, 0, suite, metered(), **kwargs)
        hooked = run_strategy(
            p,
            strategy,
            0,
            suite,
            metered(),
            checkpoint=lambda k, x, events=events: events.append((k, x)),
            **kwargs,
        )
        assert without_timing(plain) == without_timing(hooked)
        assert [k for k, _ in events if k == "settled"] == ["settled"] * len(plain["candidates"])


# == the checkpoint format =======================================================================
def test_every_settled_candidate_and_workflow_run_is_checkpointed(tmp_path):
    d = definition(ExperimentBudget(max_candidate_evaluations=5, max_tokens=500_000), seeds=(0,))
    proc, job_id, _ = reference(tmp_path, d)
    for unit in proc.jobs.get(job_id).units:
        record = json.loads(
            next(
                u.record_json
                for u in proc.store.get_job(job_id).units
                if (u.strategy, u.seed) == (unit.strategy, unit.seed)
            )
        )
        checkpoints = proc.store.checkpoints(job_id, unit.strategy, unit.seed)
        attempts = proc.store.attempts(job_id, unit.strategy, unit.seed)
        settled = [json.loads(c.payload_json) for c in checkpoints if c.kind == "settled"]
        assert len(settled) == record["usage"]["candidate_evaluations"]
        assert len(attempts) == record["usage"]["workflow_runs"]  # one row per workflow run
        assert all(a.state.value == "COMPLETED" and a.result_json for a in attempts)
        for c in settled:
            assert set(c) == {
                "evaluation",
                "round",
                "genome_hash",
                "candidate",
                "curve_point",
                "ledger",
                "champion",
                "optimizer_state",
            }
        # the ledger at every settlement is exactly the measured usage of the runs before it
        seq = [c.seq for c in checkpoints if c.kind == "settled"]
        for s, c in zip(seq, settled, strict=True):
            entries = [json.loads(a.entry_json) for a in attempts if a.seq < s]
            expected = settle_partial(zero_totals(False), entries)
            expected["candidate_evaluations"] = c["evaluation"]
            assert c["ledger"] == expected
        assert settled[-1]["ledger"] == record["usage"]
        reserved = [json.loads(c.payload_json) for c in checkpoints if c.kind == "reserved"]
        assert all(
            r["reservation"]["tokens"] == RUNS_PER_CANDIDATE * PER_RUN["tokens"] for r in reserved
        )
    aco = [p for k, p in log(proc, job_id, "aco", 0) if k == "observed"]
    state = aco[-1]["optimizer_state"]
    assert state["optimizer"] == "aco_mmas" and state["epoch"] == len(aco)
    assert {"edges", "pheromones", "board", "genomes", "global_best", "lcb_range"} <= set(state)


# == restart after a settled candidate / mid-candidate ===========================================
@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_restart_after_a_settled_candidate_resumes_exactly(tmp_path, n):
    d = definition()
    ref, ref_id, ref_backend = reference(tmp_path, d)
    proc, job_id, backend, found, outcomes, _ = crash_and_resume(tmp_path, d, "settled", n)
    assert found and all(reason is Reason.PROCESS_LOST for *_, reason in found)
    assert proc.jobs.get(job_id).state is JobState.COMPLETED
    assert without_timing(artifact(proc, job_id)) == without_timing(artifact(ref, ref_id))
    # exactly the uninterrupted calls: none twice, none extra (strategies sharing a seed share
    # run seeds for the same genome - common random numbers - so a key can recur across units)
    assert Counter(backend.calls) == Counter(ref_backend.calls)
    resumed = next(o for o in outcomes if (o.claim.strategy, o.claim.seed) == found[0][1:3])
    assert resumed.reused > 0  # the settled candidates came from the log, not the model


@pytest.mark.parametrize("n", [7, 13, 22])
def test_restart_mid_candidate_reuses_its_completed_rows(tmp_path, n):
    d = definition()
    ref, ref_id, ref_backend = reference(tmp_path, d)
    proc, job_id, backend, found, outcomes, before = crash_and_resume(tmp_path, d, "attempt", n)
    assert n % RUNS_PER_CANDIDATE  # the crash really is inside a candidate
    assert without_timing(artifact(proc, job_id)) == without_timing(artifact(ref, ref_id))
    assert Counter(backend.calls) == Counter(ref_backend.calls)
    _, strategy, seed, _ = found[0]
    resumed = next(o for o in outcomes if (o.claim.strategy, o.claim.seed) == (strategy, seed))
    stored = [a for a in proc.store.attempts(job_id, strategy, seed)]
    assert resumed.reused == sum(a.owner == "first" for a in stored) > 0
    assert resumed.executed == sum(a.owner == "second" for a in stored)
    assert len(backend.calls) - before == sum(o.executed for o in outcomes)


def test_a_restart_with_concurrent_runs_per_candidate_resumes_exactly(tmp_path):
    d = definition(workers=3)
    ref, ref_id, ref_backend = reference(tmp_path, d)
    proc, job_id, backend, *_ = crash_and_resume(tmp_path, d, "attempt", 12)
    assert without_timing(artifact(proc, job_id)) == without_timing(artifact(ref, ref_id))
    assert Counter(backend.calls) == Counter(ref_backend.calls)


def test_a_job_dying_before_assembly_is_assembled_on_recovery(tmp_path):
    d = definition(seeds=(0,))
    ref, ref_id, _ = reference(tmp_path, d)
    backend, clock, path = Backend(), Clock(), tmp_path / "jobs.sqlite3"

    class DiesBeforeAssembly(SQLiteJobStore):
        def put_artifact(self, *args, **kwargs):
            raise Crash

    first = Process(path, Runtime(backend), clock, DiesBeforeAssembly, "first")
    job_id = first.jobs.create(d).job_id
    assert first.run() == "crashed"
    assert first.jobs.get(job_id).state is JobState.RUNNING  # every unit done, no artifact yet
    calls = len(backend.calls)
    second = Process(path, Runtime(backend), clock, name="second")
    second.worker.recover()
    assert len(backend.calls) == calls
    assert without_timing(artifact(second, job_id)) == without_timing(artifact(ref, ref_id))


# == ambiguous in-flight attempts ================================================================
def test_an_in_flight_model_call_is_never_silently_replayed(tmp_path):
    d = definition()
    ref, ref_id, _ = reference(tmp_path, d)
    crash_on: list[tuple] = []

    def die(n, key):
        if n == 8:
            crash_on.append(key)
            raise Crash  # the provider received the request; the result never came back

    backend, clock, path = Backend(hook=die), Clock(), tmp_path / "jobs.sqlite3"
    first = Process(path, Runtime(backend), clock, name="first")
    job_id = first.jobs.create(d).job_id
    assert first.run() == "crashed"
    clock.restart()
    second = Process(path, Runtime(backend), clock, name="second")
    found = second.worker.recover()
    assert [(s, seed, r) for _, s, seed, r in found] == [("random", 0, Reason.AMBIGUOUS_ATTEMPT)]
    second.run()  # every other unit completes; the ambiguous one is left alone
    view = second.jobs.get(job_id)
    assert view.state is JobState.INTERRUPTED and view.reason == "ambiguous_attempt"
    unit = next(u for u in view.units if (u.strategy, u.seed) == ("random", 0))
    assert unit.state is UnitState.INTERRUPTED and unit.attempts.in_flight == 1
    assert backend.calls.count(crash_on[0]) == 1  # never re-sent
    with pytest.raises(AmbiguousAttemptError):
        second.jobs.resume(job_id)
    with pytest.raises(ArtifactUnavailable):
        second.jobs.artifact(job_id)
    calls = len(backend.calls)
    second.run()
    assert len(backend.calls) == calls

    # a runtime that can PROVE the result without a call (e.g. a deterministic cache) resolves it
    def proof(attempt):
        suite, _ = suite_and_contract()
        task = next(
            t for t in (*search_tasks(suite)[0], *search_tasks(suite)[1]) if t.id == attempt.task_id
        )
        genome = Genome.from_canonical(json.loads(attempt.genome_json))
        return backend.result(genome, task, attempt.trial, attempt.run_seed)

    third = Process(path, Runtime(backend, proof=proof), clock, name="third")
    third.jobs.resume(job_id)
    third.run()
    assert third.jobs.get(job_id).state is JobState.COMPLETED
    assert backend.calls.count(crash_on[0]) == 1
    # random/0 had 2 stored runs + 1 proven: only the rest of it is executed now
    assert len(backend.calls) == calls + len(ref.store.attempts(ref_id, "random", 0)) - 3
    assert without_timing(artifact(third, job_id)) == without_timing(artifact(ref, ref_id))
    attempts = third.store.attempts(job_id, "random", 0)
    assert sum(a.resolution == "replay_proof" for a in attempts) == 1


def test_a_late_result_from_the_original_worker_resolves_the_ambiguity(tmp_path):
    """Worker A stalls inside a call; its lease expires and B finds the call in flight (ambiguous,
    not claimed). A's result then arrives and is stored; A itself is fenced out. Resume is then
    safe and nothing ran twice."""
    d = definition(seeds=(0,), strategies=(Strategy.RANDOM,))
    ref, ref_id, ref_backend = reference(tmp_path, d)
    clock, path = Clock(), tmp_path / "jobs.sqlite3"
    b_store = SQLiteJobStore(path)

    def stall(n, key):
        if n == 3:
            clock.restart()  # A's lease expires while the call is in flight
            assert b_store.claim("B", clock(), LEASE) is None  # in flight: not claimable
            assert b_store.get_job(job[0]).state is JobState.INTERRUPTED

    backend = Backend(hook=stall)
    a = Process(path, Runtime(backend), clock, name="A")
    job = [a.jobs.create(d).job_id]
    a.run()  # A stores the late result, then is fenced out before any further call
    assert len(backend.calls) == 3
    stored = a.store.attempts(job[0], "random", 0)
    assert [s.state.value for s in stored] == ["COMPLETED"] * 3
    b = Process(path, Runtime(backend), clock, name="B")
    b.jobs.resume(job[0])
    b.run()
    assert without_timing(artifact(b, job[0])) == without_timing(artifact(ref, ref_id))
    assert Counter(backend.calls) == Counter(ref_backend.calls)


# == optimizer state =============================================================================
@pytest.mark.parametrize("strategy", [Strategy.ACO, Strategy.RANDOM])
def test_optimizer_state_after_resume_equals_the_uninterrupted_run(tmp_path, strategy):
    d = definition(
        ExperimentBudget(max_candidate_evaluations=9), seeds=(0,), strategies=(strategy,)
    )
    ref, ref_id, _ = reference(tmp_path, d)
    for kind, n in (("settled", 3), ("attempt", 17), ("observed", 2)):
        sub = tmp_path / f"{kind}-{n}"
        sub.mkdir()
        proc, job_id, *_ = crash_and_resume(sub, d, kind, n)
        # the WHOLE progress log - proposals, every pheromone / epoch / score-board / rng-call
        # state, reservations, ledgers - is the uninterrupted one
        assert log(proc, job_id, strategy.value, 0) == log(ref, ref_id, strategy.value, 0)
        resumed = json.loads(proc.store.get_job(job_id).units[0].record_json)
        expected = json.loads(ref.store.get_job(ref_id).units[0].record_json)
        assert resumed["evaluated_genome_hashes"] == expected["evaluated_genome_hashes"]
    final = [p for k, p in log(ref, ref_id, strategy.value, 0) if k in ("observed", "proposed")]
    state = final[-1]["optimizer_state"]
    if strategy is Strategy.ACO:
        assert state["epoch"] > 0 and state["pheromones"]
    else:
        assert state["seen"] and state["calls"] > 0


def _drive(optimizer, rounds: range) -> list[list[str]]:
    suite, _ = suite_and_contract()
    train, _ = search_tasks(suite)
    evaluate = metered()
    out = []
    for r in rounds:
        context = SearchContext(contract=suite.policy, checker=ConstraintChecker(), seed=7, round=r)
        proposals = optimizer.propose(2, context)
        out.append([g.genome_hash for g in proposals])
        runs = [
            evaluate(g, t, 0, run_seed(7, g.genome_hash, t.id, 0)) for g in proposals for t in train
        ]
        if runs:
            optimizer.observe(runs)
    return out


@pytest.mark.parametrize("strategy", ALL)
def test_an_optimizer_restored_from_its_checkpoint_continues_exactly(strategy):
    p = plan()
    full_opt = make_strategy(strategy, p)
    full = _drive(full_opt, range(8))
    head_opt = make_strategy(strategy, p)
    head = _drive(head_opt, range(4))
    state = json.loads(json.dumps(checkpoint_state(head_opt)))  # through storage
    restored = restore(state)
    assert type(restored) is type(head_opt)
    assert head + _drive(restored, range(4, 8)) == full
    assert checkpoint_state(restored) == checkpoint_state(full_opt)


def test_a_checkpoint_never_restores_another_optimizer():
    aco = MMASACO()
    _drive(aco, range(2))
    state = checkpoint_state(aco)
    with pytest.raises(OptimizerStateError):
        restore({**state, "optimizer": DistinctRandomSearch.name})
    with pytest.raises(OptimizerStateError):
        restore({**state, "version": "aco_mmas/0"})
    tampered = json.loads(json.dumps(state))
    tampered["pheromones"][0][2] += 0.5  # derived state that no longer follows from the edges
    with pytest.raises(OptimizerStateError):
        restore(tampered)

    class Custom(MMASACO):  # an optimizer it does not know exactly has no checkpoint
        pass

    assert checkpoint_state(Custom()) is None


# == budget continuity ===========================================================================
CAPS = {
    "model_calls": ("max_model_calls", RUNS_PER_CANDIDATE * PER_RUN["model_calls"], 2),
    "tokens": ("max_tokens", RUNS_PER_CANDIDATE * PER_RUN["tokens"], 300),
    "wall_time_s": ("max_wall_time_s", RUNS_PER_CANDIDATE * PER_RUN["wall_time_s"], 1.5),
}


@pytest.mark.parametrize("resource", sorted(CAPS))
@pytest.mark.parametrize("crash", [("attempt", 8), ("settled", 2), ("reserved", 3)])
def test_budget_usage_survives_a_restart_exactly(tmp_path, resource, crash):
    field, reservation, per_run = CAPS[resource]
    cap = reservation + 2.5 * RUNS_PER_CANDIDATE * per_run  # room for 3 candidates, not a 4th
    d = definition(ExperimentBudget(max_candidate_evaluations=100, **{field: cap}))
    ref, ref_id, ref_backend = reference(tmp_path, d)
    proc, job_id, backend, *_ = crash_and_resume(tmp_path, d, *crash)
    for got, want in zip(proc.jobs.get(job_id).units, ref.jobs.get(ref_id).units, strict=True):
        assert got.usage == want.usage and got.stop_reason == want.stop_reason
        assert got.usage[resource] <= cap
        if got.strategy != "fixed":
            assert got.stop_reason == resource  # the same hard reservation rule stopped it
    # what the backend actually spent across both processes is exactly the uninterrupted spend
    assert spend(backend.spent) == spend(ref_backend.spent)


@pytest.mark.parametrize("n", [1, 4, 6, 9, 11, 14, 16, 21])
def test_resume_can_never_gain_calls_tokens_time_or_cost(tmp_path, n):
    pricing = LinearPricing()
    budget = ExperimentBudget(
        max_candidate_evaluations=50, max_tokens=320_000, max_model_calls=70, max_cost=40.0
    )
    d = definition(budget, seeds=(0,))
    ref, ref_id, ref_backend = reference(tmp_path, d, pricing=pricing)
    proc, job_id, backend, *_ = crash_and_resume(tmp_path, d, "attempt", n, pricing=pricing)
    for got, want in zip(proc.jobs.get(job_id).units, ref.jobs.get(ref_id).units, strict=True):
        assert got.usage == want.usage
        assert got.usage["tokens"] <= budget.max_tokens
        assert got.usage["model_calls"] <= budget.max_model_calls
        assert got.usage["cost"] <= budget.max_cost
    assert len(backend.spent) == len(ref_backend.spent)
    assert spend(backend.spent) == spend(ref_backend.spent)


def test_a_tampered_run_log_cannot_buy_budget(tmp_path):
    """Rewriting a stored run's usage (e.g. tokens -> 0) is detected on resume: the job fails
    closed before a single new model call."""
    d = definition(seeds=(0,), strategies=(Strategy.RANDOM,))
    backend, clock, path = Backend(), Clock(), tmp_path / "jobs.sqlite3"
    first = Process(path, Runtime(backend), clock, dies_after("attempt", 7), "first")
    job_id = first.jobs.create(d).job_id
    assert first.run() == "crashed"
    with sqlite3.connect(path) as conn:
        attempt_id, raw = conn.execute(
            "SELECT attempt_id, result_json FROM experiment_attempts ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        run = EvaluatedRun.model_validate_json(raw)
        cheap = run.model_copy(
            update={"execution": run.execution.model_copy(update={"budget_usage": BudgetUsage()})}
        )
        conn.execute(
            "UPDATE experiment_attempts SET result_json=? WHERE attempt_id=?",
            (cheap.model_dump_json(), attempt_id),
        )
    calls = len(backend.calls)
    clock.restart()
    second = Process(path, Runtime(backend), clock, name="second")
    second.run()
    view = second.jobs.get(job_id)
    assert view.state is JobState.FAILED and view.reason == "replay_diverged"
    assert len(backend.calls) == calls


# == cancellation ================================================================================
def test_cancellation_launches_zero_subsequent_calls(tmp_path):
    clock, path = Clock(), tmp_path / "jobs.sqlite3"
    job: list[str] = []
    states: list[JobState] = []

    def user_cancels(n, key):
        if n == 8:  # while the 8th call is in flight (random/0, 3rd row of its first candidate)
            other = ExperimentJobs(SQLiteJobStore(path), None, clock=clock)  # e.g. the API
            states.append(other.cancel(job[0]).state)

    backend = Backend(hook=user_cancels)
    proc = Process(path, Runtime(backend), clock)
    job.append(proc.jobs.create(definition()).job_id)
    proc.run()
    assert states == [JobState.CANCEL_REQUESTED]  # persisted before anything else happened
    assert len(backend.calls) == 8  # the call in flight finished; not one more started
    view = proc.jobs.get(job[0])
    assert view.state is JobState.CANCELLED and view.cancel_requested_at is not None
    by_unit = {(u.strategy, u.seed): u for u in view.units}
    assert by_unit["fixed", 0].state is UnitState.COMPLETED  # finished before the request
    cancelled = by_unit["random", 0]
    assert cancelled.state is UnitState.CANCELLED and cancelled.reason == "cancelled"
    assert cancelled.attempts.completed == 3 and cancelled.attempts.in_flight == 0
    assert cancelled.candidate_evaluations == 0 and cancelled.open_reservation is None
    assert cancelled.usage["model_calls"] == 3 * FLAT["model_calls"]  # returned work is kept
    assert cancelled.usage["tokens"] == 3 * FLAT["tokens"]
    assert all(by_unit[k].state is UnitState.CANCELLED for k in by_unit if k != ("fixed", 0))
    with pytest.raises(ArtifactUnavailable):
        proc.jobs.artifact(job[0])
    with pytest.raises(NotResumable):
        proc.jobs.resume(job[0])
    with pytest.raises(NotCancellable):
        proc.jobs.cancel(job[0])
    proc.run()
    assert len(backend.calls) == 8 and proc.jobs.get(job[0]).state is JobState.CANCELLED


def test_cancellation_at_a_settled_boundary_stops_before_the_next_candidate(tmp_path):
    clock, path = Clock(), tmp_path / "jobs.sqlite3"
    job: list[str] = []
    seen: list[int] = []
    backend = Backend()

    def cancel() -> None:
        seen.append(len(backend.calls))
        ExperimentJobs(SQLiteJobStore(path), None, clock=clock).cancel(job[0])

    proc = Process(path, Runtime(backend), clock, dies_after("settled", 2, then=cancel))
    job.append(proc.jobs.create(definition()).job_id)
    proc.run()
    assert len(backend.calls) == seen[0] == 2 * RUNS_PER_CANDIDATE
    view = proc.jobs.get(job[0])
    assert view.state is JobState.CANCELLED
    random0 = next(u for u in view.units if (u.strategy, u.seed) == ("random", 0))
    assert random0.candidate_evaluations == 1 and random0.usage == random0.settled_usage


def test_a_pending_job_cancels_without_running_anything(tmp_path):
    backend = Backend()
    proc = Process(tmp_path / "jobs.sqlite3", Runtime(backend), Clock())
    job_id = proc.jobs.create(definition()).job_id
    assert proc.jobs.cancel(job_id).state is JobState.CANCELLED
    proc.run()
    assert backend.calls == []
    assert {u.state for u in proc.jobs.get(job_id).units} == {UnitState.CANCELLED}


# == concurrency =================================================================================
def test_concurrent_workers_never_claim_the_same_unit(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    proc = Process(path, Runtime(Backend()), Clock())
    proc.jobs.create(definition())  # 6 units
    barrier, claims, lock = threading.Barrier(9), [], threading.Lock()

    def grab(i: int) -> None:
        store = SQLiteJobStore(path)
        barrier.wait()
        claim = store.claim(f"w{i}", 1_000.0, LEASE)
        with lock:
            claims.append(claim)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = [c for c in claims if c is not None]
    assert len(got) == 6 and len({c.unit for c in got}) == 6
    assert SQLiteJobStore(path).claim("late", 1_000.0, LEASE) is None  # every lease is live


def test_separate_processes_never_claim_the_same_unit(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "jobs.sqlite3"
    Process(path, Runtime(Backend()), Clock()).jobs.create(definition())  # 6 units
    script = (
        "import sys, time; from store.jobs import SQLiteJobStore\n"
        "s = SQLiteJobStore(sys.argv[1]); time.sleep(float(sys.argv[3]) - time.time())\n"
        "out = []\n"
        "while (c := s.claim(sys.argv[2], 1000.0, 30.0)) is not None: out.append(c)\n"
        "print(';'.join(f'{c.strategy}/{c.seed}' for c in out))\n"
    )
    start = str(time.time() + 3.0)  # all processes start claiming at the same instant
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(path), f"proc{i}", start],
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )
        for i in range(4)
    ]
    claimed = [u for p in procs for u in p.communicate(timeout=120)[0].strip().split(";") if u]
    assert all(p.returncode == 0 for p in procs)
    assert sorted(claimed) == sorted(f"{s}/{d}" for d in (0, 1) for s in ("fixed", "random", "aco"))


def test_a_worker_that_lost_its_lease_can_neither_start_a_run_nor_write(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    proc = Process(path, Runtime(Backend()), Clock())
    proc.jobs.create(definition(seeds=(0,), strategies=(Strategy.RANDOM,)))
    a = SQLiteJobStore(path).claim("A", 1_000.0, LEASE)
    b = SQLiteJobStore(path).claim("B", 1_000.0 + 2 * LEASE, LEASE)  # A's lease expired
    assert a is not None and b is not None and a.unit == b.unit and b.fence == a.fence + 1
    attempt = NewAttempt("x" * 64, "g", "{}", "q1", 0, 1, 0)
    store = SQLiteJobStore(path)
    with pytest.raises(LeaseLost):  # no write-ahead row, hence no model call
        store.start_attempt(a, attempt, 1_000.0 + 2 * LEASE, LEASE)
    with pytest.raises(LeaseLost):
        store.append_checkpoint(a, "settled", 1, 0, "{}", 1_000.0 + 2 * LEASE, LEASE)
    with pytest.raises(LeaseLost):
        store.finish_unit(a, UnitState.FAILED, Reason.RUNNER_ERROR)
    with pytest.raises(LeaseLost):
        store.renew(a, 1_000.0 + 2 * LEASE, LEASE)
    store.start_attempt(b, attempt, 1_000.0 + 2 * LEASE, LEASE)  # the owner can
    assert [x.owner for x in store.attempts(*b.unit)] == ["B"]
    forged = Claim(b.job_id, b.strategy, b.seed, "A", b.fence)  # right fence, wrong owner
    with pytest.raises(LeaseLost):
        store.renew(forged, 1_000.0 + 2 * LEASE, LEASE)


def test_parallel_workers_complete_a_job_exactly_like_one_worker(tmp_path):
    d = definition()
    ref, ref_id, ref_backend = reference(tmp_path, d)
    backend, clock, path = Backend(), Clock(), tmp_path / "jobs.sqlite3"
    job_id = Process(path, Runtime(backend), clock).jobs.create(d).job_id
    procs = [Process(path, Runtime(backend), clock, name=f"w{i}") for i in range(3)]
    errors: list[BaseException] = []

    def work(p: Process) -> None:
        try:
            p.worker.run_until_idle()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(p,)) for p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert without_timing(artifact(procs[0], job_id)) == without_timing(artifact(ref, ref_id))
    assert Counter(backend.calls) == Counter(ref_backend.calls)
    owners = {
        a.owner
        for u in procs[0].jobs.get(job_id).units
        for a in procs[0].store.attempts(job_id, u.strategy, u.seed)
    }
    assert owners <= {"w0", "w1", "w2"}


def test_a_heartbeat_keeps_a_long_unit_owned(tmp_path):
    """With a real clock and a short lease, the heartbeat renews the lease while a slow model
    call is in flight, so no other worker can take the unit over."""
    path = tmp_path / "jobs.sqlite3"
    lease, intruder = 0.3, []

    def slow(n, key):
        if n == 2:
            time.sleep(4 * lease)
            intruder.append(SQLiteJobStore(path).claim("intruder", time.time(), lease))

    backend = Backend(hook=slow)
    runtime = Runtime(backend)
    store = SQLiteJobStore(path)
    job_id = (
        ExperimentJobs(store, runtime)
        .create(definition(seeds=(0,), strategies=(Strategy.FIXED,)))
        .job_id
    )
    JobWorker(store, runtime, lease_s=lease).run_until_idle()
    assert intruder == [None]
    assert ExperimentJobs(store, runtime).get(job_id).state is JobState.COMPLETED


# == failures and the strategies =================================================================
def test_a_unit_that_fails_closed_fails_the_job_and_stops_the_rest(tmp_path):
    def down(genome, task):
        return FLAT

    class Down(Backend):
        def __call__(self, genome, task, trial, seed):
            run = super().__call__(genome, task, trial, seed)
            if len(self.calls) <= RUNS_PER_CANDIDATE:  # the fixed baseline completes
                return run
            failure = FailureInfo(kind=FailureKind.MODEL_ERROR, message="backend down")
            return run.model_copy(
                update={"execution": run.execution.model_copy(update={"failure": failure})}
            )

    backend = Down(down)
    proc = Process(tmp_path / "jobs.sqlite3", Runtime(backend), Clock())
    job_id = proc.jobs.create(definition()).job_id
    proc.run()
    view = proc.jobs.get(job_id)
    assert view.state is JobState.FAILED and view.reason == "model_unavailable"
    by_unit = {(u.strategy, u.seed): u for u in view.units}
    assert by_unit["fixed", 0].state is UnitState.COMPLETED
    assert by_unit["random", 0].state is UnitState.FAILED
    assert {by_unit[k].reason for k in by_unit if k not in {("fixed", 0), ("random", 0)}} == {
        "job_failed"
    }
    # the baseline's candidate, then random/0's rows and its row-1 retries, then it stopped
    assert len(backend.calls) == 2 * RUNS_PER_CANDIDATE + MODEL_ATTEMPTS - 1
    with pytest.raises(NotResumable):
        proc.jobs.resume(job_id)
    with pytest.raises(ArtifactUnavailable):
        proc.jobs.artifact(job_id)


def test_a_runtime_that_no_longer_reproduces_the_job_is_refused(tmp_path):
    path, clock = tmp_path / "jobs.sqlite3", Clock()
    first = Process(path, Runtime(Backend()), clock)
    job_id = first.jobs.create(definition(seeds=(0,))).job_id
    priced = Process(path, Runtime(Backend(), pricing=LinearPricing()), clock)  # other pricing
    priced.run()
    view = priced.jobs.get(job_id)
    assert view.state is JobState.INTERRUPTED and view.reason == "runtime_unavailable"
    first.jobs.resume(job_id)
    first.run()
    assert first.jobs.get(job_id).state is JobState.COMPLETED


def test_create_fails_closed_before_storing_anything(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    with pytest.raises(BackendUnavailable):
        ExperimentJobs(store, None).create(definition())
    jobs = ExperimentJobs(store, Runtime(Backend()))
    with pytest.raises(InvalidExperiment):  # a cost cap without pricing can never be enforced
        jobs.create(definition(ExperimentBudget(max_candidate_evaluations=2, max_cost=1.0)))
    with pytest.raises(InvalidExperiment):  # a synthetic runtime never produces a "real" result
        jobs.create(definition().model_copy(update={"synthetic": False}))
    assert store.list_job_ids() == []


def test_the_job_store_refuses_to_leave_a_terminal_state(tmp_path):
    proc = Process(tmp_path / "jobs.sqlite3", Runtime(Backend()), Clock())
    job_id = proc.jobs.create(definition(seeds=(0,), strategies=(Strategy.FIXED,))).job_id
    proc.run()
    assert proc.jobs.get(job_id).state is JobState.COMPLETED
    with pytest.raises(NotCancellable):
        proc.jobs.cancel(job_id)
    assert proc.store.put_artifact(job_id, "{}") is False  # written once


def test_job_stopped_is_raised_for_a_failed_job(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    proc = Process(path, Runtime(Backend()), Clock())
    job_id = proc.jobs.create(definition(seeds=(0,))).job_id
    claim = proc.store.claim("w", 1_000.0, LEASE)
    proc.store.fail_job(job_id, Reason.ASSEMBLY_FAILED, "x")
    with pytest.raises(JobStopped):
        proc.store.start_attempt(
            claim, NewAttempt("y" * 64, "g", "{}", "q1", 0, 1, 0), 1_000.0, LEASE
        )


# == the product API =============================================================================
def _passage_runner(model_hash: str):
    """TEST DOUBLE workflow runner: answers with the passage's first word (the city)."""

    def run_workflow(genome: Genome, task: ExecutionTask, trial: int, seed: int):
        from core.payloads import Answer

        return ExecutionResult(
            key=RunKey(
                genome_hash=genome.genome_hash,
                task_id=task.id,
                contract_hash=task.contract_hash,
                trial=trial,
                seed=seed,
                versions=versions().model_copy(update={"model_hash": model_hash}),
            ),
            answer=Answer(values={"answer": str(task.example.values["passage"]).split()[0]}),
            metrics=ExecutionMetrics(model_calls=1, prompt_tokens=60, completion_tokens=40),
            budget_usage=BudgetUsage(tokens=100, wall_time_s=0.5),
        )

    return run_workflow


MODEL_HASH = "passage-runner/1"


def _runtime(service):
    def backend(definition: ExperimentJobDefinition) -> WorkflowBackend:
        return WorkflowBackend(_passage_runner(MODEL_HASH), ConstraintChecker())

    return DatasetRuntime(data=service.blobs.get, backend=backend, synthetic=True)


def _register(api) -> tuple[dict, dict]:
    from tests.test_product_api import call

    project = call(api, "POST", "/api/v1/projects", json_body={"name": "Capitals"}).body
    up = call(
        api, "POST", f"/api/v1/projects/{project['project_id']}/uploads?format=csv", UPLOAD
    ).body
    mapping = {
        "dataset_id": "capitals-upload",
        "name": "Capitals (uploaded CSV)",
        "input_columns": ["question"],
        "target_columns": ["answer"],
        "context_columns": ["passage"],
        "row_ids": "column",
        "id_column": "id",
    }
    version = call(
        api, "POST", f"/api/v1/uploads/{up['upload_id']}/register", json_body=mapping
    ).body
    splits = call(
        api,
        "POST",
        "/api/v1/datasets/capitals-upload/versions/1/splits",
        json_body={"seed": 1, "validation_bps": 2500, "test_bps": 2500},
    ).body
    return version, splits


def _create_body(version: dict, splits: dict) -> dict:
    contract = TaskContract.model_validate(
        {**uploaded_contract().model_dump(mode="json"), "dataset": version["spec"]}
    )
    p = plan(
        ExperimentBudget(max_candidate_evaluations=3, max_tokens=200_000),
        seeds=(0, 1),
        expected_model_hash=MODEL_HASH,
    )
    return {
        "dataset_id": "capitals-upload",
        "dataset_version": 1,
        "splits_hash": splits["splits_hash"],
        "contract": contract.model_dump(mode="json"),
        "plan": p.model_dump(mode="json"),
    }


def test_experiment_jobs_through_the_product_api(tmp_path):
    from api.product import build_api
    from tests.test_product_api import call, error

    api = build_api(tmp_path / "data", runtime=_runtime)
    version, splits = _register(api)
    body = _create_body(version, splits)
    res = call(api, "POST", "/api/v1/experiments", json_body=body)
    assert res.status == 201, res.body
    job_id = res.body["job_id"]
    assert res.body["state"] == "PENDING" and res.body["synthetic"] is True
    assert len(res.body["units"]) == 6
    error(call(api, "GET", f"/api/v1/experiments/{job_id}/artifact"), 409, "job_not_completed")

    JobWorker(api.jobs.store, api.jobs.runtime, heartbeat=False).run_until_idle()
    got = call(api, "GET", f"/api/v1/experiments/{job_id}")
    assert got.status == 200 and got.body["state"] == "COMPLETED"
    assert all(u["state"] == "COMPLETED" and u["curve"] for u in got.body["units"])
    listed = call(api, "GET", "/api/v1/experiments").body["jobs"]
    assert [j["job_id"] for j in listed] == [job_id] and listed[0]["artifact_available"]
    art = call(api, "GET", f"/api/v1/experiments/{job_id}/artifact")
    assert art.status == 200 and art.body["artifact"]["schema"] == ARTIFACT_SCHEMA
    assert art.body["experiment_id"] == got.body["experiment_id"]

    # the same science as the synchronous uploaded-dataset entry point
    contract = TaskContract.model_validate(body["contract"])
    record = api.service.get_splits("capitals-upload", 1, splits["splits_hash"])
    definition_ = ExperimentJobDefinition.model_validate_json(
        api.jobs.store.get_job(job_id).definition_json
    )
    sync = optimize_uploaded_dataset(
        contract,
        record.splits,
        UPLOAD,
        _passage_runner(MODEL_HASH),
        definition_.plan,
        checker=ConstraintChecker(),
        synthetic=True,
        provenance=definition_.provenance,
    )
    assert without_timing(art.body["artifact"]) == without_timing(sync)

    error(call(api, "POST", f"/api/v1/experiments/{job_id}/cancel"), 409, "job_not_cancellable")
    error(call(api, "POST", f"/api/v1/experiments/{job_id}/resume"), 409, "job_not_resumable")
    error(call(api, "GET", "/api/v1/experiments/j-nope"), 404, "job_not_found")
    error(call(api, "GET", "/api/v1/experiments/j-nope/artifact"), 404, "job_not_found")


def test_the_api_refuses_experiments_it_cannot_run_or_verify(tmp_path):
    from api.product import build_api
    from tests.test_product_api import call, error

    api = build_api(tmp_path / "data", runtime=_runtime)
    version, splits = _register(api)
    body = _create_body(version, splits)
    other = {**version["spec"], "dataset_version": 2}  # not the version the body names
    foreign = {**body, "contract": {**body["contract"], "dataset": other}}
    error(call(api, "POST", "/api/v1/experiments", json_body=foreign), 422, "invalid_experiment")
    wrong_splits = {**body, "splits_hash": "0" * 64}
    error(call(api, "POST", "/api/v1/experiments", json_body=wrong_splits), 404, "splits_not_found")
    capped = json.loads(json.dumps(body))
    capped["plan"]["budget"]["max_cost"] = 1.0  # no pricing: unenforceable, refused up front
    error(call(api, "POST", "/api/v1/experiments", json_body=capped), 422, "invalid_experiment")
    sneaky = {**body, "synthetic": False}  # the server decides; the client cannot claim it
    error(call(api, "POST", "/api/v1/experiments", json_body=sneaky), 400, "invalid_request")
    assert call(api, "GET", "/api/v1/experiments").body == {"jobs": []}

    bare = build_api(tmp_path / "bare")  # no model registry: nothing can execute here
    v2, s2 = _register(bare)
    error(
        call(bare, "POST", "/api/v1/experiments", json_body=_create_body(v2, s2)),
        503,
        "experiment_backend_unavailable",
    )


def test_a_restarted_server_recovers_and_finishes_its_jobs(tmp_path):
    from api.product import build_api, start_worker
    from tests.test_product_api import call

    first = build_api(tmp_path / "data", runtime=_runtime)  # never starts a worker: "crashes"
    version, splits = _register(first)
    job_id = call(
        first, "POST", "/api/v1/experiments", json_body=_create_body(version, splits)
    ).body["job_id"]

    second = build_api(tmp_path / "data", runtime=_runtime)
    worker = start_worker(second)
    assert worker is not None
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = call(second, "GET", f"/api/v1/experiments/{job_id}").body["state"]
            if state == "COMPLETED":
                break
            time.sleep(0.05)
    finally:
        worker.stop()
    assert state == "COMPLETED"
    assert call(second, "GET", f"/api/v1/experiments/{job_id}/artifact").status == 200


def test_job_views_are_strict_json(tmp_path):
    proc = Process(tmp_path / "jobs.sqlite3", Runtime(Backend()), Clock())
    job_id = proc.jobs.create(definition(seeds=(0,))).job_id
    proc.run()
    json.dumps(proc.jobs.get(job_id).model_dump(mode="json"), allow_nan=False)
    json.dumps(proc.jobs.list().model_dump(mode="json"), allow_nan=False)
