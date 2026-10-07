"""Durable experiment jobs: start -> persist progress -> crash -> restart -> resume exactly, with
no duplicate spend.

A job is one ``ExperimentPlan`` over one uploaded dataset (``TaskContract`` + ``DatasetSplits``),
persisted as an immutable ``ExperimentJobDefinition`` plus mutable state in ``store.jobs``. It is
made of independent work units, one per ``(strategy, seed)``. A unit is exactly one #23
``run_strategy`` call; when every unit has completed, the job's artifact is the existing
``assemble`` of their records - durable and synchronous execution share every line of
scientific code (optimizers, ledger, run identities, selection, curves).

How a unit survives a crash: deterministic re-execution over a write-ahead run log.

  * Every workflow run already has a deterministic identity (genome, row, trial, run seed); its
    n-th attempt (MODEL_ERROR retries) gets ``attempt_identity``. Before the model is invoked,
    a STARTED attempt row is committed (fenced by the worker's lease, refused once the job is
    cancelled or failed). When the run returns, its ``EvaluatedRun`` and measured, priced usage
    are committed atomically (COMPLETED). That is the checkpoint after every workflow run.
  * ``run_strategy`` reports its progress through its ``checkpoint`` hook: proposals (with the
    optimizer state), each admitted reservation, each settled candidate (candidate record, curve
    point, ledger totals, champion so far, optimizer state) and each pheromone/observe step.
    Each is appended to the unit's checkpoint log.
  * Resume runs ``run_strategy`` again from the start with the same plan, seed and optimizer.
    ``DurableEvaluate`` answers every attempt that has a COMPLETED row from that row - no model
    call - and ``Journal`` compares every checkpoint the re-execution produces with the stored
    one (clock-derived values aside). Because optimizers, the ledger and selection are pure
    functions of the results they are fed, the re-execution passes through exactly the stored
    states: same proposals, same pheromones / epoch / score board, same RNG call indices, same
    ledger. Once the log is exhausted the unit simply continues. Any disagreement fails closed
    (``replay_diverged``) before a single new call. A new model call is only allowed once the
    whole stored log has been reproduced.
  * Budget continuity follows: the ledger is rebuilt from the stored measured usage of every
    attempt (committed spend stays spent; settled reservations stay released; the unfinished
    candidate is re-reserved with the same complete reservation and its completed rows count),
    and the stored ledger totals at every settlement are checked. Process downtime never
    enters execution time: the ledger only ever sees the runtime-measured ``wall_time_s``.
  * An attempt that STARTED and has no stored result is ambiguous - the provider may have
    charged for it. It is never re-sent: the unit becomes INTERRUPTED/ambiguous_attempt and is
    not resumed automatically. ``resume`` succeeds only once every such attempt is resolved -
    its original worker stored the result late, or the runtime's ``replay_proof`` (e.g. a
    deterministic response cache) proves the result without a call.

There is no exactly-once claim beyond this: a crash between the provider charging and the
result being committed is reported (ambiguous), never hidden.

Cancellation is cooperative and durable: ``cancel`` commits CANCEL_REQUESTED first; from then on
``start_attempt`` refuses every new run and the next checkpoint stops the unit; results already
returned are still stored. Units no live worker owns are closed at once; the job becomes
CANCELLED when no unit is running. A cancelled job keeps all its records and is never assembled.

Concurrency: units are claimed through fenced leases (``store.jobs``); any number of workers or
processes can share one database, and no unit is ever executed by two of them at once.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from core.canonical import canonical_hash
from core.constraints import ConstraintChecker
from core.dataset import DatasetSplits
from core.genome import Genome
from core.models import AllowedModels
from core.results import EvaluatedRun
from core.run_contract import ContractSuite, ExecutionTask
from core.task_contract import TaskContract
from evaluation.fitness import FitnessFunction
from experiments.budget_ledger import (
    BudgetLedger,
    PricingError,
    PricingPolicy,
    ReservationOverflow,
    check_budget,
    run_reservation,
)
from experiments.learning_curves import EvaluateFn, RunWorkflowFn, search_tasks
from experiments.optimization_experiment import (
    ExperimentError,
    ExperimentPlan,
    ModelUnavailable,
    Strategy,
    assemble,
    check_models,
    heldout_evaluator,
    make_strategy,
    problem_identity,
    run_identity,
    run_strategy,
    searchable_evaluator,
    without_timing,
)
from store.datasets import Conflict
from store.jobs import (
    AttemptRow,
    AttemptState,
    CheckpointRow,
    Claim,
    JobRow,
    JobState,
    JobStopped,
    LeaseLost,
    NewAttempt,
    Reason,
    SQLiteJobStore,
    UnitRow,
    UnitState,
)

log = logging.getLogger("wynk.experiments.jobs")

JOB_SCHEMA = "wynk-experiment-job/1"
DEFAULT_LEASE_S = 60.0


# -- errors -------------------------------------------------------------------------------------
class JobError(Exception):
    code = "job_error"


class JobNotFound(JobError):
    code = "job_not_found"


class InvalidExperiment(JobError):
    """The definition cannot run under this runtime (refused before anything is stored)."""

    code = "invalid_experiment"


class BackendUnavailable(JobError):
    code = "experiment_backend_unavailable"


class NotResumable(JobError):
    code = "job_not_resumable"


class AmbiguousAttemptError(JobError):
    """Resume refused: a model call's outcome is unknown and nothing proves it."""

    code = "ambiguous_attempt"


class NotCancellable(JobError):
    code = "job_not_cancellable"


class ArtifactUnavailable(JobError):
    code = "job_not_completed"


class AmbiguousAttempt(Exception):
    """Re-execution reached an attempt that started but whose result was never stored."""


class ReplayDiverged(Exception):
    """Re-execution disagrees with the stored progress: stop before spending anything."""


class RuntimeMismatch(Exception):
    """The runtime bound now does not reproduce the identity the job was created with."""


# -- definition + runtime binding ---------------------------------------------------------------
class ExperimentJobDefinition(BaseModel):
    """Everything that decides WHAT a job computes. Immutable; hashed into the job identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_schema: Literal["wynk-experiment-job/1"] = JOB_SCHEMA
    contract: TaskContract
    splits: DatasetSplits
    plan: ExperimentPlan
    synthetic: bool  # True whenever the model or objective is a stand-in
    workers: PositiveInt = Field(default=1, le=64)  # concurrent runs per candidate (time only)
    provenance: dict[str, Any] | None = None

    @property
    def definition_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


ReplayProof = Callable[[AttemptRow], EvaluatedRun | None]


@dataclass(frozen=True)
class JobBinding:
    """The live objects one process uses to execute a job's units."""

    suite: ContractSuite
    evaluate: EvaluateFn
    checker: ConstraintChecker
    evaluator_version: str
    pricing: PricingPolicy | None = None
    models: AllowedModels | None = None
    # Proves an ambiguous attempt's result WITHOUT a model call (e.g. a deterministic cache);
    # ``None`` when it cannot. Never used to re-send anything.
    replay_proof: ReplayProof | None = None


@dataclass(frozen=True)
class HeldoutBinding:
    """What the held-out promotion gate (``experiments.promotion``) executes with: the final
    test rows and an evaluator bound to exactly their expected values. Built only when the gate
    opens; a ``JobBinding`` never holds test targets."""

    tasks: tuple[ExecutionTask, ...]
    evaluate: EvaluateFn
    evaluator_version: str
    replay_proof: ReplayProof | None = None


class JobRuntime(Protocol):
    """Binds a definition to a model backend in this process."""

    synthetic: bool  # True if this runtime's model/objective is a test double

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding: ...


class HeldoutRuntime(JobRuntime, Protocol):
    """A ``JobRuntime`` that can also bind the held-out (test) rows for the promotion gate."""

    def bind_heldout(self, definition: ExperimentJobDefinition) -> HeldoutBinding: ...


@dataclass(frozen=True)
class WorkflowBackend:
    run_workflow: RunWorkflowFn
    checker: ConstraintChecker
    pricing: PricingPolicy | None = None
    models: AllowedModels | None = None
    replay_proof: ReplayProof | None = None


@dataclass
class DatasetRuntime:
    """A ``JobRuntime`` over stored dataset bytes: ``data(content_hash)`` returns the verified
    bytes (e.g. ``BlobStore.get``), ``backend(definition)`` the workflow runner for its model.
    Only optimization + validation rows are bound to the evaluator (``searchable_evaluator``)."""

    data: Callable[[str], bytes]
    backend: Callable[[ExperimentJobDefinition], WorkflowBackend]
    synthetic: bool
    fitness: FitnessFunction | None = None

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        backend = self.backend(definition)
        suite, evaluate, version = searchable_evaluator(
            definition.contract,
            definition.splits,
            self.data(definition.contract.dataset.content_hash),
            backend.run_workflow,
            fitness=self.fitness,
        )
        return JobBinding(
            suite=suite,
            evaluate=evaluate,
            checker=backend.checker,
            evaluator_version=version,
            pricing=backend.pricing,
            models=backend.models,
            replay_proof=backend.replay_proof,
        )

    def bind_heldout(self, definition: ExperimentJobDefinition) -> HeldoutBinding:
        """Only the final test rows (and their expected values): ``heldout_evaluator``."""
        backend = self.backend(definition)
        tasks, evaluate, version = heldout_evaluator(
            definition.contract,
            definition.splits,
            self.data(definition.contract.dataset.content_hash),
            backend.run_workflow,
            fitness=self.fitness,
        )
        return HeldoutBinding(tasks, evaluate, version, backend.replay_proof)


def job_identity(definition: ExperimentJobDefinition, binding: JobBinding) -> dict[str, Any]:
    """The identity a job is created with and every later binding must reproduce. Fails closed
    (like ``run_optimization_experiment``) before anything runs."""
    plan = definition.plan
    if binding.models is not None:
        check_models(plan, binding.models)
    check_budget(plan.budget, binding.pricing)
    run_reservation(plan.budget, binding.suite.policy, binding.pricing, plan.expected_model_hash)
    search_tasks(binding.suite)
    if binding.suite.policy.contract_hash != definition.contract.contract_hash:
        raise RuntimeMismatch("the runtime bound a different contract")
    problem = problem_identity(
        binding.suite, plan, evaluator_version=binding.evaluator_version, pricing=binding.pricing
    )
    units = []
    for seed in plan.seeds:  # the artifact order of ``run_optimization_experiment``
        for strategy in plan.strategies:
            ident = run_identity(problem, plan, strategy, make_strategy(strategy, plan), seed)
            units.append({"strategy": strategy.value, "seed": seed, "run_id": ident["run_id"]})
    return {
        "definition_hash": definition.definition_hash,
        "contract_hash": definition.contract.contract_hash,
        "protocol_hash": canonical_hash(plan.protocol()),
        "budget_hash": plan.budget.identity_hash,
        "problem_id": problem["problem_id"],
        "experiment_id": canonical_hash({"problem": problem, "plan": plan.identity_dump()}),
        "evaluator_version": binding.evaluator_version,
        "pricing": binding.pricing.identity if binding.pricing is not None else None,
        "model_hash": plan.expected_model_hash,
        "synthetic": definition.synthetic,
        "units": units,
    }


def attempt_identity(
    job_id: str, run_id: str, genome_hash: str, task_id: str, trial: int, seed: int, n: int
) -> str:
    """Deterministic identity of the ``n``-th attempt of one workflow run of one unit."""
    return canonical_hash([job_id, run_id, genome_hash, task_id, trial, seed, n])


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _comparable(payload: Any) -> Any:
    """A checkpoint payload as stored (JSON round trip), minus clock-derived values."""
    return without_timing(json.loads(_json(payload)))


# -- ledger arithmetic over stored facts --------------------------------------------------------
_COUNTERS = ("model_calls", "prompt_tokens", "completion_tokens", "tokens", "tool_calls", "retries")


def zero_totals(priced: bool) -> dict[str, Any]:
    return {
        "candidate_evaluations": 0,
        "workflow_runs": 0,
        **dict.fromkeys(_COUNTERS, 0),
        "wall_time_s": 0.0,
        "cost": 0.0 if priced else None,
    }


def settle_partial(totals: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]):
    """``totals`` plus the measured usage of an unfinished candidate's completed runs: the same
    arithmetic as ``BudgetLedger.settle(entries, admitted=False)`` - real spend is committed,
    the candidate is not counted, its reservation is released."""
    out = dict(totals)
    for key in _COUNTERS:
        out[key] = totals[key] + sum(e[key] for e in entries)
    out["wall_time_s"] = totals["wall_time_s"] + sum((e["wall_time_s"] or 0.0) for e in entries)
    if totals["cost"] is not None:
        out["cost"] = totals["cost"] + sum((e["cost"] or 0.0) for e in entries)
    out["workflow_runs"] = totals["workflow_runs"] + len(entries)
    return out


# -- re-execution over the write-ahead log ------------------------------------------------------
@dataclass
class Lease:
    store: SQLiteJobStore
    claim: Claim
    lease_s: float
    clock: Callable[[], float]


class Journal:
    """``run_strategy``'s checkpoint hook: verifies re-executed progress against the stored log,
    then appends new progress (fenced) and stops the unit at the first boundary after a
    cancellation request (or a sibling unit's failure)."""

    def __init__(self, lease: Lease, stored: Sequence[CheckpointRow]) -> None:
        self.lease = lease
        self.stored = list(stored)
        self.cursor = 0
        self.durable: DurableEvaluate | None = None

    @property
    def caught_up(self) -> bool:
        return self.cursor >= len(self.stored)

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.caught_up:
            row = self.stored[self.cursor]
            if row.kind != kind or _comparable(payload) != without_timing(
                json.loads(row.payload_json)
            ):
                raise ReplayDiverged(
                    f"checkpoint {row.seq} ({row.kind}) does not reproduce: re-execution "
                    f"produced {kind!r} with different content"
                )
            self.cursor += 1
            return
        if self.durable is not None and self.durable.unconsumed():
            raise ReplayDiverged(
                f"{len(self.durable.unconsumed())} stored run(s) were never reached by the "
                "re-execution"
            )
        lz = self.lease
        stop = lz.store.append_checkpoint(
            lz.claim,
            kind,
            payload.get("evaluation"),
            payload.get("round"),
            _json(payload),
            lz.clock(),
            lz.lease_s,
        )
        if stop is not None:
            raise JobStopped(stop)

    def finish(self) -> None:
        if not self.caught_up:
            raise ReplayDiverged(
                f"re-execution ended after {self.cursor} of {len(self.stored)} stored checkpoints"
            )


class DurableEvaluate:
    """``EvaluateFn`` that never runs a workflow twice: stored attempts are answered from the
    log; a new attempt is written ahead (STARTED) before the model is invoked and its result +
    measured usage committed atomically after."""

    def __init__(
        self,
        inner: EvaluateFn,
        lease: Lease,
        run_id: str,
        stored: Sequence[AttemptRow],
        journal: Journal,
        measure: Callable[[EvaluatedRun], dict[str, Any]],
    ) -> None:
        self._inner = inner
        self._lease = lease
        self._run_id = run_id
        self._stored = {a.attempt_id: a for a in stored}
        self._journal = journal
        self._measure = measure
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, int, int], int] = {}
        self._consumed: set[str] = set()
        self._timings: dict[tuple[str, str, int, int], tuple[float, float] | None] = {}
        self.reused = 0  # answered from the log
        self.executed = 0  # sent to the model by this process
        self._inner_timing = getattr(inner, "timing", None)
        if self._inner_timing is not None:  # expose timing only if the inner evaluator has it
            self.timing = self._timing
        journal.durable = self

    def unconsumed(self) -> set[str]:
        with self._lock:
            return set(self._stored) - self._consumed

    def _timing(self, genome: Genome, task: ExecutionTask, trial: int, seed: int):
        with self._lock:
            return self._timings.pop((genome.genome_hash, task.id, trial, seed), None)

    def __call__(self, genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
        key = (genome.genome_hash, task.id, trial, seed)
        with self._lock:
            n = self._counts.get(key, 0)
            self._counts[key] = n + 1
        lz = self._lease
        attempt_id = attempt_identity(lz.claim.job_id, self._run_id, *key, n)
        row = self._stored.get(attempt_id)
        if row is not None:
            if row.state is not AttemptState.COMPLETED:
                raise AmbiguousAttempt(f"attempt {attempt_id} ({task.id}) has no stored result")
            run = EvaluatedRun.model_validate_json(row.result_json or "")
            if self._measure(run) != json.loads(row.entry_json or "null"):
                # like ``store.datasets``: detects corruption / inconsistent edits, refuses them
                raise ReplayDiverged(f"stored run {attempt_id} disagrees with its recorded usage")
            timing = json.loads(row.timing_json) if row.timing_json else None
            with self._lock:
                self._consumed.add(attempt_id)
                self._timings[key] = tuple(timing) if timing is not None else None
                self.reused += 1
            return run
        if not self._journal.caught_up:
            raise ReplayDiverged(
                f"re-execution asked for a run that is not in the log ({task.id}, trial {trial})"
                " before reproducing the stored progress"
            )
        lz.store.start_attempt(
            lz.claim,
            NewAttempt(
                attempt_id=attempt_id,
                genome_hash=genome.genome_hash,
                genome_json=genome.canonical_json(),
                task_id=task.id,
                trial=trial,
                run_seed=seed,
                attempt=n,
            ),
            lz.clock(),
            lz.lease_s,
        )
        try:
            run = self._inner(genome, task, trial, seed)
        except Exception as exc:
            lz.store.error_attempt(lz.claim, attempt_id, f"{type(exc).__name__}: {exc}")
            raise
        timing = self._inner_timing(genome, task, trial, seed) if self._inner_timing else None
        lz.store.complete_attempt(
            lz.claim,
            attempt_id,
            run.model_dump_json(),
            _json(self._measure(run)),
            _json(list(timing)) if timing is not None else None,
        )
        with self._lock:
            self._timings[key] = timing
            self.executed += 1
        return run


class _Heartbeat:
    def __init__(self, lease: Lease, interval: float) -> None:
        self.lease, self.interval = lease, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wynk-job-lease")

    def _run(self) -> None:
        lz = self.lease
        while not self._stop.wait(self.interval):
            try:
                lz.store.renew(lz.claim, lz.clock(), lz.lease_s)
            except LeaseLost:
                return  # every later fenced write fails too
            except Exception:  # a transient database error: the next renewal retries
                log.exception("lease renewal failed for %s", lz.claim)

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _failure_reason(exc: BaseException) -> Reason:
    if isinstance(exc, ModelUnavailable):
        return Reason.MODEL_UNAVAILABLE
    if isinstance(exc, ReservationOverflow):
        return Reason.RESERVATION_OVERFLOW
    if isinstance(exc, PricingError):
        return Reason.PRICING_ERROR
    if isinstance(exc, ValueError):  # ExperimentError, ContractError, model identity, budget
        return Reason.EXPERIMENT_ERROR
    return Reason.RUNNER_ERROR


@dataclass
class UnitOutcome:
    claim: Claim
    state: UnitState
    reason: Reason | None = None
    reused: int = 0
    executed: int = 0


class JobWorker:
    """Claims and executes work units; ``recover`` at process start."""

    def __init__(
        self,
        store: SQLiteJobStore,
        runtime: JobRuntime,
        *,
        worker_id: str | None = None,
        lease_s: float = DEFAULT_LEASE_S,
        clock: Callable[[], float] = time.time,
        heartbeat: bool = True,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.worker_id = worker_id or f"w-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.lease_s = lease_s
        self.clock = clock
        self.heartbeat = heartbeat
        self._bindings: dict[str, JobBinding] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- recovery ---------------------------------------------------------------------------
    def recover(self) -> list[tuple[str, str, int, Reason]]:
        """Process start: every unit whose worker is gone becomes INTERRUPTED (ambiguous when a
        call was in flight - never resumed automatically - else claimable again), and jobs
        that finished every unit but never stored their artifact are assembled."""
        found = self.store.interrupt_stale(self.clock())
        for job_id in self.store.jobs_awaiting_artifact():
            self._assemble(job_id)
        return found

    # -- execution --------------------------------------------------------------------------
    def run_once(self, job_id: str | None = None) -> UnitOutcome | None:
        """Claim and execute one unit; ``None`` when nothing is runnable."""
        claim = self.store.claim(self.worker_id, self.clock(), self.lease_s, job_id=job_id)
        if claim is None:
            return None
        lease = Lease(self.store, claim, self.lease_s, self.clock)
        if not self.heartbeat:
            return self._execute(lease)
        with _Heartbeat(lease, self.lease_s / 3):
            return self._execute(lease)

    def run_until_idle(self, job_id: str | None = None) -> list[UnitOutcome]:
        out = []
        while (outcome := self.run_once(job_id)) is not None:
            out.append(outcome)
        for jid in self.store.jobs_awaiting_artifact():
            if job_id is None or jid == job_id:
                self._assemble(jid)
        return out

    def _binding(self, job: JobRow, definition: ExperimentJobDefinition) -> JobBinding:
        binding = self._bindings.get(job.job_id)
        if binding is None:
            binding = self.runtime.bind(definition)
            identity = json.loads(job.identity_json)
            if job_identity(definition, binding) != identity:
                raise RuntimeMismatch(
                    "this runtime does not reproduce the job's identity (dataset, contract, "
                    "evaluator, model, pricing or optimizer versions changed)"
                )
            self._bindings[job.job_id] = binding
        return binding

    def _execute(self, lease: Lease) -> UnitOutcome:
        claim, store = lease.claim, lease.store
        job = store.get_job(claim.job_id)
        assert job is not None
        unit = next(u for u in job.units if (u.strategy, u.seed) == (claim.strategy, claim.seed))
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        plan = definition.plan
        try:
            binding = self._binding(job, definition)
        except Exception as exc:
            log.warning("cannot bind job %s: %s", job.job_id, exc)
            return self._finish(lease, UnitState.INTERRUPTED, Reason.RUNTIME_UNAVAILABLE, f"{exc}")
        strategy = Strategy(claim.strategy)
        journal = Journal(lease, store.checkpoints(*claim.unit))
        per_run = run_reservation(
            plan.budget, binding.suite.policy, binding.pricing, plan.expected_model_hash
        )
        meter = BudgetLedger(plan.budget, binding.pricing, per_run)  # the ledger's own measure
        durable = DurableEvaluate(
            binding.evaluate,
            lease,
            unit.run_id,
            store.attempts(*claim.unit),
            journal,
            meter.measure,
        )

        def counts(state: UnitState, reason: Reason | None = None) -> UnitOutcome:
            return UnitOutcome(claim, state, reason, durable.reused, durable.executed)

        try:
            record = run_strategy(
                plan,
                strategy,
                claim.seed,
                binding.suite,
                durable,
                checker=binding.checker,
                evaluator_version=binding.evaluator_version,
                pricing=binding.pricing,
                optimizer=make_strategy(strategy, plan),
                workers=definition.workers,
                checkpoint=journal,
            )
            journal.finish()
            if durable.unconsumed():
                raise ReplayDiverged("stored runs were never reached by the re-execution")
            if record["run_id"] != unit.run_id:
                raise ReplayDiverged("the strategy run has a different identity than the unit")
        except LeaseLost:
            return counts(UnitState.RUNNING)  # someone else owns the unit now
        except JobStopped as stop:
            self._finish(lease, UnitState.CANCELLED, stop.reason)
            return counts(UnitState.CANCELLED, stop.reason)
        except AmbiguousAttempt as exc:
            self._finish(lease, UnitState.INTERRUPTED, Reason.AMBIGUOUS_ATTEMPT, str(exc))
            return counts(UnitState.INTERRUPTED, Reason.AMBIGUOUS_ATTEMPT)
        except ReplayDiverged as exc:
            self._finish(lease, UnitState.FAILED, Reason.REPLAY_DIVERGED, str(exc))
            return counts(UnitState.FAILED, Reason.REPLAY_DIVERGED)
        except Exception as exc:
            reason = _failure_reason(exc)
            log.warning("unit %s/%s of %s failed: %r", claim.strategy, claim.seed, job.job_id, exc)
            self._finish(lease, UnitState.FAILED, reason, f"{type(exc).__name__}: {exc}")
            return counts(UnitState.FAILED, reason)
        try:
            store.finish_unit(claim, UnitState.COMPLETED, record_json=_json(record))
        except LeaseLost:
            return counts(UnitState.RUNNING)
        self._assemble(job.job_id)
        return counts(UnitState.COMPLETED)

    def _finish(
        self, lease: Lease, state: UnitState, reason: Reason, detail: str | None = None
    ) -> UnitOutcome:
        try:
            lease.store.finish_unit(lease.claim, state, reason, detail)
        except LeaseLost:
            pass
        return UnitOutcome(lease.claim, state, reason)

    def _assemble(self, job_id: str) -> None:
        """When every unit completed: the existing ``assemble`` over their records."""
        job = self.store.get_job(job_id)
        if job is None or job.artifact_json is not None:
            return
        if any(u.state is not UnitState.COMPLETED for u in job.units):
            return
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        try:
            binding = self._binding(job, definition)
        except Exception as exc:
            log.warning("cannot bind job %s for assembly: %s", job_id, exc)
            return
        records = [json.loads(u.record_json or "") for u in job.units]  # artifact order
        try:
            artifact = assemble(
                definition.plan,
                binding.suite,
                records,
                evaluator_version=binding.evaluator_version,
                synthetic=definition.synthetic,
                pricing=binding.pricing,
                provenance=definition.provenance,
            )
            if artifact["experiment_id"] != json.loads(job.identity_json)["experiment_id"]:
                raise ExperimentError("the assembled artifact has a different experiment id")
        except ExperimentError as exc:
            self.store.fail_job(job_id, Reason.ASSEMBLY_FAILED, str(exc))
            return
        self.store.put_artifact(job_id, _json(artifact))

    # -- background loop (servers) ----------------------------------------------------------
    def start(self, poll_s: float = 1.0) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            try:
                self.recover()
            except Exception:
                log.exception("job recovery failed")
            while not self._stop.is_set():
                try:
                    if self.run_once() is None:
                        self._stop.wait(poll_s)
                except Exception:
                    log.exception("job worker iteration failed")
                    self._stop.wait(poll_s)

        self._thread = threading.Thread(target=loop, daemon=True, name="wynk-job-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


# -- query surface ------------------------------------------------------------------------------
class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AttemptCounts(_View):
    completed: int
    in_flight: int  # STARTED without a stored result (ambiguous once its worker is gone)
    errored: int
    reused_by_proof: int


class UnitView(_View):
    strategy: str
    seed: int
    run_id: str
    state: UnitState
    reason: str | None
    detail: str | None
    lease_owner: str | None
    rounds: int
    candidate_evaluations: int
    # Everything measured as spent (settled candidates + completed runs of an unfinished one).
    usage: dict[str, float | int | None]
    # The ledger at the last settlement, and the reservation an in-progress candidate holds.
    settled_usage: dict[str, float | int | None]
    open_reservation: dict[str, float | None] | None
    champion_genome_hash: str | None
    champion_validation_score: float | None
    curve: list[dict[str, Any]]
    stop_reason: str | None
    attempts: AttemptCounts


class JobSummary(_View):
    job_id: str
    state: JobState
    reason: str | None
    created_at: str
    updated_at: str
    experiment_id: str
    synthetic: bool
    artifact_available: bool


class JobView(_View):
    job_id: str
    state: JobState
    reason: str | None
    detail: str | None
    created_at: str
    updated_at: str
    cancel_requested_at: str | None
    definition_hash: str
    contract_hash: str
    protocol_hash: str
    problem_id: str
    experiment_id: str
    synthetic: bool
    budget: dict[str, Any]
    strategies: list[str]
    seeds: list[int]
    units: list[UnitView]
    artifact_available: bool


class JobList(_View):
    jobs: list[JobSummary]


class ArtifactView(_View):
    job_id: str
    experiment_id: str
    artifact: dict[str, Any]


def unit_view(
    unit: UnitRow, checkpoints: Sequence[CheckpointRow], attempts: Sequence[AttemptRow], priced
) -> UnitView:
    counts = AttemptCounts(
        completed=sum(a.state is AttemptState.COMPLETED for a in attempts),
        in_flight=sum(a.state is AttemptState.STARTED for a in attempts),
        errored=sum(a.state is AttemptState.ERRORED for a in attempts),
        reused_by_proof=sum(a.resolution == "replay_proof" for a in attempts),
    )
    common = {
        "strategy": unit.strategy,
        "seed": unit.seed,
        "run_id": unit.run_id,
        "state": unit.state,
        "reason": unit.reason,
        "detail": unit.detail,
        "lease_owner": unit.lease_owner,
        "attempts": counts,
    }
    if unit.record_json is not None:
        record = json.loads(unit.record_json)
        champ = record["champion"]
        return UnitView(
            **common,
            rounds=record["rounds"],
            candidate_evaluations=record["usage"]["candidate_evaluations"],
            usage=record["usage"],
            settled_usage=record["usage"],
            open_reservation=None,
            champion_genome_hash=champ["genome_hash"] if champ else None,
            champion_validation_score=champ["validation"]["score_mean"] if champ else None,
            curve=record["curve"],
            stop_reason=record["stop_reason"],
        )
    settled = [(c.seq, json.loads(c.payload_json)) for c in checkpoints if c.kind == "settled"]
    last_seq = settled[-1][0] if settled else -1
    base = settled[-1][1]["ledger"] if settled else zero_totals(priced)
    partial = [
        json.loads(a.entry_json or "")
        for a in attempts
        if a.state is AttemptState.COMPLETED and a.seq > last_seq
    ]
    reserved = [
        json.loads(c.payload_json)["reservation"]
        for c in checkpoints
        if c.kind == "reserved" and c.seq > last_seq
    ]
    curve = [p["curve_point"] for _, p in settled]
    rounds = sum(c.kind == "proposed" for c in checkpoints)
    return UnitView(
        **common,
        rounds=rounds,
        candidate_evaluations=base["candidate_evaluations"],
        usage=settle_partial(base, partial),
        settled_usage=base,
        open_reservation=reserved[-1] if reserved and unit.state is UnitState.RUNNING else None,
        champion_genome_hash=settled[-1][1]["champion"] if settled else None,
        champion_validation_score=curve[-1]["best_so_far_score"] if curve else None,
        curve=curve,
        stop_reason=None,
    )


@dataclass
class ExperimentJobs:
    """The job service: create / inspect / cancel / resume / artifact. ``runtime`` is ``None``
    in a process that cannot execute experiments (jobs are still inspectable and cancellable)."""

    store: SQLiteJobStore
    runtime: JobRuntime | None = None
    clock: Callable[[], float] = time.time
    _ids: Callable[[], str] = field(default=lambda: f"j-{uuid.uuid4().hex[:24]}")

    def _runtime(self) -> JobRuntime:
        if self.runtime is None:
            raise BackendUnavailable(
                "this Wynk server has no experiment backend configured (model registry)"
            )
        return self.runtime

    def create(self, definition: ExperimentJobDefinition) -> JobView:
        runtime = self._runtime()
        if runtime.synthetic and not definition.synthetic:
            raise InvalidExperiment("a synthetic runtime can only run synthetic experiments")
        try:
            binding = runtime.bind(definition)
            identity = job_identity(definition, binding)
        except (ValueError, RuntimeMismatch) as exc:
            raise InvalidExperiment(str(exc)) from exc
        job_id = self._ids()
        self.store.create_job(
            job_id,
            definition.definition_hash,
            definition.model_dump_json(),
            _json(identity),
            [(u["strategy"], u["seed"], u["run_id"]) for u in identity["units"]],
        )
        return self.get(job_id)

    def _job(self, job_id: str) -> JobRow:
        job = self.store.get_job(job_id)
        if job is None:
            raise JobNotFound(f"no experiment job {job_id}")
        return job

    def get(self, job_id: str) -> JobView:
        job = self._job(job_id)
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        identity = json.loads(job.identity_json)
        priced = identity["pricing"] is not None
        units = [
            unit_view(
                u,
                self.store.checkpoints(*_key(u)),
                self.store.attempts(*_key(u)),
                priced,
            )
            for u in job.units
        ]
        return JobView(
            job_id=job.job_id,
            state=job.state,
            reason=job.reason,
            detail=job.detail,
            created_at=job.created_at,
            updated_at=job.updated_at,
            cancel_requested_at=job.cancel_requested_at,
            definition_hash=job.definition_hash,
            contract_hash=identity["contract_hash"],
            protocol_hash=identity["protocol_hash"],
            problem_id=identity["problem_id"],
            experiment_id=identity["experiment_id"],
            synthetic=definition.synthetic,
            budget=definition.plan.budget.model_dump(mode="json"),
            strategies=[s.value for s in definition.plan.strategies],
            seeds=list(definition.plan.seeds),
            units=units,
            artifact_available=job.artifact_json is not None,
        )

    def list(self) -> JobList:
        out = []
        for job_id in self.store.list_job_ids():
            job = self._job(job_id)
            identity = json.loads(job.identity_json)
            out.append(
                JobSummary(
                    job_id=job.job_id,
                    state=job.state,
                    reason=job.reason,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    experiment_id=identity["experiment_id"],
                    synthetic=identity["synthetic"],
                    artifact_available=job.artifact_json is not None,
                )
            )
        return JobList(jobs=out)

    def cancel(self, job_id: str) -> JobView:
        self._job(job_id)
        try:
            self.store.request_cancel(job_id, self.clock())
        except Conflict as exc:
            raise NotCancellable(str(exc)) from None
        return self.get(job_id)

    def resume(self, job_id: str) -> JobView:
        """Make an INTERRUPTED job runnable again - only once nothing is ambiguous: every
        in-flight attempt must have a stored result or be proven by the runtime's
        ``replay_proof``. Nothing is ever re-sent to resolve one."""
        job = self._job(job_id)
        if job.state in (JobState.PENDING, JobState.RUNNING):
            return self.get(job_id)  # already runnable
        if job.state is not JobState.INTERRUPTED:
            raise NotResumable(f"job {job_id} is {job.state.value}; only INTERRUPTED resumes")
        for unit in job.units:
            if unit.state is not UnitState.INTERRUPTED:
                continue
            open_ = [a for a in self.store.attempts(*_key(unit)) if a.state is AttemptState.STARTED]
            if open_:
                self._prove(job, open_)
        try:
            self.store.mark_resumable(job_id)
        except Conflict as exc:
            raise AmbiguousAttemptError(str(exc)) from None
        return self.get(job_id)

    def _prove(self, job: JobRow, attempts: Sequence[AttemptRow]) -> None:
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        proof = None
        binding = None
        if self.runtime is not None:
            binding = self.runtime.bind(definition)
            if job_identity(definition, binding) != json.loads(job.identity_json):
                raise NotResumable("the configured runtime does not reproduce this job")
            proof = binding.replay_proof
        if proof is None or binding is None:
            raise AmbiguousAttemptError(
                f"{len(attempts)} model call(s) started without a stored result; their outcome "
                "and spend are unknown and no replay proof is available, so the job is not "
                "resumed (nothing is re-sent)"
            )
        plan = definition.plan
        per_run = run_reservation(
            plan.budget, binding.suite.policy, binding.pricing, plan.expected_model_hash
        )
        meter = BudgetLedger(plan.budget, binding.pricing, per_run)
        for a in attempts:
            run = proof(a)
            key = run.execution.key if run is not None else None
            if key is None or (key.genome_hash, key.task_id, key.trial, key.seed) != (
                a.genome_hash,
                a.task_id,
                a.trial,
                a.run_seed,
            ):
                raise AmbiguousAttemptError(
                    f"attempt {a.attempt_id} ({a.task_id}) cannot be proven without a model call"
                )
            self.store.resolve_attempt(
                a.attempt_id, run.model_dump_json(), _json(meter.measure(run)), "replay_proof"
            )

    def artifact(self, job_id: str) -> ArtifactView:
        job = self._job(job_id)
        if job.state is not JobState.COMPLETED or job.artifact_json is None:
            raise ArtifactUnavailable(f"job {job_id} is {job.state.value}, not COMPLETED")
        identity = json.loads(job.identity_json)
        return ArtifactView(
            job_id=job_id,
            experiment_id=identity["experiment_id"],
            artifact=json.loads(job.artifact_json),
        )


def _key(unit: UnitRow) -> tuple[str, str, int]:
    return (unit.job_id, unit.strategy, unit.seed)
