"""Durable experiment jobs: SQLite persistence for definitions, work units, checkpoints and the
write-ahead log of workflow-run attempts.

Same conventions as ``store.datasets``: stdlib ``sqlite3``, WAL, ``synchronous=FULL``, one
``BEGIN IMMEDIATE`` transaction per write, a fresh connection per operation (safe across threads
and processes). The engine that drives these records is ``experiments.jobs``; this module knows
nothing about optimizers or ledgers - it stores JSON documents and enforces the state machine,
leases and fencing.

Records

    experiment_jobs         one row per job: the immutable definition + bound identity (JSON),
                            the derived job state, a cancellation request, the final artifact
    experiment_units        one row per (strategy, seed) work unit: state, lease, fence token,
                            the completed strategy-run record
    experiment_checkpoints  append-only progress of a unit (``run_strategy``'s checkpoint events)
    experiment_attempts     append-only write-ahead log of workflow-run attempts: a row is
                            written STARTED before the model is invoked and COMPLETED (result +
                            measured usage) atomically after it returns

State machine (``JobState``; derived from the units on every write, never set directly)

    PENDING -> RUNNING -> COMPLETED
       |          |-> CANCEL_REQUESTED -> CANCELLED
       |          |-> FAILED
       |          '-> INTERRUPTED -> RUNNING (resume)
       '-> CANCELLED

Leases. A worker owns a unit through ``claim``: ``lease_owner`` + ``lease_until`` + a ``fence``
token that increments on every claim. Every unit write the worker makes is conditional on its
(owner, fence) in the same transaction, so a worker whose lease was taken over can no longer
start an attempt, append a checkpoint or finish the unit - two workers never execute the same
unit. A lease is renewed by every fenced write and by the worker's heartbeat. A completed
attempt's result is recorded by the worker that started it even after it lost the lease: it is
a fact about money already spent, and recording it is what resolves the attempt.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from store.datasets import Conflict, IntegrityViolation, RepositoryError, _Conn


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL = frozenset({JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED})


class UnitState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class Reason(StrEnum):
    """Why a unit (or job) is where it is. Stable strings: part of the API."""

    PROCESS_LOST = "process_lost"  # its worker died; nothing in flight: safe to resume
    AMBIGUOUS_ATTEMPT = "ambiguous_attempt"  # a model call started, its result was never stored
    RESUME_REQUESTED = "resume_requested"  # explicitly resumed; claimable again
    RUNTIME_UNAVAILABLE = "runtime_unavailable"  # this process could not bind the job's runtime
    CANCELLED = "cancelled"  # the job was cancelled
    JOB_FAILED = "job_failed"  # another unit of the job failed
    REPLAY_DIVERGED = "replay_diverged"  # re-execution disagreed with the persisted progress
    MODEL_UNAVAILABLE = "model_unavailable"
    EXPERIMENT_ERROR = "experiment_error"  # an invariant of the experiment broke (fail closed)
    RESERVATION_OVERFLOW = "reservation_overflow"
    PRICING_ERROR = "pricing_error"
    RUNNER_ERROR = "runner_error"  # the workflow runner / evaluator raised
    ASSEMBLY_FAILED = "assembly_failed"


# A worker may claim a unit in these states without anyone asking (automatic recovery).
AUTO_RESUMABLE = frozenset({Reason.PROCESS_LOST, Reason.RESUME_REQUESTED})


class AttemptState(StrEnum):
    STARTED = "STARTED"  # write-ahead: the model may have been invoked
    COMPLETED = "COMPLETED"  # result + measured usage stored atomically
    ERRORED = "ERRORED"  # the runner raised; the unit failed


class LeaseLost(RepositoryError):
    """This worker no longer owns the unit (another worker claimed it, or it was stopped)."""


class JobStopped(Exception):
    """The job was cancelled or failed: no new workflow run may start."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class Claim:
    job_id: str
    strategy: str
    seed: int
    owner: str
    fence: int

    @property
    def unit(self) -> tuple[str, str, int]:
        return (self.job_id, self.strategy, self.seed)


@dataclass(frozen=True)
class UnitRow:
    job_id: str
    strategy: str
    seed: int
    position: int
    run_id: str
    state: UnitState
    reason: str | None
    detail: str | None
    lease_owner: str | None
    lease_until: float | None
    fence: int
    updated_at: str
    record_json: str | None


@dataclass(frozen=True)
class JobRow:
    job_id: str
    created_at: str
    updated_at: str
    state: JobState
    reason: str | None
    detail: str | None
    cancel_requested_at: str | None
    definition_hash: str
    definition_json: str
    identity_json: str
    artifact_json: str | None
    units: tuple[UnitRow, ...]


@dataclass(frozen=True)
class CheckpointRow:
    seq: int
    kind: str
    evaluation: int | None
    round: int | None
    payload_json: str


@dataclass(frozen=True)
class AttemptRow:
    attempt_id: str
    seq: int
    genome_hash: str
    genome_json: str
    task_id: str
    trial: int
    run_seed: int
    attempt: int
    state: AttemptState
    owner: str
    fence: int
    resolution: str | None
    result_json: str | None
    entry_json: str | None
    timing_json: str | None
    error: str | None


@dataclass(frozen=True)
class NewAttempt:
    attempt_id: str
    genome_hash: str
    genome_json: str
    task_id: str
    trial: int
    run_seed: int
    attempt: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_jobs (
    job_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    state               TEXT NOT NULL,
    reason              TEXT,
    detail              TEXT,
    cancel_requested_at TEXT,
    failed_reason       TEXT,              -- a job-level failure (e.g. assembly), not a unit's
    definition_hash     TEXT NOT NULL,
    definition_json     TEXT NOT NULL,     -- immutable
    identity_json       TEXT NOT NULL,     -- immutable: problem / protocol / contract / run ids
    artifact_json       TEXT               -- written once, only when every unit completed
);
CREATE TABLE IF NOT EXISTS experiment_units (
    job_id      TEXT    NOT NULL REFERENCES experiment_jobs(job_id),
    strategy    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    position    INTEGER NOT NULL,         -- order of the record in the artifact
    run_id      TEXT    NOT NULL,         -- deterministic strategy-run identity
    state       TEXT    NOT NULL,
    reason      TEXT,
    detail      TEXT,
    lease_owner TEXT,
    lease_until REAL,
    fence       INTEGER NOT NULL DEFAULT 0,
    next_seq    INTEGER NOT NULL DEFAULT 0,   -- shared by checkpoints and attempts
    updated_at  TEXT    NOT NULL,
    record_json TEXT,                     -- the strategy-run record, when COMPLETED
    PRIMARY KEY (job_id, strategy, seed)
);
CREATE TABLE IF NOT EXISTS experiment_checkpoints (
    job_id       TEXT    NOT NULL,
    strategy     TEXT    NOT NULL,
    seed         INTEGER NOT NULL,
    seq          INTEGER NOT NULL,
    kind         TEXT    NOT NULL,
    evaluation   INTEGER,
    round        INTEGER,
    payload_json TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    PRIMARY KEY (job_id, strategy, seed, seq),
    FOREIGN KEY (job_id, strategy, seed) REFERENCES experiment_units(job_id, strategy, seed)
);
CREATE TABLE IF NOT EXISTS experiment_attempts (
    attempt_id  TEXT    PRIMARY KEY,      -- deterministic: job, unit run, genome, row, trial, n
    job_id      TEXT    NOT NULL,
    strategy    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    genome_hash TEXT    NOT NULL,
    genome_json TEXT    NOT NULL,
    task_id     TEXT    NOT NULL,
    trial       INTEGER NOT NULL,
    run_seed    INTEGER NOT NULL,
    attempt     INTEGER NOT NULL,
    state       TEXT    NOT NULL CHECK (state IN ('STARTED', 'COMPLETED', 'ERRORED')),
    owner       TEXT    NOT NULL,
    fence       INTEGER NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    resolution  TEXT,                     -- 'executed' | 'replay_proof'
    result_json TEXT,                     -- EvaluatedRun
    entry_json  TEXT,                     -- measured (priced) usage, as the ledger measures it
    timing_json TEXT,
    error       TEXT,
    FOREIGN KEY (job_id, strategy, seed) REFERENCES experiment_units(job_id, strategy, seed)
);
CREATE INDEX IF NOT EXISTS attempts_by_unit ON experiment_attempts(job_id, strategy, seed, seq);
CREATE INDEX IF NOT EXISTS units_by_state ON experiment_units(state);
"""

_UNIT_COLS = (
    "job_id, strategy, seed, position, run_id, state, reason, detail, lease_owner, lease_until, "
    "fence, updated_at, record_json"
)
_ATTEMPT_COLS = (
    "attempt_id, seq, genome_hash, genome_json, task_id, trial, run_seed, attempt, state, owner, "
    "fence, resolution, result_json, entry_json, timing_json, error"
)


def derive_job_state(
    *,
    has_artifact: bool,
    failed: bool,
    cancel_requested: bool,
    units: Sequence[UnitState],
) -> JobState:
    """The job's state from its units: the single rule, applied on every write."""
    if has_artifact:
        return JobState.COMPLETED
    if failed or UnitState.FAILED in units:
        return JobState.FAILED
    if cancel_requested:
        return JobState.CANCEL_REQUESTED if UnitState.RUNNING in units else JobState.CANCELLED
    if UnitState.RUNNING in units:
        return JobState.RUNNING
    if UnitState.INTERRUPTED in units:
        return JobState.INTERRUPTED
    if UnitState.COMPLETED in units:
        return JobState.RUNNING  # some units done, the rest waiting for a worker (or assembly)
    return JobState.PENDING


class SQLiteJobStore:
    def __init__(self, path: Path | str, clock: Callable[[], str] = utc_now) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._write(conn, lambda c: c.executescript_safe(SCHEMA))
            row = conn.execute("SELECT value FROM meta WHERE key='jobs_schema_version'").fetchone()
            if row is None:
                self._write(
                    conn,
                    lambda c: c.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES ('jobs_schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    ),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise RepositoryError(f"unsupported job store schema version {row[0]}")

    @contextmanager
    def _connect(self) -> Iterator[_Conn]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            yield _Conn(conn)
        finally:
            conn.close()

    def _tx(self, fn: Callable[[_Conn], Any]) -> Any:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return out

    @staticmethod
    def _write(conn: _Conn, fn: Any) -> Any:
        conn.execute("BEGIN IMMEDIATE")
        try:
            out = fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return out

    # -- jobs -------------------------------------------------------------------------------
    def create_job(
        self,
        job_id: str,
        definition_hash: str,
        definition_json: str,
        identity_json: str,
        units: Sequence[tuple[str, int, str]],  # (strategy, seed, run_id) in artifact order
    ) -> None:
        now = self.clock()

        def insert(c: _Conn) -> None:
            c.execute(
                "INSERT INTO experiment_jobs(job_id, created_at, updated_at, state, "
                "definition_hash, definition_json, identity_json) VALUES (?,?,?,?,?,?,?)",
                (
                    job_id,
                    now,
                    now,
                    JobState.PENDING.value,
                    definition_hash,
                    definition_json,
                    identity_json,
                ),
            )
            for position, (strategy, seed, run_id) in enumerate(units):
                c.execute(
                    "INSERT INTO experiment_units(job_id, strategy, seed, position, run_id, state,"
                    " updated_at) VALUES (?,?,?,?,?,?,?)",
                    (job_id, strategy, seed, position, run_id, UnitState.PENDING.value, now),
                )

        try:
            self._tx(insert)
        except sqlite3.IntegrityError as exc:
            raise Conflict(f"job {job_id}: {exc}") from None

    def get_job(self, job_id: str) -> JobRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id, created_at, updated_at, state, reason, detail, "
                "cancel_requested_at, definition_hash, definition_json, identity_json, "
                "artifact_json FROM experiment_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            units = self._units(conn, job_id)
        return JobRow(
            job_id=row[0],
            created_at=row[1],
            updated_at=row[2],
            state=JobState(row[3]),
            reason=row[4],
            detail=row[5],
            cancel_requested_at=row[6],
            definition_hash=row[7],
            definition_json=row[8],
            identity_json=row[9],
            artifact_json=row[10],
            units=units,
        )

    def list_job_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM experiment_jobs ORDER BY created_at, job_id"
            ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _units(conn: _Conn, job_id: str) -> tuple[UnitRow, ...]:
        rows = conn.execute(
            f"SELECT {_UNIT_COLS} FROM experiment_units WHERE job_id=? ORDER BY position",
            (job_id,),
        ).fetchall()
        return tuple(_unit(r) for r in rows)

    def checkpoints(self, job_id: str, strategy: str, seed: int) -> list[CheckpointRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, kind, evaluation, round, payload_json FROM experiment_checkpoints "
                "WHERE job_id=? AND strategy=? AND seed=? ORDER BY seq",
                (job_id, strategy, seed),
            ).fetchall()
        return [CheckpointRow(*r) for r in rows]

    def attempts(self, job_id: str, strategy: str, seed: int) -> list[AttemptRow]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ATTEMPT_COLS} FROM experiment_attempts "
                "WHERE job_id=? AND strategy=? AND seed=? ORDER BY seq",
                (job_id, strategy, seed),
            ).fetchall()
        return [_attempt(r) for r in rows]

    # -- derived job state ------------------------------------------------------------------
    def _refresh(self, c: _Conn, job_id: str, now_s: float | None = None) -> JobState:
        """Re-derive the job state; on reaching FAILED / CANCELLED, close every unit no live
        worker owns (a live worker closes its own unit at its next boundary)."""
        row = c.execute(
            "SELECT artifact_json IS NOT NULL, failed_reason, cancel_requested_at, state "
            "FROM experiment_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        has_artifact, failed_reason, cancel_at, old = row
        failed = failed_reason is not None or bool(
            c.execute(
                "SELECT 1 FROM experiment_units WHERE job_id=? AND state=?",
                (job_id, UnitState.FAILED.value),
            ).fetchone()
        )
        if not has_artifact and (failed or cancel_at is not None):
            reason = Reason.JOB_FAILED if failed else Reason.CANCELLED
            live = "" if now_s is None else " OR (state='RUNNING' AND lease_until < ?)"
            params: tuple[Any, ...] = (reason.value, self.clock(), job_id)
            c.execute(
                "UPDATE experiment_units SET state='CANCELLED', reason=?, lease_owner=NULL, "
                "lease_until=NULL, updated_at=? WHERE job_id=? AND (state IN "
                f"('PENDING', 'INTERRUPTED'){live})",
                params + ((now_s,) if now_s is not None else ()),
            )
        states = [
            UnitState(r[0])
            for r in c.execute(
                "SELECT state FROM experiment_units WHERE job_id=?", (job_id,)
            ).fetchall()
        ]
        new = derive_job_state(
            has_artifact=bool(has_artifact),
            failed=failed,
            cancel_requested=cancel_at is not None,
            units=states,
        )
        reason, detail = self._job_reason(c, job_id, new, failed_reason)
        c.execute(
            "UPDATE experiment_jobs SET state=?, reason=?, detail=?, updated_at=? WHERE job_id=?",
            (new.value, reason, detail, self.clock(), job_id),
        )
        if JobState(old) in TERMINAL and new != JobState(old):
            raise IntegrityViolation(f"job {job_id} left terminal state {old} for {new}")
        return new

    @staticmethod
    def _job_reason(
        c: _Conn, job_id: str, state: JobState, failed_reason: str | None
    ) -> tuple[str | None, str | None]:
        if state is JobState.FAILED and failed_reason is not None:
            detail = c.execute(
                "SELECT detail FROM experiment_jobs WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            return failed_reason, detail
        wanted = {
            JobState.FAILED: (UnitState.FAILED,),
            JobState.INTERRUPTED: (UnitState.INTERRUPTED,),
        }.get(state)
        if wanted is None:
            return (Reason.CANCELLED.value, None) if state is JobState.CANCELLED else (None, None)
        rows = c.execute(
            "SELECT reason, detail, strategy, seed FROM experiment_units WHERE job_id=? AND "
            "state=? ORDER BY position",
            (job_id, wanted[0].value),
        ).fetchall()
        # an ambiguous attempt dominates: it is the one an operator must look at
        rows.sort(key=lambda r: r[0] != Reason.AMBIGUOUS_ATTEMPT.value)
        reason, detail, strategy, seed = rows[0]
        return reason, f"{strategy}/seed {seed}: {detail}" if detail else f"{strategy}/seed {seed}"

    # -- claims and leases ------------------------------------------------------------------
    def claim(
        self, owner: str, now_s: float, lease_s: float, *, job_id: str | None = None
    ) -> Claim | None:
        """Take the next runnable unit: PENDING, auto-resumable INTERRUPTED, or RUNNING with an
        expired lease and nothing in flight (with something in flight it becomes INTERRUPTED /
        ambiguous_attempt instead and is skipped)."""

        def take(c: _Conn) -> Claim | None:
            rows = c.execute(
                "SELECT u.job_id, u.strategy, u.seed, u.state, u.reason, u.lease_until "
                "FROM experiment_units u JOIN experiment_jobs j ON j.job_id = u.job_id "
                "WHERE j.artifact_json IS NULL AND j.failed_reason IS NULL "
                "AND j.cancel_requested_at IS NULL AND j.state NOT IN ('FAILED') "
                "AND u.state IN ('PENDING', 'INTERRUPTED', 'RUNNING') "
                + ("AND u.job_id=? " if job_id is not None else "")
                + "ORDER BY j.created_at, j.job_id, u.position",
                (job_id,) if job_id is not None else (),
            ).fetchall()
            for jid, strategy, seed, state, reason, lease_until in rows:
                unit = (jid, strategy, seed)
                if state == UnitState.INTERRUPTED and reason not in AUTO_RESUMABLE:
                    continue
                if state == UnitState.RUNNING:
                    if lease_until is not None and lease_until >= now_s:
                        continue  # a live worker owns it
                    if self._interrupt_if_ambiguous(c, unit):
                        continue
                c.execute(
                    "UPDATE experiment_units SET state='RUNNING', reason=NULL, detail=NULL, "
                    "lease_owner=?, lease_until=?, fence=fence+1, updated_at=? "
                    "WHERE job_id=? AND strategy=? AND seed=?",
                    (owner, now_s + lease_s, self.clock(), *unit),
                )
                (fence,) = c.execute(
                    "SELECT fence FROM experiment_units WHERE job_id=? AND strategy=? AND seed=?",
                    unit,
                ).fetchone()
                self._refresh(c, jid)
                return Claim(jid, strategy, seed, owner, fence)
            return None

        return self._tx(take)

    def _interrupt_if_ambiguous(self, c: _Conn, unit: tuple[str, str, int]) -> bool:
        n = c.execute(
            "SELECT COUNT(*) FROM experiment_attempts WHERE job_id=? AND strategy=? AND seed=? "
            "AND state='STARTED'",
            unit,
        ).fetchone()[0]
        if not n:
            return False
        c.execute(
            "UPDATE experiment_units SET state='INTERRUPTED', reason=?, detail=?, "
            "lease_owner=NULL, lease_until=NULL, updated_at=? "
            "WHERE job_id=? AND strategy=? AND seed=?",
            (
                Reason.AMBIGUOUS_ATTEMPT.value,
                f"{n} model call(s) started without a stored result; their spend is unknown",
                self.clock(),
                *unit,
            ),
        )
        self._refresh(c, unit[0])
        return True

    def interrupt_stale(self, now_s: float) -> list[tuple[str, str, int, Reason]]:
        """Recovery: every RUNNING unit whose lease expired becomes INTERRUPTED - ambiguous if
        an attempt is in flight, else process_lost (which any worker may resume)."""

        def scan(c: _Conn) -> list[tuple[str, str, int, Reason]]:
            out = []
            rows = c.execute(
                "SELECT job_id, strategy, seed FROM experiment_units "
                "WHERE state='RUNNING' AND (lease_until IS NULL OR lease_until < ?) "
                "ORDER BY job_id, position",
                (now_s,),
            ).fetchall()
            for unit in rows:
                if self._interrupt_if_ambiguous(c, unit):
                    out.append((*unit, Reason.AMBIGUOUS_ATTEMPT))
                    continue
                c.execute(
                    "UPDATE experiment_units SET state='INTERRUPTED', reason=?, detail=NULL, "
                    "lease_owner=NULL, lease_until=NULL, updated_at=? "
                    "WHERE job_id=? AND strategy=? AND seed=?",
                    (Reason.PROCESS_LOST.value, self.clock(), *unit),
                )
                self._refresh(c, unit[0])
                out.append((*unit, Reason.PROCESS_LOST))
            return out

        return self._tx(scan)

    def _guard(self, c: _Conn, claim: Claim, now_s: float, lease_s: float) -> None:
        cur = c.execute(
            "UPDATE experiment_units SET lease_until=?, updated_at=? WHERE job_id=? AND "
            "strategy=? AND seed=? AND state='RUNNING' AND lease_owner=? AND fence=?",
            (now_s + lease_s, self.clock(), *claim.unit, claim.owner, claim.fence),
        )
        if cur.rowcount != 1:
            raise LeaseLost(f"{claim.owner} no longer owns {claim.strategy}/seed {claim.seed}")

    @staticmethod
    def _stop_reason(c: _Conn, job_id: str) -> Reason | None:
        cancel_at, failed_reason, state = c.execute(
            "SELECT cancel_requested_at, failed_reason, state FROM experiment_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if cancel_at is not None:
            return Reason.CANCELLED
        if failed_reason is not None or state == JobState.FAILED:
            return Reason.JOB_FAILED
        return None

    def renew(self, claim: Claim, now_s: float, lease_s: float) -> None:
        self._tx(lambda c: self._guard(c, claim, now_s, lease_s))

    def stop_reason(self, job_id: str) -> Reason | None:
        with self._connect() as conn:
            return self._stop_reason(conn, job_id)

    # -- write-ahead attempts ---------------------------------------------------------------
    def start_attempt(
        self, claim: Claim, attempt: NewAttempt, now_s: float, lease_s: float
    ) -> None:
        """Record that a model call is about to happen. Refused (nothing written, so nothing
        may be called) unless this worker still owns the unit and the job is not stopping."""

        def start(c: _Conn) -> None:
            self._guard(c, claim, now_s, lease_s)
            stop = self._stop_reason(c, claim.job_id)
            if stop is not None:
                raise JobStopped(stop)
            seq = self._next_seq(c, claim)
            try:
                c.execute(
                    "INSERT INTO experiment_attempts(attempt_id, job_id, strategy, seed, seq, "
                    "genome_hash, genome_json, task_id, trial, run_seed, attempt, state, owner, "
                    "fence, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,'STARTED',?,?,?)",
                    (
                        attempt.attempt_id,
                        *claim.unit,
                        seq,
                        attempt.genome_hash,
                        attempt.genome_json,
                        attempt.task_id,
                        attempt.trial,
                        attempt.run_seed,
                        attempt.attempt,
                        claim.owner,
                        claim.fence,
                        self.clock(),
                    ),
                )
            except sqlite3.IntegrityError:
                raise Conflict(f"attempt {attempt.attempt_id} was already started") from None

        self._tx(start)

    def complete_attempt(
        self,
        claim: Claim,
        attempt_id: str,
        result_json: str,
        entry_json: str,
        timing_json: str | None,
    ) -> None:
        """Store the result + measured usage of an attempt this worker started (atomic)."""

        def complete(c: _Conn) -> None:
            cur = c.execute(
                "UPDATE experiment_attempts SET state='COMPLETED', resolution='executed', "
                "result_json=?, entry_json=?, timing_json=?, finished_at=? WHERE attempt_id=? "
                "AND state='STARTED' AND owner=? AND fence=?",
                (
                    result_json,
                    entry_json,
                    timing_json,
                    self.clock(),
                    attempt_id,
                    claim.owner,
                    claim.fence,
                ),
            )
            if cur.rowcount != 1:
                raise IntegrityViolation(f"attempt {attempt_id} is not in flight for this worker")

        self._tx(complete)

    def error_attempt(self, claim: Claim, attempt_id: str, error: str) -> None:
        self._tx(
            lambda c: c.execute(
                "UPDATE experiment_attempts SET state='ERRORED', error=?, finished_at=? "
                "WHERE attempt_id=? AND state='STARTED' AND owner=? AND fence=?",
                (error[:2000], self.clock(), attempt_id, claim.owner, claim.fence),
            )
        )

    def resolve_attempt(
        self, attempt_id: str, result_json: str, entry_json: str, resolution: str
    ) -> None:
        """Settle an ambiguous (STARTED) attempt with a result proven by other means."""

        def resolve(c: _Conn) -> None:
            cur = c.execute(
                "UPDATE experiment_attempts SET state='COMPLETED', resolution=?, result_json=?, "
                "entry_json=?, finished_at=? WHERE attempt_id=? AND state='STARTED'",
                (resolution, result_json, entry_json, self.clock(), attempt_id),
            )
            if cur.rowcount != 1:
                raise Conflict(f"attempt {attempt_id} is not in flight")

        self._tx(resolve)

    @staticmethod
    def _next_seq(c: _Conn, claim: Claim) -> int:
        (seq,) = c.execute(
            "SELECT next_seq FROM experiment_units WHERE job_id=? AND strategy=? AND seed=?",
            claim.unit,
        ).fetchone()
        c.execute(
            "UPDATE experiment_units SET next_seq=? WHERE job_id=? AND strategy=? AND seed=?",
            (seq + 1, *claim.unit),
        )
        return seq

    # -- checkpoints ------------------------------------------------------------------------
    def append_checkpoint(
        self,
        claim: Claim,
        kind: str,
        evaluation: int | None,
        round_: int | None,
        payload_json: str,
        now_s: float,
        lease_s: float,
    ) -> Reason | None:
        """Append a progress checkpoint (fenced). Returns why the job is stopping, if it is."""

        def append(c: _Conn) -> Reason | None:
            self._guard(c, claim, now_s, lease_s)
            seq = self._next_seq(c, claim)
            c.execute(
                "INSERT INTO experiment_checkpoints(job_id, strategy, seed, seq, kind, "
                "evaluation, round, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (*claim.unit, seq, kind, evaluation, round_, payload_json, self.clock()),
            )
            return self._stop_reason(c, claim.job_id)

        return self._tx(append)

    # -- unit / job transitions -------------------------------------------------------------
    def finish_unit(
        self,
        claim: Claim,
        state: UnitState,
        reason: Reason | None = None,
        detail: str | None = None,
        record_json: str | None = None,
    ) -> JobState:
        """Close the claimed unit (fenced) and re-derive the job state."""
        if (state is UnitState.COMPLETED) != (record_json is not None):
            raise ValueError("exactly a COMPLETED unit carries its record")
        if state in (UnitState.PENDING, UnitState.RUNNING):
            raise ValueError(f"a unit cannot be finished as {state}")

        def finish(c: _Conn) -> JobState:
            cur = c.execute(
                "UPDATE experiment_units SET state=?, reason=?, detail=?, record_json=?, "
                "lease_owner=NULL, lease_until=NULL, updated_at=? WHERE job_id=? AND strategy=? "
                "AND seed=? AND state='RUNNING' AND lease_owner=? AND fence=?",
                (
                    state.value,
                    reason.value if reason is not None else None,
                    (detail or None) and detail[:2000],
                    record_json,
                    self.clock(),
                    *claim.unit,
                    claim.owner,
                    claim.fence,
                ),
            )
            if cur.rowcount != 1:
                raise LeaseLost(f"{claim.owner} no longer owns {claim.strategy}/seed {claim.seed}")
            return self._refresh(c, claim.job_id)

        return self._tx(finish)

    def request_cancel(self, job_id: str, now_s: float) -> JobState:
        """Persist the cancellation request (idempotent) and close every unit no live worker
        owns. Refused for a terminal job."""

        def cancel(c: _Conn) -> JobState:
            row = c.execute(
                "SELECT state, cancel_requested_at FROM experiment_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            if JobState(row[0]) in TERMINAL:
                raise Conflict(f"job {job_id} is already {row[0]}")
            if row[1] is None:
                c.execute(
                    "UPDATE experiment_jobs SET cancel_requested_at=? WHERE job_id=?",
                    (self.clock(), job_id),
                )
            return self._refresh(c, job_id, now_s)

        return self._tx(cancel)

    def mark_resumable(self, job_id: str) -> JobState:
        """Explicit resume: every INTERRUPTED unit becomes claimable again. The caller must
        first have established that nothing is in flight (no STARTED attempt)."""

        def resume(c: _Conn) -> JobState:
            n = c.execute(
                "SELECT COUNT(*) FROM experiment_attempts a JOIN experiment_units u ON "
                "a.job_id=u.job_id AND a.strategy=u.strategy AND a.seed=u.seed WHERE "
                "u.job_id=? AND u.state='INTERRUPTED' AND a.state='STARTED'",
                (job_id,),
            ).fetchone()[0]
            if n:
                raise Conflict(f"job {job_id} still has {n} unresolved in-flight attempt(s)")
            c.execute(
                "UPDATE experiment_units SET reason=?, detail=NULL, updated_at=? "
                "WHERE job_id=? AND state='INTERRUPTED'",
                (Reason.RESUME_REQUESTED.value, self.clock(), job_id),
            )
            return self._refresh(c, job_id)

        return self._tx(resume)

    def fail_job(self, job_id: str, reason: Reason, detail: str) -> JobState:
        def fail(c: _Conn) -> JobState:
            c.execute(
                "UPDATE experiment_jobs SET failed_reason=?, detail=? WHERE job_id=? AND "
                "artifact_json IS NULL",
                (reason.value, detail[:2000], job_id),
            )
            return self._refresh(c, job_id)

        return self._tx(fail)

    def put_artifact(self, job_id: str, artifact_json: str) -> bool:
        """Store the final artifact once - only if every unit completed and the job was neither
        cancelled nor failed. ``False``: not stored (already there, or not allowed)."""

        def put(c: _Conn) -> bool:
            open_units = c.execute(
                "SELECT COUNT(*) FROM experiment_units WHERE job_id=? AND state != 'COMPLETED'",
                (job_id,),
            ).fetchone()[0]
            if open_units:
                return False
            cur = c.execute(
                "UPDATE experiment_jobs SET artifact_json=? WHERE job_id=? AND artifact_json IS "
                "NULL AND cancel_requested_at IS NULL AND failed_reason IS NULL AND state != ?",
                (artifact_json, job_id, JobState.FAILED.value),
            )
            if cur.rowcount == 1:
                self._refresh(c, job_id)
            return cur.rowcount == 1

        return self._tx(put)

    def jobs_awaiting_artifact(self) -> list[str]:
        """Jobs whose units all completed but whose artifact was never stored (e.g. the process
        died between the last unit and assembly)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT j.job_id FROM experiment_jobs j WHERE j.artifact_json IS NULL "
                "AND j.cancel_requested_at IS NULL AND j.failed_reason IS NULL "
                "AND j.state != 'FAILED' AND NOT EXISTS (SELECT 1 FROM experiment_units u "
                "WHERE u.job_id = j.job_id AND u.state != 'COMPLETED') ORDER BY j.created_at"
            ).fetchall()
        return [r[0] for r in rows]


def _unit(r: tuple[Any, ...]) -> UnitRow:
    return UnitRow(
        job_id=r[0],
        strategy=r[1],
        seed=r[2],
        position=r[3],
        run_id=r[4],
        state=UnitState(r[5]),
        reason=r[6],
        detail=r[7],
        lease_owner=r[8],
        lease_until=r[9],
        fence=r[10],
        updated_at=r[11],
        record_json=r[12],
    )


def _attempt(r: tuple[Any, ...]) -> AttemptRow:
    return AttemptRow(
        attempt_id=r[0],
        seq=r[1],
        genome_hash=r[2],
        genome_json=r[3],
        task_id=r[4],
        trial=r[5],
        run_seed=r[6],
        attempt=r[7],
        state=AttemptState(r[8]),
        owner=r[9],
        fence=r[10],
        resolution=r[11],
        result_json=r[12],
        entry_json=r[13],
        timing_json=r[14],
        error=r[15],
    )
