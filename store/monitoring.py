"""Production monitoring: SQLite persistence for quality feedback, versioned monitoring policies,
trigger decisions and the re-optimizations they start.

Same conventions as ``store.deployments``: stdlib ``sqlite3``, WAL, ``synchronous=FULL``, one
``BEGIN IMMEDIATE`` transaction per write, a fresh connection per operation. The engine is
``experiments.monitoring``. Inference telemetry is NOT stored here: the #30 ``inference_records``
in ``deployments.sqlite3`` are the only telemetry authority, and every row here cites one.

Records

    monitoring_policies  content-addressed (``mp-<hash>``) ``wynk-monitoring-policy/1`` documents:
                         drift statistics, thresholds, trigger rules. Immutable (trigger).
    feedback             one labelled (or unlabelled input-only) observation per inference
                         record, bound to it by ``request_sha256`` / ``output_sha256``. Immutable
                         (trigger); ``UNIQUE(inference_id)``: a second, different feedback for the
                         same inference is a conflict, never an overwrite.
    trigger_decisions    one immutable decision per deterministic trigger identity
                         (``tr-<hash of the evidence>``): NOT_TRIGGERED / TRIGGERED / SUPPRESSED
                         with machine-readable reasons and evidence. The same evidence can only
                         ever have one decision (PRIMARY KEY, inserted under BEGIN IMMEDIATE),
                         and at most one TRIGGERED per workflow version is open at a time.
    frozen_examples      which feedback a TRIGGERED decision froze into a new dataset version
                         (``feedback_id`` PRIMARY KEY: a label enters at most one new version).
    reoptimizations      per TRIGGERED decision: the fenced claim and the forward-only progress
                         CLAIMED -> DATASET_CREATED -> JOB_CREATED (or FAILED), with the new
                         dataset version, splits and challenger job it produced.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from store.datasets import Conflict, RepositoryError, _Conn
from store.jobs import utc_now
from store.tenancy import bind_sqlite


class TriggerOutcome(StrEnum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    TRIGGERED = "TRIGGERED"
    SUPPRESSED = "SUPPRESSED"


class ReoptState(StrEnum):
    CLAIMED = "CLAIMED"
    DATASET_CREATED = "DATASET_CREATED"
    JOB_CREATED = "JOB_CREATED"
    FAILED = "FAILED"


class FeedbackConflict(Conflict):
    """A different feedback already exists for this inference record."""


class ReoptClaimLost(Conflict):
    """Another monitor holds (or advanced) this re-optimization: nothing was written."""


@dataclass(frozen=True)
class PolicyRow:
    policy_id: str
    policy_json: str
    created_at: str


@dataclass(frozen=True)
class FeedbackRow:
    feedback_id: str
    inference_id: str
    workflow_version: str
    lineage_id: str
    labelled: bool
    inference_created_at: str
    record_json: str
    received_at: str


@dataclass(frozen=True)
class DecisionRow:
    trigger_id: str
    workflow_version: str
    lineage_id: str
    policy_id: str
    outcome: TriggerOutcome
    window_until: str
    decision_json: str
    created_at: str


@dataclass(frozen=True)
class ReoptRow:
    trigger_id: str
    state: ReoptState
    owner: str
    fence: int
    lease_until: float
    dataset_id: str | None
    dataset_version: int | None
    splits_hash: str | None
    job_id: str | None
    detail: str | None
    created_at: str
    updated_at: str


SCHEMA_VERSION = 1
_REOPT_TRANSITIONS = (
    "CLAIMED>CLAIMED",  # a re-claim (fence bump) of an unfinished re-optimization
    "CLAIMED>DATASET_CREATED",
    "DATASET_CREATED>DATASET_CREATED",
    "DATASET_CREATED>JOB_CREATED",
    "CLAIMED>FAILED",
    "DATASET_CREATED>FAILED",
)
SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS monitoring_policies (
        policy_id   TEXT PRIMARY KEY,               -- content-addressed: mp-<document hash>
        policy_json TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS feedback (
        feedback_id          TEXT PRIMARY KEY,
        inference_id         TEXT NOT NULL UNIQUE,  -- one feedback per inference record
        workflow_version     TEXT NOT NULL,
        lineage_id           TEXT NOT NULL,
        labelled             INTEGER NOT NULL CHECK (labelled IN (0, 1)),
        inference_created_at TEXT NOT NULL,         -- copied from the cited inference record
        record_json          TEXT NOT NULL,
        received_at          TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS feedback_by_version ON feedback(workflow_version)",
    """CREATE TABLE IF NOT EXISTS trigger_decisions (
        trigger_id       TEXT PRIMARY KEY,          -- tr-<hash of the evidence identity>
        workflow_version TEXT NOT NULL,
        lineage_id       TEXT NOT NULL,
        policy_id        TEXT NOT NULL REFERENCES monitoring_policies(policy_id),
        outcome          TEXT NOT NULL
                         CHECK (outcome IN ('NOT_TRIGGERED', 'TRIGGERED', 'SUPPRESSED')),
        window_until     TEXT NOT NULL,
        decision_json    TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS decisions_by_version
        ON trigger_decisions(workflow_version, created_at)""",
    """CREATE TABLE IF NOT EXISTS frozen_examples (
        feedback_id TEXT PRIMARY KEY REFERENCES feedback(feedback_id),
        trigger_id  TEXT NOT NULL REFERENCES trigger_decisions(trigger_id)
    )""",
    """CREATE TABLE IF NOT EXISTS reoptimizations (
        trigger_id      TEXT PRIMARY KEY REFERENCES trigger_decisions(trigger_id),
        state           TEXT NOT NULL
                        CHECK (state IN ('CLAIMED', 'DATASET_CREATED', 'JOB_CREATED', 'FAILED')),
        owner           TEXT NOT NULL,
        fence           INTEGER NOT NULL,
        lease_until     REAL NOT NULL,
        dataset_id      TEXT,
        dataset_version INTEGER,
        splits_hash     TEXT,
        job_id          TEXT UNIQUE,
        detail          TEXT,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )""",
    # -- immutability, enforced by the database itself -----------------------------------------
    """CREATE TRIGGER IF NOT EXISTS monitoring_policies_are_immutable
    BEFORE UPDATE ON monitoring_policies
    BEGIN SELECT RAISE(ABORT, 'monitoring policies are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS monitoring_policies_are_permanent
    BEFORE DELETE ON monitoring_policies
    BEGIN SELECT RAISE(ABORT, 'monitoring policies are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS feedback_is_immutable BEFORE UPDATE ON feedback
    BEGIN SELECT RAISE(ABORT, 'feedback is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS feedback_is_permanent BEFORE DELETE ON feedback
    BEGIN SELECT RAISE(ABORT, 'feedback is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS trigger_decisions_are_immutable
    BEFORE UPDATE ON trigger_decisions
    BEGIN SELECT RAISE(ABORT, 'trigger decisions are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS trigger_decisions_are_permanent
    BEFORE DELETE ON trigger_decisions
    BEGIN SELECT RAISE(ABORT, 'trigger decisions are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS frozen_examples_are_immutable
    BEFORE UPDATE ON frozen_examples
    BEGIN SELECT RAISE(ABORT, 'frozen examples are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS frozen_examples_are_permanent
    BEFORE DELETE ON frozen_examples
    BEGIN SELECT RAISE(ABORT, 'frozen examples are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS reoptimizations_move_forward
    BEFORE UPDATE ON reoptimizations
    WHEN NEW.trigger_id IS NOT OLD.trigger_id OR NEW.fence < OLD.fence
      OR (OLD.state || '>' || NEW.state) NOT IN ("""
    + ", ".join(f"'{t}'" for t in _REOPT_TRANSITIONS)
    + """)
      OR (OLD.job_id IS NOT NULL AND NEW.job_id IS NOT OLD.job_id)
      OR (OLD.dataset_version IS NOT NULL AND NEW.dataset_version IS NOT OLD.dataset_version)
    BEGIN SELECT RAISE(ABORT, 're-optimization progress only moves forward'); END""",
    """CREATE TRIGGER IF NOT EXISTS reoptimizations_are_permanent
    BEFORE DELETE ON reoptimizations
    BEGIN SELECT RAISE(ABORT, 're-optimizations are permanent'); END""",
)

_FEEDBACK_COLS = (
    "feedback_id, inference_id, workflow_version, lineage_id, labelled, inference_created_at, "
    "record_json, received_at"
)
_DECISION_COLS = (
    "trigger_id, workflow_version, lineage_id, policy_id, outcome, window_until, decision_json, "
    "created_at"
)
_REOPT_COLS = (
    "trigger_id, state, owner, fence, lease_until, dataset_id, dataset_version, splits_hash, "
    "job_id, detail, created_at, updated_at"
)


class SQLiteMonitoringStore:
    def __init__(
        self,
        path: Path | str,
        clock: Callable[[], str] = utc_now,
        *,
        workspace_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        bind_sqlite(self.path, workspace_id)  # #32: before any read or write
        self.workspace_id = workspace_id
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            def create(c: _Conn) -> None:
                for statement in SCHEMA:
                    c.execute(statement)
                row = c.execute(
                    "SELECT value FROM meta WHERE key='monitoring_schema_version'"
                ).fetchone()
                if row is None:
                    c.execute(
                        "INSERT INTO meta(key, value) VALUES ('monitoring_schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                elif row[0] != str(SCHEMA_VERSION):
                    raise RepositoryError(f"unsupported monitoring store schema version {row[0]}")

            self._write(conn, create)

    @contextmanager
    def _connect(self) -> Iterator[_Conn]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            yield _Conn(conn)
        finally:
            conn.close()

    @staticmethod
    def _write(conn: _Conn, fn: Callable[[_Conn], Any]) -> Any:
        conn.execute("BEGIN IMMEDIATE")
        try:
            out = fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return out

    def _tx(self, fn: Callable[[_Conn], Any]) -> Any:
        with self._connect() as conn:
            return self._write(conn, fn)

    # -- policies ---------------------------------------------------------------------------
    def put_policy(self, policy_id: str, policy_json: str) -> tuple[PolicyRow, bool]:
        now = self.clock()

        def put(c: _Conn) -> tuple[PolicyRow, bool]:
            row = c.execute(
                "SELECT policy_id, policy_json, created_at FROM monitoring_policies "
                "WHERE policy_id=?",
                (policy_id,),
            ).fetchone()
            if row is not None:
                return PolicyRow(*row), False
            c.execute(
                "INSERT INTO monitoring_policies(policy_id, policy_json, created_at) "
                "VALUES (?,?,?)",
                (policy_id, policy_json, now),
            )
            return PolicyRow(policy_id, policy_json, now), True

        return self._tx(put)

    def policy(self, policy_id: str) -> PolicyRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT policy_id, policy_json, created_at FROM monitoring_policies "
                "WHERE policy_id=?",
                (policy_id,),
            ).fetchone()
        return PolicyRow(*row) if row is not None else None

    def policies(self) -> list[PolicyRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT policy_id, policy_json, created_at FROM monitoring_policies "
                "ORDER BY created_at, policy_id"
            ).fetchall()
        return [PolicyRow(*r) for r in rows]

    # -- feedback ---------------------------------------------------------------------------
    def put_feedback(self, row: FeedbackRow) -> tuple[FeedbackRow, bool]:
        """Insert ``row``, or return the identical feedback already stored for its inference
        (``bool``: inserted). A different feedback for the same inference is refused."""

        def put(c: _Conn) -> tuple[FeedbackRow, bool]:
            existing = c.execute(
                f"SELECT {_FEEDBACK_COLS} FROM feedback WHERE inference_id=?", (row.inference_id,)
            ).fetchone()
            if existing is not None:
                stored = _feedback(existing)
                if stored.feedback_id != row.feedback_id:
                    raise FeedbackConflict(
                        f"inference {row.inference_id} already has feedback {stored.feedback_id}"
                    )
                return stored, False
            c.execute(
                f"INSERT INTO feedback({_FEEDBACK_COLS}) VALUES (?,?,?,?,?,?,?,?)",
                (
                    row.feedback_id,
                    row.inference_id,
                    row.workflow_version,
                    row.lineage_id,
                    int(row.labelled),
                    row.inference_created_at,
                    row.record_json,
                    row.received_at,
                ),
            )
            return row, True

        return self._tx(put)

    def feedback(self, feedback_id: str) -> FeedbackRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_FEEDBACK_COLS} FROM feedback WHERE feedback_id=?", (feedback_id,)
            ).fetchone()
        return _feedback(row) if row is not None else None

    def feedback_for_version(self, workflow_version: str) -> list[FeedbackRow]:
        """Every feedback citing an inference of exactly ``workflow_version``."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_FEEDBACK_COLS} FROM feedback WHERE workflow_version=? "
                "ORDER BY inference_created_at, feedback_id",
                (workflow_version,),
            ).fetchall()
        return [_feedback(r) for r in rows]

    def frozen_feedback_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT feedback_id FROM frozen_examples").fetchall()
        return {r[0] for r in rows}

    def frozen_by(self, trigger_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT feedback_id FROM frozen_examples WHERE trigger_id=? ORDER BY feedback_id",
                (trigger_id,),
            ).fetchall()
        return [r[0] for r in rows]

    # -- decisions --------------------------------------------------------------------------
    def decide(
        self,
        trigger_id: str,
        workflow_version: str,
        lineage_id: str,
        policy_id: str,
        window_until: str,
        decide: Callable[[list[DecisionRow], set[str]], tuple[TriggerOutcome, str, list[str]]],
    ) -> tuple[DecisionRow, bool]:
        """The decision for ``trigger_id``: the stored one if this evidence was already decided
        (``bool`` False), else ``decide(prior decisions of this version, frozen feedback ids)``
        evaluated and inserted - with the feedback a TRIGGERED decision freezes - inside ONE
        ``BEGIN IMMEDIATE`` transaction. Concurrent monitors are serialised by the write lock,
        so one evidence never gets two decisions and two TRIGGERED decisions are never both
        written on the strength of the same history."""
        now = self.clock()

        def write(c: _Conn) -> tuple[DecisionRow, bool]:
            existing = c.execute(
                f"SELECT {_DECISION_COLS} FROM trigger_decisions WHERE trigger_id=?",
                (trigger_id,),
            ).fetchone()
            if existing is not None:
                return _decision(existing), False
            prior = [
                _decision(r)
                for r in c.execute(
                    f"SELECT {_DECISION_COLS} FROM trigger_decisions WHERE workflow_version=? "
                    "ORDER BY created_at, trigger_id",
                    (workflow_version,),
                ).fetchall()
            ]
            frozen = {r[0] for r in c.execute("SELECT feedback_id FROM frozen_examples")}
            outcome, decision_json, freeze = decide(prior, frozen)
            c.execute(
                f"INSERT INTO trigger_decisions({_DECISION_COLS}) VALUES (?,?,?,?,?,?,?,?)",
                (
                    trigger_id,
                    workflow_version,
                    lineage_id,
                    policy_id,
                    outcome.value,
                    window_until,
                    decision_json,
                    now,
                ),
            )
            if outcome is TriggerOutcome.TRIGGERED:
                for feedback_id in sorted(freeze):
                    c.execute(
                        "INSERT INTO frozen_examples(feedback_id, trigger_id) VALUES (?,?)",
                        (feedback_id, trigger_id),
                    )
            row = DecisionRow(
                trigger_id,
                workflow_version,
                lineage_id,
                policy_id,
                outcome,
                window_until,
                decision_json,
                now,
            )
            return row, True

        return self._tx(write)

    def decision(self, trigger_id: str) -> DecisionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_DECISION_COLS} FROM trigger_decisions WHERE trigger_id=?",
                (trigger_id,),
            ).fetchone()
        return _decision(row) if row is not None else None

    def decisions(self, workflow_version: str) -> list[DecisionRow]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_DECISION_COLS} FROM trigger_decisions WHERE workflow_version=? "
                "ORDER BY created_at, trigger_id",
                (workflow_version,),
            ).fetchall()
        return [_decision(r) for r in rows]

    # -- re-optimizations -------------------------------------------------------------------
    def reoptimization(self, trigger_id: str) -> ReoptRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_REOPT_COLS} FROM reoptimizations WHERE trigger_id=?", (trigger_id,)
            ).fetchone()
        return _reopt(row) if row is not None else None

    def claim_reoptimization(
        self, trigger_id: str, owner: str, now: float, lease_s: float
    ) -> ReoptRow | None:
        """Claim the (unfinished) re-optimization of a TRIGGERED decision: a fresh claim, or a
        re-claim of one whose lease expired or that ``owner`` already holds. Every claim bumps
        the fence. ``None``: finished (JOB_CREATED / FAILED) or held by a live other monitor."""
        stamp = self.clock()

        def claim(c: _Conn) -> ReoptRow | None:
            decision = c.execute(
                "SELECT outcome FROM trigger_decisions WHERE trigger_id=?", (trigger_id,)
            ).fetchone()
            if decision is None or decision[0] != TriggerOutcome.TRIGGERED.value:
                raise RepositoryError(f"{trigger_id} is not a TRIGGERED decision")
            row = c.execute(
                f"SELECT {_REOPT_COLS} FROM reoptimizations WHERE trigger_id=?", (trigger_id,)
            ).fetchone()
            if row is None:
                c.execute(
                    f"INSERT INTO reoptimizations({_REOPT_COLS}) "
                    "VALUES (?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,?,?)",
                    (trigger_id, ReoptState.CLAIMED.value, owner, 1, now + lease_s, stamp, stamp),
                )
            else:
                current = _reopt(row)
                if current.state in (ReoptState.JOB_CREATED, ReoptState.FAILED):
                    return None
                if current.owner != owner and current.lease_until > now:
                    return None
                c.execute(
                    "UPDATE reoptimizations SET owner=?, fence=fence+1, lease_until=?, "
                    "updated_at=? WHERE trigger_id=?",
                    (owner, now + lease_s, stamp, trigger_id),
                )
            return _reopt(
                c.execute(
                    f"SELECT {_REOPT_COLS} FROM reoptimizations WHERE trigger_id=?", (trigger_id,)
                ).fetchone()
            )

        return self._tx(claim)

    def release_reoptimization(self, trigger_id: str, owner: str, fence: int) -> None:
        """End ``owner``'s lease without moving progress (e.g. job admission was refused), so
        any monitor can resume the same re-optimization at once."""
        self._tx(
            lambda c: c.execute(
                "UPDATE reoptimizations SET lease_until=0, updated_at=? "
                "WHERE trigger_id=? AND owner=? AND fence=?",
                (self.clock(), trigger_id, owner, fence),
            )
        )

    def advance_reoptimization(
        self, trigger_id: str, owner: str, fence: int, state: ReoptState, **fields: Any
    ) -> ReoptRow:
        """Move a claimed re-optimization forward, only while (owner, fence) still holds it."""
        allowed = {"dataset_id", "dataset_version", "splits_hash", "job_id", "detail"}
        if set(fields) - allowed:
            raise ValueError(f"unknown fields {sorted(set(fields) - allowed)}")
        stamp = self.clock()

        def advance(c: _Conn) -> ReoptRow:
            sets = ", ".join(f"{k}=?" for k in sorted(fields))
            cur = c.execute(
                f"UPDATE reoptimizations SET state=?, updated_at=?{', ' + sets if sets else ''} "
                "WHERE trigger_id=? AND owner=? AND fence=?",
                (
                    state.value,
                    stamp,
                    *(fields[k] for k in sorted(fields)),
                    trigger_id,
                    owner,
                    fence,
                ),
            )
            if cur.rowcount != 1:
                raise ReoptClaimLost(f"re-optimization {trigger_id} is no longer held by {owner}")
            return _reopt(
                c.execute(
                    f"SELECT {_REOPT_COLS} FROM reoptimizations WHERE trigger_id=?", (trigger_id,)
                ).fetchone()
            )

        return self._tx(advance)


def _feedback(r: tuple[Any, ...]) -> FeedbackRow:
    return FeedbackRow(r[0], r[1], r[2], r[3], bool(r[4]), r[5], r[6], r[7])


def _decision(r: tuple[Any, ...]) -> DecisionRow:
    return DecisionRow(r[0], r[1], r[2], r[3], TriggerOutcome(r[4]), r[5], r[6], r[7])


def _reopt(r: tuple[Any, ...]) -> ReoptRow:
    return ReoptRow(r[0], ReoptState(r[1]), *r[2:])


__all__ = [
    "DecisionRow",
    "FeedbackConflict",
    "FeedbackRow",
    "PolicyRow",
    "ReoptClaimLost",
    "ReoptRow",
    "ReoptState",
    "SQLiteMonitoringStore",
    "TriggerOutcome",
]
