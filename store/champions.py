"""Champion promotion: SQLite persistence for promotions, the write-ahead log of held-out runs,
and the immutable champion history of every lineage.

Same conventions as ``store.jobs`` / ``store.datasets``: stdlib ``sqlite3``, WAL,
``synchronous=FULL``, one ``BEGIN IMMEDIATE`` transaction per write, a fresh connection per
operation. The engine is ``experiments.promotion``; this module stores JSON documents and
enforces the state machine, the lease/fence of a held-out evaluation, the one-look rule for an
experiment's test split, and compare-and-promote.

Records

    promotions          one row per experiment that entered promotion. ``job_id`` is UNIQUE: an
                        experiment's test split is opened at most once, so a rejected challenger
                        stays rejected and no other candidate of that experiment ever reaches
                        test. The validation selection, the pinned incumbent and the moment the
                        test split was opened are immutable (trigger); the decision is written
                        once, and a DECIDED row can never change again (trigger).
    promotion_attempts  write-ahead log of held-out workflow runs, exactly like
                        ``experiment_attempts``: STARTED before the model is invoked, COMPLETED
                        (result + measured usage) atomically after. A completed attempt is
                        immutable (trigger).
    champion_lineages   per lineage: the current champion and its ``version`` - the fence every
                        promotion compares against before it may promote.
    champions           append-only champion records, one per (lineage, version). Immutable
                        (trigger): history is never rewritten, a newer champion is a new row.

Promotion states

    OPEN --> DECIDED                 (PROMOTED or REJECTED; terminal)
      |----> FAILED                  (held-out evaluation could not finish; terminal, no decision)
      '----> INTERRUPTED --> OPEN    (resume: same challenger, same incumbent, same rows)

Compare-and-promote. ``decide`` with a new champion runs in ONE transaction: it re-reads the
lineage version, refuses with ``StaleIncumbent`` (writing nothing) unless it is still the version
the promotion pinned when it opened the test split, then appends the champion, advances the
lineage and records the decision. Two promotions racing on one lineage can never both win, and a
promotion judged against an older incumbent can never overwrite a newer champion.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from store.datasets import Conflict, IntegrityViolation, RepositoryError, _Conn
from store.jobs import AttemptRow, AttemptState, LeaseLost, NewAttempt, _attempt, utc_now


class PromotionState(StrEnum):
    OPEN = "OPEN"  # the test split is open; the held-out evaluation holds a lease
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    DECIDED = "DECIDED"


TERMINAL = frozenset({PromotionState.DECIDED, PromotionState.FAILED})


class PromotionReason(StrEnum):
    """Why a promotion is INTERRUPTED or FAILED. Stable strings: part of the API."""

    PROCESS_LOST = "process_lost"  # its process died; nothing in flight: safe to resume
    AMBIGUOUS_ATTEMPT = "ambiguous_attempt"  # a held-out call started; its result was never stored
    RUNTIME_UNAVAILABLE = "runtime_unavailable"  # the held-out runtime could not be bound
    MODEL_UNAVAILABLE = "model_unavailable"
    HELDOUT_ERROR = "heldout_error"  # the runner / evaluator raised, or a run broke an invariant


RESUMABLE = frozenset({PromotionReason.PROCESS_LOST, PromotionReason.RUNTIME_UNAVAILABLE})


class StaleIncumbent(Conflict):
    """Compare-and-promote lost: the lineage's champion changed after this promotion pinned it."""


@dataclass(frozen=True)
class PromotionClaim:
    promotion_id: str
    owner: str
    fence: int


@dataclass(frozen=True)
class PromotionRow:
    promotion_id: str
    job_id: str
    lineage_id: str
    state: PromotionState
    reason: str | None
    detail: str | None
    created_at: str
    updated_at: str
    test_opened_at: str | None  # None: the test split was never opened for this experiment
    incumbent_id: str | None
    incumbent_version: int  # the lineage version pinned when the test split was opened
    selection_json: str
    lease_owner: str | None
    lease_until: float | None
    fence: int
    decision: str | None
    decision_json: str | None
    decided_at: str | None


@dataclass(frozen=True)
class ChampionRow:
    champion_id: str
    lineage_id: str
    version: int
    promotion_id: str
    compat_hash: str
    record_json: str
    created_at: str


@dataclass(frozen=True)
class NewChampion:
    champion_id: str
    compat_hash: str  # every champion of one lineage shares it
    record_json: str


SCHEMA_VERSION = 1
SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS promotions (
        promotion_id      TEXT PRIMARY KEY,
        job_id            TEXT NOT NULL UNIQUE,   -- one look at an experiment's test split
        lineage_id        TEXT NOT NULL,
        state             TEXT NOT NULL
                          CHECK (state IN ('OPEN', 'INTERRUPTED', 'FAILED', 'DECIDED')),
        reason            TEXT,
        detail            TEXT,
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        test_opened_at    TEXT,                   -- durable: test split opened (immutable)
        incumbent_id      TEXT,                   -- pinned incumbent (immutable)
        incumbent_version INTEGER NOT NULL,       -- lineage version it was pinned at (immutable)
        selection_json    TEXT NOT NULL,          -- validation selection + reasons (immutable)
        lease_owner       TEXT,
        lease_until       REAL,
        fence             INTEGER NOT NULL DEFAULT 0,
        next_seq          INTEGER NOT NULL DEFAULT 0,
        decision          TEXT CHECK (decision IS NULL OR decision IN ('PROMOTED', 'REJECTED')),
        decision_json     TEXT,                   -- the PromotionDecision, written once
        decided_at        TEXT,
        CHECK ((state = 'DECIDED') = (decision IS NOT NULL AND decision_json IS NOT NULL))
    )""",
    """CREATE TABLE IF NOT EXISTS promotion_attempts (
        attempt_id   TEXT    PRIMARY KEY,   -- deterministic: promotion, subject, genome, row, n
        promotion_id TEXT    NOT NULL REFERENCES promotions(promotion_id),
        subject      TEXT    NOT NULL CHECK (subject IN ('challenger', 'incumbent')),
        seq          INTEGER NOT NULL,
        genome_hash  TEXT    NOT NULL,
        genome_json  TEXT    NOT NULL,
        task_id      TEXT    NOT NULL,
        trial        INTEGER NOT NULL,
        run_seed     INTEGER NOT NULL,
        attempt      INTEGER NOT NULL,
        state        TEXT    NOT NULL CHECK (state IN ('STARTED', 'COMPLETED', 'ERRORED')),
        owner        TEXT    NOT NULL,
        fence        INTEGER NOT NULL,
        started_at   TEXT    NOT NULL,
        finished_at  TEXT,
        resolution   TEXT,                  -- 'executed' | 'replay_proof'
        result_json  TEXT,                  -- EvaluatedRun
        entry_json   TEXT,                  -- measured usage
        timing_json  TEXT,
        error        TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS promotion_attempts_by_seq ON promotion_attempts(promotion_id, seq)",
    """CREATE TABLE IF NOT EXISTS champion_lineages (
        lineage_id   TEXT    PRIMARY KEY,
        compat_hash  TEXT    NOT NULL,      -- what every champion of the lineage is comparable on
        version      INTEGER NOT NULL,      -- the current champion's version: the CAS fence
        champion_id  TEXT    NOT NULL,
        created_at   TEXT    NOT NULL,
        updated_at   TEXT    NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS champions (
        champion_id  TEXT    PRIMARY KEY,
        lineage_id   TEXT    NOT NULL REFERENCES champion_lineages(lineage_id),
        version      INTEGER NOT NULL CHECK (version >= 1),
        promotion_id TEXT    NOT NULL UNIQUE REFERENCES promotions(promotion_id),
        compat_hash  TEXT    NOT NULL,
        record_json  TEXT    NOT NULL,
        created_at   TEXT    NOT NULL,
        UNIQUE (lineage_id, version)
    )""",
    # -- immutability, enforced by the database itself -----------------------------------------
    """CREATE TRIGGER IF NOT EXISTS champions_are_immutable BEFORE UPDATE ON champions
    BEGIN SELECT RAISE(ABORT, 'champion records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS champions_are_permanent BEFORE DELETE ON champions
    BEGIN SELECT RAISE(ABORT, 'champion records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS promotions_are_permanent BEFORE DELETE ON promotions
    BEGIN SELECT RAISE(ABORT, 'promotion records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS decided_promotions_are_final BEFORE UPDATE ON promotions
    WHEN OLD.state IN ('DECIDED', 'FAILED')
    BEGIN SELECT RAISE(ABORT, 'a decided or failed promotion is final'); END""",
    """CREATE TRIGGER IF NOT EXISTS promotion_evidence_is_pinned BEFORE UPDATE ON promotions
    WHEN NEW.job_id IS NOT OLD.job_id OR NEW.lineage_id IS NOT OLD.lineage_id
      OR NEW.selection_json IS NOT OLD.selection_json
      OR NEW.test_opened_at IS NOT OLD.test_opened_at
      OR NEW.incumbent_id IS NOT OLD.incumbent_id
      OR NEW.incumbent_version IS NOT OLD.incumbent_version
    BEGIN SELECT RAISE(ABORT, 'selection, incumbent and test opening are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS completed_attempts_are_immutable
    BEFORE UPDATE ON promotion_attempts WHEN OLD.state != 'STARTED'
    BEGIN SELECT RAISE(ABORT, 'a finished held-out attempt is immutable'); END""",
)

_PROMOTION_COLS = (
    "promotion_id, job_id, lineage_id, state, reason, detail, created_at, updated_at, "
    "test_opened_at, incumbent_id, incumbent_version, selection_json, lease_owner, lease_until, "
    "fence, decision, decision_json, decided_at"
)
_ATTEMPT_COLS = (
    "attempt_id, seq, genome_hash, genome_json, task_id, trial, run_seed, attempt, state, owner, "
    "fence, resolution, result_json, entry_json, timing_json, error"
)
_CHAMPION_COLS = (
    "champion_id, lineage_id, version, promotion_id, compat_hash, record_json, created_at"
)


class SQLiteChampionStore:
    def __init__(self, path: Path | str, clock: Callable[[], str] = utc_now) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            def create(c: _Conn) -> None:
                for statement in SCHEMA:
                    c.execute(statement)

            self._write(conn, create)
            row = conn.execute(
                "SELECT value FROM meta WHERE key='champions_schema_version'"
            ).fetchone()
            if row is None:
                self._write(
                    conn,
                    lambda c: c.execute(
                        "INSERT OR IGNORE INTO meta(key, value) "
                        "VALUES ('champions_schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    ),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise RepositoryError(f"unsupported champion store schema version {row[0]}")

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

    # -- reads ------------------------------------------------------------------------------
    def get(self, promotion_id: str) -> PromotionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_PROMOTION_COLS} FROM promotions WHERE promotion_id=?", (promotion_id,)
            ).fetchone()
        return _promotion(row) if row is not None else None

    def for_job(self, job_id: str) -> PromotionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_PROMOTION_COLS} FROM promotions WHERE job_id=?", (job_id,)
            ).fetchone()
        return _promotion(row) if row is not None else None

    def promotions(self, lineage_id: str | None = None) -> list[PromotionRow]:
        where, params = ("WHERE lineage_id=? ", (lineage_id,)) if lineage_id else ("", ())
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_PROMOTION_COLS} FROM promotions {where}"
                "ORDER BY created_at, promotion_id",
                params,
            ).fetchall()
        return [_promotion(r) for r in rows]

    def attempts(self, promotion_id: str) -> list[tuple[str, AttemptRow]]:
        """``(subject, attempt)`` in write order."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT subject, {_ATTEMPT_COLS} FROM promotion_attempts "
                "WHERE promotion_id=? ORDER BY seq",
                (promotion_id,),
            ).fetchall()
        return [(r[0], _attempt(r[1:])) for r in rows]

    def current(self, lineage_id: str) -> ChampionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {', '.join('c.' + c for c in _CHAMPION_COLS.split(', '))} "
                "FROM champion_lineages l JOIN champions c ON c.champion_id = l.champion_id "
                "WHERE l.lineage_id=?",
                (lineage_id,),
            ).fetchone()
        return ChampionRow(*row) if row is not None else None

    def lineage_version(self, lineage_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM champion_lineages WHERE lineage_id=?", (lineage_id,)
            ).fetchone()
        return row[0] if row is not None else 0

    def history(self, lineage_id: str) -> list[ChampionRow]:
        """Every champion the lineage ever had, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHAMPION_COLS} FROM champions WHERE lineage_id=? ORDER BY version",
                (lineage_id,),
            ).fetchall()
        return [ChampionRow(*r) for r in rows]

    def champion(self, champion_id: str) -> ChampionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_CHAMPION_COLS} FROM champions WHERE champion_id=?", (champion_id,)
            ).fetchone()
        return ChampionRow(*row) if row is not None else None

    def lineages(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT lineage_id FROM champion_lineages ORDER BY lineage_id")
            return [r[0] for r in rows.fetchall()]

    # -- entering promotion -----------------------------------------------------------------
    def _insert(self, c: _Conn, promotion_id: str, job_id: str, lineage_id: str, **cols: Any):
        now = self.clock()
        fields = {
            "promotion_id": promotion_id,
            "job_id": job_id,
            "lineage_id": lineage_id,
            "created_at": now,
            "updated_at": now,
            **cols,
        }
        try:
            c.execute(
                f"INSERT INTO promotions({', '.join(fields)}) "
                f"VALUES ({', '.join('?' for _ in fields)})",
                tuple(fields.values()),
            )
        except sqlite3.IntegrityError:
            raise Conflict(f"experiment {job_id} already entered promotion") from None

    def record_without_test(
        self,
        promotion_id: str,
        job_id: str,
        lineage_id: str,
        selection_json: str,
        decision: str,
        decision_json: str,
    ) -> None:
        """A decision reached before the test split was opened (e.g. no feasible challenger).
        The experiment's promotion is final all the same."""
        now = self.clock()
        self._tx(
            lambda c: self._insert(
                c,
                promotion_id,
                job_id,
                lineage_id,
                state=PromotionState.DECIDED.value,
                incumbent_version=self.lineage_version_in(c, lineage_id),
                selection_json=selection_json,
                decision=decision,
                decision_json=decision_json,
                decided_at=now,
            )
        )

    def open(
        self,
        promotion_id: str,
        job_id: str,
        lineage_id: str,
        selection_json: str,
        incumbent_id: str | None,
        incumbent_version: int,
        owner: str,
        now_s: float,
        lease_s: float,
    ) -> PromotionClaim:
        """Durably open the experiment's test split for its ONE validation-selected challenger,
        pinning the incumbent (and its lineage version). Refused - nothing written - if the
        experiment already entered promotion, or if the incumbent is no longer current."""

        def open_(c: _Conn) -> PromotionClaim:
            current = c.execute(
                "SELECT champion_id, version FROM champion_lineages WHERE lineage_id=?",
                (lineage_id,),
            ).fetchone()
            if (current or (None, 0)) != (incumbent_id, incumbent_version):
                raise StaleIncumbent(f"lineage {lineage_id} changed before the test split opened")
            self._insert(
                c,
                promotion_id,
                job_id,
                lineage_id,
                state=PromotionState.OPEN.value,
                test_opened_at=self.clock(),
                incumbent_id=incumbent_id,
                incumbent_version=incumbent_version,
                selection_json=selection_json,
                lease_owner=owner,
                lease_until=now_s + lease_s,
                fence=1,
            )
            return PromotionClaim(promotion_id, owner, 1)

        return self._tx(open_)

    @staticmethod
    def lineage_version_in(c: _Conn, lineage_id: str) -> int:
        row = c.execute(
            "SELECT version FROM champion_lineages WHERE lineage_id=?", (lineage_id,)
        ).fetchone()
        return row[0] if row is not None else 0

    # -- leases -----------------------------------------------------------------------------
    def claim(
        self, promotion_id: str, owner: str, now_s: float, lease_s: float
    ) -> PromotionClaim | None:
        """Take over an unfinished held-out evaluation: OPEN with an expired lease, or
        INTERRUPTED with nothing in flight. With a call in flight it becomes INTERRUPTED /
        ambiguous_attempt instead (never re-sent); ``None`` when it is not claimable."""

        def take(c: _Conn) -> PromotionClaim | None:
            row = c.execute(
                "SELECT state, lease_until FROM promotions WHERE promotion_id=?", (promotion_id,)
            ).fetchone()
            if row is None:
                return None
            state, lease_until = PromotionState(row[0]), row[1]
            if state in TERMINAL:
                return None
            if state is PromotionState.OPEN and lease_until is not None and lease_until >= now_s:
                return None  # a live process owns it
            if self._interrupt_if_ambiguous(c, promotion_id):
                return None
            c.execute(
                "UPDATE promotions SET state='OPEN', reason=NULL, detail=NULL, lease_owner=?, "
                "lease_until=?, fence=fence+1, updated_at=? WHERE promotion_id=?",
                (owner, now_s + lease_s, self.clock(), promotion_id),
            )
            (fence,) = c.execute(
                "SELECT fence FROM promotions WHERE promotion_id=?", (promotion_id,)
            ).fetchone()
            return PromotionClaim(promotion_id, owner, fence)

        return self._tx(take)

    def _interrupt_if_ambiguous(self, c: _Conn, promotion_id: str) -> bool:
        (n,) = c.execute(
            "SELECT COUNT(*) FROM promotion_attempts WHERE promotion_id=? AND state='STARTED'",
            (promotion_id,),
        ).fetchone()
        if not n:
            return False
        c.execute(
            "UPDATE promotions SET state='INTERRUPTED', reason=?, detail=?, lease_owner=NULL, "
            "lease_until=NULL, updated_at=? WHERE promotion_id=?",
            (
                PromotionReason.AMBIGUOUS_ATTEMPT.value,
                f"{n} held-out model call(s) started without a stored result; their outcome and "
                "spend are unknown",
                self.clock(),
                promotion_id,
            ),
        )
        return True

    def interrupt_stale(self, now_s: float) -> list[tuple[str, PromotionReason]]:
        """Recovery: every OPEN promotion whose lease expired becomes INTERRUPTED - ambiguous if
        a held-out call is in flight, else process_lost (resumable)."""

        def scan(c: _Conn) -> list[tuple[str, PromotionReason]]:
            out = []
            rows = c.execute(
                "SELECT promotion_id FROM promotions WHERE state='OPEN' AND "
                "(lease_until IS NULL OR lease_until < ?) ORDER BY created_at, promotion_id",
                (now_s,),
            ).fetchall()
            for (pid,) in rows:
                if self._interrupt_if_ambiguous(c, pid):
                    out.append((pid, PromotionReason.AMBIGUOUS_ATTEMPT))
                    continue
                c.execute(
                    "UPDATE promotions SET state='INTERRUPTED', reason=?, detail=NULL, "
                    "lease_owner=NULL, lease_until=NULL, updated_at=? WHERE promotion_id=?",
                    (PromotionReason.PROCESS_LOST.value, self.clock(), pid),
                )
                out.append((pid, PromotionReason.PROCESS_LOST))
            return out

        return self._tx(scan)

    def _guard(self, c: _Conn, claim: PromotionClaim, now_s: float, lease_s: float) -> None:
        cur = c.execute(
            "UPDATE promotions SET lease_until=?, updated_at=? WHERE promotion_id=? AND "
            "state='OPEN' AND lease_owner=? AND fence=?",
            (now_s + lease_s, self.clock(), claim.promotion_id, claim.owner, claim.fence),
        )
        if cur.rowcount != 1:
            raise LeaseLost(f"{claim.owner} no longer owns promotion {claim.promotion_id}")

    def renew(self, claim: PromotionClaim, now_s: float, lease_s: float) -> None:
        self._tx(lambda c: self._guard(c, claim, now_s, lease_s))

    # -- write-ahead held-out attempts ------------------------------------------------------
    def start_attempt(
        self,
        claim: PromotionClaim,
        subject: str,
        attempt: NewAttempt,
        now_s: float,
        lease_s: float,
    ) -> None:
        """Record that a held-out model call is about to happen (fenced). Refused - nothing
        written, so nothing may be called - unless this process still owns the evaluation."""

        def start(c: _Conn) -> None:
            self._guard(c, claim, now_s, lease_s)
            (seq,) = c.execute(
                "SELECT next_seq FROM promotions WHERE promotion_id=?", (claim.promotion_id,)
            ).fetchone()
            c.execute(
                "UPDATE promotions SET next_seq=? WHERE promotion_id=?",
                (seq + 1, claim.promotion_id),
            )
            try:
                c.execute(
                    "INSERT INTO promotion_attempts(attempt_id, promotion_id, subject, seq, "
                    "genome_hash, genome_json, task_id, trial, run_seed, attempt, state, owner, "
                    "fence, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,'STARTED',?,?,?)",
                    (
                        attempt.attempt_id,
                        claim.promotion_id,
                        subject,
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
                raise Conflict(f"held-out attempt {attempt.attempt_id} already started") from None

        self._tx(start)

    def complete_attempt(
        self,
        claim: PromotionClaim,
        attempt_id: str,
        result_json: str,
        entry_json: str,
        timing_json: str | None,
    ) -> None:
        """Store the result + measured usage of an attempt this process started (atomic)."""

        def complete(c: _Conn) -> None:
            cur = c.execute(
                "UPDATE promotion_attempts SET state='COMPLETED', resolution='executed', "
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
                raise IntegrityViolation(f"held-out attempt {attempt_id} is not in flight here")

        self._tx(complete)

    def error_attempt(self, claim: PromotionClaim, attempt_id: str, error: str) -> None:
        self._tx(
            lambda c: c.execute(
                "UPDATE promotion_attempts SET state='ERRORED', error=?, finished_at=? "
                "WHERE attempt_id=? AND state='STARTED' AND owner=? AND fence=?",
                (error[:2000], self.clock(), attempt_id, claim.owner, claim.fence),
            )
        )

    def resolve_attempt(
        self, attempt_id: str, result_json: str, entry_json: str, resolution: str
    ) -> None:
        """Settle an ambiguous (STARTED) held-out attempt with a result proven by other means."""

        def resolve(c: _Conn) -> None:
            cur = c.execute(
                "UPDATE promotion_attempts SET state='COMPLETED', resolution=?, result_json=?, "
                "entry_json=?, finished_at=? WHERE attempt_id=? AND state='STARTED'",
                (resolution, result_json, entry_json, self.clock(), attempt_id),
            )
            if cur.rowcount != 1:
                raise Conflict(f"held-out attempt {attempt_id} is not in flight")

        self._tx(resolve)

    # -- outcomes ---------------------------------------------------------------------------
    def _close(self, claim: PromotionClaim, state: PromotionState, reason, detail) -> None:
        def close(c: _Conn) -> None:
            cur = c.execute(
                "UPDATE promotions SET state=?, reason=?, detail=?, lease_owner=NULL, "
                "lease_until=NULL, updated_at=? WHERE promotion_id=? AND state='OPEN' AND "
                "lease_owner=? AND fence=?",
                (
                    state.value,
                    reason.value,
                    (detail or None) and detail[:2000],
                    self.clock(),
                    claim.promotion_id,
                    claim.owner,
                    claim.fence,
                ),
            )
            if cur.rowcount != 1:
                raise LeaseLost(f"{claim.owner} no longer owns promotion {claim.promotion_id}")

        self._tx(close)

    def interrupt(
        self, claim: PromotionClaim, reason: PromotionReason, detail: str | None = None
    ) -> None:
        self._close(claim, PromotionState.INTERRUPTED, reason, detail)

    def fail(self, claim: PromotionClaim, reason: PromotionReason, detail: str | None = None):
        """Terminal: the held-out evaluation cannot finish. No decision, no champion change; the
        experiment's test split stays opened, so it can never be promoted."""
        self._close(claim, PromotionState.FAILED, reason, detail)

    def decide(
        self,
        claim: PromotionClaim,
        decision: str,
        decision_json: str,
        champion: NewChampion | None = None,
    ) -> None:
        """Record the decision (fenced, once) and - for a promotion - compare-and-promote in the
        same transaction: refused with ``StaleIncumbent`` (nothing written) unless the lineage
        is still at the version this promotion pinned when it opened the test split."""
        if (decision == "PROMOTED") != (champion is not None):
            raise ValueError("exactly a PROMOTED decision carries a new champion")

        def decide_(c: _Conn) -> None:
            row = c.execute(
                "SELECT lineage_id, incumbent_id, incumbent_version FROM promotions WHERE "
                "promotion_id=? AND state='OPEN' AND lease_owner=? AND fence=?",
                (claim.promotion_id, claim.owner, claim.fence),
            ).fetchone()
            if row is None:
                raise LeaseLost(f"{claim.owner} no longer owns promotion {claim.promotion_id}")
            lineage_id, incumbent_id, pinned = row
            now = self.clock()
            if champion is not None:
                current = c.execute(
                    "SELECT champion_id, version, compat_hash FROM champion_lineages "
                    "WHERE lineage_id=?",
                    (lineage_id,),
                ).fetchone()
                if (current[:2] if current else (None, 0)) != (incumbent_id, pinned):
                    raise StaleIncumbent(
                        f"lineage {lineage_id} is at version {current[1] if current else 0}; "
                        f"this promotion was judged against version {pinned}"
                    )
                if current is not None and current[2] != champion.compat_hash:
                    raise IntegrityViolation(
                        f"champion is not comparable with lineage {lineage_id}"
                    )
                version = pinned + 1
                if current is None:
                    c.execute(
                        "INSERT INTO champion_lineages(lineage_id, compat_hash, version, "
                        "champion_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                        (lineage_id, champion.compat_hash, version, champion.champion_id, now, now),
                    )
                else:
                    c.execute(
                        "UPDATE champion_lineages SET version=?, champion_id=?, updated_at=? "
                        "WHERE lineage_id=? AND version=?",
                        (version, champion.champion_id, now, lineage_id, pinned),
                    )
            c.execute(
                "UPDATE promotions SET state='DECIDED', reason=NULL, detail=NULL, "
                "lease_owner=NULL, lease_until=NULL, decision=?, decision_json=?, decided_at=?, "
                "updated_at=? WHERE promotion_id=?",
                (decision, decision_json, now, now, claim.promotion_id),
            )
            if champion is not None:  # after the promotion row: it references the decision
                c.execute(
                    "INSERT INTO champions(champion_id, lineage_id, version, promotion_id, "
                    "compat_hash, record_json, created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        champion.champion_id,
                        lineage_id,
                        version,
                        claim.promotion_id,
                        champion.compat_hash,
                        champion.record_json,
                        now,
                    ),
                )

        self._tx(decide_)


def _promotion(r: tuple[Any, ...]) -> PromotionRow:
    return PromotionRow(
        promotion_id=r[0],
        job_id=r[1],
        lineage_id=r[2],
        state=PromotionState(r[3]),
        reason=r[4],
        detail=r[5],
        created_at=r[6],
        updated_at=r[7],
        test_opened_at=r[8],
        incumbent_id=r[9],
        incumbent_version=r[10],
        selection_json=r[11],
        lease_owner=r[12],
        lease_until=r[13],
        fence=r[14],
        decision=r[15],
        decision_json=r[16],
        decided_at=r[17],
    )


__all__ = [
    "AttemptRow",
    "AttemptState",
    "ChampionRow",
    "Conflict",
    "LeaseLost",
    "NewAttempt",
    "NewChampion",
    "PromotionClaim",
    "PromotionReason",
    "PromotionRow",
    "PromotionState",
    "SQLiteChampionStore",
    "StaleIncumbent",
]
