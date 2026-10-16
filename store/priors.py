"""Durable, tenant-scoped workflow-memory priors (#28).

One SQLite store per workspace partition (``<data>/workspaces/<ws>/priors.sqlite3``), bound to
that workspace before any read or write (#32 ``bind_sqlite``): a prior can never be read through
another workspace's handle. It replaces the legacy JSON-file-per-task-class memory
(``memory.store``) for production; that one remains for the synthetic demo only.

    priors          one immutable ``PriorRecord`` per row, content-addressed (``prior_id`` is the
                    hash of ``record_json``: deterministic, no timestamps). State is ACTIVE or
                    RETIRED; retirement is written once (reason, time, superseding prior) and
                    never undone. ``invalidated_*`` is written once too, when the prior's
                    authority stops holding (dataset deleted, provenance no longer verifies):
                    an invalidated prior is unusable even by a job that already pinned it.
    prior_datasets  every dataset version a prior depends on (transitively), for #33 deletion.
    learned_jobs    one row per job that learning has decided on (learned or refused), so a
                    job is never learned twice and refusals stay auditable.

Nothing is ever deleted (triggers enforce it): history stays queryable. At most one ACTIVE
prior exists per ``(kind, key_hash)`` (a partial unique index), and every write runs in one
``BEGIN IMMEDIATE`` transaction, so concurrent learners serialize and never leave two heads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory.priors import (
    INVALIDATING,
    DatasetRef,
    PriorKind,
    PriorRecord,
    RetireReason,
)
from store.datasets import RepositoryError, _Conn
from store.jobs import utc_now
from store.tenancy import bind_sqlite

SCHEMA_VERSION = 1
_REASONS = ", ".join(f"'{r.value}'" for r in RetireReason)
_INVALIDATING = ", ".join(f"'{r.value}'" for r in sorted(INVALIDATING))
SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS priors (
        prior_id           TEXT    PRIMARY KEY,     -- hash of record_json (content-addressed)
        seq                INTEGER NOT NULL UNIQUE, -- write order
        kind               TEXT    NOT NULL CHECK (kind IN ('dataset_champion', 'task_family')),
        key_hash           TEXT    NOT NULL,        -- exact validity key
        anchor             TEXT    NOT NULL,        -- lookup only (never a compatibility test)
        source_job_id      TEXT    NOT NULL,
        record_json        TEXT    NOT NULL,
        created_at         TEXT    NOT NULL,
        state              TEXT    NOT NULL CHECK (state IN ('ACTIVE', 'RETIRED')),
        retired_at         TEXT,
        retired_reason     TEXT    CHECK (retired_reason IS NULL OR retired_reason IN ({_REASONS})),
        superseded_by      TEXT,
        invalidated_at     TEXT,
        invalidated_reason TEXT    CHECK (invalidated_reason IS NULL
                                          OR invalidated_reason IN ({_INVALIDATING})),
        detail             TEXT,
        CHECK ((state = 'RETIRED') = (retired_at IS NOT NULL AND retired_reason IS NOT NULL)),
        CHECK ((invalidated_at IS NULL) = (invalidated_reason IS NULL)),
        CHECK (invalidated_at IS NULL OR state = 'RETIRED')
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS one_active_prior_per_key "
    "ON priors(kind, key_hash) WHERE state = 'ACTIVE'",
    "CREATE INDEX IF NOT EXISTS priors_by_anchor ON priors(kind, anchor, seq)",
    """CREATE TABLE IF NOT EXISTS prior_datasets (
        prior_id        TEXT    NOT NULL REFERENCES priors(prior_id),
        dataset_id      TEXT    NOT NULL,
        dataset_version INTEGER NOT NULL,
        content_hash    TEXT    NOT NULL,
        PRIMARY KEY (prior_id, dataset_id, dataset_version, content_hash)
    )""",
    """CREATE TABLE IF NOT EXISTS learned_jobs (
        job_id         TEXT PRIMARY KEY,
        outcome        TEXT NOT NULL CHECK (outcome IN ('learned', 'refused')),
        detail         TEXT,
        prior_ids_json TEXT NOT NULL,
        decided_at     TEXT NOT NULL
    )""",
    # -- history is permanent; content is immutable; retirement and invalidation are final ------
    """CREATE TRIGGER IF NOT EXISTS priors_are_permanent BEFORE DELETE ON priors
    BEGIN SELECT RAISE(ABORT, 'priors are never deleted'); END""",
    """CREATE TRIGGER IF NOT EXISTS prior_datasets_are_permanent BEFORE DELETE ON prior_datasets
    BEGIN SELECT RAISE(ABORT, 'priors are never deleted'); END""",
    """CREATE TRIGGER IF NOT EXISTS learned_jobs_are_permanent BEFORE DELETE ON learned_jobs
    BEGIN SELECT RAISE(ABORT, 'learning decisions are permanent'); END""",
    """CREATE TRIGGER IF NOT EXISTS learned_jobs_are_immutable BEFORE UPDATE ON learned_jobs
    BEGIN SELECT RAISE(ABORT, 'learning decisions are permanent'); END""",
    """CREATE TRIGGER IF NOT EXISTS prior_content_is_immutable BEFORE UPDATE ON priors
    WHEN NEW.prior_id IS NOT OLD.prior_id OR NEW.seq IS NOT OLD.seq OR NEW.kind IS NOT OLD.kind
      OR NEW.key_hash IS NOT OLD.key_hash OR NEW.anchor IS NOT OLD.anchor
      OR NEW.source_job_id IS NOT OLD.source_job_id OR NEW.record_json IS NOT OLD.record_json
      OR NEW.created_at IS NOT OLD.created_at
    BEGIN SELECT RAISE(ABORT, 'prior records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS prior_retirement_is_final BEFORE UPDATE ON priors
    WHEN OLD.state = 'RETIRED' AND (NEW.state IS NOT 'RETIRED'
      OR NEW.retired_at IS NOT OLD.retired_at OR NEW.retired_reason IS NOT OLD.retired_reason
      OR NEW.superseded_by IS NOT OLD.superseded_by)
    BEGIN SELECT RAISE(ABORT, 'a retired prior stays retired'); END""",
    """CREATE TRIGGER IF NOT EXISTS prior_invalidation_is_final BEFORE UPDATE ON priors
    WHEN OLD.invalidated_at IS NOT NULL AND (NEW.invalidated_at IS NOT OLD.invalidated_at
      OR NEW.invalidated_reason IS NOT OLD.invalidated_reason)
    BEGIN SELECT RAISE(ABORT, 'an invalidated prior stays invalidated'); END""",
)
_COLS = (
    "prior_id, seq, record_json, created_at, state, retired_at, retired_reason, superseded_by, "
    "invalidated_at, invalidated_reason, detail"
)


class PriorIntegrityError(RepositoryError):
    """A stored prior does not hash to its id (tampered or corrupted): fail closed."""


@dataclass(frozen=True)
class PriorRow:
    prior_id: str
    seq: int
    record: PriorRecord
    created_at: str
    state: str  # ACTIVE | RETIRED
    retired_at: str | None
    retired_reason: RetireReason | None
    superseded_by: str | None
    invalidated_at: str | None
    invalidated_reason: RetireReason | None
    detail: str | None

    @property
    def active(self) -> bool:
        return self.state == "ACTIVE"

    @property
    def invalidated(self) -> bool:
        return self.invalidated_at is not None


@dataclass(frozen=True)
class Learned:
    job_id: str
    outcome: str  # learned | refused
    detail: str | None
    prior_ids: tuple[str, ...]
    decided_at: str


def _row(r: Sequence[Any]) -> PriorRow:
    prior_id, record_json = r[0], r[2]
    try:
        record = PriorRecord.model_validate_json(record_json)
    except ValueError as exc:
        raise PriorIntegrityError(f"prior {prior_id} does not parse: {exc}") from None
    if record.prior_id != prior_id:
        raise PriorIntegrityError(f"prior {prior_id} does not hash to its id")
    return PriorRow(
        prior_id=prior_id,
        seq=r[1],
        record=record,
        created_at=r[3],
        state=r[4],
        retired_at=r[5],
        retired_reason=RetireReason(r[6]) if r[6] else None,
        superseded_by=r[7],
        invalidated_at=r[8],
        invalidated_reason=RetireReason(r[9]) if r[9] else None,
        detail=r[10],
    )


def _rank(record: PriorRecord) -> tuple[float, ...]:
    assert record.champion is not None
    return tuple(record.champion.rank_key)


class SQLitePriorStore:
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
                row = c.execute("SELECT value FROM meta WHERE key='priors_schema_version'")
                found = row.fetchone()
                if found is None:
                    c.execute(
                        "INSERT INTO meta(key, value) VALUES ('priors_schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                elif int(found[0]) != SCHEMA_VERSION:
                    raise RepositoryError(f"unsupported prior store schema version {found[0]}")

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

    # -- reads ------------------------------------------------------------------------------
    def get(self, prior_id: str) -> PriorRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLS} FROM priors WHERE prior_id=?", (prior_id,)
            ).fetchone()
        return _row(row) if row is not None else None

    def candidates(self, kind: PriorKind, anchor: str) -> list[tuple[str, PriorRow | None]]:
        """Every prior of ``kind`` under ``anchor``, newest first, as ``(prior_id, row)``; the
        row is ``None`` when it does not verify (the caller excludes it, fail closed)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLS} FROM priors WHERE kind=? AND anchor=? ORDER BY seq DESC",
                (kind.value, anchor),
            ).fetchall()
        out: list[tuple[str, PriorRow | None]] = []
        for r in rows:
            try:
                out.append((r[0], _row(r)))
            except PriorIntegrityError:
                out.append((r[0], None))
        return out

    def state_of(self, prior_id: str) -> tuple[str, str | None] | None:
        """``(state, invalidated_reason)`` without parsing the record (``None``: no such prior)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state, invalidated_reason FROM priors WHERE prior_id=?", (prior_id,)
            ).fetchone()
        return (row[0], row[1]) if row is not None else None

    def history(self, kind: PriorKind | None = None) -> list[PriorRow]:
        """Every prior ever stored (active and retired), oldest first."""
        where, params = ("WHERE kind=? ", (kind.value,)) if kind is not None else ("", ())
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLS} FROM priors {where}ORDER BY seq", params
            ).fetchall()
        return [_row(r) for r in rows]

    def active(self, kind: PriorKind | None = None) -> list[PriorRow]:
        return [r for r in self.history(kind) if r.active]

    def learned(self, job_id: str) -> Learned | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id, outcome, detail, prior_ids_json, decided_at FROM learned_jobs "
                "WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return Learned(row[0], row[1], row[2], tuple(json.loads(row[3])), row[4])

    def dataset_refs(self) -> dict[str, list[DatasetRef]]:
        """``prior_id -> dataset versions it depends on`` for every not-yet-invalidated prior."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.prior_id, d.dataset_id, d.dataset_version, d.content_hash "
                "FROM prior_datasets d JOIN priors p ON p.prior_id = d.prior_id "
                "WHERE p.invalidated_at IS NULL ORDER BY p.seq, d.dataset_id, d.dataset_version"
            ).fetchall()
        out: dict[str, list[DatasetRef]] = {}
        for prior_id, dataset_id, version, content_hash in rows:
            out.setdefault(prior_id, []).append(
                DatasetRef(
                    dataset_id=dataset_id, dataset_version=version, content_hash=content_hash
                )
            )
        return out

    # -- writes -----------------------------------------------------------------------------
    def learn(
        self, job_id: str, records: Sequence[PriorRecord], detail: str | None = None
    ) -> Learned:
        """Record what one job taught, in one transaction - exactly once per job.

        A task-family prior becomes the ACTIVE head of its key and retires the previous head
        (``superseded``): pheromone maps are never merged. A dataset champion becomes the head
        only if it ranks strictly higher than the current one (same key = same validation rows
        and contract, so ranks are comparable); otherwise it is stored already retired
        (``dominated``). A prior already stored (same content) is not written again."""

        def write(c: _Conn) -> Learned:
            found = c.execute(
                "SELECT job_id, outcome, detail, prior_ids_json, decided_at FROM learned_jobs "
                "WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if found is not None:
                return Learned(found[0], found[1], found[2], tuple(json.loads(found[3])), found[4])
            now = self.clock()
            ids = []
            for record in records:
                ids.append(self._insert(c, record, now))
            outcome = "learned" if records else "refused"
            c.execute(
                "INSERT INTO learned_jobs(job_id, outcome, detail, prior_ids_json, decided_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, outcome, detail, json.dumps(ids), now),
            )
            return Learned(job_id, outcome, detail, tuple(ids), now)

        return self._tx(write)

    def _insert(self, c: _Conn, record: PriorRecord, now: str) -> str:
        prior_id = record.prior_id
        if c.execute("SELECT 1 FROM priors WHERE prior_id=?", (prior_id,)).fetchone():
            return prior_id
        head = c.execute(
            f"SELECT {_COLS} FROM priors WHERE kind=? AND key_hash=? AND state='ACTIVE'",
            (record.kind.value, record.key_hash),
        ).fetchone()
        state, reason, by = "ACTIVE", None, None
        if head is not None:
            current = _row(head)
            if record.kind is PriorKind.DATASET_CHAMPION and _rank(record) <= _rank(current.record):
                state, reason, by = "RETIRED", RetireReason.DOMINATED, current.prior_id
            else:
                c.execute(
                    "UPDATE priors SET state='RETIRED', retired_at=?, retired_reason=?, "
                    "superseded_by=? WHERE prior_id=?",
                    (now, RetireReason.SUPERSEDED.value, prior_id, current.prior_id),
                )
        seq = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM priors").fetchone()[0]
        c.execute(
            f"INSERT INTO priors({_COLS}, kind, key_hash, anchor, source_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)",
            (
                prior_id,
                seq,
                json.dumps(record.dump(), sort_keys=True, separators=(",", ":")),
                now,
                state,
                now if reason else None,
                reason.value if reason else None,
                by,
                record.kind.value,
                record.key_hash,
                record.anchor,
                record.source.job_id,
            ),
        )
        for d in record.evidence.datasets:
            c.execute(
                "INSERT INTO prior_datasets(prior_id, dataset_id, dataset_version, content_hash) "
                "VALUES (?, ?, ?, ?)",
                (prior_id, d.dataset_id, d.dataset_version, d.content_hash),
            )
        return prior_id

    def invalidate(self, prior_id: str, reason: RetireReason, detail: str | None = None) -> bool:
        """The prior's authority no longer holds: retire it (if active) and mark it unusable,
        also for jobs that already pinned it. Idempotent; returns whether anything changed."""
        if reason not in INVALIDATING:
            raise ValueError(f"{reason.value} does not invalidate a prior")

        def write(c: _Conn) -> bool:
            row = c.execute(
                "SELECT state, invalidated_at FROM priors WHERE prior_id=?", (prior_id,)
            ).fetchone()
            if row is None or row[1] is not None:
                return False
            now = self.clock()
            if row[0] == "ACTIVE":
                c.execute(
                    "UPDATE priors SET state='RETIRED', retired_at=?, retired_reason=? "
                    "WHERE prior_id=?",
                    (now, reason.value, prior_id),
                )
            c.execute(
                "UPDATE priors SET invalidated_at=?, invalidated_reason=?, detail=? "
                "WHERE prior_id=?",
                (now, reason.value, detail, prior_id),
            )
            return True

        return bool(self._tx(write))


__all__ = ["Learned", "PriorIntegrityError", "PriorRow", "SQLitePriorStore"]
