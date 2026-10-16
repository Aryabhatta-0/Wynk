"""Append-only audit trail of security-sensitive dataset actions (#33).

One ``audit.sqlite3`` per workspace partition (bound to it like every other store, #32), plus one
unbound ``security_audit.sqlite3`` at the data-dir root for requests that never reached a
workspace (failed authentication).

Each event records actor (user id + API key id, or a ``system:`` actor), workspace, action,
resource type + id, time, result, the stable error code and a few allowlisted details. Results:
``requested`` (written BEFORE an irreversible action - deletion, upload - so it can never happen
unaudited), ``ok``, ``refused`` (validation / not found / policy), ``denied`` (authentication or
scope) and ``error``. Append-only by construction:

* SQLite triggers refuse every UPDATE and DELETE on ``audit_events``;
* events are hash-chained: ``event_hash = sha256(prev_hash || canonical event)``, so a removed,
  reordered or edited row is detected by ``verify_chain``.

Redaction by construction: the schema has no free-text field. ``details`` keys are an allowlist
and every string value must match ``_SAFE`` (identifiers, hashes, codes, timestamps), so a raw
uploaded value, an API key secret, a filename or a detected PII string cannot be recorded even
by mistake - the append refuses (``ValueError``) instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.canonical import canonical_json
from store.tenancy import bind_sqlite

Result = Literal["requested", "ok", "refused", "denied", "error"]

# identifiers, hex hashes, error codes, versions (``sensitive-data/1``), ISO timestamps and
# comma lists of categories: never whitespace, quotes, ``@`` or free text
_SAFE = re.compile(r"^[A-Za-z0-9_.:/+,-]{0,160}$")
DETAIL_KEYS = frozenset(
    {
        "content_hash",
        "identity_hash",
        "format",
        "export_format",
        "size_bytes",
        "row_count",
        "created",
        "categories",
        "category",
        "redacted_values",
        "formula_like_values",
        "neutralized_values",
        "findings",
        "neutralized_cells",
        "content_removed",
        "content_shared_with",
        "affected_versions",
        "reason",
        "scope",
        "route",
        "status",
        "key_id",
        "policy_version",
        "dataset_version",
        "uploads",
        "upload_id",
    }
)
GENESIS = "0" * 64


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    event_id: str
    at: str
    workspace_id: str | None
    actor_user_id: str | None
    actor_key_id: str | None
    action: str = Field(pattern=r"^[a-z_]+(\.[a-z_]+)+$")
    resource_type: str = Field(pattern=r"^[a-z_]{1,32}$")
    resource_id: str | None
    result: Result
    code: str | None
    details: dict[str, str | int | bool | None]
    prev_hash: str
    event_hash: str


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS audit_events (
        seq           INTEGER PRIMARY KEY,
        event_id      TEXT NOT NULL UNIQUE,
        at            TEXT NOT NULL,
        workspace_id  TEXT,
        actor_user_id TEXT,
        actor_key_id  TEXT,
        action        TEXT NOT NULL,
        resource_type TEXT NOT NULL,
        resource_id   TEXT,
        result        TEXT NOT NULL
                      CHECK (result IN ('requested', 'ok', 'refused', 'denied', 'error')),
        code          TEXT,
        details_json  TEXT NOT NULL,
        prev_hash     TEXT NOT NULL,
        event_hash    TEXT NOT NULL
    )""",
    "CREATE TRIGGER IF NOT EXISTS audit_events_append_only_u BEFORE UPDATE ON audit_events "
    "BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS audit_events_append_only_d BEFORE DELETE ON audit_events "
    "BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END",
)


def _safe(name: str, value: str | None) -> str | None:
    if value is not None and not _SAFE.match(value):
        raise ValueError(f"audit field {name} holds a value that is not an identifier")
    return value


def _event_hash(prev: str, body: Mapping[str, object]) -> str:
    return hashlib.sha256((prev + canonical_json(dict(body))).encode()).hexdigest()


class SQLiteAuditLog:
    def __init__(self, path: Path | str, *, workspace_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if workspace_id is not None:
            bind_sqlite(self.path, workspace_id)  # #32: before any read or write
        self.workspace_id = workspace_id
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            for statement in SCHEMA:
                conn.execute(statement)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
        finally:
            conn.close()

    def append(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str | None,
        result: Result,
        actor_user_id: str | None = None,
        actor_key_id: str | None = None,
        code: str | None = None,
        details: Mapping[str, str | int | bool | None] | None = None,
        at: str | None = None,
        workspace_id: str | None = None,
    ) -> AuditEvent:
        clean: dict[str, str | int | bool | None] = {}
        for key, value in sorted((details or {}).items()):
            if key not in DETAIL_KEYS:
                raise ValueError(f"audit detail {key!r} is not allowlisted")
            if isinstance(value, str):
                _safe(key, value)
            elif value is not None and not isinstance(value, (bool, int)):
                raise ValueError(f"audit detail {key!r} must be a scalar")
            clean[key] = value
        body = {
            "event_id": "ae-" + uuid.uuid4().hex,
            "at": at or datetime.now(UTC).isoformat(timespec="milliseconds"),
            "workspace_id": _safe("workspace_id", workspace_id or self.workspace_id),
            "actor_user_id": _safe("actor_user_id", actor_user_id),
            "actor_key_id": _safe("actor_key_id", actor_key_id),
            "action": action,
            "resource_type": resource_type,
            "resource_id": _safe("resource_id", resource_id),
            "result": result,
            "code": _safe("code", code),
            "details": clean,
        }
        AuditEvent.model_validate(body | {"seq": 0, "prev_hash": GENESIS, "event_hash": GENESIS})
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS)
                digest = _event_hash(prev, body | {"seq": seq})
                conn.execute(
                    "INSERT INTO audit_events(seq, event_id, at, workspace_id, actor_user_id, "
                    "actor_key_id, action, resource_type, resource_id, result, code, "
                    "details_json, prev_hash, event_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        seq,
                        body["event_id"],
                        body["at"],
                        body["workspace_id"],
                        body["actor_user_id"],
                        body["actor_key_id"],
                        action,
                        resource_type,
                        body["resource_id"],
                        result,
                        body["code"],
                        canonical_json(clean),
                        prev,
                        digest,
                    ),
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        return AuditEvent(**(body | {"seq": seq, "prev_hash": prev, "event_hash": digest}))

    def events(self, after: int = 0, limit: int = 100) -> list[AuditEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, event_id, at, workspace_id, actor_user_id, actor_key_id, action, "
                "resource_type, resource_id, result, code, details_json, prev_hash, event_hash "
                "FROM audit_events WHERE seq > ? ORDER BY seq LIMIT ?",
                (after, limit),
            ).fetchall()
        return [
            AuditEvent(
                seq=r[0],
                event_id=r[1],
                at=r[2],
                workspace_id=r[3],
                actor_user_id=r[4],
                actor_key_id=r[5],
                action=r[6],
                resource_type=r[7],
                resource_id=r[8],
                result=r[9],
                code=r[10],
                details=json.loads(r[11]),
                prev_hash=r[12],
                event_hash=r[13],
            )
            for r in rows
        ]

    def verify_chain(self) -> int:
        """Re-derive every hash in order. Returns the event count; ``ValueError`` on a break."""
        prev, seq, n = GENESIS, 0, 0
        while True:
            batch = self.events(after=seq, limit=1000)
            if not batch:
                return n
            for e in batch:
                if e.seq != seq + 1 or e.prev_hash != prev:
                    raise ValueError(f"audit chain broken at event {e.seq}")
                body = e.model_dump(exclude={"prev_hash", "event_hash"})
                if _event_hash(prev, body) != e.event_hash:
                    raise ValueError(f"audit event {e.seq} was altered")
                prev, seq, n = e.event_hash, e.seq, n + 1
