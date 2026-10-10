"""Identity repository (#32): users, workspaces, memberships, API keys, legacy migrations.

One SQLite file at the data-dir root (``identity.sqlite3``); it holds no tenant resource data,
only who may act on which workspace partition. Every table is append-only except the few
columns that record a one-way transition, each guarded by a trigger:

* ``users`` / ``workspaces``: immutable rows (UPDATE / DELETE refused).
* ``memberships``: immutable except ``removed_at`` / ``removed_by``, set once (NULL -> value).
  A user holds at most one ACTIVE membership per workspace (partial unique index); a role
  change is a removal plus a new membership, so history is never rewritten.
* ``api_keys``: immutable except ``revoked_at`` / ``revoked_by`` (set once) and
  ``last_used_at`` (monotonic). Only a hash of the secret is stored - never the plaintext.
* ``legacy_migrations``: one row per migrated data dir; ``completed_at`` set once.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Role(StrEnum):
    OWNER = "OWNER"
    MEMBER = "MEMBER"


class IdentityError(Exception):
    pass


class IdentityConflict(IdentityError):
    pass


SCHEMA_VERSION = "1"
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS users (
        user_id      TEXT PRIMARY KEY,
        email        TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        created_at   TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS workspaces (
        workspace_id TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        created_by   TEXT NOT NULL REFERENCES users(user_id),
        created_at   TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS memberships (
        membership_id TEXT PRIMARY KEY,
        workspace_id  TEXT NOT NULL REFERENCES workspaces(workspace_id),
        user_id       TEXT NOT NULL REFERENCES users(user_id),
        role          TEXT NOT NULL CHECK (role IN ('OWNER', 'MEMBER')),
        created_by    TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        removed_by    TEXT,
        removed_at    TEXT)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS one_active_membership "
    "ON memberships(workspace_id, user_id) WHERE removed_at IS NULL",
    """CREATE TABLE IF NOT EXISTS api_keys (
        key_id       TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
        user_id      TEXT NOT NULL REFERENCES users(user_id),
        name         TEXT NOT NULL,
        scopes       TEXT NOT NULL,
        secret_hash  TEXT NOT NULL,
        created_by   TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        expires_at   TEXT,
        revoked_by   TEXT,
        revoked_at   TEXT,
        last_used_at TEXT)""",
    "CREATE INDEX IF NOT EXISTS keys_by_workspace ON api_keys(workspace_id)",
    """CREATE TABLE IF NOT EXISTS legacy_migrations (
        source        TEXT PRIMARY KEY,
        workspace_id  TEXT NOT NULL REFERENCES workspaces(workspace_id),
        owner_user_id TEXT NOT NULL REFERENCES users(user_id),
        started_at    TEXT NOT NULL,
        completed_at  TEXT)""",
    # -- immutability ---------------------------------------------------------------------
    "CREATE TRIGGER IF NOT EXISTS users_immutable BEFORE UPDATE ON users "
    "BEGIN SELECT RAISE(ABORT, 'users are immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS users_no_delete BEFORE DELETE ON users "
    "BEGIN SELECT RAISE(ABORT, 'users are immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS workspaces_immutable BEFORE UPDATE ON workspaces "
    "BEGIN SELECT RAISE(ABORT, 'workspaces are immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS workspaces_no_delete BEFORE DELETE ON workspaces "
    "BEGIN SELECT RAISE(ABORT, 'workspaces are immutable'); END",
    "CREATE TRIGGER IF NOT EXISTS memberships_no_delete BEFORE DELETE ON memberships "
    "BEGIN SELECT RAISE(ABORT, 'memberships are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS memberships_removal_only BEFORE UPDATE ON memberships "
    "WHEN OLD.membership_id IS NOT NEW.membership_id OR OLD.workspace_id IS NOT NEW.workspace_id "
    "OR OLD.user_id IS NOT NEW.user_id OR OLD.role IS NOT NEW.role "
    "OR OLD.created_by IS NOT NEW.created_by OR OLD.created_at IS NOT NEW.created_at "
    "OR OLD.removed_at IS NOT NULL "
    "BEGIN SELECT RAISE(ABORT, 'a membership is only ever removed, once'); END",
    "CREATE TRIGGER IF NOT EXISTS api_keys_no_delete BEFORE DELETE ON api_keys "
    "BEGIN SELECT RAISE(ABORT, 'api keys are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS api_keys_identity BEFORE UPDATE ON api_keys "
    "WHEN OLD.key_id IS NOT NEW.key_id OR OLD.workspace_id IS NOT NEW.workspace_id "
    "OR OLD.user_id IS NOT NEW.user_id OR OLD.name IS NOT NEW.name "
    "OR OLD.scopes IS NOT NEW.scopes OR OLD.secret_hash IS NOT NEW.secret_hash "
    "OR OLD.created_by IS NOT NEW.created_by OR OLD.created_at IS NOT NEW.created_at "
    "OR OLD.expires_at IS NOT NEW.expires_at "
    "OR (OLD.revoked_at IS NOT NULL AND OLD.revoked_at IS NOT NEW.revoked_at) "
    "OR (OLD.revoked_by IS NOT NULL AND OLD.revoked_by IS NOT NEW.revoked_by) "
    "OR (NEW.last_used_at IS NULL AND OLD.last_used_at IS NOT NULL) "
    "OR (OLD.last_used_at IS NOT NULL AND NEW.last_used_at < OLD.last_used_at) "
    "BEGIN SELECT RAISE(ABORT, 'an api key only records revocation and use'); END",
    "CREATE TRIGGER IF NOT EXISTS migrations_no_delete BEFORE DELETE ON legacy_migrations "
    "BEGIN SELECT RAISE(ABORT, 'migrations are append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS migrations_complete_once BEFORE UPDATE ON legacy_migrations "
    "WHEN OLD.source IS NOT NEW.source OR OLD.workspace_id IS NOT NEW.workspace_id "
    "OR OLD.owner_user_id IS NOT NEW.owner_user_id OR OLD.started_at IS NOT NEW.started_at "
    "OR OLD.completed_at IS NOT NULL "
    "BEGIN SELECT RAISE(ABORT, 'a migration completes once'); END",
)


@dataclass(frozen=True)
class UserRow:
    user_id: str
    email: str
    display_name: str
    created_at: str


@dataclass(frozen=True)
class WorkspaceRow:
    workspace_id: str
    name: str
    created_by: str
    created_at: str


@dataclass(frozen=True)
class MembershipRow:
    membership_id: str
    workspace_id: str
    user_id: str
    role: Role
    created_by: str
    created_at: str
    removed_by: str | None
    removed_at: str | None


@dataclass(frozen=True)
class KeyRow:
    key_id: str
    workspace_id: str
    user_id: str
    name: str
    scopes: tuple[str, ...]
    secret_hash: str
    created_by: str
    created_at: str
    expires_at: str | None
    revoked_by: str | None
    revoked_at: str | None
    last_used_at: str | None


@dataclass(frozen=True)
class MigrationRow:
    source: str
    workspace_id: str
    owner_user_id: str
    started_at: str
    completed_at: str | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


_KEY_COLS = (
    "key_id, workspace_id, user_id, name, scopes, secret_hash, created_by, created_at, "
    "expires_at, revoked_by, revoked_at, last_used_at"
)
_MEMBER_COLS = (
    "membership_id, workspace_id, user_id, role, created_by, created_at, removed_by, removed_at"
)


class SQLiteIdentityStore:
    def __init__(self, path: Path | str, clock: Callable[[], str] = utc_now) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            def create(c: sqlite3.Connection) -> None:
                for statement in SCHEMA:
                    c.execute(statement)
                row = c.execute("SELECT value FROM meta WHERE key='identity_schema'").fetchone()
                if row is None:
                    c.execute(
                        "INSERT INTO meta(key, value) VALUES ('identity_schema', ?)",
                        (SCHEMA_VERSION,),
                    )
                elif row[0] != SCHEMA_VERSION:
                    raise IdentityError(f"unsupported identity schema {row[0]}")

            self._tx(conn, create)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _tx(conn: sqlite3.Connection, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn.execute("BEGIN IMMEDIATE")
        try:
            out = fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return out

    def write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """One atomic write transaction (several rows commit together or not at all)."""
        with self._connect() as conn:
            try:
                return self._tx(conn, fn)
            except sqlite3.IntegrityError as exc:
                raise IdentityConflict(str(exc)) from None

    # -- users ------------------------------------------------------------------------------
    @staticmethod
    def insert_user(c: sqlite3.Connection, row: UserRow) -> None:
        c.execute(
            "INSERT INTO users(user_id, email, display_name, created_at) VALUES (?,?,?,?)",
            (row.user_id, row.email, row.display_name, row.created_at),
        )

    def user(self, user_id: str) -> UserRow | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT user_id, email, display_name, created_at FROM users WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return UserRow(*r) if r else None

    def user_by_email(self, email: str) -> UserRow | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT user_id, email, display_name, created_at FROM users WHERE email=?",
                (email,),
            ).fetchone()
        return UserRow(*r) if r else None

    # -- workspaces -------------------------------------------------------------------------
    @staticmethod
    def insert_workspace(c: sqlite3.Connection, row: WorkspaceRow) -> None:
        c.execute(
            "INSERT INTO workspaces(workspace_id, name, created_by, created_at) VALUES (?,?,?,?)",
            (row.workspace_id, row.name, row.created_by, row.created_at),
        )

    def workspace(self, workspace_id: str) -> WorkspaceRow | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT workspace_id, name, created_by, created_at FROM workspaces "
                "WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
        return WorkspaceRow(*r) if r else None

    def workspaces(self) -> list[WorkspaceRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT workspace_id, name, created_by, created_at FROM workspaces "
                "ORDER BY created_at, workspace_id"
            ).fetchall()
        return [WorkspaceRow(*r) for r in rows]

    # -- memberships ------------------------------------------------------------------------
    @staticmethod
    def insert_membership(c: sqlite3.Connection, row: MembershipRow) -> None:
        c.execute(
            f"INSERT INTO memberships({_MEMBER_COLS}) VALUES (?,?,?,?,?,?,?,?)",
            (
                row.membership_id,
                row.workspace_id,
                row.user_id,
                row.role.value,
                row.created_by,
                row.created_at,
                None,
                None,
            ),
        )

    def membership(self, workspace_id: str, user_id: str) -> MembershipRow | None:
        """The user's ACTIVE membership of the workspace."""
        with self._connect() as conn:
            r = conn.execute(
                f"SELECT {_MEMBER_COLS} FROM memberships WHERE workspace_id=? AND user_id=? "
                "AND removed_at IS NULL",
                (workspace_id, user_id),
            ).fetchone()
        return _member(r) if r else None

    def memberships(self, workspace_id: str) -> list[MembershipRow]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_MEMBER_COLS} FROM memberships WHERE workspace_id=? "
                "AND removed_at IS NULL ORDER BY created_at, membership_id",
                (workspace_id,),
            ).fetchall()
        return [_member(r) for r in rows]

    def remove_membership(self, workspace_id: str, user_id: str, by: str) -> bool:
        """Remove the active membership, refusing to remove the last OWNER. ``bool``: removed."""

        def remove(c: sqlite3.Connection) -> bool:
            r = c.execute(
                "SELECT membership_id, role FROM memberships WHERE workspace_id=? AND user_id=? "
                "AND removed_at IS NULL",
                (workspace_id, user_id),
            ).fetchone()
            if r is None:
                return False
            if r[1] == Role.OWNER.value:
                owners = c.execute(
                    "SELECT COUNT(*) FROM memberships WHERE workspace_id=? AND role='OWNER' "
                    "AND removed_at IS NULL",
                    (workspace_id,),
                ).fetchone()[0]
                if owners <= 1:
                    raise IdentityConflict("a workspace keeps at least one OWNER")
            c.execute(
                "UPDATE memberships SET removed_at=?, removed_by=? WHERE membership_id=?",
                (self.clock(), by, r[0]),
            )
            return True

        return bool(self.write(remove))

    # -- api keys ---------------------------------------------------------------------------
    @staticmethod
    def insert_key(c: sqlite3.Connection, row: KeyRow) -> None:
        c.execute(
            f"INSERT INTO api_keys({_KEY_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row.key_id,
                row.workspace_id,
                row.user_id,
                row.name,
                ",".join(row.scopes),
                row.secret_hash,
                row.created_by,
                row.created_at,
                row.expires_at,
                None,
                None,
                None,
            ),
        )

    def key(self, key_id: str) -> KeyRow | None:
        with self._connect() as conn:
            r = conn.execute(
                f"SELECT {_KEY_COLS} FROM api_keys WHERE key_id=?", (key_id,)
            ).fetchone()
        return _key(r) if r else None

    def keys(self, workspace_id: str) -> list[KeyRow]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_KEY_COLS} FROM api_keys WHERE workspace_id=? "
                "ORDER BY created_at, key_id",
                (workspace_id,),
            ).fetchall()
        return [_key(r) for r in rows]

    def revoke_key(self, workspace_id: str, key_id: str, by: str) -> KeyRow | None:
        """Revoke a key OF THIS WORKSPACE (idempotent). ``None``: no such key here."""

        def revoke(c: sqlite3.Connection) -> None:
            c.execute(
                "UPDATE api_keys SET revoked_at=?, revoked_by=? "
                "WHERE key_id=? AND workspace_id=? AND revoked_at IS NULL",
                (self.clock(), by, key_id, workspace_id),
            )

        self.write(revoke)
        row = self.key(key_id)
        return row if row is not None and row.workspace_id == workspace_id else None

    def touch_key(self, key_id: str, when: str) -> None:
        with self._connect() as conn:
            self._tx(
                conn,
                lambda c: c.execute(
                    "UPDATE api_keys SET last_used_at=? WHERE key_id=? "
                    "AND (last_used_at IS NULL OR last_used_at < ?)",
                    (when, key_id, when),
                ),
            )

    # -- legacy migrations ------------------------------------------------------------------
    def migration(self, source: str) -> MigrationRow | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT source, workspace_id, owner_user_id, started_at, completed_at "
                "FROM legacy_migrations WHERE source=?",
                (source,),
            ).fetchone()
        return MigrationRow(*r) if r else None

    @staticmethod
    def insert_migration(c: sqlite3.Connection, row: MigrationRow) -> None:
        c.execute(
            "INSERT INTO legacy_migrations(source, workspace_id, owner_user_id, started_at, "
            "completed_at) VALUES (?,?,?,?,NULL)",
            (row.source, row.workspace_id, row.owner_user_id, row.started_at),
        )

    def complete_migration(self, source: str) -> None:
        self.write(
            lambda c: c.execute(
                "UPDATE legacy_migrations SET completed_at=? WHERE source=? "
                "AND completed_at IS NULL",
                (self.clock(), source),
            )
        )


def _member(r: tuple[Any, ...]) -> MembershipRow:
    return MembershipRow(r[0], r[1], r[2], Role(r[3]), *r[4:])


def _key(r: tuple[Any, ...]) -> KeyRow:
    return KeyRow(r[0], r[1], r[2], r[3], tuple(s for s in r[4].split(",") if s), *r[5:])


KEY_COLS = _KEY_COLS
key_row = _key  # #32: for callers that read api_keys inside their own write transaction


__all__ = [
    "KEY_COLS",
    "key_row",
    "IdentityConflict",
    "IdentityError",
    "KeyRow",
    "MembershipRow",
    "MigrationRow",
    "Role",
    "SQLiteIdentityStore",
    "UserRow",
    "WorkspaceRow",
]
