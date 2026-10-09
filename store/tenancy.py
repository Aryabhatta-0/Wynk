"""Store-level tenant binding (#32).

Every durable store of a workspace lives in that workspace's own partition
(``<data>/workspaces/<workspace_id>/``) and carries an immutable binding to it:

* a SQLite store holds one ``tenant_binding`` row (``singleton = 1``) with the workspace id,
  guarded by UPDATE / DELETE triggers: a store can never change workspace after creation;
* a blob store holds a ``TENANT`` file, written once (``O_EXCL``).

Opening a store checks its binding BEFORE any read or write, independently of the API layer:

* opened for workspace ``W``: an unbound store is bound to ``W``; a store bound to another
  workspace is refused (``TenantBindingError``);
* opened with no workspace (library / single-tenant use): a store that IS bound is refused, so
  tenant data is never reachable through an unscoped handle.

The partition is the canonical tenant authority for everything inside it (projects, uploads,
datasets, splits, jobs, artifacts, promotions, workflow versions, deployments, inferences,
feedback, policies, trigger decisions, re-optimizations): every derived resource inherits its
workspace from the store it is written to, never from a caller-supplied id.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE_ID = re.compile(r"^ws-[0-9a-z][0-9a-z-]{1,62}$")
BLOB_BINDING = "TENANT"

_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS tenant_binding ("
    " singleton INTEGER PRIMARY KEY CHECK (singleton = 1),"
    " workspace_id TEXT NOT NULL,"
    " bound_at TEXT NOT NULL)",
    "CREATE TRIGGER IF NOT EXISTS tenant_binding_no_update BEFORE UPDATE ON tenant_binding "
    "BEGIN SELECT RAISE(ABORT, 'a store never changes workspace'); END",
    "CREATE TRIGGER IF NOT EXISTS tenant_binding_no_delete BEFORE DELETE ON tenant_binding "
    "BEGIN SELECT RAISE(ABORT, 'a store never changes workspace'); END",
)


class TenantBindingError(Exception):
    """A store is bound to a different workspace than the one it is opened for (or to one at
    all, when opened unscoped): fail closed, never rebind."""


def check_workspace_id(workspace_id: str) -> str:
    if not WORKSPACE_ID.match(workspace_id):
        raise ValueError(f"not a workspace id: {workspace_id!r}")
    return workspace_id


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def sqlite_binding(path: Path | str) -> str | None:
    """The workspace a SQLite store is bound to (``None``: unbound or no such file)."""
    path = Path(path)
    if not path.exists():
        return None
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tenant_binding'"
        ).fetchone()
        if not has:
            return None
        row = conn.execute("SELECT workspace_id FROM tenant_binding WHERE singleton=1").fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def bind_sqlite(path: Path | str, workspace_id: str | None) -> None:
    """Check (and on first use, create) a SQLite store's workspace binding. Atomic: the check
    and the insert run in one ``BEGIN IMMEDIATE`` transaction."""
    if workspace_id is not None:
        check_workspace_id(workspace_id)
    conn = sqlite3.connect(Path(path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tenant_binding'"
            ).fetchone()
            row = None
            if exists:
                row = conn.execute(
                    "SELECT workspace_id FROM tenant_binding WHERE singleton=1"
                ).fetchone()
            if row is None:
                if workspace_id is not None:
                    for statement in _STATEMENTS:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO tenant_binding(singleton, workspace_id, bound_at) "
                        "VALUES (1, ?, ?)",
                        (workspace_id, _now()),
                    )
            elif row[0] != workspace_id:
                raise TenantBindingError(
                    f"{Path(path).name} belongs to another workspace"
                    if workspace_id is not None
                    else f"{Path(path).name} belongs to a workspace; open it scoped to one"
                )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def bind_directory(root: Path | str, workspace_id: str | None) -> None:
    """The same binding for a directory store (blobs): a ``TENANT`` file written once."""
    root = Path(root)
    marker = root / BLOB_BINDING
    if workspace_id is not None:
        check_workspace_id(workspace_id)
        root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w") as f:
                f.write(workspace_id)
                f.flush()
                os.fsync(f.fileno())
    bound = marker.read_text().strip() if marker.exists() else None
    if bound != workspace_id and not (bound is None and workspace_id is None):
        raise TenantBindingError(
            f"{root.name} belongs to another workspace"
            if workspace_id is not None
            else f"{root.name} belongs to a workspace; open it scoped to one"
        )


__all__ = [
    "BLOB_BINDING",
    "WORKSPACE_ID",
    "TenantBindingError",
    "bind_directory",
    "bind_sqlite",
    "check_workspace_id",
    "sqlite_binding",
]
