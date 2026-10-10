"""Metadata repository for projects, uploads, dataset versions and splits.

Records wrap the contract models from ``core/dataset.py`` (``DatasetSpec``, ``DatasetSplits``);
they add only what persistence needs (owner project, source upload, row ids, timestamps). Every
record is re-validated when it is read back: a stored ``DatasetSpec`` must still hash to its
recorded identity, stored seeded splits are re-derived from their plan by ``DatasetSplits`` itself,
and stored row ids must match their hash. Anything that fails is refused, never repaired.
This detects corruption and inconsistent edits, not a coherent rewrite (records are unsigned).

``DatasetRepository`` is the narrow interface the ingestion service uses.
``SQLiteDatasetRepository`` is the durable local implementation (stdlib ``sqlite3``, WAL,
``synchronous=FULL``, one transaction per write); its tables map one-to-one onto a later
PostgreSQL schema.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

from core.dataset import SLUG, DatasetFormat, DatasetSpec, DatasetSplits
from ingestion.parse import ColumnProfile, row_ids_hash
from store.tenancy import StoreQuota, admit, bind_sqlite

_SHA256 = r"^[0-9a-f]{64}$"


class RowIdSource(StrEnum):
    COLUMN = "column"  # taken from the dataset's id column
    GENERATED = "generated"  # derived from row content (``ingestion.parse.generated_row_ids``)


class ProjectRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(pattern=SLUG)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    created_at: str


class UploadRecord(BaseModel):
    """An inspected upload. Every field except ``filename`` was computed from the bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    upload_id: str = Field(pattern=SLUG)
    project_id: str = Field(pattern=SLUG)
    filename: str | None = Field(default=None, max_length=255)  # display only
    format: DatasetFormat
    content_hash: str = Field(pattern=_SHA256)
    size_bytes: PositiveInt
    row_count: PositiveInt
    columns: tuple[ColumnProfile, ...] = Field(min_length=1)
    preview: tuple[dict[str, Any], ...]
    parser_version: str
    created_at: str


class DatasetVersionRecord(BaseModel):
    """One registered, immutable dataset version: its ``DatasetSpec`` and how rows are named."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(pattern=SLUG)
    upload_id: str = Field(pattern=SLUG)
    spec: DatasetSpec
    identity_hash: str = Field(pattern=_SHA256)  # == spec.identity_hash
    row_id_source: RowIdSource
    row_id_scheme: str | None  # set iff row ids are generated
    row_ids_hash: str = Field(pattern=_SHA256)
    created_at: str

    @model_validator(mode="after")
    def _consistent(self) -> DatasetVersionRecord:
        if self.identity_hash != self.spec.identity_hash:
            raise ValueError("identity_hash does not match the stored DatasetSpec")
        if (self.row_id_source is RowIdSource.COLUMN) != (self.spec.id_column is not None):
            raise ValueError("row_id_source 'column' requires the spec's id_column, and only it")
        if (self.row_id_source is RowIdSource.GENERATED) != (self.row_id_scheme is not None):
            raise ValueError("row_id_scheme is set iff row ids are generated")
        return self

    @property
    def dataset_id(self) -> str:
        return self.spec.dataset_id

    @property
    def dataset_version(self) -> int:
        return self.spec.dataset_version


class SplitsRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(pattern=SLUG)
    dataset_version: PositiveInt
    splits_hash: str = Field(pattern=_SHA256)  # == splits.identity_hash
    splits: DatasetSplits
    sizes: dict[str, NonNegativeInt]
    created_at: str

    @model_validator(mode="after")
    def _consistent(self) -> SplitsRecord:
        if self.splits_hash != self.splits.identity_hash:
            raise ValueError("splits_hash does not match the stored DatasetSplits")
        if self.sizes != {s.role.value: len(s.row_ids) for s in self.splits.splits}:
            raise ValueError("sizes do not match the stored DatasetSplits")
        return self


# -- errors -------------------------------------------------------------------------------------
class RepositoryError(Exception):
    pass


class Conflict(RepositoryError):
    """The write collides with existing state (e.g. a dataset version number already taken)."""


class IntegrityViolation(RepositoryError):
    """Stored data failed re-validation on read."""


# -- interface ----------------------------------------------------------------------------------
class DatasetRepository(Protocol):
    def create_project(self, project: ProjectRecord) -> None: ...
    def get_project(self, project_id: str) -> ProjectRecord | None: ...
    def list_projects(self) -> list[ProjectRecord]: ...

    def put_upload(self, upload: UploadRecord) -> tuple[UploadRecord, bool]:
        """Insert, or return the stored record with the same id. ``bool``: inserted."""
        ...

    def get_upload(self, upload_id: str) -> UploadRecord | None: ...

    def dataset_owner(self, dataset_id: str) -> str | None:
        """The project that owns ``dataset_id``, or ``None`` if it does not exist."""
        ...

    def add_version(self, record: DatasetVersionRecord, row_ids: tuple[str, ...]) -> None:
        """Insert a new version. ``Conflict`` if the number is taken or another project owns it."""
        ...

    def get_version(self, dataset_id: str, version: int) -> DatasetVersionRecord | None: ...
    def list_versions(self, dataset_id: str) -> list[DatasetVersionRecord]: ...
    def list_dataset_ids(self, project_id: str) -> list[str]: ...
    def get_row_ids(self, dataset_id: str, version: int) -> tuple[str, ...]: ...

    def put_splits(self, record: SplitsRecord) -> tuple[SplitsRecord, bool]: ...
    def get_splits(
        self, dataset_id: str, version: int, splits_hash: str
    ) -> SplitsRecord | None: ...
    def list_splits(self, dataset_id: str, version: int) -> list[SplitsRecord]: ...


# -- SQLite -------------------------------------------------------------------------------------
SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
    upload_id    TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(project_id),
    content_hash TEXT NOT NULL,             -- blob name in the BlobStore
    record_json  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id)
);
CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id      TEXT    NOT NULL REFERENCES datasets(dataset_id),
    dataset_version INTEGER NOT NULL CHECK (dataset_version > 0),
    identity_hash   TEXT    NOT NULL,       -- DatasetSpec.identity_hash
    upload_id       TEXT    NOT NULL REFERENCES uploads(upload_id),
    record_json     TEXT    NOT NULL,
    row_ids_json    TEXT    NOT NULL,       -- JSON array, file order
    PRIMARY KEY (dataset_id, dataset_version)
);
CREATE TABLE IF NOT EXISTS dataset_splits (
    dataset_id      TEXT    NOT NULL,
    dataset_version INTEGER NOT NULL,
    splits_hash     TEXT    NOT NULL,       -- DatasetSplits.identity_hash
    created_at      TEXT    NOT NULL,
    record_json     TEXT    NOT NULL,
    PRIMARY KEY (dataset_id, dataset_version, splits_hash),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES dataset_versions(dataset_id, dataset_version)
);
CREATE INDEX IF NOT EXISTS uploads_by_project ON uploads(project_id);
CREATE INDEX IF NOT EXISTS datasets_by_project ON datasets(project_id);
"""


def _load(model: type[BaseModel], raw: str, what: str) -> Any:
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise IntegrityViolation(f"stored {what} failed validation: {exc}") from None


class SQLiteDatasetRepository:
    quota: StoreQuota | None = None  # #32: set by the tenant router; enforced in-transaction

    def __init__(self, path: Path | str, *, workspace_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        bind_sqlite(self.path, workspace_id)  # #32: before any read or write
        self.workspace_id = workspace_id
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._write(conn, lambda c: c.executescript_safe(SCHEMA))
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._write(
                    conn,
                    lambda c: c.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    ),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise RepositoryError(f"unsupported repository schema version {row[0]}")

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
    def _write(conn: _Conn, fn: Any) -> Any:
        conn.execute("BEGIN IMMEDIATE")
        try:
            out = fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return out

    # -- projects ---------------------------------------------------------------------------
    def create_project(self, project: ProjectRecord) -> None:
        def insert(c: _Conn) -> None:
            if self.quota is not None:
                used = c.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
                admit("projects", self.quota.max_projects, used)
            c.execute(
                "INSERT INTO projects(project_id, created_at, record_json) VALUES (?,?,?)",
                (project.project_id, project.created_at, project.model_dump_json()),
            )

        with self._connect() as conn:
            try:
                self._write(conn, insert)
            except sqlite3.IntegrityError:
                raise Conflict(f"project {project.project_id} already exists") from None

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM projects WHERE project_id=?", (project_id,)
            ).fetchone()
        return self._project(row[0], project_id) if row else None

    def list_projects(self) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT project_id, record_json FROM projects ORDER BY created_at, project_id"
            ).fetchall()
        return [self._project(raw, pid) for pid, raw in rows]

    @staticmethod
    def _project(raw: str, project_id: str) -> ProjectRecord:
        rec: ProjectRecord = _load(ProjectRecord, raw, "project")
        if rec.project_id != project_id:
            raise IntegrityViolation(f"project row {project_id} holds {rec.project_id}")
        return rec

    # -- uploads ----------------------------------------------------------------------------
    def put_upload(self, upload: UploadRecord) -> tuple[UploadRecord, bool]:
        def insert(c: _Conn) -> bool:
            self._admit_upload(c, upload.upload_id, upload.size_bytes)
            cur = c.execute(
                "INSERT INTO uploads(upload_id, project_id, content_hash, record_json) "
                "VALUES (?,?,?,?) ON CONFLICT(upload_id) DO NOTHING",
                (
                    upload.upload_id,
                    upload.project_id,
                    upload.content_hash,
                    upload.model_dump_json(),
                ),
            )
            return cur.rowcount == 1

        with self._connect() as conn:
            try:
                created = self._write(conn, insert)
            except sqlite3.IntegrityError as exc:  # unknown project (foreign key)
                raise Conflict(str(exc)) from None
        stored = self.get_upload(upload.upload_id)
        if stored is None:
            raise IntegrityViolation(f"upload {upload.upload_id} vanished after insert")
        return stored, created

    def _admit_upload(self, c: _Conn, upload_id: str, size: int) -> None:
        if self.quota is None:
            return
        if c.execute("SELECT 1 FROM uploads WHERE upload_id=?", (upload_id,)).fetchone():
            return  # the identical upload already exists: nothing new is stored
        n, total = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(json_extract(record_json, '$.size_bytes')), 0) "
            "FROM uploads"
        ).fetchone()
        admit("uploads", self.quota.max_uploads, n)
        admit("stored_bytes", self.quota.max_stored_bytes, total, size)

    def admit_upload(self, upload_id: str, size: int) -> None:
        """ADVISORY: refuse a NEW upload the quota clearly cannot take before its bytes reach
        the blob store. ``put_upload`` re-checks inside its insert transaction - that row quota
        is the authority. Concurrent unique uploads can both pass this check and write their
        blobs; the loser's metadata insert is refused and its blob may stay unreferenced
        (logical usage never exceeds the quota; orphan-blob cleanup is #33)."""
        with self._connect() as conn:
            self._admit_upload(conn, upload_id, size)

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json, content_hash FROM uploads WHERE upload_id=?", (upload_id,)
            ).fetchone()
        if row is None:
            return None
        rec: UploadRecord = _load(UploadRecord, row[0], "upload")
        if rec.upload_id != upload_id or rec.content_hash != row[1]:
            raise IntegrityViolation(f"upload row {upload_id} does not match its record")
        return rec

    # -- dataset versions -------------------------------------------------------------------
    def dataset_owner(self, dataset_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        return row[0] if row else None

    def add_version(self, record: DatasetVersionRecord, row_ids: tuple[str, ...]) -> None:
        if row_ids_hash(row_ids) != record.row_ids_hash:
            raise IntegrityViolation("row ids do not match the record's row_ids_hash")
        if len(row_ids) != record.spec.row_count:
            raise IntegrityViolation("row id count does not match the spec's row_count")

        def insert(c: _Conn) -> None:
            c.execute(
                "INSERT INTO datasets(dataset_id, project_id) VALUES (?,?) "
                "ON CONFLICT(dataset_id) DO NOTHING",
                (record.dataset_id, record.project_id),
            )
            owner = c.execute(
                "SELECT project_id FROM datasets WHERE dataset_id=?", (record.dataset_id,)
            ).fetchone()[0]
            if owner != record.project_id:
                raise Conflict(f"dataset {record.dataset_id} belongs to another project")
            if self.quota is not None:
                used = c.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
                admit("dataset_versions", self.quota.max_dataset_versions, used)
            c.execute(
                "INSERT INTO dataset_versions(dataset_id, dataset_version, identity_hash, "
                "upload_id, record_json, row_ids_json) VALUES (?,?,?,?,?,?)",
                (
                    record.dataset_id,
                    record.dataset_version,
                    record.identity_hash,
                    record.upload_id,
                    record.model_dump_json(),
                    json.dumps(list(row_ids), ensure_ascii=True, separators=(",", ":")),
                ),
            )

        with self._connect() as conn:
            try:
                self._write(conn, insert)
            except sqlite3.IntegrityError as exc:
                raise Conflict(
                    f"dataset {record.dataset_id} version {record.dataset_version}: {exc}"
                ) from None

    def get_version(self, dataset_id: str, version: int) -> DatasetVersionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dataset_id, dataset_version, identity_hash, record_json "
                "FROM dataset_versions WHERE dataset_id=? AND dataset_version=?",
                (dataset_id, version),
            ).fetchone()
        return self._version(row) if row else None

    def list_versions(self, dataset_id: str) -> list[DatasetVersionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dataset_id, dataset_version, identity_hash, record_json "
                "FROM dataset_versions WHERE dataset_id=? ORDER BY dataset_version",
                (dataset_id,),
            ).fetchall()
        return [self._version(r) for r in rows]

    @staticmethod
    def _version(row: tuple[Any, ...]) -> DatasetVersionRecord:
        dataset_id, version, identity, raw = row
        rec: DatasetVersionRecord = _load(DatasetVersionRecord, raw, "dataset version")
        if (rec.dataset_id, rec.dataset_version, rec.identity_hash) != (
            dataset_id,
            version,
            identity,
        ):
            raise IntegrityViolation(f"dataset version row {dataset_id}/{version} was altered")
        return rec

    def list_dataset_ids(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dataset_id FROM datasets WHERE project_id=? ORDER BY dataset_id",
                (project_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_row_ids(self, dataset_id: str, version: int) -> tuple[str, ...]:
        record = self.get_version(dataset_id, version)
        if record is None:
            raise KeyError(f"{dataset_id}/{version}")
        with self._connect() as conn:
            (raw,) = conn.execute(
                "SELECT row_ids_json FROM dataset_versions "
                "WHERE dataset_id=? AND dataset_version=?",
                (dataset_id, version),
            ).fetchone()
        ids = json.loads(raw)
        if not isinstance(ids, list) or not all(isinstance(r, str) for r in ids):
            raise IntegrityViolation(f"row ids of {dataset_id}/{version} are malformed")
        row_ids = tuple(ids)
        if row_ids_hash(row_ids) != record.row_ids_hash:
            raise IntegrityViolation(f"row ids of {dataset_id}/{version} were altered")
        return row_ids

    # -- splits -----------------------------------------------------------------------------
    def put_splits(self, record: SplitsRecord) -> tuple[SplitsRecord, bool]:
        def insert(c: _Conn) -> bool:
            cur = c.execute(
                "INSERT INTO dataset_splits(dataset_id, dataset_version, splits_hash, "
                "created_at, record_json) VALUES (?,?,?,?,?) "
                "ON CONFLICT(dataset_id, dataset_version, splits_hash) DO NOTHING",
                (
                    record.dataset_id,
                    record.dataset_version,
                    record.splits_hash,
                    record.created_at,
                    record.model_dump_json(),
                ),
            )
            return cur.rowcount == 1

        with self._connect() as conn:
            try:
                created = self._write(conn, insert)
            except sqlite3.IntegrityError as exc:  # unknown dataset version (foreign key)
                raise Conflict(str(exc)) from None
        stored = self.get_splits(record.dataset_id, record.dataset_version, record.splits_hash)
        if stored is None:
            raise IntegrityViolation("splits vanished after insert")
        return stored, created

    def get_splits(self, dataset_id: str, version: int, splits_hash: str) -> SplitsRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dataset_id, dataset_version, splits_hash, record_json FROM dataset_splits "
                "WHERE dataset_id=? AND dataset_version=? AND splits_hash=?",
                (dataset_id, version, splits_hash),
            ).fetchone()
        return self._splits(row) if row else None

    def list_splits(self, dataset_id: str, version: int) -> list[SplitsRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dataset_id, dataset_version, splits_hash, record_json FROM dataset_splits "
                "WHERE dataset_id=? AND dataset_version=? ORDER BY created_at, splits_hash",
                (dataset_id, version),
            ).fetchall()
        return [self._splits(r) for r in rows]

    @staticmethod
    def _splits(row: tuple[Any, ...]) -> SplitsRecord:
        dataset_id, version, splits_hash, raw = row
        rec: SplitsRecord = _load(SplitsRecord, raw, "splits")
        if (rec.dataset_id, rec.dataset_version, rec.splits_hash) != (
            dataset_id,
            version,
            splits_hash,
        ):
            raise IntegrityViolation(f"splits row {dataset_id}/{version} was altered")
        return rec


class _Conn:
    """A sqlite3 connection with an explicit-transaction ``executescript``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executescript_safe(self, script: str) -> None:
        # ``executescript`` would COMMIT the surrounding transaction; run statements one by one.
        for statement in script.split(";"):
            if statement.strip():
                self._conn.execute(statement)
