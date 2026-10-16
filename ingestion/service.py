"""Dataset ingestion service: upload -> inspect -> map roles -> register DatasetSpec -> split.

The service is the only writer of dataset state. It owns no storage itself: original bytes go to a
``BlobStore`` (content-addressed), metadata to a ``DatasetRepository``. Everything that identifies
a dataset is computed here from the stored bytes; nothing a client sends about the data (hash, row
count, column types) is accepted.

* ``upload``: bytes are size-checked, parsed and inspected *before* anything is stored, so a
  malformed file leaves no trace. The blob is written first, then the upload record; a crash in
  between leaves at most an unreferenced blob, which the next identical upload reuses. An
  identical re-upload also goes through ``BlobStore.put``, so a missing or corrupt stored blob is
  replaced by the supplied (hash-matching) bytes before the existing record is returned.
* Every read of an upload or a dataset version re-hashes its blob (``BlobStore.verify``) and fails
  closed with ``StorageFailure`` if it is missing or corrupt: nothing is reported as available
  while its content is not.
* ``register``: re-reads the blob (verified against its hash), re-parses it, checks the role
  mapping, derives row ids, builds the real ``core.dataset.DatasetSpec`` and stores a new immutable
  version. Registering the same upload with the same mapping again returns the existing version.
* ``create_splits``: ``core.dataset.seeded_splits`` over the version's stored row ids; the
  resulting ``DatasetSplits`` is stored under its identity hash. Same plan, same splits.

Task-specific rules (which column types a task type accepts, evaluator choice) are not checked
here; they belong to ``core.task_contract.TaskContract``.
"""

from __future__ import annotations

import csv
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.canonical import sha256_hex
from core.dataset import (
    SLUG,
    ColumnType,
    DatasetSpec,
    SplitPlan,
    seeded_splits,
)
from ingestion import export as exports
from ingestion.parse import (
    PARSER_VERSION,
    ROW_ID_SCHEME,
    IngestError,
    IngestLimits,
    ParsedDataset,
    column_row_ids,
    generated_row_ids,
    head_rows,
    parse_dataset,
    preview_rows,
    row_ids_hash,
    serialize_rows,
    sha256_bytes,
)
from ingestion.sensitive import SensitiveDataPolicy, screen
from store.blobs import BlobError, BlobNotFound, BlobStore
from store.datasets import (
    Conflict,
    DatasetRepository,
    DatasetVersionRecord,
    ProjectRecord,
    RowIdSource,
    SplitsRecord,
    UploadDeletion,
    UploadRecord,
)

MAX_VERSION_RETRIES = 3
SYSTEM_RETENTION_ACTOR = "system:retention"

# Display filenames (#33): never used as a path, but still refused when they look like one, or
# carry control / shell-hostile characters, so nothing downstream (logs, headers, a user's
# download dialog) ever handles one. The extension, if any, must match the declared format.
_FILENAME = re.compile(r'^[^\x00-\x1f\x7f-\x9f/\\:*?"<>|]{1,255}$')
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)
_EXTENSIONS = {"csv": (".csv", ".txt"), "jsonl": (".jsonl", ".ndjson", ".txt")}


class NotFound(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SourceDeleted(Exception):
    """The upload behind a request was deleted (#33): its content is not retrievable. ``code``:
    ``upload_deleted`` or ``dataset_source_deleted``; details carry identifiers only."""

    def __init__(self, code: str, message: str, deletion: UploadDeletion) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, str | int | None] = {
            "upload_id": deletion.upload_id,
            "deleted_at": deletion.deleted_at,
            "reason": deletion.reason,
        }


class StorageFailure(Exception):
    """Stored state is missing or corrupt. Never reported as success; never repaired silently."""


class SourceStatus(BaseModel):
    """Whether a dataset version's source content is still held (#33)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    dataset_version: int
    upload_id: str
    content_hash: str
    identity_hash: str
    status: Literal["available", "deleted"]
    deleted_at: str | None = None
    deletion_reason: Literal["request", "retention"] | None = None


class DeletionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deletion: UploadDeletion
    # live uploads (other projects of this workspace) with the same bytes: their content is
    # still theirs, so the shared blob stays until the last of them is deleted too
    content_shared_with: int
    content_removed: bool


class DatasetDeletion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    deletions: tuple[DeletionResult, ...]


class ExportResult(NamedTuple):
    data: bytes
    content_type: str
    filename: str
    content_hash: str  # the version's (stored canonical plaintext) content hash
    export_format: str
    safety: str  # raw | spreadsheet-neutralized | json
    neutralized_cells: int


def check_filename(filename: str | None, fmt: str) -> str | None:
    if filename is None:
        return None
    if (
        not _FILENAME.match(filename)
        or filename != filename.strip()
        or filename.strip(".") == ""
        or ".." in filename
        or filename.endswith(".")
        or _RESERVED.match(filename)
    ):
        raise IngestError(
            "invalid_filename",
            "filename must be 1-255 characters without path separators, '..', control "
            'characters or any of : * ? " < > |',
        )
    if "." in filename.lstrip("."):
        ext = "." + filename.rsplit(".", 1)[1].lower()
        allowed = _EXTENSIONS.get(fmt)  # an unsupported format is refused by the parser
        if allowed is not None and ext not in allowed:
            raise IngestError(
                "invalid_filename",
                f"a {fmt} upload's filename must end in {' / '.join(allowed)}",
            )
    return filename


# -- requests -----------------------------------------------------------------------------------
class NewProject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class RegisterDataset(BaseModel):
    """Role mapping for an inspected upload. No field describes the data itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(pattern=SLUG)
    name: str = Field(min_length=1, max_length=200)
    input_columns: tuple[str, ...] = Field(min_length=1)
    target_columns: tuple[str, ...] = Field(min_length=1)
    context_columns: tuple[str, ...] = ()
    row_ids: Literal["column", "generated"]
    id_column: str | None = None

    @model_validator(mode="after")
    def _row_ids(self) -> RegisterDataset:
        if (self.row_ids == "column") != (self.id_column is not None):
            raise ValueError("row_ids 'column' requires id_column; 'generated' forbids it")
        return self


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DatasetService:
    def __init__(
        self,
        repository: DatasetRepository,
        blobs: BlobStore,
        limits: IngestLimits | None = None,
        clock: Callable[[], str] = _utc_now,
        policy: SensitiveDataPolicy | None = None,
        retention_days: int | None = None,
    ) -> None:
        self.repo = repository
        self.blobs = blobs
        self.limits = limits or IngestLimits()
        self.clock = clock
        self.policy = policy or SensitiveDataPolicy()
        if retention_days is not None and retention_days < 1:
            raise ValueError("retention_days must be at least 1 (or None: keep until deleted)")
        self.retention_days = retention_days

    # -- projects ---------------------------------------------------------------------------
    def create_project(self, request: NewProject) -> ProjectRecord:
        project = ProjectRecord(
            project_id=f"p-{uuid.uuid4().hex[:20]}",
            name=request.name,
            description=request.description,
            created_at=self.clock(),
        )
        self.repo.create_project(project)
        return project

    def get_project(self, project_id: str) -> ProjectRecord:
        project = self.repo.get_project(project_id)
        if project is None:
            raise NotFound("project_not_found", f"no project {project_id}")
        return project

    def list_projects(self) -> list[ProjectRecord]:
        return self.repo.list_projects()

    # -- uploads ----------------------------------------------------------------------------
    def upload(
        self, project_id: str, data: bytes, fmt: str, filename: str | None = None
    ) -> tuple[UploadRecord, bool]:
        """Inspect, screen and store one upload. ``bool``: a new upload (False: an identical one
        existed).

        Order (#33): bounds + parse, then the secrets / PII policy, and only then storage, so a
        refused file leaves no trace. The stored bytes are the *canonical plaintext*: the
        uploaded bytes verbatim, or - iff the policy redacted a value - their deterministic
        re-serialization (``serialize_rows``) with the redacted values. ``content_hash`` (and
        therefore the upload id and every dataset identity) is the sha256 of exactly those
        plaintext bytes; the blob store encrypts them, and no ciphertext is ever hashed."""
        self.get_project(project_id)
        filename = check_filename(filename, fmt)
        parsed = parse_dataset(data, fmt, self.limits)  # fails before anything is stored
        names = [c.name for c in parsed.columns]
        screened = screen(names, parsed.rows, self.policy, filename)
        if screened.report.blocked:
            first = screened.report.blocked[0]
            where = (
                f"column {first.column!r}, row {first.first_row}" if first.column else "filename"
            )
            raise IngestError(
                "sensitive_data_blocked",
                f"the upload contains {first.kind.value} data ({first.category.value}, {where}); "
                "the workspace's sensitive-data policy refuses it. Nothing was stored.",
                category=first.category.value,
                column=first.column,
                row=first.first_row,
                categories=",".join(sorted({f.category.value for f in screened.report.blocked})),
                policy_version=screened.report.policy_version,
            )
        stored = data
        if screened.rows is not None:  # values were redacted: store the canonical rewrite
            stored = serialize_rows(parsed.format, names, screened.rows)
            redacted = parse_dataset(stored, parsed.format, self.limits.relaxed(len(stored)))
            if [c.name for c in redacted.columns] != names or redacted.rows != screened.rows:
                raise StorageFailure("the redacted dataset does not round-trip")
            parsed = redacted
        content_hash = sha256_bytes(stored)
        upload_id = "u-" + sha256_hex(f"{project_id}:{parsed.format.value}:{content_hash}")[:32]
        deletion = self.repo.get_deletion(upload_id)
        if deletion is not None:  # deletion is final for this (project, format, content)
            raise SourceDeleted("upload_deleted", f"upload {upload_id} was deleted", deletion)
        admit = getattr(self.repo, "admit_upload", None)
        if admit is not None:  # #32 quota (advisory; put_upload re-checks authoritatively)
            admit(upload_id, len(stored))
        ref = self.blobs.put(stored)  # verifies a stored duplicate; replaces a missing/corrupt one
        if ref.sha256 != content_hash or ref.size_bytes != len(stored):
            raise StorageFailure("blob store returned a different digest for the upload")
        existing = self.repo.get_upload(upload_id)
        if existing is not None:
            return self._with_preview(existing, parsed.rows), False
        record = UploadRecord(
            upload_id=upload_id,
            project_id=project_id,
            filename=screened.filename,
            format=parsed.format,
            content_hash=content_hash,
            size_bytes=len(stored),
            row_count=parsed.row_count,
            columns=parsed.columns,
            preview=(),  # never persisted: derived from the encrypted blob on read
            parser_version=PARSER_VERSION,
            created_at=self.clock(),
            sensitive_data=screened.report,
        )
        stored_record, created = self.repo.put_upload(record)
        return self._with_preview(stored_record, parsed.rows), created

    def _with_preview(
        self, upload: UploadRecord, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]
    ) -> UploadRecord:
        names = [c.name for c in upload.columns]
        return upload.model_copy(update={"preview": preview_rows(names, rows, self.limits)})

    def get_upload(self, upload_id: str) -> UploadRecord:
        """The upload with its preview, derived from the verified, decrypted blob."""
        upload = self._upload_record(upload_id)
        self._require_live(upload.upload_id)
        data = self._blob(upload.content_hash)
        names = [c.name for c in upload.columns]
        try:
            rows = head_rows(data, upload.format, names, self.limits.preview_rows)
        except (ValueError, KeyError, csv.Error) as exc:
            raise StorageFailure(
                f"upload {upload.upload_id} no longer parses: {type(exc).__name__}"
            ) from None
        return self._with_preview(upload, rows)

    def _require_live(self, upload_id: str, code: str = "upload_deleted") -> None:
        deletion = self.repo.get_deletion(upload_id)
        if deletion is not None:
            what = "upload" if code == "upload_deleted" else "the source upload"
            raise SourceDeleted(code, f"{what} {upload_id} was deleted", deletion)

    def _blob(self, content_hash: str) -> bytes:
        try:
            return self.blobs.get(content_hash)
        except BlobError as exc:
            raise StorageFailure(f"stored dataset bytes {content_hash}: {exc!r}") from None

    def _upload_record(self, upload_id: str) -> UploadRecord:
        upload = self.repo.get_upload(upload_id)
        if upload is None:
            raise NotFound("upload_not_found", f"no upload {upload_id}")
        return upload

    def _verify_blob(self, content_hash: str) -> None:
        try:
            self.blobs.verify(content_hash)
        except BlobError as exc:
            raise StorageFailure(f"stored dataset bytes {content_hash}: {exc!r}") from None

    def _verified(self, versions: list[DatasetVersionRecord]) -> list[DatasetVersionRecord]:
        """Every version whose source is still held must have a healthy blob. A version whose
        upload was deleted (#33) is reported as it is - immutable provenance - and its absent
        content is not a storage failure (``source_status`` makes the state explicit)."""
        live = {
            v.spec.content_hash for v in versions if self.repo.get_deletion(v.upload_id) is None
        }
        for content_hash in sorted(live):
            self._verify_blob(content_hash)
        return versions

    def _reparse(self, upload: UploadRecord) -> ParsedDataset:
        """The upload's stored bytes, parsed again. Must reproduce the recorded inspection."""
        data = self._blob(upload.content_hash)
        if upload.parser_version != PARSER_VERSION:
            raise StorageFailure(
                f"upload {upload.upload_id} was inspected by {upload.parser_version}; "
                f"this server runs {PARSER_VERSION}. Upload the file again."
            )
        # the bytes were admitted under the bounds in force at upload time
        parsed = parse_dataset(data, upload.format, self.limits.relaxed(len(data)))
        if parsed.columns != upload.columns or parsed.row_count != upload.row_count:
            raise StorageFailure(f"upload {upload.upload_id} no longer parses as inspected")
        return parsed

    # -- registration -----------------------------------------------------------------------
    def register(
        self, upload_id: str, request: RegisterDataset
    ) -> tuple[DatasetVersionRecord, bool]:
        """Create (or return the identical existing) dataset version. ``bool``: created."""
        upload = self._upload_record(upload_id)
        self._require_live(upload.upload_id)
        parsed = self._reparse(upload)  # reads the blob through BlobStore.get: hash-verified
        self._check_mapping(parsed, request)
        if request.row_ids == "column":
            assert request.id_column is not None
            self._check_id_values(parsed, request.id_column)
            row_ids = column_row_ids(parsed, request.id_column)
            source, scheme = RowIdSource.COLUMN, None
        else:
            row_ids = generated_row_ids(parsed)
            source, scheme = RowIdSource.GENERATED, ROW_ID_SCHEME

        for _ in range(MAX_VERSION_RETRIES):
            owner = self.repo.dataset_owner(request.dataset_id)
            if owner is not None and owner != upload.project_id:
                raise IngestError(
                    "dataset_conflict",
                    f"dataset id {request.dataset_id!r} belongs to another project",
                    dataset_id=request.dataset_id,
                )
            versions = self.repo.list_versions(request.dataset_id)
            spec = self._spec(upload, request, len(versions) + 1)
            for existing in versions:  # same bytes + same mapping: same version, not a new one
                same = spec.model_copy(update={"dataset_version": existing.dataset_version})
                if (
                    existing.identity_hash == same.identity_hash
                    and existing.row_id_source is source
                    and existing.row_ids_hash == row_ids_hash(row_ids)
                ):
                    return existing, False
            record = DatasetVersionRecord(
                project_id=upload.project_id,
                upload_id=upload.upload_id,
                spec=spec,
                identity_hash=spec.identity_hash,
                row_id_source=source,
                row_id_scheme=scheme,
                row_ids_hash=row_ids_hash(row_ids),
                created_at=self.clock(),
            )
            try:
                self.repo.add_version(record, row_ids)
            except Conflict:
                continue  # a concurrent registration took the version number (or the id)
            return record, True
        raise StorageFailure(f"could not allocate a version for {request.dataset_id}")

    def _check_mapping(self, parsed: ParsedDataset, request: RegisterDataset) -> None:
        roles = {
            "input": request.input_columns,
            "target": request.target_columns,
            "context": request.context_columns,
            "id": (request.id_column,) if request.id_column is not None else (),
        }
        seen: dict[str, str] = {}
        for role, cols in roles.items():
            for name in cols:
                col = parsed.column(name)
                if col is None:
                    raise IngestError(
                        "unknown_column", f"{role} column {name!r} is not in the file", column=name
                    )
                if col.type is ColumnType.JSON:
                    raise IngestError(
                        "json_column_role",
                        f"{name!r} holds nested JSON values and cannot take a schema role",
                        column=name,
                    )
                if name in seen:
                    raise IngestError(
                        "invalid_mapping",
                        f"column {name!r} is mapped as both {seen[name]} and {role}",
                        column=name,
                    )
                seen[name] = role

    def _check_id_values(self, parsed: ParsedDataset, id_column: str) -> None:
        """Row ids are stored in plaintext metadata and kept in split / experiment records
        after the source is deleted, so an id column must not hold sensitive values: any rule
        match (whatever the policy's action) or a redaction placeholder refuses it (#33)."""
        values = [{id_column: r[id_column]} for r in parsed.rows]
        report = screen([id_column], values, self.policy).report
        redacted = any(
            isinstance(r[id_column], str) and "[REDACTED:" in r[id_column] for r in parsed.rows
        )
        if report.findings or redacted:
            cats = sorted({f.category.value for f in report.findings}) or ["redacted"]
            raise IngestError(
                "sensitive_row_ids",
                f"id column {id_column!r} holds sensitive values ({', '.join(cats)}); row ids "
                "outlive the data in split and experiment records. Use row_ids: generated.",
                column=id_column,
                categories=",".join(cats),
            )

    def _spec(self, upload: UploadRecord, request: RegisterDataset, version: int) -> DatasetSpec:
        metadata: dict[str, str | int | float | bool] = {"row_ids": request.row_ids}
        if request.row_ids == "generated":
            metadata["row_id_scheme"] = ROW_ID_SCHEME
        try:
            return DatasetSpec(
                dataset_id=request.dataset_id,
                dataset_version=version,
                name=request.name,
                content_hash=upload.content_hash,
                format=upload.format,
                columns=tuple(c.spec() for c in upload.columns),
                id_column=request.id_column,
                input_columns=request.input_columns,
                target_columns=request.target_columns,
                context_columns=request.context_columns,
                row_count=upload.row_count,
                metadata=metadata,
            )
        except ValidationError as exc:  # the contract is the authority; report its reason
            reason = exc.errors()[0]["msg"] if exc.errors() else str(exc)
            raise IngestError(
                "invalid_mapping", f"DatasetSpec rejected the mapping: {reason}"
            ) from None

    # -- datasets ---------------------------------------------------------------------------
    def list_datasets(self, project_id: str) -> list[list[DatasetVersionRecord]]:
        self.get_project(project_id)
        return [
            self._verified(self.repo.list_versions(d))
            for d in self.repo.list_dataset_ids(project_id)
        ]

    def get_dataset(self, dataset_id: str) -> list[DatasetVersionRecord]:
        versions = self.repo.list_versions(dataset_id)
        if not versions:
            raise NotFound("dataset_not_found", f"no dataset {dataset_id}")
        return self._verified(versions)

    def get_version(self, dataset_id: str, version: int) -> DatasetVersionRecord:
        record = self.repo.get_version(dataset_id, version)
        if record is None:
            self.get_dataset(dataset_id)
            raise NotFound(
                "dataset_version_not_found", f"dataset {dataset_id} has no version {version}"
            )
        self._verified([record])
        return record

    def require_source(self, record: DatasetVersionRecord) -> None:
        """``dataset_source_deleted`` unless the version's source upload is still held."""
        self._require_live(record.upload_id, "dataset_source_deleted")

    def source_status(self, dataset_id: str, version: int) -> SourceStatus:
        record = self.get_version(dataset_id, version)
        deletion = self.repo.get_deletion(record.upload_id)
        return SourceStatus(
            dataset_id=dataset_id,
            dataset_version=version,
            upload_id=record.upload_id,
            content_hash=record.spec.content_hash,
            identity_hash=record.identity_hash,
            status="available" if deletion is None else "deleted",
            deleted_at=deletion.deleted_at if deletion else None,
            deletion_reason=deletion.reason if deletion else None,
        )

    def dataset_bytes(self, content_hash: str) -> bytes:
        """Verified plaintext for an experiment / monitoring run (by the pinned content hash).
        Refused once no live upload of this workspace holds the content (#33 deletion)."""
        if self.repo.live_uploads_with_hash(content_hash) == 0:
            raise StorageFailure(
                f"dataset content {content_hash} is not held by any live upload (deleted)"
            )
        return self._blob(content_hash)

    # -- splits -----------------------------------------------------------------------------
    def create_splits(
        self, dataset_id: str, version: int, plan: SplitPlan
    ) -> tuple[SplitsRecord, bool]:
        record = self.get_version(dataset_id, version)
        self.require_source(record)
        row_ids = self.repo.get_row_ids(dataset_id, version)
        try:
            splits = seeded_splits(record.identity_hash, row_ids, plan)
        except ValueError as exc:  # e.g. too few rows for the plan to leave optimization rows
            raise IngestError("invalid_split_plan", str(exc)) from None
        return self.repo.put_splits(
            SplitsRecord(
                dataset_id=dataset_id,
                dataset_version=version,
                splits_hash=splits.identity_hash,
                splits=splits,
                sizes={s.role.value: len(s.row_ids) for s in splits.splits},
                created_at=self.clock(),
            )
        )

    def get_splits(self, dataset_id: str, version: int, splits_hash: str) -> SplitsRecord:
        version_record = self.get_version(dataset_id, version)
        record = self.repo.get_splits(dataset_id, version, splits_hash)
        if record is None:
            raise NotFound("splits_not_found", f"no splits {splits_hash} for {dataset_id}")
        if record.splits.dataset_hash != version_record.identity_hash:
            raise StorageFailure("stored splits belong to a different dataset identity")
        return record

    def list_splits(self, dataset_id: str, version: int) -> list[SplitsRecord]:
        self.get_version(dataset_id, version)
        return self.repo.list_splits(dataset_id, version)

    # -- export (#33) -----------------------------------------------------------------------
    def export(self, dataset_id: str, version: int, fmt: str) -> ExportResult:
        """One version's content in ``original`` / ``csv`` / ``jsonl`` (``ingestion.export``)."""
        if fmt not in exports.EXPORT_FORMATS:
            raise IngestError(
                "invalid_request", f"format must be one of: {', '.join(exports.EXPORT_FORMATS)}"
            )
        record = self.get_version(dataset_id, version)
        self.require_source(record)
        data = self._blob(record.spec.content_hash)
        base = f"{dataset_id}-v{version}"
        source = record.spec.format.value
        if fmt == "original":
            return ExportResult(
                data,
                "application/octet-stream",
                f"{base}.original.{source}",
                record.spec.content_hash,
                fmt,
                "raw",
                0,
            )
        parsed = parse_dataset(data, record.spec.format, self.limits.relaxed(len(data)))
        names = [c.name for c in parsed.columns]
        if fmt == "csv":
            body, neutralized = exports.safe_csv(names, parsed.rows)
            return ExportResult(
                body,
                "text/csv; charset=utf-8",
                f"{base}.csv",
                record.spec.content_hash,
                fmt,
                "spreadsheet-neutralized",
                neutralized,
            )
        return ExportResult(
            exports.jsonl(names, parsed.rows),
            "application/x-ndjson; charset=utf-8",
            f"{base}.jsonl",
            record.spec.content_hash,
            fmt,
            "json",
            0,
        )

    # -- deletion / retention (#33) ---------------------------------------------------------
    def delete_upload(
        self, upload_id: str, actor: str, reason: Literal["request", "retention"] = "request"
    ) -> DeletionResult:
        """Delete an upload's content: tombstone (immutable), scrubbed record, purged row ids,
        then the blob once no live upload of this workspace holds the same bytes. Idempotent:
        deleting again returns the first tombstone and re-attempts a blob removal that a crash
        interrupted. Dataset versions and splits stay (immutable provenance), marked
        source-deleted; experiment artifacts are never touched."""
        upload = self._upload_record(upload_id)
        deletion = self.repo.get_deletion(upload_id)
        if deletion is None:
            deletion, _ = self.repo.delete_upload(
                UploadDeletion(
                    upload_id=upload.upload_id,
                    project_id=upload.project_id,
                    content_hash=upload.content_hash,
                    deleted_at=self.clock(),
                    deleted_by=actor,
                    reason=reason,
                    affected_versions=tuple(self.repo.versions_of_upload(upload_id)),
                )
            )
        shared = self.repo.live_uploads_with_hash(deletion.content_hash)
        if shared == 0:
            self.blobs.delete(deletion.content_hash)
            try:
                self.blobs.get(deletion.content_hash)
            except BlobNotFound:
                pass  # proven gone: not retrievable through the store any more
            except BlobError:
                raise StorageFailure(
                    f"blob {deletion.content_hash} survived its deletion"
                ) from None
            else:
                raise StorageFailure(f"blob {deletion.content_hash} survived its deletion")
        return DeletionResult(
            deletion=deletion, content_shared_with=shared, content_removed=shared == 0
        )

    def delete_dataset(self, dataset_id: str, actor: str) -> DatasetDeletion:
        """Delete the source uploads of every version of ``dataset_id``. Refused (nothing is
        deleted) if one of them also backs a version of ANOTHER dataset: deleting it would
        silently take that dataset's content too - delete that upload explicitly instead."""
        self.get_dataset(dataset_id)
        uploads = self.repo.uploads_of_dataset(dataset_id)
        for upload_id in uploads:
            if self.repo.get_deletion(upload_id) is not None:
                continue
            others = [
                v for v in self.repo.versions_of_upload(upload_id) if v.dataset_id != dataset_id
            ]
            if others:
                raise IngestError(
                    "dataset_source_shared",
                    f"upload {upload_id} also backs {others[0].dataset_id} version "
                    f"{others[0].dataset_version}; delete the upload itself to remove both",
                    upload_id=upload_id,
                    dataset_id=others[0].dataset_id,
                )
        return DatasetDeletion(
            dataset_id=dataset_id,
            deletions=tuple(self.delete_upload(u, actor) for u in uploads),
        )

    def expired_uploads(self, now: datetime | None = None) -> list[str]:
        """Live uploads older than ``retention_days`` (none when retention is unlimited)."""
        if self.retention_days is None:
            return []
        cutoff = (now or datetime.now(UTC)) - timedelta(days=self.retention_days)
        out = []
        for upload in self.repo.list_uploads():
            created = datetime.fromisoformat(upload.created_at)
            created = created if created.tzinfo else created.replace(tzinfo=UTC)
            if created < cutoff and self.repo.get_deletion(upload.upload_id) is None:
                out.append(upload.upload_id)
        return out

    def purge_expired(self, now: datetime | None = None) -> list[DeletionResult]:
        """Apply the declared retention policy: delete every expired upload."""
        return [
            self.delete_upload(u, SYSTEM_RETENTION_ACTOR, "retention")
            for u in self.expired_uploads(now)
        ]

    def collect_garbage(self, older_than_s: float = 3600.0) -> list[str]:
        """Remove blobs no live upload references (e.g. the loser of a #32 quota race)."""
        collect = getattr(self.blobs, "collect", None)
        if collect is None:
            return []
        removed: list[str] = collect(self.repo.live_content_hashes(), older_than_s=older_than_s)
        return removed
