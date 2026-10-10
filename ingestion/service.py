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

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.canonical import sha256_hex
from core.dataset import (
    SLUG,
    ColumnType,
    DatasetSpec,
    SplitPlan,
    seeded_splits,
)
from ingestion.parse import (
    PARSER_VERSION,
    ROW_ID_SCHEME,
    IngestError,
    IngestLimits,
    ParsedDataset,
    column_row_ids,
    generated_row_ids,
    parse_dataset,
    row_ids_hash,
    sha256_bytes,
)
from store.blobs import BlobCorrupted, BlobNotFound, BlobStore
from store.datasets import (
    Conflict,
    DatasetRepository,
    DatasetVersionRecord,
    ProjectRecord,
    RowIdSource,
    SplitsRecord,
    UploadRecord,
)

MAX_VERSION_RETRIES = 3


class NotFound(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class StorageFailure(Exception):
    """Stored state is missing or corrupt. Never reported as success; never repaired silently."""


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
    ) -> None:
        self.repo = repository
        self.blobs = blobs
        self.limits = limits or IngestLimits()
        self.clock = clock

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
        """Inspect and store one upload. ``bool``: a new upload (False: identical one existed)."""
        self.get_project(project_id)
        if filename is not None and (len(filename) > 255 or not filename.strip()):
            raise IngestError("invalid_request", "filename must be 1-255 characters")
        parsed = parse_dataset(data, fmt, self.limits)  # fails before anything is stored
        content_hash = sha256_bytes(data)
        upload_id = "u-" + sha256_hex(f"{project_id}:{parsed.format.value}:{content_hash}")[:32]
        admit = getattr(self.repo, "admit_upload", None)
        if admit is not None:  # #32 quota (advisory; put_upload re-checks authoritatively)
            admit(upload_id, len(data))
        ref = self.blobs.put(data)  # verifies a stored duplicate; replaces a missing/corrupt one
        if ref.sha256 != content_hash or ref.size_bytes != len(data):
            raise StorageFailure("blob store returned a different digest for the upload")
        existing = self.repo.get_upload(upload_id)
        if existing is not None:
            return existing, False
        record = UploadRecord(
            upload_id=upload_id,
            project_id=project_id,
            filename=filename,
            format=parsed.format,
            content_hash=content_hash,
            size_bytes=len(data),
            row_count=parsed.row_count,
            columns=parsed.columns,
            preview=parsed.preview(self.limits),
            parser_version=PARSER_VERSION,
            created_at=self.clock(),
        )
        return self.repo.put_upload(record)

    def get_upload(self, upload_id: str) -> UploadRecord:
        upload = self._upload_record(upload_id)
        self._verify_blob(upload.content_hash)
        return upload

    def _upload_record(self, upload_id: str) -> UploadRecord:
        upload = self.repo.get_upload(upload_id)
        if upload is None:
            raise NotFound("upload_not_found", f"no upload {upload_id}")
        return upload

    def _verify_blob(self, content_hash: str) -> None:
        try:
            self.blobs.verify(content_hash)
        except (BlobNotFound, BlobCorrupted) as exc:
            raise StorageFailure(f"stored dataset bytes {content_hash}: {exc!r}") from None

    def _verified(self, versions: list[DatasetVersionRecord]) -> list[DatasetVersionRecord]:
        for content_hash in sorted({v.spec.content_hash for v in versions}):
            self._verify_blob(content_hash)
        return versions

    def _reparse(self, upload: UploadRecord) -> ParsedDataset:
        """The upload's stored bytes, parsed again. Must reproduce the recorded inspection."""
        try:
            data = self.blobs.get(upload.content_hash)
        except (BlobNotFound, BlobCorrupted) as exc:
            raise StorageFailure(f"stored bytes of upload {upload.upload_id}: {exc!r}") from None
        if upload.parser_version != PARSER_VERSION:
            raise StorageFailure(
                f"upload {upload.upload_id} was inspected by {upload.parser_version}; "
                f"this server runs {PARSER_VERSION}. Upload the file again."
            )
        # the bytes were accepted under the limit in force at upload time
        limits = self.limits.model_copy(update={"max_upload_bytes": len(data)})
        parsed = parse_dataset(data, upload.format, limits)
        if parsed.columns != upload.columns or parsed.row_count != upload.row_count:
            raise StorageFailure(f"upload {upload.upload_id} no longer parses as inspected")
        return parsed

    # -- registration -----------------------------------------------------------------------
    def register(
        self, upload_id: str, request: RegisterDataset
    ) -> tuple[DatasetVersionRecord, bool]:
        """Create (or return the identical existing) dataset version. ``bool``: created."""
        upload = self._upload_record(upload_id)
        parsed = self._reparse(upload)  # reads the blob through BlobStore.get: hash-verified
        self._check_mapping(parsed, request)
        if request.row_ids == "column":
            assert request.id_column is not None
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
        self._verify_blob(record.spec.content_hash)
        return record

    # -- splits -----------------------------------------------------------------------------
    def create_splits(
        self, dataset_id: str, version: int, plan: SplitPlan
    ) -> tuple[SplitsRecord, bool]:
        record = self.get_version(dataset_id, version)
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
