# Dataset ingestion (product API v1)

How a user's file becomes a registered `DatasetSpec` with durable, deterministic splits. The
contracts themselves (`DatasetSpec`, `SplitPlan`, `DatasetSplits`, `seeded_splits`) are in
`core/dataset.py` and described in [dataset_contract.md](dataset_contract.md); nothing here
redefines them.

```
upload bytes ──► parse + inspect ──► blob (sha256) + UploadRecord
                                          │
            role mapping ──► re-read blob, re-parse, row ids ──► DatasetSpec ──► DatasetVersionRecord
                                                                                     │
                                         SplitPlan ──► core.dataset.seeded_splits ──► SplitsRecord
```

| Module | Role |
|---|---|
| `ingestion/parse.py` | parse + inspect CSV / JSONL from exact bytes; type/null inference; preview; row ids |
| `ingestion/service.py` | `DatasetService`: upload, register, list/get, create/get splits |
| `store/blobs.py` | `BlobStore` interface; `LocalBlobStore` (content-addressed files) |
| `store/datasets.py` | `DatasetRepository` interface + records; `SQLiteDatasetRepository` |
| `api/product.py` | HTTP routes, strict response schemas, stable error codes |

## Running

```
python -m api.product --data-dir .wynk-data          # standalone, http://127.0.0.1:8788/api/v1/
python -m api.chat --env-file .env                   # chat server also mounts /api/v1/ (UI proxy)
```

`--data-dir` (or `WYNK_DATA_DIR`, default `.wynk-data/`, git-ignored) holds `metadata.sqlite3` and
`blobs/`. `--max-upload-mb` (or `WYNK_MAX_UPLOAD_MB`, default 50) sets the upload limit.

## API

| Method + path | Body | Result |
|---|---|---|
| `POST /api/v1/projects` | `{name, description?}` | 201 project |
| `GET /api/v1/projects`, `GET /api/v1/projects/{id}` | | projects / project |
| `POST /api/v1/projects/{id}/uploads?format=csv\|jsonl&filename=` | raw file bytes | 201 new / 200 identical upload: hash, size, row count, typed columns, preview |
| `GET /api/v1/uploads/{upload_id}` | | upload |
| `POST /api/v1/uploads/{upload_id}/register` | `{dataset_id, name, input_columns, target_columns, context_columns?, row_ids: "column"\|"generated", id_column?}` | 201 new / 200 identical version: `spec` (a `DatasetSpec`), `identity_hash`, row-id source |
| `GET /api/v1/projects/{id}/datasets`, `GET /api/v1/datasets/{dataset_id}` | | datasets with all versions |
| `GET /api/v1/datasets/{dataset_id}/versions/{n}` | | one version |
| `POST /api/v1/datasets/{dataset_id}/versions/{n}/splits` | `SplitPlan` `{seed, validation_bps, test_bps}` | 201 new / 200 identical: `DatasetSplits`, `splits_hash`, sizes |
| `GET .../versions/{n}/splits`, `GET .../versions/{n}/splits/{splits_hash}` | | splits |

Request bodies are validated in pydantic strict mode with unknown fields refused; the upload route
accepts only `format` and `filename`. A client therefore cannot supply a content hash, row count,
column type or row id. Errors are `{"error": {"code", "message", "details"}}`; codes are listed in
`api.product.ERROR_STATUS` and never renamed. Unexpected failures are `500 internal_error` with no
internals; missing or corrupt stored state is `500 storage_error` (never reported as success).

## Validation rules

* **Size.** `Content-Length` is required (chunked bodies are refused) and checked against the limit
  before any byte is read (`413 payload_too_large`).
* **Encoding.** Strict UTF-8; a leading BOM is ignored by the parser (the hash still covers it); NUL
  characters are refused.
* **CSV.** Comma, `"` quoting, header first; every record has exactly the header's field count; a
  blank line is malformed; text after a closing quote is malformed. An empty cell is null.
* **JSONL.** One object per line; a blank line is malformed; duplicate keys, `NaN`/`Infinity` and
  overflowing numbers are refused; a missing key is null. Columns are ordered by first appearance.
* **Columns.** Unique, identifier-like names (`core.dataset.COLUMN_NAME`); at most 512 columns.
* **Rows.** Any unreadable row rejects the whole file; rows are never dropped. At least one row.
* **Types** (deterministic, over every non-null value):
  CSV: boolean > integer (no leading zeros, within int64) > number > date (`YYYY-MM-DD`) > string.
  JSONL: boolean > integer > number > date > string > string_list; any other mix is `json`.
  `nullable` is true iff some row has no value; `null_count` is reported per column.
* **Preview.** At most 20 rows, cells truncated to 200 characters.
* **Roles.** Columns must exist; a column takes at most one role; `json` columns take none. Then the
  real `DatasetSpec` validates the mapping again and is the authority. Task-type rules stay in
  `TaskContract`.

## Row identity

* `row_ids: "column"`: the id column must be a non-nullable string or integer column; every value
  is non-empty, at most 256 characters, and unique (`duplicate_row_id` otherwise). Integer ids are
  their decimal text.
* `row_ids: "generated"`: `r-` + the first 24 hex digits of
  `sha256("wynk-row/1:" + canonical_json(row))`; the k-th identical copy of a row gets `-k`. Ids
  depend only on row content (not on line endings or position), so they are stable for the
  immutable version, and a row keeps its id when others are added, removed or reordered. The
  spec's `id_column` is `None`; the scheme is recorded on the version (`row_id_scheme`) and in the
  spec's non-authoritative metadata. A collision fails closed.

Row ids are stored with the version, in file order, guarded by `row_ids_hash`.

## Identity and versions

* `content_hash` is sha256 of the exact uploaded bytes, computed by the server.
* The upload id is derived from (project, format, content hash): re-uploading identical bytes
  returns the existing upload (200) and reuses the same blob, if that blob still verifies (see
  [Blob integrity](#blob-integrity)).
* Registering creates version `n + 1` of `dataset_id`, unless an existing version has the same
  identity (same bytes, columns, roles, row count, id handling): then that version is returned
  (200). Versions are immutable; a dataset id belongs to one project.
* Splits are `core.dataset.seeded_splits(spec.identity_hash, row_ids, plan)`, stored under
  `DatasetSplits.identity_hash`. Same version + same plan gives the same splits, in any store.

## Persistence

* **Blobs** (`LocalBlobStore`): `blobs/sha256/ab/cd/<digest>`, written to a temp file in the same
  directory, fsynced, then atomically renamed (`os.replace`); the directory is fsynced on POSIX.
  Paths never leave the store. See [Blob integrity](#blob-integrity).
* **Metadata** (`SQLiteDatasetRepository`): stdlib `sqlite3`, WAL, `synchronous=FULL`, foreign
  keys, one `BEGIN IMMEDIATE` transaction per write. Records are stored as canonical pydantic JSON
  and re-validated on every read: a spec must still hash to its stored identity, seeded splits are
  re-derived from their plan by `DatasetSplits`, row ids must match their hash. This catches
  corruption and *inconsistent* edits (refused, never repaired); it is not tamper-proofing: a
  deliberate rewrite of both a record and its stored hashes is not detected (nothing is signed).
* Upload order is: parse (nothing stored on failure) → blob → record. A crash between the last two
  leaves an unreferenced blob that the next identical upload reuses.

### Blob integrity

Invariant: no operation reports an upload or dataset version as available while its blob is
missing or does not hash to its recorded sha256. `BlobStore` has no existence-only check;
`verify(sha256)` re-hashes the stored bytes (streamed, 1 MiB chunks).

* `GET` of an upload, a dataset, a dataset version, a project's dataset list or a version's splits,
  and creating splits, verify every referenced blob first; a missing or corrupt blob is
  `500 storage_error`. Registration reads the blob through `get`, which also re-hashes it.
* `put` reuses a stored blob only if it verifies (healthy duplicate: no write, `created=False`).
* **Policy for a missing or corrupt blob on re-upload:** `put` atomically replaces it (temp file,
  fsync, rename) with the newly supplied bytes. Those bytes hash to the blob's name by
  construction, so the replacement is exactly the recorded content; the existing upload record is
  then returned (200) and is healthy again. Nothing else ever repairs a blob.
* Cost: each verified read hashes the whole blob (up to the upload limit); there is no cache.

Both stores sit behind `BlobStore` / `DatasetRepository`, so object storage and PostgreSQL can
replace them without touching the service or the API.

## Limitations (deferred)

* Uploads are held in memory (bounded by the size limit); no streaming parse, no resumable upload.
* No Parquet, no CSV dialect sniffing (comma only), no per-column type overrides.
* No auth or multi-tenancy: any caller can read any project.
* No garbage collection of unreferenced blobs; no deletion endpoints.
* Distinct-value statistics are not computed.
* The UI's live adapter (`ui/src/api/live.ts`) is not wired to these endpoints yet.
* Optimizer execution on uploaded datasets, background workers and deployment are later phases.
