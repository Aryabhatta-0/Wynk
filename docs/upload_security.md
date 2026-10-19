# Uploaded-data security and privacy (#33)

User-uploaded datasets are bounded, safely parsed, screened for secrets and PII, encrypted at rest,
exportable and deletable per workspace, and every sensitive action is audited. Dataset identities,
deterministic splits, provenance, experiment reproducibility and #32 tenant isolation are
unchanged.

Code: `ingestion/parse.py` (bounds, encoding), `ingestion/sensitive.py` (secrets / PII policy, log
scrubbing), `ingestion/export.py` (export formats, CSV-injection rule), `store/crypto.py`
(AES-256-GCM envelopes, keyring), `store/blobs.py` (encrypted content-addressed blobs),
`store/datasets.py` (tombstones), `store/audit.py` (append-only audit trail),
`ingestion/service.py` (upload, export, deletion, retention), `api/product.py` /
`api/tenancy.py` (routes, scopes, audit wiring, CLI). Tests: `tests/test_upload_security.py`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WYNK_DATA_KEYS` | **required** | Keyring `kid:<base64url 32 bytes>[,kid2:...]`. The first key encrypts; all decrypt. |
| `WYNK_DATA_KEYS_FILE` | - | Same, read from a file. Refused if it is inside the data dir, or if both are set. |
| `WYNK_MAX_UPLOAD_MB` | 50 | Upload size (`--max-upload-mb`). |
| `WYNK_MAX_UPLOAD_ROWS` | 1 000 000 | Data rows per upload. |
| `WYNK_MAX_UPLOAD_COLUMNS` | 512 | Columns per upload. |
| `WYNK_MAX_UPLOAD_CELL_CHARS` | 131 072 | One CSV field, one JSON string or key. |
| `WYNK_MAX_UPLOAD_RECORD_CHARS` | 4 MiB | One CSV record, one JSONL line. |
| `WYNK_MAX_UPLOAD_JSON_DEPTH` | 32 | Nesting inside one JSONL row (the row object is level 1). |
| `WYNK_SENSITIVE_POLICY` | defaults below | Per-category overrides, e.g. `email=allow,phone=block`. |
| `WYNK_UPLOAD_RETENTION_DAYS` | unset (keep until deleted) | Declared retention, enforced by `purge-expired`. |

An unparsable value refuses to start. Without a data key the server refuses to start
(`KeyConfigError`): there is no plaintext or "development" mode. Generate a key with
`python -m api.tenancy generate-data-key --key-id k1` and keep it in a secret manager or an
environment variable, never in the data dir. Tests use the fixed, public test-only key in
`tests/conftest.py`; it must never be used for real data.

## 1. Upload boundaries

* `Content-Length` is checked before a body byte is read (`413 payload_too_large`).
* Rows, columns, cell size, record size and JSON depth are checked while parsing, before the next
  row or value is materialized: `too_many_rows`, `too_many_columns`, `cell_too_large`,
  `record_too_large`, `json_too_deep` (all 422). JSONL line count is checked before any line is
  parsed. Depth is checked iteratively, so a deeply nested line cannot exhaust the stack.
* Every refusal is a structured `{"error": {"code", "message", "details"}}` body with a stable
  code. Unexpected failures are `500 internal_error` with no trace; the trace goes to the log only.
* Stored bytes are re-read with relaxed bounds (`IngestLimits.relaxed`), so tightening a limit never
  makes an accepted upload unreadable.

## 2. Safe parsing

* Only `csv` and `jsonl` are accepted (`unsupported_format`). The parsers are the stdlib `csv`
  and `json` modules on decoded text: no code execution, no `eval`, no pickle, no spreadsheet or
  archive formats, no decompression (so no zip bombs).
* Strict UTF-8 (`invalid_encoding`). NUL, other C0 controls except tab / LF / CR, DEL and C1
  controls (terminal escapes such as ESC / CSI) are refused, also when a JSON escape (`\u001b`)
  hides them. Malformed or truncated input is `malformed_csv` / `malformed_jsonl`.
* **Filenames are display metadata only.** Storage paths are `blobs/sha256/ab/cd/<sha256>`; the
  filename never forms a path. `check_filename` refuses (`invalid_filename`): path separators,
  `..`, drive prefixes (`:`), control characters, Windows-reserved names (`CON`, `LPT1`, ...),
  a trailing space or dot, more than 255 characters, and an extension that is not the format's
  (`.csv` / `.txt` for CSV; `.jsonl` / `.ndjson` / `.txt` for JSONL). The filename is also
  screened by the sensitive-data policy.

## 3. CSV formula injection

CSV injection is neutralized **on ingest and on export**, with one rule (`ingestion/export.py`),
for every CSV cell `v`:

    v is formula-like  iff  v.lstrip(" ") starts with  =  +  -  @  TAB  CR
                            or the full-width forms  ＝ ＋ － ＠
                        and v is not a plain signed number (-12, +3.5, -1e3, -.5)
    a formula-like v is rewritten as  "'" + v

A rewritten value starts with `'`, so it is not formula-like: the rule is idempotent.

**Ingest order and hashing** (`DatasetService.upload`):

    uploaded bytes
      -> bounds + strict parse                      (sections 1, 2)
      -> secrets / PII policy: block, or redact     (section 4)
      -> CSV only: neutralize every formula-like cell (neutralize_rows)
      -> canonical serialization                    (serialize_rows; only if a value changed)
      -> content_hash = sha256(those plaintext bytes)
      -> encrypt + store                            (section 5)

* If no value changed, the stored bytes are the uploaded bytes verbatim, so a safe CSV keeps
  exactly the identity it had before. If a value was redacted or neutralized, the stored bytes are
  the deterministic re-serialization (UTF-8, `
`, header, minimal `"` quoting), and the upload
  is refused (`500 storage_error`) unless it parses back to exactly the rewritten rows.
* `content_hash`, the upload id, the dataset identity, the row ids (also a generated or an id
  column), the splits and the experiment provenance are all over that one canonical, neutralized
  representation. There is no second scientific representation, and **no copy of the
  formula-capable upload is kept**. Identical input always gives identical stored bytes and hashes.
* Workflows see the neutralized value (`'=1+1`), not the formula. This is a deliberate change to
  the logical dataset, made once and recorded in the report.
* JSONL is not rewritten (JSON is not evaluated by spreadsheets).
* The upload's `sensitive_data` report records `formula_like_values` (detected after redaction,
  in any format) and `neutralized_values` (rewritten in the stored bytes: all of them for CSV, `0`
  for JSONL). It never records a cell value; the audit `dataset.upload` event holds only the two
  counts.

`GET /api/v1/datasets/{id}/versions/{v}/export?format=`:

| `format` | Body | `X-Wynk-Export-Safety` |
|---|---|---|
| `original` | the stored canonical bytes, byte for byte (`sha256` = `content_hash`); for CSV already neutralized | `canonical` |
| `csv` (default) | logical rows, formula-like cells neutralized (again, for legacy blobs) | `spreadsheet-neutralized` |
| `jsonl` | logical rows as JSON lines, values exact | `json` |

A CSV version stored before ingest neutralization existed (a pre-#33 blob, kept unchanged so its
hashes and provenance stay valid) can still hold formula-like cells. Its `original` export is
refused (`409 unsafe_original_export`, with the cell count only), and its `csv` export
neutralizes them. So no API path re-emits formula-capable CSV content.

Every export is `Content-Disposition: attachment`, `X-Content-Type-Options: nosniff`, and carries
`X-Wynk-Content-Hash` and `X-Wynk-Neutralized-Cells` (cells rewritten by this export: `0` for a
version admitted with ingest neutralization).

## 4. Secrets and PII

Deterministic rules, no model (`ingestion/sensitive.py`, `POLICY_VERSION = sensitive-data/1`).
Each rule is one bounded regular expression, some confirmed by a checksum.

| Category | Kind | Default | Rule |
|---|---|---|---|
| `private_key` | secret | block | `-----BEGIN ... PRIVATE KEY-----` |
| `aws_access_key` | secret | block | `AKIA` / `ASIA` + 16 |
| `github_token` | secret | block | `gh[pousr]_...`, `github_pat_...` |
| `slack_token` | secret | block | `xox[abposr]-...` |
| `stripe_secret_key` | secret | block | `sk_live_` / `rk_live_` |
| `google_api_key` | secret | block | `AIza` + 35 |
| `model_api_key` | secret | block | `sk-...`, `sk-ant-...`, `sk-proj-...` |
| `wynk_api_key` | secret | block | `wynk_sk_<16 hex>_<43>` |
| `jwt` | secret | block | three base64url segments starting `eyJ` |
| `payment_card` | pii | block | 13-19 digits, known IIN, Luhn |
| `us_ssn` | pii | block | `AAA-GG-SSSS`, SSA validity rules |
| `iban` | pii | block | country + check digits, ISO 7064 mod-97 |
| `email` | pii | redact | address with a dotted domain |
| `phone` | pii | redact | E.164 with a leading `+` |

* **block**: the upload is refused (`422 sensitive_data_blocked`). The details carry the category,
  column and row, never the value. Nothing is stored.
* **redact**: each match becomes `[REDACTED:<category>]` before anything is stored. The stored
  canonical bytes are the deterministic re-serialization of the redacted rows
  (`parse.serialize_rows`), so the content hash, the upload id, the dataset identity and every
  experiment see the redacted value. The change is recorded, never silent.
* **allow**: the value is kept verbatim; the finding is still recorded. A secret category can only
  block or redact (`allow` refuses to start).
* The decision is recorded on the upload (`sensitive_data`: policy version, and per category and
  column: action, how many values matched, first row) and in the audit trail. Neither holds a
  matched value.
* An id column (`row_ids: column`) with any finding or a placeholder is refused
  (`sensitive_row_ids`): row ids outlive the data in split and experiment records.
* Rules find the listed formats only. Names, addresses, free-text health data and anything
  without a fixed format are **not** detected.
* Logs: `install_log_redaction` (installed by the router) scrubs every log record of the process
  (message, arguments and traceback) with the same rules, as it is created.

## 5. Encryption at rest

* Every blob is an AES-256-GCM envelope (`cryptography`):
  `WYNKBLOB | format 1 | alg 1 | len(key_id) | key_id | nonce(12) | ciphertext+tag`.
* The per-workspace data key is `HKDF-SHA256(master, salt="wynk-blob-key/1",
  info="workspace:<id>")`. The associated data binds the header, the workspace id and the blob's
  name, so a blob copied to another workspace or renamed fails authentication.
* A fresh 96-bit random nonce per write. **Hashes are over the plaintext**: a blob is named by
  `sha256(stored canonical plaintext)`, which is the `content_hash`. Nonces and ciphertext never
  enter a hash, so identical content keeps one identity across writes, workspaces and keys.
* Reads decrypt, authenticate and re-hash before returning. Tampering, truncation, a moved blob or a
  wrong key is `BlobCorrupted`; an unknown key id is `BlobKeyUnavailable`. Both fail closed
  (`500 storage_error`).
* Key material never sits beside the ciphertext: it comes from the environment or from a file
  outside the data dir. Only the key id is stored in each envelope.
* Rotation: prepend a new key (`WYNK_DATA_KEYS="k2:...,k1:..."`), restart, then run
  `python -m api.tenancy secure-data --data-dir ...`. It re-encrypts blobs under the active key.
  Retire the old key only after that.
* Upload previews are not persisted (before #33 they were plaintext in `metadata.sqlite3`). They
  are derived from the decrypted blob on read. `secure-data` also encrypts pre-#33 plaintext blobs
  (only if they still hash to their name) and scrubs stored previews. Until then a plaintext blob
  is refused (`BlobNotEncrypted`).
* **Not encrypted:** metadata in SQLite (project names, column names and types, row ids, row
  counts, hashes, split assignments), the identity store and the audit trail. Disk or filesystem
  encryption covers those.

## 6. Export, deletion and retention

* Export: `GET .../versions/{v}/export` (scope `read`), tenant-scoped like every route (#32).
* `DELETE /api/v1/uploads/{id}` and `DELETE /api/v1/datasets/{id}` (scope `admin`, so an OWNER
  only). A deletion, in one transaction:
  1. inserts an immutable tombstone (`upload_deletions`; triggers refuse UPDATE and DELETE):
     upload id, project, content hash, time, actor, reason (`request` / `retention`) and the
     affected dataset versions;
  2. scrubs the upload record (filename, preview) and purges the stored row-id list
     (`PRAGMA secure_delete=ON`);
  3. then unlinks the encrypted blob if no live upload of the workspace holds the same bytes, and
     proves it is gone (`BlobStore.get` must raise `BlobNotFound`).
* After deletion, every content path refuses with an explicit state:
  `GET /uploads/{id}`, registration and re-upload of the same bytes to the same project answer
  `410 upload_deleted`. Export, new splits and new experiments over an affected version answer
  `410 dataset_source_deleted`. Experiment and monitoring runs fail (`dataset_bytes` refuses). The
  details are `upload_id`, `deleted_at` and `reason` only.
* `GET .../versions/{v}/source` reports `available` or `deleted` (with time and reason).
* Kept, because they are immutable provenance and hold no uploaded values: dataset versions
  (spec, identity hash, content hash), split records, experiment artifacts, promotions and
  deployments. Nothing historical is mutated. Row ids in kept split records are why an id column
  must not hold sensitive values.
* `DELETE /datasets/{id}` deletes the source uploads of every version. It is refused
  (`409 dataset_source_shared`) when one of them also backs another dataset; delete that upload
  explicitly instead.
* Deletion is final for its upload id (a deterministic function of project, format and content):
  identical bytes cannot be re-uploaded to the same project. Upload them to a new project.
* Retention: `WYNK_UPLOAD_RETENTION_DAYS=N` declares it; `python -m api.tenancy purge-expired
  --data-dir ...` (run it on a schedule) deletes every live upload older than N days with reason
  `retention` and actor `system:retention`. It also removes blobs no live upload references
  (older than `--gc-older-than-s`, default 3600 s). This closes the #32 orphan-blob limitation.
* Scope of "not retrievable": through the application, its APIs and its stores. The OS, the
  filesystem, SQLite free pages before `secure_delete`, backups and snapshots are outside the
  application's control.

## 7. Audit trail

* `workspaces/<ws>/audit.sqlite3`, bound to its workspace (#32); `security_audit.sqlite3` at the
  data-dir root for failed authentication (no workspace was reached).
* Append-only: triggers refuse UPDATE and DELETE. Events are hash-chained
  (`event_hash = sha256(prev_hash || canonical event)`); `SQLiteAuditLog.verify_chain` detects
  an edited, removed or reordered event even if the triggers were dropped.
* Each event: sequence, id, time, workspace, actor (user id + API key id, or `system:retention`),
  action, resource type + id, result, stable error code, allowlisted details.
* Events: `dataset.upload`, `dataset.register`, `dataset.read` (upload with preview),
  `dataset.export`, `dataset.delete`, `audit.read`; refusals of those (`refused`, with the code);
  `security.access_refused` (a not-found or deleted dataset resource on any other route: a foreign
  workspace's id looks exactly like a missing one); `security.scope_denied`;
  `security.authentication_failed` (key id only if the token is well-formed).
* Results: `requested` (written **before** every state-changing action, so none happens
  unaudited), `ok`, `refused`, `denied`, `error`. If the trail cannot be written, the request
  fails closed (`503 audit_unavailable`) and no data (preview, export) is released.
* Redaction by construction: no free-text field. Detail keys are an allowlist and every string
  must match `^[A-Za-z0-9_.:/+,-]{0,160}$` (identifiers, hashes, codes, versions). An uploaded
  value, filename, column name, API key, secret or PII value cannot be recorded; the append
  refuses instead.
* `GET /api/v1/audit?after=&limit=` (scope `admin`) pages through the workspace's trail.

## 8. No user-data training

What the code guarantees, and tests enforce (`test_no_product_code_can_send_user_data_*`):

* Wynk trains no model. It has no training, fine-tuning, file-upload or batch code. The only
  outbound model call in the product is the OpenAI-compatible `POST <base_url>/chat/completions`
  (`runtime/backends/openai_compatible.py`). The request has exactly `model`, `messages`,
  `temperature`, `seed`, `max_tokens` and optionally `response_format`, and never asks the
  provider to store it.
* The only other outbound calls are optional. One is the hosted JEV decision POST
  (`runtime/backends/typesafe_jev.py`, request keys exactly `model`, `state`, `questions`, no
  storage opt-in). It runs only when `WYNK_JEV_API_KEY` and a pinned `WYNK_JEV_MODEL` are configured and only for GATHER `jev`
  over benchmark snapshots: GATHER `jev` refuses uploaded (inline csv/jsonl) rows before any call
  (`test_jev_never_reads_uploaded_dataset_rows`).
* The other is the hosted web-search GET (`runtime/backends/brave_search.py`, query
  parameters exactly `q`, `count` and four pinned filters, no storage opt-in). It runs only when
  `WYNK_WEBSEARCH_API_KEY` is configured AND the contract lists `websearch` in
  `allowed_sources`; no default, chat or benchmark contract does. The query is the current row's
  input values only, at most 400 characters: never the instructions, context columns (such as a
  passage), other rows, targets, secrets or workspace metadata
  (`test_an_uploaded_rows_context_and_other_rows_are_never_sent`). Opting a contract over
  uploaded data in to websearch therefore sends each row's inputs to the search provider.
* Uploaded rows reach a provider only as the inputs of an experiment or an invocation that a user
  of the workspace started, against a model the workspace's registry names.

What the code **cannot** guarantee: how a third-party provider retains, logs or uses requests it
receives. That is the provider's contract (choose one with zero data retention or no-training
terms, or a self-hosted endpoint). Secrets are always blocked or redacted before storage, so they
are never sent. Under the default policy, detected contact data is redacted too. Other personal
data in free text is not detected and is sent as uploaded.

## Invariants

* Cross-tenant access is impossible (#32 partitions; the audit trail is bound the same way).
* #32 API-key scopes and quotas are unchanged; deletion and the audit trail need `admin`. Quotas
  count live uploads only.
* Identical content gives the same `content_hash`, upload id, dataset identity and splits, across
  writes, workspaces and keys. If neither sensitive-data redaction nor CSV formula
  neutralization changes the upload, `content_hash = sha256(uploaded bytes)`, as before #33. If
  either one changes it, `content_hash = sha256(canonical transformed plaintext)` (section 3).
  Dataset identity, row ids, splits and provenance all use that same stored canonical
  representation.
* Splits stay `seeded_splits(identity_hash, row_ids, plan)`.
* Nonces never enter a hash.
* A deleted blob is not retrievable through the application.
* No raw value, key or secret reaches a log line, an error body or the audit trail.
* Security failures (missing key, unknown key, tamper, audit unavailable) fail closed.

## Deferred

* Metadata (SQLite) encryption; key management service integration; per-tenant master keys.
* Rate limiting and an audit-trail size bound (failed authentication is recorded per request).
* Detection of unstructured PII (names, addresses, free text); a policy per workspace.
* Streaming upload and parse (bodies are still held in memory, bounded by the size limit).
* Malware scanning (only plain-text CSV / JSONL is accepted, which is never executed).
* Crypto-shredding of backups; deletion of experiment artifacts (immutable by design).
