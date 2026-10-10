# Authentication, workspaces and tenant isolation (#32)

`authenticated principal -> workspace -> project -> tenant-scoped resources`

No workspace can read, modify, invoke, monitor or discover another workspace's resources.
Privacy / security hardening (encryption at rest, PII handling, rate limiting, audit export) is
#33 and not part of this change.

Code: `store/identity.py` (users, workspaces, memberships, API keys, migrations),
`store/tenancy.py` (store-level workspace binding), `api/tenancy.py` (`TenantRouter`:
authentication, scopes, quotas, workspace / member / key endpoints, legacy migration, CLI),
`api/product.py` (`build_api` now returns the router; `ProductAPI` is one workspace's API).
Tests: `tests/test_tenancy.py`.

## 1. Identity model

| record | identity | mutable |
| --- | --- | --- |
| User | `usr-<20 hex>`, unique email, display name | never |
| Workspace | `ws-<id>`, name, creator | never |
| WorkspaceMembership | `mem-<20 hex>`: workspace, user, role `OWNER` / `MEMBER` | `removed_at/by` once |
| API key | `key_id` (16 hex): workspace, user, name, scopes, secret hash, expiry | `revoked_at/by` once, `last_used_at` (monotonic) |

All of it lives in `<data>/identity.sqlite3`; triggers refuse every other UPDATE and every
DELETE. A user has at most one ACTIVE membership per workspace; a role change is a removal plus
a new membership. A workspace always keeps at least one OWNER. No OAuth / social login: the ids
are internal and an external identity provider can later map onto `users`.

## 2. Ownership: the partition is the tenant authority

Each workspace has its own partition, `<data>/workspaces/<workspace_id>/`, holding its own
`metadata.sqlite3` (projects, uploads, datasets, versions, splits), `blobs/`, `jobs.sqlite3`
(jobs, units, attempts, artifacts), `champions.sqlite3` (promotions, champions),
`deployments.sqlite3` (workflow versions, deployments, inference records) and
`monitoring.sqlite3` (feedback, policies, trigger decisions, re-optimizations).

* **One canonical authority.** A resource belongs to the workspace whose partition it was
  written to. Every derived resource is created by the services of the partition that holds its
  source (project -> upload -> dataset -> splits -> job -> artifact -> promotion -> champion ->
  workflow version -> deployment -> inference -> feedback -> trigger -> re-optimization job), so
  it inherits the workspace of its whole chain. No `workspace_id` is copied into records, and no
  caller-supplied workspace or project id is ever trusted.
* **Stores enforce it, not just routes.** Every SQLite store holds an immutable
  `tenant_binding` row (UPDATE / DELETE triggers) and the blob store a write-once `TENANT` file.
  Each store constructor checks the binding before any read or write: a store opened for
  another workspace, or opened unscoped while bound, raises `TenantBindingError`. When a
  partition opens, the router also checks that all six handles name the same workspace. Any
  disagreement fails closed (`tenant_integrity_error`, 500) for that workspace only.
* **Never changes workspace.** A binding cannot be rewritten, and partitions are never merged.
* **Overlapping ids are fine.** Two workspaces can both have dataset `capitals`, lineage
  `capitals.capitals`, the same content-addressed policy id or the same upload bytes. Neither
  sees the other's, and idempotent creates answer "created" in each workspace, so dedup reveals
  nothing.
* **No identifier oracle.** Another workspace's id is absent from your partition, so the
  answer is byte-for-byte the `*_not_found` you get for an id that never existed (the e2e test
  compares both responses). Admin endpoints work the same way: another workspace's key id or
  member gives `api_key_not_found` / `member_not_found`, and another workspace id gives
  `workspace_not_found`.

## 3. API keys

Format: `wynk_sk_<key_id: 16 hex>_<secret: 43 url-safe base64 chars>` (a 256-bit secret).

* The plaintext is returned once, in the create response (`secret`), and stored nowhere. It is
  never logged or persisted (the test scans every file in the data dir and the captured logs).
* Stored: `key_id` (the lookup prefix), name, scopes, `secret_hash =
  sha256("wynk-api-key/1" NUL key_id NUL secret)` compared with `hmac.compare_digest`,
  `created_at`, optional `expires_at` (`expires_in_s`), `revoked_at/by`, and `last_used_at`
  (written at most once a minute). A salted fast hash fits a uniformly random 256-bit secret;
  a password KDF would add latency without adding security.
* `Authorization: Bearer <key>`. An unknown, malformed, wrong, revoked or expired key, a key
  whose user is no longer a member, and a key of a missing workspace all get the same `401
  unauthenticated`.

## 4. Scopes and roles

| scope | allows |
| --- | --- |
| `read` | every product `GET`, `GET /workspace`, members list, quota |
| `write` | create / change: projects, uploads, register, splits, experiments (create / cancel / resume / promote), publish / stage / promote / rollback, monitoring policies, **feedback**, trigger evaluation, re-optimization |
| `invoke` | `POST /workflows/{id}/invoke`, `POST /deployments/{lineage}/invoke` |
| `admin` | create workspaces, add / remove members, create / list / revoke keys |

Scopes don't imply one another. `admin` counts only for an `OWNER`: it's dropped at
authentication for a `MEMBER`, and a `MEMBER` can't be issued it (`invalid_scopes`). A new key
can't exceed its creator's scopes. Removing a member disables all of their keys immediately.
Scope checks run before quota checks and before any resource is read (`403
insufficient_scope`).

## 5. Endpoints

Public: `GET /api/v1/health`. Everything else needs a key. Authentication comes first, so an
unauthenticated caller can't even probe routes.

| method + path | scope |
| --- | --- |
| `POST /workspaces {name}` -> workspace, OWNER membership, one-time all-scope key | admin |
| `GET /workspace`, `GET /workspaces/{id}` (own only) | read |
| `GET /workspace/members` | read |
| `POST /workspace/members {email, display_name?, role}` | admin |
| `POST /workspace/members/{user_id}/remove` | admin |
| `POST /workspace/keys {name, scopes, user_id?, expires_in_s?}` -> key + one-time `secret` | admin |
| `GET /workspace/keys` (metadata only) | admin |
| `POST /workspace/keys/{key_id}/revoke` (idempotent) | admin |
| `GET /workspace/quota` -> limits + usage | read |

The existing product routes keep their paths and are scoped to the key's workspace. There's
no `/workspaces/{id}/...` duplicate family.

## 6. #30 / #31 principals

Inference records and feedback gain `authenticated_principal` = `{workspace_id, user_id,
key_id, authenticated_by: "api_key"}`, written by the server from the key. It is `null` for
in-process calls and for records written before #32. The #30 / #31 `actor` / `source` fields
stay `untrusted_metadata` with `metadata_trusted: false`: they are still not authenticated, and
the new field doesn't change that. Neither field enters any identity or decision.

## 7. Quotas

`Quotas` (defaults in brackets; configure them with `build_api(quotas=...)` or
`WYNK_QUOTA_<NAME>`, e.g. `WYNK_QUOTA_PROJECTS=20`):
`projects` [50], `uploads` [500], `stored_bytes` [1 GiB of uploads], `dataset_versions` [500],
`active_jobs` [4: create / resume experiment, re-optimize], `workflow_versions` [200],
`api_keys` [50 active].

* **Authority: the stores' own insert transactions.** Every limit except API keys is a
  `store.tenancy.StoreQuota` that the partition's stores check INSIDE the `BEGIN IMMEDIATE`
  transaction that inserts the resource:
  `create_project` / `put_upload` / `add_version` (`store/datasets.py`), `create_job`
  (`store/jobs.py`) and `publish` (`store/deployments.py`). The count and the insert commit
  together, so no path can exceed a limit: not an API route, not a trigger's automatic
  re-optimization, not a second process. A refusal (`429 quota_exceeded`, `details: {resource,
  limit, used}`) writes nothing. The counts are rows that exist, so there's no separate counter
  to drift. API keys are counted in the identity store under its write lock.
* **Job admission point:** `SQLiteJobStore.create_job`. Every new #24 job goes through it:
  `POST /experiments` and the challenger that `ProductionMonitor._run` creates for both
  `POST /triggers/{id}/reoptimize` and `POST /workflows/{v}/triggers/evaluate` (which
  re-optimizes automatically on a TRIGGERED decision). `reoptimize` also asks
  `SQLiteJobStore.admit_job()` before it claims or builds anything. That check is advisory, but
  it means a refusal leaves no claim and no new dataset version behind. If the in-transaction
  check refuses anyway (a race), the claim is released.
  * Trigger evaluation records the decision (evidence, not a job) and defers the challenger,
    the same way it already does when no backend is configured. The response is the normal
    201 / 200. `GET /triggers/{id}/challenger` shows no job, and `POST /triggers/{id}/reoptimize`
    returns `429 quota_exceeded` with `trigger_id`. Once capacity frees, the same trigger
    creates its single deterministic challenger (`j-<hash(trigger_id)>`, never duplicated).
    NOT_TRIGGERED, SUPPRESSED and idempotent re-evaluations are unaffected.
* **Retries that create nothing never hit a full quota.** The check runs only when the insert
  would create a row:
  * An identical upload: it's checked before its bytes reach the blob store, and an existing
    upload id skips the check.
  * An identical dataset registration: dedup returns before `add_version`.
  * Republishing a champion: its existing version returns before the count.
  * An existing challenger job: it's looked up before `create_job`.
  * Resume: an INTERRUPTED job already counts as active, so resuming creates no new active job
    and has no check.
* **Limitations (documented, not fixed here):**
  * The byte quota is checked after the body is read. It's still bounded by the per-upload
    `max_upload_bytes` limit, and nothing is stored before the check.
  * A re-optimization's derived dataset version counts against `dataset_versions` like any
    registration. So a full `dataset_versions` quota also defers a trigger's challenger, with the
    same 429 on `reoptimize`.
* No billing.

## 8. Existing single-tenant data (migration)

A data dir that still has stores at its root (the pre-#32 layout) is refused:
`build_api` raises `LegacyDataNotMigrated`, so it is never served and ownership is never
guessed. An operator assigns it explicitly:

```bash
python -m api.tenancy migrate-legacy --data-dir .wynk-data --owner-email ops@example.com \
    [--workspace-id ws-legacy] [--workspace-name "Legacy workspace"]
python -m api.tenancy issue-key --data-dir .wynk-data --workspace-id ws-legacy \
    --email ops@example.com --scopes read,write,invoke,admin      # prints the key once
```

What the migration does:

1. It records a `legacy_migrations` row (source = this data dir) for the named workspace, and
   in the same transaction creates that workspace, the owner user (or reuses that email's user)
   and the OWNER membership.
2. It checkpoints each SQLite WAL, then renames every legacy store (`metadata`, `jobs`,
   `champions`, `deployments`, `monitoring`, `blobs`) into `workspaces/<id>/`. Records are
   moved, never rewritten, so every hash, id and provenance chain stays valid.
3. It binds every store to the workspace and marks the migration completed.

It's idempotent and resumable. A re-run after completion does nothing (`already_migrated`). A
re-run after a crash finishes the remaining steps. Naming a different workspace for the same
data, finding a store both at the root and in the partition, or legacy stores reappearing after
completion all refuse. Every old project and its whole derived chain belong to that one
workspace. Nothing is split or inferred.

**Fresh / dev:** `python -m api.tenancy bootstrap --data-dir D --owner-email E` creates
`ws-default` and prints an owner key once. For dev and E2E servers,
`WYNK_BOOTSTRAP_API_KEY=<key in the format above>` makes `python -m api.product` / `api.chat`
idempotently register that exact key (hash only) for `ws-default`. The UI dev proxy sends
`$WYNK_API_KEY` server-side, so the browser bundle never holds a key.

## 9. Operations

* Job workers: one per workspace partition, each over that workspace's job store only
  (`start_worker(router)`; workspaces created later get theirs when first opened). Failures
  are isolated: a workspace whose partition fails its binding or integrity check is logged, gets
  no worker and keeps answering `tenant_integrity_error`, while every other workspace starts
  normally. With no experiment runtime configured, `start_workers` opens no partition at all.
* Artifact CLI: `python -m experiments.artifacts verify --data-dir D --workspace ws-... --job
  j-...` (the partition's stores are checked against `--workspace`).

## 10. Proof

`tests/test_tenancy.py`:

* **Two-tenant e2e (`@maf`).** A and B upload the same capitals bytes under the same dataset
  id, lineage and policy, and each runs project -> upload -> register -> splits -> Fixed /
  Random / ACO experiment -> held-out promotion -> publish -> stage -> promote -> production
  inference -> labelled feedback -> trigger -> challenger job, all through the authenticated
  API. Then 27 boundary checks use B's key on A's ids (project, project datasets, upload,
  register, splits, job, artifact, provenance, verify, cancel, promote, promotion, publish A's
  champion, workflow get / stage / invoke, monitoring, drift, triggers list / evaluate,
  inference get / feedback, feedback, trigger, challenger, reoptimize). Each one must equal the
  response for a random id with the id substituted (publishing A's champion is compared by
  status and code). The one exception is the content-derived splits hash, which resolves to B's
  own identical splits. A's stores must be byte-identical afterwards.
  Invoking the shared lineage name with B's key runs B's own version and records B's
  principal. The recorded principal is checked against the untrusted actor.
* Revoked, expired, wrong, unknown and malformed keys get identical 401s. Every product route
  is 401 without a key. Missing scope is 403 with nothing written. Plaintext keys appear in no
  file and no log. Caller-supplied workspace headers and params are ignored or refused.
  Cross-tenant project ids are not reusable. Role rules: no admin scope for members, the last
  owner can't be removed, and removal disables keys. Quotas refuse atomically for projects,
  bytes, uploads, keys and active jobs. Stores refuse foreign and unscoped opens, and a
  tampered binding fails closed. A restart preserves workspaces, keys and isolation. Legacy
  migration is explicit, idempotent and resumable after a crash. The bootstrap key is
  idempotent and stored as a hash only.
* Existing #18-#31 API tests (`test_product_api`, `test_durable_jobs`, the UI fixture) now run
  through the router with an authenticated dev workspace (`tests/tenancy_support.py`). The
  in-process service suites are unchanged.
* **Mutation checks.** Each of these makes the suite fail: removing the store tenant predicate,
  accepting a revoked key, trusting a caller workspace header, skipping the scope check, and
  resolving the tenant from a project id, removing job admission from `create_job`, letting
  trigger evaluation propagate the deferred challenger's refusal, and un-isolating worker
  start.
* Review regressions: with `max_active_jobs=1` and one experiment active, a TRIGGERED
  evaluation admits no hidden challenger. After the slot frees, the same trigger creates
  exactly one. A corrupted workspace doesn't stop `start_workers`, and the healthy workspace's
  worker runs its jobs. Identical upload, registration and publish retries pass a full quota.
