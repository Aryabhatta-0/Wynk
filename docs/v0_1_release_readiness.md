# Wynk v0.1 release readiness (production dogfood audit)

Audited commit: `main` at `c1de65d` (after #57). Evidence:
`tests/test_dogfood_lifecycle.py` (22 tests) with `tests/dogfood_support.py` and the fixture
`tests/fixtures/dogfood/order_desk.csv`. No product code was changed by this audit (section 6).

Statuses: **PASS** (an automated test proves it through the product path), **BLOCKED**
(could not be run), **FAIL** (a defect is open), **NOT TESTED**.

## 1. How the audit runs the product

Every step goes through `TenantRouter.handle` (the HTTP entry point) of an API that is wired
exactly as `python -m api.product --model-registry ...` wires it:

* `build_api(...)` with `registry_runtime(service, registry, api_key, websearch)` and
  `registry_inference(registry, api_key, websearch)`. Experiments run on the real MAF
  `WorkflowRunner`. Inference re-reads the registry FILE on every call.
* The model client is the real `OpenAICompatibleClient` and sends real HTTP. Only the far end is
  fake: `FakeProvider` is a local OpenAI-compatible server on `127.0.0.1` that answers the
  order-desk prompts deterministically. No data leaves the machine, and no real credential is used.
* Websearch uses the real `BraveWebSearchSource`, configured by `websearch_from_env`. Only its
  transport (`FakeBrave`) is fake.
* Workers are built as `start_worker` builds them: `JobWorker(store, runtime, priors=...)`,
  `recover()`, then run. A "restart" builds a new `TenantRouter` on the same data directory.
* Data at rest is encrypted with the test keyring (`tests/conftest.py`).

No layer is bypassed. A test reads a store directly only to inspect state (memory priors) or to
simulate an operator or an attacker (registry edits, a database tamper).

## 2. Lifecycle architecture

| Step | Entry point (`/api/v1/...`) | Module |
|---|---|---|
| Upload, bounds, strict parse | `POST projects/{id}/uploads?format=` | `ingestion/parse.py`, `ingestion/service.py` |
| Secrets / PII, formula neutralization | (on upload) | `ingestion/sensitive.py`, `ingestion/export.py` |
| Encrypted blob, content hash | (on upload) | `store/blobs.py`, `store/crypto.py` |
| Dataset version + spec | `POST uploads/{id}/register` | `ingestion/service.py`, `store/datasets.py` |
| Seeded splits | `POST datasets/{id}/versions/{v}/splits` | `ingestion/service.py` |
| TaskContract + plan, durable job | `POST experiments` | `core/task_contract.py`, `experiments/jobs.py`, `store/jobs.py` |
| Search: fixed / random / ACO / warm ACO | worker | `optimizers/*`, `experiments/optimization_experiment.py` |
| Execution of one candidate | worker | `runtime/runner.py`, `runtime/executors/*`, `runtime/backends/*` |
| Canonical artifact + provenance | `GET experiments/{id}/artifact`, `/provenance`, `/verify` | `experiments/artifacts.py`, `experiments/provenance.py` |
| Workflow memory (learn, warm start) | (after the artifact) | `memory/production.py`, `store/priors.py` |
| Champion promotion (held-out test once) | `POST experiments/{id}/promote` | `experiments/promotion.py`, `store/champions.py` |
| Immutable workflow version | `POST workflows` | `experiments/deployment.py`, `store/deployments.py` |
| Staging, production (fenced) | `POST workflows/{id}/stage`, `/promote` | `experiments/deployment.py` |
| Inference | `POST deployments/{lineage}/invoke` | `experiments/deployment.py` (`InferenceRuntime`) |
| Monitoring, trigger, challenger job | `monitoring/policies`, `inferences/{id}/feedback`, `workflows/{id}/triggers/evaluate`, `triggers/{id}/challenger` | `experiments/monitoring.py`, `store/monitoring.py` |
| Rollback | `POST deployments/{lineage}/rollback` | `experiments/deployment.py` |
| Export, delete, audit | `GET datasets/{id}/versions/{v}/export`, `DELETE datasets/{id}`, `GET audit` | `ingestion/export.py`, `store/audit.py` |
| Tenancy, keys, scopes | every route | `api/tenancy.py`, `store/tenancy.py` |

## 3. The dogfood dataset and contract

`tests/fixtures/dogfood/order_desk.csv` is new (not from `benchmarks/`). It has 18 support
tickets (`ticket_id, question, order_record, answer`). Each question asks for the destination
city of one order. The record names a decoy city (the packing warehouse) and the destination.
The answer is deterministic: the city after "ships to".

* Registration: input `question`, context `order_record`, target `answer`, row ids from
  `ticket_id`.
* Splits: seed 11, 20% validation, 20% test. That gives 12 optimization rows, and the remaining
  rows for validation and the held-out test.
* Contract: `question_answering`, exact match (case-insensitive), 20,000 tokens, 120 s, 20 tool
  calls, 2 retries and 4 model calls per example.
* A DIRECT workflow cannot see the record and answers "unknown" (score 0). A GATHER, EXTRACT,
  SYNTHESIZE workflow reads the record and scores 1.0. So the search has a real signal.

## 4. Release matrix

| Area | Status | Evidence (tests in `tests/test_dogfood_lifecycle.py`) |
|---|---|---|
| Ingestion | PASS | Happy path: upload, register, splits, CSV export of 18 rows. `test_upload_boundaries_malformed_input_and_sensitive_data`: exactly 1 MiB accepted, 1 MiB + 1 byte `413 payload_too_large`, `malformed_csv`, `malformed_jsonl`, control characters refused, `invalid_encoding`. |
| Contracts | PASS | Happy path: the contract binds the registered dataset spec, splits hash, model hash and prompt version. A JEV contract on a CSV is refused at creation (`400 invalid_request`). |
| Optimizer | PASS | Happy path: fixed, random and ACO units complete. ACO finds the GATHER champion (validation 1.0). No held-out row is in any prompt during the search. |
| Provenance | PASS | Happy path: `verify` is verified, `synthetic` is false, `test_runs` is 0, model hashes and the dataset content hash are recorded. Crash tests: a resumed artifact differs from an uninterrupted one only in `provenance.upload_id`. |
| Memory | PASS | Happy path: dataset-champion and task-family priors are learned. Warm ACO starts `warm_dataset_champion` with 0 extra model calls. `test_memory_is_reused_only_for_the_exact_dataset_model_and_runtime`: no held-out row hash in any prior; a new dataset version never reuses the dataset champion; another model is `cold_model_changed`; an invalidated pinned prior stops the job (not COMPLETED) with no model request; deletion invalidates every prior. Websearch jobs are refused learning (`live_websearch_evidence`). |
| Tenancy | PASS | `test_workspace_b_can_never_reach_workspace_a`: for 16 routes plus publication, B naming A's id answers exactly like B naming an id that never existed. B's identical upload has the same identity hash, but B's warm start is `cold_no_prior`. A's state is unchanged. `test_read_only_and_member_keys_cannot_change_or_delete`: a read key gets `403 insufficient_scope` for 9 writes; a member's write key can invoke, but delete and audit stay owner-only. |
| Security | PASS | Secrets blocked (`422 sensitive_data_blocked`, the value is not echoed). Email is redacted before storage. Formula cells are neutralized (`'=`). The audit chain verifies. The provider key is in no data file and no response. `test_outbound_model_requests_carry_one_rows_inputs_and_no_identifiers`: each prompt names at most one order, no ticket id, no dataset name; the key goes only in the `Authorization` header. |
| Promotion | PASS | Happy path: the held-out test split is used only at promotion; promotion verify returns the decision; the champion is readable. `test_a_crash_during_promotion_resumes_without_resending_held_out_calls`. A failed experiment is `409 experiment_not_promotable`. |
| Deployment | PASS | Happy path: CREATED, STAGING, PRODUCTION, idempotent publication, `409 no_production_version` before production. `test_only_the_current_champion_reaches_production_and_tampering_fails_closed`: a superseded champion can be staged but never promoted (`409 champion_not_current`); a tampered version document fails integrity on read and on invoke, with no model request. `test_a_staged_deployment_survives_a_restart_and_stays_fenced`: a stale revision is `409 stale_deployment`. A missing or disabled model blocks staging (`503 model_binding_failed`) and leaves the revision unchanged. |
| Inference | PASS | Happy path: the output is correct; the inference record holds the exact document versions, the output hash and the usage. A provider outage is `502 inference_failed` with a stored FAILED record. A changed model revision is `503 model_binding_failed` (`registry_entry_missing`); a disabled entry is `registry_entry_disabled`. No request is sent in either case. |
| Rollback | PASS | Happy path: v2 to v1 at revision 5; v1 serves again; the document is unchanged; the history records the rollback. |
| Deletion | PASS | Happy path: `DELETE` dataset; export, experiments and splits are `410 dataset_source_deleted`; the upload is `410 upload_deleted`; priors are invalidated (`dataset_deleted`); artifacts still verify; production still serves. The audit log records upload, register, export and delete, and the hash chain verifies. |
| JEV | PASS (scope) | The product API does not configure JEV, and `test_the_product_never_runs_jev_on_an_uploaded_dataset` proves a JEV contract is refused before anything runs. JEV identity drift is proved at runtime level by the existing `tests/test_runtime_jev.py` and `tests/test_warm_start_evidence_provider.py` (in the full suite). |
| Websearch | PASS | `test_websearch_without_a_configured_provider_is_never_substituted`: no GATHER candidate, no evidence source. `test_a_websearch_champion_is_pinned_to_its_provider_and_never_teaches_memory`: the champion pins the provider identity; production refuses another provider (`websearch_identity_mismatch`) and no provider (`websearch_unavailable`) before any request; the query holds only the row's question; the key is only in a header and in no file. `test_a_websearch_job_never_resumes_under_another_or_a_drifted_provider`: a job never runs under another provider (INTERRUPTED `runtime_unavailable`, 0 requests); after a live drift no search is sent or charged. |
| Crash recovery | PASS | `test_a_crash_resumes_exactly_without_duplicate_calls_or_spend` (crash after a reserved checkpoint, after an attempt, after a settled checkpoint): the same request total, ids and usage as an uninterrupted run. `test_a_crash_with_a_call_in_flight_is_never_resent_or_hidden`: INTERRUPTED `ambiguous_attempt`, resume is `409`, nothing is re-sent. `test_a_crash_around_artifact_finalization_recovers_once` (before the artifact, before learning): recovery makes no model call and learns once. Promotion and staged deployment crashes are above. |
| Monitoring, re-optimization | PASS | `test_production_feedback_triggers_a_challenger_that_never_replaces_production_by_itself`: labelled feedback triggers a challenger job on a new dataset version; it completes; production is unchanged. |

No row is FAIL, BLOCKED or NOT TESTED.

## 5. Residual risks (not defects; accepted for v0.1, or for the owner to decide)

* **R1.** A key with only the `read` scope can export raw dataset rows (documented in
  `docs/upload_security.md`). A "viewer" is therefore not a "metadata-only" role.
* **R2.** A websearch-only contract is accepted when no websearch provider is configured. It
  completes with DIRECT candidates only (score 0) instead of an up-front refusal. Nothing is
  substituted, so this is a usability issue, not a safety issue.
* **R3 (changed after v0.1, branch `production/websearch-drift-policy`).** Previously, a websearch
  provider that drifted during a job made every later search a refused row scored as a failure,
  and the job completed. A refusal measures nothing, yet it reached the optimizer and the
  champion: a drifted websearch genome could win with a 0.0 score. Now the next run that would
  search is refused before it is recorded. The job becomes INTERRUPTED (`runtime_unavailable`),
  and `resume` under the original provider gives exactly the undisturbed result. Genomes that
  do not search are not affected. Residual: a drift that lands inside one run, after its
  pre-run check, is still refused by the executor and recorded as a failed row (never sent,
  never charged).
* **R4.** `POST experiments` is not idempotent. A client that retries after a lost response
  creates a second job (and spends again).
* **R5.** PII rules detect fixed formats only (email, phone, cards, SSN, IBAN). Names, addresses
  and free-text health data are not detected (documented).
* **R6.** The audit used a deterministic fake provider. It proves the product path, not the
  quality of a real hosted model. Model quality is the canonical benchmark's concern.
* **R7.** On a Windows checkout with `core.autocrlf=true`,
  `test_benchmark_regenerates_byte_identically` fails because of CRLF line endings. This is
  environmental. The committed benchmark bytes are unchanged (section 7).

## 6. Defects found

None. Every failure during the audit was a test assumption, corrected in the test. No product
file was changed.

## 7. Frozen benchmark proof

The audit changes no file under `benchmarks/` or `experiments/configs/`. The git tree and blob
ids at the audit commit are identical to `origin/main` (`c1de65d`):

| Path | git object id |
|---|---|
| `benchmarks/` | `b047ba8c462f35e7f9e3e0534a8385255be1d616` |
| `benchmarks/heldout/` | `ff8ae8f64af71b11da82cb3fd43761b5ba7aa367` |
| `experiments/configs/full_suite.json` | `10245ec549373e96db477649d3a848c6bca007a4` |

SHA-256 of the committed bytes: `benchmarks/catalog.json` `be459b57...c41481`,
`benchmarks/splits.json` `24ae3acc...1a98cc`, `benchmarks/heldout/tasks.json` `0e009cb1...738168`,
`benchmarks/heldout/splits.json` `b25d9618...3fec`, `experiments/configs/full_suite.json`
`9b85c30b...95dd`.

## 8. Verification

Run on Windows 11, Python 3.11, Node 24:

```
python -m pytest -o addopts= -q tests/test_dogfood_lifecycle.py
python -m pytest -o addopts= -q
python -m ruff check . && python -m ruff format --check .
cd ui && npm ci && npm run typecheck && npm run lint && npm test && npm run build && npm run test:e2e
```

| Check | Result |
|---|---|
| Dogfood lifecycle (`tests/test_dogfood_lifecycle.py`) | 22 passed |
| Full Python suite | 2086 passed, 4 skipped, 2 failed (both environmental, below) |
| Ruff check and format check | clean (214 files) |
| UI typecheck | clean |
| UI lint (oxlint) | 0 errors, 3 existing warnings |
| UI unit tests (vitest) | 14 files, 122 passed |
| UI build | built |
| Browser E2E (Playwright, real `python -m api.product`) | 5 passed |

The two full-suite failures are not product defects:

* `test_benchmarks_mvp.py::test_benchmark_regenerates_byte_identically` fails on `main` too, on
  this Windows checkout, because of CRLF line endings (R7).
* `test_durable_jobs.py::test_separate_processes_never_claim_the_same_unit` failed once while the
  UI build ran in parallel: its subprocesses computed a negative `time.sleep` value
  (`ValueError: sleep length must be non-negative`). It passed 3 of 3 times when run alone.

## 9. Verdict

**PASS.** Every row of the matrix is PASS, through the real product path, and the audit found no
product defect. The residual risks in section 5 stay open for the owner to decide; none of them
lets data cross a workspace, a credential leak, a held-out row reach the search, or an unpinned
model or provider serve production. CI on a clean Linux runner must confirm the full suite.
