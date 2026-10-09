# Versioned champion inference

A #25 champion is a decision, not a service. Champion deployment turns it into one:

```
CHAMPION ─► immutable workflow version ─► STAGING ─► PRODUCTION ─► inference
                                                        ▲   │
                                                        └───┘ rollback (a previous version)
```

Code: `experiments/deployment.py` (engine and service), `store/deployments.py` (SQLite,
`deployments.sqlite3` next to `champions.sqlite3`), and `api/product.py` (`/api/v1/workflows`,
`/deployments`, `/inferences`). Nothing is re-derived by hand: every identity a version pins is read
from the experiment's #26 canonical artifact; execution is the existing compiler →
`WorkflowRunner` → #27 `ModelClient` path. A champion from fixed, random or ACO search is published,
deployed and served identically: the optimizer appears only in the version's provenance.

## 1. Publishing: an immutable workflow version

`ChampionDeployments.publish(champion_id)` accepts only a row of `champions` - which only a
`PROMOTED` decision writes. A candidate, a rejected challenger, a promotion id or a genome hash is
`champion_not_found`. Before anything is written:

1. the champion's promotion is re-derived from stored evidence (`ChampionPromotions.verify`); it
   must be `PROMOTED`, name this champion and cite a canonical artifact (a champion that predates
   #26 artifacts is `not_publishable`);
2. the artifact is loaded and verified end to end (`load_canonical`) and must be the one the
   promotion cited;
3. `workflow_version_document` cross-checks every component and fails closed
   (`not_publishable`, naming each disagreement):

| pinned | authority | cross-checked against |
| --- | --- | --- |
| genome + `genome_hash` | champion record | `Genome.from_canonical` hash; the provenance strategy run's `selected_genome_hash`; the contract grammar admits it |
| champion id / version / promotion / `decision_hash` / record sha256 / `compat_hash` | `champions` row + decision | the decision's champion, challenger status `CHAMPION`, cited artifact |
| `artifact_id`, `provenance_id`, `experiment_sha256`, `experiment_id`, `definition_hash`, run id / strategy / seed / optimizer | #26 artifact + `ProvenanceRecord` | the champion record's provenance; `compatibility_identity(definition, body, provenance)` equals the champion's |
| TaskContract (full), `contract_hash`, input + output schema | `ProvenanceRecord.contract` | its hash; the champion's `task_contract_hash` |
| grammar version, stage kinds, `grammar_hash` | `ProvenanceRecord.grammar` | `workflow_grammar(contract)`; the champion's grammar version |
| registry entry (full), `registry_entry_hash`, `model_hash`, configuration | `ProvenanceRecord.model` | entry hash; entry `model_hash`; the champion's `model_hash` / model; a provenance without a registry entry is `not_publishable` |
| model / prompt / compiler / grammar / benchmark versions | `ProvenanceRecord.versions.run_versions` (exactly one) | the champion's `run_versions`, model hash, prompt version, grammar version; benchmark must be `inline` |

The `wynk-workflow-version/1` document holds no timestamp and is content-addressed:
`version_id = "wv-" + canonical_hash(document)[:24]`. Publishing the same champion again returns the
same version (`200`); a stored document that no longer hashes to its id answers
`workflow_version_integrity_error` on every read. `workflow_versions` rows are immutable (UPDATE and
DELETE abort in a trigger), so a newer champion is always a new version.

Publishing creates the version `CREATED`: it is deployed nowhere and serves nothing.

## 2. Deployment lifecycle

```
CREATED ──stage──► STAGING ──promote──► PRODUCTION ──(replaced)──► RETIRED
                      └──(displaced by another stage)──► RETIRED      │
                                 PRODUCTION ◄──rollback── RETIRED ◄───┘ (only a former production version)
```

One deployment per lineage (the champion lineage id) holds a staging slot, a production slot and a
`revision`. `version_states` accepts only the transitions above (trigger), and partial unique
indexes keep at most one `STAGING` and one `PRODUCTION` version per lineage in the database itself.

| move | requires | fails with |
| --- | --- | --- |
| `stage(version, expected_revision)` | version `CREATED`; its model binds in this process | `invalid_deployment_transition`, `model_binding_failed`, `stale_deployment` |
| `promote(version, expected_revision)` | version `STAGING`; its champion is the lineage's **current** champion (record hash, version and `compat_hash` re-checked); its model binds | `invalid_deployment_transition`, `champion_not_current`, `model_binding_failed`, `stale_deployment` |
| `rollback(lineage, expected_revision[, version])` | a production version exists; the target was in production on this lineage before (default: the version the current one replaced) | `no_production_version`, `invalid_deployment_transition`, `stale_deployment` |

**Fencing.** Every move names the `revision` it was decided against. The store re-reads the
revision inside one `BEGIN IMMEDIATE` transaction and refuses with `stale_deployment` - writing
nothing - unless it is unchanged; otherwise it moves the slots and states, advances the revision by
exactly one (trigger-enforced) and appends the event, atomically. Two simultaneous promotions can
never both win, and a write based on an older view of the deployment never overwrites a newer one.

**Promotion is atomic.** The replaced production version becomes `RETIRED` in the same transaction
the new one becomes `PRODUCTION`: the lineage is never without, or with two, active versions.

**Rollback** re-activates the old immutable version itself - nothing is copied, re-optimized,
re-evaluated or re-promoted, and no version document changes. It records who (`actor`) replaced
which version with which.

**History.** `deployment_events` is append-only (trigger): `publish`, `stage`, `promote`,
`rollback`, each with the version, the version it replaced, the revision before and after, the
actor and the time. Everything is in SQLite, so a restarted server serves the same production
version.

`actor` is recorded as given by the client; the product API has no authentication yet.

## 3. Inference

```
POST /api/v1/workflows/{version_id}/invoke     {"inputs": {...}}   exact version (STAGING or PRODUCTION)
POST /api/v1/deployments/{lineage}/invoke      {"inputs": {...}}   the lineage's production version
```

1. **Request schema first.** `inputs` is validated against the pinned TaskContract
   `input_schema` (required fields, exact types, no unknown field - a target column is unknown;
   `null` on an optional field means absent) before anything is bound: `invalid_inference_request`
   (422, `details.field`) and no model call.
2. **Deployability.** Only `STAGING` and `PRODUCTION` versions serve (`workflow_version_not_deployed`
   otherwise); a lineage without production is `no_production_version`.
3. **Binding (every invocation, fail closed).** `bind_version` re-reads the model registry and
   refuses with `model_binding_failed` (503, `details.reason`) before any model call when:

   | reason | |
   | --- | --- |
   | `registry_entry_missing` | no entry pins the version's `model_hash` any more |
   | `registry_entry_disabled` | the entry is disabled |
   | `registry_entry_changed` | the entry differs from the pinned one (capabilities, revision, endpoint, ...) |
   | `model_hash_mismatch` | the client built for the entry reports another `model_hash` |
   | `model_identity_mismatch` | the client is bound to another registry entry |
   | `capability_unavailable` | the entry cannot serve the workflow's model stages |
   | `runtime_incompatible` | this runtime's model / prompt / compiler / grammar / benchmark versions differ from the pinned ones |
   | `grammar_incompatible` | the runtime's checker does not admit the pinned genome |

   Another model is never substituted.
4. **Execution.** The frozen genome runs through `WorkflowRunner.run_sync` (compiler → MAF →
   executors → `ModelClient`). No optimizer, evaluator, search or promotion is involved; there are
   no target values.
5. **Run identity + output schema.** The run's key must be the pinned genome, contract and versions;
   the answer must satisfy the pinned `output_schema`. Otherwise no output is returned:
   `output_schema_violation` / `inference_failed` (502, `details.inference_id`,
   `details.failure_kind`).

Every executed invocation appends an immutable `inference_records` row (`wynk-inference/1`):

| field | |
| --- | --- |
| `workflow_version`, `lineage_id`, `addressed_by` (`version` / `production`), `deployment_revision` | what served it |
| `champion_id`, `genome_hash` | the champion |
| `provenance` | `artifact_id`, `provenance_id`, `experiment_id`, `job_id`, `promotion_id`, `run_id`, `strategy`, `seed` |
| `model` | `model_hash`, `registry_entry_hash`, entry name |
| `versions` | model / prompt / compiler / grammar / benchmark versions |
| `run_id`, `request_sha256`, `output_sha256`, `failure` | the run; inputs and output are kept as hashes only |
| `usage` | measured `model_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_s`; `cost` only when the pinned registry entry has prices (`cost_authoritative`) |

`GET /api/v1/inferences/{inference_id}` returns the record (without the output). Monitoring, drift
detection and re-optimization (#31) are in [production_monitoring.md](production_monitoring.md).

## API

```
POST /api/v1/workflows                          {"champion_id", "actor"?}                201 | 200
GET  /api/v1/workflows?lineage=...
GET  /api/v1/workflows/{version_id}
POST /api/v1/workflows/{version_id}/stage       {"expected_revision", "actor"?}
POST /api/v1/workflows/{version_id}/promote     {"expected_revision", "actor"?}
POST /api/v1/workflows/{version_id}/invoke      {"inputs"}
GET  /api/v1/deployments/{lineage}              slots + revision
GET  /api/v1/deployments/{lineage}/history
POST /api/v1/deployments/{lineage}/rollback     {"expected_revision", "version_id"?, "actor"?}
POST /api/v1/deployments/{lineage}/invoke       {"inputs"}
GET  /api/v1/inferences/{inference_id}
```

A server without `--model-registry` keeps versions and deployments inspectable, but staging,
promotion and inference answer `inference_backend_unavailable`.

## Tests

`tests/test_champion_inference.py`: only a promoted champion publishes; every identity is pinned
from provenance and cross-checked; a champion without a registry entry is refused; versions are
immutable (triggers + content address); publishing deploys nothing; staging ≠ production;
explicit, fenced, atomic promotion; one production version per lineage; only the current champion
is promoted; rollback restores the exact old version and records history; restart preserves
production; stale and concurrent promotions fail closed; missing / disabled / changed registry
entries, swapped clients and incompatible prompt or compiler versions are refused before any model
call; bad requests are refused before binding; output schema enforced; fixed, random and ACO
champions deploy identically; the exact champion genome runs with every optimizer, evaluator and
promotion entry point trapped; inference records trace to the #26 artifact and provenance.
