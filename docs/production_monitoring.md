# Production monitoring and re-optimization triggers

Closes the production loop:

```
deployed workflow version (#30)
  ─► #30 inference records  +  quality feedback bound to them
  ─► windowed aggregates · drift statistics · trigger rules            (deterministic, no LLM)
  ─► NOT_TRIGGERED | TRIGGERED | SUPPRESSED   (immutable, with reasons + evidence)
  ─► TRIGGERED: freeze labels ─► NEW dataset version ─► splits ─► NEW challenger job (#23/#24)
  ─► nothing promoted, nothing deployed: #25 promotion and #30 stage/promote stay explicit
```

Code: `experiments/monitoring.py` (engine), `store/monitoring.py` (SQLite, `monitoring.sqlite3`
next to `deployments.sqlite3`), `api/product.py` (routes). Tests:
`tests/test_production_monitoring.py`.

## 1. Telemetry authority

The only telemetry is the #30 `inference_records` row (`wynk-inference/1`): workflow version,
provenance (artifact / provenance / experiment / job / promotion ids), `model_hash`, tokens,
latency, model calls, cost when the pinned registry entry prices the model, and failures.
A request the pinned input schema REJECTS on a STAGING / PRODUCTION version is recorded there
too, before any binding or model call: `FAILED`, `failure.kind = "input_schema_invalid"`, zero
usage, no run, the same version / provenance / model pins, the request hash, and value-free
`failure.violations` (`{field, code: missing|type, observed_type}` for a pinned field;
`{field_sha256, code: unknown}` for an unknown key - its name is never stored verbatim, and no
value ever is). The 422 names its `inference_id`.
`store/monitoring.py` has no telemetry table. The deployment store gained one read
(`inferences(version_id)`); records stay append-only (triggers). Every record read for an
aggregate is re-checked to cite exactly the version, its artifact, provenance, genome and model
hash, or the read fails closed (`monitoring_integrity_error`).

## 2. Quality feedback (`wynk-feedback/1`)

`POST /api/v1/inferences/{inference_id}/feedback`
`{"inputs", "output"?, "expected"?, "actor"?, "source"?}`

* `inputs` must validate against the pinned input schema and hash to the record's
  `request_sha256`; `output` must hash to `output_sha256` (and be absent for a FAILED inference).
  Feedback can only describe the request that was actually served (`feedback_mismatch`).
* `expected` is validated against the pinned output schema and judged by the pinned
  TaskContract evaluator (`evaluate_prediction`): the stored `evaluation` is the deterministic
  `EvaluatorRecord` (quality, passed, evaluator version, spec hash). Omitted: an unlabelled input
  observation (drift only).
* Persisted: feedback id, inference id, workflow version, lineage (champion, artifact,
  provenance, experiment, source job, promotion, dataset id/version/hash, contract and model
  hash), inputs, output, expected, evaluation, the inference's timestamp, received timestamp,
  and `untrusted_metadata` `{actor, source}` with `metadata_trusted: false`.
* Identity `fb-<hash(inference_id, expected)>`: one feedback per inference. The same label again
  returns the stored one (200, metadata NOT overwritten); a different label is
  `feedback_conflict` (409). Feedback rows are immutable (UPDATE/DELETE triggers).
* Until #32, actor/source are opaque: they enter no identity, aggregate, drift statistic or
  decision.

## 3. Aggregates (`wynk-monitoring-summary/1`)

`GET /api/v1/workflows/{version_id}/monitoring?since=&until=&policy_id=`

Per workflow version, over inference records with `since <= created_at < until` (default: the
policy window ending now): count / succeeded / failed / failure rate / failure kinds; latency
mean, p50, p95 (nearest rank); model calls total + mean; prompt / completion / total tokens
total + mean; cost total + mean only when every record is cost-authoritative (else `null`);
feedback count, labelled count, mean quality, pass rate. `evidence` lists every inference id
and feedback id used plus their hash. Rejected requests (`input_schema_invalid`) are reported as
`rejected_inputs` and are NOT inferences: they are excluded from count / failure rate / latency /
tokens / cost (a client error is not a workflow failure, so it cannot fire `max_failure_rate`),
but their ids stay in `evidence`. Records of another version are never read, and an
aggregate handed one refuses to compute. All statistics rounded to 6 digits.

## 4. Drift (`wynk-drift-report/1`)

`GET /api/v1/workflows/{version_id}/drift?since=&until=&policy_id=`

Reference = the champion's own selection evidence: the **optimization + validation** rows of the
dataset version it was selected on (the test split stays sealed for #25), and the champion
genome's stored selection-time runs in the source job (prompt tokens, latency, quality).
Observed = inputs bound to feedback (hash-verified) and the window's inference records.
Feedback inputs are schema-valid by construction, so schema violations are read ONLY from #30's
rejected-request records; a schema feature drifts when its rate exceeds
`schema_violation_rate` (default 0.05) over at least `min_samples` requests.

| feature | statistic | applies to |
| --- | --- | --- |
| `<field>.presence` | TVD of present/absent | optional input fields (schema drift) |
| `<field>.type` | share of ALL requests rejected with a type violation on the field | every input field (schema drift) |
| `<field>.missing` | share of ALL requests rejected for missing the field | required input fields (schema drift) |
| `input.unknown_fields` | share of ALL requests rejected for an unknown (new) field | every request (schema drift) |
| `<field>.distribution` | total-variation distance | booleans; strings/dates with ≤ `max_categories` distinct reference values |
| `<field>.distribution` | two-sample KS | integers, numbers, (other) dates as ordinals |
| `<field>.length` | two-sample KS of length | free text, string lists |
| `input.length` | two-sample KS of canonical input length | every request |
| `prompt_tokens` | two-sample KS | inference records vs the champion's runs |
| `quality` | selection-time mean quality − labelled production quality | needs `min_labelled` labels |

Each feature is `DRIFTED | STABLE | INSUFFICIENT_EVIDENCE` (fewer than `min_samples`
observations) `| UNAVAILABLE` (no reference). `material` = at least `material_features` data
features drifted.

## 5. Policy (`wynk-monitoring-policy/1`)

Content-addressed (`mp-<hash>`), immutable, persisted (`POST /api/v1/monitoring/policies`,
`GET .../{policy_id}`); every decision names the policy it applied. Defaults:

| | default |
| --- | --- |
| window | 7 days |
| drift: `min_samples`, `presence_delta`, `categorical_tvd`, `numeric_ks`, `length_ks`, `token_ks` | 30, 0.2, 0.25, 0.3, 0.3, 0.3 |
| drift: `max_categories`, `material_features`, `min_labelled`, `quality_max_drop` | 20, 1, 20, 0.15 |
| trigger: `min_inferences`, `min_labelled` | 30, 20 (schema floor: 2) |
| trigger: `min_quality`, `max_failure_rate` | 0.7, 0.2 |
| trigger: `on_material_drift`, `on_quality_drift` | true, true |
| trigger: `max_p95_latency_ratio`, `max_mean_cost_ratio` | off (vs the champion's selection runs; cost priced by the pinned registry entry) |
| trigger: `min_new_labelled_examples`, `cooldown_s` | 10, 7 days |
| `new_row_splits` (only when the parent splits have no seeded plan) | seed 0, 20 % validation, 20 % test |

## 6. Trigger decisions (`wynk-trigger-decision/1`)

`POST /api/v1/workflows/{version_id}/triggers/evaluate {since?, until?, policy_id?}`,
`GET /api/v1/workflows/{version_id}/triggers`, `GET /api/v1/triggers/{trigger_id}`

Rules (`checks` lists every one): `quality_below_threshold`, `failure_rate_above_threshold`,
`material_data_drift`, `quality_drift`, `latency_regression`, `cost_regression`. Each has a
minimum sample count; below it the rule is `INSUFFICIENT_EVIDENCE` and cannot fire, so one bad
request never triggers anything.

* No rule fired: **NOT_TRIGGERED**.
* A rule fired, but a TRIGGERED decision of this version is still inside the cooldown
  (`cooldown_open_trigger`) or fewer than `min_new_labelled_examples` unfrozen labels exist
  (`insufficient_new_labelled_examples`): **SUPPRESSED**.
* Otherwise **TRIGGERED**, and the eligible labels are frozen in the same transaction.

Decision `reasons` and `checks` are machine-readable (`code`, `observed`, `threshold`,
`samples`, `min_samples`); `evidence` holds the window, the exact inference / feedback ids, the
summary and drift report; `links` the champion, artifact, provenance, source job, contract and
model hash.

## 7. Re-optimization (`wynk-reoptimization/1`)

`GET /api/v1/triggers/{trigger_id}/challenger`, `POST .../reoptimize` (resume)

1. **Freeze.** The TRIGGERED decision's labels go to `frozen_examples` (`feedback_id` PRIMARY
   KEY: a label enters at most one new version). Labels arriving later are not in it.
2. **New dataset version.** Parent rows (exact typed values, file order) + one row per frozen
   label (inputs + expected; id column = feedback id). Excluded deterministically: a label whose
   inputs equal a parent row's (`duplicate_of_parent_row`: no copy of a test row can reach the
   optimizer), conflicting labels for the same inputs, duplicates. Uploaded and registered
   through `DatasetService` as version n+1 of the same dataset; identical bytes register as the
   identical version. Column types must not change (`dataset_schema_changed`).
3. **Splits.** Every parent row keeps its parent role (`EXPLICIT` splits); only the new rows are
   assigned, by the `seeded_hash/1` ordering and fractions of the parent's plan (or
   `new_row_splits`). Stored with the new version; the parent's splits are untouched.
4. **Challenger job.** `ExperimentJobDefinition` with the SAME TaskContract (objective,
   constraints, evaluator, workflow vocabulary) re-bound only to the new dataset, the SAME plan
   (model, expected model hash, budget, seeds, trials, batch size, LCB z, fixed rule) and always
   `strategies = fixed, random, aco`: the normal fair comparison, no strategy preferred.
   Created through `ExperimentJobs.create` (#24); it runs on the normal worker.
5. **Link.** `definition.provenance.reoptimization` names the trigger, policy, workflow version,
   lineage, champion, artifact, provenance, experiment, source job, contract hash, model hash and
   parent dataset version.

Nothing promotes or deploys it. A completed challenger is promoted only by an explicit
`POST /experiments/{job}/promote` (#25) and served only after an explicit #30 publish → stage →
promote. The challenger is promoted into the incumbent's OWN lineage (`dataset_id.task_id`):
its lineage identity (`wynk-lineage/1`) names the task semantics, evaluator, objective,
constraints, grammar, model and runtime, never the dataset version, splits or test rows (see
[champion_promotion.md](champion_promotion.md)). #25 opens the challenger's v(n+1) test split
once and re-evaluates the CURRENT incumbent on exactly those rows, trials and seeds; the
incumbent's old held-out score is never reused. Any change of task semantics still fails closed
(`incompatible_incumbent`).

## 8. Idempotency and concurrency

* **Trigger identity** `tr-<hash(workflow version, policy id, inference ids, feedback ids)>`:
  the evidence, not the window bounds or the caller. Decided once, under `BEGIN IMMEDIATE`,
  together with the history it was decided against; re-evaluating returns the stored decision.
* **One open trigger per version**: the cooldown check runs in the same write transaction, so
  concurrent monitors with different evidence cannot both trigger.
* **Fenced re-optimization**: a claim with owner, fence and lease; every write is conditional on
  (owner, fence); progress `CLAIMED → DATASET_CREATED → JOB_CREATED | FAILED` only moves forward
  (trigger). A live claim is never taken over; an expired one is.
* **Exactly one dataset version and job**: the dataset bytes are deterministic (content-
  addressed upload, identity-matched registration) and the challenger job id is
  `j-<hash(trigger id)>`; a monitor that crashed after creating the job is adopted, never
  duplicated.
* **Restart**: everything is in SQLite; summaries, drift reports, history and challenger views
  are reproduced byte for byte.

## 9. Leakage and non-mutation

Production labels enter only the NEW version, only after the decision froze them. Parent test
rows stay test rows, new test rows are test rows, and the challenger's optimizer view
(`DatasetSplits.optimizer_view`) holds no test row. Feedback never reaches pheromones, champion
state, the old experiment or deployments: monitoring writes only `monitoring.sqlite3`, new
dataset versions/splits, and one new job.

## Tests (`tests/test_production_monitoring.py`)

Telemetry tied to version + provenance; failures from records; feedback hash-bound, immutable,
never alters records; untrusted actor metadata decides nothing; aggregates never mix versions;
deterministic statistics; minimum-sample gates (quality, failure rate); drift against the
champion's dataset with test sealed; trigger creates a new version and challenger and mutates
nothing (dataset, splits, blob, job definition/identity/artifact/attempts/checkpoints,
champions, promotions, versions, deployments, inference records); parent roles preserved and
test isolated; same evidence never double-triggers; concurrent monitors → one decision, one
dataset version, one job; crash before / after job creation resumes without duplicates; fair
Fixed / Random / ACO challenger (even from a Fixed-only champion); no deployment, #25 still
required; restart reproduces state; API.

Mutation checks (each makes the suite fail): mixing versions in aggregates; writing the new
splits onto the parent version; re-splitting parent rows; bypassing sample minimums; deciding
the same evidence twice; a non-deterministic challenger job id.

## Limitations

* No authentication (#32) and no privacy controls (#33): feedback stores the inputs it is
  bound to; actor/source are untrusted.
* Input distribution drift sees inputs that arrive with feedback (labelled or not); token
  drift sees all traffic; schema drift sees every rejected request but only its value-free
  violations. Inference records keep only request hashes, by #30's design.
* **Delayed labels.** Feedback is one immutable record per inference (`UNIQUE(inference_id)`):
  once an UNLABELLED observation is stored for an inference, a later label for the same
  inference is `feedback_conflict` (409) and cannot be added. Submit feedback with its label, or
  wait until the label is known. Follow-up: an append-only label record that references the
  observation (never editing it) and is frozen into at most one dataset version.
* **Source experiment.** Monitoring tolerates exactly one absence: the champion's job is not in
  this process's job store (`source_experiment_not_in_this_store`; re-optimization answers
  `source_experiment_not_found`). A job that IS there but is not COMPLETED, not finalized,
  corrupt, hash-invalid or inconsistent with its provenance or the version's pins - or a
  registered dataset version that is not the pinned one - fails closed with
  `monitoring_integrity_error`; it is never reported as absent.
* No scheduler: triggers are evaluated on request (`POST .../triggers/evaluate`).
