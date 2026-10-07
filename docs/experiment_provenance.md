# Experiment provenance and reproducible artifacts

Every experiment has one canonical artifact, and every number Wynk reports can be traced to an
immutable field in it:

```
dataset ─► experiment ─► strategy runs ─► selected challenger ─► promotion evidence
   └──────────────── one ProvenanceRecord, one artifact_id ────────────────┘
```

Code: `core/provenance.py` (the record), `experiments/provenance.py` (build, verify, trace,
declared-metric checks, frozen-artifact migration), `experiments/artifacts.py` (service, evidence
replay, CLI), `store/jobs.py` (`experiment_artifacts`), and `api/product.py`.

There is no new result format. The #23 result body (`wynk-optimization-experiment/2`, produced
by `assemble`) is unchanged and stored once. The canonical artifact adds a small envelope next to
it.

## The canonical artifact (`wynk-experiment-artifact/1`)

```json
{
  "schema": "wynk-experiment-artifact/1",
  "experiment_id": "…",
  "parent": {"kind": "job", "job_id": "j-…", "definition_hash": "…"},
  "provenance_id": "…",
  "provenance": { "schema": "wynk-provenance/1", "…": "…" },
  "experiment": {
    "schema": "wynk-optimization-experiment/2",
    "sha256": "sha256 of the body's canonical JSON",
    "bytes": 123456,
    "scientific_sha256": "hash of the body's scientific fields only",
    "gzip_sha256": null
  },
  "field_classes": "wynk-field-classes/1"
}
```

* `artifact_id` is the canonical hash of the envelope. Jobs, promotion decisions and traces refer
  to this identity.
* `experiment.sha256` links the envelope to the exact stored body. Neither can change without
  the other noticing, and the body is never duplicated.
* `scientific_sha256` is equal for two runs that computed the same science, whatever their
  clocks measured.
* `gzip_sha256` is set for frozen `write_compact` directories, whose `.gz` is the stored form.
  The canonical JSON digest is the authority, because gzip bytes can differ across zlib builds.
* There are no timestamps in the envelope. The same job always finalizes to the same
  `artifact_id`, which is what makes finalization idempotent.

## ProvenanceRecord (`wynk-provenance/1`)

Every component is copied from its existing authority. None is re-derived by hand.

| component | authority | fields |
| --- | --- | --- |
| `dataset` | `TaskContract.dataset` (`DatasetSpec`) | id, version, content hash, identity hash, manifest hash |
| `splits` | `DatasetSplits` | splits hash, dataset hash, method, plan (config), rows per role |
| `contract` | `TaskContract` | the full contract, contract hash, objective/constraints hashes |
| `grammar` | `workflow_grammar(contract)` | version, stage vocabulary, grammar hash |
| `evaluator` | `EvaluationSpec` + the bound evaluator | kind, spec version, spec hash, config, run version |
| `model` | `ModelRegistry` entry pinned by `expected_model_hash` | configuration, config hash, exact `model_hash`, registry entry + hash, prompt version |
| `budget` | `ExperimentBudget`, `PricingPolicy`, `ExperimentPlan.protocol()` | budget + hash, pricing identity, protocol + hash, protocol id, fixed rule, trials, batch size |
| `runs[]` | `run_identity` + `make_strategy` | strategy, seed, `run_id`, optimizer name/version/config, candidate count, candidate-order hash, selected workflow |
| `versions` | runner, ledger, job and identity schemas | runner/ledger versions, artifact + job schema, every `RunVersions` |

`build_provenance` builds the record from the job's authorities. It then checks the record
(`check_provenance`) against the identity the body recorded while it ran: the problem identity,
the plan, and every strategy-run identity. It also checks the record against the job's bound
identity. These are some of the checks:

* `problem_id` and `experiment_id` re-hash.
* Every `run_id` re-hashes.
* The contract re-validates and re-hashes.
* The splits were made for the contract's dataset.
* The registry entry pins the plan's `model_hash` and describes `plan.model`.
* Every run reported that model and those prompts.

Any disagreement raises `ProvenanceMismatch`, and nothing is repaired. A real (non-synthetic)
experiment must pin its registry entry.

## Field classes

Every leaf of a result body is exactly one of three classes:

| class | what | reproduced by |
| --- | --- | --- |
| `scientific` | scores, verdicts, candidate order, selected workflows, tokens, calls, cost, stop reasons, curves, declared budgets/limits | evidence replay **and** re-execution on a deterministic backend |
| `measured_telemetry` | runtime-measured workflow wall time (`latency_s`, `wall_time_s`, `execution_s`, …) | evidence replay only (never re-measured equal) |
| `clock_telemetry` | process clocks: end-to-end time, overheads, throughput (`CLOCK_KEYS`) | never compared |

Deterministic equality means equality of the scientific projection only. No wall-clock latency
is ever promised to reproduce byte for byte.

## Traceability: "where did this number come from?"

A field path names any stored number. Strategy runs are addressed by key, not by list position,
and candidates and curve points by their evaluation number:

```
strategy.<strategy>.seed.<seed>[.field…]       one strategy run
  ….candidate.<evaluation>[.runs.<i>.field]    one candidate evaluation / one workflow run
  ….curve.<evaluation>.<field>                 one learning-curve point
summary.by_strategy.<s>.metrics.<metric>.<stat>
summary.by_strategy.<s>.per_seed.<seed>.<metric>
summary.by_strategy.<s>.best_so_far_curve.<evaluation>.best_so_far_score.<stat>
provenance.<component>.<field>                 the ProvenanceRecord
promotion.<field>                              the decided PromotionDecision (API only)
```

`GET /api/v1/experiments/{job_id}/trace?path=strategy.aco.seed.1.usage.tokens` answers:

```json
{
  "path": "strategy.aco.seed.1.usage.tokens",
  "artifact_id": "…", "experiment_sha256": "…", "provenance_id": "…",
  "document": "experiment",
  "pointer": "/runs/5/usage/tokens",
  "value": 6000,
  "field_class": "scientific",
  "run": {"strategy": "aco", "seed": 1, "run_id": "…"},
  "derivation": {
    "rule": "sum/1",
    "sources": ["strategy.aco.seed.1.candidate.1.usage.tokens", "…"],
    "recomputed": 6000,
    "consistent": true
  }
}
```

* A derived number (summary statistics, run and candidate usage, cumulative curve points,
  best-so-far scores and selected workflows) carries its `derivation`. Each source is itself a
  traceable stored field, and `consistent` says whether the recomputation matches.
* A field of one workflow run carries its `evidence`: genome, row, trial, run seed, and the
  write-ahead `attempt_id` of the #24 job that measured it.
* Every number of a completed unit in `GET /experiments/{job_id}` has a `sources` entry naming
  its field path.
* `promotion.challenger.*` links to the challenger's strategy run
  (`strategy.<s>.seed.<n>.champion`) and checks that the two agree.

`recompute_declared` re-derives every declared aggregate from the per-run evidence the body
carries: the summary from the runs, each run's usage and curve from its candidates, each
candidate's usage and validation score from its workflow runs, and every selected workflow from
its curve.

## Integrity: verify on every read, fail closed

`load_canonical` runs on every artifact, provenance, trace, verify, reproduce and promotion read.
It checks all of the following:

* The envelope schema is known.
* The envelope hashes to `artifact_id`.
* The body is in canonical form and matches `experiment.sha256` and `scientific_sha256`.
* The provenance re-hashes to `provenance_id`.
* The provenance agrees with the body, the job's bound identity and its immutable definition
  (contract, splits, plan, `synthetic`).

Any failure raises `artifact_integrity_error` (HTTP 500), and nothing is repaired. A forger who
rewrites the body and recomputes every hash still fails twice. First, `verify` finds declared
metrics that no longer follow from the runs. Second, `reproduce` finds stored attempts that no
longer replay to that body.

In SQLite, triggers make `experiment_artifacts` rows immutable (no `UPDATE` or `DELETE`) and the
job's result body write-once.

## Durable finalization (#24)

When the last unit completes, the worker assembles the body exactly as before. It builds the
envelope and calls `put_artifact(job_id, body, canonical)`, which writes the body and the
envelope in **one transaction**:

| situation | result |
| --- | --- |
| crash before the transaction | neither is stored; `JobWorker.recover` finalizes on restart |
| crash after the commit | both are stored; recovery and retries are no-ops |
| finalizing again with the same envelope | no-op (`False`) |
| a different envelope for the same job | `IntegrityViolation`: a job never has two artifacts |
| an envelope that does not link to the body | `IntegrityViolation` |

Jobs completed before this change have a body and no envelope. They are **explicitly migrated**:
`JobWorker.recover` re-assembles them from their unit records. It finalizes the envelope only if
the re-assembly reproduces the stored body byte for byte. Until then they answer
`artifact_not_finalized` (409) and cannot be promoted.

## Promotion (#25) cites the artifact

Promotion pins `art.ref()` (`artifact_id`, `provenance_id`, `experiment_sha256`,
`scientific_sha256`) when it opens the test split. Its decision (`wynk-promotion-decision/2`)
carries that reference as `artifact`. The champion record's `provenance` carries `artifact_id`
and `provenance_id`.

The compatibility identity is read from the ProvenanceRecord, not copied from the body. It is
byte-identical to the previous derivation, so existing lineages stay comparable. Continuing or
verifying a promotion fails closed (`promotion_evidence_mismatch`) unless the experiment's
artifact is exactly the pinned one. Decisions recorded before this change
(`wynk-promotion-decision/1`) still verify exactly as they were written.

## Reproduction

`GET /api/v1/experiments/{job_id}/reproduce` (or `python -m experiments.artifacts reproduce`)
rebinds the experiment: the same contract, splits and evaluator, and the same model registry
entry, which must be the one in the provenance. The runtime must reproduce the job's identity
exactly. The unchanged #23 `run_strategy` then re-runs every (strategy, seed) with the same plan,
seed and optimizer configuration. Its evaluator answers **only from the job's stored write-ahead
attempts**, so no model is called.

Optimizers, the ledger and selection are pure functions of the results they are fed. The replay
must therefore propose the same candidates in the same order and select the same workflows. The
re-assembled body must match the stored one:

* scientific fields: equal
* measured telemetry: equal (replayed from the stored runs)
* clock telemetry: counted, never compared

A missing or edited stored run fails with `reproduction_mismatch` (500).

A hosted model may be nondeterministic, so its outputs are never regenerated and "hoped equal".
The stored outputs are the evidence, and reproduction proves that the declared results follow
from them. For a backend known to be deterministic, `replay(..., reexecute=evaluate)` re-executes
the runs live and compares the scientific fields only.

Frozen MuSiQue and MMLU-Pro results (`experiments/results/*/protocol-*`) have no job and no
attempt log. `load_frozen` verifies both recorded digests and migrates them explicitly: the
provenance is the identity they recorded, marked `migration`, with the authorities that survive
only as hashes listed in `missing`. It then re-derives their declared metrics from the per-run
evidence they carry. The migration is deterministic, and the tests pin each `artifact_id`.

## API

| method | path | |
| --- | --- | --- |
| GET | `/experiments/{job_id}/artifact` | the result body plus `artifact_id`, `provenance_id`, `experiment_sha256`, `artifact_schema` |
| GET | `/experiments/{job_id}/provenance` | the ProvenanceRecord (+ `promotion_id`) |
| GET | `/experiments/{job_id}/trace?path=…` | where a number comes from |
| GET | `/experiments/{job_id}/verify` | integrity, identities, declared metrics, promotion citation |
| GET | `/experiments/{job_id}/reproduce` | replay from stored evidence (needs the experiment runtime) |

Error codes: `job_not_completed`, `artifact_not_finalized` (409); `trace_path_not_found` (404);
`invalid_request` (400, malformed path); `experiment_backend_unavailable` (503, reproduce without
a runtime); `artifact_integrity_error`, `reproduction_mismatch` (500).

```bash
python -m experiments.artifacts verify --data-dir .wynk-data --job j-…
python -m experiments.artifacts trace --data-dir .wynk-data --job j-… --path strategy.aco.seed.1.usage.tokens
python -m experiments.artifacts reproduce --data-dir .wynk-data --job j-… --model-registry registry.json
python -m experiments.artifacts verify --frozen experiments/results/musique/protocol-v2
```

Out of scope: deploying a champion (#30).
