# Workflow memory priors (#28)

Production workflow memory reuses what earlier experiments learned, as a **hint** for the next
search. A prior never bypasses validation, held-out promotion (#25) or production gates: it only
chooses where the `aco_warm` search starts. Same budget, same ledger, same validation.

Code: `memory/priors.py` (domain), `memory/production.py` (`PriorService`),
`store/priors.py` (`SQLitePriorStore`). The legacy file-based demo (`memory/warm_start.py`,
keyed by `task_class`) is unchanged and never consulted by production experiments.

## Two levels

| Kind | Holds | Warm start |
| --- | --- | --- |
| `dataset_champion` | the validation-selected feasible workflow of one exact dataset context: genome + hash, contract rank key, VALIDATION stats | the MMAS fixed point of a colony converged on it: its path edges at `tau_max`, epoch `converged_epoch(config)` (every other edge at `tau_min`) |
| `task_family` | the final MMAS pheromone state of an ACO run: every explored edge, the ACO epoch | exactly that state |

Both start through `MMASACO.from_pheromones(state, config, epoch)`; MMAS itself is unchanged.
A pheromone map is used whole or not at all: maps are never merged or blended.

## Validity key (`wynk-prior-key/1`)

Every component is read from the #26 `ProvenanceRecord` of the source artifact (and re-derived
for the new job from its definition and binding). A prior is usable only on **exact** key
equality - never on loose names such as a task class.

| Component | Fields | Cold reason if different |
| --- | --- | --- |
| scope | `schema`, `kind`, `workspace_id` | `cold_provenance_invalid` |
| dataset (champion only) | `dataset_id`, `dataset_version`, `dataset_content_hash`, `dataset_identity_hash`, `splits_hash` | `cold_dataset_changed` |
| task contract | `task_id`, `task_semantics_hash` (the authoritative contract minus dataset / evaluation / workflow); champion also `task_contract_hash` | `cold_contract_changed` |
| grammar | `workflow_hash`, `grammar_version`, `grammar_hash` | `cold_grammar_changed` |
| evaluator | `evaluation_hash` (EvaluationSpec), `evaluator_version` (spec + fitness) | `cold_evaluator_changed` |
| model | `model_hash`, `model_config_hash` | `cold_model_changed` |
| runtime | `prompt_template_version`, `compiler_version`, `runtime_grammar_version`; champion also `benchmark_hash` | `cold_runtime_changed` |
| evidence provider | `evidence_provider` (the live JEV identity; absent when none) | `cold_evidence_source_changed` |
| optimizer | `optimizer`, `optimizer_version`, `aco_config_hash` (canonical MMAS config) | `cold_optimizer_changed` |

The runtime fields come from the RunVersions the runtime **declares** before any run
(`WorkflowBackend.versions` = `WorkflowRunner.versions`). A runtime that declares none never
warm-starts (`cold_runtime_undeclared`), and every warm run must then report exactly the pinned
versions or it fails.

The evidence provider is not part of the shared runtime identity: one search may run JEV and
fetch genomes, and a fetch genome reports no `evidence_source`. It is a key field of its own.

* A job's provider is the binding's `JobBinding.evidence_provider` (`WorkflowBackend` /
  `WorkflowRunner.jev.identity`): the pinned JEV identity, or none when the runtime has no JEV.
* A prior's provider is the one its learned state reflects (`learned_provider`): the
  `evidence_source` the source strategy run's candidates reported (its pheromones and its
  champion selection both depend on every candidate's outcome), plus the provider of the prior
  that run was warm-started from (recorded in the run protocol's `warm_start` summary).
* Exact equality: JEV A to JEV B, JEV to none, and none to JEV are all cold
  (`cold_evidence_source_changed`). None to none is unchanged.
* The field is absent while it is none, so every fetch-only key, key hash and `prior_id` stays
  what it was. A prior learned from JEV evidence before this field existed claims no provider;
  re-verification re-derives its key from the provenance (with the provider), so it is
  invalidated (`cold_provenance_invalid`) and never used.
* A warm pin records the provider (`WarmStart.evidence_provider`, absent when none). A worker
  whose runtime has another provider does not run the warm unit (interrupted,
  `runtime_unavailable`), `verify_pin` fails a pin whose prior learned from another provider,
  and a warm run that reports another `evidence_source` fails.
* Elite routing (#10/#54) ignores this key field: each elite is judged by its own runs'
  `evidence_source`, so a fetch elite of a mixed search stays usable.

## Warm-start policy

`PriorService.resolve` runs once, when an `aco_warm` job is created:

1. build both expected keys; 2. load the candidates of the same anchor (workspace + task, and
dataset for a champion); 3. verify each candidate (record hash, dataset liveness, the source
artifact's full #26 verification, eligibility, run identity, re-derived key, champion equality,
pheromones re-derived from the source unit's checkpoint log); 4. compare keys exactly;
5. warm-start on an exact match - dataset champion first, then task family; 6. otherwise a true
cold start with a structured reason.

The decision (`WarmStart`, `wynk-warm-start/1`) is pinned into `plan.warm_start`, hence into the
definition, experiment and run identities: resuming a job always reproduces it. A client can
never supply it. At execution the worker re-verifies the pin (`verify_pin`, no model call); a
prior invalidated since then fails the unit closed instead of silently starting another way.

Start reasons: `warm_dataset_champion`, `warm_task_family`, `cold_baseline` (plain `aco`),
`cold_no_prior`, `cold_dataset_changed`, `cold_contract_changed`, `cold_grammar_changed`,
`cold_evaluator_changed`, `cold_model_changed`, `cold_runtime_changed`,
`cold_evidence_source_changed`, `cold_optimizer_changed`, `cold_provenance_invalid`, `cold_dataset_deleted`,
`cold_heldout_overlap`, `cold_synthetic`, `cold_runtime_undeclared`, `cold_memory_unavailable`.

## Honest comparison

`aco` is always cold. Only the opt-in `aco_warm` strategy consults memory; no default plan has
it, so fixed / random / aco baselines are byte-identical with or without memory (a plan without
`aco_warm` serializes exactly as before #28). Every artifact has a `memory` section
(`wynk-memory-start/1`): each run's mode, reason, prior kind, prior id and state hash, and
`prior_loading_model_calls: 0`.

## Learning rules

After a job's canonical artifact is finalized, `PriorService.learn(job_id)` decides once per
job (recorded in `learned_jobs`, also on refusal). It learns only from a verified canonical
artifact of a real, durable job: never from synthetic runs or evaluators, frozen / migrated
fixtures, jobs without a registry entry, corrupted artifacts, failed / cancelled / incomplete
jobs, deleted datasets, or a run that executed test rows. The champion is the best feasible
validation-selected champion by contract rank; pheromones come from the best ACO run's last
checkpoint. Only optimization and validation rows are evidence; held-out rows never are.

## Held-out isolation

* Learning reads only the job store (search + validation evidence); never the promotion store.
* A champion's key includes `splits_hash`: it is reused only with the same validation and test
  rows.
* A prior records the (hashed) rows its evidence came from, transitively; a task-family prior
  whose evidence rows are test rows of the new job is refused (`cold_heldout_overlap`).

## Persistence and invalidation

`priors.sqlite3` per workspace (#32 `bind_sqlite`): content-addressed priors (`prior_id` =
hash of the record, no timestamps), WAL, `BEGIN IMMEDIATE` transactions, one active prior per
key (partial unique index). Triggers forbid deletes, content updates and un-retiring.

Retirement is deterministic and keeps history: a newer task-family prior supersedes the old
one (`superseded`, `superseded_by`); a champion replaces the head only with a strictly higher
rank, else it is stored retired as `dominated`. Invalidation (`dataset_deleted`,
`provenance_invalid`) also blocks jobs that already pinned the prior. Dataset deletion (#33) and
retention purges call `invalidate_deleted`, and liveness is checked again on every use.

## Per-task champions (#10)

A finished search also selects, for every VALIDATION row (task), the validated genome that the
contract ranks best on that row's own final validation runs (`experiments/task_champions.py`).
The order is total and never depends on execution order: feasible first, then the contract rank
key (`rank_measured`, the same authority as `rank_candidate`), then mean evaluator fitness, then
the genome hash (greatest wins). Optimization rows, optimizer scores and test rows are never
read. The suite champion is selected exactly as before; per-task champions are an extra layer.

* Artifact: every strategy-run record has `task_champions` (schema `wynk-task-champions/1`):
  `task_id`, `genome_hash`, `genome`, `feasible`, `rank_key`, `violations`, `validation`
  (`runs`, `pass_rate`, `fitness_mean`, `score_mean`), `compared_genomes`, `evidence_sources`,
  `run_versions`, `contract_hashes`, `run_ids` and `run_paths` (trace paths
  `candidate.<n>.runs.<j>`). `recompute_declared` re-derives the mapping from the stored
  validation-run entries; a mismatch fails closed. The learning-curves `run_search` result has
  the same field.
* An infeasible winner is recorded as diagnostic evidence only (`feasible: false`). It is never
  stored in a prior and never routed.
* Production memory: the `dataset_champion` prior's `Champion` gains `tasks` (feasible task
  champions keyed by `row_hash`, sorted) and `evidence_source` (the pinned JEV identity of the
  source run, or absent). Both are omitted when empty, so every #28 prior parses and keeps its
  `prior_id`. Task champions share the prior's validity: `PriorService.elite` finds the prior
  exactly like a warm start (same workspace, exact key, live dataset, verified provenance, no
  held-out overlap). It returns the row's task elite, else the dataset elite, else none. A
  champion learned with one JEV identity gives no elite to a runtime with another
  (`evidence_source_changed`). Task-family pheromones are unchanged.
* Chat: the elite ant is the routed template task's champion from the recorded ACO search when
  it is feasible and its contract hash and RunVersions equal what this runtime reports, else the
  suite champion, else none. The `ants` event marks it (`elite`: `task` or `suite`,
  `elite_task`).
* JEV is an evidence source only. Nothing asks JEV, or any model, which genome is best.
