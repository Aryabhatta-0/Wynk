# Champion promotion

A completed experiment (#23 run as a durable #24 job) does not change anything by itself. Champion
promotion is a separate, durable step after it:

```
experiment COMPLETED ─► ONE validation-selected challenger ─► held-out gate ─► PROMOTED | REJECTED
```

Code: `experiments/promotion.py` (engine and service), `store/champions.py` (SQLite), and
`api/product.py` (`/api/v1/experiments/{job_id}/promote`, `/promotions`, `/champions`). Nothing
scientific is reimplemented: candidates come from the #23 artifact, ranks from
`TaskContract.rank`, held-out evaluation from the experiment's own runtime binding, crash safety
from the #24 write-ahead attempt rule. Optimizers and evaluators are unchanged.

## Lifecycle

```
CANDIDATE ─► VALIDATED ─► CHAMPION
    │            │
    └────────────┴─► REJECTED
```

| status | meaning |
| --- | --- |
| `CANDIDATE` | a (strategy, seed) run's validation-selected workflow, feasible, outranked |
| `VALIDATED` | **the** challenger: the one candidate allowed to reach the test split |
| `CHAMPION` | the challenger passed the held-out gate and won compare-and-promote |
| `REJECTED` | infeasible on validation, or rejected by the held-out gate |

Only a `COMPLETED` job with its assembled artifact may enter promotion. `PENDING`, `RUNNING`,
`INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED` and `FAILED` jobs are refused
(`experiment_not_promotable`) before anything is recorded.

## 1. Validation-only challenger selection (`select_challenger`)

The candidates are the validation-selected champions of every (strategy, seed) run in the
artifact. Each run already chose its best under `TaskContract.rank`, so the best of them is the
best of every candidate the experiment evaluated. For each candidate, the validation rank is
**recomputed** from the job's stored validation runs (`experiment_attempts`, last attempt per
row × trial) with `rank_candidate` → `TaskContract.rank`. The recomputed rank must equal the
artifact's, or promotion fails closed (`promotion_evidence_mismatch`).

1. Hard constraints first. Infeasible candidates are `REJECTED` (`validation_infeasible`), and no
   score can lift them.
2. Feasible candidates are ranked by `CandidateRank.sort_key` (the contract's objective).
3. Ties are broken by test-free values only: higher mean validation score, then earlier first
   evaluation, then plan strategy order, then plan seed order.

Exactly one candidate becomes `VALIDATED`. It can come from fixed, random or ACO, and nothing
assumes search wins. With no feasible candidate there is no challenger. The decision is
`REJECTED` / `no_feasible_challenger`, and the test split is **never opened**.

The selection, each candidate's status and reason codes, and the ranking are pinned in the
promotion record.

## 2. Incumbent and lineages

A lineage has one current champion. The default lineage is `<dataset_id>.<task_id>`, and a caller
may name another. Every champion pins its **compatibility identity**: dataset id, version,
content hash and identity hash; splits hash and test row ids; TaskContract hash with objective and
constraints; evaluation hash and evaluator version; grammar version; model configuration and
exact `model_hash`; prompt template version; runtime run versions; `synthetic`; and the held-out
protocol. The plan (budget, seeds, strategies) is not part of it.

If the lineage's champion has a different compatibility identity, promotion **fails closed**
(`incompatible_incumbent`). Nothing is recorded and the test split stays closed. The two are never
compared. To start a new champion line, name a new lineage.

## 3. Test confidentiality

Before any test target is bound, `SQLiteChampionStore.open` durably records that the experiment's
test split is open (`test_opened_at`). The same record pins the selection and the incumbent (its
id and lineage version). `job_id` is `UNIQUE`, so an experiment's test split is opened at most once.

* Test rows never reach an optimizer, candidate generation, validation selection or
  tie-breaking. Those finished inside the experiment (whose evaluator never had test targets) or
  in step 1, before the test split was opened. `heldout_evaluator` binds test targets only after
  that.
* `SplitUse.PROMOTION_GATE` is a new split use allowed for the **test** role only. It means
  "decides the binary gate of one already-selected challenger". Selection (`SELECTION`) stays
  forbidden for test rows.
* Held-out runs are stored in `promotion_attempts`, never in the experiment. The job stays
  `COMPLETED` and unchanged.
* A second `promote` of the same job returns (or resumes) the same promotion. Naming another
  lineage is refused. There is no API to try another candidate. A rejected challenger stays
  rejected evidence.

## 4. Held-out gate (`gate`)

Only the challenger runs on the test rows. If there is an incumbent, it runs on **exactly the
same** rows, trial indices (`20000 + i`, for `plan.trials` trials) and execution seeds
(`heldout_seed(row, trial)`, independent of the workflow). If the challenger is the incumbent's
workflow, it runs once and the result is used for both (`challenger_is_incumbent`).

1. The challenger must satisfy every hard constraint on held-out, via `TaskContract.rank`
   (`heldout_constraint_violation`).
2. No incumbent: it may become the initial champion (`initial_champion`). The test split is still
   used and the constraints still apply.
3. With an incumbent: the challenger must not rank below it under `CandidateRank.sort_key`, the
   contract's own comparator. Maximize objectives promote at `>=`, minimize objectives at `<=`
   (`(-metric, quality)`), and balanced objectives compare their utility. Equal is not a regression
   (`no_regression`, `equal_to_incumbent`). Below is `regression`. There is no confidence score.

## 5. Compare-and-promote

`SQLiteChampionStore.decide` runs in one transaction. It re-reads the lineage version and refuses
with `StaleIncumbent` (writing nothing) unless the version is still the one pinned when the test
split opened. On success it appends the champion `(lineage, version + 1)`, advances the lineage
and records the decision. A promotion that loses this race is recorded as `REJECTED` with
`incumbent_superseded`; its gate result stays in the record as evidence. A newer champion is never
overwritten. Two racing promotions can never both win.

SQLite triggers make champion records, decided or failed promotions, the pinned
selection/incumbent/test-opening and finished held-out attempts immutable.

## 6. Crash safety (the #24 rule)

Each held-out run goes through a write-ahead attempt log. The attempt is `STARTED` before the
model is invoked and `COMPLETED` with its result and measured usage atomically after. The
evaluation holds a fenced lease, renewed by the #24 heartbeat.

| promotion state / reason | meaning | resumed by `promote(job_id)`? |
| --- | --- | --- |
| `OPEN` | held-out evaluation in progress (live lease) | no (`promotion_in_progress`) |
| `INTERRUPTED` / `process_lost` | its process died with nothing in flight | yes; completed runs are reused, never re-sent |
| `INTERRUPTED` / `runtime_unavailable` | the held-out runtime could not be bound | yes |
| `INTERRUPTED` / `ambiguous_attempt` | a held-out call started and its result was never stored | only after the runtime's `replay_proof` proves it; nothing is re-sent |
| `FAILED` / `model_unavailable`, `heldout_error` | the evaluation cannot finish | never (terminal: no decision, the test split stays spent) |
| `DECIDED` | `PROMOTED` or `REJECTED` | terminal |

`ChampionPromotions.recover()` (run by the product server at start) marks every `OPEN` promotion
whose lease expired as `INTERRUPTED`.

## 7. PromotionDecision

Every decision is an immutable JSON record (`wynk-promotion-decision/1`). It contains:

* the experiment: job id, experiment id, definition hash, problem id, `synthetic`;
* the challenger's identity (strategy, seed, run id, optimizer and version, genome) and its final
  status;
* the pinned incumbent (champion id, version, genome, provenance), if any;
* validation evidence: the whole selection with every candidate's recomputed rank, validation
  summary and attempt ids;
* held-out evidence for the challenger and incumbent: measurements, rank key, feasibility,
  violations and attempt ids, under the held-out protocol;
* the constraint results, the objective comparison (comparator, both keys, relation) and the
  `gate` outcome from evidence alone;
* the final `decision`, machine-readable `reason_codes`, and the exact identities (evaluator,
  model, prompt, grammar, run versions, held-out protocol, selection rule);
* a `decision_hash`.

`ChampionPromotions.verify(promotion_id)` (`GET /api/v1/promotions/{id}/verify`) re-derives the
decision from stored evidence alone: validation runs in the job store, held-out runs in
`promotion_attempts`, and the contract. It fails closed unless the result reproduces the recorded
decision and champion record exactly.

## API

```
POST /api/v1/experiments/{job_id}/promote?lineage=...   promote (or return the promotion); 201/200
GET  /api/v1/experiments/{job_id}/promotion
GET  /api/v1/promotions?lineage=...
GET  /api/v1/promotions/{promotion_id}
GET  /api/v1/promotions/{promotion_id}/verify
GET  /api/v1/champions/{lineage}                         current champion
GET  /api/v1/champions/{lineage}/history                 every champion, oldest first
```

The promote request runs the held-out evaluation before it answers. Error codes:
`experiment_not_promotable`, `incompatible_incumbent`, `promotion_in_progress`,
`promotion_ambiguous_attempt`, `promotion_failed` (409); `promotion_not_found`,
`champion_not_found` (404); `heldout_backend_unavailable` (503);
`promotion_evidence_mismatch` (500).

Out of scope: serving or deploying a champion (#30) and monitoring it (#31).
