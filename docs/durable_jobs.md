# Durable experiment jobs

A fixed / random / ACO experiment (#23) runs as a durable job. It survives a process crash or a
browser disconnect, resumes exactly where it stopped, can be cancelled cleanly, and never pays
twice for a workflow run. Code: `experiments/jobs.py` (engine and service), `store/jobs.py`
(SQLite), and `api/product.py` (`/api/v1/experiments`).

Nothing scientific is reimplemented. A job is made of independent work units, one per
`(strategy, seed)`. Each unit is exactly one `run_strategy` call. When every unit has completed,
the job's artifact is the existing `assemble` of their records. The optimizers, the
`BudgetLedger`, run identities, selection and curves are the #23 code. The only addition to
`run_strategy` is an observe-only `checkpoint` hook.

```
create ── bind runtime, job_identity() ── persist definition + identity + units   (PENDING)
   │
worker: claim(unit) ── fenced lease ──► run_strategy(..., evaluate=DurableEvaluate,
   │                                                   checkpoint=Journal)
   │      DurableEvaluate: stored attempt? → its result (no model call)
   │                       else write-ahead STARTED → model → COMPLETED(result + usage)
   │      Journal:         re-executed checkpoint == stored one? (else fail closed)
   │                       else append it; stop here if the job is stopping
   ▼
finish_unit(record) ── all units COMPLETED → assemble(records) → artifact       (COMPLETED)
```

## State machine

The job state is derived from its units on every write, by one rule (`derive_job_state`):

```
PENDING ─► RUNNING ─► COMPLETED                  (artifact stored)
   │          ├─► CANCEL_REQUESTED ─► CANCELLED  (no unit running any more)
   │          ├─► FAILED                         (a unit failed closed, or assembly did)
   │          └─► INTERRUPTED ─► RUNNING         (resume)
   └─► CANCELLED
```

Units are `PENDING | RUNNING | COMPLETED | CANCELLED | FAILED | INTERRUPTED`, with a stable
`reason`:

| reason | meaning | resumed automatically? |
| --- | --- | --- |
| `process_lost` | the unit's worker died with nothing in flight | yes, by any worker |
| `ambiguous_attempt` | a model call started and its result was never stored | **no** |
| `runtime_unavailable` | this process could not bind the job, or its binding no longer reproduces the job's identity | no (explicit resume) |
| `resume_requested` | explicitly resumed | yes |
| `cancelled` / `job_failed` | stopped because the job was cancelled, or another unit failed | never (terminal) |
| `model_unavailable`, `experiment_error`, `reservation_overflow`, `pricing_error`, `runner_error`, `replay_diverged` | failed closed | never (terminal) |

`COMPLETED`, `CANCELLED` and `FAILED` are terminal. The store refuses any write that would leave
a terminal state. A cancelled or failed job is never assembled, so it is never reported
`COMPLETED`.

## What is persisted (`jobs.sqlite3`)

The store follows `store.datasets` conventions: WAL, `synchronous=FULL`, one `BEGIN IMMEDIATE`
transaction per write, and a connection per operation.

* **`experiment_jobs`.** The immutable `ExperimentJobDefinition` (TaskContract, DatasetSplits,
  ExperimentPlan, `synthetic`, workers, provenance) and the immutable bound identity: definition,
  contract, protocol and budget hashes, `problem_id`, `experiment_id`, evaluator version, pricing
  identity, model hash, and the `run_id` of every unit. Also the derived state, the cancellation
  request and the final artifact.
* **`experiment_units`.** State, reason, lease (`lease_owner`, `lease_until`, `fence`), and the
  completed strategy-run record.
* **`experiment_attempts`.** The write-ahead log of workflow runs. Each row holds a deterministic
  `attempt_id` = hash(job, unit run id, genome, row, trial, run seed, attempt n), the genome, and
  a state of `STARTED`, `COMPLETED` or `ERRORED`. A completed row also holds the `EvaluatedRun`,
  the measured and priced usage exactly as `BudgetLedger.measure` reports it, and timing.
* **`experiment_checkpoints`.** The unit's progress, in order (`run_strategy`'s
  `CHECKPOINT_KINDS`):
  * `proposed`: round, proposals, optimizer state
  * `reserved`: candidate index, round, genome, the admitted complete reservation, ledger totals
  * `settled`: candidate index, round, the full candidate record, the curve point, ledger
    totals, champion so far, optimizer state
  * `observed`: round, ledger totals, optimizer state after `observe`

Optimizer state is plain JSON (`checkpoint_state()` / `from_checkpoint()`):

* **MMAS ACO:** config, `epoch` (which also fixes the iteration-best vs. global-best phase), the
  lazily evaporated edges `(value, epoch set at)`, every proposed genome, the full score board
  (every observation, hence every LCB and the normalisation range), and the propose-call index
  that seeds the next ant. Current pheromones, the global best and the LCB range are recorded too
  and re-checked on restore.
* **Random:** the propose-call index (each proposal's rng is `derive_rng(name, seed, round,
  calls)`), the proposed set, and the exhausted flag. Proposal order is the candidate log.
* **Fixed:** whether its one workflow was proposed.

## Crash and recovery

Resume re-executes `run_strategy` from the start with the same plan, seed and optimizer.
`DurableEvaluate` answers every attempt that has a `COMPLETED` row from that row: no model call,
and the timing and usage are the stored ones. `Journal` compares every checkpoint the
re-execution produces with the stored one (clock-derived values aside). Optimizers, the ledger
and selection are pure functions of the results they are fed, so the re-execution passes through
exactly the stored states: proposals, pheromones and epoch, score board, rng call indices,
ledger. When the log is exhausted, the unit simply continues. Any disagreement fails the unit
closed with `replay_diverged` before a single new call. A new model call is only possible once
the whole stored log has been reproduced. A stored run whose result disagrees with its recorded
usage is refused the same way, so an edited log cannot buy budget.

On process start, `JobWorker.recover()` does two things:

1. Every `RUNNING` unit whose lease expired becomes `INTERRUPTED`. It is `ambiguous_attempt` if
   it has a `STARTED` attempt, otherwise `process_lost`.
2. Jobs whose units all completed but whose artifact was never stored get assembled.

Workers then claim `PENDING` and auto-resumable units. Leases are fenced: every claim increments
`fence`, and every unit write is conditional on (owner, fence) in the same transaction. A worker
that lost its unit therefore cannot start an attempt, append a checkpoint or finish the unit.
Two workers never execute the same unit. A heartbeat renews the lease during long model calls.

## No double spend

* A completed deterministic run is reused from the log. It is never sent again, and its usage is
  rebuilt into the ledger exactly once, because the re-executed ledger is the uninterrupted
  ledger, checked at every stored settlement.
* `start_attempt` is the write-ahead record. It commits before the model is invoked and is
  refused once the job is stopping or the lease is lost. Refused means nothing was written, so
  nothing may be called.
* An attempt with `STARTED` and no result has ambiguous spend. It is never re-sent. The unit
  stays `INTERRUPTED/ambiguous_attempt`, and `resume` answers `ambiguous_attempt` until every
  such attempt is resolved. It resolves either because the original worker stored its result
  late (allowed even after losing the lease, because it is a fact about money already spent), or
  because the runtime's `replay_proof` proves the result without a call, for example from a
  deterministic response cache. Nothing stronger than this is claimed. A crash between the
  provider charging and the commit is reported, never hidden.

## Budget continuity

The ledger is the #23 ledger, rebuilt from stored measured usage:

* Committed calls, tokens, execution time and cost stay spent.
* Settled reservations stay released.
* The unfinished candidate is re-reserved with the same complete reservation, using the same
  admission rule, and its completed rows count toward it.

Execution time is the runtime-measured `wall_time_s` of each stored run, so process downtime
never counts as execution time. For an unfinished unit, the job view reports `usage` (everything
measured as spent), `settled_usage` (the ledger at the last settlement) and `open_reservation`.

## Cancellation

`cancel` commits `CANCEL_REQUESTED` first. From then on `start_attempt` refuses every new run,
and the next checkpoint raises `JobStopped`. A result already returned is still stored. Units no
live worker owns are closed at once. The job becomes `CANCELLED` once no unit is running. The
unfinished candidate's reservation is released, and its completed runs remain as spent usage. A
cancelled job keeps every record, stays inspectable, and is never resumed or assembled.

## API (`/api/v1/experiments`)

| method | path | |
| --- | --- | --- |
| POST | `/experiments` | `{dataset_id, dataset_version, splits_hash, contract, plan, workers?}` → 201 job |
| GET | `/experiments` | job summaries |
| GET | `/experiments/{job_id}` | job + every unit's state, usage, open reservation, champion so far, learning curve, attempt counts |
| POST | `/experiments/{job_id}/cancel` | 409 `job_not_cancellable` once terminal |
| POST | `/experiments/{job_id}/resume` | 409 `job_not_resumable` / `ambiguous_attempt` |
| GET | `/experiments/{job_id}/artifact` | the `assemble` artifact; 409 `job_not_completed` otherwise |

The contract's dataset must be exactly the registered version, and the splits are the stored
ones. The server, not the client, decides `synthetic`. Experiments execute only when the server
has a model registry (`--model-registry` or `$WYNK_MODEL_REGISTRY`, API key from
`$WYNK_MODEL_API_KEY`). Without one, a create answers 503 `experiment_backend_unavailable`;
stored jobs stay inspectable and cancellable. A background worker in the server process recovers
and executes jobs.

UI status mapping, when the UI is wired: `PENDING`→`queued`;
`RUNNING`/`CANCEL_REQUESTED`/`INTERRUPTED`→`running` (with `reason`); `COMPLETED`→`completed`;
`FAILED`→`failed`; `CANCELLED`→`cancelled`.
