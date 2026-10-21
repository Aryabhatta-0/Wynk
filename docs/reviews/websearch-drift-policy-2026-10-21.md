# Websearch provider drift policy (residual risk R3)

Decision: **HYBRID POLICY** (source-aware interrupt, before the run is recorded).
Branch `production/websearch-drift-policy`. JEV behavior is unchanged.

## 1. Lifecycle on main (before this change)

```
websearch_from_env (startup, frozen BraveSearchConfig; identity = endpoint + count + fixed params)
 -> registry_runtime.backend(definition): new WorkflowRunner, pin_websearch (identity, max_results)
 -> WorkflowRunner.evidence_authority(contract)  (websearch member only if the contract allows it)
 -> JobBinding.evidence_provider -> job_identity["evidence_provider"]   (compared exactly on rebind)
 -> RunVersions.evidence_source = the PINNED identity (runner._identities, never the live one)
 -> GatherExecutor: PinnedWebSearchSource.drifted() -> _fail(EXECUTOR_ERROR), 0 tool calls
 -> DurableEvaluate: attempt COMPLETED (a normal run with a failure)
 -> run_strategy.check(): evidence_source is the pinned one, so `unauthorized` passes
 -> entry score 0 -> ScoreBoard, val_runs, suite.rank, champion, optimizer.observe (pheromones)
 -> unit COMPLETED -> job COMPLETED -> canonical artifact (failure kind only, no cause)
 -> memory: refused (`live_websearch_evidence`, any websearch authority)
 -> promotion: re-binds (identity checked), held-out runs must report the champion's versions;
    with no incumbent a feasible challenger is PROMOTED
```

Rebinding (new worker, restart, resume, promotion) builds a NEW runner that pins the live
identity, so a changed provider there gives `INTERRUPTED/runtime_unavailable` with 0 requests.
Only a worker that already holds the job's binding sees the in-process refusal path.

## 2. Evidence (product path, `FakeBrave`, ACO + random, budget 8)

| case | main result |
|---|---|
| websearch-only, drift after search 9 | COMPLETED; every champion scores 0.0; ACO observed 0 for websearch genomes |
| websearch-only, drift after search 1, seeds 2 and 3 | COMPLETED; **champion is the drifted websearch genome** (14/15 runs refused, validation 0.0, wins on first-seen tie-break) |
| mixed fetch+websearch, drift after search 3 or 9 | COMPLETED; random champion flips websearch -> fetch; ACO trajectory and final state differ |
| drift, then the original identity returns | COMPLETED; refused rows inside the window are scored |
| drift before the worker binds | INTERRUPTED/runtime_unavailable, 0 requests |
| any drift case | the artifact never says why a row failed (`executor_error` only) |

So the same event (the provider changed after the job was created) gives INTERRUPTED or
COMPLETED depending only on whether the worker had already bound the job. The R3 statement
"the drifted candidate cannot win" was false.

## 3. Violated invariant

Every score that reaches `optimizer.observe`, the score board, selection, the champion or a
COMPLETED artifact is a measurement of the genome under the job's bound runtime identity.
The model authority already follows it: a MODEL_ERROR is never scored (retried, then
`ModelUnavailable` fails the unit); a run reporting another provider fails the unit. A drift
refusal sends nothing, so it measures nothing, but it was scored.

## 4. Policies

| | correctness | comparability | complexity | resume | cost | compat | mixed search |
|---|---|---|---|---|---|---|---|
| A current | safe, scores non-measurements | broken (time-of-drift dependent) | none | n/a (completes) | spends on a result that is not valid | — | fetch genomes win by default |
| B interrupt on any drift | valid | valid | low | resumable | stops even jobs that never search | interrupts jobs whose authority lacks websearch, or fetch-only remainders | over-broad |
| C source-aware interrupt (chosen) | valid | resumed job = undisturbed job | low (one preflight) | resumable, nothing ambiguous | non-search spend kept and reused | only drift-affected jobs change state | fetch/direct genomes keep running until a websearch run is due |
| D quarantine websearch genomes | valid rows, but skips proposals | broken: the run is not the plan its identity declares; RNG/checkpoint replay changes | high | fragile | low | changes optimizer semantics | search space silently shrinks |

## 5. Chosen behavior

1. Trigger: a new attempt of a genome with a websearch GATHER while the pinned provider reports
   another identity or result count (`WorkflowRunner.check_evidence`).
2. Layer: `DurableEvaluate`, before `start_attempt` (via `JobBinding.preflight`).
3. State: unit INTERRUPTED, reason `runtime_unavailable`; job INTERRUPTED. The cached binding
   is dropped.
4. Resume: `POST experiments/{id}/resume` (explicit, as for every `runtime_unavailable`).
5. Must match: the full `job_identity` on rebind (evidence authority included), and the pinned
   provider must not have drifted.
6. Settled candidates: kept; replayed from the log with no call.
7. Fetch/direct genomes: never refused; they run until a websearch run is due.
8. Optimizer state: nothing refused is observed; replay reproduces the stored states exactly.
9. Spend: committed spend stays committed; the open candidate's completed rows are reused, so
   no search or model call is sent twice.
10. Provenance: no refused row is stored; the unit's `detail` says the provider drifted. The
    resumed artifact equals an undisturbed job's.

## 6. JEV

JEV has the same two paths: rebinding under another JEV provider interrupts, an in-process JEV
drift is an EXECUTOR_ERROR row that is scored. This change leaves JEV as it is (the product
never runs JEV). One rule for every live provider would be: "a bound provider that no longer
reports its pinned identity interrupts the attempt that would use it, before it is written".
JEV needs a `drifted()` probe on `PinnedJevSource` to join it.

## 7. Residuals

* A drift that lands inside one run (after the preflight, or before a `regather`) is still
  refused by the executor and stored as a failed row (never sent, never charged).
* `PinnedWebSearchSource.search` checks `drifted()` and then the Brave adapter reads
  `self.config` again; a config swap between the two would send to the new endpoint. No
  production code swaps the config (it is frozen and built once per process). Pre-existing.
* A provider outage (HTTP error, timeout) is not drift: the request was sent and charged, and
  the row is scored as a live-evidence failure. Unchanged; a separate policy question.
