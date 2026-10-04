# Code review — `main` @ `1272326` (2026-10-04)

**Scope:** `origin/main` @ `1272326` ("Merge pull request #4 … integrate/mvp-demo"), reviewed in a detached worktree.
Local `main` is stale (it only has `1d5951f` init) — run `git pull` / `git branch -f main origin/main`.

**Reviewers**
- Codex CLI 0.149.0, model **`gpt-5.6-sol`, reasoning effort high**, read-only sandbox.
  `gpt-6.1-sol` (requested) was rejected by the API: *"not supported when using Codex with a ChatGPT account"*;
  `gpt-5.6-sol` is the newest Sol model in this account's Codex catalog.
- Claude (Opus 5.5): independent pass + verification of Codex's High findings against the source.

**Baseline health on main:** `pytest` 357 passed · `ruff check` clean · `ruff format --check` clean.

Source tag: **[C]** = Codex, **[CL]** = Claude, **[C+CL]** = found independently by both.
Severity reflects impact on the project's purpose (a valid ACO-vs-random comparison), not just crash risk.

---

## Summary

No crash-level bugs, and the codebase is clean and well-tested. But **the committed ACO-vs-random
results on `main` should not be treated as valid evidence yet**:

1. A run that failed can still score **PASS** (evaluator ignores non-budget failures when an answer exists).
2. Fitness includes **wall-clock time**, which is infrastructure noise; and **HTTP retry backoff counts toward the
   wall-time cap**, so rate limits turn into INFEASIBLE (−1) verdicts that are **cached permanently**.
3. The search space includes operators the runtime **always rejects** (`jev`, `self_consistency`), and by
   `docs/HANDOFF.md`'s own account avoiding them is "a large part of" ACO's Class A lead.
4. There is **no held-out test split**: the "best" seed is picked on validation and the "final test" is a validation task.
5. `main` is **missing the fixes from PR #5** (`b72db1d`), which were merged into `integrate/mvp-demo`
   *after* that branch was merged into `main`.

---

## High

### H1. Failed runs can be scored PASS — [C], verified
`evaluation/gate.py:67-95`, `runtime/stage_runner.py:143-147`
`DeterministicEvaluator.evaluate` only short-circuits on `BUDGET_EXCEEDED`. `StageRunner.result()` returns
`_last_answer` even when a later stage failed. So `SYNTHESIZE` → correct answer → `VERIFY(self_consistency)`
(always `EXECUTOR_ERROR`) produces `failure=EXECUTOR_ERROR` **plus** an answer, which grades as PASS (Codex reproduced this).
The same applies to a `MODEL_ERROR` in a trailing stage.
**Fix:** any non-null `failure` other than budget ⇒ `FAIL`, or drop the answer on terminal failure; add a failure+answer regression test.

### H2. Wall-clock time in the fitness breaks determinism and the `--workers` claim — [C+CL], verified
`evaluation/fitness.py:43-48`, `runtime/executors/base.py:95-96`, `experiments/learning_curves.py:17-19`, `experiments/run_mvp.py` (`--workers` help: "results are identical to 1")
PASS fitness = `1 + 0.1/(1+tokens/5000) + 0.1/(1+wall_time/30)`. Wall time is measured, so it depends on endpoint
load and on how many runs execute concurrently. Changing `--workers` therefore changes fitness, LCB ranking and pheromone
deposits for the same seed. The "identical to sequential" claim only holds for the synthetic zero-time evaluator used in tests.
**Fix:** already done on `integrate/mvp-demo` (`b72db1d`: wall time removed, headroom vs `caps.tokens`) — merge it (see H6).

### H3. Retry backoff counts toward the wall-time cap → spurious INFEASIBLE, cached forever — [CL]
`runtime/gemma_client.py:159-173`, `runtime/executors/base.py:95-96`, `experiments/real_runtime.py:323-331`
`_post` sleeps up to 60 s per retry (with `Retry-After` honoured) **inside** the measured stage time. With the 180 s
wall caps and `--workers 4` hitting a rate-limited gateway, a couple of 429s push a run over the cap → `BUDGET_EXCEEDED` →
INFEASIBLE (fitness −1). `RunCache.put` only skips `MODEL_ERROR`, so this infra-caused verdict is persisted to
`run_cache.jsonl` and reused on every rerun.
Related [C]: HTTP retries are also invisible to `BudgetUsage.retries` / `model_calls` (`gemma_client.py:162`),
contradicting the "every retry is charged" contract.
**Fix:** exclude backoff time from measured wall time (or measure only time-in-request), surface attempt counts from the
client, and don't cache `BUDGET_EXCEEDED` results whose breach is wall-time-only.

### H4. Search space contains operators the runtime always rejects — [C], verified
`runtime/executors/gather.py:52-53` (`jev` → "not available (MVP)"), `runtime/executors/verify.py:107-112` (`self_consistency`)
The grammar offers them to both optimizers. Per `docs/HANDOFF.md:83-86`, ~350/500 of random's Class-A train runs
vs ~200 of ACO's hit these, and "ACO's pheromone steers away from them, which is a large part of its Class A lead".
That measures "learns to avoid unimplemented options", not workflow quality.
**Fix:** exclude unimplemented options via a shared runtime-capability constraint in `ConstraintChecker` (both optimizers),
then rerun the comparison.

### H5. No held-out test set; selection and "final test" both use validation — [C+CL], verified
`experiments/run_mvp.py:186-189`, `experiments/run_mvp.py:215-229`, `benchmarks/splits.json`
The best ACO seed is chosen by **max validation fitness** across seeds (winner's curse), and `final` defaults to
`A-001`, a **validation** task, written out as `final_test.json`. Splits are also very small (5 train / 3 validation per class),
so one lucky task flips the headline numbers.
**Fix:** add a frozen `test` split that's only touched after all genome/seed/config selection; report all seeds'
validation numbers rather than the max; consider more tasks per class.

### H6. `main` is missing PR #5 fixes — [CL]
`git rev-list origin/main...integrate/mvp-demo` → main is 2 commits behind. PR #5 (`b72db1d`) targeted
`integrate/mvp-demo` after PR #4 had already merged that branch into `main`. Missing on main:
cap-relative token fitness (fixed 5000-token scale scores 8k tokens the same on a 9k vs 90k cap), wall-time removal (H2),
ACO batch de-duplication (a repeat burns an LLM evaluation and adds a non-independent sample that narrows the LCB),
`rho` 0.15 → 0.30, corrected FAIL-band docstring (`[0, 0.7]`, not `[0, 0.8]`).
Note `integrate/mvp-demo` also *removes* `memory/` (warm start, PR #3), so a straight merge needs a decision on that.
**Fix:** open a PR `integrate/mvp-demo → main` (or cherry-pick `b72db1d`), resolving the `memory/` delta deliberately.

---

## Medium

| # | Where | Issue | Src |
|---|---|---|---|
| M1 | `runtime/executors/gemma_stages.py:78-79`, `evaluation/gate.py:77-80` | Transient `MODEL_ERROR` (backend down after retries) becomes `FAIL` with fitness 0.0 and is fed to the ScoreBoard/pheromone — infra failures penalise the genome. It isn't cached, but it's already been observed by the optimizer. Exclude/re-run instead of scoring. | CL |
| M2 | `core/cost_model.py:67`, `core/constraints.py:133` | Uncalibrated placeholder costs are used as *proven lower bounds* for hard rejection; e.g. `GATHER:api` charged 2 tool calls though Class-B snapshots expose one endpoint, so feasible genomes can be pruned. | C |
| M3 | `benchmarks/loader.py:47` | `benchmark_hash(bench_dir=…)` reads tasks/splits from `bench_dir` but hashes snapshots from the default store → modified snapshots can keep the old identity and reuse stale cached runs. | C |
| M4 | `runtime/gemma_client.py:123`, `experiments/real_runtime.py` | Run identity hashes only `model@revision`; endpoint, `GEMMA_STRUCTURED` and empty revision aren't in the key → switching providers/modes silently reuses incompatible cached runs. | C |
| M5 | `memory/models.py:24`, `memory/warm_start.py:88` | Warm-start compatibility key omits prompt-template/compiler versions and ACO params (`rho`, tau bounds) → stale pheromones reused. | C |
| M6 | `evaluation/gate.py:89` | Optional answer field (`required=False`) omitted → passes schema validation, then `answer.values[name]` raises `KeyError`. | C |
| M7 | `experiments/learning_curves.py:72-75` | `ExperimentConfig.lcb_z` drives incumbent selection, but `MMASACO()` is constructed with default `lcb_z=1.0` → reported config ≠ what ACO used. | C, verified |
| M8 | `experiments/learning_curves.py:48-61` | `make_evaluate_fn` looks up ground truth by `task.id` only; a same-ID task with changed question/caps/snapshot is graded against the original truth. Also keeps `TaskSpec` outside the evaluator boundary. | C |
| M9 | `store/runs.py:54` | `save_run()` silently drops a non-identical result with the same `run_id` (e.g. re-evaluation under a new evaluator version keeps the stale verdict). | C |
| M10 | `runtime/executors/gather.py:60` | One `SourceError` among parallel reads returns zero usage, erasing the tool calls that did happen from caps/metrics. | C |
| M11 | `experiments/run_mvp.py:101` | `EvalLog.best_so_far_fitness` is the max *single run*, not the genome LCB used for selection → logs look better than the incumbent. | C |
| M12 | `experiments/run_mvp.py:126` | `evaluations.jsonl` is always appended with no run/config ID while `results.json` is overwritten → reruns silently mix data. | C |
| M13 | `runtime/runner.py:33`, `runtime/maf_nodes.py:17` | `runtime.runner` eagerly imports optional `agent_framework`, so the documented base `.[dev]` install can't import the runner/real runtime. | C |

## Low

| # | Where | Issue | Src |
|---|---|---|---|
| L1 | `runtime/sources.py:46,72` | `DirectorySnapshotSource`/`DirectoryApiSource` join `snapshot_id` onto root without containment checks (`..`, absolute/UNC paths). Codex rated High; downgraded because snapshot IDs currently come only from the in-repo frozen benchmark — becomes real if tasks are ever user-supplied. | C (sev. adjusted by CL) |
| L2 | `experiments/real_runtime.py:314-317` | `RunCache` load fails entirely on a truncated last JSONL line (e.g. killed mid-write). Skip/repair the tail line. | CL |
| L3 | `experiments/run_mvp.py:247` | `final --task` defaults to `A-001` regardless of the class the saved genome was optimised for; a Class-B genome on a Class-A task can be inadmissible. | CL |
| L4 | `optimizers/aco_mmas.py:42` | `global_best_period` not validated; `0` → `ZeroDivisionError` in `observe`. | C |
| L5 | `evaluation/evidence.py:7` | `evaluation` imports `benchmarks`, violating the `evaluation → core` dependency map in `ARCHITECTURE_CONTRACTS.md`; the authority test doesn't cover this direction. | C |
| L6 | `.gitignore` vs tree | `experiments/results/` is ignored but 20 result files are tracked; intentional? If so, un-ignore the subpath to avoid confusion. | CL |

## Aside — current branch (`integrate/mvp-demo`), not main
`tests/test_authority_boundaries.py:59` `rglob`s `experiments/` and trips over the untracked
`experiments/oss_baselines/.venvs/` (882 MB, includes a Big5-encoded joblib test file) → 1 failure / 335 passed.
Not a code bug: move the venvs outside the repo or have the test iterate `git ls-files` / skip dot-dirs.

## Suggested order of work
1. Merge PR #5 into main (H6) — fixes H2 and the duplicate-sample issue.
2. H1 (failure ⇒ FAIL) and H3/M1 (infra failures must not become genome verdicts, and must not be cached).
3. H4 (prune unimplemented operators) and H5 (add a test split), then **rerun** the ACO-vs-random experiment; current
   `experiments/results/real-gemma4/*` numbers are confounded by H1–H5.
4. Mediums, starting with M4/M3/M5 (cache/identity correctness) and M7.

---
*Raw Codex output preserved alongside this file: `main-review-2026-10-04.codex-raw.md`.*
