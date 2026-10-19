# Benchmark and full-suite protocol (#13)

This document covers the frozen benchmark, how it is split, and the one canonical protocol that
runs ACO and random search across all of it.

## The benchmark

The benchmark has 100 frozen tasks, 50 Class A and 50 Class B:

* `benchmarks/` holds the 60 search tasks.
* `benchmarks/heldout/` holds the 40 held-out tasks.

Every task has a `TaskSpec` in `tasks.json` and frozen fixtures under `snapshots/<id>/`
(`pages/*.txt`, plus `records.json` for Class B). It also has a catalog entry in `catalog.json`
giving its suite and shape.

All tasks come from deterministic generators:

* `benchmarks/build.py` and `benchmarks/heldout.py` produce the 16 + 16 MVP tasks, which are
  byte-identical to before #13.
* `benchmarks/expansion.py` produces every new task.

Regenerating gives identical bytes, and tests pin the hashes.

| Suite | Class | Search | Held-out | Pages | Fields | Caps (tokens / tool calls) | Sources |
|---|---|---|---|---|---|---|---|
| `A` | A | 16 | 12 | 3-4 | 2-3 | 9000 / 12 | fetch, jev |
| `Along` | A | 8 | 4 | 12 | 3-5 | 16000 / 20 | fetch, jev |
| `Ajev` | A | 6 | 4 | 18 | 2-3 | 12000 / 10 | **jev only** (`interaction_required`) |
| `B` | B | 16 | 12 | 1 endpoint | 1-2 | 7000 / 10 | fetch, api |
| `Bjoin` | B | 14 | 8 | 4 endpoints | 1-2 | 14000 / 12 | fetch, api |

A **suite** is a set of tasks that share one search policy (workflow constraints, objective,
evaluator), so a single search can cover the whole suite. The suite choices:

* The original MVP tasks keep their suites (`A`, `B`), so the old search policy is unchanged.
* `Along`, `Ajev` and `Bjoin` exist because they need larger caps or different sources.

### Shapes (reporting only)

| Suite | Search | Held-out |
|---|---|---|
| `A` | single_page_lookup 7, multi_page 5, distractor 4 | 6, 4, 2 |
| `Along` | cross_document_chain 6, wide_schema 2 | 3, 1 |
| `Ajev` | interactive_navigation 6 | 4 |
| `B` | filter_aggregate 6, filter_list 2, top_k_sort 3, date_aggregate 3, group_by 2 | 5, 1, 3, 2, 1 |
| `Bjoin` | join_aggregate 4, dependency_chain 2, anti_join 1, multi_endpoint_join 1, top_k_sort 1, date_aggregate 3, group_by 2 | 2, 1, 1, 1, 1, 1, 1 |

How each new suite is built:

* **`Along` dossiers** have 12 pages: overview, leadership, financials, segments, governance,
  acquisition, a profile of the acquired company, industry peers, press, and three subsidiary
  pages.
  - `cross_document_chain`: one answer leads to the next page. For example, the oldest
    subsidiary → its page for city and year; the acquired company → its profile for its original
    HQ; the CEO's previous employer → the peers page for its HQ.
  - `wide_schema`: the question asks for 4-5 fields spread over several pages.
* **`Ajev` investor portals** have 18 pages under a 10-tool-call cap. Fetch reads every page, so
  it can never fit the contract; only a browser that chooses which pages to open can.
  - The contract sets `interaction_required`. The constraint checker rejects any fetch genome
    for it and accepts a jev genome.
  - Without a JEV provider, the runtime marks `jev` unavailable, so these tasks have no
    admissible workflow. They are then reported as skipped and never substituted.
* **`Bjoin`** joins `items`, `orders`, `stores` and `suppliers`. Every page is byte-identical to
  the unfiltered API response, and the truth is computed from `records.json`.
  - A template whose answer would be ambiguous (a tie) is re-drawn with a different seed.

### What tests enforce

The checks live in `tests/test_benchmarks_mvp.py`:

* Size and coverage:
  - at least 80 tasks, with Class A and Class B balanced
  - unique ids
  - every task has a catalog entry and a snapshot
  - every suite has train, validation and held-out rows
  - the required shapes are covered
* Truth and fixtures:
  - answer types and matchers agree with the truth
  - Class A truth appears literally on the pages
  - Class B pages equal the API response
* No copies:
  - question templates are not renamed copies within a directory
  - new held-out questions never repeat a search question
  - search and held-out share no entities
* Splits: every search task is in exactly one split, and the splits follow the split rule.
* JEV:
  - interaction tasks are jev-only and have more pages than their tool cap
  - every non-JEV task fits its cap with fetch or api
* Every contract admits some genome.
* Hashes are frozen, and regeneration gives identical bytes.

## Splits and authority

| Rows | Split rule | Allowed use |
|---|---|---|
| optimization (`train`, 44) | every search row not in validation | optimizer feedback (pheromones, proposals) |
| validation (16) | per suite, ⌈n/4⌉ rows: the frozen pre-#13 validation rows, then the last of the new rows in `sha256('wynk-split-v1:' + id)` order | selection: incumbent curve, suite champion, per-task champions |
| held-out test (`benchmarks/heldout/`, 40) | separate directory, separate entities | reporting only |

Frozen pre-#13 assignments take precedence; hash ordering fills only the remaining validation
quota. The 16 original MVP rows keep the split they had before #13
(`benchmarks.build.FROZEN_SPLITS`):

| Split | Frozen rows |
|---|---|
| train (10) | A-002, A-003, A-004, A-005, A-008, B-002, B-003, B-004, B-005, B-007 |
| validation (6) | A-001, A-006, A-007, B-001, B-006, B-008 |

Per suite, the validation target is ⌈n/4⌉ rows. The suite's other rows are sorted by the hash,
and only the last ⌈n/4⌉ − (frozen validation rows) of them become validation; the rest are
train. Suites A and B have 16 rows each, so a target of 4: 3 frozen + 1 new (A-011, B-010).
Suites with no frozen rows (Along, Ajev, Bjoin) use the plain hash split. Filtering the
expanded benchmark to the original 16 ids, e.g. `legacy_suite("A", task_ids=...)`, reproduces
the pre-#13 membership and order exactly.

`ContractSuite.check_feedback` refuses any non-optimization row before it reaches
`optimizer.observe`.

The full-suite recorder raises `SplitAccessError` if a held-out row reaches the search phase.
Search results are written to `search.json` before any held-out row runs. The final phase reads
only that frozen file, writes only `final.json`, and touches no optimizer, champion or workflow
memory.

`benchmarks.heldout.OSS_TEST_IDS` is the 16-task subset the OSS starter comparison was frozen on.
`benchmark_hash(HELDOUT_DIR, task_ids=OSS_TEST_IDS)` reproduces its pinned hash `f88b7fc4…`.

## Canonical protocol

The protocol is defined in `experiments/configs/full_suite.json` and run with
`python -m experiments.full_suite run`.

| Setting | Value |
|---|---|
| suites | A, Along, Ajev, B, Bjoin |
| optimizers | `aco_mmas`, `random_search` (compared: ACO − random) |
| seeds | 0, 1, 2, 3, 4 |
| candidate budget | 16 genomes × every optimization row × 1 search trial |
| batch | 2 genomes per round (= ants per pheromone update) |
| final | each frozen suite champion, 2 trials on every held-out row of its suite |

The candidate budget works out per suite as follows, identical for both optimizers:

| Suite | Train evaluations per search |
|---|---|
| A | 192 |
| Along | 96 |
| Ajev | 64 |
| B | 192 |
| Bjoin | 160 |

`python -m experiments.full_suite plan` prints the full plan. Command-line flags override any
setting, and the run records the protocol it actually used in `protocol.json`.

Execution seeds:

* Execution seeds depend on (seed, genome, task, trial) but **not** on the optimizer. ACO and
  random therefore see identical noise for the same candidate, and share the run cache.
* Validation runs use trial offset 10 000, and final runs use 20 000.
* Proposal order inside a search depends only on its seed (`run_search` is order-preserving), so
  the same seed gives the same candidates whatever the concurrency.

## Artifacts (`experiments/results/<run>/`)

| File | Contents |
|---|---|
| `protocol.json` | effective protocol, budgets, benchmark / held-out hashes, suite hashes, model hash, JEV provider, `synthetic` flag |
| `search.json` | per (suite, optimizer, seed): status / reason, config, `run_search` result (curve, suite champion, per-task champions), genome order, genomes, every run |
| `final.json` | every held-out run of every frozen champion |
| `evaluations.jsonl` | every search run, as it happened |
| `per_task.json` / `per_task.csv` | one row per task × optimizer × seed (completed, skipped or failed) |
| `summary.json` | aggregates, paired comparisons, per-task wins/losses/ties, champions |
| `REPORT.md` | the human-readable report |
| `learning_curve_<suite>.png` | validation fitness of the incumbent vs train evaluations |
| `run_cache.jsonl`, `final_run_cache.jsonl` | run cache: re-running the same command resumes |

Each per-task row records:

* identity: suite, task id, class, shape, split, optimizer, seed, status, reason
* the selection: search trials, selected genome hash and stage path, whether the champion is
  feasible, and the task champion (validation rows only)
* outcome: trials, passes, pass rate, `passed` (at least half of the trials pass), mean fitness,
  mean evaluator score
* cost: mean tokens, model calls, tool calls and wall time
* evidence source (the gather source) and evidence provider (the JEV identity)
* failure kinds: the runtime failure kind, otherwise `infeasible` / `wrong_answer`
* the runs themselves

For train and validation rows, the runs are the selected genome's own search-phase runs. For test
rows, they are its final runs.

### Statistics

* **Seed-level aggregates.** Within one (optimizer, seed), a group's value (overall, per class,
  suite or shape) is the mean over its tasks. Across seeds the report gives the mean, the SD and
  a 95% Student-t CI.
* **ACO vs random.**
  - ACO and random are paired by seed. The test is an exact two-sided sign-flip permutation test
    on the per-seed differences.
  - With 5 seeds the smallest attainable p is 0.0625, so the canonical protocol **cannot** report
    a significant seed-level difference at 0.05. The report states this instead of implying one.
* **Per task.**
  - ACO and random are compared on mean pass rate over seeds; ties are broken by mean evaluator
    score.
  - Tasks are counted as wins, losses or ties, with an exact sign test.
  - *Rescued* means ACO's pass rate is ≥ 0.5 while random's is < 0.5; *broken* is the reverse.
  - Every task where ACO loses is listed by name, so a regression stays visible even when the
    mean improves.

A `--synthetic` run uses the labelled fake objective (`experiments.synthetic`). Its
`protocol.json` says `"synthetic": true` and `REPORT.md` opens with a warning. It exercises the
pipeline and is never a result.
