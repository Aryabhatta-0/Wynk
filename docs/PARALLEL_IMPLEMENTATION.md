# Parallel Implementation Plan (after Phase 0)

Read `ARCHITECTURE_CONTRACTS.md` first. Everything under `core/` is **frozen**: to change a shared model,
make the change in its own PR, add a row to the "Contract change log", and tell the other developer.

## Developer A - Benchmark + Evaluator

Owns: `benchmarks/`, `evaluation/`. May add test files under `tests/` named `test_evaluation_*` / `test_benchmarks_*`.

Implement:

* Class A / Class B frozen snapshots under `benchmarks/snapshots/`, `benchmarks/classes/{A,B}/`
* `TaskSpec`s with ground truth and per-field matcher config; `splits.json`; benchmark hash
* `mirror_server.py` (offline snapshot mirror) and `mock_api.py` (deterministic Class B API)
* deterministic matchers (`evaluation/matchers.py`), evidence verification against snapshots
  (`evaluation/evidence.py`), the PASS/FAIL/INFEASIBLE gate (`evaluation/gate.py`), shaped fitness
  (`evaluation/fitness.py`)

Rules: implement the `Evaluator` / `Matcher` / `EvidenceVerifier` / `FitnessFunction` protocols as written;
INFEASIBLE comes from `BUDGET_EXCEEDED` failures or `usage_exceeds`; never call an LLM; the evaluator is the only
code that reads `TaskSpec.ground_truth`.

## Developer B - Grammar + Compiler + Runtime

Owns: `core/` (grammar/constraint/cost behaviour only - not the shared data models), `compiler/`, `runtime/`.

Implement:

* final grammar / constraint / cost-model behaviour (calibrated `CostTable`)
* workflow compilation incl. MAF `Executor` wrappers around `StageExecutor`s and running a workflow end-to-end
  (`compiler/maf_compiler.py`; MAF imports stay in `compiler/` and `runtime/`)
* executors: `gather_fetch`, `gather_api`, `gather_jev`, `filter`, Gemma stages, schema / evidence /
  self-consistency verifiers, `runtime/prompts/` (versioned)
* real `ModelClient` for Gemma, budget enforcement on every executor via `GuardedExecutor`
* a runner that turns `(Genome, RuntimeTask, seed)` into an `ExecutionResult`

Rules: only `RuntimeTask` is visible at runtime; no verdicts; keep `compile_genome` pure.

## Shared / hand-off

* Hand-off object: `ExecutionResult` (B -> A). A produces `Evaluation` -> `EvaluatedRun` -> `RunStore`.
* Until both tracks exist, each can test alone: A builds `ExecutionResult`s by hand; B inspects results without an evaluator.
* Frozen-model changes: separate PR + change-log row.
* **Optimizers (`optimizers/`) start only when hand-built workflows can be executed (B) and evaluated (A) end to end.**
  The `Optimizer` interface is already fixed.

## Suggested branches

```
main                       <- Phase 0 foundation (commit this first)
track-a/benchmark-evaluator
track-b/compiler-runtime
```

Run `python -m pytest` and `python -m ruff check .` before every PR.
