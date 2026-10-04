# Architecture Contracts (Phase 0 - FROZEN)

These contracts are shared by both parallel tracks. Treat them as frozen: if you must change one,
record it in **Contract change log** at the bottom *in the same PR* - never change silently.

> **Provenance note.** The "Final Architecture v2" document was not present in the workspace
> when Phase 0 was built. These contracts were derived from the Phase 0 brief. Items that went
> beyond the brief are listed under **Decisions to confirm** - check them against the source
> document first.

## 1. Four authorities

| Authority | Only does | Never does |
|---|---|---|
| **Gemma** | extract, reason, synthesize (+ self-consistency sampling) | judge correctness, choose workflow, see ground truth, steer the optimizer |
| **Optimizer / ACO** | decide *how* to execute (workflow structure/config) | execute workflows, judge answers, see ground truth |
| **MAF** (via compiler/runtime) | execute the compiled graph | plan the workflow, judge correctness |
| **Deterministic evaluator** | PASS / FAIL / INFEASIBLE + search fitness | ask an LLM for a verdict |

How the code makes violations hard:

* `ModelRole` (runtime/gemma_client.py) lists the four permitted Gemma uses; nothing else is representable.
* `ExecutionResult` (runtime output) has **no** verdict/fitness field. Only `Evaluation` does, and only the evaluator builds it.
* Optimizers receive `RuntimeTask` and `EvaluatedRun`; they never receive `TaskSpec`, a model client, or the evaluator.
* Budget breach is decided by `usage_exceeds()` (pure arithmetic), never by a model.

## 2. Module map and dependency rules

```
core/          frozen shared contracts + pure deterministic rules (no I/O, no LLM, no MAF)
optimizers/    search algorithms           -> may import core only
compiler/      Genome -> DAG -> framework  -> core (+ MAF, lazily, maf_compiler.py only)
runtime/       executors, budget guard, model client -> core (+ MAF if/when needed)
evaluation/    deterministic evaluator     -> core (may hold TaskSpec / ground truth)
benchmarks/    snapshots, TaskSpecs, splits -> core
store/         run-store contract          -> core
memory/        persistent workflow memory  -> core, optimizers, experiments (JSON per class;
               written only from evaluator-measured results + MMAS pheromones, never an LLM)
router/ experiments/ api/ ui/              later phases
```

Mechanically enforced by `tests/test_authority_boundaries.py`:

* `runtime/`, `compiler/`, `optimizers/`, `router/` and every `core/` module except
  `core/task_spec.py` must **not** import `TaskSpec` / `GroundTruth`, import
  `evaluation`/`benchmarks`/`store`, or contain the token `ground_truth`.
* `agent_framework` may only be imported from `compiler/` and `runtime/`.

## 3. Shared models (where they live)

| Concept | Module | Notes |
|---|---|---|
| Stage specs (`GatherStage` ... `SynthesizeStage`), option enums | `core/stages.py` | frozen, discriminated on `kind`; option strings are the architecture's (`parallel-2`, `retry-1`, ...) |
| `Genome` | `core/genome.py` | ordered tuple of stage specs; may be partial |
| `Grammar`, `DataType` | `core/grammar.py` | types + successors + placement |
| `ConstraintChecker`, `ConstraintConfig` | `core/constraints.py` | the one hard-constraint layer |
| `Violation`, `ViolationCode` | `core/violations.py` | shared by grammar and constraints |
| `CostModel`, `CostEstimate`, `StaticCostModel` | `core/cost_model.py` | placeholder numbers, see 8 |
| `RuntimeTask`, `TaskSpec`, `Caps`, `GroundTruth`, matcher config | `core/task_spec.py` | see 5 |
| `EvidenceSpan`, `FieldEvidence` | `core/evidence.py` | `page_id`, `char_start`, `char_end`, `content_hash` |
| `Page`, `Pages`, `Fact`, `Facts`, `Answer` | `core/payloads.py` | data flowing between stages |
| `RunKey`, `RunVersions`, `ExecutionResult`, `StageTrace`, `FailureInfo`, `BudgetUsage`, `ExecutionMetrics`, `Verdict`, `Evaluation`, `EvaluatedRun` | `core/results.py` | see 6 |
| `canonical_json`, `canonical_hash` | `core/canonical.py` | every identity goes through here |

## 4. Genome invariants

* A genome is an **ordered** tuple of typed stages. `EXTRACT->VERIFY->SYNTHESIZE` != `EXTRACT->SYNTHESIZE->VERIFY`.
* Canonical form: `{"schema":"genome/1","stages":[{...}]}` as compact, key-sorted ASCII JSON;
  hash = SHA-256 of that string. No timestamps, UUIDs, dict-order or environment dependence.
  (`tests/test_genome.py` pins a golden hash and checks it across processes / `PYTHONHASHSEED`.)
* Equality **iff** canonical JSON equal. Genomes are immutable; `extend()` returns a new one.
* Changing the canonical form requires bumping `GENOME_SCHEMA_VERSION` (and logging it below).
* A `Genome` is plain data and may be partial. Validity is decided by `Grammar` and
  `ConstraintChecker` - never by the optimizer.

### Grammar (`core/grammar.py`)

| Stage | Input -> Output | Required |
|---|---|---|
| GATHER | Task -> Pages | yes |
| FILTER | Pages -> Pages | no (max 1) |
| EXTRACT | Pages -> Facts | yes |
| REASON | Facts -> Facts | no (max 1) |
| VERIFY | Facts -> Facts **or** Answer -> Answer (by position) | no (no two adjacent) |
| SYNTHESIZE | Facts -> Answer | yes |

A complete workflow starts from Task and ends in Answer. Optimizers call
`Grammar.valid_successors(partial)` / `valid_successor_specs(partial)`; they must not copy these rules.

### Hard constraints (`core/constraints.py`)

Jev cannot use `parallel-4` - at most 2 active verifiers (configurable) - `self_consistency` at most
once - `regather` not allowed when GATHER source is Jev - required stages present - stage typing valid -
workflow terminates in an Answer - source must be in `task.allowed_sources` - `interaction_required`
forces source `jev` - budget feasibility via the cost model.
Optimizers use `ConstraintChecker.check(...)` and `admissible_successors(partial, task)`.
All rules are monotone, so `check(..., complete=False)` is a sound prefix test.

## 5. Task contract and the no-ground-truth runtime rule

* `RuntimeTask` = id, task_class, question, answer_schema, caps (tokens, wall_time_s, tool_calls, retries),
  allowed_sources, snapshot_id, interaction_required. **This is the only task type runtime-side code may hold.**
* `TaskSpec` = `RuntimeTask` + `GroundTruth` + per-field `MatcherConfig`. Offline only
  (evaluation/, benchmarks/, store/, tests/). Composition not inheritance, so a `TaskSpec` is *not* a
  `RuntimeTask`; `GroundTruth.__repr__` is redacted; `TaskSpec.runtime_view()` is the only hand-off.

## 6. Run / evaluation contracts

* `RunKey` = genome_hash + task_id + trial + seed + `RunVersions` (model hash, prompt-template version,
  benchmark hash, compiler version, grammar version). `RunKey.run_id` is a deterministic hash = cache key.
* `ExecutionResult` (runtime): key, answer (+structured evidence), metrics, stage trace, budget usage,
  failure. **No verdict.**
* `Evaluation` (evaluator): `Verdict` in {PASS, FAIL, INFEASIBLE}, finite `fitness`, evaluator version,
  per-field results. `EvaluatedRun` = both; this is what optimizers observe and the store persists.
* A `BUDGET_EXCEEDED` failure (or usage above caps, per `usage_exceeds`) is mapped to INFEASIBLE by the evaluator.
* Runtime VERIFY may inspect runtime outputs/evidence and retry/correct; it cannot see ground truth.

## 7. Optimizer interface (`optimizers/base.py`)

```python
class Optimizer(ABC):
    def propose(self, k: int, context: SearchContext) -> list[Genome]: ...
    def observe(self, results: Sequence[EvaluatedRun]) -> None: ...

SearchContext(task: RuntimeTask, checker: ConstraintChecker, seed: int, round: int = 0)
```

Proposals must be complete and admissible (`ensure_admissible` checks it). No algorithm-specific concepts
live in the base. All randomness derives from `context.seed`.

## 8. Compiler and runtime boundaries

* `compile_genome(genome) -> WorkflowDAG` (`compiler/dag.py`) is pure: same genome + compiler version =>
  identical DAG (same `dag_hash`). The DAG is validated with NetworkX (acyclic, connected, one Task source,
  one Answer sink, edge types match). Failure strategies are node attributes, not back-edges.
* `Compiler` protocol: `to_dag(genome)` and `build(dag, executors)`. `MAFCompiler.build` is the only MAF touchpoint.
  MAF calls used (`WorkflowBuilder(start_executor=, output_from=)`, `.add_edge`, `.build`) were checked against the
  official docs **and** built against `agent-framework-core 1.20.0` in a scratch venv
  (`tests/test_maf_integration.py`, skipped when MAF is not installed). Writing MAF `Executor` wrappers around
  `StageExecutor`s and *running* workflows is Track B work. A LangGraph compiler would add a second `Compiler`
  without touching Genome/optimizers/evaluator/benchmarks.
* Executors (`runtime/executors/base.py`): `StageExecutor.run(ExecutorInput, RunContext) -> ExecutorOutput`
  (exactly one of payload/failure). `GuardedExecutor` wraps *any* executor with budget accounting;
  `RunContext` holds a `RuntimeTask`, never a `TaskSpec`.
* `ModelClient` (`runtime/gemma_client.py`): structured-output schema, prompt-template id+version, seed,
  token usage, model hash, deterministic `cache_key(model_hash)`. No backend is wired.

## 9. Cost model

`CostModel.estimate(genome, task)` returns a **best-case lower bound** (tokens, latency, tool calls) plus
`max_retries` / `retry_risk`. Because it is a lower bound, "estimate > cap" *proves* a cap cannot be met and the
checker rejects; retries are reported but never used to reject. For partial genomes the cheapest completion of
missing required stages is added. `StaticCostModel` ships **uncalibrated placeholder** numbers (`CostTable`);
calibrate from the run store later.

## 10. Run store

`store/runs.py` defines the `RunStore` protocol (+ `InMemoryRunStore` for tests). `store/schema.sql` is a
DRAFT DDL. Core models know nothing about DuckDB.

## Decisions to confirm against the architecture document

These go beyond the literal Phase 0 brief:

1. GATHER first and only once; FILTER and REASON at most once; no two adjacent VERIFY stages (keeps the grammar finite).
2. VERIFY has a mandatory `on_failure`; its Facts/Answer typing is inferred from position.
3. `interaction_required` => `jev` must be allowed (TaskSpec) and used (constraint).
4. Matcher kinds (exact, normalized_text, numeric_tolerance, date, set_equal) and answer field types are provisional.
5. `fitness` is any finite float (higher = better); scale is Track A's to define in `evaluation/fitness.py`.
6. Cost numbers are placeholders; retries never trigger static rejection.
7. Task classes A/B/C are an enum only; no Class C behaviour.

## Contract change log

| Date | Change | Reason | Affects |
|---|---|---|---|
| 2026-10-04 | Phase 0 initial freeze | - | all |
| 2026-10-04 | `run_search` result gains additive `task_class`, `workflows` (validated incumbents: genome, validation fitness/pass rate, mean tokens/wall time) and `versions`; `MMASACO` gains `version`, `from_pheromones`, `explored_pheromones` | persistent workflow memory + ACO warm start (`memory/`); cold-start results unchanged (golden-hash test) | experiments, optimizers |
