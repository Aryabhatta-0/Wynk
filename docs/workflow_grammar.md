# Workflow grammar

A workflow genome is an ordered tuple of typed, configured stages. The grammar
(`core/grammar.py`) decides which genomes are structurally valid, and the constraint layer
(`core/constraints.py`) decides which of those are admissible for a task. Both run before
compilation and before any model call. The compiler and the runtime never discover structural
invalidity late.

## Capabilities: what existed and what was added

Each requested capability maps to the smallest representation that fits the existing linear
genome → DAG → executor architecture. `core.grammar.capabilities(spec)` returns the mapping in code.

| Capability | Representation | Status |
|---|---|---|
| Direct | `DIRECT(answer \| cot)` | **new** stage: Task → Answer, one model call, no retrieval |
| Gather | `GATHER(source, mode)` | existing (`fetch`, `api` executable; `jev` stays unsupported) |
| Filter | `FILTER(keyword_chunk \| section_select)` | existing |
| Reason | `REASON(single)` | existing |
| Decompose | `REASON(decompose)` | existing configuration |
| Parallel | `GATHER(mode=parallel-2 \| parallel-4)` | existing configuration: bounded fan-out width of page reads |
| Verify | `VERIFY(method, on_failure)` | existing |
| Synthesize | `SYNTHESIZE(direct \| cite_evidence)` | existing |
| Confidence Gate | `CONFIDENCE_GATE(support-50 \| support-100)` | **new** stage: terminal Answer → Answer |
| *(bridge)* | `EXTRACT(direct \| schema_guided \| cot)` | existing: Pages → Facts |

Parallel and Decompose are bounded configurations of linear stages. A genome therefore always
compiles to a linear DAG, so no fan-out or fan-in structure exists that could be malformed. There is
no free-form, arbitrary-code or planner stage: every option is an enum, and a genome containing an
unknown kind, option or extra field fails to parse.

## Stage contracts

Shapes are the grammar's data types: `Task` (the `ExecutionTask`: a contract + one row's inputs),
`Pages`, `Facts` and `Answer`.
Costs are the placeholder `CostTable` values (uncalibrated, advisory unless the table is marked
`proven_lower_bound`).

| Stage | Input → Output | Valid predecessors | Valid successors | Bounded configuration | Cost / budget |
|---|---|---|---|---|---|
| GATHER | Task → Pages | start | FILTER, EXTRACT | source {fetch, api, jev} × mode {sequential, parallel-2, parallel-4} = 9 | tool calls: fetch 3, api 2, jev 10; latency scaled by mode |
| FILTER | Pages → Pages | GATHER | EXTRACT | {keyword_chunk, section_select}; at most 1 | deterministic, no tokens |
| EXTRACT | Pages → Facts | GATHER, FILTER | REASON, VERIFY, SYNTHESIZE | {direct, schema_guided, cot} | 1500 / 2000 / 3500 tokens |
| REASON | Facts → Facts | EXTRACT, VERIFY | VERIFY, SYNTHESIZE | {single, decompose}; at most 1 | 1000 / 3000 tokens |
| VERIFY | Facts → Facts or Answer → Answer (by position) | EXTRACT, REASON, SYNTHESIZE, DIRECT | REASON, SYNTHESIZE (on Facts); CONFIDENCE_GATE (on Answer) | method {schema_check, evidence_span, self_consistency} × on_failure {retry-1, retry-2, regather} = 9; never adjacent; ≤ 2 active (constraint) | at most 2 retries each, reported as risk and never used to reject |
| SYNTHESIZE | Facts → Answer | EXTRACT, REASON, VERIFY | VERIFY, CONFIDENCE_GATE | {direct, cite_evidence} | 800 / 1200 tokens |
| DIRECT | Task → Answer | start | VERIFY (schema_check / self_consistency, retry-1 / retry-2) | {answer, cot} | 800 / 1500 tokens, 0 tool calls |
| CONFIDENCE_GATE | Answer → Answer, **terminal** | SYNTHESIZE, VERIFY on the retrieval path | none | {support-50, support-100}; at most 1 | deterministic, no tokens; abstains with `LOW_CONFIDENCE` |

Dependencies (`core.grammar.requires`): `VERIFY(evidence_span)`, `VERIFY(on_failure=regather)` and
`CONFIDENCE_GATE` need pages from an upstream GATHER. On the DIRECT path no pages exist, so these
stages could never succeed there, and the grammar rejects them.

The confidence gate's confidence is *evidence support*: the fraction of the answer's required fields
whose value is backed by a span that verifies against the gathered pages
(`runtime.executors.verify.answer_support`). It is deterministic, does not use a model, and never
sees ground truth.

## Vocabularies: the contract selects the grammar

The authority flow is:

    TaskContract.workflow.stages -> workflow_grammar(contract) -> ConstraintChecker
                                 -> optimizer (SearchContext) / runtime (WorkflowRunner)

* `WorkflowSpec.stages` (`core/task_contract.py`) is the stage vocabulary, next to the GATHER
  policy (`allowed_sources`, `interaction_required`) from #37. There is one representation:
  `workflow_grammar(contract)` and `ConstraintChecker.check(genome, contract)` both build the
  grammar from that field (`core.grammar.grammar_for`), so a checker cannot disagree with its
  contract. `ConstraintChecker(grammar=...)` only applies to checks made without a contract.
* Left empty, `stages` means every kind the dataset supports, and it is filled in when the
  contract is validated. A validated contract therefore always states its vocabulary, and the
  vocabulary is part of `contract_hash`.
  * Every dataset supports `DIRECT` and `VERIFY`, which need only a row's inputs.
  * Retrieval kinds and `CONFIDENCE_GATE` read pages, so they need `context_columns`. Through
    #37's runtime these are a snapshot id (`SnapshotSource`) or the row's own context values
    (`InlineSource`, gathered with `fetch`).
* A contract may narrow the vocabulary. It is refused when it lists a page-reading kind the
  dataset cannot support, cannot produce an Answer, or lists retrieval-path kinds without the
  full `GATHER → EXTRACT → SYNTHESIZE` chain.
* The legacy benchmark reaches this only through `benchmarks/legacy_adapter.py`, which declares
  `stages=LEGACY_STAGE_KINDS`. `Grammar()` admits those six kinds and reports `grammar/1`. Its
  language is byte-for-byte the pre-#36 language: tests pin the adapter-path admissible set
  against hashes measured on `main@845d79b`. Generic code never branches on a task class or a
  legacy evaluator.
* Any other vocabulary reports `grammar/2[<kinds>]`. `WorkflowRunner.versions(task)` records
  the task's contract grammar in each `RunKey`.
* `compile_genome` validates structure with every kind enabled. Whether a task supports a kind is
  decided at admission, which runs first.

## Admission invariants

Every rule is monotone in the prefix, so `check(..., complete=False)` is a sound prefix test and
`admissible_successors` can never offer a stage that makes the genome invalid.

| Rejected | Code | Where |
|---|---|---|
| Missing required producers (no `DIRECT`, or an incomplete `GATHER → EXTRACT → SYNTHESIZE`) | `missing_required_stage`, `no_answer_terminal` | grammar |
| Incompatible ordering (a stage that cannot consume the current type, or a second Answer producer) | `invalid_transition` | grammar |
| Impossible dependencies (evidence check, regather or gate without GATHER) | `unsatisfied_dependency` | grammar |
| Duplicate or illegal terminal stages (anything after `CONFIDENCE_GATE`) | `after_terminal` | grammar |
| Per-kind limits and stacked verifiers (FILTER / REASON / DECOMPOSE twice, VERIFY → VERIFY) | `placement` | grammar |
| Stage outside the task's vocabulary | `stage_unsupported` | grammar |
| Unbounded fan-out widths, retry counts or free-form options | pydantic `ValidationError` | parsing |
| jev + parallel-4, jev + regather, > 2 verifiers, repeated self_consistency | existing codes | constraints |
| Source not allowed, interaction without jev (including DIRECT), runtime-unavailable option, provable budget breach | existing codes | constraints |
| More stages / model calls than the contract allows (#37), counting the cheapest completion the vocabulary allows (DIRECT = 1 stage, 1 call) | `step_limit`, `model_call_limit` | constraints |
| A vocabulary the dataset cannot support or that cannot produce an Answer | `ContractError` | contract validation |

`WorkflowRunner.run` raises `InadmissibleGenome` before compiling, so no model is called.

## Search-space bounds

The whole language is finite because types only move forward (Task → Pages → Facts → Answer),
FILTER and REASON appear at most once, verifiers are never adjacent, and nothing follows a terminal
stage. `Grammar.language_size()` and `Grammar.max_genome_length()` compute the language size and
the longest genome exactly, by memoizing over grammar states, without materializing the
genomes. Both computations would recurse forever on a cyclic grammar, so they also serve as
termination checks.

| Vocabulary | Grammar language | Longest | Admissible on the shipped runtime |
|---|---|---|---|
| legacy `grammar/1` (adapter contracts) | 340,200 (unchanged) | 8 | unchanged: 16,362 (benchmark A-001), 32,724 (B-001), same set hashes as before #36 |
| contract without context (`DIRECT`, `VERIFY`) | 10 | 2 | 6 |
| contract with context (all 8 kinds) | 1,020,610 | 9 | 49,092 = 3 × 16,362 + 6 (`fetch` only) |

The growth over legacy is a constant factor of 3: the terminal gate is absent, `support-50` or
`support-100`. DIRECT adds 10 more genomes. Optimizers sample this space through
`admissible_successors`. `ConstraintChecker.enumerate_admissible(contract)` lists it exactly, and
construction is capped at `MAX_GENOME_STAGES + 1` steps.

## Reconciliation with #20 / #37 (TaskContract authority)

#37 made `TaskContract` + `ExecutionTask` the authority for search and runtime. This grammar plugs
into that rather than beside it:

* **Admission.** The vocabulary is a field of #37's `WorkflowSpec`; there is no second
  workflow policy and no `RuntimeTask` field.
* **Lower bounds.** #37's step and model-call bounds and the cost bound all count the cheapest
  completion the contract's vocabulary allows (`core.grammar.completions`). They no longer
  assume the retrieval chain, and the legacy bounds are unchanged.
* **No-context datasets.** For a CSV/JSONL row without context columns, GATHER is outside the
  vocabulary. `allowed_sources` (still at least one source in #37's model) is never consulted.
* **Identity.** Adding the vocabulary to `contract_hash` changes run identities (`RunKey`). It
  does not change admission or proposals. `tests/test_workflow_memory.py` proves this by
  reproducing the previous synthetic-search golden exactly when contracts are hashed without
  `stages`.
