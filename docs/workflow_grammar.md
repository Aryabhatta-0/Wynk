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

Shapes are the grammar's data types: `Task` (the `RuntimeTask`), `Pages`, `Facts` and `Answer`.
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

## Vocabularies: which stages a task supports

A `Grammar` admits a fixed set of stage kinds:

* `Grammar()` admits the frozen benchmark's six kinds and reports `grammar/1`. Its language is
  byte-for-byte the pre-change language. Tests pin the enumerated admissible set against a hash
  measured on the pre-change code.
* `core.task_contract.workflow_grammar(contract)` selects a contract's vocabulary:
  * `legacy_field_match` (benchmark) contracts use the legacy six kinds, so a benchmark task has
    the same search space through either path.
  * Other contracts always support `DIRECT` and `VERIFY`. Retrieval stages and `CONFIDENCE_GATE`
    are supported only when the dataset declares `context_columns`, which are the pages that
    `fetch` serves.
  * Any other vocabulary reports `grammar/2[<kinds>]`.
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
| legacy `grammar/1` | 340,200 (unchanged) | 8 | unchanged: e.g. 16,362 (benchmark A-001), 32,724 (B-001) |
| contract without context (`DIRECT`, `VERIFY`) | 10 | 2 | 6 |
| contract with context (all 8 kinds) | 1,020,610 | 9 | 49,092 = 3 × 16,362 + 6 |

The growth over legacy is a constant factor of 3: the terminal gate is absent, `support-50` or
`support-100`. DIRECT adds 10 more genomes. Optimizers sample this space through
`admissible_successors`. `ConstraintChecker.enumerate_admissible(task)` lists it exactly, and
construction is capped at `MAX_GENOME_STAGES + 1` steps.

## TaskContract-driven callers (issue #20)

This grammar does not depend on any unmerged runtime-authority work. A contract-driven caller
builds `ConstraintChecker(grammar=workflow_grammar(contract), ...)` and passes it to
`WorkflowRunner(checker=...)` and to `SearchContext`. The runner's compiler accepts every
vocabulary. Existing `RuntimeTask` callers keep the legacy default without any changes.
`RuntimeTask.allowed_sources` still requires at least one source. For a dataset without context
columns, the contract vocabulary contains no GATHER, so the source list is never consulted.
