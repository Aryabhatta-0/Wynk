# Runtime MVP (Track B)

```
ExecutionTask (TaskContract + row inputs) + Genome -> WorkflowRunner.run -> compile_genome (pure DAG) -> MAFCompiler.build (MAF workflow)
   -> one MAF Executor per node (runtime/maf_nodes.py) -> StageRunner -> stage executors -> ExecutionResult
```

Single public entry point: `runtime.runner.WorkflowRunner(model=..., benchmark_hash=..., pages=..., api=...).run(genome, task, trial=, seed=)`
(`run_sync` for scripts). Three hand-written genomes live in `runtime/mvp_genomes.py` (A minimal, B filter+reason, C verify).

## What each stage does

| Stage | MVP behaviour |
|---|---|
| GATHER `fetch` | reads every page of the task's frozen snapshot; 1 tool call per page; `parallel-2/4` read concurrently (deterministic order) |
| GATHER `api` | one call per mock endpoint; 1 tool call each |
| GATHER `jev` | **not implemented** - fails with `EXECUTOR_ERROR` |
| FILTER `keyword_chunk` / `section_select` | deterministic keyword overlap with question + field names; keeps best 6 chunks; chunks keep original `page_id` + offset so spans stay valid; passes pages through if nothing matches |
| EXTRACT / REASON / SYNTHESIZE | one Gemma call each (`direct\|schema_guided\|cot`, `single\|decompose`, `direct\|cite_evidence` change the prompt text only). Model returns values + **verbatim quotes**; the runtime locates quotes and builds `EvidenceSpan`s pinned to the original page's content hash |
| VERIFY `schema_check` | required fields present, no unknown fields, JSON types valid (answers) |
| VERIFY `evidence_span` | every fact / answer field has spans that exist on the original page (hash + range) and whose text contains the value |
| VERIFY `self_consistency` | **not implemented** - fails with `EXECUTOR_ERROR` |

Recovery (bounded, from the genome): on a failed VERIFY, `retry-1/2` re-run the producing stage (fresh per-attempt seed) then re-verify; `regather` re-runs GATHER..verifier once. Each retry is charged to the `retries` cap; exhausting retries ends the run with the verifier's `SCHEMA_INVALID` failure and no answer.

Budget: every executor is wrapped by `GuardedExecutor`; a breach yields a `BUDGET_EXCEEDED` failure and downstream nodes never run. A genome whose best-case estimate already exceeds the caps is not executed (zero-usage `BUDGET_EXCEEDED` result). Structurally invalid genomes raise `InadmissibleGenome`.

Ground truth: the runtime only ever receives `ExecutionTask` (a `TaskContract` + one row's inputs); `tests/test_authority_boundaries.py` enforces that `runtime/` and `compiler/` never import `TaskSpec`/`GroundTruth`/`evaluation`.

## Model backend

`runtime/backends/openai_compatible.py` provides `OpenAICompatibleClient` (stdlib only), an adapter behind the provider-neutral `ModelClient` protocol (`runtime/model_client.py`). Pinned models come from the registry (`client_for(entry)`, see [MODEL_REGISTRY.md](MODEL_REGISTRY.md)); for ad-hoc runs configure env vars, then `client_from_env()`. Each `WYNK_MODEL_*` name falls back to its legacy `GEMMA_*` name (`runtime/gemma_client.py` remains as a deprecated import path):

| Variable | Meaning |
|---|---|
| `WYNK_MODEL_BASE_URL` (`GEMMA_BASE_URL`) | OpenAI-compatible base URL (e.g. `http://localhost:8000/v1`) |
| `WYNK_MODEL` (`GEMMA_MODEL`) | model name sent to the backend |
| `WYNK_MODEL_API_KEY` (`GEMMA_API_KEY`) | optional bearer token |
| `WYNK_MODEL_REVISION` (`GEMMA_MODEL_REVISION`) | optional; part of `model_hash` |
| `WYNK_MODEL_STRUCTURED=0` (`GEMMA_STRUCTURED`) | disable `response_format` json_schema |
| `WYNK_MODEL_TIMEOUT_S` (`GEMMA_TIMEOUT_S`) | default 120 |
| `WYNK_MODEL_MAX_RETRIES` (`GEMMA_MAX_RETRIES`) | default 3 |

Missing config raises `ModelUnavailableError`; a failing backend becomes a `MODEL_ERROR` run failure. Nothing is ever faked. Backend usage (`prompt_tokens`/`completion_tokens`) is required and is what the budget guard charges. **No live model backend was available in the build environment**; the client is tested against a local OpenAI-compatible fake server only.

## Local data layout (Track A can satisfy the same protocols)

```
<root>/<snapshot_id>/*.txt|*.md|*.html     frozen pages   (runtime.sources.PageSource)
<root>/<snapshot_id>/api/*.json            mock endpoints (runtime.sources.ApiSource)
```

## Contract notes (no frozen model was changed)

* `FailureKind` has no evidence-specific value; failed `evidence_span` checks use `SCHEMA_INVALID` (message names the verifier). Smallest possible improvement: add `EVIDENCE_INVALID` to `core/results.FailureKind`.
* `ExecutionMetrics` has no latency field; wall time is `budget_usage.wall_time_s` (sum of measured stage times).
* `runtime/executors/base.py` (runtime-owned) gained two additive `ExecutorInput` fields: `attempt` and `source_pages`.
* `maf` extra now points at `agent-framework-core` (what the code imports; verified at 1.20.0). The full `agent-framework` meta-package failed to install on Windows because of path length.
