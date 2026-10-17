# Runtime MVP (Track B)

```
ExecutionTask (TaskContract + row inputs) + Genome -> WorkflowRunner.run -> compile_genome (pure DAG) -> MAFCompiler.build (MAF workflow)
   -> one MAF Executor per node (runtime/maf_nodes.py) -> StageRunner -> stage executors -> ExecutionResult
```

Single public entry point: `runtime.runner.WorkflowRunner(model=..., benchmark_hash=..., pages=..., api=..., jev=None).run(genome, task, trial=, seed=)`
(`run_sync` for scripts). Three hand-written genomes live in `runtime/mvp_genomes.py` (A minimal, B filter+reason, C verify).

## What each stage does

| Stage | MVP behaviour |
|---|---|
| GATHER `fetch` | reads every page of the task's frozen snapshot; 1 tool call per page; `parallel-2/4` read concurrently (deterministic order) |
| GATHER `api` | one call per mock endpoint; 1 tool call each |
| GATHER `jev` | a JEV browsing agent over the task's snapshot pages (see [JEV](#gather-jev)); 1 tool call per decision + 1 per opened page; `sequential` opens 1 page per decision, `parallel-2` up to 2; available only with a configured `JevSource` |
| FILTER `keyword_chunk` / `section_select` | deterministic keyword overlap with question + field names; keeps best 6 chunks; chunks keep original `page_id` + offset so spans stay valid; passes pages through if nothing matches |
| EXTRACT / REASON / SYNTHESIZE | one Gemma call each (`direct\|schema_guided\|cot`, `single\|decompose`, `direct\|cite_evidence` change the prompt text only). Model returns values + **verbatim quotes**; the runtime locates quotes and builds `EvidenceSpan`s pinned to the original page's content hash |
| VERIFY `schema_check` | required fields present, no unknown fields, JSON types valid (answers) |
| VERIFY `evidence_span` | every fact / answer field has spans that exist on the original page (hash + range) and whose text contains the value |
| VERIFY `self_consistency` | **not implemented** - fails with `EXECUTOR_ERROR` |

Recovery (bounded, from the genome): on a failed VERIFY, `retry-1/2` re-run the producing stage (fresh per-attempt seed) then re-verify; `regather` re-runs GATHER..verifier once. Each retry is charged to the `retries` cap; exhausting retries ends the run with the verifier's `SCHEMA_INVALID` failure and no answer.

Budget: every executor is wrapped by `GuardedExecutor`; a breach yields a `BUDGET_EXCEEDED` failure and downstream nodes never run. A genome whose best-case estimate already exceeds the caps is not executed (zero-usage `BUDGET_EXCEEDED` result). Structurally invalid genomes raise `InadmissibleGenome`.

Ground truth: the runtime only ever receives `ExecutionTask` (a `TaskContract` + one row's inputs); `tests/test_authority_boundaries.py` enforces that `runtime/` and `compiler/` never import `TaskSpec`/`GroundTruth`/`evaluation`.

## GATHER jev

`GATHER(source="jev")` runs through the same path as every other stage:
`WorkflowRunner -> compiler -> StageRunner -> GatherExecutor -> JevSource -> Pages`.

* **Interface** (`runtime/sources.py`): a `JevSource` has an `identity` and one method,
  `decide(JevObservation) -> JevDecision`. Wynk owns the loop: each step it shows the agent the
  goal (`ctx.task.question`), the surface (`snapshot://<id>`), the pages not opened yet and the
  pages already opened, and accepts only `open` (1..width of the offered pages), `done` or
  `blocked`. `checked_decision` rejects anything else (unknown or extra pages, free text,
  selectors, code) as a `SourceError`; nothing a provider returns is ever executed. Wynk then
  reads the chosen pages through the runner's `PageSource`.
* **Result**: normal `Pages`; `page_id` and `source_ref` are the snapshot's own (`p1`,
  `snapshot://<id>/p1.txt`), sorted by `page_id` whatever the opening or completion order.
  Downstream FILTER / EXTRACT / VERIFY / SYNTHESIZE contracts are unchanged.
* **Budget**: one decision = 1 tool call, one opened page = 1 tool call (`BudgetUsage.tool_calls`).
  Each action's cost is known before it runs, so the tool-call cap is checked first: a
  decision runs only if one more call fits, and the chosen pages are read only if all of them
  fit (a `parallel-2` choice of two pages with one call left reads neither). An action that does
  not fit is never started; the stage stops with `BUDGET_EXCEEDED` (cap `tool_calls`) and the
  run's tool calls stay at or below the cap. The JEV model's own tokens belong to the provider and are not charged as Wynk model tokens;
  wall time is measured by the guarded executor as for every stage. `core/cost_model.py` keeps
  its conservative admission estimate (10 tool calls) - it is not runtime usage.
* **Failures**: a provider error, an invalid decision, a page read error, an unknown snapshot,
  or an agent that stops without opening any page is an `EXECUTOR_ERROR`. A jev stage never
  falls back to `fetch` or `api`.
* **Constraints** (now enforced against a real executor): `JEV_PARALLEL_4` (the executor also
  refuses `parallel-4`), `JEV_REGATHER` (a regather would re-run and re-charge the whole JEV
  loop), `INTERACTION_REQUIRES_JEV` (an `interaction_required` contract admits only jev gathers).
* **Reproducibility**: a jev run's `RunVersions.evidence_source` is the source's `identity`, so
  its `run_id` never collides with a frozen fetch/api run. Runs that do not use jev serialize
  without the field, so their identities and the frozen benchmark are unchanged, with or
  without a configured `JevSource`.

### What "JEV available" requires

1. A `JevSource` passed to `WorkflowRunner(jev=...)` (or `experiments.real_runtime.build_runner(...,
   jev=...)`), together with the `PageSource` it browses (the runner refuses a `JevSource` without
   one).
2. A contract whose `workflow.allowed_sources` contains `jev` and a `wynk_snapshot` dataset (jev
   cannot read inline csv/jsonl rows).

Without (1) the runner's checker marks jev `RUNTIME_UNAVAILABLE`: every jev genome is
inadmissible and nothing is substituted.

### Hosted provider

`runtime/backends/typesafe_jev.py` implements `JevSource` with TypeSafe's hosted JEV model (the
`systemone` choice API): each decision asks two bounded choice questions (operation; which
listed page) and accepts only a well-formed probability distribution over exactly the offered
options whose `choice` is its most probable one. `parallel-2` also opens the second most
probable page when it has non-zero probability. Requests use stdlib `urllib`, a bounded timeout
and no retries; the key is sent only in the `Authorization` header and never appears in
`repr`, `identity`, errors or results. A decision request sends the goal and the text of the
pages already opened (at most 6000 characters) to the provider. It is the product's second
outbound call site; `tests/test_upload_security.py` pins exactly these two, and GATHER jev never
reads uploaded inline datasets (see [upload_security.md](upload_security.md#8-no-user-data-training)).

| Variable | Meaning |
|---|---|
| `WYNK_JEV_API_KEY` (`TYPESAFE_API_KEY`) | required; without it JEV stays unavailable |
| `WYNK_JEV_MODEL` (`TYPESAFE_MODEL`) | required with a key; an explicit, pinned model version. No default. Floating aliases are refused: an id whose terminal identity is `latest` / `current` / `stable` / `default` (`latest`, `jev-latest`, `jev:latest`, `typesafe/jev-latest`). A versioned id that only contains such a word (`jev-stable-2026-09`) is accepted |
| `WYNK_JEV_ENDPOINT` | default `https://api.typesafe.ai/v1/systemone`; must be https |
| `WYNK_JEV_TIMEOUT_S` | default 25, at most 120 |

**Model pinning.** A run's `evidence_source` is the source's `identity`,
`jev:typesafe-jev/<model>@<host><path>`, so it is only as reproducible as `<model>`. An alias the
provider can re-point (such as `jev-latest`) would let two different models share one run
identity, so Wynk refuses it. A key set without `WYNK_JEV_MODEL` fails closed with
`JevConfigError`: JEV is neither enabled on a default nor silently left off. Wynk records the
pinned id exactly as configured. It does not invent a provider revision that the API does not
expose, so pin the most specific model id your provider account offers. Changing the pinned
model changes the identity of every jev run and of no other run. Test doubles
(`ScriptedJev`) keep their fixed fixture identity.

`api/chat.py` builds its runner with `jev_from_env(env)`; a JEV configuration error stops the
engine from starting and `/api/health` reports the reason. `/api/health` reports the JEV
identity (never the key). Tests use `tests/runtime_helpers.ScriptedJev` and a fake transport
only; no test touches the network or a browser.

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
<root>/<snapshot_id>/*.txt|*.md|*.html     frozen pages   (runtime.sources.PageSource; also the jev surface)
<root>/<snapshot_id>/api/*.json            mock endpoints (runtime.sources.ApiSource)
```

## Contract notes (no frozen model was changed)

* `FailureKind` has no evidence-specific value; failed `evidence_span` checks use `SCHEMA_INVALID` (message names the verifier). Smallest possible improvement: add `EVIDENCE_INVALID` to `core/results.FailureKind`.
* `ExecutionMetrics` has no latency field; wall time is `budget_usage.wall_time_s` (sum of measured stage times).
* `runtime/executors/base.py` (runtime-owned) gained two additive `ExecutorInput` fields: `attempt` and `source_pages`.
* `maf` extra now points at `agent-framework-core` (what the code imports; verified at 1.20.0). The full `agent-framework` meta-package failed to install on Windows because of path length.
