# Wynk vs. official OSS *starter* workflows

**Question tested:** can Wynk automatically discover a workflow that beats the common *starter*
workflows a developer gets from popular open-source agent frameworks?

This is **not** a comparison of Wynk with those frameworks. smolagents, CrewAI, LlamaIndex and
LangGraph can all build far stronger workflows than their starters. Every result is labelled with
the exact starter workflow that was run.

| id | Label used in results | Upstream (pinned) | Workflow |
|---|---|---|---|
| `smolagents` | smolagents Starter Agent (CodeAgent) | huggingface/smolagents `v1.26.0` @ `12c1bc8` | README quickstart `CodeAgent(tools, model)` |
| `crewai` | CrewAI Starter Research Workflow (1-agent sequential crew) | crewAIInc/crewAI `1.15.23` @ `deaa71e` | `crewai create crew` template: `researcher` + `research_task`, sequential |
| `llamaindex` | LlamaIndex Starter Agentic RAG (FunctionAgent, tool retrieval) | run-llama/llama_index `v0.14.25` @ `f12d46a` | starter tutorial `FunctionAgent(tools, llm, system_prompt)` |
| `langgraph` (P2) | LangGraph Basic ReAct Agent (create_react_agent) | langchain-ai/langgraph `1.2.12` @ `49cce0c` | quickstart `create_react_agent(model, tools, prompt)` |
| `wynk` | Wynk Optimized Workflow (frozen) | this repo | `frozen_wynk.json` (one genome per task class) |

`manifest.json` has the full record: repository, tag, commit, lock file and hash, install command,
agent type, configuration, and every prompt or tool modification.

## Protocol

1. **Held-out TEST split** (`benchmarks/heldout.py` -> `benchmarks/heldout/`, hash
   `f88b7fc4…`). The frozen MVP benchmark only has `train` (used by the search) and `validation`
   (used to select the incumbent), so neither may be used to report a result. The test split has
   16 tasks rendered by the same Class A / Class B generators, question templates, caps and
   matchers, over 8 new fictional companies and 8 new inventory seeds. No optimizer has read it.
   It sits outside `benchmarks/snapshots`, so the MVP `benchmark_hash` is unchanged.
2. **Freeze Wynk first** (`harness freeze` -> `frozen_wynk.json`). This step stores Wynk's own
   selected workflow per task class: the incumbent of the ACO seed with the best validation
   fitness, from the committed `experiments/results/real-gemma4` run at `ff2358b`. It also
   records the genome hash, optimizer run id, results hash, validation score, model assignments,
   tool configuration and held-out hash. The freeze refuses to overwrite itself. Every run
   re-checks the genome hashes and the held-out hash. Nothing in this directory updates
   pheromones, runs a search, or edits a genome.
3. **Same tasks, same runs.** Every system solves every held-out task `runs_per_task` times
   (default 3). The job order is shuffled with a fixed seed so backend load hits all systems
   alike.
4. **Same model and settings.** All systems use `google/gemma-4-31b-it` on the same
   OpenAI-compatible endpoint, at temperature 0 with max_tokens 1024 per call. The per-call
   timeout is 120 s, the client retries 3 times, and the run-level hang guard is 360 s. These are
   Wynk's own client settings. The settings each framework actually sent are recorded per call by
   the metering proxy.
5. **Same information (CONTROLLED_SOURCE).** Wynk has no web-search tool; it reads each task's
   frozen pages, or the mock API whose response is byte-identical to the `items` page. Every
   external starter gets two tools over the same frozen pages, `list_pages()` and
   `read_page(page_id)`, with the same names, docstrings and behaviour. These replace the
   starters' web-search or example tools. NATIVE_SEARCH is not applicable and is not run.
6. **Same query.** Each external system receives `task_prompt(task)`. It contains the question,
   the answer fields with their JSON types (rendered exactly like Wynk's runtime prompts), and an
   output contract: a JSON `answer` plus per-field `evidence` quotes. That is the same
   information and the same verbatim-quote requirement as Wynk's prompts. Nothing about Wynk's
   workflow, caps, or answers is in it.
7. **Isolation.** Each framework runs in its own pinned venv (`.oss-venvs/<id>`, built from
   `requirements/<id>.txt`). It runs as a subprocess (`runners/<id>_runner.py`) using a
   JSON-in/JSON-out protocol. Wynk runs the same way, in its own env and with the same timeout.
   Wynk's main env needs none of the competitor dependencies.
8. **Metering.** All model traffic, Wynk's included, goes through `proxy.py`. The proxy records
   backend-reported token usage and cost, call counts, latencies, and the raw request and
   response. It injects the real API key (framework code never holds it) and never retries.

## Evaluation (common, deterministic, no LLM judge)

Each output is normalised into the Wynk `ExecutionResult` contract and scored by the
**shared** `DeterministicEvaluator` (`evaluator/mvp-2+fitness/mvp-2`), using each task's own
caps.

- **Wynk:** its runtime's `ExecutionResult` is used as-is. The only change is that tokens and
  wall time are re-measured exactly like the externals (proxy usage and runner clock).
- **External systems:** the final JSON is parsed leniently (fenced, embedded in prose,
  Python-literal, or bare fields). Values are restricted to the schema fields with no type
  coercion, which matches Wynk. Each cited quote becomes evidence spans in two steps:
  1. First, an exact match using Wynk's own `runtime.spans.locate_quote`.
  2. Otherwise, a whitespace-insensitive match of each `...`-separated excerpt (8 or more
     non-whitespace chars). Every excerpt must exist in the page, and any changed character
     fails.

  This is deliberately **more lenient than Wynk's runtime**, so formatting never costs a
  competitor its evidence. How each quote matched is recorded.
- **Execution status**, with the same rule for every system:
  - `TIMEOUT` > `BUDGET_EXCEEDED` (any cap breached) > `RUNTIME_ERROR` > `INVALID_OUTPUT`
    (no parseable answer, or a schema-invalid one) > `MISSING_EVIDENCE` > `SUCCESS`.
  - A failed competitor never stops the experiment.
  - Only completed executions are skipped when a crashed run is resumed. Failures are never
    retried.

### Metrics (fixed before the full run; see `report.py`)

| Metric | Definition |
|---|---|
| **fitness** (primary) | evaluator fitness: INFEASIBLE -1; FAIL 0.7·matched + 0.1·evidence; PASS 1 + 0.1·token headroom vs the task's own cap |
| success rate | evaluator verdict PASS (all fields match, every field has valid evidence, within caps) |
| answer correct\* | all fields match ground truth, caps ignored (diagnostic) |
| evidence valid\* | fraction of fields with >= 1 valid evidence span, caps ignored |
| within caps | run within the task's token / wall-time / tool-call / retry caps |
| failure rate | no usable answer: TIMEOUT, RUNTIME_ERROR or INVALID_OUTPUT |
| latency, tokens, cost, LLM calls, tool calls | per execution (runner clock / metering proxy); `null` when unavailable |

The **strongest starter baseline** is the external system with the highest mean fitness over all
held-out tasks. Relative changes are reported only where the baseline value is > 0. 95% CIs
resample tasks (2000 resamples, runs averaged per task). The paired per-task difference is
reported alongside.

## Known fairness caveats (stated before the full run)

- **Task-specific vs. generic.** Wynk's workflow was optimised for each task class (it has one
  genome for A and one for B). Each starter is one generic workflow. That is the claim under
  test, not a confound. However, Wynk's search did see 10 train and 6 validation tasks of the
  same classes, while the starters were not tuned at all.
- **Caps are part of the task.** The token caps (A: 9000, B: 7000) and the 180 s wall cap are
  hard task constraints. Wynk was selected under them; the starters know nothing about them.
  "Answer correct\*" and "evidence valid\*" are reported with caps lifted, so correctness can be
  read separately from budget compliance.
- **Selection-time evaluator.** The frozen genomes were selected under `fitness/mvp-1`.
  `fitness/mvp-2` (the cap-relative token bonus, without wall time) was merged afterwards and is
  what scores this comparison, for every system alike. The genomes were not re-selected.
- **Tools.** The starters' web-search tools are replaced by page tools over the same frozen pages.
  Wynk's Class B workflow reads the mock API; the starters read the byte-identical `items` page.
- **Prompt.** The output contract is part of the shared query. CrewAI's `{topic}` is set to one
  generic value; its template's second agent (a markdown report writer) is not used. LlamaIndex
  uses tool-based retrieval, not a vector index, so it is "agentic RAG" in the
  agent-retrieves-documents sense.
- **Model settings.** smolagents' CodeAgent sends no `tools` field (it acts in Python code);
  the others use native function calling. Wynk passes a per-call `seed`; the starters do not
  (their defaults). LlamaIndex streams; the others do not.
- **Citation locator.** The externals' locator is more lenient than Wynk's (see above).
- **Clock.** Latency is measured from just before the workflow is built to just after it
  returns, inside each runner. Interpreter and framework import time are excluded for every
  system.

## Results: `full-r3` (2026-10-04, 240 executions, 16 held-out tasks x 3 runs x 5 systems)

All numbers come from `experiments/results/oss_baselines/full-r3/` (`report.md`, `summary.json`,
`runs.jsonl`). The run used 10 concurrent workers in shuffled order, so every system ran under
the same backend load. 720 of 721 model calls returned HTTP 200 (see the caveat below), and no
run hit a runtime error or timed out.

**Primary analysis (pre-registered rules):**

| Workflow | Success | Fitness (95% CI) | Answer correct\* | Evidence valid\* | Within caps | Failure | Latency (s) | Tokens | Cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| smolagents Starter Agent (CodeAgent) | 35% | -0.289 [-0.749, 0.171] | 100% | 100% | 35% | 0% | 85.5 | 10524 | 0.001614 |
| CrewAI Starter Research Workflow | 94% | 1.010 [0.886, 1.074] | 94% | 100% | 100% | 0% | 62.4 | 2332 | 0.000387 |
| LlamaIndex Starter Agentic RAG | 94% | 1.013 [0.888, 1.077] | 94% | 100% | 100% | 0% | 54.1 | 2097 | 0.000349 |
| LangGraph Basic ReAct Agent | 94% | 1.014 [0.889, 1.079] | 94% | 100% | 100% | 0% | 54.0 | 1953 | 0.000329 |
| **Wynk Optimized Workflow (frozen)** | 81% | 0.970 [0.811, 1.083] | 94% | 81% | 100% | 0% | 33.5 | 1221 | 0.000215 |

The strongest starter is the **LangGraph Basic ReAct Agent** (fitness 1.014). Compared with it,
Wynk has:

- fitness **-4.4%** (0.970 vs 1.014). The paired per-task difference is -0.045, with 95% CI
  [-0.114, +0.008]: Wynk is better on 13 of 16 tasks and worse on 3.
- success rate 81% vs 94%;
- the same answer correctness (94%), within-caps rate (100%) and failure rate (0%);
- **37.5% fewer tokens** per run, and fewer tokens on 16 of 16 tasks against every starter;
- **34.6% lower cost** (also lower on 16 of 16 tasks);
- **38.1% lower mean latency** (lower on 12 of 16 tasks against LangGraph).

What drives the result:

- **Class A:** every system except smolagents passes every run. Wynk has the top fitness (1.093
  vs 1.084) at about half the tokens.
- **Class B, TB-005 and TB-006:** Wynk's frozen B workflow gives the correct answer but attaches
  no evidence span to the aggregate field (`MISSING_EVIDENCE`, 9 of 24 runs). That workflow was
  selected with only a 0.33 validation pass rate.
- **Class B, TB-008:** every system except smolagents gets the arithmetic wrong.
- **smolagents CodeAgent:** it answers 48 of 48 runs correctly (it computes in Python), but
  exceeds the token cap in 31 of 48 runs, including every Class B run. Its system prompt alone is
  about 2.4k tokens and is re-sent at every step.

**Sensitivity analysis (secondary; `full-r3/sensitivity_exact_citations/`).** The tool-calling
starters' Class B evidence depends on the lenient citation locator. If their recorded outputs
are re-scored with the exact-quote rule Wynk's runtime uses (no new executions), the following
changes:

- Their success drops to 75%.
- Wynk becomes the strongest workflow on fitness: 0.970 vs 0.940, **+3.2%**. The paired CI is
  [-0.041, +0.121].

So the sign of the fitness gap depends on the citation-normalisation rule, and in **neither**
analysis does the paired CI exclude zero.

**Headline (the one conclusion the data supports):** On 16 held-out tasks with the same Gemma 4
31B model, Wynk's frozen workflow matched the answer correctness of the three tool-calling
starter workflows (94%) and stayed within every task cap. It used 37.5% fewer tokens than the
strongest starter (LangGraph Basic ReAct Agent), with fewer tokens on every task. **It did not
achieve higher deterministic fitness:** fitness was -4.4% vs that starter, and the paired 95% CI
includes zero.

### Caveats found during the run

- **Lenient locator.** Tool-calling starters needed the lenient locator for 9 Class B passes
  each (see the sensitivity analysis). Wynk's 9 evidence failures have zero spans, so the
  locator is not what explains them.
- **One retried Wynk call.** In run `wynk.TB-001.r2`, one model call exceeded the 120 s per-call
  timeout, and Wynk's client retried it. Externals have the same timeout and retry policy. The
  run passed in 144 s, under the 180 s cap. The abandoned call never returned usage, so that
  run's token total is reported as `null`. Its cap check used the 1911 tokens of the calls that
  did return; that rule is identical for every system.
- **Latency under load.** Latency was measured with 10 concurrent runs. Model time is 92–99% of
  every system's latency.
- **smolagents and the token cap.** Its token usage comes from the CodeAgent default system
  prompt. No step-count or prompt changes were made: that would no longer be the starter
  workflow.

## Run

```bash
set -a && . ./.env && set +a                       # GEMMA_BASE_URL / GEMMA_MODEL / GEMMA_API_KEY
for f in smolagents crewai llamaindex langgraph; do
  uv venv --python 3.13 .oss-venvs/$f
  uv pip install --python .oss-venvs/$f/Scripts/python.exe -r experiments/oss_baselines/requirements/$f.txt
done
P=.venv/Scripts/python.exe                          # Wynk env: pip install -e ".[dev,maf]"
$P -m experiments.oss_baselines.harness freeze      # once; refuses to overwrite
$P -m experiments.oss_baselines.harness manifest
$P -m experiments.oss_baselines.harness smoke --tasks TA-001
$P -m experiments.oss_baselines.harness run --runs 3 --workers 10
$P -m experiments.oss_baselines.harness report --out experiments/results/oss_baselines/<run>
```

Each run directory has these outputs:

- `runs.jsonl`: one line per execution, with the input task, raw final output, normalised
  answer, quotes and how they matched, evidence spans with their text, status, verdict,
  fitness, field results, caps-lifted diagnostics, telemetry, per-call proxy records, error,
  framework version, and the full `ExecutionResult`.
- `raw/<system>.<task>.r<k>/`: `input.json`, `output.json`, `stdout.log`, `stderr.log`, and
  `proxy_calls.json` with the raw model requests and responses.
- `run_meta.json`: the manifest snapshot, `frozen_wynk.json` hash, and start time.
- `summary.json` and `report.md`.
