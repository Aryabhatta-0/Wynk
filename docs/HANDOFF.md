# Handoff: real MVP (ACO vs random over real Gemma + MAF)

Branch `integrate/mvp-demo`. Status: **real end-to-end MVP works on Class A.** Paused mid-way through
moving the experiment onto Modal (nothing Modal-related is written yet).

## What works

```
RuntimeTask -> optimizer proposes Genome -> WorkflowRunner (MAF) -> ExecutionResult
            -> DeterministicEvaluator (PASS/FAIL/INFEASIBLE + fitness) -> optimizer.observe
```

* `experiments/real_runtime.py` - the only glue: `SnapshotPageSource` / `MockApiSource` (benchmark
  snapshots -> runtime `PageSource` / `ApiSource`, page ids + bytes unchanged so evidence spans
  validate), `build_runner`, `real_evaluate_fn` (uses the existing `make_evaluate_fn`), `RunCache`
  (by `RunKey.run_id`; MODEL_ERROR results are never cached).
* `experiments/run_mvp.py` - CLI: `smoke`, `experiment`, `final`. Logs every evaluation to
  `evaluations.jsonl` (optimizer, seed, split, evaluation #, genome hash, fitness, verdict, best-so-far).
* `experiments/report.py` - x-axis is now "Training workflow evaluations" (validation runs are not
  charged to the budget).
* No `core/` contract was changed.

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev,maf]"
export GEMMA_BASE_URL=https://openrouter.ai/api/v1
export GEMMA_MODEL=google/gemma-3-27b-it
export GEMMA_API_KEY=<your OpenRouter key>      # never commit
```

Backend notes: OpenRouter `google/gemma-3-27b-it` works (structured output + token usage, ~2 s/call).
NVIDIA NIM Gemma returned HTTP 403 for our key. No local server was running.

## Run

```bash
.venv/Scripts/python -m experiments.run_mvp smoke                              # A-002, hand-built genome -> PASS
.venv/Scripts/python -m experiments.run_mvp experiment --budget 100 --seeds 3  # ~16 min on a laptop
.venv/Scripts/python -m experiments.run_mvp final --task A-001                 # best ACO genome on validation
```

## Results so far (Class A, 3 seeds, budget 100, 2 trials/candidate) - committed in `experiments/results/real/`

| | final validation fitness (mean, per seed) | validation pass | train runs PASS |
|---|---|---|---|
| ACO (MMAS) | 1.175 (1.172 / 1.177 / 1.176) | 100% | 47% |
| Random | 1.130 (1.178 / 1.035 / 1.177) | 89% | 39% |

ACO's lead comes from one bad random seed; 3 seeds is not statistically strong. Class A is easy for
Gemma 27B, so both saturate. Not tuned. Best ACO workflow:
`GATHER(fetch, parallel-4) -> EXTRACT(direct) -> SYNTHESIZE(cite_evidence) -> VERIFY(evidence_span, retry-1)`
-> PASS on validation task A-001 (Lisbon / 1987, valid evidence span, 611 tokens, 5.3 s, 3 tool calls).

Verified: runtime only receives `RuntimeTask`; evaluator alone flips verdicts (wrong value -> FAIL,
forged span hash -> FAIL, over token cap -> INFEASIBLE). All tests + ruff pass.

## Known issues

* **Class B mostly FAILs** (not run as an experiment yet). Gemma often quotes whole JSON blocks as
  evidence (long output, can hit `MAX_OUTPUT_TOKENS = 1024` in `runtime/executors/gemma_stages.py`
  -> "model output is not a JSON object"), and the quotes contain escaped `\n` so they don't locate
  verbatim in the page. Real result, not faked; prompt work would be needed to make B interesting.
* Runs are slow when sequential (~5 s Class A, ~20 s Class B per workflow run). Fix below.
* Local machine quirk (original dev only): shell writes under the OneDrive folder were sometimes
  discarded; irrelevant on a normal checkout.

## Next steps (planned, not started)

1. **Parallelise runs**: in `experiments/learning_curves.run_search` the per-genome train runs (line
   ~142) and validation runs (~115) are list comprehensions - map them over a `ThreadPoolExecutor`
   (order-preserving; seeds are per-run so results stay deterministic). Lock the counter update in
   `run_mvp.EvalLog.wrap` when doing this.
2. **Retry 429/5xx** in `runtime/gemma_client.OpenAICompatibleClient._post` (2-3 tries, backoff) so
   rate limits don't turn into MODEL_ERROR FAILs under concurrency.
3. **Modal** (user has ~$3 credit; `modal` 1.4.3 works, `pip install modal==1.4.3`): add
   `experiments/modal_app.py` - one cheap CPU container (`cpu=0.25`) per (class, optimizer, seed),
   image = debian_slim + pydantic, networkx, agent-framework-core==1.20.0, matplotlib + repo via
   `add_local_dir` (ignore `.venv`, `.git`, `experiments/results`), key via
   `modal.Secret.from_dict`. Each container returns its `run_search` dict + evaluation rows +
   genomes; the local entrypoint writes `results.json`, the plot, best genome. 20 containers
   (A+B x 2 optimizers x 5 seeds) should cost well under $0.10 of Modal CPU; Gemma tokens are
   billed by OpenRouter (estimate < $1). No GPU needed - nothing is trained; "training" = ACO search.
4. Optional tiny demo surface: task -> candidate workflows -> winner -> evidence -> curve.
