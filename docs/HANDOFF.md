# Handoff: real MVP (ACO vs random over real Gemma + MAF)

Branch `integrate/mvp-demo`. Status: **full A + B experiment (2 optimizers x 5 seeds) runs locally
in ~17 min per class** against hosted Gemma 4 31B. Modal is no longer needed: parallel runs inside
each search make a laptop fast enough.

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
* `experiments/learning_curves.run_search(..., workers=N)` - the train runs of a round and each
  validation pass go through an order-preserving thread pool. Seeds are per run, so results are
  identical to `workers=1` (tested); only wall time changes. `EvalLog` updates are locked.
* `runtime/gemma_client.OpenAICompatibleClient` retries HTTP 429/5xx and dropped connections
  (exponential backoff, honours `Retry-After`, `GEMMA_MAX_RETRIES`, default 3). Other 4xx errors
  are raised right away.
* No `core/` contract was changed.

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev,maf]"
# .env (gitignored) - load with: set -a && . ./.env && set +a
GEMMA_BASE_URL=https://zenmux.ai/api/v1
GEMMA_MODEL=google/gemma-4-31b-it
GEMMA_API_KEY=<ZenMux key with gemma-4-31b-it in its allowed model list>   # never commit
```

Backend notes: ZenMux `google/gemma-4-31b-it` works (structured output + token usage). Latency is
2-28 s per call and grows under load (median ~10 s at 12 concurrent calls). ZenMux has no Gemma 3.
Earlier: OpenRouter `google/gemma-3-27b-it` also worked (~2 s/call); NVIDIA NIM returned 403.

## Run

```bash
.venv/Scripts/python -m experiments.run_mvp smoke                          # A-002, hand-built genome -> PASS
.venv/Scripts/python -m experiments.run_mvp experiment --task-class A --budget 100 --seeds 5 \
    --workers 4 --out experiments/results/real-gemma4/A                    # ~17 min
.venv/Scripts/python -m experiments.run_mvp experiment --task-class B --budget 100 --seeds 5 \
    --workers 4 --out experiments/results/real-gemma4/B                    # ~16 min
.venv/Scripts/python -m experiments.run_mvp final --task A-001 --out experiments/results/real-gemma4/A
```

## Changes needed for Gemma 4

* **Strict EXTRACT schema** (`runtime/prompts/templates.py`, prompt version `mvp-2`): `page_id` and
  `quote` are required and `value` is a scalar. With the old untyped `value`, Gemma 4 under
  constrained decoding wrote `"Jaan Kask own page_id: page overview, quote: \"...\""` into the value,
  so no evidence span could be located. REASON keeps `page_id`/`quote` optional (derived values).
* **Wall-time caps 45 s (A) / 30 s (B) -> 180 s** (`benchmarks/build.py`, rebuilt `tasks.json`, new
  golden benchmark hash). Only the cap values changed. With the old caps, provider latency alone
  made runs INFEASIBLE. Wall time still feeds the fitness cheapness bonus (<= 0.1), so latency
  adds a little noise to PASS fitness.

## Results (Gemma 4 31B, 5 seeds, budget 100, 2 trials/candidate) - `experiments/results/real-gemma4/`

| Class | Optimizer | final validation fitness (mean, per seed) | validation pass | train runs PASS |
|---|---|---|---|---|
| A | ACO (MMAS) | **1.078** (1.155 / 1.153 / 1.149 / 0.775 / 1.159) | **93%** | 41% |
| A | Random | 0.926 (1.163 / 1.011 / 1.165 / 0.385 / 0.906) | 73% | 28% |
| B | ACO (MMAS) | 0.584 (0.536 / 0.611 / 0.619 / 0.616 / 0.536) | 27% | 60% |
| B | Random | 0.615 (0.541 / 0.612 / 0.540 / 0.617 / 0.765) | 33% | 54% |

* **Class A:** ACO is ahead on every summary: 4/5 ACO seeds end at >= 1.149 with 100% validation
  pass, vs 2/5 random seeds. ACO also finds a passing workflow sooner (see `A/learning_curve.png`).
  Still only 5 seeds and nothing was tuned.
* **Class B:** no meaningful difference. ACO picks better workflows on train (60% vs 54% runs
  PASS), but validation is capped by two tasks the model cannot pass with any workflow:
  * B-001 (total stock): Gemma gets the sum right (507) but a computed total has no single quote
    on the page, so the evidence check fails (correct answer, FAIL). The final run of the best B
    workflow shows exactly this (`B/final_test.json`).
  * B-008 (inventory value): Gemma's arithmetic is wrong (65311.14 vs 68844.76).
  B-006 passes most of the time, so validation pass sits near 1/3 for both optimizers.
* Many proposed genomes fail instantly on options the MVP runtime does not implement yet (gather
  source `jev`, verifier `self_consistency`). Class A: ~350 of random's 500 train runs vs ~200 of
  ACO's; Class B: ~60 vs ~10. ACO's pheromone steers away from them, which is a large part of its
  Class A lead.
* Best ACO workflows:
  * A: `GATHER(fetch, sequential) -> EXTRACT(cot) -> VERIFY(schema_check, retry-1) -> SYNTHESIZE(cite_evidence) -> VERIFY(schema_check, regather)`
    -> PASS on validation task A-001 (Lisbon / 1987, both evidence spans valid, 620 tokens, 34 s).
  * B: `GATHER(api, parallel-4) -> EXTRACT(direct) -> VERIFY(schema_check, regather) -> SYNTHESIZE(cite_evidence) -> VERIFY(schema_check, regather)`
* Cost: 2,336 unique workflow runs, 3.2M tokens, about $0.45-1.28 at ZenMux list prices.

The earlier Gemma 3 27B results (Class A, 3 seeds, old prompts and caps) are kept in
`experiments/results/real/` for reference. They are not directly comparable (different model,
prompt version and benchmark hash).

Verified: runtime only receives `RuntimeTask`; evaluator alone flips verdicts (wrong value -> FAIL,
forged span hash -> FAIL, over token cap -> INFEASIBLE). All tests + ruff pass.

## Next steps

1. **Evidence for derived values (Class B):** let a REASON/aggregate fact cite the set of source
   spans it was computed from (e.g. every row summed), and teach the evaluator to accept that.
   This is what keeps B-001 at FAIL with a correct answer.
2. **Arithmetic:** route sums/products through a deterministic tool instead of the model (B-008).
3. Implement or remove the grammar options the MVP runtime rejects (`jev`, `self_consistency`) so
   search budget isn't spent on genomes that cannot run.
4. More seeds (10+) for a statistically meaningful A comparison; the runs are cheap now.
5. Optional tiny demo surface: task -> candidate workflows -> winner -> evidence -> curve.
