# External benchmark matrix: fixed vs random vs ACO

The two tables below are copied verbatim from `TABLES.md`, which
`python -m experiments.external.matrix` generates from the committed, SHA-256-verified
`experiment.json.gz` of each benchmark; a test checks that they match. Everything else here is
written by hand.

**Status of each benchmark:**
- MuSiQue protocol-v2 is **immutable prior evidence**: the PR #39 artifact, read unchanged and not rerun.
- MMLU-Pro protocol-v1 is new. Its protocol and manifest were frozen in `22f2c11`, before any MMLU-Pro run.

**Setup shared by both benchmarks:**
- model `google/gemma-4-31b-it`, exact model_hash `e2cd2c74…0b6e`, prompts `mvp-3`;
- a 6-candidate budget, 1 trial and seeds 0/1/2;
- the same per-run limits and the unchanged `random_distinct/1` and `aco_mmas/1` implementations.

**Fixed baselines** were chosen from the task shape only:
- MuSiQue: `fixed_context/1`, the retrieval chain;
- MMLU-Pro: `fixed_reasoning/1`, `DIRECT(cot)`. On MuSiQue this rule picks the same retrieval chain.

## Benchmark matrix

| Benchmark | Task family | Fixed | Random | ACO | Calls (F / R / A) | Tokens (F / R / A) | E2E s (F / R / A) | Winner |
|---|---|---|---|---|---|---|---|---|
| MuSiQue-Answerable v1.0 dev (token F1, protocol-v2) | multi-hop extractive QA over provided paragraphs | 0.670 ± 0.011 | 0.652 ± 0.041 | 0.657 ± 0.032 | 84 / 561 / 538 | 115772 / 429023 / 373114 | 33 / 213 / 283 | fixed |
| MMLU-Pro (test) (accuracy, protocol-v1) | context-free multiple-choice reasoning (10-way classification) | 0.836 ± 0.024 | 0.836 ± 0.009 | 0.836 ± 0.024 | 147 / 967 / 963 | 61698 / 351724 / 354668 | 97 / 378 / 467 | aco, fixed, random |

Scores are the mean ± sample std over 3 seeds of the champion's VALIDATION score. Calls, tokens
and E2E are means per strategy run. Winner is the pre-registered rule (highest mean; ties listed
jointly). Test splits were never executed. Cost is not reported: the runtime has no
authoritative pricing.

## Search characteristics

| Benchmark | context columns | admissible workflows | candidate budget | champion on optimization rows (F / R / A) | tokens/example (F / R / A) | exec s (F / R / A) | random-fixed | ACO-fixed | ACO-random | search tokens x fixed (R / A) |
|---|---|---|---|---|---|---|---|---|---|---|
| MuSiQue-Answerable v1.0 dev | yes | 49092 | 6 | 0.705 / 0.641 / 0.653 | 2756 / 1702 / 1481 | 487 / 2364 / 2827 | -0.017 | -0.012 | +0.005 | 3.7x / 3.2x |
| MMLU-Pro (test) | no | 6 | 6 | 0.794 / 0.798 / 0.780 | 420 / 399 / 402 | 1301 / 4697 / 5959 | +0.000 | +0.000 | +0.000 | 5.7x / 5.7x |

"Champion on optimization rows" scores each champion on rows that champion selection never
reads. It is a check that a best-of-6 validation pick cannot inflate.

## Answers

**1. Does search beat a strong fixed workflow? Not on either benchmark.**
- MuSiQue: fixed had the highest mean (0.670 vs 0.652 / 0.657). On the selection-free
  optimization rows it is clearly ahead (0.705 vs 0.641 / 0.653).
- MMLU-Pro: all three got exactly 158 of 189 validation answers right, a three-way tie. On the
  optimization rows fixed sits between the two searchers (0.794 vs 0.798 / 0.780).
- Per seed, the searchers beat fixed in 2 of 12 (strategy, seed) comparisons (both on MMLU-Pro
  seed 0), tied in 5 and lost in 5.

**2. Does ACO beat random? No.**
- The differences are +0.005 (MuSiQue) and 0.000 (MMLU-Pro), well inside one question's worth
  of noise (1/18 and 1/63 of the validation split).
- On MMLU-Pro, ACO re-evaluated workflows it had already tried (3–4 distinct of 6), while
  random covered the whole 6-workflow space.

**3. What extra resources did search consume?** Measured, per strategy run, against fixed:

| | model calls | tokens | summed execution time | end-to-end time |
|---|---|---|---|---|
| MuSiQue | 6.4–6.7× | 3.2–3.7× | 4.9–5.8× | 6.5–8.6× |
| MMLU-Pro | 6.6× | 5.7× | 3.6–4.6× | 3.9–4.8× |

That spend bought no quality. Optimizer and evaluator overheads are negligible (under 1 s per
run); model execution is the whole cost.

**4. On what task characteristics does search appear useful?** On this evidence, nowhere yet.
These are observations from two benchmarks, not rules:
- **When a strong default exists, search cannot add much.** On both tasks one obvious design
  decision carries almost all of the gain. For context-dependent QA that is reading the context
  (the retrieval chain). For context-free reasoning it is chain of thought (+0.18 over direct
  answering on MMLU-Pro). A shape-based rule picks it for free, and the remaining
  workflow-structure effects (VERIFY retries, extraction variants) are within noise.
- **When the space is small, the choice is between strategies that cost the same.** MMLU-Pro's
  contract admits 6 workflows. A 6-candidate budget is exhaustive, so random search is an
  enumeration and ACO has nothing to learn.
- **When the space is large, 6 candidates is too few.** MuSiQue's contract admits 49,092
  workflows. Six candidates is a tiny sample: neither searcher reliably even found the plain
  retrieval chain's quality.
- **Selection bias favours search on validation.** A best-of-N validation score is optimistic
  (ACO's MMLU-Pro seed-1 curve drops from 0.873 to 0.857 when the same workflow is re-run). This
  bias favours search, and search still did not win.

So search would have to earn its cost on tasks with all three of:
- no strong shape-implied default;
- real interactions between workflow choices;
- a candidate budget that is large relative to the noise of a validation estimate.

Neither benchmark here has that profile.

**Not claimed.** Two benchmarks, one model, 6 candidates, 3 seeds and 18 / 63 validation rows
cannot establish that search never helps, or anything general about ACO. The adapter
(`experiments/external/adapter.py`) is built so that HotpotQA, LongBench, BFCL or GAIA can be
added the same way: a source pin, a sanitizer, an evaluator, a stratum and a fixed rule.
