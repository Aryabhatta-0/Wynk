# Reading protocol-v2

`REPORT.md` is generated; these notes are written by hand and add context only. Protocol-v2 was
committed and pushed in `da0c9a3` before any v2 run. Its canonical hash is `7bc15caa…af9d`, and
`protocols.lock.json` holds the full value. Nothing was tuned after v1 or v2 results. All three
strategies were re-run under v2; no v1 measurement appears in the v2 tables.

## Headline: validation token F1 of each strategy's champion

| strategy | seed 0 | seed 1 | seed 2 | mean ± std | candidates | model calls (mean) | tokens (mean) | strategy E2E s (mean) |
|---|---|---|---|---|---|---|---|---|
| fixed (`GATHER → EXTRACT → SYNTHESIZE`) | 0.657 | 0.676 | 0.676 | **0.670 ± 0.011** | 1 | 84 | 116k | 33 |
| random | 0.606 | 0.676 | 0.676 | 0.652 ± 0.041 | 6 | 561 | 429k | 213 |
| ACO (MMAS) | 0.620 | 0.676 | 0.676 | 0.657 ± 0.032 | 6 | 538 | 373k | 283 |

The official MuSiQue answer F1 (answer + aliases, `evaluate_v1.0.py`) equals Wynk's token F1
for all nine champions.

## Answers to the questions

**Does search beat a reasonable fixed workflow? No, not on this run.**
- The plain retrieval chain had the highest mean validation F1.
- Neither searcher beat it on any seed.
- Seed 0: neither random nor ACO ever matched it. Their best scores were 0.606 and 0.620, against fixed's 0.657.
- Seeds 1 and 2: both reached the same 0.676, but later and at a higher price.

**When did the searchers match the fixed workflow's score?**

| seed | random | ACO | fixed |
|---|---|---|---|
| 0 | never | never | — |
| 1 | candidate 5 (370k tokens, 388 calls) | candidate 5 (267k tokens, 430 calls) | 1 candidate, 116k tokens, 84 calls |
| 2 | candidate 1 (127k tokens, 126 calls) | candidate 3 (229k tokens, 235 calls) | 1 candidate, 116k tokens, 84 calls |

**Does ACO beat random? No.**
- The means are 0.657 and 0.652, and the per-seed champions are identical except seed 0 (0.620 vs 0.606, a one-question difference).
- ACO used about 13% fewer tokens on average, but it had higher latency and end-to-end time.
- Its seed-2 run had a heavy p95 latency tail (84.8 s), which is gateway behaviour rather than an algorithm property.
- None of this supports a claim that ACO is better.

**How large is the seed variance? Large relative to the differences.**
- The validation split has 18 rows, so one question moves F1 by roughly 0.02–0.06.
- Three seeds and 18 rows support no claim of statistical superiority in any direction.

**What does the extra search cost?** Each searcher spent 6 candidate evaluations. Compared with the fixed workflow that is:
- about 6.5× the model calls;
- 3.2–3.7× the tokens;
- 6.4–8.5× the strategy end-to-end time;

all for no gain in validation F1. Optimizer overhead was about 0.06 s per strategy and evaluator overhead about 0.23 s. Model and workflow execution dominates everything.

## Reproducibility observations

- Under the same seeds, v2's random and ACO proposed exactly the same candidates in the same order as in v1. The proposals depend only on the seed and the frozen search implementations.
- Because the protocol identity changed, every v2 run_id differs from v1.
- The same backend at temperature 0, with the same request seeds, still returned slightly different per-candidate F1 on some candidates and slightly different token counts. Champion F1 values reproduced exactly.
- Backend non-determinism is therefore real but small here. Byte-identical reproduction is only claimed for deterministic backends.

## Caveats

- **Timing load:** the three seeds ran as three concurrent processes (24 workers each) against one shared gateway, the same as in v1. Compare latency and E2E within a seed.
- **Cost:** not reported. The runtime has no authoritative price for this gateway, and no pricing policy was supplied (#27 owns pricing).
- **Test split:** the 18 test rows were never executed (`test_runs: 0`). Final test reporting belongs to promotion (#25).
- **Generality:** this is one task (MuSiQue-Answerable), one model and a short search budget. It shows that on this setup, 6-candidate search does not beat the plain retrieval chain. It does not show that search never helps.
