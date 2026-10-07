# Reading MMLU-Pro protocol-v1

`REPORT.md`, `summary.json` and `learning_curves.png` are generated. These notes are written by
hand and add context only.

Protocol-v1, the manifest and the protocol lock were committed and pushed in `22f2c11` before any
MMLU-Pro run. The protocol's canonical hash is `34f3065e…8265`. Nothing was changed after the
results: not the protocol, the fixed rule, the prompts, the ACO/random implementations
(byte-pinned by `tests/test_benchmark_adapter.py`) or the budgets.

Before freezing, one plumbing check ran 3 rows from MMLU-Pro's separate *validation* split. That
split is disjoint from the frozen test subset. The check confirmed only that the prompt shows
the question, options and instructions with nothing else, and that outputs parse to a letter.
Correctness was not looked at, and nothing was changed afterwards.

## Headline: champion validation accuracy (63 rows)

| strategy | seed 0 | seed 1 | seed 2 | mean ± std | candidates | calls (mean) | tokens (mean) | E2E s (mean) |
|---|---|---|---|---|---|---|---|---|
| fixed `DIRECT(cot)` | 0.810 (51) | 0.857 (54) | 0.841 (53) | **0.836 ± 0.024** | 1 | 147 | 61.7k | 97 |
| random (distinct) | 0.841 (53) | 0.841 (53) | 0.825 (52) | **0.836 ± 0.009** | 6 | 967 | 351.7k | 378 |
| ACO (MMAS) | 0.841 (53) | 0.857 (54) | 0.810 (51) | **0.836 ± 0.024** | 6 | 963 | 354.7k | 467 |

Numbers in parentheses are correct answers out of 63. All three strategies got 158 of 189
validation answers right, so the means are identical. Under the pre-registered rule the winner
is a **three-way tie**.

## Answers

**Does search beat the strong fixed workflow? No.** It ties on the mean.
- Seed 0: both searchers found a workflow 2 answers better than fixed.
- Seed 1: random was 1 answer worse; ACO tied.
- Seed 2: both searchers were worse (by 1 and 2 answers).

The champion's accuracy on the optimization rows is a second check that champion selection never
reads, so it is free of best-of-6 selection bias. It agrees: fixed 0.794, random 0.798, ACO 0.780
(84 rows, means over seeds).

**Does ACO beat random? No.** Identical means. ACO's spread is larger (0.024 vs 0.009). ACO
re-evaluated workflows it had already tried: 4, 3 and 4 distinct workflows in its 6 candidates.
Random is exhaustive here (6 of 6 distinct).

**What did search cost?** Compared with fixed, per strategy run:
- about 6.6× the model calls;
- 5.7× the tokens;
- 3.6–4.6× the summed workflow execution time;
- 3.9–4.8× the end-to-end time.

All of that bought no accuracy. Optimizer overhead was about 0.01 s per run and evaluator
overhead about 0.5 s. Model execution dominates everything.

## What the search space contained

There is no context column, so the contract admits exactly 6 workflows. Their results over every
evaluation in all runs:

| workflow | evaluations | mean validation | mean optimization | abstentions / failures |
|---|---|---|---|---|
| `DIRECT(cot) -> VERIFY(schema_check, retry-2)` | 8 | 0.841 | 0.793 | 29 schema failures in 1176 runs |
| `DIRECT(cot)` (fixed) | 8 | 0.821 | 0.784 | 58 no-answer in 1176 runs |
| `DIRECT(cot) -> VERIFY(schema_check, retry-1)` | 6 | 0.812 | 0.798 | 30 in 882 |
| `DIRECT(answer) -> VERIFY(schema_check, retry-2)` | 4 | 0.738 | 0.643 | 50 in 588 |
| `DIRECT(answer) -> VERIFY(schema_check, retry-1)` | 8 | 0.694 | 0.634 | 136 in 1176 |
| `DIRECT(answer)` | 5 | 0.641 | 0.617 | 109 no-answer in 735 |

- **Chain of thought is the one large effect** (about +0.18 accuracy). The fixed rule picked it
  from the task shape, before any run.
- **`VERIFY(schema_check, retry)` only re-asks when the model abstains.** The mvp-3 DIRECT prompt
  says "do not guess a field the input does not support", and Gemma then sometimes returns no
  letter. A retry recovers some of those rows. The gain is large without CoT and within noise
  with it. Search picked `cot + retry-2` as champion on 5 of 6 search runs, but on the
  selection-free optimization rows it does not beat plain CoT (0.793 vs 0.784, about one row).
- **Selection bias is visible.** ACO seed 1's best-so-far curve *falls* from 0.873 to 0.857 as
  the same workflow is re-evaluated. A best-of-6 validation score is optimistic, and fixed has
  no such inflation.

## Caveats

- **Timing load.** The 3 seeds ran as 3 concurrent processes (24 workers each) on one shared
  gateway. Compare latency and E2E within a seed.
- **Cost.** Not reported: the runtime has no authoritative price, and no pricing policy was
  supplied.
- **Test split.** The 63 test rows were never executed (`test_runs: 0`).
- **Evaluator.** The evaluator is stricter than MMLU-Pro's official regex extraction: the answer
  must be exactly one letter. This applies identically to every strategy.
- **Scale.** 63 validation rows × 3 seeds: one answer is 0.016. Nothing here supports a claim of
  statistical superiority in any direction.
