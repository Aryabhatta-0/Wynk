# How to read this result

The numbers are in `REPORT.md` (generated) and `experiment.json` (the canonical artifact).
These notes are written by hand and add context only. Nothing here was tuned after the run.

## What was frozen before any strategy ran

The protocol, manifest and runner were committed in `2733834`, before any MuSiQue result existed.
Frozen were:

- the protocol: sampling, split, contract, the `token_f1` config with `pass_threshold` 0.5, the model and its exact hash, the budget, the seeds and the trials;
- the manifest: provenance, file SHA-256, selected ids and hashes;
- the runner.

Nothing was changed after results were seen.

## Findings

1. **The pre-registered fixed baseline scores 0, and that says more about the baseline rule than about search.**
   - The rule `fixed_shortest/1` picks the shortest admissible workflow, which is `DIRECT(answer)`.
   - DIRECT never sees the context paragraphs (they are context columns, read only by GATHER).
   - The contract's instructions say to answer "using only the context paragraphs".
   - So Gemma abstains (`{"answer": {}}`) on every row. One raw call confirmed this; it is not a parsing failure.
   - A retrieval baseline such as `GATHER → EXTRACT → SYNTHESIZE` would be the stronger and fairer fixed comparator. It would have to be pre-registered as a new experiment, not swapped in here.
2. **Random and ACO are statistically indistinguishable here.**
   - Champion validation token F1, mean ± std: random 0.652 ± 0.041, ACO 0.657 ± 0.032.
   - Two of the three seeds tie at exactly 0.676 for both strategies.
   - The validation split has 18 rows, so one question moves F1 by about 0.03–0.06.
   - Six candidates per strategy is too short a run to show learning.
3. **Resource axes reorder the strategies (see `learning_curves.png`).**
   - Seed 2: random reached its best (0.676) at candidate 1, after 127k tokens. ACO needed 3 candidates and 228k tokens.
   - Seed 1: both reached 0.676 at candidate 5. ACO used 267k tokens, random used 370k.
   - Seed 0: ACO ended higher (0.620 vs 0.606), but got there at 333k tokens versus random's 226k.
4. **Both searchers spend budget on no-context DIRECT variants.**
   - These score F1 0 for the same abstention reason as the fixed baseline.
   - ACO re-proposed `DIRECT → VERIFY` in seed 0. That is consistent with flat early pheromone and only 6 candidates.

## Caveats

- **E2E wall time and latency:** the three seeds ran as three concurrent processes (24 workers each) against one shared gateway. Gateway latency was bimodal (5–65 s in a pre-run probe). Compare these timings within a seed, not as absolute performance.
- **Cost:** not reported. The runtime has no authoritative price for this gateway, and no pricing policy was supplied (#27 owns pricing).
- **Test split:** the 18 test rows were never executed (`test_runs: 0`). Final test reporting belongs to promotion (#25).
- **Official cross-check:** MuSiQue's official answer F1 (answer + aliases) equals Wynk's `token_f1` (answer only) for every champion. EM is lower because the official EM counts only exact strings.
