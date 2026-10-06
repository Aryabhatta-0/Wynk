# MuSiQue-Answerable results (real model)

There is one directory per frozen protocol. Every protocol uses the same 60-row manifest
(`experiments/musique_frozen/manifest.json`) and the same 24 / 18 / 18 split. The 18 test rows
are never executed.

| directory | protocol | fixed-baseline rule | frozen in | status |
|---|---|---|---|---|
| `protocol-v1/` | `protocol-v1.json` (canonical hash `00536ebc…4422`) | `fixed_shortest/1` → `DIRECT(answer)`, which never sees context | `2733834` | kept unchanged |
| `protocol-v2/` | `protocol-v2.json` (canonical hash `7bc15caa…af9d`) | `fixed_context/1` → `GATHER → EXTRACT → SYNTHESIZE` when the dataset has context columns | see `musique_frozen/protocols.lock.json` | headline |

Protocol-v2 differs from v1 only in the fixed-baseline rule; `tests/test_musique_protocols.py`
checks this.

Each directory holds:

- `REPORT.md`: the generated report.
- `NOTES.md`: hand-written interpretation.
- `learning_curves.png`: validation token F1 against candidates, model calls, tokens, execution time, end-to-end time and cost.
- `official_crosscheck.json`: the official MuSiQue evaluator's answer F1 compared with Wynk's token F1.
- `champion_validation_predictions.jsonl`: each champion's validation predictions, in the official format.
- `summary.json`: identity, plan, the per-run summary and curves, the multi-seed aggregates, and the SHA-256 of the raw artifact.
- `experiment.json.gz`: the canonical raw artifact, stored once and compressed deterministically.

`load_compact` verifies both hashes of `experiment.json.gz` before using it. The per-seed `runs/` records
are regenerable inputs and are not committed.

The protocol-v1 report and notes were written when its files lived in `musique-real/` and named
`experiment.json`. That artifact is now `protocol-v1/experiment.json.gz`. Its uncompressed bytes
are exactly the committed `experiment.json` (SHA-256 `00dbb370…2888`).
