# ACO Agent Workflow Optimizer

Phase 0 = the contract/framework foundation (no ACO, benchmarks, UI or real model serving yet).

* `docs/ARCHITECTURE_CONTRACTS.md` - frozen contracts and module boundaries
* `docs/PARALLEL_IMPLEMENTATION.md` - how two developers split the next phase
* `docs/RUNTIME_MVP.md` - Track B runtime: run a genome end-to-end through MAF

```bash
pip install -e ".[dev]"      # pydantic, networkx, pytest, ruff
python -m pytest
python -m ruff check . && python -m ruff format --check .
pip install -e ".[dev,maf]"  # adds Microsoft Agent Framework: enables the MAF/e2e tests
```

## MVP benchmark + optimizers (Track B)

* `benchmarks/` - frozen Class A (fact sheets) / Class B (mock JSON API) snapshots, `tasks.json`, `splits.json`
  (regenerate with `python -m benchmarks.build`; pinned by a golden hash in `tests/test_benchmarks_mvp.py`)
* `evaluation/` - deterministic schema / matcher / evidence checks, PASS/FAIL/INFEASIBLE, shaped fitness
* `optimizers/` - `RandomSearch` and `MMASACO` (edge pheromone, shared grammar/constraints)
* `experiments/` - `run_experiment(evaluate_fn, train, val, ...)`; plug in the real runtime with
  `make_evaluate_fn(run_workflow, DeterministicEvaluator(), load_task_specs())`

```bash
python -m experiments.learning_curves --synthetic --task-class B   # SYNTHETIC objective demo -> experiments/results/
```
