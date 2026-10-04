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
