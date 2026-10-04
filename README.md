# ACO Agent Workflow Optimizer

Phase 0 = the contract/framework foundation (no ACO, benchmarks, UI or real model serving yet).

* `docs/ARCHITECTURE_CONTRACTS.md` - frozen contracts and module boundaries
* `docs/PARALLEL_IMPLEMENTATION.md` - how two developers split the next phase

```bash
pip install -e ".[dev]"      # pydantic, networkx, pytest, ruff
python -m pytest
python -m ruff check . && python -m ruff format --check .
pip install -e ".[maf]"      # optional: enables tests/test_maf_integration.py
```
