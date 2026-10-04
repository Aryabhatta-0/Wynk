"""Wynk - Optimized Workflow (frozen genome), run exactly like the external runners.

Executes the FROZEN genome through the unchanged Wynk runtime (``WorkflowRunner`` -> MAF) in the
main Wynk environment, as a subprocess with the same protocol, timeout and metering proxy as
every external framework. Receives only the ``RuntimeTask`` (no ground truth).

``final`` is the runtime's own ``ExecutionResult`` (answer + evidence spans built by Wynk).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

# framework imports happen before timing starts (not part of the workflow latency)
from benchmarks.heldout import HELDOUT_DIR, HELDOUT_SNAPSHOTS  # noqa: E402
from benchmarks.loader import benchmark_hash  # noqa: E402
from benchmarks.mock_api import MockAPI  # noqa: E402
from benchmarks.snapshot_store import SnapshotStore  # noqa: E402
from core.genome import Genome  # noqa: E402
from core.task_spec import RuntimeTask  # noqa: E402
from experiments.real_runtime import MockApiSource, SnapshotPageSource  # noqa: E402
from runtime.gemma_client import GemmaConfig, OpenAICompatibleClient  # noqa: E402
from runtime.runner import WorkflowRunner  # noqa: E402


def framework() -> dict[str, str]:
    return {"name": "wynk", "version": "runtime@" + _git_head()}


def _git_head() -> str:
    head = ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            return (ROOT / ".git" / ref[5:]).read_text().strip()[:12]
        return ref[:12]
    except OSError:
        return "unknown"


def run(inp: dict, _pages: _common.PageTools):
    m = inp["model"]
    w = inp["wynk"]
    client = OpenAICompatibleClient(
        GemmaConfig(
            base_url=m["base_url"],
            model=m["model"],
            api_key=m["api_key"],
            timeout_s=m["timeout_s"],
            max_retries=m["max_retries"],
        )
    )
    store = SnapshotStore(HELDOUT_SNAPSHOTS)
    runner = WorkflowRunner(
        model=client,
        benchmark_hash=benchmark_hash(HELDOUT_DIR, store),
        pages=SnapshotPageSource(store),
        api=MockApiSource(MockAPI(store)),
    )
    genome = Genome.from_stages(w["genome_stages"])
    assert genome.genome_hash == w["genome_hash"], "frozen genome hash mismatch"
    task = RuntimeTask.model_validate(inp["task"])
    result = runner.run_sync(genome, task, trial=w["trial"], seed=w["seed"])
    return result.model_dump(mode="json"), {
        "stage_trace": [t.model_dump(mode="json") for t in result.stage_trace]
    }


if __name__ == "__main__":
    _common.main(framework, run)
