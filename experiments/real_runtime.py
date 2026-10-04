"""Glue between the frozen benchmark (Track B) and the MAF runtime (Track A).

* ``SnapshotPageSource`` / ``MockApiSource`` satisfy ``runtime.sources.PageSource`` /
  ``ApiSource`` on top of ``SnapshotStore`` / ``MockAPI``. Page ids and bytes are passed through
  unchanged, so runtime evidence spans verify against the evaluator's view of the snapshot.
* ``real_evaluate_fn`` = ``WorkflowRunner.run_sync`` -> ``ExecutionResult`` ->
  ``DeterministicEvaluator`` via ``make_evaluate_fn``. The runtime only ever receives
  ``RuntimeTask``; the ``TaskSpec`` map stays on the evaluator side of the closure.

Runs are cached by ``RunKey.run_id`` (genome, task, trial, seed, versions): the backend is called
at temperature 0 with that seed, so an identical key is the same run. Transient model failures
are never cached.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from benchmarks.loader import benchmark_hash, load_task_specs
from benchmarks.mock_api import MockAPI
from benchmarks.snapshot_store import SnapshotStore
from core.genome import Genome
from core.payloads import Page
from core.results import EvaluatedRun, ExecutionResult, FailureKind
from core.task_spec import RuntimeTask
from evaluation.gate import DeterministicEvaluator
from experiments.learning_curves import EvaluateFn, make_evaluate_fn
from runtime.gemma_client import ModelClient
from runtime.runner import WorkflowRunner
from runtime.sources import SourceError


class SnapshotPageSource:
    """``PageSource`` over the frozen snapshot pages (``fetch``)."""

    def __init__(self, store: SnapshotStore) -> None:
        self.store = store

    def list_page_ids(self, snapshot_id: str) -> list[str]:
        if not self.store.has_snapshot(snapshot_id):
            raise SourceError(f"unknown snapshot: {snapshot_id}")
        return [p.page_id for p in self.store.pages(snapshot_id).pages]

    def read_page(self, snapshot_id: str, page_id: str) -> Page:
        page = self.store.page(snapshot_id, page_id)
        if page is None:
            raise SourceError(f"page not found: {snapshot_id}/{page_id}")
        return page


class MockApiSource:
    """``ApiSource`` over the mock JSON API (``api``). An unfiltered endpoint response is
    byte-identical to the snapshot page of the same name, so it keeps that page id."""

    def __init__(self, api: MockAPI) -> None:
        self.api = api

    def list_endpoints(self, snapshot_id: str) -> list[str]:
        if not self.api.store.has_snapshot(snapshot_id):
            raise SourceError(f"unknown snapshot: {snapshot_id}")
        return sorted(self.api.store.records(snapshot_id))

    def call(self, snapshot_id: str, endpoint: str) -> Page:
        try:
            content = self.api.get(snapshot_id, endpoint)
        except KeyError as exc:
            raise SourceError(str(exc)) from exc
        return Page(page_id=endpoint, source_ref=f"api://{snapshot_id}/{endpoint}", content=content)


def build_runner(model: ModelClient, store: SnapshotStore | None = None) -> WorkflowRunner:
    store = store or SnapshotStore()
    return WorkflowRunner(
        model=model,
        benchmark_hash=benchmark_hash(store=store),
        pages=SnapshotPageSource(store),
        api=MockApiSource(MockAPI(store)),
    )


class RunCache:
    """Thread-safe ``run_id -> ExecutionResult`` cache, optionally persisted as JSONL."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._runs: dict[str, ExecutionResult] = {}
        if path and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = ExecutionResult.model_validate_json(line)
                self._runs[r.key.run_id] = r

    def get(self, run_id: str) -> ExecutionResult | None:
        with self._lock:
            return self._runs.get(run_id)

    def put(self, result: ExecutionResult) -> None:
        if result.failure is not None and result.failure.kind is FailureKind.MODEL_ERROR:
            return  # transient backend failure: never reuse
        with self._lock:
            self._runs[result.key.run_id] = result
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(result.model_dump_json() + "\n")


def real_evaluate_fn(
    runner: WorkflowRunner,
    evaluator: DeterministicEvaluator | None = None,
    cache: RunCache | None = None,
) -> EvaluateFn:
    cache = cache or RunCache()

    def run_workflow(genome: Genome, task: RuntimeTask, trial: int, seed: int) -> ExecutionResult:
        from core.results import RunKey

        key = RunKey(
            genome_hash=genome.genome_hash,
            task_id=task.id,
            trial=trial,
            seed=seed,
            versions=runner.versions(),
        )
        hit = cache.get(key.run_id)
        if hit is not None:
            return hit
        result = runner.run_sync(genome, task, trial=trial, seed=seed)
        cache.put(result)
        return result

    return make_evaluate_fn(run_workflow, evaluator or DeterministicEvaluator(), load_task_specs())


def run_summary(run: EvaluatedRun) -> dict:
    """Plain-data view of one evaluated run for printing / JSON."""
    ex, ev = run.execution, run.evaluation
    answer = ex.answer
    return {
        "verdict": ev.verdict.value,
        "fitness": ev.fitness,
        "answer": answer.values if answer else None,
        "evidence": [
            {"field": fe.field, "spans": [s.model_dump(mode="json") for s in fe.spans]}
            for fe in (answer.evidence if answer else ())
        ],
        "field_results": [fr.model_dump(mode="json") for fr in ev.field_results],
        "tokens": ex.budget_usage.tokens,
        "wall_time_s": round(ex.budget_usage.wall_time_s, 2),
        "tool_calls": ex.budget_usage.tool_calls,
        "retries": ex.budget_usage.retries,
        "failure": ex.failure.model_dump(mode="json") if ex.failure else None,
        "model_calls": ex.metrics.model_calls,
        "stages": [
            {"kind": t.kind.value, "status": t.status.value, "tokens": t.usage.tokens}
            for t in ex.stage_trace
        ],
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)
