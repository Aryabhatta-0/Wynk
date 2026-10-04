"""Run-store contract (storage-agnostic; DuckDB implementation comes later).

Core models are plain pydantic and know nothing about persistence. Identity is ``RunKey``
(genome hash + task + trial/seed + model/prompt/benchmark/compiler/grammar versions), so a
future cache can look up an already-executed run deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core.genome import Genome
from core.results import EvaluatedRun, RunKey
from core.task_spec import TaskSpec


class RunStore(Protocol):
    def save_task(self, task: TaskSpec) -> None: ...

    def save_genome(self, genome: Genome) -> None:
        """Persist the canonical genome under its ``genome_hash``."""
        ...

    def save_run(self, run: EvaluatedRun) -> None:
        """Idempotent on ``run.run_id``: re-saving an identical run is a no-op."""
        ...

    def get_run(self, run_id: str) -> EvaluatedRun | None: ...

    def lookup(self, key: RunKey) -> EvaluatedRun | None:
        """Cache hit for an identical (genome, task, trial, seed, versions) run."""
        ...

    def runs_for(
        self, *, genome_hash: str | None = None, task_id: str | None = None
    ) -> Sequence[EvaluatedRun]: ...


class InMemoryRunStore:
    """Reference implementation for tests and for tracks developing in isolation."""

    def __init__(self) -> None:
        self._runs: dict[str, EvaluatedRun] = {}
        self._tasks: dict[str, TaskSpec] = {}
        self._genomes: dict[str, Genome] = {}

    def save_task(self, task: TaskSpec) -> None:
        self._tasks[task.id] = task

    def save_genome(self, genome: Genome) -> None:
        self._genomes[genome.genome_hash] = genome

    def save_run(self, run: EvaluatedRun) -> None:
        self._runs.setdefault(run.run_id, run)

    def get_run(self, run_id: str) -> EvaluatedRun | None:
        return self._runs.get(run_id)

    def lookup(self, key: RunKey) -> EvaluatedRun | None:
        return self._runs.get(key.run_id)

    def runs_for(
        self, *, genome_hash: str | None = None, task_id: str | None = None
    ) -> list[EvaluatedRun]:
        return [
            r
            for r in self._runs.values()
            if (genome_hash is None or r.execution.genome_hash == genome_hash)
            and (task_id is None or r.execution.task_id == task_id)
        ]
