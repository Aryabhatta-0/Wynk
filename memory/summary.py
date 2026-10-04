"""Tiny deterministic, human-readable summary of a class's workflow memory (no LLM).

python -m memory.summary B
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from memory.models import WorkflowMemory
from memory.store import DEFAULT_DIR, WorkflowMemoryStore

STRONG_EDGES = 4

_BANNER = {
    "real": "",
    "synthetic": "  [SYNTHETIC objective - not a benchmark result]",
    "TEST/FIXTURE": "  [TEST/FIXTURE data - not a learned result]",
}


def render(memory: WorkflowMemory) -> str:
    k = memory.key
    lines = [f"Class {k.task_class} memory:{_BANNER[memory.source]}"]
    if memory.workflows:
        best = memory.workflows[0]
        lines += [
            f"Best workflow: {best.path}",
            f"Validation fitness: {best.validation_fitness:.3f}",
            f"Pass rate: {best.pass_rate:.2f}",
            f"Tokens (mean): {best.mean_tokens:.0f}   Wall time (mean): "
            f"{best.mean_wall_time_s:.2f}s",
        ]
        for i, w in enumerate(memory.workflows[1:], start=2):
            lines.append(
                f"#{i}: {w.path}  (fitness {w.validation_fitness:.3f}, pass {w.pass_rate:.2f})"
            )
    else:
        lines.append("Best workflow: (none recorded)")
    inner = [
        e
        for e in memory.pheromones
        if "START" not in (e.src, e.dst) and "END" not in (e.src, e.dst)
    ]
    strong = sorted(inner, key=lambda e: (-e.tau, e.label))[:STRONG_EDGES]
    lines.append("Strong edges:")
    lines += [f"  {e.label}  (tau {e.tau:.3f})" for e in strong] or ["  (none)"]
    lines.append(
        f"Runs merged: {memory.runs_merged}   Updated: {memory.updated_at}   "
        f"Optimizer: {k.optimizer_version}   Grammar: {k.grammar_version}"
    )
    lines.append(
        f"Benchmark: {k.benchmark_hash[:12]}   Model: {(k.model_hash or '-')[:12]}   "
        f"Evaluator: {k.evaluator_version}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Print the learned workflow memory for a task class.")
    p.add_argument("task_class")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)
    memory = WorkflowMemoryStore(args.dir).load(args.task_class)
    if memory is None:
        print(
            f"No workflow memory for class {args.task_class} in {args.dir}.\n"
            "Create one with: python -m memory.warm_start --synthetic --task-class <A|B>"
        )
        return 1
    print(render(memory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
