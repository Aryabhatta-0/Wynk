"""Load the frozen MVP benchmark: TaskSpecs (offline), runtime views, splits, benchmark hash.

``TaskSpec`` (with ground truth) must only be handed to ``evaluation/``. Everything else - the
optimizer, the runtime, the experiment harness - works with ``runtime_tasks`` (``RuntimeTask``).
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.snapshot_store import SnapshotStore
from core.canonical import canonical_hash
from core.task_spec import RuntimeTask, TaskClass, TaskSpec

BENCH_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "validation")


def load_task_specs(bench_dir: Path = BENCH_DIR) -> dict[str, TaskSpec]:
    raw = json.loads((bench_dir / "tasks.json").read_bytes().decode("utf-8"))
    specs = [TaskSpec.model_validate(t) for t in raw["tasks"]]
    return {s.id: s for s in specs}


def load_splits(bench_dir: Path = BENCH_DIR) -> dict[str, tuple[str, ...]]:
    raw = json.loads((bench_dir / "splits.json").read_bytes().decode("utf-8"))
    return {name: tuple(ids) for name, ids in raw["splits"].items()}


def split_specs(split: str, task_class: TaskClass | None = None) -> list[TaskSpec]:
    """Offline view: TaskSpecs of a split (sorted by id), optionally for one class."""
    specs = load_task_specs()
    ids = load_splits()[split]
    return [
        specs[i]
        for i in sorted(ids)
        if task_class is None or specs[i].runtime.task_class == task_class
    ]


def runtime_tasks(split: str, task_class: TaskClass | None = None) -> list[RuntimeTask]:
    """Ground-truth-free view of a split: what optimizers/runtime may hold."""
    return [s.runtime_view() for s in split_specs(split, task_class)]


def benchmark_hash(bench_dir: Path = BENCH_DIR, store: SnapshotStore | None = None) -> str:
    """Identity of the frozen benchmark: every TaskSpec, the splits and every snapshot's bytes."""
    store = store or SnapshotStore()
    specs = load_task_specs(bench_dir)
    return canonical_hash(
        {
            "tasks": {tid: s.content_hash for tid, s in sorted(specs.items())},
            "splits": {k: list(v) for k, v in sorted(load_splits(bench_dir).items())},
            "snapshots": {sid: store.snapshot_hash(sid) for sid in store.snapshot_ids()},
        }
    )
