"""Load the frozen benchmark: TaskSpecs (offline), runtime views, splits, catalog, benchmark hash.

``TaskSpec`` (with ground truth) must only be handed to ``evaluation/``. Everything else - the
optimizer, the runtime, the experiment harness - works with ``runtime_tasks`` (``RuntimeTask``).
The catalog (suite + shape per task) is benchmark metadata for choosing a suite's rows and for
reports; it carries no ground truth and never reaches the runtime.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from benchmarks.snapshot_store import SnapshotStore
from core.canonical import canonical_hash
from core.task_spec import RuntimeTask, TaskClass, TaskSpec

BENCH_DIR = Path(__file__).resolve().parent
HELDOUT_DIR = BENCH_DIR / "heldout"
SPLIT_NAMES = ("train", "validation")


@dataclass(frozen=True)
class TaskMeta:
    suite: str  # the search policy the task is searched / reported under
    shape: str  # what kind of task it is (reporting only)


def load_task_specs(bench_dir: Path = BENCH_DIR) -> dict[str, TaskSpec]:
    raw = json.loads((bench_dir / "tasks.json").read_bytes().decode("utf-8"))
    specs = [TaskSpec.model_validate(t) for t in raw["tasks"]]
    return {s.id: s for s in specs}


def load_splits(bench_dir: Path = BENCH_DIR) -> dict[str, tuple[str, ...]]:
    raw = json.loads((bench_dir / "splits.json").read_bytes().decode("utf-8"))
    return {name: tuple(ids) for name, ids in raw["splits"].items()}


def _catalog_raw(bench_dir: Path) -> dict | None:
    path = bench_dir / "catalog.json"
    return json.loads(path.read_bytes().decode("utf-8")) if path.is_file() else None


def load_catalog(bench_dir: Path = BENCH_DIR) -> dict[str, TaskMeta]:
    """Suite + shape per task id. A directory without ``catalog.json`` (a pre-#13 layout) puts
    every task in the suite named after its class, shape ``unknown``."""
    raw = _catalog_raw(bench_dir)
    if raw is None:
        return {
            tid: TaskMeta(suite=s.runtime.task_class.value, shape="unknown")
            for tid, s in load_task_specs(bench_dir).items()
        }
    return {tid: TaskMeta(suite=e["suite"], shape=e["shape"]) for tid, e in raw["tasks"].items()}


def suite_names(*bench_dirs: Path) -> tuple[str, ...]:
    """Every suite named by the given benchmark directories (default: search + held-out)."""
    names: set[str] = set()
    for d in bench_dirs or (BENCH_DIR, HELDOUT_DIR):
        names |= {m.suite for m in load_catalog(d).values()}
    return tuple(sorted(names))


def split_specs(
    split: str, task_class: TaskClass | None = None, bench_dir: Path = BENCH_DIR
) -> list[TaskSpec]:
    """Offline view: TaskSpecs of a split (sorted by id), optionally for one class."""
    specs = load_task_specs(bench_dir)
    ids = load_splits(bench_dir)[split]
    return [
        specs[i]
        for i in sorted(ids)
        if task_class is None or specs[i].runtime.task_class == task_class
    ]


def runtime_tasks(
    split: str, task_class: TaskClass | None = None, bench_dir: Path = BENCH_DIR
) -> list[RuntimeTask]:
    """Ground-truth-free view of a split: what optimizers/runtime may hold."""
    return [s.runtime_view() for s in split_specs(split, task_class, bench_dir)]


def benchmark_hash(
    bench_dir: Path = BENCH_DIR,
    store: SnapshotStore | None = None,
    *,
    task_ids: Iterable[str] | None = None,
) -> str:
    """Identity of a frozen benchmark: every TaskSpec, the splits, every snapshot's bytes and
    (when the directory has one) the catalog.

    ``task_ids`` hashes a SUBSET instead: only those tasks, their snapshots and their split
    membership, without the catalog - exactly the formula the directory had before the catalog
    existed, so a subset pinned then (``benchmarks.heldout.OSS_TEST_IDS``) keeps its hash.
    """
    store = store or SnapshotStore(bench_dir / "snapshots")
    specs = load_task_specs(bench_dir)
    splits = load_splits(bench_dir)
    if task_ids is not None:
        keep = set(task_ids)
        if missing := keep - set(specs):
            raise KeyError(f"not tasks of {bench_dir}: {sorted(missing)}")
        snaps = {specs[t].runtime.snapshot_id for t in keep}
        return canonical_hash(
            {
                "tasks": {tid: s.content_hash for tid, s in sorted(specs.items()) if tid in keep},
                "splits": {
                    k: [i for i in v if i in keep]
                    for k, v in sorted(splits.items())
                    if any(i in keep for i in v)
                },
                "snapshots": {sid: store.snapshot_hash(sid) for sid in sorted(snaps)},
            }
        )
    identity = {
        "tasks": {tid: s.content_hash for tid, s in sorted(specs.items())},
        "splits": {k: list(v) for k, v in sorted(splits.items())},
        "snapshots": {sid: store.snapshot_hash(sid) for sid in store.snapshot_ids()},
    }
    catalog = _catalog_raw(bench_dir)
    if catalog is not None:
        identity["catalog"] = catalog
    return canonical_hash(identity)
