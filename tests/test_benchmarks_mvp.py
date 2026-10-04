"""Frozen MVP benchmark: shape, split determinism, snapshot/API consistency."""

import json

import pytest

from benchmarks.build import N_VALIDATION, compute_splits
from benchmarks.loader import (
    benchmark_hash,
    load_splits,
    load_task_specs,
    runtime_tasks,
    split_specs,
)
from benchmarks.mock_api import MockAPI, UnknownEndpoint
from benchmarks.snapshot_store import SnapshotStore
from core.task_spec import RuntimeTask, TaskClass

# Pin of the committed benchmark bytes. A change means the frozen benchmark changed.
# wall_time_s caps raised 45/30 -> 180 s for hosted Gemma 4 latency (other caps unchanged)
GOLDEN_BENCHMARK_HASH = "4c44edcd2fa1e27012b4fe7f072002c44ae3136f224221fa2e90a4b1a96529bc"

STORE = SnapshotStore()
SPECS = load_task_specs()


def test_benchmark_is_class_a_and_b_only_with_5_to_8_tasks_each():
    counts = {c: 0 for c in TaskClass}
    for s in SPECS.values():
        counts[s.runtime.task_class] += 1
    assert counts[TaskClass.C] == 0
    assert 5 <= counts[TaskClass.A] <= 8 and 5 <= counts[TaskClass.B] <= 8


def test_benchmark_hash_is_frozen():
    assert benchmark_hash() == GOLDEN_BENCHMARK_HASH


def test_splits_are_disjoint_cover_all_tasks_and_match_the_documented_rule():
    splits = load_splits()
    train, val = set(splits["train"]), set(splits["validation"])
    assert not train & val
    assert train | val == set(SPECS)
    by_class: dict[str, list[str]] = {}
    for s in SPECS.values():
        by_class.setdefault(s.runtime.task_class.value, []).append(s.id)
    assert compute_splits(by_class) == {k: sorted(v) for k, v in splits.items()}
    for cls in ("A", "B"):
        assert sum(SPECS[i].runtime.task_class.value == cls for i in val) == N_VALIDATION


def test_every_task_has_a_snapshot_with_pages_and_records():
    for s in SPECS.values():
        sid = s.runtime.snapshot_id
        assert STORE.has_snapshot(sid)
        assert STORE.pages(sid).pages
        assert STORE.records(sid)


def test_class_a_ground_truth_is_literally_present_on_the_snapshot_pages():
    for s in SPECS.values():
        if s.runtime.task_class is not TaskClass.A:
            continue
        text = " ".join(p.content for p in STORE.pages(s.runtime.snapshot_id).pages)
        for field, value in s.ground_truth.values.items():
            for item in value if isinstance(value, list) else [value]:
                assert str(item) in text, (s.id, field, item)


def test_mock_api_unfiltered_response_is_byte_identical_to_the_class_b_page():
    api = MockAPI(STORE)
    for s in SPECS.values():
        if s.runtime.task_class is not TaskClass.B:
            continue
        sid = s.runtime.snapshot_id
        assert api.get(sid, "items") == STORE.page(sid, "items").content


def test_mock_api_filters_come_from_the_same_records_and_are_deterministic():
    api = MockAPI(STORE)
    records = STORE.records("B-001")["items"]
    body = json.loads(api.get("B-001", "items", category="tools"))
    assert body["results"] == [r for r in records if r["category"] == "tools"]
    assert api.get("B-001", "items", category="tools") == api.get(
        "B-001", "items", category="tools"
    )
    with pytest.raises(UnknownEndpoint):
        api.get("B-001", "nope")


def test_class_b_ground_truth_is_exact_over_the_api_records():
    records = STORE.records("B-001")["items"]
    assert SPECS["B-001"].ground_truth.values["total_stock"] == sum(
        r["stock"] for r in records if r["category"] == "tools"
    )


def test_runtime_view_contains_no_ground_truth_and_is_a_runtime_task():
    for split in ("train", "validation"):
        for t in runtime_tasks(split):
            assert isinstance(t, RuntimeTask)
            assert "ground_truth" not in t.model_dump_json()


def test_split_specs_are_sorted_and_class_filtered():
    for cls in (TaskClass.A, TaskClass.B):
        specs = split_specs("train", cls)
        assert [s.id for s in specs] == sorted(s.id for s in specs)
        assert all(s.runtime.task_class is cls for s in specs)


def test_all_tasks_in_a_class_share_constraints_so_one_genome_serves_the_class():
    for cls in (TaskClass.A, TaskClass.B):
        tasks = [s.runtime for s in SPECS.values() if s.runtime.task_class is cls]
        ref = tasks[0]
        assert all((t.caps, t.allowed_sources) == (ref.caps, ref.allowed_sources) for t in tasks)
