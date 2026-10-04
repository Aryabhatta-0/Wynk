"""Benchmark snapshots satisfy the runtime source protocols without changing page identity."""

from benchmarks.mock_api import MockAPI
from benchmarks.snapshot_store import SnapshotStore
from evaluation.evidence import SnapshotEvidenceVerifier
from experiments.real_runtime import MockApiSource, SnapshotPageSource


def test_page_source_preserves_snapshot_page_ids_and_hashes():
    store = SnapshotStore()
    src = SnapshotPageSource(store)
    ids = src.list_page_ids("A-001")
    assert ids == [p.page_id for p in store.pages("A-001").pages]
    for pid in ids:
        assert src.read_page("A-001", pid).content_hash == store.page("A-001", pid).content_hash


def test_api_pages_produce_spans_the_evaluator_accepts():
    store = SnapshotStore()
    src = MockApiSource(MockAPI(store))
    verifier = SnapshotEvidenceVerifier(store)
    for endpoint in src.list_endpoints("B-001"):
        page = src.call("B-001", endpoint)
        assert verifier.is_valid(page.span(0, len(page.content)), "B-001")
