import asyncio

import pytest

from core.payloads import Pages
from core.results import FailureKind
from core.stages import FilterMethod, GatherMode, GatherSource, StageKind
from runtime.executors.base import ExecutorInput
from runtime.executors.filter import FilterExecutor
from runtime.executors.gather import GatherExecutor
from runtime.sources import DirectoryApiSource, DirectorySnapshotSource, SourceError
from runtime.spans import locate_quote, span_text, supports
from tests.conftest import flt, gather, make_runtime_task
from tests.runtime_helpers import PAGES, QUOTE, make_ctx, write_snapshot


@pytest.fixture
def sources(tmp_path):
    write_snapshot(tmp_path)
    return DirectorySnapshotSource(tmp_path), DirectoryApiSource(tmp_path)


def run_gather(sources, **stage):
    task = make_runtime_task()
    ex = GatherExecutor(pages=sources[0], api=sources[1])
    inp = ExecutorInput(stage_index=0, stage=gather(**stage), payload=task)
    return asyncio.run(ex.run(inp, make_ctx(task)))


def test_fetch_reads_every_frozen_page_in_sorted_order_and_counts_tool_calls(sources):
    out = run_gather(sources)
    assert [p.page_id for p in out.payload.pages] == ["p1", "p2", "p3"]
    assert out.payload.pages[0].content == PAGES["p1"]
    assert out.usage.tool_calls == 3 and out.metrics.pages_fetched == 3


@pytest.mark.parametrize("mode", list(GatherMode))
def test_parallel_modes_return_identical_deterministic_pages(sources, mode):
    base = run_gather(sources).payload
    assert run_gather(sources, mode=mode).payload == base


def test_api_source_returns_mock_endpoint_records(sources):
    out = run_gather(sources, source=GatherSource.API)
    assert [p.page_id for p in out.payload.pages] == ["api:country"]
    assert '"capital":"Paris"' in out.payload.pages[0].content
    assert out.usage.tool_calls == 1


def test_jev_and_missing_sources_fail_explicitly_instead_of_faking(sources, tmp_path):
    out = run_gather(sources, source=GatherSource.JEV)
    assert out.payload is None and out.failure.kind is FailureKind.EXECUTOR_ERROR
    task = make_runtime_task(snapshot_id="nope")
    ex = GatherExecutor(pages=sources[0])
    inp = ExecutorInput(stage_index=0, stage=gather(), payload=task)
    assert "unknown snapshot" in asyncio.run(ex.run(inp, make_ctx(task))).failure.message
    with pytest.raises(SourceError):
        sources[0].read_page("snap-001", "missing")


def _filter(pages: Pages, method=FilterMethod.KEYWORD_CHUNK):
    ex = FilterExecutor()
    inp = ExecutorInput(stage_index=1, stage=flt(method), payload=pages)
    assert ex.kind is StageKind.FILTER
    return asyncio.run(ex.run(inp, make_ctx())).payload


def gathered(sources) -> Pages:
    return run_gather(sources).payload


def test_keyword_chunk_keeps_relevant_chunks_and_drops_irrelevant_ones(sources):
    kept = _filter(gathered(sources))
    text = " ".join(p.content for p in kept.pages)
    assert "Paris is the capital" in text
    assert "bananas" not in text
    assert _filter(gathered(sources)) == kept  # deterministic


def test_filtered_chunks_still_yield_spans_on_the_original_page(sources):
    pages = gathered(sources)
    originals = {p.page_id: p for p in pages.pages}
    kept = _filter(pages)
    span = locate_quote(QUOTE, kept.pages, originals, page_id="p1")
    assert span is not None and span.page_id == "p1"
    assert span.content_hash == originals["p1"].content_hash
    assert span_text(span, originals) == QUOTE
    assert span.char_start == PAGES["p1"].index(QUOTE)  # offset is in the ORIGINAL page


def test_section_select_splits_on_headings(sources):
    kept = _filter(gathered(sources), FilterMethod.SECTION_SELECT)
    assert {p.page_id for p in kept.pages} == {"p1", "p2"}
    assert all(p.content.startswith("# ") for p in kept.pages)


def test_filter_never_drops_everything():
    pages = Pages(pages=())
    assert _filter(pages) == pages
    from core.payloads import Page

    zz = Pages(pages=(Page(page_id="z", source_ref="x", content="zzz qqq"),))
    assert _filter(zz) == zz  # nothing matches -> pass through


def test_span_validation_helpers_detect_bad_spans(sources):
    pages = gathered(sources)
    originals = {p.page_id: p for p in pages.pages}
    good = originals["p1"].span(0, 5)
    assert span_text(good, originals) == PAGES["p1"][:5]
    stale = good.model_copy(update={"content_hash": "0" * 64})
    assert span_text(stale, originals) is None
    assert span_text(originals["p1"].span(0, 10_000), originals) is None
    assert locate_quote("not in any page", pages.pages, originals) is None
    assert supports("2,100,000", "about 2100000 people")
    assert not supports("Lyon", "Paris is the capital")
