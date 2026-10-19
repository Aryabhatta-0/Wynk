"""GATHER websearch (#12) as a real, opt-in execution step: WorkflowRunner -> compiler ->
StageRunner -> GatherExecutor -> WebSearchSource -> Pages.

Every provider here is a deterministic local double (``FakeWebSearch``) or the hosted Brave
adapter over a fake transport; nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from core.constraints import ConstraintChecker, ConstraintConfig
from core.cost_model import StaticCostModel
from core.evidence_authority import (
    authority_members,
    compose_authority,
    is_websearch_identity,
    live_websearch,
    unauthorized,
)
from core.genome import Genome
from core.grammar import Grammar
from core.results import BudgetCap, BudgetUsage, FailureKind, StageStatus
from core.stages import (
    ALL_STAGE_KINDS,
    FailureStrategy,
    GatherMode,
    GatherSource,
    StageKind,
    VerifyMethod,
    all_stage_specs,
)
from core.violations import ViolationCode as V
from runtime.backends.brave_search import (
    DEFAULT_ENDPOINT,
    FIXED_PARAMS,
    BraveSearchConfig,
    BraveWebSearchSource,
    WebSearchConfigError,
    _NoRedirect,
    parse_response,
    websearch_from_env,
)
from runtime.executors.base import ExecutorInput, GuardedExecutor
from runtime.executors.gather import GatherExecutor
from runtime.sources import (
    MAX_QUERY_CHARS,
    DirectoryApiSource,
    DirectorySnapshotSource,
    SourceError,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
    search_pages,
    web_query,
)
from tests.conftest import SECRET_ANSWER, extract, gather, make_caps, make_task, synth, verify
from tests.runtime_helpers import (
    DONE,
    QUOTE,
    ScriptedJev,
    ScriptedModel,
    make_ctx,
    opens,
    write_snapshot,
)

WS = GatherSource.WEBSEARCH
WITH_WS = (GatherSource.FETCH, GatherSource.API, GatherSource.JEV, WS)
WS_ID = "websearch:fake/test@1"
WS_B = "websearch:fake/test@2"
JEV_ID = "jev:scripted/test@1"
FAKE_KEY = "test-websearch-key-0000-not-a-real-credential"
QUESTION = "What is the capital of France?"

RESULTS = (
    WebSearchResult(
        rank=1,
        url="https://example.org/france",
        title="France",
        snippet=f"{QUOTE}. It lies on the Seine.",
    ),
    WebSearchResult(
        rank=2, url="https://example.org/germany", title="", snippet="Berlin is in Germany."
    ),
)


class FakeWebSearch:
    """Deterministic ``WebSearchSource``: answers every request with ``response`` (an exception
    is raised instead), recording each request it was sent."""

    def __init__(self, response=None, identity: str = WS_ID, max_results: int = 5) -> None:
        self.response = WebSearchResponse(results=RESULTS) if response is None else response
        self._identity = identity
        self._max_results = max_results
        self.requests: list[WebSearchRequest] = []

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def max_results(self) -> int:
        return self._max_results

    def search(self, request: WebSearchRequest):
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class CountingPages(DirectorySnapshotSource):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.reads: list[str] = []

    def list_page_ids(self, snapshot_id):
        self.reads.append("*list*")
        return super().list_page_ids(snapshot_id)

    def read_page(self, snapshot_id, page_id):
        self.reads.append(page_id)
        return super().read_page(snapshot_id, page_id)


@pytest.fixture
def root(tmp_path):
    write_snapshot(tmp_path)
    return tmp_path


def ws_task(**kw):
    return make_task(allowed_sources=WITH_WS, **kw)


def run_gather(root, websearch, mode=GatherMode.SEQUENTIAL, task=None, jev=None):
    task = task or ws_task()
    pages = CountingPages(root)
    ex = GatherExecutor(pages=pages, api=DirectoryApiSource(root), jev=jev, websearch=websearch)
    inp = ExecutorInput(stage_index=0, stage=gather(WS, mode), payload=task)
    return asyncio.run(ex.run(inp, make_ctx(task))), pages


def guarded_gather(root, websearch, *, cap, used=0):
    task = ws_task(caps=make_caps(tool_calls=cap))
    ctx = make_ctx(task)
    ctx.guard.charge(BudgetUsage(tool_calls=used))
    pages = CountingPages(root)
    ex = GuardedExecutor(GatherExecutor(pages=pages, websearch=websearch))
    inp = ExecutorInput(stage_index=0, stage=gather(WS), payload=task)
    return asyncio.run(ex.run(inp, ctx)), pages, ctx


def workflow_runner(root, model, websearch=None, jev=None, pages=None):
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner

    return WorkflowRunner(
        model=model,
        benchmark_hash="bench-1",
        pages=pages or DirectorySnapshotSource(root),
        api=DirectoryApiSource(root),
        jev=jev,
        websearch=websearch,
    )


WS_GENOME = Genome.of(gather(WS), extract(), synth())
FETCH_GENOME = Genome.of(gather(), extract(), synth())


# == execution ===================================================================================
def test_websearch_genome_executes_end_to_end_through_the_workflow_runner(root):
    ws = FakeWebSearch()
    model = ScriptedModel()
    pages = CountingPages(root)
    result = workflow_runner(root, model, ws, pages=pages).run_sync(WS_GENOME, ws_task(), seed=3)
    assert result.failure is None
    assert result.answer.values == {"capital": "Paris"}
    assert [t.status for t in result.stage_trace] == [StageStatus.OK] * 3
    assert result.stage_trace[0].kind is StageKind.GATHER
    # one search request is one tool call; the model stages never count as tool calls
    assert result.stage_trace[0].usage.tool_calls == 1 and result.budget_usage.tool_calls == 1
    assert result.metrics.pages_fetched == 2
    assert pages.reads == []  # the frozen snapshot is never read instead
    assert len(ws.requests) == 1
    extract_input = next(r for r in model.requests if r.prompt_template_id.startswith("extract"))
    assert QUOTE in extract_input.input_text and "Berlin is in Germany" in extract_input.input_text


def test_websearch_pages_are_exactly_the_results_with_deterministic_ids():
    pages = search_pages(WebSearchResponse(results=RESULTS))
    assert [p.source_ref for p in pages] == [r.url for r in RESULTS]  # the real URL, unopened
    assert pages[0].page_id.startswith("web-01-") and pages[1].page_id.startswith("web-02-")
    assert pages == search_pages(WebSearchResponse(results=RESULTS))  # stable
    assert pages[0].content == f"France\n\n{QUOTE}. It lies on the Seine."
    assert pages[1].content == "Berlin is in Germany."  # no title: the snippet alone


def test_websearch_evidence_cites_the_result_it_came_from(root):
    result = workflow_runner(root, ScriptedModel(), FakeWebSearch()).run_sync(WS_GENOME, ws_task())
    span = result.evidence[0].spans[0]
    assert span.page_id == search_pages(WebSearchResponse(results=RESULTS))[0].page_id


# == the query: minimal, deterministic, bounded ==================================================
def test_the_query_is_only_the_rows_inputs(root):
    ws = FakeWebSearch()
    out, _ = run_gather(root, ws)
    assert out.failure is None
    (request,) = ws.requests
    assert request.query == QUESTION and request.max_results == 5
    task = ws_task()
    for private in (task.contract.instructions, "snap-001", SECRET_ANSWER, "capital:"):
        assert private not in request.query  # no instructions, context, target or schema


def test_an_uploaded_rows_context_and_other_rows_are_never_sent(root):
    from core.run_contract import ExampleInput, ExecutionTask
    from core.task_contract import ConstraintLimits, WorkflowSpec
    from tests.contract_helpers import qa_contract
    from tests.test_contract_runtime import FULL_CAPS

    contract = qa_contract(
        constraints=ConstraintLimits(**FULL_CAPS),
        workflow=WorkflowSpec(allowed_sources=(GatherSource.FETCH, WS)),
    )
    task = ExecutionTask(
        contract=contract,
        example=ExampleInput(row_id="r1", values={"question": "who?", "passage": "secret text"}),
    )
    ws = FakeWebSearch()
    out, _ = run_gather(root, ws, task=task)
    assert out.failure is None
    assert [r.query for r in ws.requests] == ["who?"]  # the passage (context) stays local


def test_web_query_is_bounded_at_a_word_boundary():
    assert web_query("  a \n b\t c ") == "a b c"
    assert web_query(" \n ") == ""
    long = " ".join(["word"] * 200)
    q = web_query(long)
    assert len(q) <= MAX_QUERY_CHARS and q.endswith("word") and long.startswith(q)
    assert web_query("x" * 1000) == "x" * MAX_QUERY_CHARS


def test_an_empty_query_sends_nothing(root):
    ws = FakeWebSearch()
    out, _ = run_gather(root, ws, task=make_task(question="   ", allowed_sources=WITH_WS))
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR and ws.requests == []
    assert out.usage.tool_calls == 0


# == budget ======================================================================================
def test_one_request_with_exactly_one_call_left_completes_at_the_cap(root):
    ws = FakeWebSearch()
    out, _, ctx = guarded_gather(root, ws, cap=3, used=2)
    assert out.failure is None
    assert out.usage.tool_calls == 1 and ctx.guard.usage.tool_calls == 3
    assert not ctx.guard.exceeded()


@pytest.mark.parametrize(("cap", "used"), [(3, 3), (1, 1), (2, 2)])
def test_no_call_left_is_refused_before_the_request_starts(root, cap, used):
    ws = FakeWebSearch()
    out, pages, ctx = guarded_gather(root, ws, cap=cap, used=used)
    assert out.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert out.failure.cap is BudgetCap.TOOL_CALLS and out.failure.stage_index == 0
    assert ws.requests == [] and pages.reads == []  # nothing sent, nothing substituted
    assert out.usage.tool_calls == 0 and ctx.guard.usage.tool_calls == used  # no overshoot


def test_the_request_deadline_is_the_runs_remaining_wall_time(root):
    ws = FakeWebSearch()
    task = ws_task()
    ctx = make_ctx(task)
    ex = GatherExecutor(websearch=ws)
    inp = ExecutorInput(stage_index=0, stage=gather(WS), payload=task)
    asyncio.run(ex.run(inp, ctx))
    (request,) = ws.requests
    assert 0 < request.timeout_s <= task.caps.wall_time_s


def test_websearch_costs_one_tool_call_in_the_static_cost_model():
    contract = ws_task().contract
    model = StaticCostModel()
    ws, fetch = model.estimate(WS_GENOME, contract), model.estimate(FETCH_GENOME, contract)
    assert fetch.tool_calls - ws.tool_calls == 3 - 1  # GATHER:fetch 3 calls, websearch 1
    assert ws.tokens == fetch.tokens  # a search request spends no model tokens


# == failures: fail closed, never fall back ======================================================
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"results": []}, "unstructured"),  # not a WebSearchResponse
        (WebSearchResponse(results=RESULTS), "returned 2 results; 1 were requested"),
        (SourceError("websearch provider timed out after 1.0s"), "timed out"),
        (SourceError("websearch provider returned HTTP 503"), "HTTP 503"),
        (WebSearchResponse(), "no results"),
    ],
)
def test_bad_responses_and_errors_fail_cleanly_without_a_fallback(root, response, message):
    ws = FakeWebSearch(response, max_results=1)
    jev = ScriptedJev()
    out, pages = run_gather(root, ws, jev=jev)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert message in out.failure.message
    assert out.usage.tool_calls == 1  # the request was made: it is charged
    assert pages.reads == [] and jev.observations == []  # no fetch / api / jev substitute
    assert len(ws.requests) == 1  # never retried


def test_executor_without_a_websearch_source_fails_explicitly(root):
    out, pages = run_gather(root, None)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert "websearch" in out.failure.message and "not available" in out.failure.message
    assert pages.reads == [] and out.usage.tool_calls == 0


@pytest.mark.parametrize("mode", [GatherMode.PARALLEL_2, GatherMode.PARALLEL_4])
def test_executor_refuses_parallel_websearch_even_past_admission(root, mode):
    ws = FakeWebSearch()
    out, _ = run_gather(root, ws, mode)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR and ws.requests == []


# == grammar / constraints =======================================================================
def checker(**cfg) -> ConstraintChecker:
    return ConstraintChecker(
        config=ConstraintConfig(unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,), **cfg)
    )


def codes(genome, contract, **cfg):
    return {v.code for v in checker(**cfg).check(genome, contract, complete=True)}


def test_websearch_is_admissible_only_where_the_contract_lists_it():
    assert codes(WS_GENOME, ws_task().contract) == set()
    default = make_task().contract  # fetch / api / jev: never websearch by default
    assert WS not in default.workflow.allowed_sources
    assert V.SOURCE_NOT_ALLOWED in codes(WS_GENOME, default)


def test_websearch_is_sequential_only():
    for mode in (GatherMode.PARALLEL_2, GatherMode.PARALLEL_4):
        genome = Genome.of(gather(WS, mode), extract(), synth())
        assert V.WEBSEARCH_PARALLEL in codes(genome, ws_task().contract)


def test_an_unconfigured_provider_makes_websearch_unavailable_not_substituted():
    found = checker(unavailable_sources=(WS,)).check(WS_GENOME, ws_task().contract, complete=True)
    (v,) = [v for v in found if v.code is V.RUNTIME_UNAVAILABLE]
    assert "websearch" in v.message and "nothing is substituted" in v.message


def test_the_runner_without_a_provider_refuses_before_any_work(root):
    model = ScriptedModel()
    pages = CountingPages(root)
    runner = workflow_runner(root, model, websearch=None, pages=pages)
    with pytest.raises(Exception, match="runtime_unavailable") as info:
        runner.run_sync(WS_GENOME, ws_task())
    assert [v.code for v in info.value.violations] == [V.RUNTIME_UNAVAILABLE]
    assert pages.reads == [] and model.requests == []


def test_the_runner_with_a_provider_refuses_a_contract_that_does_not_allow_it(root):
    ws = FakeWebSearch()
    runner = workflow_runner(root, ScriptedModel(), ws)
    with pytest.raises(Exception, match="source_not_allowed"):
        runner.run_sync(WS_GENOME, make_task())
    assert ws.requests == []


def test_a_failed_verification_retries_extraction_without_a_second_search(root):
    ws = FakeWebSearch()
    span = verify(VerifyMethod.EVIDENCE_SPAN, FailureStrategy.RETRY_1)
    genome = Genome.of(gather(WS), extract(), span, synth())
    model = ScriptedModel(extract_quotes=["this sentence is nowhere in the results", QUOTE])
    result = workflow_runner(root, model, ws).run_sync(genome, ws_task())
    assert result.failure is None
    extracts = [r for r in model.requests if r.prompt_template_id.startswith("extract")]
    assert len(extracts) == 2  # the verifier failed once and EXTRACT was retried
    assert len(ws.requests) == 1  # ... over the same results: one search request in total


def test_websearch_never_regathers():
    regather = verify(VerifyMethod.EVIDENCE_SPAN, FailureStrategy.REGATHER)
    genome = Genome.of(gather(WS), extract(), regather, synth())
    assert codes(genome, ws_task().contract) == {V.WEBSEARCH_REGATHER}
    # fetch keeps its regather (no websearch rule applies)
    frozen = Genome.of(gather(), extract(), regather, synth())
    assert V.WEBSEARCH_REGATHER not in codes(frozen, ws_task().contract)


def test_the_runner_refuses_a_websearch_regather_before_any_request(root):
    ws = FakeWebSearch()
    regather = verify(VerifyMethod.EVIDENCE_SPAN, FailureStrategy.REGATHER)
    genome = Genome.of(gather(WS), extract(), regather, synth())
    with pytest.raises(Exception, match="websearch_regather"):
        workflow_runner(root, ScriptedModel(), ws).run_sync(genome, ws_task())
    assert ws.requests == []


def test_the_default_language_and_every_frozen_genome_are_unchanged():
    assert Grammar().language_size() == 340_200
    assert Grammar(ALL_STAGE_KINDS).language_size() == 1_020_610
    gathers = all_stage_specs(StageKind.GATHER)
    assert len(gathers) == 9 and all(s.source is not WS for s in gathers)
    assert [s.source for s in all_stage_specs(StageKind.GATHER, (WS,))][-1] is WS
    assert all_stage_specs(StageKind.GATHER, (WS,))[:9] == gathers
    assert Grammar().version == "grammar/1"
    live = Grammar(live_sources=(WS,))
    assert live.version != Grammar().version and "websearch" in live.version
    # a websearch genome is outside the default vocabulary, inside the opted-in one
    assert Grammar().validate(WS_GENOME) and not live.validate(WS_GENOME)


def test_admissible_successors_of_a_frozen_contract_ignore_websearch():
    contract = make_task().contract
    a = list(checker().admissible_successors(Genome.of(), contract))
    b = list(checker(unavailable_sources=(WS,)).admissible_successors(Genome.of(), contract))
    assert a == b and all(getattr(s, "source", None) is not WS for s in a)


# == provenance: the exact provider identity =====================================================
def test_websearch_runs_record_the_exact_provider_identity(root):
    ws = FakeWebSearch()
    model = ScriptedModel()
    with_ws = workflow_runner(root, model, ws)
    without = workflow_runner(root, model)
    task = ws_task()
    key = with_ws.run_key(WS_GENOME, task)
    assert key.versions.evidence_source == WS_ID
    assert key.versions.model_dump(mode="json")["evidence_source"] == WS_ID
    # a fetch run has the same identity whether or not websearch is configured
    assert with_ws.run_key(FETCH_GENOME, task) == without.run_key(FETCH_GENOME, task)
    assert "evidence_source" not in without.run_key(FETCH_GENOME, task).versions.model_dump(
        mode="json"
    )
    frozen = make_task()
    assert with_ws.run_key(FETCH_GENOME, frozen) == without.run_key(FETCH_GENOME, frozen)
    assert with_ws.versions(frozen) == without.versions(frozen)


def test_jev_and_websearch_runs_in_one_runtime_each_record_their_own_provider(root):
    runner = workflow_runner(root, ScriptedModel(), FakeWebSearch(), ScriptedJev())
    task = ws_task()
    jev_genome = Genome.of(gather(GatherSource.JEV), extract(), synth())
    assert runner.run_key(jev_genome, task).versions.evidence_source == JEV_ID
    assert runner.run_key(WS_GENOME, task).versions.evidence_source == WS_ID
    assert runner.run_key(FETCH_GENOME, task).versions.evidence_source is None


def test_the_evidence_authority_adds_websearch_only_for_opted_in_contracts(root):
    model = ScriptedModel()
    both = workflow_runner(root, model, FakeWebSearch(), ScriptedJev())
    frozen, opted = make_task().contract, ws_task().contract
    assert both.evidence_authority(frozen) == JEV_ID  # #55's JEV-only value, byte for byte
    assert both.evidence_authority(opted) == json.dumps(
        sorted([JEV_ID, WS_ID]), separators=(",", ":")
    )
    assert workflow_runner(root, model).evidence_authority(opted) is None  # fetch-only: unchanged
    assert workflow_runner(root, model, FakeWebSearch()).evidence_authority(frozen) is None
    assert workflow_runner(root, model, FakeWebSearch()).evidence_authority(opted) == WS_ID
    assert workflow_runner(root, model, jev=ScriptedJev()).evidence_authority(opted) == JEV_ID


def test_compose_authority_keeps_single_values_and_encodes_sets_canonically():
    assert compose_authority() is None and compose_authority(None, None) is None
    assert compose_authority(JEV_ID) == JEV_ID and compose_authority(None, JEV_ID) == JEV_ID
    both = compose_authority(WS_ID, JEV_ID)
    assert both == compose_authority(JEV_ID, WS_ID, JEV_ID) == f'["{JEV_ID}","{WS_ID}"]'
    assert authority_members(both) == {JEV_ID, WS_ID}
    assert authority_members(JEV_ID) == {JEV_ID} and authority_members(None) == frozenset()
    # changing either provider changes the authority (exact equality pins fail closed)
    assert compose_authority(JEV_ID, WS_B) != both
    assert compose_authority("jev:scripted/test@2", WS_ID) != both
    for bad in ('["a"]', '["b","a"]', "[1,2]", "[oops", '["a","a"]'):
        with pytest.raises(ValueError):
            authority_members(bad)
    with pytest.raises(ValueError):
        compose_authority('["smuggled"]')


def test_a_candidate_must_use_a_provider_the_authority_binds():
    both = compose_authority(JEV_ID, WS_ID)
    assert unauthorized(both, [JEV_ID, WS_ID, None]) == ()
    assert unauthorized(JEV_ID, [WS_ID]) == (WS_ID,)  # websearch used, JEV-only binding
    assert unauthorized(both, [WS_B]) == (WS_B,)  # another websearch provider
    assert unauthorized(None, [None]) == ()


def test_source_provider_cross_checks_websearch_candidates():
    from memory.priors import source_provider
    from tests.test_memory_priors import DECLARED

    fetch = DECLARED.model_dump(mode="json")
    ws = {**fetch, "evidence_source": WS_ID}
    jev = {**fetch, "evidence_source": JEV_ID}
    both = compose_authority(JEV_ID, WS_ID)
    assert source_provider(both, [fetch, jev, ws]) == both
    with pytest.raises(ValueError, match="bound with 'jev"):
        source_provider(JEV_ID, [ws])
    with pytest.raises(ValueError, match="bound with None"):
        source_provider(None, [ws])


# == live-source reproducibility =================================================================
def test_live_websearch_is_detected_from_the_authority_or_any_candidate():
    fetch = {"evidence_source": None}
    assert not live_websearch(None, [fetch]) and not live_websearch(JEV_ID, [{}])
    assert live_websearch(WS_ID, [fetch])
    assert live_websearch(compose_authority(JEV_ID, WS_ID), [fetch])
    assert live_websearch(JEV_ID, [{"evidence_source": WS_ID}])  # unauthorized, still live
    assert is_websearch_identity(WS_ID) and not is_websearch_identity(JEV_ID)


def test_websearch_runs_are_never_cached(root, tmp_path):
    from experiments.real_runtime import RunCache

    runner = workflow_runner(root, ScriptedModel(), FakeWebSearch())
    live = runner.run_sync(WS_GENOME, ws_task())
    frozen = runner.run_sync(FETCH_GENOME, ws_task())
    cache = RunCache(tmp_path / "runs.jsonl")
    cache.put(live)
    cache.put(frozen)
    assert cache.get(live.run_id) is None and cache.get(frozen.run_id) == frozen
    reloaded = RunCache(tmp_path / "runs.jsonl")
    assert reloaded.get(live.run_id) is None


class _LiveWorld:
    """A durable job world (#55) whose runtime is bound with a #12 provider authority."""

    @staticmethod
    def make(root, provider, versions=None):
        from tests.test_memory_priors import DECLARED
        from tests.test_mixed_run_versions import MixedBackend, uses_jev
        from tests.test_warm_start_evidence_provider import world_at

        if versions is None:  # the "jev" genomes report the bound JEV provider (one member)

            def versions(g, t, tr):
                source = JEV_ID if uses_jev(g) else None
                return DECLARED.model_copy(update={"evidence_source": source})

        return world_at(root, provider, MixedBackend(versions))


def test_a_websearch_bound_job_pins_its_authority_and_teaches_memory_nothing(tmp_path):
    from experiments.jobs import load_canonical
    from memory.priors import PriorKind
    from store.jobs import JobState
    from tests.test_mixed_run_versions import SEARCH, definition

    both = compose_authority(JEV_ID, WS_ID)
    world = _LiveWorld.make(tmp_path / "ws", both)
    job_id = world.run(definition(SEARCH))
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    assert json.loads(world.store.get_job(job_id).identity_json)["evidence_provider"] == both
    _, art = load_canonical(world.store, job_id)
    assert art.provenance.versions.evidence_provider == both
    learned = world.prior_store.learned(job_id)
    assert (learned.outcome, learned.detail) == ("refused", "live_websearch_evidence")
    assert not world.prior_store.active(PriorKind.DATASET_CHAMPION)


def test_a_candidate_that_searched_the_web_is_refused_by_a_jev_bound_job(tmp_path):
    from memory.priors import PriorKind
    from store.jobs import JobState, UnitState
    from tests.test_memory_priors import DECLARED
    from tests.test_mixed_run_versions import SEARCH, definition, uses_jev

    def versions(g, t, tr):  # the "jev" genomes report a websearch provider instead
        return DECLARED.model_copy(update={"evidence_source": WS_ID if uses_jev(g) else None})

    world = _LiveWorld.make(tmp_path / "ws", JEV_ID, versions)
    job_id = world.run(definition(SEARCH))
    # the job was bound with JEV only: a candidate that searched the web is refused at once
    assert world.jobs.get(job_id).state is not JobState.COMPLETED
    failed = [u for u in world.store.get_job(job_id).units if u.state is UnitState.FAILED]
    assert failed and all("evidence provider" in (u.detail or "") for u in failed)
    assert all("websearch:fake/test@1" in (u.detail or "") for u in failed)
    learned = world.prior_store.learned(job_id)
    assert learned is None or learned.outcome == "refused"
    assert not world.prior_store.active(PriorKind.DATASET_CHAMPION)


def test_changing_either_provider_never_runs_or_resumes_the_job(tmp_path):
    from experiments.jobs import Reason
    from store.jobs import JobState, UnitState
    from tests.test_mixed_run_versions import SEARCH, definition
    from tests.test_warm_start_evidence_provider import ProviderRuntime

    both = compose_authority(JEV_ID, WS_ID)
    for i, now in enumerate(
        (
            compose_authority(JEV_ID, WS_B),
            compose_authority("jev:scripted/test@2", WS_ID),
            JEV_ID,
            WS_ID,
        )
    ):
        world = _LiveWorld.make(tmp_path / f"ws{i}", both)
        job_id = world.jobs.create(definition(SEARCH)).job_id
        world.worker_for(ProviderRuntime(now)).run_until_idle()
        for unit in world.store.get_job(job_id).units:
            assert (unit.state, unit.reason) == (UnitState.INTERRUPTED, Reason.RUNTIME_UNAVAILABLE)
            assert world.store.attempts(job_id, unit.strategy, unit.seed) == []
        world.jobs.resume(job_id)  # the bound providers are back
        world.worker.run_until_idle()
        assert world.jobs.get(job_id).state is JobState.COMPLETED


def test_a_websearch_job_is_never_reproduced_from_stored_evidence(tmp_path):
    from experiments.artifacts import LiveEvidenceNotReproducible, replay
    from tests.test_mixed_run_versions import SEARCH, definition

    world = _LiveWorld.make(tmp_path / "ws", compose_authority(JEV_ID, WS_ID))
    job_id = world.run(definition(SEARCH))
    with pytest.raises(LiveEvidenceNotReproducible, match="live web search") as info:
        replay(world.store, job_id, world.runtime)
    assert info.value.code == "live_evidence_not_reproducible"


# == the hosted adapter (fake transport only) ====================================================
class FakeTransport:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url, headers, timeout_s):
        self.calls.append((url, dict(headers), timeout_s))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r if isinstance(r, bytes) else json.dumps(r).encode()


def brave_body(n=2, **extra):
    results = [
        {"url": f"https://example.org/{i}", "title": f" T{i} ", "description": f"snip  {i}"}
        for i in range(1, n + 1)
    ]
    return {"type": "search", "web": {"results": results}, **extra}


def brave(*responses, **cfg) -> tuple[BraveWebSearchSource, FakeTransport]:
    transport = FakeTransport(*responses)
    config = BraveSearchConfig(api_key=FAKE_KEY, **cfg)
    return BraveWebSearchSource(config, transport), transport


def request(n=5, timeout_s=10.0, query=QUESTION) -> WebSearchRequest:
    return WebSearchRequest(query=query, max_results=n, timeout_s=timeout_s)


def test_one_search_is_one_bounded_https_get_with_only_the_query():
    src, transport = brave(brave_body())
    response = src.search(request(timeout_s=3.0))
    assert [r.title for r in response.results] == ["T1", "T2"]
    assert [r.snippet for r in response.results] == ["snip 1", "snip 2"]
    ((url, headers, timeout_s),) = transport.calls
    assert url.startswith(DEFAULT_ENDPOINT + "?")
    from urllib.parse import parse_qsl, urlsplit

    params = dict(parse_qsl(urlsplit(url).query))
    assert params == {"q": QUESTION, "count": "5", **dict(FIXED_PARAMS)}
    assert headers == {"Accept": "application/json", "X-Subscription-Token": FAKE_KEY}
    assert timeout_s == 3.0  # the run's remaining time when it is below the configured timeout
    src.config.__class__  # noqa: B018 - config stays frozen
    src2, t2 = brave(brave_body(), timeout_s=2.0)
    src2.search(request(timeout_s=30.0))
    assert t2.calls[0][2] == 2.0


def test_the_identity_names_every_result_shaping_setting_and_never_the_key():
    src, _ = brave()
    assert src.identity == (
        "websearch:brave-search@api.search.brave.com/res/v1/web/search?count=5&result_filter=web"
        "&safesearch=strict&spellcheck=false&text_decorations=false"
    )
    assert FAKE_KEY not in src.identity and FAKE_KEY not in repr(src)
    assert FAKE_KEY not in repr(src.config)
    assert brave(max_results=3)[0].identity != src.identity
    other = brave(endpoint="https://search.example.com/v1/web")[0].identity
    assert other != src.identity and "search.example.com" in other
    assert brave(timeout_s=30.0)[0].identity == src.identity  # timeout does not shape results


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"not json", "invalid JSON"),
        (b"\xff\xfe", "invalid JSON"),
        ([1, 2], "unexpected response shape"),
        ({"type": "error"}, "unexpected response shape"),
        ({"type": "search", "web": {"results": "x"}}, "unexpected response shape"),
        ({"type": "search", "web": []}, "unexpected response shape"),
        (brave_body(6), "returned 6 results; 5 were requested"),
        ({"type": "search", "web": {"results": [1]}}, "malformed result"),
        ({"type": "search", "web": {"results": [{"url": "x", "title": 1}]}}, "malformed result"),
        (
            {"type": "search", "web": {"results": [{"url": "javascript:alert(1)", "title": "t"}]}},
            "malformed result",
        ),
        ({"type": "search", "web": {"results": [{"url": "ftp://h/x", "title": "t"}]}}, "malformed"),
        (
            {
                "type": "search",
                "web": {"results": [{"url": "https://h/x", "title": "t", "description": 3}]},
            },
            "malformed result",
        ),
    ],
)
def test_malformed_responses_fail_closed(raw, message):
    src, _ = brave(raw)
    with pytest.raises(SourceError, match=message) as info:
        src.search(request())
    assert FAKE_KEY not in str(info.value)


def test_no_web_section_is_zero_results_not_an_error():
    assert parse_response(json.dumps({"type": "search"}).encode(), 5).results == ()


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (TimeoutError(), "timed out after 10.0s"),
        (urllib.error.HTTPError("u", 401, "nope " + FAKE_KEY, {}, None), "HTTP 401"),
        (urllib.error.URLError("dns " + FAKE_KEY), "unreachable"),
        (ConnectionResetError(), "unreachable"),
    ],
)
def test_transport_errors_fail_cleanly_once_without_the_key(exc, message):
    src, transport = brave(exc, brave_body())
    with pytest.raises(SourceError, match=message) as info:
        src.search(request())
    assert FAKE_KEY not in str(info.value) and info.value.__cause__ is None
    assert len(transport.calls) == 1  # no hidden retry


def test_a_request_must_match_the_configured_result_count():
    src, transport = brave(brave_body())
    with pytest.raises(SourceError, match="result count"):
        src.search(request(n=3))
    assert transport.calls == []


def test_redirects_are_never_followed():
    handler = _NoRedirect()
    req = urllib.request.Request(DEFAULT_ENDPOINT)
    assert handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example/") is None


@pytest.mark.parametrize(
    "cfg",
    [
        {"api_key": " "},
        {"endpoint": "http://api.search.brave.com/res/v1/web/search"},
        {"endpoint": "https://api.search.brave.com/res/v1/web/search?x=1"},
        {"endpoint": "https:///nohost"},
        {"timeout_s": 0},
        {"timeout_s": 61},
        {"max_results": 0},
        {"max_results": 11},
    ],
)
def test_unsafe_configurations_are_refused(cfg):
    with pytest.raises(WebSearchConfigError) as info:
        BraveSearchConfig(**{"api_key": FAKE_KEY, **cfg})
    assert FAKE_KEY not in str(info.value)


def test_websearch_is_configured_explicitly_from_the_environment():
    assert websearch_from_env({}) is None  # no key: websearch unavailable
    assert websearch_from_env({"WYNK_WEBSEARCH_API_KEY": "  "}) is None
    src = websearch_from_env(
        {
            "WYNK_WEBSEARCH_API_KEY": FAKE_KEY,
            "WYNK_WEBSEARCH_MAX_RESULTS": "3",
            "WYNK_WEBSEARCH_TIMEOUT_S": "4.5",
        }
    )
    assert (src.max_results, src.config.timeout_s) == (3, 4.5)
    assert websearch_from_env({"BRAVE_SEARCH_API_KEY": FAKE_KEY}).max_results == 5
    for bad in ({"WYNK_WEBSEARCH_TIMEOUT_S": "soon"}, {"WYNK_WEBSEARCH_MAX_RESULTS": "many"}):
        with pytest.raises(WebSearchConfigError):
            websearch_from_env({"WYNK_WEBSEARCH_API_KEY": FAKE_KEY, **bad})


def test_the_hosted_adapter_runs_a_genome_end_to_end(root):
    body = {
        "type": "search",
        "web": {
            "results": [{"url": "https://example.org/fr", "title": "France", "description": QUOTE}]
        },
    }
    src, transport = brave(body, max_results=1)
    result = workflow_runner(root, ScriptedModel(), src).run_sync(WS_GENOME, ws_task())
    assert result.failure is None and len(transport.calls) == 1
    assert result.key.versions.evidence_source == src.identity
    assert FAKE_KEY not in result.model_dump_json()


# == benchmark fairness ==========================================================================
def test_no_frozen_benchmark_suite_allows_websearch():
    from benchmarks.legacy_adapter import legacy_suite
    from benchmarks.loader import suite_names

    for name in suite_names():
        workflow = legacy_suite(name).policy.workflow
        assert WS not in workflow.allowed_sources, name
        assert workflow.live_sources == ()


def test_full_suite_eligibility_is_identical_with_or_without_websearch():
    from benchmarks.legacy_adapter import legacy_suite
    from benchmarks.loader import suite_names
    from experiments.full_suite import unavailable_reason

    base = (VerifyMethod.SELF_CONSISTENCY,)
    off = ConstraintChecker(config=ConstraintConfig(unavailable_verifiers=base))
    on = ConstraintChecker(
        config=ConstraintConfig(unavailable_verifiers=base, unavailable_sources=(WS,))
    )
    for name in suite_names():
        suite = legacy_suite(name)
        assert unavailable_reason(suite, off) == unavailable_reason(suite, on), name


def test_oss_baselines_keep_only_the_frozen_page_tools():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "experiments" / "oss_baselines"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "brave_search" not in text and "WebSearchSource" not in text, path


# == product / UI events =========================================================================
def test_streamed_stage_events_name_the_gather_source():
    from api.chat import stage_source

    genome = Genome.of(gather(WS), extract(), synth())
    assert stage_source(genome, 0) == {"source": "websearch"}
    assert stage_source(FETCH_GENOME, 0) == {"source": "fetch"}
    assert stage_source(genome, 1) == {} and stage_source(genome, 9) == {}


# == review round 1 (#57) ========================================================================
def question_only(**workflow):
    """A JSONL contract whose rows have a question and no context column."""
    from core.task_contract import ConstraintLimits, WorkflowSpec
    from core.task_spec import FieldType
    from tests.contract_helpers import col, qa_contract, qa_dataset, schema
    from tests.test_contract_runtime import FULL_CAPS

    dataset = qa_dataset(columns=(col("qid"), col("question"), col("answer")), context_columns=())
    return qa_contract(
        dataset=dataset,
        input_schema=schema(("question", FieldType.STRING)),
        constraints=ConstraintLimits(**FULL_CAPS),
        workflow=WorkflowSpec(**workflow),
    )


def test_a_question_only_dataset_can_opt_in_to_websearch():
    from core.task_contract import ContractError

    without = question_only()
    assert StageKind.GATHER not in without.workflow.stages  # unchanged without websearch
    opted = question_only(allowed_sources=(WS,))
    assert StageKind.GATHER in opted.workflow.stages
    assert codes(WS_GENOME, opted) == set()
    explicit = question_only(
        allowed_sources=(WS,),
        stages=(StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE),
    )
    assert codes(WS_GENOME, explicit) == set()
    with pytest.raises((ContractError, ValueError), match="can only be gathered with websearch"):
        question_only(allowed_sources=(GatherSource.FETCH, WS))  # fetch has nothing to read


@pytest.mark.parametrize("bad", ["key\nX-Evil: 1", "key\r\n", "k e y", "kéy", "key\x00"])
def test_a_key_an_http_header_cannot_carry_is_refused_without_echoing_it(bad):
    with pytest.raises(WebSearchConfigError) as info:
        BraveSearchConfig(api_key=bad)
    assert bad not in str(info.value)


def test_unexpected_transport_errors_never_carry_their_message():
    src, _ = brave(ValueError(f"Invalid header value {FAKE_KEY!r}"))
    with pytest.raises(SourceError, match=r"request failed \(ValueError\)") as info:
        src.search(request())
    assert FAKE_KEY not in str(info.value) and info.value.__cause__ is None


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://search.example.com/v1/web", "https://search.example.com:8443/v1/web"),
        ("https://search.example.com:8443/v1/web", "https://search.example.com:9443/v1/web"),
        ("https://search.example.com/v1/web", "https://search.example.com/v2/web"),
    ],
)
def test_different_endpoints_never_share_an_identity(a, b):
    assert brave(endpoint=a)[0].identity != brave(endpoint=b)[0].identity


def test_equivalent_endpoints_share_one_identity():
    plain = brave(endpoint="https://Search.Example.com/v1/web")[0].identity
    assert plain == brave(endpoint="https://search.example.com:443/v1/web")[0].identity
    assert "search.example.com/v1/web" in plain


@pytest.mark.parametrize(
    "endpoint", ["https://user:pw@search.example.com/v1", "https://search.example.com:99999/v1"]
)
def test_endpoints_with_credentials_or_bad_ports_are_refused(endpoint):
    with pytest.raises(WebSearchConfigError):
        BraveSearchConfig(api_key=FAKE_KEY, endpoint=endpoint)


@pytest.mark.parametrize("identity", ["custom-provider@1", "jev:pretend", "", "[x]"])
def test_a_websearch_provider_outside_the_websearch_namespace_is_refused(root, identity):
    with pytest.raises(ValueError, match="identity"):
        GatherExecutor(websearch=FakeWebSearch(identity=identity))
    with pytest.raises(ValueError, match="identity"):
        workflow_runner(root, ScriptedModel(), FakeWebSearch(identity=identity))


def test_a_jev_provider_cannot_claim_the_websearch_namespace(root):
    with pytest.raises(ValueError, match="must not start with 'websearch:'"):
        workflow_runner(root, ScriptedModel(), jev=ScriptedJev(identity="websearch:fake/x@1"))


# == review round 2 (#57) ========================================================================
def test_ipv6_endpoints_never_share_an_identity():
    a = brave(endpoint="https://[2001:db8::1]:8443/v1")[0].identity
    b = brave(endpoint="https://[2001:db8::1:8443]/v1")[0].identity
    assert a != b
    assert "@[2001:db8::1]:8443/v1?" in a and "@[2001:db8::1:8443]/v1?" in b
    assert (
        brave(endpoint="https://[2001:db8::1]:443/v1")[0].identity
        == brave(endpoint="https://[2001:db8::1]/v1")[0].identity
    )


class DriftingWebSearch(FakeWebSearch):
    """A provider whose identity changes after it was attached."""

    def drift(self, identity: str) -> None:
        self._identity = identity


@pytest.mark.parametrize("drifted", ["custom-provider@1", "websearch:fake/test@2"])
def test_a_provider_whose_identity_drifts_is_refused_and_not_recorded(root, drifted):
    ws = DriftingWebSearch()
    runner = workflow_runner(root, ScriptedModel(), ws)
    ws.drift(drifted)
    task = ws_task()
    # provenance keeps the identity the provider was attached with
    assert runner.run_key(WS_GENOME, task).versions.evidence_source == WS_ID
    assert runner.evidence_authority(task.contract) == WS_ID
    result = runner.run_sync(WS_GENOME, task)
    assert result.failure is not None and "identity changed" in result.failure.message
    assert ws.requests == []  # nothing was sent under the drifted identity
    assert is_websearch_identity(result.key.versions.evidence_source)  # never cacheable


def test_an_unbound_provider_is_refused_in_a_cold_job_too(tmp_path):
    """A durable job admits a run only if it reports a provider the job was bound with."""
    from store.jobs import JobState, UnitState
    from tests.test_memory_priors import DECLARED
    from tests.test_mixed_run_versions import SEARCH, definition, uses_jev

    def versions(g, t, tr):
        return DECLARED.model_copy(update={"evidence_source": WS_ID if uses_jev(g) else None})

    both = compose_authority(JEV_ID, WS_ID)
    ok = _LiveWorld.make(tmp_path / "bound", both, versions)  # websearch IS bound: admitted
    assert ok.jobs.get(ok.run(definition(SEARCH))).state is JobState.COMPLETED
    cold = _LiveWorld.make(tmp_path / "cold", JEV_ID, versions)  # no warm start involved
    job_id = cold.run(definition(SEARCH))
    units = cold.store.get_job(job_id).units
    assert any(u.state is UnitState.FAILED for u in units)
    assert cold.jobs.get(job_id).state is not JobState.COMPLETED


def test_a_direct_experiment_without_an_authority_is_not_checked():
    from experiments.optimization_experiment import UNBOUND, _Unbound

    assert isinstance(UNBOUND, _Unbound)  # run_optimization_experiment passes nothing


def test_a_jev_provider_whose_identity_drifts_is_refused(root):
    jev = ScriptedJev()
    runner = workflow_runner(root, ScriptedModel(), jev=jev)
    genome = Genome.of(gather(GatherSource.JEV), extract(), synth())
    jev._identity = "jev:scripted/test@2"
    task = ws_task()
    assert runner.run_key(genome, task).versions.evidence_source == JEV_ID  # pinned
    result = runner.run_sync(genome, task)
    assert result.failure is not None and "identity changed" in result.failure.message
    assert jev.observations == []  # no decision was requested under the drifted identity


# == review round 3 (#57) ========================================================================
class ShiftingWebSearch(FakeWebSearch):
    """A provider that reports ``first`` on its first identity read and its real one after."""

    def __init__(self, first: str) -> None:
        super().__init__()
        self._reads = [first]

    @property
    def identity(self) -> str:
        return self._reads.pop(0) if self._reads else self._identity


def test_a_provider_identity_is_read_and_validated_once(root):
    # the one read is the one validated: a first answer outside the namespace never passes
    with pytest.raises(ValueError, match="must start with 'websearch:'"):
        workflow_runner(root, ScriptedModel(), ShiftingWebSearch("jev:pretend"))
    with pytest.raises(ValueError, match="must start with 'websearch:'"):
        GatherExecutor(websearch=ShiftingWebSearch("jev:pretend"))
    # the runner records exactly the identity its executor runs under (one pinned object)
    ws = ShiftingWebSearch("websearch:fake/test@0")
    runner = workflow_runner(root, ScriptedModel(), ws)
    task = ws_task()
    assert runner.run_key(WS_GENOME, task).versions.evidence_source == "websearch:fake/test@0"
    result = runner.run_sync(WS_GENOME, task)  # the provider now reports another identity
    assert result.failure is not None and "identity changed" in result.failure.message
    assert ws.requests == []


class DriftingJev(ScriptedJev):
    """A JEV provider whose identity changes after its first decision."""

    def decide(self, observation):
        step = super().decide(observation)
        self._identity = "jev:scripted/test@2"
        return step


def test_a_jev_provider_that_drifts_between_decisions_is_refused(root):
    jev = DriftingJev([opens("p1"), opens("p2"), DONE])
    runner = workflow_runner(root, ScriptedModel(), jev=jev)
    genome = Genome.of(gather(GatherSource.JEV), extract(), synth())
    result = runner.run_sync(genome, ws_task())
    assert result.failure is not None and "identity changed" in result.failure.message
    assert len(jev.observations) == 1  # the second decision was never requested
    assert result.key.versions.evidence_source == JEV_ID


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://search.example.com/v1/web?",
        "https://search.example.com/v1/web#",
        "https://search.example.com/v1/web?x=1",
        "https://search.example.com/v1/web#top",
    ],
)
def test_endpoints_with_a_query_or_fragment_delimiter_are_refused(endpoint):
    with pytest.raises(WebSearchConfigError, match="'\\?' or '#'"):
        BraveSearchConfig(api_key=FAKE_KEY, endpoint=endpoint)


def test_the_request_goes_to_exactly_the_endpoint_the_identity_names():
    src, transport = brave(brave_body(), endpoint="https://Search.Example.com:443/v1/web")
    src.search(WebSearchRequest(query="capital of France", max_results=5, timeout_s=5))
    url = transport.calls[0][0]
    assert url.startswith("https://search.example.com/v1/web?q=capital+of+France&count=5&")
    assert "@search.example.com/v1/web?" in src.identity


def test_replay_refuses_a_stored_run_from_a_provider_the_job_was_not_bound_with(
    tmp_path, monkeypatch
):
    """A historical cold job bound with JEV A whose candidates reported B (old code did not
    check cold runs): replaying it fails closed instead of reproducing it."""
    from experiments import jobs
    from experiments.artifacts import ReproductionMismatch, replay
    from store.jobs import JobState
    from tests.test_mixed_run_versions import SEARCH, definition
    from tests.test_warm_start_evidence_provider import JEV_B, UNCHECKED, backend_for, world_at

    world = world_at(tmp_path / "ws", JEV_ID, backend_for(JEV_B))
    world.worker.priors = None
    run_strategy = jobs.run_strategy
    with monkeypatch.context() as m:  # old code: the worker did not check cold runs
        m.setattr(jobs, "run_strategy", lambda *a, **kw: run_strategy(*a, **{**kw, **UNCHECKED}))
        job_id = world.run(definition(SEARCH))
    assert world.jobs.get(job_id).state is JobState.COMPLETED
    with pytest.raises(ReproductionMismatch, match="bound with"):
        replay(world.store, job_id, world.runtime)


# == review round 4 (#57) ========================================================================
def test_a_drift_refusal_is_never_charged(root):
    jev = ScriptedJev([opens("p1"), DONE])
    ex = GatherExecutor(pages=CountingPages(root), jev=jev)
    jev._identity = "jev:scripted/test@2"  # drifted before the first decision
    task = ws_task()
    inp = ExecutorInput(stage_index=0, stage=gather(GatherSource.JEV), payload=task)
    out = asyncio.run(ex.run(inp, make_ctx(task)))
    assert "identity changed" in out.failure.message
    assert out.usage.tool_calls == 0 and jev.observations == []

    class LateDrift(FakeWebSearch):  # drifts between the executor's check and the request
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        @property
        def identity(self) -> str:
            self.reads += 1
            return self._identity if self.reads <= 2 else "websearch:fake/test@2"

    ws = LateDrift()
    out, _ = run_gather(root, ws)
    assert "identity changed" in out.failure.message
    assert out.usage.tool_calls == 0 and ws.requests == []
