"""GATHER jev as a real execution step: WorkflowRunner -> compiler -> StageRunner ->
GatherExecutor -> JevSource -> Pages. Every provider here is a deterministic local double
(``ScriptedJev`` or a fake HTTP transport); nothing touches the network or a browser."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error

import pytest

from core.constraints import ConstraintChecker, ConstraintConfig
from core.genome import Genome
from core.results import BudgetCap, FailureKind, RunVersions, StageStatus
from core.stages import FailureStrategy, GatherMode, GatherSource, StageKind, VerifyMethod
from core.violations import ViolationCode as V
from runtime.backends.typesafe_jev import (
    DEFAULT_ENDPOINT,
    TypeSafeJevConfig,
    TypeSafeJevSource,
    jev_from_env,
)
from runtime.executors.base import ExecutorInput
from runtime.executors.gather import GatherExecutor
from runtime.sources import (
    DirectoryApiSource,
    DirectorySnapshotSource,
    JevDecision,
    JevElement,
    JevObservation,
    SourceError,
    checked_decision,
)
from tests.conftest import extract, flt, gather, make_caps, make_task, synth, verify
from tests.runtime_helpers import (
    DONE,
    PAGES,
    QUOTE,
    ScriptedJev,
    ScriptedModel,
    build_runner,
    drive,
    make_ctx,
    opens,
    write_snapshot,
)

JEV = GatherSource.JEV
FAKE_KEY = "test-jev-key-0000-not-a-real-credential"
BOGUS = "this sentence is nowhere in the pages"


class CountingPages(DirectorySnapshotSource):
    """The snapshot surface, recording every page actually opened."""

    def __init__(self, root, delays: dict[str, float] | None = None) -> None:
        super().__init__(root)
        self.reads: list[str] = []
        self.delays = delays or {}

    def read_page(self, snapshot_id, page_id):
        time.sleep(self.delays.get(page_id, 0.0))
        self.reads.append(page_id)
        return super().read_page(snapshot_id, page_id)


@pytest.fixture
def root(tmp_path):
    write_snapshot(tmp_path)
    return tmp_path


def run_gather(root, jev, mode=GatherMode.SEQUENTIAL, pages=None, task=None):
    task = task or make_task()
    pages = pages or CountingPages(root)
    ex = GatherExecutor(pages=pages, api=DirectoryApiSource(root), jev=jev)
    inp = ExecutorInput(stage_index=0, stage=gather(JEV, mode), payload=task)
    return asyncio.run(ex.run(inp, make_ctx(task))), pages


def workflow_runner(root, model, jev=None, pages=None):
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner

    return WorkflowRunner(
        model=model,
        benchmark_hash="bench-1",
        pages=pages or DirectorySnapshotSource(root),
        api=DirectoryApiSource(root),
        jev=jev,
    )


JEV_GENOME = Genome.of(gather(JEV), extract(), synth())


# -- execution ----------------------------------------------------------------------------------


def test_jev_genome_executes_end_to_end_through_the_workflow_runner(root):
    jev = ScriptedJev([opens("p1"), DONE])
    model = ScriptedModel()
    result = workflow_runner(root, model, jev).run_sync(JEV_GENOME, make_task(), seed=3)
    assert result.failure is None
    assert result.answer.values == {"capital": "Paris"}
    assert [t.status for t in result.stage_trace] == [StageStatus.OK] * 3
    assert result.stage_trace[0].kind is StageKind.GATHER
    # 2 decisions + 1 opened page; the model stages never count as tool calls
    assert result.stage_trace[0].usage.tool_calls == 3
    assert result.budget_usage.tool_calls == 3
    assert result.metrics.pages_fetched == 1


def test_jev_evidence_reaches_extraction_and_synthesis(root):
    jev = ScriptedJev([opens("p1"), DONE])
    model = ScriptedModel()
    result = workflow_runner(root, model, jev).run_sync(JEV_GENOME, make_task())
    extract_input = next(r for r in model.requests if r.prompt_template_id.startswith("extract"))
    assert QUOTE in extract_input.input_text  # the opened page is what EXTRACT read
    assert "Berlin" not in extract_input.input_text  # an unopened page is not
    span = result.evidence[0].spans[0]
    assert span.page_id == "p1"


def test_jev_uses_the_task_question_as_goal_and_the_snapshot_as_surface(root):
    jev = ScriptedJev([opens("p2"), DONE])
    task = make_task(question="Which city is the capital of Germany?")
    out, _ = run_gather(root, jev, task=task)
    first = jev.observations[0]
    assert first.goal == task.question and "Germany" in first.goal
    assert first.surface == "snapshot://snap-001"
    assert [e.element_id for e in first.elements] == ["p1", "p2", "p3"]
    second = jev.observations[1]
    assert [e.element_id for e in second.elements] == ["p1", "p3"]  # only unopened pages
    assert [p.page_id for p in second.opened] == ["p2"]
    # the agent sees the question, the page ids and the pages it opened - no answer schema,
    # targets or other run state exists on an observation
    assert set(JevObservation.model_fields) == {
        "goal",
        "surface",
        "elements",
        "opened",
        "width",
        "step",
    }


def test_jev_pages_keep_snapshot_ids_and_refs_and_come_back_sorted(root):
    out, pages = run_gather(root, ScriptedJev([opens("p3"), opens("p1"), DONE]))
    assert pages.reads == ["p3", "p1"]  # opened in the agent's order...
    assert [p.page_id for p in out.payload.pages] == ["p1", "p3"]  # ...returned sorted
    assert out.payload.pages[0].source_ref == "snapshot://snap-001/p1.txt"
    assert out.payload.pages[0].content == PAGES["p1"]
    again, _ = run_gather(root, ScriptedJev([opens("p3"), opens("p1"), DONE]))
    assert again.payload == out.payload


def test_parallel_2_ordering_does_not_depend_on_completion_order(root):
    slow_first = CountingPages(root, delays={"p1": 0.2})
    out, pages = run_gather(
        root, ScriptedJev([opens("p1", "p2"), DONE]), GatherMode.PARALLEL_2, slow_first
    )
    assert pages.reads == ["p2", "p1"]  # p1 finished last
    assert [p.page_id for p in out.payload.pages] == ["p1", "p2"]


def test_every_page_open_ends_the_loop_without_another_decision(root):
    jev = ScriptedJev([opens("p1"), opens("p2"), opens("p3")])
    out, _ = run_gather(root, jev)
    assert len(jev.observations) == 3  # no 4th decision: nothing left to choose
    assert out.usage.tool_calls == 6


# -- budget -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "script", "calls"),
    [
        (GatherMode.SEQUENTIAL, [opens("p1"), DONE], 3),
        (GatherMode.SEQUENTIAL, [opens("p1"), opens("p2"), DONE], 5),
        (GatherMode.PARALLEL_2, [opens("p1", "p2"), DONE], 4),
        (GatherMode.PARALLEL_2, [DONE], 1),
    ],
)
def test_jev_actions_are_charged_as_tool_calls(root, mode, script, calls):
    out, pages = run_gather(root, ScriptedJev(script), mode)
    if out.failure is None:
        assert out.usage.tool_calls == calls
    else:  # DONE before opening anything: no evidence, but the decision was still paid for
        assert out.failure.kind is FailureKind.EXECUTOR_ERROR
        assert out.usage.tool_calls == calls
    assert out.usage.tokens == 0  # the JEV model is the provider's, not a Wynk model call
    assert out.metrics.pages_fetched == len(pages.reads)


def test_tool_call_cap_terminates_jev_mid_stage(root):
    write_snapshot(root, "snap-big", extra_pages=8)  # 11 pages
    want = ["p1", "p2", "p3", "x0", "x1", "x2", "x3"]
    jev = ScriptedJev([opens(n) for n in want])
    model = ScriptedModel()
    pages = CountingPages(root)
    # 10 = the admission estimate for jev, so the run is admitted; the agent wants 14 actions
    task = make_task(snapshot_id="snap-big", caps=make_caps(tool_calls=10))
    result = workflow_runner(root, model, jev, pages).run_sync(JEV_GENOME, task)
    assert result.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert result.failure.cap is BudgetCap.TOOL_CALLS
    assert result.failure.stage_index == 0
    # 5 x (decision + open) = 10 = cap; the 6th decision crosses it and nothing more is opened
    assert result.budget_usage.tool_calls == 11
    assert len(jev.observations) == 6 and pages.reads == want[:5]
    assert model.requests == []  # nothing downstream ran


def test_regather_would_charge_jev_again_which_is_why_admission_forbids_it(root):
    genome = Genome.of(
        gather(JEV),
        extract(),
        verify(VerifyMethod.EVIDENCE_SPAN, FailureStrategy.REGATHER),
        synth(),
    )
    with pytest.raises(Exception, match="jev_regather"):
        workflow_runner(root, ScriptedModel(), ScriptedJev([opens("p1")])).run_sync(
            genome, make_task()
        )
    # Below admission: the stage runner's regather really re-runs the JEV gather and pays again.
    jev = ScriptedJev([opens("p1"), DONE, opens("p1"), DONE])
    task = make_task()
    dag, runner = build_runner(genome, task, root, ScriptedModel([BOGUS, QUOTE]), jev=jev)
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure is None
    assert len(jev.observations) == 4
    gathers = [t for t in result.stage_trace if t.kind is StageKind.GATHER]
    assert [t.usage.tool_calls for t in gathers] == [3, 3]
    assert result.budget_usage.tool_calls == 6 and result.budget_usage.retries == 1


def test_retry_after_a_jev_gather_does_not_rerun_the_gather(root):
    genome = Genome.of(gather(JEV), extract(), verify(VerifyMethod.EVIDENCE_SPAN), synth())
    jev = ScriptedJev([opens("p1"), DONE])
    task = make_task()
    dag, runner = build_runner(genome, task, root, ScriptedModel([BOGUS, QUOTE]), jev=jev)
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure is None
    assert len(jev.observations) == 2 and result.budget_usage.tool_calls == 3


# -- availability: no fallback ------------------------------------------------------------------


def test_unavailable_jev_is_refused_before_any_work_and_never_replaced_by_fetch(root):
    model = ScriptedModel()
    pages = CountingPages(root)
    runner = workflow_runner(root, model, jev=None, pages=pages)
    with pytest.raises(Exception, match="runtime_unavailable") as info:
        runner.run_sync(JEV_GENOME, make_task())
    assert [v.code for v in info.value.violations] == [V.RUNTIME_UNAVAILABLE]
    assert pages.reads == [] and model.requests == []


def test_executor_without_a_jev_source_fails_explicitly_without_reading_pages(root):
    out, pages = run_gather(root, jev=None)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert "jev" in out.failure.message and "not available" in out.failure.message
    assert pages.reads == [] and out.usage.tool_calls == 0


def test_a_jev_source_needs_the_page_surface_it_browses():
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner

    with pytest.raises(ValueError, match="PageSource"):
        WorkflowRunner(model=None, benchmark_hash="b", jev=ScriptedJev())


# -- source errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ([SourceError("provider down")], "provider down"),
        ([opens("p9")], "not offered: p9"),
        ([opens("p1", "p2")], "allows 1"),  # sequential: one page per decision
        (["open p1 please"], "unstructured decision"),
        ([JevDecision(operation="blocked")], "agent stopped: blocked"),
        ([DONE], "gathered no pages"),
    ],
)
def test_source_errors_become_executor_errors(root, script, message):
    out, pages = run_gather(root, ScriptedJev(script))
    assert out.payload is None
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert message in out.failure.message
    assert pages.reads == []  # an invalid decision opens nothing
    assert out.usage.tool_calls == 1


def test_a_page_read_failure_is_an_executor_error(root):
    class Broken(CountingPages):
        def read_page(self, snapshot_id, page_id):
            raise SourceError(f"page not found: {snapshot_id}/{page_id}")

    out, _ = run_gather(root, ScriptedJev([opens("p1")]), pages=Broken(root))
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert "page not found" in out.failure.message
    assert out.usage.tool_calls == 2


def test_unknown_snapshot_fails_before_any_decision(root):
    jev = ScriptedJev([opens("p1")])
    out, _ = run_gather(root, jev, task=make_task(snapshot_id="nope"))
    assert "unknown snapshot" in out.failure.message
    assert jev.observations == []


def test_decisions_are_structurally_bounded():
    with pytest.raises(ValueError):
        JevDecision(operation="open")
    with pytest.raises(ValueError):
        JevDecision(operation="done", targets=("p1",))
    with pytest.raises(ValueError):
        JevDecision(operation="open", targets=("p1", "p1"))
    with pytest.raises(ValueError):
        JevDecision.model_validate({"operation": "eval", "targets": ["document.cookie"]})
    obs = JevObservation(
        goal="g", surface="snapshot://s", elements=(JevElement(element_id="a", label="a"),)
    )
    assert checked_decision(opens("a"), obs) == opens("a")


# -- constraints --------------------------------------------------------------------------------


def jev_checker() -> ConstraintChecker:
    return ConstraintChecker(
        config=ConstraintConfig(unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,))
    )


def codes(genome, **contract_kw):
    contract = make_task(**contract_kw).contract
    return {v.code for v in jev_checker().check(genome, contract, complete=True)}


@pytest.mark.parametrize(
    ("genome", "expected", "contract"),
    [
        (Genome.of(gather(JEV), extract(), synth()), set(), {}),
        (Genome.of(gather(JEV, GatherMode.PARALLEL_2), extract(), synth()), set(), {}),
        (
            Genome.of(gather(JEV), extract(), verify(on_failure=FailureStrategy.RETRY_1), synth()),
            set(),
            {},
        ),
        (
            Genome.of(gather(JEV, GatherMode.PARALLEL_4), extract(), synth()),
            {V.JEV_PARALLEL_4},
            {},
        ),
        (
            Genome.of(gather(JEV), extract(), verify(on_failure=FailureStrategy.REGATHER), synth()),
            {V.JEV_REGATHER},
            {},
        ),
        (Genome.of(gather(), extract(), synth()), {V.INTERACTION_REQUIRES_JEV}, {"i": True}),
        (
            Genome.of(gather(GatherSource.API), extract(), synth()),
            {V.INTERACTION_REQUIRES_JEV},
            {"i": True},
        ),
        (Genome.of(gather(JEV), extract(), synth()), set(), {"i": True}),
        (
            Genome.of(gather(JEV), extract(), synth()),
            {V.SOURCE_NOT_ALLOWED},
            {"allowed_sources": (GatherSource.FETCH,)},
        ),
    ],
)
def test_jev_constraints_with_jev_available(genome, expected, contract):
    kw = {"interaction_required": True} if contract.pop("i", False) else {}
    assert codes(genome, **kw, **contract) == expected


def test_jev_constraints_are_enforced_by_the_runner_that_can_execute_jev(root):
    runner = workflow_runner(root, ScriptedModel(), ScriptedJev([opens("p1"), DONE]))
    bad = Genome.of(gather(JEV, GatherMode.PARALLEL_4), extract(), synth())
    with pytest.raises(Exception, match="jev_parallel_4"):
        runner.run_sync(bad, make_task())
    with pytest.raises(Exception, match="interaction_requires_jev"):
        runner.run_sync(
            Genome.of(gather(), extract(), synth()), make_task(interaction_required=True)
        )
    ok = runner.run_sync(
        Genome.of(gather(JEV, GatherMode.PARALLEL_2), extract(), synth()),
        make_task(interaction_required=True),
    )
    assert ok.failure is None


def test_executor_refuses_parallel_4_even_past_admission(root):
    jev = ScriptedJev([opens("p1")])
    out, pages = run_gather(root, jev, GatherMode.PARALLEL_4)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR and "parallel-4" in out.failure.message
    assert jev.observations == [] and pages.reads == []


def test_contracts_without_jev_behave_exactly_as_before():
    fetch_only = {"allowed_sources": (GatherSource.FETCH, GatherSource.API)}
    off = ConstraintChecker(
        config=ConstraintConfig(
            unavailable_sources=(JEV,), unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,)
        )
    )
    contract = make_task(**fetch_only).contract
    for genome in (
        Genome.of(gather(), extract(), synth()),
        Genome.of(gather(GatherSource.API, GatherMode.PARALLEL_4), extract(), synth()),
        Genome.of(gather(), extract(), verify(on_failure=FailureStrategy.REGATHER), synth()),
        Genome.of(gather(), flt(), extract(), synth()),
    ):
        assert off.check(genome, contract, complete=True) == jev_checker().check(
            genome, contract, complete=True
        )
    # a jev genome stays inadmissible on a contract that never allows jev, available or not
    jev_genome = Genome.of(gather(JEV), extract(), synth())
    assert {v.code for v in jev_checker().check(jev_genome, contract, complete=True)} == {
        V.SOURCE_NOT_ALLOWED
    }
    assert V.SOURCE_NOT_ALLOWED in {v.code for v in off.check(jev_genome, contract, complete=True)}


# -- reproducibility ----------------------------------------------------------------------------


def test_jev_runs_record_the_provider_identity_and_frozen_runs_do_not(root):
    model = ScriptedModel()
    with_jev = workflow_runner(root, model, ScriptedJev([opens("p1"), DONE]))
    without = workflow_runner(root, model)
    fetch = Genome.of(gather(), extract(), synth())
    task = make_task()

    jev_key = with_jev.run_key(JEV_GENOME, task)
    assert jev_key.versions.evidence_source == "jev:scripted/test@1"
    assert jev_key.versions.model_dump(mode="json")["evidence_source"] == "jev:scripted/test@1"

    # a frozen fetch run has the same identity whether or not JEV is configured
    assert with_jev.run_key(fetch, task) == without.run_key(fetch, task)
    assert "evidence_source" not in without.run_key(fetch, task).versions.model_dump(mode="json")
    assert jev_key.versions.benchmark_hash == without.versions(task).benchmark_hash
    a = with_jev.run_sync(fetch, task)
    b = without.run_sync(fetch, task)
    assert a.run_id == b.run_id and a.answer == b.answer and a.evidence == b.evidence
    assert a.budget_usage.tool_calls == b.budget_usage.tool_calls == 3


def test_run_versions_without_a_live_source_serialize_exactly_as_before():
    legacy = {
        "model_hash": "m",
        "prompt_template_version": "p",
        "benchmark_hash": "b",
        "compiler_version": "c",
        "grammar_version": "g",
    }
    v = RunVersions.model_validate(legacy)
    assert v.model_dump(mode="json") == legacy
    assert RunVersions.model_validate_json(v.model_dump_json()) == v
    live = RunVersions(**legacy, evidence_source="jev:x/y@z")
    assert RunVersions.model_validate(live.model_dump(mode="json")) == live
    assert live != v


# -- hosted adapter (fake transport only) -------------------------------------------------------


def choice(chosen, probs, confidence=0.9):
    return {"choice": chosen, "probabilities": probs, "confidence": confidence}


class FakeTransport:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, dict, float]] = []

    def __call__(self, url, body, headers, timeout_s):
        self.calls.append((url, json.loads(body), dict(headers), timeout_s))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r if isinstance(r, bytes) else json.dumps(r).encode()


def typesafe(*responses, **cfg):
    transport = FakeTransport(*responses)
    return TypeSafeJevSource(TypeSafeJevConfig(api_key=FAKE_KEY, **cfg), transport), transport


def observation(width=1, opened=()):
    return JevObservation(
        goal="What is the capital of France?",
        surface="snapshot://snap-001",
        elements=tuple(JevElement(element_id=n, label=n) for n in ("p1", "p2", "p3")),
        opened=opened,
        width=width,
    )


OPEN_P2 = {
    "model": "jev-1",
    "answers": {
        "operation": choice("OPEN", {"OPEN": 0.8, "DONE": 0.1, "BLOCKED": 0.1}),
        "open_target": choice("2", {"1": 0.3, "2": 0.6, "3": 0.1}),
    },
}


def test_typesafe_adapter_asks_bounded_questions_and_maps_the_answer():
    jev, transport = typesafe(OPEN_P2)
    decision = jev.decide(observation())
    assert decision == JevDecision(operation="open", targets=("p2",), confidence=0.9)
    url, body, headers, timeout = transport.calls[0]
    assert url == DEFAULT_ENDPOINT and timeout == 25.0
    assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(body)
    assert set(body["questions"]) == {"operation", "open_target"}
    assert set(body["questions"]["operation"]["criteria"]) == {"OPEN", "DONE", "BLOCKED"}
    assert set(body["questions"]["open_target"]["criteria"]) == {"1", "2", "3"}


def test_typesafe_parallel_2_adds_the_next_most_probable_page_only_if_possible():
    jev, _ = typesafe(OPEN_P2)
    assert jev.decide(observation(width=2)).targets == ("p2", "p1")
    certain = {
        "answers": {
            "operation": choice("OPEN", {"OPEN": 1.0, "DONE": 0.0, "BLOCKED": 0.0}),
            "open_target": choice("3", {"1": 0.0, "2": 0.0, "3": 1.0}),
        }
    }
    jev, _ = typesafe(certain)
    assert jev.decide(observation(width=2)).targets == ("p3",)


def test_typesafe_done_and_blocked():
    for op in ("DONE", "BLOCKED"):
        probs = {"OPEN": 0.1, "DONE": 0.1, "BLOCKED": 0.1, op: 0.8}
        jev, _ = typesafe({"answers": {"operation": choice(op, probs)}})
        assert jev.decide(observation()).operation == op.lower()


@pytest.mark.parametrize(
    "answers",
    [
        {},  # no operation answer
        {"operation": choice("CLICK", {"OPEN": 0.5, "DONE": 0.5, "BLOCKED": 0.0})},
        {"operation": choice("DONE", {"OPEN": 0.9, "DONE": 0.1, "BLOCKED": 0.0})},  # not argmax
        {"operation": choice("DONE", {"OPEN": 0.5, "DONE": 0.9, "BLOCKED": 0.0})},  # sum != 1
        {"operation": choice("DONE", {"DONE": 1.0})},  # not the offered set
        {"operation": choice("DONE", {"OPEN": 0, "DONE": 1, "BLOCKED": 0}, confidence=2)},
        {
            "operation": choice("OPEN", {"OPEN": 1.0, "DONE": 0.0, "BLOCKED": 0.0}),
            "open_target": choice("4", {"1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}),
        },
        {
            "operation": choice("OPEN", {"OPEN": 1.0, "DONE": 0.0, "BLOCKED": 0.0}),
            "open_target": {"choice": "//a[@id='x']", "selector": "#x"},
        },
    ],
)
def test_typesafe_rejects_anything_but_a_valid_offered_choice(answers):
    jev, _ = typesafe({"answers": answers})
    with pytest.raises(SourceError, match="no action taken"):
        jev.decide(observation())


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError(DEFAULT_ENDPOINT, 401, f"bad key {FAKE_KEY}", {}, None),
        urllib.error.URLError(f"refused for {FAKE_KEY}"),
        TimeoutError(f"slow {FAKE_KEY}"),
        ConnectionResetError(FAKE_KEY),
        b"<html>not json</html>",
        b"[]",
    ],
)
def test_typesafe_failures_are_source_errors_without_credentials(failure):
    jev, transport = typesafe(failure)
    with pytest.raises(SourceError) as info:
        jev.decide(observation())
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None  # the transport error (and its text) is not chained
    assert info.value.__suppress_context__ or info.value.__context__ is None
    assert len(transport.calls) == 1  # never retried


def test_credentials_never_reach_results_reprs_or_identity(root):
    jev, _ = typesafe(urllib.error.HTTPError(DEFAULT_ENDPOINT, 403, FAKE_KEY, {}, None))
    for text in (repr(jev), repr(jev.config), jev.identity):
        assert FAKE_KEY not in text
    assert jev.identity == "jev:typesafe-jev/jev-latest@api.typesafe.ai/v1/systemone"
    result = workflow_runner(root, ScriptedModel(), jev).run_sync(JEV_GENOME, make_task())
    assert result.failure.kind is FailureKind.EXECUTOR_ERROR
    assert "HTTP 403" in result.failure.message
    assert FAKE_KEY not in result.model_dump_json()


def test_typesafe_over_the_full_runtime_with_a_fake_transport(root):
    jev, transport = typesafe(
        {
            "answers": {
                "operation": choice("OPEN", {"OPEN": 0.9, "DONE": 0.05, "BLOCKED": 0.05}),
                "open_target": choice("1", {"1": 0.9, "2": 0.05, "3": 0.05}),
            }
        },
        {"answers": {"operation": choice("DONE", {"OPEN": 0.1, "DONE": 0.9, "BLOCKED": 0.0})}},
    )
    result = workflow_runner(root, ScriptedModel(), jev).run_sync(JEV_GENOME, make_task())
    assert result.failure is None and result.answer.values == {"capital": "Paris"}
    assert result.budget_usage.tool_calls == 3
    second = transport.calls[1][1]
    assert "Paris is the capital" in second["state"]["page"]["text"]  # it saw what it opened
    assert set(second["questions"]["open_target"]["criteria"]) == {"1", "2"}  # p2, p3 left


def test_jev_configuration_is_explicit():
    assert jev_from_env({}) is None
    assert jev_from_env({"WYNK_JEV_API_KEY": "  "}) is None
    jev = jev_from_env({"WYNK_JEV_API_KEY": FAKE_KEY, "WYNK_JEV_MODEL": "jev-2026-09"})
    assert jev is not None and jev.identity.startswith("jev:typesafe-jev/jev-2026-09@")
    assert jev_from_env({"TYPESAFE_API_KEY": FAKE_KEY}) is not None
    for bad in (
        {"endpoint": "http://api.typesafe.ai/v1/systemone"},
        {"timeout_s": 0},
        {"timeout_s": 10_000},
        {"model": " "},
    ):
        with pytest.raises(ValueError) as info:
            TypeSafeJevConfig(api_key=FAKE_KEY, **bad)
        assert FAKE_KEY not in str(info.value)
    with pytest.raises(ValueError):
        TypeSafeJevConfig(api_key="")


# -- product event path -------------------------------------------------------------------------


def test_chat_streams_a_jev_stage_like_any_other_gather_stage(root):
    pytest.importorskip("agent_framework")
    from api.chat import run_with_progress  # noqa: PLC0415

    events = []
    runner = workflow_runner(root, ScriptedModel(), ScriptedJev([opens("p1"), DONE]))
    result = run_with_progress(runner, JEV_GENOME, make_task(), lambda *e: events.append(e))
    assert result.failure is None
    assert events[0] == (0, "running", 0, None, 0)
    assert events[1] == (0, "ok", 0, None, 3)  # the GATHER stage: 3 JEV tool calls, no tokens
    assert [e[1] for e in events] == ["running", "ok"] * 3
    assert result.key.versions.evidence_source == "jev:scripted/test@1"
