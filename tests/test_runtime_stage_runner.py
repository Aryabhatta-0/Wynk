"""Stage execution, evidence, bounded retries and budget enforcement (framework-neutral)."""

import asyncio

import pytest

from core.genome import Genome
from core.results import BudgetCap, FailureKind, StageStatus
from core.stages import FailureStrategy, VerifyMethod
from runtime.mvp_genomes import GENOME_A, GENOME_B, GENOME_C
from runtime.spans import span_text
from tests.conftest import extract, gather, make_caps, make_task, synth, verify
from tests.runtime_helpers import (
    PAGES,
    QUOTE,
    FailingModel,
    ScriptedModel,
    build_runner,
    drive,
    write_snapshot,
)

BOGUS = "this sentence is nowhere in the pages"


@pytest.fixture
def root(tmp_path):
    write_snapshot(tmp_path)
    write_snapshot(tmp_path, "snap-big", extra_pages=3)  # 6 pages
    return tmp_path


def execute(genome, root, model, **task_kw):
    task = make_task(**task_kw)
    dag, runner = build_runner(genome, task, root, model)
    return asyncio.run(drive(dag, runner, task)), task


def templates(model):
    return [r.prompt_template_id for r in model.requests]


@pytest.mark.parametrize("genome", [GENOME_A, GENOME_B, GENOME_C], ids=["A", "B", "C"])
def test_mvp_genomes_execute_and_return_answer_with_structured_evidence(genome, root):
    model = ScriptedModel()
    result, _ = execute(genome, root, model)
    assert result.failure is None
    assert result.answer.values == {"capital": "Paris"}
    assert all(t.status is StageStatus.OK for t in result.stage_trace)
    assert len(result.stage_trace) == len(genome)
    [fe] = result.evidence
    assert fe.field == "capital"
    span = fe.spans[0]
    originals = {
        pid: __import__("core.payloads", fromlist=["Page"]).Page(
            page_id=pid, source_ref="x", content=text
        )
        for pid, text in PAGES.items()
    }
    assert span_text(span, originals) == QUOTE  # span really points at the quote
    assert span.page_id == "p1" and span.char_start == PAGES["p1"].index(QUOTE)


def test_real_metrics_are_populated_from_what_actually_ran(root):
    model = ScriptedModel()
    result, _ = execute(GENOME_B, root, model)  # filter + extract + reason + synthesize
    assert result.metrics.model_calls == 3 == len(model.requests)
    assert result.metrics.prompt_tokens == 300 and result.metrics.completion_tokens == 60
    assert result.budget_usage.tokens == 360
    assert result.budget_usage.tool_calls == 3 == result.metrics.pages_fetched
    assert result.budget_usage.retries == 0
    assert result.budget_usage.wall_time_s > 0
    assert templates(model) == [
        "extract.schema_guided",
        "reason.single",
        "synthesize.cite_evidence",
    ]


def test_genome_a_makes_exactly_two_model_calls_with_derived_seeds(root):
    model = ScriptedModel()
    execute(GENOME_A, root, model)
    assert templates(model) == ["extract.direct", "synthesize.direct"]
    seeds = [r.seed for r in model.requests]
    assert seeds[0] != seeds[1]
    model2 = ScriptedModel()
    execute(GENOME_A, root, model2)
    assert [r.seed for r in model2.requests] == seeds  # deterministic


def test_filter_reduces_what_the_model_sees(root):
    model = ScriptedModel()
    execute(GENOME_B, root, model)
    prompt = model.requests[0].input_text
    assert "Paris is the capital" in prompt and "bananas" not in prompt


def test_verify_retry_reruns_the_producer_once_and_recovers(root):
    model = ScriptedModel(extract_quotes=[BOGUS, QUOTE])
    result, _ = execute(GENOME_C, root, model)
    assert result.failure is None and result.answer.values == {"capital": "Paris"}
    assert templates(model) == ["extract.direct", "extract.direct", "synthesize.direct"]
    assert result.budget_usage.retries == 1
    statuses = [(t.kind.value, t.status.value) for t in result.stage_trace]
    assert ("VERIFY", "failed") in statuses and statuses.count(("VERIFY", "ok")) == 2
    assert model.requests[0].seed != model.requests[1].seed  # retry uses a fresh seed


def test_retries_are_bounded_by_the_genome_and_end_in_failure(root):
    for strategy, extra in ((FailureStrategy.RETRY_1, 1), (FailureStrategy.RETRY_2, 2)):
        genome = Genome.of(
            gather(), extract(), verify(VerifyMethod.EVIDENCE_SPAN, strategy), synth()
        )
        model = ScriptedModel(extract_quotes=[BOGUS])  # never fixes itself
        result, _ = execute(genome, root, model)
        assert templates(model).count("extract.direct") == 1 + extra
        assert "synthesize.direct" not in templates(model)  # never reached
        assert result.failure.kind is FailureKind.SCHEMA_INVALID
        assert result.failure.stage_index == 2
        assert result.budget_usage.retries == extra
        assert result.answer is None


def test_regather_reruns_gather_and_everything_up_to_the_verifier(root):
    genome = Genome.of(
        gather(),
        extract(),
        verify(VerifyMethod.EVIDENCE_SPAN, FailureStrategy.REGATHER),
        synth(),
    )
    model = ScriptedModel(extract_quotes=[BOGUS, QUOTE])
    result, _ = execute(genome, root, model)
    assert result.failure is None
    assert result.budget_usage.tool_calls == 6  # gathered twice
    assert result.budget_usage.retries == 1
    assert templates(model).count("extract.direct") == 2


def test_retry_that_would_cross_the_retries_cap_stops_the_run(root):
    model = ScriptedModel(extract_quotes=[BOGUS, QUOTE])
    result, _ = execute(GENOME_C, root, model, caps=make_caps(retries=0))
    assert result.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert result.failure.cap is BudgetCap.RETRIES
    assert templates(model) == ["extract.direct"]  # the retry never ran


def test_tool_call_breach_stops_execution_before_any_model_call(root):
    # static estimate says 3 calls (ok for the cap); the 6-page snapshot actually needs 6
    result, _ = execute(
        GENOME_A, root, ScriptedModel(), snapshot_id="snap-big", caps=make_caps(tool_calls=3)
    )
    assert result.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert result.failure.cap is BudgetCap.TOOL_CALLS and result.failure.stage_index == 0
    assert [t.kind.value for t in result.stage_trace] == ["GATHER"]
    assert result.stage_trace[0].status is StageStatus.FAILED
    assert result.answer is None


def test_token_breach_stops_after_the_offending_stage(root):
    model = ScriptedModel()
    model.tokens = (1500, 500)  # 2000 per call; cap 2300 allows extract (2000) but not synth
    genome = Genome.of(gather(), extract(), synth(), verify())
    result, _ = execute(genome, root, model, caps=make_caps(tokens=2300))
    assert result.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert result.failure.cap is BudgetCap.TOKENS and result.failure.stage_index == 2
    assert [t.kind.value for t in result.stage_trace] == ["GATHER", "EXTRACT", "SYNTHESIZE"]
    assert result.budget_usage.tokens == 4000  # real usage is reported, not hidden


def test_wall_time_cap_is_enforced_by_the_guard(root):
    task = make_task(caps=make_caps(wall_time_s=0.001))
    dag, runner = build_runner(GENOME_A, task, root, ScriptedModel())

    async def slow_run():
        import time

        orig = runner._guarded[next(iter(runner._guarded))].inner.run

        async def slow(inp, ctx):
            time.sleep(0.01)
            return await orig(inp, ctx)

        runner._guarded[next(iter(runner._guarded))].inner.run = slow
        return await drive(dag, runner, task)

    result = asyncio.run(slow_run())
    assert result.failure.cap is BudgetCap.WALL_TIME


def test_failures_are_reported_never_faked(root):
    result, _ = execute(GENOME_A, root, None)  # no model configured
    assert result.failure.kind is FailureKind.MODEL_ERROR and result.answer is None
    result, _ = execute(GENOME_A, root, FailingModel())
    assert result.failure.kind is FailureKind.MODEL_ERROR
    assert "backend down" in result.failure.message
    assert [t.status for t in result.stage_trace] == [StageStatus.OK, StageStatus.FAILED]


def test_unparseable_model_output_is_a_schema_failure_not_a_guess(root):
    class Garbage(ScriptedModel):
        async def generate(self, request):
            from runtime.gemma_client import GenerationResponse

            return GenerationResponse(
                text="I think it's Paris!", prompt_tokens=5, completion_tokens=5, model_hash="g"
            )

    result, _ = execute(GENOME_A, root, Garbage())
    assert result.failure.kind is FailureKind.SCHEMA_INVALID
    assert result.budget_usage.tokens == 10  # tokens were still spent and are accounted


def test_unexpected_executor_exception_becomes_a_failure_not_a_crash(root):
    task = make_task()
    dag, runner = build_runner(GENOME_A, task, root, ScriptedModel())

    async def boom(inp, ctx):
        raise RuntimeError("kaput")

    for g in runner._guarded.values():
        g.inner.run = boom
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure.kind is FailureKind.EXECUTOR_ERROR and "kaput" in result.failure.message


def test_self_consistency_is_explicitly_unsupported_in_the_mvp(root):
    genome = Genome.of(gather(), extract(), verify(VerifyMethod.SELF_CONSISTENCY), synth())
    result, _ = execute(genome, root, ScriptedModel())
    assert result.failure.kind is FailureKind.EXECUTOR_ERROR
    assert "not in MVP" in result.failure.message
