"""Shared result / evidence / cost / store contracts."""

import pytest
from pydantic import ValidationError

from core.cost_model import CostTable, StaticCostModel, exceeded_caps
from core.evidence import EvidenceSpan, FieldEvidence
from core.genome import Genome
from core.payloads import Answer, Page
from core.results import (
    BudgetCap,
    BudgetUsage,
    EvaluatedRun,
    Evaluation,
    ExecutionResult,
    RunKey,
    RunVersions,
    Verdict,
)
from core.stages import FailureStrategy, GatherMode, GatherSource
from store.runs import InMemoryRunStore
from tests.conftest import (
    extract,
    gather,
    make_caps,
    make_runtime_task,
    make_task_spec,
    minimal_genome,
    synth,
    verify,
)

H = "a" * 64


def versions(**kw) -> RunVersions:
    base = dict(
        model_hash="m1",
        prompt_template_version="p1",
        benchmark_hash="b1",
        compiler_version="c1",
        grammar_version="g1",
    )
    return RunVersions(**{**base, **kw})


def key(**kw) -> RunKey:
    base = dict(
        genome_hash=minimal_genome().genome_hash,
        task_id="task-001",
        contract_hash="k1",
        trial=0,
        seed=7,
        versions=versions(),
    )
    return RunKey(**{**base, **kw})


# -- evidence -----------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        {"char_start": 5, "char_end": 5},
        {"char_start": 9, "char_end": 3},
        {"char_start": -1, "char_end": 3},
        {"content_hash": "not-a-hash"},
        {"content_hash": "A" * 64},
        {"page_id": ""},
    ],
)
def test_evidence_span_validation(bad):
    ok = dict(page_id="p1", char_start=0, char_end=10, content_hash=H)
    EvidenceSpan(**ok)
    with pytest.raises(ValidationError):
        EvidenceSpan(**{**ok, **bad})


def test_page_spans_are_pinned_to_the_page_content_hash():
    page = Page(page_id="p1", source_ref="snap://x", content="Paris is the capital.")
    span = page.span(0, 5)
    assert span.content_hash == page.content_hash
    assert (
        Page(page_id="p1", source_ref="snap://x", content="other").content_hash != span.content_hash
    )


def test_field_evidence_needs_at_least_one_span():
    with pytest.raises(ValidationError):
        FieldEvidence(field="capital", spans=())


# -- run identity -------------------------------------------------------------
def test_run_id_is_deterministic_and_sensitive_to_every_identity_part():
    assert key().run_id == key().run_id
    variants = [
        key(trial=1),
        key(seed=8),
        key(task_id="t2"),
        key(contract_hash="k2"),
        key(genome_hash="0" * 64),
        key(versions=versions(model_hash="m2")),
        key(versions=versions(prompt_template_version="p2")),
        key(versions=versions(benchmark_hash="b2")),
        key(versions=versions(compiler_version="c2")),
        key(versions=versions(grammar_version="g2")),
    ]
    assert len({v.run_id for v in variants} | {key().run_id}) == len(variants) + 1


def _run(verdict=Verdict.PASS, k=None) -> EvaluatedRun:
    answer = Answer(
        values={"capital": "Paris"},
        evidence=(
            FieldEvidence(
                field="capital",
                spans=(EvidenceSpan(page_id="p1", char_start=0, char_end=5, content_hash=H),),
            ),
        ),
    )
    ex = ExecutionResult(key=k or key(), answer=answer, budget_usage=BudgetUsage(tokens=10))
    return EvaluatedRun(
        execution=ex,
        evaluation=Evaluation(verdict=verdict, fitness=1.0, evaluator_version="e1"),
    )


def test_execution_result_exposes_identity_and_evidence_from_the_answer():
    run = _run()
    assert run.execution.run_id == key().run_id
    assert run.execution.genome_hash == minimal_genome().genome_hash
    assert run.execution.evidence[0].spans[0].page_id == "p1"
    assert ExecutionResult(key=key()).evidence == ()


def test_evaluated_run_json_roundtrip():
    run = _run()
    assert EvaluatedRun.model_validate_json(run.model_dump_json()) == run


def test_fitness_must_be_finite():
    with pytest.raises(ValidationError):
        Evaluation(verdict=Verdict.FAIL, fitness=float("nan"), evaluator_version="e")


def test_in_memory_store_is_keyed_by_deterministic_identity():
    store = InMemoryRunStore()
    store.save_task(make_task_spec())
    store.save_genome(minimal_genome())
    run = _run()
    store.save_run(run)
    store.save_run(run)  # identical re-save is idempotent
    assert store.lookup(key()).evaluation.verdict is Verdict.PASS
    assert store.lookup(key(seed=99)) is None
    assert store.get_run(run.run_id) == run
    assert store.runs_for(genome_hash=minimal_genome().genome_hash) == [run]
    assert store.runs_for(task_id="nope") == []


# -- cost model ---------------------------------------------------------------
M = StaticCostModel(CostTable(proven_lower_bound=True))


def test_cost_estimate_is_deterministic_and_sums_stage_costs(task):
    e = M.estimate(minimal_genome(), task)
    assert e == M.estimate(minimal_genome(), task)
    assert (e.tokens, e.tool_calls) == (1500 + 800, 3)
    assert e.latency_s == pytest.approx(2.0 + 3.0 + 2.0)
    assert e.max_retries == 0 and e.retry_risk == 0.0


def test_gather_mode_scales_latency_but_not_calls_and_retries_are_reported(task):
    seq = M.estimate(minimal_genome(), task)
    par = M.estimate(Genome.of(gather(mode=GatherMode.PARALLEL_4), extract(), synth()), task)
    assert par.latency_s < seq.latency_s and par.tool_calls == seq.tool_calls
    risky = M.estimate(
        Genome.of(gather(), extract(), verify(on_failure=FailureStrategy.RETRY_2), synth()), task
    )
    assert risky.max_retries == 2 and 0 < risky.retry_risk < 1


def test_partial_estimate_never_exceeds_any_completion(task):
    prefix = M.estimate(Genome.of(gather(GatherSource.JEV)), task)
    full = M.estimate(Genome.of(gather(GatherSource.JEV), extract(), synth()), task)
    assert prefix.tokens <= full.tokens and prefix.latency_s <= full.latency_s
    assert prefix.tool_calls <= full.tool_calls


def test_exceeded_caps_reports_each_cap_independently():
    e = M.estimate(minimal_genome(), make_runtime_task())
    assert exceeded_caps(e, make_caps(tokens=10)) == (BudgetCap.TOKENS,)
    assert exceeded_caps(e, make_caps(tool_calls=1)) == (BudgetCap.TOOL_CALLS,)
    assert exceeded_caps(e, make_caps(wall_time_s=1.0)) == (BudgetCap.WALL_TIME,)
    assert exceeded_caps(e, make_caps()) == ()
