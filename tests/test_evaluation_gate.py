"""Deterministic evaluator: schema, matching, evidence, caps, verdicts, fitness."""

import itertools

import pytest

from benchmarks.loader import load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.evidence import EvidenceSpan, FieldEvidence
from core.payloads import Answer
from core.results import (
    BudgetCap,
    BudgetUsage,
    ExecutionResult,
    FailureInfo,
    FailureKind,
    FieldResult,
    RunKey,
    RunVersions,
    Verdict,
)
from core.task_spec import MatcherConfig, MatcherKind
from evaluation.fitness import ShapedFitness
from evaluation.gate import DeterministicEvaluator
from evaluation.matchers import DefaultMatcher

SPECS = load_task_specs()
STORE = SnapshotStore()
EVALUATOR = DeterministicEvaluator()
VERSIONS = RunVersions(
    model_hash="m",
    prompt_template_version="p",
    benchmark_hash="b",
    compiler_version="c",
    grammar_version="g",
)


def _key(task_id="A-001"):
    return RunKey(
        genome_hash="g" * 8, task_id=task_id, contract_hash="k", trial=0, seed=0, versions=VERSIONS
    )


def good_evidence(spec, fields=None):
    page = STORE.pages(spec.runtime.snapshot_id).pages[0]
    return tuple(
        FieldEvidence(field=f, spans=(page.span(0, 10),))
        for f in (fields or spec.ground_truth.values)
    )


def result_for(spec, values=None, evidence=None, usage=None, failure=None, answer=True):
    values = dict(spec.ground_truth.values) if values is None else values
    ans = (
        Answer(values=values, evidence=good_evidence(spec) if evidence is None else evidence)
        if answer
        else None
    )
    return ExecutionResult(
        key=_key(spec.id),
        answer=ans,
        budget_usage=usage or BudgetUsage(tokens=1000),
        failure=failure,
    )


@pytest.mark.parametrize("task_id", sorted(SPECS))
def test_correct_answer_with_valid_evidence_passes_on_every_benchmark_task(task_id):
    spec = SPECS[task_id]
    ev = EVALUATOR.evaluate(spec, result_for(spec))
    assert ev.verdict is Verdict.PASS
    assert all(r.matched and r.evidence_valid for r in ev.field_results)


def test_wrong_answer_fails():
    spec = SPECS["A-001"]
    ev = EVALUATOR.evaluate(spec, result_for(spec, values={"hq": "Madrid", "founded": 1987}))
    assert ev.verdict is Verdict.FAIL
    assert {r.field: r.matched for r in ev.field_results} == {"hq": False, "founded": True}


def test_schema_invalid_answers_fail():
    spec = SPECS["A-001"]
    for values in (
        {"hq": "Lisbon"},  # missing field
        {"hq": "Lisbon", "founded": "1987"},  # wrong type
        {"hq": "Lisbon", "founded": 1987, "extra": 1},  # unknown field
    ):
        assert EVALUATOR.evaluate(spec, result_for(spec, values=values)).verdict is Verdict.FAIL


def test_no_answer_fails_not_infeasible():
    spec = SPECS["A-001"]
    failure = FailureInfo(kind=FailureKind.NO_ANSWER, message="none")
    ev = EVALUATOR.evaluate(spec, result_for(spec, answer=False, failure=failure))
    assert ev.verdict is Verdict.FAIL


def _bad_spans(spec):
    page = STORE.pages(spec.runtime.snapshot_id).pages[0]
    good = page.span(0, 10)
    return {
        "wrong_hash": good.model_copy(update={"content_hash": "0" * 64}),
        "out_of_range": EvidenceSpan(
            page_id=page.page_id,
            char_start=0,
            char_end=len(page.content) + 5,
            content_hash=page.content_hash,
        ),
        "unknown_page": good.model_copy(update={"page_id": "no-such-page"}),
    }


@pytest.mark.parametrize("kind", ["wrong_hash", "out_of_range", "unknown_page"])
def test_bad_evidence_fails_even_when_the_values_are_correct(kind):
    spec = SPECS["A-001"]
    span = _bad_spans(spec)[kind]
    evidence = tuple(FieldEvidence(field=f, spans=(span,)) for f in spec.ground_truth.values)
    ev = EVALUATOR.evaluate(spec, result_for(spec, evidence=evidence))
    assert ev.verdict is Verdict.FAIL
    assert all(r.matched and not r.evidence_valid for r in ev.field_results)


def test_missing_evidence_fails_but_is_optional_when_not_required():
    spec = SPECS["A-001"]
    assert EVALUATOR.evaluate(spec, result_for(spec, evidence=())).verdict is Verdict.FAIL
    lenient = DeterministicEvaluator(require_evidence=False)
    assert lenient.evaluate(spec, result_for(spec, evidence=())).verdict is Verdict.PASS


def test_evidence_for_one_field_only_is_not_enough():
    spec = SPECS["A-001"]
    ev = EVALUATOR.evaluate(spec, result_for(spec, evidence=good_evidence(spec, ["hq"])))
    assert ev.verdict is Verdict.FAIL


@pytest.mark.parametrize(
    "usage",
    [
        BudgetUsage(tokens=10**6),
        BudgetUsage(wall_time_s=10**4),
        BudgetUsage(tool_calls=10**3),
        BudgetUsage(retries=10),
    ],
)
def test_cap_breach_is_infeasible_even_if_the_answer_is_perfect(usage):
    spec = SPECS["A-001"]
    ev = EVALUATOR.evaluate(spec, result_for(spec, usage=usage))
    assert ev.verdict is Verdict.INFEASIBLE


def test_usage_exactly_at_the_cap_is_allowed():
    spec = SPECS["A-001"]
    caps = spec.caps
    usage = BudgetUsage(
        tokens=caps.tokens,
        wall_time_s=caps.wall_time_s,
        tool_calls=caps.tool_calls,
        retries=caps.retries,
    )
    assert EVALUATOR.evaluate(spec, result_for(spec, usage=usage)).verdict is Verdict.PASS


def test_budget_exceeded_failure_is_infeasible_regardless_of_usage():
    spec = SPECS["A-001"]
    failure = FailureInfo(kind=FailureKind.BUDGET_EXCEEDED, message="x", cap=BudgetCap.TOKENS)
    ev = EVALUATOR.evaluate(spec, result_for(spec, failure=failure, answer=False))
    assert ev.verdict is Verdict.INFEASIBLE


def test_evaluation_is_deterministic():
    spec = SPECS["B-002"]
    r = result_for(spec)
    assert EVALUATOR.evaluate(spec, r) == EVALUATOR.evaluate(spec, r)


# -- fitness ---------------------------------------------------------------------------------
def _fr(matched, evidence):
    return FieldResult(field="f", matched=matched, evidence_valid=evidence)


def test_every_pass_fitness_exceeds_every_fail_fitness_and_infeasible_is_lowest():
    fit = ShapedFitness()
    caps = SPECS["A-001"].runtime.caps
    usages = [BudgetUsage(), BudgetUsage(tokens=1), BudgetUsage(tokens=10**9, wall_time_s=10**9)]
    per_field = list(itertools.product([True, False], repeat=2))
    passes = [fit.fitness(Verdict.PASS, (_fr(True, True),), u, caps) for u in usages]
    fails = [
        fit.fitness(Verdict.FAIL, tuple(_fr(m, e) for m, e in combo), u, caps)
        for u in usages
        for n in range(0, 4)
        for combo in itertools.product(per_field, repeat=n)
    ]
    infeasible = fit.fitness(Verdict.INFEASIBLE, (), BudgetUsage(), caps)
    assert min(passes) > max(fails)
    assert min(fails) > infeasible
    assert all(f == f and abs(f) < 10 for f in passes + fails)  # finite


def test_fail_fitness_is_shaped_by_partial_credit_and_pass_by_budget_headroom():
    fit = ShapedFitness()
    caps = SPECS["A-001"].runtime.caps
    none = fit.fitness(Verdict.FAIL, (_fr(False, False), _fr(False, False)), BudgetUsage(), caps)
    half = fit.fitness(Verdict.FAIL, (_fr(True, True), _fr(False, False)), BudgetUsage(), caps)
    assert none < half
    cheap = fit.fitness(Verdict.PASS, (_fr(True, True),), BudgetUsage(tokens=500), caps)
    dear = fit.fitness(Verdict.PASS, (_fr(True, True),), BudgetUsage(tokens=8000), caps)
    assert cheap > dear > 1.0


def test_pass_headroom_is_relative_to_the_runs_own_caps_not_a_fixed_constant():
    """The same absolute spend must score differently under a tight vs a loose cap."""
    fit = ShapedFitness()
    fr = (_fr(True, True),)
    spend = BudgetUsage(tokens=8000)
    tight = SPECS["A-001"].runtime.caps
    loose = tight.model_copy(update={"tokens": tight.tokens * 10})
    at_tight = fit.fitness(Verdict.PASS, fr, spend, tight)
    at_loose = fit.fitness(Verdict.PASS, fr, spend, loose)
    assert at_loose > at_tight  # 8k is 80% of the tight cap but only 8% of the loose one


def test_pass_fitness_ignores_wall_clock_time():
    """Wall time is a hard cap already; as a soft term it is pure infrastructure noise."""
    fit = ShapedFitness()
    caps = SPECS["A-001"].runtime.caps
    fr = (_fr(True, True),)
    fast = fit.fitness(Verdict.PASS, fr, BudgetUsage(tokens=1000, wall_time_s=1.0), caps)
    slow = fit.fitness(Verdict.PASS, fr, BudgetUsage(tokens=1000, wall_time_s=900.0), caps)
    assert fast == slow


def test_fail_band_ceiling_is_07_because_full_credit_is_the_pass_condition():
    fit = ShapedFitness()
    caps = SPECS["A-001"].runtime.caps
    best_reachable_fail = fit.fitness(Verdict.FAIL, (_fr(True, False),), BudgetUsage(), caps)
    assert best_reachable_fail == pytest.approx(0.7)
    assert best_reachable_fail < 1.0  # band ordering still cannot invert


def test_pass_beats_fail_end_to_end_through_the_evaluator():
    spec = SPECS["A-001"]
    passed = EVALUATOR.evaluate(spec, result_for(spec))
    nearly = EVALUATOR.evaluate(spec, result_for(spec, values={"hq": "Lisbon", "founded": 1900}))
    assert passed.fitness > nearly.fitness


# -- matchers --------------------------------------------------------------------------------
M = DefaultMatcher()


@pytest.mark.parametrize(
    ("kind", "expected", "actual", "ok", "kw"),
    [
        (MatcherKind.EXACT, 1987, 1987, True, {}),
        (MatcherKind.EXACT, 1987, "1987", False, {}),
        (MatcherKind.EXACT, 1, True, False, {}),
        (MatcherKind.NORMALIZED_TEXT, "Lisbon", "  lisbon. ", True, {}),
        (MatcherKind.NORMALIZED_TEXT, "Lisbon", "Porto", False, {}),
        (MatcherKind.NORMALIZED_TEXT, "Lisbon", 5, False, {}),
        (MatcherKind.NUMERIC_TOLERANCE, 10.0, 10.04, True, {"abs_tol": 0.05}),
        (MatcherKind.NUMERIC_TOLERANCE, 10.0, 10.2, False, {"abs_tol": 0.05}),
        (MatcherKind.NUMERIC_TOLERANCE, 100, 101, True, {"rel_tol": 0.02}),
        (MatcherKind.NUMERIC_TOLERANCE, 5, 5.0, True, {}),
        (MatcherKind.NUMERIC_TOLERANCE, 5, "5", False, {}),
        (MatcherKind.NUMERIC_TOLERANCE, 1, True, False, {}),
        (MatcherKind.DATE, "2024-06-30", "2024-06-30", True, {}),
        (MatcherKind.DATE, "2024-06-30", "2024-06-30T00:00:00", True, {}),
        (MatcherKind.DATE, "2024-06-30", "June 30 2024", False, {}),
        (MatcherKind.DATE, "2024-06-30", "2024-07-01", False, {}),
        (MatcherKind.SET_EQUAL, ["a", "B"], ["b ", "A"], True, {}),
        (MatcherKind.SET_EQUAL, ["a", "b"], ["a"], False, {}),
        (MatcherKind.SET_EQUAL, ["a"], "a", False, {}),
    ],
)
def test_matchers(kind, expected, actual, ok, kw):
    assert M.matches(expected, actual, MatcherConfig(kind=kind, **kw)) is ok
