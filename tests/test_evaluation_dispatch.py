"""The EvaluationSpec-driven evaluator dispatcher: fail-closed behaviour, what every evaluation
records, the legacy compatibility boundary, target isolation and one end-to-end candidate run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.legacy_adapter import (
    legacy_evaluator,
    legacy_execution_tasks,
    legacy_references,
    legacy_task_contract,
)
from benchmarks.loader import load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.constraints import ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitRole,
    SplitUse,
)
from core.evaluation_spec import EvaluationSpec
from core.genome import Genome
from core.payloads import Answer
from core.results import (
    BudgetUsage,
    Evaluation,
    EvaluatorFailure,
    EvaluatorRecord,
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
    Verdict,
)
from core.run_contract import ExecutionTask
from core.task_contract import ContractError, TaskContract, TaskType
from core.task_spec import AnswerField, AnswerSchema, FieldType
from evaluation import metrics
from evaluation.contract_eval import ContractEvaluator
from evaluation.dispatch import EvaluationFailed, evaluate_prediction, resolve
from evaluation.fitness import INFEASIBLE_FITNESS, ShapedFitness
from evaluation.metrics import Metric
from experiments.contract_run import contract_suite
from experiments.learning_curves import make_evaluate_fn
from ingestion.parse import sha256_bytes
from tests.conftest import extract, gather, synth
from tests.runtime_helpers import versions
from tests.test_evaluation_gate import good_evidence

ROOT = Path(__file__).resolve().parent.parent
ANSWER = AnswerSchema(fields=(AnswerField(name="answer", type=FieldType.STRING),))
EXACT = EvaluationSpec(evaluator="exact_match")


def raw(spec: EvaluationSpec, **changes) -> dict:
    """A spec in its persisted (mapping) form, optionally tampered with."""
    return {**spec.model_dump(mode="json"), **changes}


# -- fail closed --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "failure"),
    [
        (raw(EXACT, evaluator="llm_judge"), EvaluatorFailure.UNKNOWN_KIND),
        (raw(EXACT, evaluator=""), EvaluatorFailure.UNKNOWN_KIND),
        (raw(EXACT, evaluator=None), EvaluatorFailure.UNKNOWN_KIND),
        (raw(EXACT, evaluator=["exact_match"]), EvaluatorFailure.UNKNOWN_KIND),
        (raw(EXACT, evaluator_version="exact_match/2"), EvaluatorFailure.VERSION_MISMATCH),
        (raw(EXACT, evaluator_version="token_f1/1"), EvaluatorFailure.VERSION_MISMATCH),
        (raw(EXACT, evaluator_version=""), EvaluatorFailure.VERSION_MISMATCH),
        ({"evaluator": "exact_match"}, EvaluatorFailure.VERSION_MISMATCH),  # nothing pinned
        (raw(EXACT, config={"case_insensitive": True}), EvaluatorFailure.INVALID_CONFIG),
        (raw(EXACT, config={"case_sensitive": "maybe"}), EvaluatorFailure.INVALID_CONFIG),
        (raw(EXACT, config=["case_sensitive"]), EvaluatorFailure.INVALID_CONFIG),
        (raw(EXACT, judge_model="gpt"), EvaluatorFailure.INVALID_CONFIG),  # unknown spec key
        (raw(EXACT, schema_version="evaluationspec/2"), EvaluatorFailure.UNSUPPORTED),
        (
            {
                "evaluator": "token_f1",
                "evaluator_version": "token_f1/1",
                "config": {"pass_threshold": 0},
            },
            EvaluatorFailure.INVALID_CONFIG,
        ),
        (
            {
                "evaluator": "classification_accuracy",
                "evaluator_version": "classification_accuracy/1",
            },
            EvaluatorFailure.INVALID_CONFIG,  # labels are required
        ),
        ("exact_match", EvaluatorFailure.INVALID_CONFIG),  # not a spec at all
    ],
)
def test_unrunnable_specs_fail_closed_with_a_reason(spec, failure):
    record = evaluate_prediction(spec, ANSWER, {"answer": "x"}, {"answer": "x"})
    assert record.failure is failure and record.detail
    assert not record.ok
    assert (record.quality, record.passed) == (None, None)  # nothing measured, nothing passed
    with pytest.raises(EvaluationFailed) as exc:
        resolve(spec)
    assert exc.value.record == record


def test_unknown_kind_is_recorded_as_given():
    record = evaluate_prediction(raw(EXACT, evaluator="llm_judge"), ANSWER, {"answer": "x"}, None)
    assert (record.kind, record.version) == ("llm_judge", "exact_match/1")
    assert record.spec_hash is None  # it never resolved to a spec


def test_an_implementation_that_moved_version_is_never_silently_substituted(monkeypatch):
    old = metrics.METRICS[EXACT.evaluator]
    monkeypatch.setitem(metrics.METRICS, EXACT.evaluator, Metric("exact_match/2", old.measure))
    record = evaluate_prediction(EXACT, ANSWER, {"answer": "x"}, {"answer": "x"})
    assert record.failure is EvaluatorFailure.VERSION_MISMATCH
    assert "exact_match/2" in record.detail


def test_an_implementation_that_breaks_its_contract_fails_closed(monkeypatch):
    for bad in ((1.5, True), (float("nan"), True), (1.0, "yes"), (True, True)):
        monkeypatch.setitem(
            metrics.METRICS,
            EXACT.evaluator,
            Metric("exact_match/1", lambda *_, out=bad: out),
        )
        record = evaluate_prediction(EXACT, ANSWER, {"answer": "x"}, {"answer": "x"})
        assert record.failure is EvaluatorFailure.EVALUATOR_ERROR and not record.passed


@pytest.mark.parametrize(
    ("expected", "failure"),
    [
        (None, EvaluatorFailure.MISSING_TARGET),
        ({}, EvaluatorFailure.MISSING_TARGET),
        ({"other": "x"}, EvaluatorFailure.MISSING_TARGET),
        ({"answer": "x", "other": "y"}, EvaluatorFailure.MISSING_TARGET),
        ({"answer": None}, EvaluatorFailure.MISSING_TARGET),  # required field, null target
        ({"answer": 3}, EvaluatorFailure.INVALID_TARGET),  # does not fit the output schema
    ],
)
def test_missing_or_unusable_targets_fail_closed(expected, failure):
    for predicted in ({"answer": "x"}, None):  # caught even when there is no prediction
        record = evaluate_prediction(EXACT, ANSWER, expected, predicted)
        assert record.failure is failure and record.quality is None


def test_a_token_f1_spec_over_several_fields_is_unsupported():
    out = AnswerSchema(
        fields=(
            AnswerField(name="a", type=FieldType.STRING),
            AnswerField(name="b", type=FieldType.STRING),
        )
    )
    spec = EvaluationSpec(evaluator="token_f1")
    record = evaluate_prediction(spec, out, {"a": "x", "b": "y"}, {"a": "x", "b": "y"})
    assert record.failure is EvaluatorFailure.UNSUPPORTED


def test_a_missing_prediction_is_a_measured_fail_not_an_evaluator_failure():
    for predicted in (None, {}, {"answer": 3}, {"answer": "x", "extra": 1}):
        record = evaluate_prediction(EXACT, ANSWER, {"answer": "x"}, predicted)
        assert record.ok and (record.quality, record.passed) == (0.0, False)
    # json_schema_validity: a missing prediction is invalid, never "valid by default"
    record = evaluate_prediction(
        EvaluationSpec(evaluator="json_schema_validity"), ANSWER, None, None
    )
    assert record.ok and (record.quality, record.passed) == (0.0, False)


def test_failure_details_never_contain_target_values():
    secret = "SECRET-TARGET-7f3a"
    spec = EvaluationSpec(evaluator="classification_accuracy", config={"labels": ["a", "b"]})
    record = evaluate_prediction(spec, ANSWER, {"answer": secret}, {"answer": "a"})
    assert record.failure is EvaluatorFailure.INVALID_TARGET
    assert secret not in record.model_dump_json()
    assert secret not in str(EvaluationFailed(record))
    record = evaluate_prediction(EXACT, ANSWER, {"answer": secret}, {"answer": "a"})
    assert record.ok and secret not in record.model_dump_json()


# -- no PASS / fitness from a failed evaluator --------------------------------------------------
def test_an_evaluation_cannot_be_built_from_a_failed_evaluator():
    failed = evaluate_prediction(raw(EXACT, evaluator="llm_judge"), ANSWER, {"answer": "x"}, None)
    for verdict in Verdict:
        with pytest.raises(ValidationError, match="no verdict or fitness"):
            Evaluation(
                verdict=verdict, fitness=1.0, evaluator_version="exact_match/1", evaluator=failed
            )


def test_an_evaluation_must_agree_with_its_evaluator_record():
    passed = evaluate_prediction(EXACT, ANSWER, {"answer": "x"}, {"answer": "x"})
    failed = evaluate_prediction(EXACT, ANSWER, {"answer": "x"}, {"answer": "y"})
    v = "exact_match/1+fitness/mvp-2"
    Evaluation(verdict=Verdict.PASS, fitness=1.0, evaluator_version=v, evaluator=passed)
    with pytest.raises(ValidationError, match="disagrees"):
        Evaluation(verdict=Verdict.PASS, fitness=1.0, evaluator_version=v, evaluator=failed)
    with pytest.raises(ValidationError, match="disagrees"):
        Evaluation(verdict=Verdict.FAIL, fitness=0.0, evaluator_version=v, evaluator=passed)
    with pytest.raises(ValidationError, match="never measured"):
        Evaluation(verdict=Verdict.INFEASIBLE, fitness=-1.0, evaluator_version=v, evaluator=passed)
    with pytest.raises(ValidationError, match="name the evaluator"):
        Evaluation(
            verdict=Verdict.PASS, fitness=1.0, evaluator_version="token_f1/1", evaluator=passed
        )
    with pytest.raises(ValidationError):
        EvaluatorRecord(kind="exact_match", version="exact_match/1", quality=1.0)  # no passed


# -- determinism and what is recorded -----------------------------------------------------------
def test_evaluation_is_deterministic_and_identified_by_the_spec():
    spec = EvaluationSpec(evaluator="token_f1", config={"pass_threshold": 0.5})
    args = (ANSWER, {"answer": "the cat sat"}, {"answer": "a cat"})
    records = [evaluate_prediction(spec, *args) for _ in range(5)]
    assert all(r == records[0] for r in records)
    assert records[0].spec_hash == spec.identity_hash
    # the persisted form, and an equivalent spec spelled with explicit defaults, are the same
    assert evaluate_prediction(spec.model_dump(mode="json"), *args) == records[0]
    explicit = EvaluationSpec(
        evaluator="token_f1", evaluator_version="token_f1/1", config={"pass_threshold": 0.5}
    )
    assert evaluate_prediction(explicit, *args) == records[0]
    # a different config is a different evaluator identity
    other = EvaluationSpec(evaluator="token_f1", config={"pass_threshold": 0.6})
    assert evaluate_prediction(other, *args).spec_hash != records[0].spec_hash


# -- a generic contract, judged through the ContractEvaluator ------------------------------------
TICKETS = [
    ("t1", "I was charged twice this month", "billing"),
    ("t2", "The export button crashes the app", "bugs"),
    ("t3", "Can I get a quote for 50 seats?", "sales"),
    ("t4", "Refund my last invoice please", "billing"),
    ("t5", "Login page throws an error", "bugs"),
    ("t6", "Interested in the enterprise plan", "sales"),
]
TICKET_DATA = (
    "\n".join(json.dumps({"id": i, "text": text, "label": label}) for i, text, label in TICKETS)
    + "\n"
).encode()
LABELS = ["billing", "bugs", "sales"]
CAPS = dict(
    maximum_tokens_per_example=1_000,
    maximum_wall_time_s=60.0,
    maximum_tool_calls=5,
    maximum_retries=1,
)


def ticket_contract(evaluation: EvaluationSpec | None = None) -> TaskContract:
    col = lambda n: ColumnSpec(name=n, type=ColumnType.STRING)  # noqa: E731
    return TaskContract(
        task_id="ticket-routing",
        contract_version=1,
        task_type=TaskType.CLASSIFICATION,
        instructions="Route the ticket to exactly one of: billing, bugs, sales.",
        input_schema=AnswerSchema(fields=(AnswerField(name="text", type=FieldType.STRING),)),
        output_schema=AnswerSchema(fields=(AnswerField(name="label", type=FieldType.STRING),)),
        dataset=DatasetSpec(
            dataset_id="tickets",
            dataset_version=1,
            name="Tickets",
            content_hash=sha256_bytes(TICKET_DATA),
            format=DatasetFormat.JSONL,
            columns=(col("id"), col("text"), col("label")),
            id_column="id",
            input_columns=("text",),
            target_columns=("label",),
            row_count=len(TICKETS),
        ),
        evaluation=evaluation
        or EvaluationSpec(evaluator="classification_accuracy", config={"labels": LABELS}),
        constraints=ConstraintLimits(**CAPS),
    )


def ticket_splits(contract: TaskContract) -> DatasetSplits:
    return DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("t1", "t2", "t3")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("t4", "t5")),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=("t6",)),
        ),
    )


def result_for(task: ExecutionTask, values, *, usage=None, failure=None, genome_hash="g" * 8):
    return ExecutionResult(
        key=RunKey(
            genome_hash=genome_hash,
            task_id=task.id,
            contract_hash=task.contract_hash,
            trial=0,
            seed=0,
            versions=versions(),
        ),
        answer=None if values is None else Answer(values=values),
        budget_usage=usage or BudgetUsage(tokens=100),
        failure=failure,
    )


def ticket_task(contract: TaskContract, row: str = "t1"):
    suite, refs = contract_suite(contract, ticket_splits(contract), TICKET_DATA)
    return next(t for t in suite.tasks if t.id == row), refs


def test_every_contract_evaluation_records_kind_version_spec_and_quality():
    contract = ticket_contract()
    task, refs = ticket_task(contract)  # t1 is billing
    evaluator = ContractEvaluator(refs)
    ev = evaluator.evaluate(task, result_for(task, {"label": "billing"}))
    rec = ev.evaluator
    assert ev.verdict is Verdict.PASS and ev.fitness >= 1.0
    assert rec.kind == "classification_accuracy"
    assert rec.version == contract.evaluation.evaluator_version == "classification_accuracy/1"
    assert rec.spec_hash == contract.evaluation.identity_hash
    assert (rec.quality, rec.passed, rec.failure) == (1.0, True, None)
    assert ev.evaluator_version == f"{rec.version}+{evaluator.fitness_fn.version}"

    wrong = evaluator.evaluate(task, result_for(task, {"label": "sales"}))
    assert wrong.verdict is Verdict.FAIL and wrong.fitness == 0.0
    assert (wrong.evaluator.quality, wrong.evaluator.passed) == (0.0, False)


def test_no_prediction_is_a_fail_never_a_pass():
    task, refs = ticket_task(ticket_contract())
    broken = FailureInfo(kind=FailureKind.MODEL_ERROR, message="backend down")
    for result in (
        result_for(task, None),
        result_for(task, {"label": "billing"}, failure=broken),  # an answer from a failed run
        result_for(task, {"label": 3}),
    ):
        ev = ContractEvaluator(refs).evaluate(task, result)
        assert ev.verdict is Verdict.FAIL and ev.fitness == 0.0
        assert ev.evaluator.quality == 0.0


def test_infeasible_is_decided_by_the_constraints_and_quality_is_not_measured():
    task, refs = ticket_task(ticket_contract())
    over = result_for(task, {"label": "billing"}, usage=BudgetUsage(tokens=1_001))
    ev = ContractEvaluator(refs).evaluate(task, over)
    assert ev.verdict is Verdict.INFEASIBLE and ev.fitness == INFEASIBLE_FITNESS
    assert ev.evaluator.kind == "classification_accuracy"
    assert ev.evaluator.quality is None and ev.evaluator.passed is None


def test_the_evaluator_follows_the_spec_not_the_task():
    """Same task, same answer: changing only the EvaluationSpec changes the judgement."""
    exact = EvaluationSpec(evaluator="exact_match", config={"case_sensitive": False})
    strict = ticket_contract()
    folded = ticket_contract(exact)
    verdicts = {}
    for name, contract in (("strict", strict), ("folded", folded)):
        task, refs = ticket_task(contract)
        ev = ContractEvaluator(refs).evaluate(task, result_for(task, {"label": "BILLING"}))
        verdicts[name] = (ev.verdict, ev.evaluator.kind)
    assert verdicts == {
        "strict": (Verdict.FAIL, "classification_accuracy"),
        "folded": (Verdict.PASS, "exact_match"),
    }


def test_contract_evaluation_fails_closed_before_execution_and_yields_no_fitness(monkeypatch):
    contract = ticket_contract()
    suite, refs = contract_suite(contract, ticket_splits(contract), TICKET_DATA)

    def run_workflow(*args):
        raise AssertionError("the runtime must not be reached")

    tasks = suite.tasks[:5]
    # a target outside the configured label set is caught in preflight, before any model call
    bad_refs = {t.id: {"label": "billing"} for t in tasks} | {"t1": {"label": "hr"}}
    with pytest.raises(EvaluationFailed) as exc:
        make_evaluate_fn(run_workflow, ContractEvaluator(bad_refs), tasks)
    assert exc.value.record.failure is EvaluatorFailure.INVALID_TARGET
    # an implementation that moved version: preflight refuses, and so does a direct evaluate
    impl = metrics.METRICS[contract.evaluation.evaluator]
    monkeypatch.setitem(
        metrics.METRICS,
        contract.evaluation.evaluator,
        Metric("classification_accuracy/2", impl.measure),
    )
    with pytest.raises(EvaluationFailed, match="version_mismatch"):
        make_evaluate_fn(run_workflow, ContractEvaluator(refs), tasks)
    with pytest.raises(EvaluationFailed, match="version_mismatch"):
        ContractEvaluator(refs).evaluate(tasks[0], result_for(tasks[0], {"label": "billing"}))


def test_a_missing_target_row_fails_closed():
    task, _ = ticket_task(ticket_contract())
    with pytest.raises(ContractError, match="no expected values"):
        ContractEvaluator({}).evaluate(task, result_for(task, {"label": "billing"}))


# -- legacy_field_match stays behind the benchmark boundary -------------------------------------
def test_the_generic_dispatcher_refuses_legacy_field_match():
    spec = legacy_execution_tasks()["A-001"].contract.evaluation
    record = evaluate_prediction(spec, ANSWER, {"answer": "x"}, {"answer": "x"})
    assert record.failure is EvaluatorFailure.UNSUPPORTED
    assert "benchmark compatibility boundary" in record.detail


def test_a_plain_contract_evaluator_refuses_legacy_tasks():
    task = legacy_execution_tasks()["A-001"]
    with pytest.raises(EvaluationFailed, match="unsupported") as exc:
        ContractEvaluator(legacy_references()).check_task(task)
    assert exc.value.record.kind == "legacy_field_match"


def test_the_legacy_evaluator_keeps_legacy_semantics_and_records_them():
    from evaluation.gate import EVALUATOR_VERSION, DeterministicEvaluator

    spec = load_task_specs()["A-001"]
    task = legacy_execution_tasks()["A-001"]
    result = result_for(task, dict(spec.ground_truth.values))
    result = result.model_copy(
        update={"answer": Answer(values=result.answer.values, evidence=good_evidence(spec))}
    )
    judged = legacy_evaluator().evaluate(task, result)
    assert judged.verdict is Verdict.PASS
    assert judged == DeterministicEvaluator().evaluate(spec, result)
    rec = judged.evaluator
    assert (rec.kind, rec.version) == ("legacy_field_match", EVALUATOR_VERSION)
    assert rec.spec_hash == legacy_task_contract(spec, SnapshotStore()).evaluation.identity_hash
    assert (rec.quality, rec.passed) == (1.0, True)


def test_only_the_benchmark_boundary_switches_the_legacy_evaluator_on():
    enabling = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".venv/", "ui/")):
            continue
        if re.search(r"legacy_verifier\s*=", path.read_text(encoding="utf-8")):
            enabling.append(rel)
    assert enabling == ["benchmarks/legacy_adapter.py"]


# -- targets never reach execution ---------------------------------------------------------------
SECRET_LABELS = ["billing-7f3a", "bugs-91c2", "sales-0d4e"]


def test_targets_never_reach_what_execution_receives():
    data = TICKET_DATA
    for plain, secret in zip(LABELS, SECRET_LABELS, strict=True):
        data = data.replace(f'"label": "{plain}"'.encode(), f'"label": "{secret}"'.encode())
    contract = ticket_contract(
        EvaluationSpec(evaluator="exact_match")  # labels must not be in the contract either
    )
    contract = TaskContract.model_validate(
        {
            **contract.model_dump(),
            "dataset": {
                **contract.dataset.model_dump(),
                "content_hash": sha256_bytes(data),
            },
        }
    )
    splits = DatasetSplits.model_validate(
        {**ticket_splits(contract).model_dump(), "dataset_hash": contract.dataset.identity_hash}
    )
    suite, refs = contract_suite(contract, splits, data)
    seen: list[str] = []

    def run_workflow(genome, task, trial, seed):
        # everything the runtime is handed: the task (contract + row inputs) and the genome
        seen.append(task.model_dump_json() + task.question + genome.model_dump_json())
        return result_for(task, {"label": "unknown"}, genome_hash=genome.genome_hash)

    train = suite.tasks[:3]
    evaluate = make_evaluate_fn(run_workflow, ContractEvaluator(refs), train)
    runs = [evaluate(Genome.of(gather(), extract(), synth()), t, 0, 0) for t in train]
    assert len(seen) == 3 and all(r.evaluation.verdict is Verdict.FAIL for r in runs)
    for blob in seen:
        for secret in SECRET_LABELS:
            assert secret not in blob
    # the evaluator did hold them, and never prints them
    assert refs.expected("t1") == {"label": SECRET_LABELS[0]}
    assert all(s not in repr(refs) for s in SECRET_LABELS)
    for r in runs:
        assert all(s not in r.evaluation.model_dump_json() for s in SECRET_LABELS)


# -- end to end ---------------------------------------------------------------------------------
KEYWORDS = {
    "billing": ("charged", "refund", "invoice"),
    "bugs": ("crash", "error"),
    "sales": ("quote", "plan"),
}


def keyword_router(genome, task, trial, seed):
    """TEST DOUBLE runtime: routes by keywords in the row's input text (no targets in reach)."""
    text = task.inputs_text.lower()
    label = next((k for k, words in KEYWORDS.items() if any(w in text for w in words)), "sales")
    if task.id == "t5":
        label = "billing"  # one deliberate mistake
    return result_for(task, {"label": label}, genome_hash=genome.genome_hash)


def test_end_to_end_candidate_evaluation_from_an_evaluation_spec():
    contract = ticket_contract()
    suite, refs = contract_suite(contract, ticket_splits(contract), TICKET_DATA)
    evaluator = ContractEvaluator(refs)
    rows = suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK) + suite.tasks_for(
        SplitRole.VALIDATION, SplitUse.SELECTION
    )
    evaluate = make_evaluate_fn(keyword_router, evaluator, rows)
    genome = Genome.of(gather(), extract(), synth())
    runs = [evaluate(genome, t, 0, 0) for t in rows]

    assert [r.execution.task_id for r in runs] == ["t1", "t2", "t3", "t4", "t5"]
    assert [r.evaluation.verdict for r in runs] == [Verdict.PASS] * 4 + [Verdict.FAIL]
    for r in runs:
        rec = r.evaluation.evaluator
        assert (rec.kind, rec.version) == ("classification_accuracy", "classification_accuracy/1")
        assert rec.spec_hash == contract.evaluation.identity_hash
    # fitness is the separate shaping layer over the measured quality ...
    fitness = ShapedFitness()
    assert all(r.evaluation.fitness >= 1.0 for r in runs[:4]) and runs[4].evaluation.fitness == 0.0
    assert runs[0].evaluation.fitness == fitness.score_fitness(
        Verdict.PASS, 1.0, runs[0].execution.budget_usage, rows[0].caps
    )
    # ... and ranking is the contract's objective over feasible candidates
    rank = suite.rank(genome, runs)
    assert rank.feasible and rank.objective_key == (pytest.approx(0.8),)
    # re-running is bit-for-bit identical
    assert [evaluate(genome, t, 0, 0) for t in rows] == runs
