"""TaskContract authority (Issue #20).

A ``TaskContract`` (+ its dataset bytes and splits) alone drives search and execution; the frozen
A/B benchmark reaches the same pipeline only through ``benchmarks.legacy_adapter``. Every model
backend here is a scripted TEST DOUBLE; nothing is a benchmark result.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path

import pytest

from benchmarks.legacy_adapter import (
    legacy_execution_tasks,
    legacy_references,
    legacy_suite,
)
from benchmarks.loader import load_task_specs
from core.constraints import ConstraintChecker, ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplit,
    DatasetSplits,
    SplitAccessError,
    SplitMethod,
    SplitRole,
    SplitUse,
)
from core.evaluation_spec import EvaluationSpec
from core.genome import Genome
from core.objective import ObjectiveSpec
from core.payloads import Answer
from core.results import BudgetUsage, ExecutionResult, FailureKind, RunKey, Verdict
from core.run_contract import ContractSuite, ExecutionTask
from core.stages import GatherSource, StageKind
from core.task_contract import ContractError, TaskContract, TaskType, WorkflowSpec
from core.task_spec import AnswerField, AnswerSchema, FieldType
from evaluation.contract_eval import ContractEvaluator
from evaluation.evidence import SnapshotEvidenceVerifier
from evaluation.gate import DeterministicEvaluator
from experiments.contract_run import contract_suite
from experiments.learning_curves import (
    ExperimentConfig,
    make_evaluate_fn,
    run_search,
    search_tasks,
)
from experiments.synthetic import synthetic_evaluate, synthetic_score
from ingestion.parse import sha256_bytes
from optimizers.aco_mmas import MMASACO
from optimizers.base import SearchContext
from runtime.gemma_client import GenerationResponse
from tests.conftest import extract, gather, make_caps, make_contract, make_suite, reason, synth
from tests.runtime_helpers import build_runner, drive, versions
from tests.test_evaluation_gate import good_evidence

ROOT = Path(__file__).resolve().parent.parent

# -- a generic dataset + contract (no benchmark anywhere) ---------------------------------------
CAPITALS = [
    ("q1", "France", "Paris"),
    ("q2", "Germany", "Berlin"),
    ("q3", "Italy", "Rome"),
    ("q4", "Spain", "Madrid"),
    ("q5", "Japan", "Tokyo"),
    ("q6", "Canada", "Ottawa"),
]
DATA = (
    "\n".join(
        json.dumps(
            {
                "qid": qid,
                "question": f"Which city is the capital of {country}?",
                "passage": f"{city} is the capital of {country}. It is a large city.",
                "answer": city,
            }
        )
        for qid, country, city in CAPITALS
    )
    + "\n"
).encode()
INSTRUCTIONS = "Name the capital city, using only the passage."
FULL_CAPS = dict(
    maximum_tokens_per_example=20_000,
    maximum_wall_time_s=120.0,
    maximum_tool_calls=20,
    maximum_retries=2,
)


def _schema(*names: str) -> AnswerSchema:
    return AnswerSchema(fields=tuple(AnswerField(name=n, type=FieldType.STRING) for n in names))


def qa_contract(data: bytes = DATA, **overrides) -> TaskContract:
    dataset = DatasetSpec(
        dataset_id="capitals",
        dataset_version=1,
        name="Capitals",
        content_hash=sha256_bytes(data),
        format=DatasetFormat.JSONL,
        columns=tuple(
            ColumnSpec(name=n, type=ColumnType.STRING)
            for n in ("qid", "question", "passage", "answer")
        ),
        id_column="qid",
        input_columns=("question",),
        context_columns=("passage",),
        target_columns=("answer",),
        row_count=len(CAPITALS),
    )
    base = dict(
        task_id="capitals",
        contract_version=1,
        task_type=TaskType.QUESTION_ANSWERING,
        instructions=INSTRUCTIONS,
        input_schema=_schema("question", "passage"),
        output_schema=_schema("answer"),
        dataset=dataset,
        evaluation=EvaluationSpec(evaluator="exact_match", config={"case_sensitive": False}),
        constraints=ConstraintLimits(**FULL_CAPS),
    )
    return TaskContract(**{**base, **overrides})


def splits_for(contract: TaskContract) -> DatasetSplits:
    return DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("q1", "q2", "q3")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("q4", "q5")),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=("q6",)),
        ),
    )


class PassageModel:
    """TEST DOUBLE: reads "<City> is the capital of" from the gathered passage."""

    model_hash = "passage-model/1"

    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        text = request.input_text
        if request.prompt_template_id.startswith("extract"):
            m = re.search(r"\b([A-Z]\w+) is the capital of \w+", text.split("Pages:")[-1])
            facts = (
                [{"field": "answer", "value": m.group(1), "page_id": "passage", "quote": m[0]}]
                if m
                else []
            )
            body = {"facts": facts}
        else:
            m = re.search(r'\] answer = "([^"]*)"', text)
            value = m.group(1) if m else ""
            if request.prompt_template_id.startswith("reason"):
                body = {"facts": [{"field": "answer", "value": value}]}
            else:
                body = {"answer": {"answer": value}, "citations": {"answer": [0]}}
        return GenerationResponse(
            text=json.dumps(body),
            parsed=body,
            prompt_tokens=50,
            completion_tokens=10,
            model_hash=self.model_hash,
        )


def result_for(task: ExecutionTask, values, usage=None) -> ExecutionResult:
    return ExecutionResult(
        key=RunKey(
            genome_hash="g" * 8,
            task_id=task.id,
            contract_hash=task.contract_hash,
            trial=0,
            seed=0,
            versions=versions(),
        ),
        answer=Answer(values=values),
        budget_usage=usage or BudgetUsage(tokens=100),
    )


# -- 1. a contract alone drives an end-to-end run -----------------------------------------------
def test_a_task_contract_alone_drives_an_end_to_end_search():
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner

    contract = qa_contract()
    suite, references = contract_suite(contract, splits_for(contract), DATA)
    model = PassageModel()
    runner = WorkflowRunner(model=model, benchmark_hash="inline")
    executed: list[str] = []

    def run_workflow(genome, task, trial, seed):
        executed.append(task.id)
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    train, val = search_tasks(suite)
    evaluate = make_evaluate_fn(run_workflow, ContractEvaluator(references), (*train, *val))
    config = ExperimentConfig(budget=6, batch_size=2, trials=1)
    result = run_search(MMASACO(), evaluate, suite, config, seed=0, checker=runner.checker)

    assert result["workflow_evaluations"] == 6
    assert result["train_pass_rate"] == 1.0  # the double reads the right city from each row
    assert result["champion"]["feasible"] and result["champion"]["validation_pass_rate"] == 1.0
    # the dataset rows executed are the contract's, by split role; the test row never ran
    assert set(executed) == {"q1", "q2", "q3", "q4", "q5"}
    # every prompt carried the contract's instructions and its own row's question
    assert model.requests and all(INSTRUCTIONS in r.input_text for r in model.requests)
    assert all("Which city is the capital of" in r.input_text for r in model.requests)
    # and every run is identified by the contract
    run = runner.run_sync(Genome.of(gather(), extract(), synth()), train[0])
    assert run.key.contract_hash == contract.contract_hash
    assert run.answer.values == {"answer": "Paris"}


def test_the_contract_pins_the_exact_dataset_bytes():
    contract = qa_contract()
    with pytest.raises(ContractError, match="bytes do not match"):
        contract_suite(contract, splits_for(contract), DATA.replace(b"Paris", b"Lyon "))


def test_runtime_examples_carry_inputs_and_context_but_never_targets():
    contract = qa_contract()
    suite, references = contract_suite(contract, splits_for(contract), DATA)
    for task in suite.tasks:
        assert set(task.example.values) == {"question", "passage"}
        assert "answer" not in task.example.values
    assert references.expected("q1") == {"answer": "Paris"}
    assert "Paris" not in repr(references)


# -- 2. legacy benchmark runs still work through the adapter --------------------------------
def test_legacy_benchmark_task_runs_and_is_judged_through_the_adapter(tmp_path):
    task = legacy_execution_tasks()["A-001"]
    spec = load_task_specs()["A-001"]
    assert task.contract.workflow.allowed_sources == spec.runtime.allowed_sources
    assert task.caps == spec.runtime.caps
    evaluator = ContractEvaluator(legacy_references())
    evaluator.check_task(task)

    ok = result_for(task, dict(spec.ground_truth.values))
    ok = ok.model_copy(
        update={"answer": Answer(values=ok.answer.values, evidence=good_evidence(spec))}
    )
    judged = evaluator.evaluate(task, ok)
    assert judged.verdict is Verdict.PASS
    assert judged == DeterministicEvaluator().evaluate(spec, ok)  # same legacy semantics


def test_legacy_suite_search_runs_with_the_synthetic_objective():
    suite = legacy_suite("B")
    result = run_search(MMASACO(), synthetic_evaluate, suite, ExperimentConfig(budget=20), seed=0)
    assert result["workflow_evaluations"] == 20
    assert result["suite"] == "B" and result["champion"] is not None


# -- 3. invalid / incomplete contracts fail before any model call ------------------------------
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"constraints": ConstraintLimits(maximum_tokens_per_example=20_000)}, "runtime caps"),
        (
            {
                "objective": ObjectiveSpec(mode="minimize_cost"),
                "constraints": ConstraintLimits(**FULL_CAPS, minimum_quality=0.5),
            },
            "does not measure",
        ),
        (
            {"constraints": ConstraintLimits(**FULL_CAPS, maximum_cost_per_example=0.01)},
            "does not measure",
        ),
    ],
)
def test_incomplete_or_unmeasurable_contract_is_refused_before_execution(overrides, match):
    contract = qa_contract(**overrides)  # a valid contract document ...
    with pytest.raises(ValueError, match=match):  # ... that the runtime cannot honour
        contract_suite(contract, splits_for(contract), DATA)


def test_unrunnable_evaluator_or_missing_expected_values_stop_before_the_model(monkeypatch):
    from evaluation import metrics

    contract = qa_contract()
    suite, references = contract_suite(contract, splits_for(contract), DATA)
    train, val = search_tasks(suite)
    model = PassageModel()

    def run_workflow(*args):
        raise AssertionError("the runtime must not be reached")

    monkeypatch.delitem(metrics.METRICS, contract.evaluation.evaluator)
    with pytest.raises(metrics.EvaluatorUnavailable):
        make_evaluate_fn(run_workflow, ContractEvaluator(references), (*train, *val))
    monkeypatch.undo()
    partial = ContractEvaluator({"q1": {"answer": "Paris"}})
    with pytest.raises(ContractError, match="no expected values"):
        make_evaluate_fn(run_workflow, partial, (*train, *val))
    assert model.requests == []


def test_a_row_missing_a_required_input_is_refused():
    contract = qa_contract()
    with pytest.raises(ValueError, match="missing inputs"):
        ExecutionTask(contract=contract, example={"row_id": "x", "values": {"passage": "p"}})


# -- 4. changing evaluator / objective / constraints changes behaviour ----------------------
def _task(contract: TaskContract, row: str = "q1") -> ExecutionTask:
    suite, _ = contract_suite(contract, splits_for(contract), DATA)
    return next(t for t in suite.tasks if t.id == row)


def test_changing_the_evaluator_changes_the_verdict_and_the_run_identity():
    strict = qa_contract(evaluation=EvaluationSpec(evaluator="exact_match"))
    lenient = qa_contract()  # case-insensitive
    _, refs = contract_suite(lenient, splits_for(lenient), DATA)
    verdicts = {}
    for name, contract in (("strict", strict), ("lenient", lenient)):
        task = _task(contract)
        verdicts[name] = ContractEvaluator(refs).evaluate(
            task, result_for(task, {"answer": "paris"})
        )
    assert verdicts["strict"].verdict is Verdict.FAIL
    assert verdicts["lenient"].verdict is Verdict.PASS
    assert strict.contract_hash != lenient.contract_hash  # new contract -> new runs, new cache keys


def test_changing_workflow_limits_changes_what_the_search_may_build():
    roomy, steps, calls = (
        qa_contract(),
        qa_contract(constraints=ConstraintLimits(**FULL_CAPS, maximum_workflow_steps=3)),
        qa_contract(constraints=ConstraintLimits(**FULL_CAPS, maximum_model_calls=2)),
    )

    def proposals(contract):
        ctx = SearchContext(contract=contract, checker=ConstraintChecker(), seed=4)
        return MMASACO().propose(30, ctx)

    def retrieval(gs):  # the vocabulary also offers DIRECT (1-2 stages, one model call)
        return [g for g in gs if g.stages[0].kind == StageKind.GATHER]

    assert any(len(g) > 3 for g in proposals(roomy))
    assert all(len(g) <= 3 for g in proposals(steps))
    assert retrieval(proposals(steps)) and all(len(g) == 3 for g in retrieval(proposals(steps)))
    assert all(not any(s.kind == StageKind.REASON for s in g.stages) for g in proposals(calls))
    assert any(any(s.kind == StageKind.REASON for s in g.stages) for g in proposals(roomy))
    # the source a contract does not allow is never built (csv/jsonl rows: fetch only)
    assert all(g.stages[0].source is GatherSource.FETCH for g in retrieval(proposals(roomy)))


def test_changing_runtime_caps_changes_execution(tmp_path):
    genome = Genome.of(gather(), extract(), synth())
    outcomes = {}
    for tokens in (20_000, 100):  # PassageModel spends 60 tokens per call, 2 calls
        caps = {**FULL_CAPS, "maximum_tokens_per_example": tokens}
        contract = qa_contract(constraints=ConstraintLimits(**caps))
        task = _task(contract)
        dag, runner = build_runner(genome, task, tmp_path, PassageModel())
        result = asyncio.run(drive(dag, runner, task))
        _, refs = contract_suite(contract, splits_for(contract), DATA)
        outcomes[tokens] = (result, ContractEvaluator(refs).evaluate(task, result).verdict)
    assert outcomes[20_000][0].failure is None and outcomes[20_000][1] is Verdict.PASS
    assert outcomes[100][0].failure.kind is FailureKind.BUDGET_EXCEEDED
    assert outcomes[100][1] is Verdict.INFEASIBLE


def _latency_evaluate(genome, task, trial, seed):
    """Noise-free synthetic fitness; every run passes; better workflows are slower."""
    run = synthetic_evaluate(genome, task, trial, seed)
    usage = BudgetUsage(tokens=1000 * len(genome), wall_time_s=10.0 * synthetic_score(genome))
    return run.model_copy(
        update={
            "execution": run.execution.model_copy(update={"budget_usage": usage}),
            "evaluation": run.evaluation.model_copy(
                update={"verdict": Verdict.PASS, "fitness": synthetic_score(genome)}
            ),
        }
    )


def test_changing_the_objective_changes_the_selected_champion():
    floor = ConstraintLimits.from_caps(make_caps()).model_copy(update={"minimum_quality": 0.5})
    config = ExperimentConfig(budget=60, trials=1)

    def search(objective):
        suite = make_suite(
            train=("t1", "t2"), val=("v1", "v2"), constraints=floor, objective=objective
        )
        return run_search(MMASACO(), _latency_evaluate, suite, config, seed=1)

    quality = search(ObjectiveSpec())
    latency = search(ObjectiveSpec(mode="minimize_latency"))
    # same evaluations, same optimizer trajectory: only the contract's objective differs ...
    assert quality["curve"] == latency["curve"]
    # ... and it decides which validated workflow is promoted
    wall = {w["genome_hash"]: w["mean_wall_time_s"] for w in latency["workflows"]}
    assert len(wall) > 1
    assert wall[latency["champion"]["genome_hash"]] == min(wall.values())
    fit = {w["genome_hash"]: w["validation_fitness"] for w in quality["workflows"]}
    assert fit[quality["champion"]["genome_hash"]] == max(fit.values())
    assert quality["champion"]["genome_hash"] != latency["champion"]["genome_hash"]
    assert quality["suite_hash"] != latency["suite_hash"]


def test_an_unmet_hard_limit_is_reported_never_promoted():
    impossible = ConstraintLimits.from_caps(make_caps()).model_copy(
        update={"maximum_mean_latency_s": 0.1}  # far below any workflow's measured latency
    )
    suite = make_suite(constraints=impossible)
    result = run_search(MMASACO(), _latency_evaluate, suite, ExperimentConfig(budget=6), seed=0)
    assert result["champion"]["feasible"] is False
    assert result["champion"]["violations"]


# -- 5. A/B/C cannot leak into generic runtime / search --------------------------------------
GENERIC = [
    *(
        p
        for pkg in ("runtime", "compiler", "optimizers", "router")
        for p in (ROOT / pkg).rglob("*.py")
    ),
    *(p for p in (ROOT / "core").glob("*.py") if p.name != "task_spec.py"),
    ROOT / "evaluation" / "contract_eval.py",
    ROOT / "evaluation" / "metrics.py",
    ROOT / "evaluation" / "fitness.py",
    ROOT / "experiments" / "learning_curves.py",
    ROOT / "experiments" / "contract_run.py",
    ROOT / "experiments" / "synthetic.py",
]
LEGACY_AUTHORITY = {"RuntimeTask", "TaskSpec", "TaskClass", "GroundTruth"}


@pytest.mark.parametrize("path", GENERIC, ids=lambda p: str(p.relative_to(ROOT)))
def test_generic_code_never_reads_legacy_task_authority(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            assert not names & LEGACY_AUTHORITY, f"{path.name} imports {names & LEGACY_AUTHORITY}"
            module = node.module or ""
            if module.split(".")[0] == "benchmarks":  # the adapter is the only way in
                assert module == "benchmarks.legacy_adapter", f"{path.name} imports {module}"
        if isinstance(node, ast.Attribute):
            assert node.attr != "task_class", f"{path.name} reads .task_class"
    assert "task_class" not in path.read_text(encoding="utf-8")


def test_search_and_execution_inputs_have_no_place_for_a_task_class():
    assert set(ExecutionTask.model_fields) == {"contract", "example"}
    assert set(SearchContext.__dataclass_fields__) == {"contract", "checker", "seed", "round"}
    result = run_search(MMASACO(), synthetic_evaluate, make_suite(), ExperimentConfig(budget=4), 0)
    assert "task_class" not in result


def test_relabelling_the_legacy_class_changes_nothing_downstream():
    suite = legacy_suite("A")

    def relabel(task: ExecutionTask) -> ExecutionTask:
        dataset = task.contract.dataset.model_copy(update={"metadata": {"task_class": "C"}})
        contract = TaskContract.model_validate(
            {**task.contract.model_dump(), "dataset": dataset.model_dump()}
        )
        return ExecutionTask(contract=contract, example=task.example)

    relabelled = ContractSuite(
        name=suite.name, tasks=tuple(relabel(t) for t in suite.tasks), splits=suite.splits
    )
    assert [t.contract_hash for t in relabelled.tasks] == [t.contract_hash for t in suite.tasks]
    config = ExperimentConfig(budget=24)
    assert run_search(MMASACO(), synthetic_evaluate, relabelled, config, 0) == run_search(
        MMASACO(), synthetic_evaluate, suite, config, 0
    )


# -- 6. the test split cannot feed the optimizer --------------------------------------------
def test_split_roles_gate_what_each_row_may_be_used_for():
    suite = make_suite(train=("t1",), val=("v1",), test=("x1",))
    with pytest.raises(SplitAccessError):
        suite.tasks_for(SplitRole.TEST, SplitUse.OPTIMIZER_FEEDBACK)
    with pytest.raises(SplitAccessError):
        suite.tasks_for(SplitRole.TEST, SplitUse.SELECTION)
    with pytest.raises(SplitAccessError):
        suite.tasks_for(SplitRole.VALIDATION, SplitUse.OPTIMIZER_FEEDBACK)
    assert [t.id for t in suite.tasks_for(SplitRole.TEST, SplitUse.REPORTING)] == ["x1"]


class SpyACO(MMASACO):
    def __init__(self):
        super().__init__()
        self.observed: set[str] = set()

    def observe(self, results):
        self.observed |= {r.execution.task_id for r in results}
        super().observe(results)


def test_only_optimization_rows_reach_the_optimizer_and_test_rows_never_run():
    suite = make_suite(train=("t1", "t2"), val=("v1",), test=("x1", "x2"))
    evaluated: set[str] = set()

    def spy(genome, task, trial, seed):
        evaluated.add(task.id)
        return synthetic_evaluate(genome, task, trial, seed)

    opt = SpyACO()
    run_search(opt, spy, suite, ExperimentConfig(budget=12, trials=1), seed=0)
    assert opt.observed == {"t1", "t2"}
    assert evaluated == {"t1", "t2", "v1"}


def test_a_test_row_result_smuggled_into_feedback_is_refused_before_state_changes():
    suite = make_suite(train=("t1",), val=("v1",), test=("x1",))
    test_task = suite.tasks_for(SplitRole.TEST, SplitUse.REPORTING)[0]

    def leaky(genome, task, trial, seed):  # a buggy evaluator reporting a test row as feedback
        return synthetic_evaluate(genome, test_task if task.id == "t1" else task, trial, seed)

    opt = SpyACO()
    with pytest.raises(SplitAccessError):
        run_search(opt, leaky, suite, ExperimentConfig(budget=4, trials=1), seed=0)
    assert opt.epoch == 0 and not opt.observed


# -- 7. no dual authority after adaptation ---------------------------------------------------
def _with(task: ExecutionTask, **fields) -> ExecutionTask:
    contract = TaskContract.model_validate({**task.contract.model_dump(), **fields})
    return ExecutionTask(contract=contract, example=task.example)


def test_legacy_examples_hold_inputs_only():
    for task in legacy_execution_tasks().values():
        assert set(task.example.values) == {"question", "snapshot_id"}


def test_after_adaptation_the_contract_not_the_task_spec_decides_evaluation():
    spec = load_task_specs()["A-001"]
    task = legacy_execution_tasks()["A-001"]
    near = {**spec.ground_truth.values, "founded": spec.ground_truth.values["founded"] + 2}
    result = result_for(task, near).model_copy(
        update={"answer": Answer(values=near, evidence=good_evidence(spec))}
    )
    # the frozen TaskSpec's exact matcher rejects it ...
    assert DeterministicEvaluator().evaluate(spec, result).verdict is Verdict.FAIL
    # ... a contract with a tolerant matcher accepts it - and the pipeline follows the contract
    config = dict(task.contract.evaluation.config)
    config["matchers"] = {
        **config["matchers"],
        "founded": {"kind": "numeric_tolerance", "abs_tol": 5.0},
    }
    tolerant = _with(task, evaluation={"evaluator": "legacy_field_match", "config": config})
    result = result.model_copy(
        update={"key": result.key.model_copy(update={"contract_hash": tolerant.contract_hash})}
    )
    verifier = SnapshotEvidenceVerifier()
    judged = ContractEvaluator(legacy_references(), verifier=verifier).evaluate(tolerant, result)
    assert judged.verdict is Verdict.PASS


def test_after_adaptation_the_contract_decides_caps_and_admission():
    task = legacy_execution_tasks()["B-001"]
    assert GatherSource.API in task.allowed_sources
    fetch_only = _with(
        task, workflow=WorkflowSpec(allowed_sources=(GatherSource.FETCH,)).model_dump()
    )
    tight = _with(
        task,
        constraints=task.contract.constraints.model_copy(
            update={"maximum_tokens_per_example": 50}
        ).model_dump(),
    )
    api = Genome.of(gather(GatherSource.API), extract(), synth())
    checker = ConstraintChecker()
    assert checker.is_valid(api, task.contract)
    assert not checker.is_valid(api, fetch_only.contract)  # RuntimeTask still says api is fine
    usage = BudgetUsage(tokens=1000)
    refs = legacy_references()
    keyed = result_for(tight, {"total_stock": 507}, usage)
    assert ContractEvaluator(refs).evaluate(tight, keyed).verdict is Verdict.INFEASIBLE
    assert tight.caps.tokens == 50


def test_a_result_for_another_contract_is_rejected():
    task = legacy_execution_tasks()["B-001"]
    other = _with(task, instructions="Different instructions.")
    with pytest.raises(ContractError, match="not produced for this task"):
        ContractEvaluator(legacy_references()).evaluate(other, result_for(task, {"total_stock": 1}))


def test_reason_stage_is_counted_as_a_model_call():
    contract = make_contract(
        constraints=ConstraintLimits.from_caps(make_caps()).model_copy(
            update={"maximum_model_calls": 2}
        )
    )
    checker = ConstraintChecker()
    assert checker.is_valid(Genome.of(gather(), extract(), synth()), contract)
    assert not checker.is_valid(Genome.of(gather(), extract(), reason(), synth()), contract)
