"""Generic workflow grammar (issues #21 + #20): the typed stage vocabulary, admission invariants,
the bounded search space, and execution against uploaded datasets - all selected by the
``TaskContract``:

    TaskContract.workflow.stages -> workflow_grammar(contract) -> ConstraintChecker
                                 -> optimizer (SearchContext) / runtime (WorkflowRunner)

Every model backend is a scripted test double; nothing here calls a real model.
"""

from __future__ import annotations

import ast
import asyncio
import csv
import inspect
import io
import json
import random
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.legacy_adapter import legacy_contracts, legacy_execution_tasks
from compiler.dag import CompileError, compile_genome
from core.canonical import canonical_hash
from core.constraints import ConstraintChecker, ConstraintConfig, ConstraintLimits
from core.cost_model import CostTable, StaticCostModel
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
)
from core.evaluation_spec import EvaluationSpec
from core.evidence import FieldEvidence
from core.genome import Genome
from core.grammar import (
    GRAMMAR_VERSION,
    MAX_GENOME_STAGES,
    SIGNATURES,
    Capability,
    DataType,
    Grammar,
    capabilities,
)
from core.payloads import Answer, Page
from core.results import FailureKind
from core.run_contract import ExampleInput, ExecutionTask
from core.stages import (
    ALL_STAGE_KINDS,
    LEGACY_STAGE_KINDS,
    ConfidenceGateStage,
    DirectMethod,
    DirectStage,
    FailureStrategy,
    GatherMode,
    GatherSource,
    ReasonMethod,
    StageKind,
    SupportThreshold,
    SynthesizeMethod,
    VerifyMethod,
    all_stage_specs,
)
from core.task_contract import (
    ContractError,
    TaskContract,
    TaskType,
    WorkflowSpec,
    workflow_grammar,
)
from core.task_spec import AnswerField, AnswerSchema, FieldType
from core.violations import ViolationCode as V
from evaluation.contract_eval import ContractEvaluator
from experiments.contract_run import contract_suite
from experiments.learning_curves import ExperimentConfig, make_evaluate_fn, run_search, search_tasks
from ingestion.parse import sha256_bytes
from optimizers.aco_mmas import MMASACO
from optimizers.base import SearchContext, ensure_admissible
from optimizers.construct import MAX_STEPS, construct_genome
from optimizers.random_search import RandomSearch
from runtime.executors.base import ExecutorInput
from runtime.executors.gate import ConfidenceGateExecutor
from runtime.executors.gemma_stages import DirectExecutor
from runtime.executors.registry import default_executors
from runtime.executors.verify import answer_support
from runtime.gemma_client import GenerationResponse
from runtime.mvp_genomes import MVP_GENOMES
from runtime.runner import InadmissibleGenome, WorkflowRunner
from runtime.stage_runner import StageRunner
from tests.conftest import extract, flt, gather, make_caps, make_contract, reason, synth, verify
from tests.runtime_helpers import drive, make_ctx

ROOT = Path(__file__).resolve().parent.parent
FULL = Grammar(ALL_STAGE_KINDS)
JEV = GatherSource.JEV
# What ``WorkflowRunner`` admits by default: jev and self_consistency are not executable.
RUNTIME = ConstraintConfig(
    unavailable_sources=(GatherSource.JEV,),
    unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,),
)
ONE_VERIFIER = RUNTIME.model_copy(update={"max_active_verifiers": 1})
CAPS = ConstraintLimits(
    maximum_tokens_per_example=20_000,
    maximum_wall_time_s=120.0,
    maximum_tool_calls=20,
    maximum_retries=2,
)


def direct(method=DirectMethod.ANSWER) -> DirectStage:
    return DirectStage(method=method)


def gate(min_support=SupportThreshold.HALF) -> ConfidenceGateStage:
    return ConfidenceGateStage(min_support=min_support)


def codes(violations) -> set[V]:
    return {v.code for v in violations}


def snapshot_contract(stages=ALL_STAGE_KINDS, **kw) -> TaskContract:
    """A snapshot-backed test contract (``make_contract``) with an explicit vocabulary."""
    return make_contract(stages=stages, **kw)


# -- two uploaded datasets: one with a context column, one without -----------------------------
def _schema(**fields: FieldType) -> AnswerSchema:
    return AnswerSchema(fields=tuple(AnswerField(name=n, type=t) for n, t in fields.items()))


QA_ROWS = [
    ("q1", "Which river flows through Paris?", "Paris is the capital of France.\n\n"
     "The Seine flows through Paris.\n\nBananas grow in warm climates.", "Seine"),
    ("q2", "Which river flows through Rome?", "Rome is the capital of Italy.\n\n"
     "The Tiber flows through Rome.", "Tiber"),
]  # fmt: skip
QA_DATA = (
    "\n".join(
        json.dumps({"qid": q, "question": question, "passage": passage, "answer": answer})
        for q, question, passage, answer in QA_ROWS
    )
    + "\n"
).encode()
QA_QUOTE = "The Seine flows through Paris"


def qa_contract(**workflow: object) -> TaskContract:
    """Question answering over an uploaded JSONL with a ``passage`` context column."""
    return TaskContract(
        task_id="rivers",
        contract_version=1,
        task_type=TaskType.QUESTION_ANSWERING,
        instructions="Answer with the river's name, using only the passage.",
        input_schema=_schema(question=FieldType.STRING, passage=FieldType.STRING),
        output_schema=_schema(answer=FieldType.STRING),
        dataset=DatasetSpec(
            dataset_id="rivers",
            dataset_version=1,
            name="Rivers",
            content_hash=sha256_bytes(QA_DATA),
            format=DatasetFormat.JSONL,
            columns=tuple(
                ColumnSpec(name=n, type=ColumnType.STRING)
                for n in ("qid", "question", "passage", "answer")
            ),
            id_column="qid",
            input_columns=("question",),
            context_columns=("passage",),
            target_columns=("answer",),
            row_count=len(QA_ROWS),
        ),
        evaluation=EvaluationSpec(evaluator="exact_match"),
        constraints=CAPS,
        workflow=WorkflowSpec(**workflow),
    )


TICKETS = [
    ("t1", "I was charged twice for my plan this month.", "billing"),
    ("t2", "The export button crashes the app.", "bugs"),
    ("t3", "Can I get a quote for 40 seats?", "sales"),
    ("t4", "My invoice shows the wrong VAT number.", "billing"),
    ("t5", "Search returns nothing since the update.", "bugs"),
]


def _tickets_csv() -> bytes:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["id", "text", "label"])
    w.writerows(TICKETS)
    return out.getvalue().encode()


TICKET_DATA = _tickets_csv()


def ticket_contract(**workflow: object) -> TaskContract:
    """Ticket routing over an uploaded CSV with NO context column."""
    return TaskContract(
        task_id="ticket-routing",
        contract_version=1,
        task_type=TaskType.CLASSIFICATION,
        instructions="Route the support ticket to exactly one team: billing, bugs or sales.",
        input_schema=_schema(text=FieldType.STRING),
        output_schema=_schema(label=FieldType.STRING),
        dataset=DatasetSpec(
            dataset_id="tickets",
            dataset_version=1,
            name="Support tickets",
            content_hash=sha256_bytes(TICKET_DATA),
            format=DatasetFormat.CSV,
            columns=tuple(
                ColumnSpec(name=n, type=ColumnType.STRING) for n in ("id", "text", "label")
            ),
            id_column="id",
            input_columns=("text",),
            target_columns=("label",),
            row_count=len(TICKETS),
        ),
        evaluation=EvaluationSpec(
            evaluator="classification_accuracy", config={"labels": ["billing", "bugs", "sales"]}
        ),
        constraints=CAPS,
        workflow=WorkflowSpec(**workflow),
    )


def qa_task(contract: TaskContract | None = None, row: int = 0) -> ExecutionTask:
    qid, question, passage, _ = QA_ROWS[row]
    return ExecutionTask(
        contract=contract or qa_contract(),
        example=ExampleInput(row_id=qid, values={"question": question, "passage": passage}),
    )


def ticket_task(contract: TaskContract | None = None, row: int = 0) -> ExecutionTask:
    tid, text, _ = TICKETS[row]
    return ExecutionTask(
        contract=contract or ticket_contract(),
        example=ExampleInput(row_id=tid, values={"text": text}),
    )


class RowModel:
    """Scripted backend answering one field; quotes ``quote`` from page ``page_id``."""

    model_hash = "row-model"

    def __init__(self, field: str, value: str, quote: str = QA_QUOTE, page_id: str = "passage"):
        self.field, self.value, self.quote, self.page_id = field, value, quote, page_id
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        fact = {"field": self.field, "value": self.value, "quote": self.quote}
        if request.prompt_template_id.startswith("extract"):
            body = {"facts": [{**fact, "page_id": self.page_id}]}
        elif request.prompt_template_id.startswith("reason"):
            body = {"facts": [fact]}
        else:  # synthesize.* / direct.*
            body = {"answer": {self.field: self.value}, "citations": {self.field: [0]}}
        return GenerationResponse(
            text=json.dumps(body),
            parsed=body,
            prompt_tokens=100,
            completion_tokens=20,
            model_hash=self.model_hash,
        )


class TicketModel:
    """Scripted classifier: routes on keywords in the ticket text the prompt carries."""

    model_hash = "ticket-model"

    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        text = request.input_text.lower()
        label = (
            "billing"
            if any(w in text for w in ("charged", "invoice"))
            else "sales"
            if "quote" in text
            else "bugs"
        )
        body = {"answer": {"label": label}}
        return GenerationResponse(
            text=json.dumps(body), parsed=body, prompt_tokens=40, completion_tokens=5,
            model_hash=self.model_hash,
        )  # fmt: skip


def run_row(genome: Genome, task: ExecutionTask, model):
    """Compile and execute one genome through the real executors (no MAF needed)."""
    dag = compile_genome(genome)
    runner = StageRunner(dag, default_executors(), make_ctx(task, model))
    return asyncio.run(drive(dag, runner, task))


@cache
def qa_space() -> tuple[Genome, ...]:
    """Every admissible genome for the context-backed QA contract on the shipped runtime, with one
    active verifier (keeps the exhaustive tests fast; with two it is 49,092)."""
    return tuple(ConstraintChecker(config=ONE_VERIFIER).enumerate_admissible(qa_contract()))


# == 1. vocabulary: every requested capability is a typed, bounded stage =========================
def test_every_requested_capability_is_a_typed_bounded_stage_configuration():
    specs = [s for k in StageKind for s in all_stage_specs(k)]
    assert {c for s in specs for c in capabilities(s)} == set(Capability)
    assert all(capabilities(s) for s in specs)
    assert {Capability.PARALLEL} < capabilities(gather(mode=GatherMode.PARALLEL_2))
    assert capabilities(reason(ReasonMethod.DECOMPOSE)) == {Capability.DECOMPOSE}
    # every kind has a finite enumerated configuration space; there is no free-form option
    assert {k: len(all_stage_specs(k)) for k in StageKind} == {
        StageKind.GATHER: 9,  # source x mode (Parallel = mode parallel-2/parallel-4)
        StageKind.FILTER: 2,
        StageKind.EXTRACT: 3,
        StageKind.REASON: 2,  # single | decompose
        StageKind.VERIFY: 9,  # method x on_failure
        StageKind.SYNTHESIZE: 2,
        StageKind.DIRECT: 2,
        StageKind.CONFIDENCE_GATE: 2,
    }


def test_input_and_output_shapes_of_every_stage():
    T, P, F, A = DataType.TASK, DataType.PAGES, DataType.FACTS, DataType.ANSWER
    assert SIGNATURES == {
        StageKind.GATHER: {T: P},
        StageKind.FILTER: {P: P},
        StageKind.EXTRACT: {P: F},
        StageKind.REASON: {F: F},
        StageKind.VERIFY: {F: F, A: A},
        StageKind.SYNTHESIZE: {F: A},
        StageKind.DIRECT: {T: A},
        StageKind.CONFIDENCE_GATE: {A: A},
    }


def test_valid_successors_of_every_stage():
    k = StageKind
    after = {  # a representative prefix ending in each kind
        "START": (),
        k.GATHER: (gather(),),
        k.FILTER: (gather(), flt()),
        k.EXTRACT: (gather(), extract()),
        k.REASON: (gather(), extract(), reason()),
        k.SYNTHESIZE: (gather(), extract(), synth()),
        k.DIRECT: (direct(),),
        k.CONFIDENCE_GATE: (gather(), extract(), synth(), gate()),
    }
    got = {
        name: {s.kind for s in FULL.valid_successor_specs(Genome.from_stages(p))}
        for name, p in after.items()
    }
    assert got == {
        "START": {k.GATHER, k.DIRECT},
        k.GATHER: {k.FILTER, k.EXTRACT},
        k.FILTER: {k.EXTRACT},  # FILTER at most once
        k.EXTRACT: {k.REASON, k.VERIFY, k.SYNTHESIZE},
        k.REASON: {k.VERIFY, k.SYNTHESIZE},  # REASON/DECOMPOSE at most once
        k.SYNTHESIZE: {k.VERIFY, k.CONFIDENCE_GATE},
        k.DIRECT: {k.VERIFY},  # no gate: DIRECT has no evidence to gate on
        k.CONFIDENCE_GATE: set(),  # terminal
    }
    after_verify = Genome.of(gather(), extract(), verify())
    assert {s.kind for s in FULL.valid_successor_specs(after_verify)} == {k.REASON, k.SYNTHESIZE}
    after_answer_verify = Genome.of(gather(), extract(), synth(), verify())
    assert {s.kind for s in FULL.valid_successor_specs(after_answer_verify)} == {k.CONFIDENCE_GATE}
    # on the DIRECT path only stages that need no gathered pages are offered
    assert FULL.valid_successors(Genome.of(direct())) == (k.VERIFY,)
    assert {
        (s.method, s.on_failure)
        for s in FULL.valid_successor_specs(Genome.of(direct()))
        if s.kind == k.VERIFY
    } == {
        (m, f)
        for m in (VerifyMethod.SCHEMA_CHECK, VerifyMethod.SELF_CONSISTENCY)
        for f in (FailureStrategy.RETRY_1, FailureStrategy.RETRY_2)
    }


def test_cost_step_and_model_call_bounds_follow_the_contract_vocabulary():
    m = StaticCostModel(CostTable(proven_lower_bound=True))
    full = snapshot_contract()
    for spec in (s for k in StageKind for s in all_stage_specs(k)):
        m.estimate(Genome.of(spec), full)  # every configuration is costed
    assert m.estimate(Genome.of(direct()), full).tool_calls == 0
    # the empty prefix's cheapest completion depends on the producers the contract supports
    legacy = m.estimate(Genome(), make_contract())
    assert (legacy.tokens, legacy.tool_calls) == (1500 + 800, 2)  # unchanged legacy bound
    generic = m.estimate(Genome(), full)
    assert (generic.tokens, generic.tool_calls) == (800, 0)  # DIRECT(answer)
    # a token cap that rules out retrieval leaves exactly the cheap DIRECT configuration
    proven = ConstraintChecker(cost_model=m)
    tight = make_caps(tokens=1000)
    assert proven.admissible_successors(Genome(), make_contract(caps=tight)) == ()
    assert proven.admissible_successors(Genome(), snapshot_contract(caps=tight)) == (direct(),)
    # #37's step / model-call lower bounds count the cheapest completion too (DIRECT = 1 + 1)
    steps = snapshot_contract(
        constraints=ConstraintLimits.from_caps(make_caps()).model_copy(
            update={"maximum_workflow_steps": 2, "maximum_model_calls": 1}
        )
    )
    checker = ConstraintChecker()
    assert checker.is_valid(Genome.of(direct(), verify()), steps)
    assert codes(checker.check(Genome.of(gather(), extract(), synth()), steps)) == {
        V.STEP_LIMIT,
        V.MODEL_CALL_LIMIT,
    }
    assert codes(checker.check(Genome.of(gather()), steps, complete=False)) == {
        V.STEP_LIMIT,
        V.MODEL_CALL_LIMIT,
    }
    assert checker.is_valid(Genome(), steps, complete=False)  # DIRECT can still finish


def test_the_synthetic_objective_scores_every_stage_configuration():
    from experiments.synthetic import synthetic_score

    for spec in (s for k in StageKind for s in all_stage_specs(k)):
        synthetic_score(Genome.of(spec))  # total over the vocabulary: no KeyError / AttributeError
    assert synthetic_score(MVP_GENOMES["A"]) == pytest.approx(0.2)  # legacy scores unchanged


# == 2. representative valid compositions ========================================================
VALID = {
    "direct": (direct(),),
    "direct-cot-verified": (direct(DirectMethod.COT), verify()),
    "direct-verified-retry-2": (direct(), verify(on_failure=FailureStrategy.RETRY_2)),
    "retrieval-minimal": (gather(), extract(), synth()),
    "parallel-gather": (gather(mode=GatherMode.PARALLEL_4), extract(), synth()),
    "filter-decompose": (gather(), flt(), extract(), reason(ReasonMethod.DECOMPOSE), synth()),
    "gated": (gather(), extract(), synth(SynthesizeMethod.CITE_EVIDENCE), gate()),
    "verified-then-gated": (
        gather(),
        extract(),
        verify(VerifyMethod.EVIDENCE_SPAN),
        synth(),
        verify(),
        gate(SupportThreshold.ALL),
    ),
    "regather": (gather(), extract(), verify(on_failure=FailureStrategy.REGATHER), synth()),
}


@pytest.mark.parametrize("name", sorted(VALID))
def test_representative_compositions_are_admissible_and_compile(name):
    genome = Genome.from_stages(VALID[name])
    assert FULL.validate(genome) == ()
    assert ConstraintChecker().check(genome, snapshot_contract()) == ()
    dag = compile_genome(genome)
    assert [n.stage for n in dag.nodes] == list(genome.stages)
    assert dag.nodes[0].input_type == DataType.TASK
    assert dag.nodes[-1].output_type == DataType.ANSWER
    assert dag.topological_order() == tuple(n.node_id for n in dag.nodes)  # linear


def test_the_longest_genome_is_grammar_valid_and_hits_the_bound():
    longest = Genome.of(
        gather(), flt(), extract(), verify(), reason(), verify(), synth(), verify(), gate()
    )
    assert len(longest) == MAX_GENOME_STAGES
    assert FULL.validate(longest) == ()
    assert FULL.valid_successor_specs(longest) == ()
    # three verifiers: grammar-valid, but the hard constraint (<= 2) rejects it
    assert codes(ConstraintChecker().check(longest, snapshot_contract())) == {V.TOO_MANY_VERIFIERS}


# == 3. invalid compositions are rejected with a reason ==========================================
INVALID = {
    # missing required producers
    "empty": ((), {V.MISSING_REQUIRED_STAGE, V.NO_ANSWER_TERMINAL}),
    "no-synthesize": ((gather(), extract()), {V.MISSING_REQUIRED_STAGE, V.NO_ANSWER_TERMINAL}),
    "no-extract": ((gather(), flt()), {V.MISSING_REQUIRED_STAGE, V.NO_ANSWER_TERMINAL}),
    # incompatible ordering (including a second Answer producer)
    "direct-after-gather": ((gather(), direct()), {V.INVALID_TRANSITION}),
    "synthesize-after-direct": ((direct(), synth()), {V.INVALID_TRANSITION}),
    "two-directs": ((direct(), direct()), {V.INVALID_TRANSITION}),
    "direct-after-answer": ((gather(), extract(), synth(), direct()), {V.INVALID_TRANSITION}),
    "synthesize-twice": ((gather(), extract(), synth(), synth()), {V.INVALID_TRANSITION}),
    "gate-first": ((gate(),), {V.INVALID_TRANSITION}),
    "gate-on-facts": ((gather(), extract(), gate()), {V.INVALID_TRANSITION}),
    "reason-after-direct": ((direct(), reason()), {V.INVALID_TRANSITION}),
    # impossible dependencies: these need pages that the DIRECT path never gathers
    "evidence-check-without-pages": (
        (direct(), verify(VerifyMethod.EVIDENCE_SPAN)),
        {V.UNSATISFIED_DEPENDENCY},
    ),
    "regather-without-gather": (
        (direct(), verify(on_failure=FailureStrategy.REGATHER)),
        {V.UNSATISFIED_DEPENDENCY},
    ),
    "gate-without-evidence": ((direct(), verify(), gate()), {V.UNSATISFIED_DEPENDENCY}),
    # duplicate / illegal terminal stages
    "second-gate": (
        (gather(), extract(), synth(), gate(), gate(SupportThreshold.ALL)),
        {V.AFTER_TERMINAL},
    ),
    "verify-after-gate": ((gather(), extract(), synth(), gate(), verify()), {V.AFTER_TERMINAL}),
    # parallel / decomposition structure
    "decompose-twice": (
        (gather(), extract(), reason(ReasonMethod.DECOMPOSE), reason(ReasonMethod.DECOMPOSE)),
        {V.PLACEMENT},
    ),
    "decompose-on-pages": ((gather(), reason(ReasonMethod.DECOMPOSE)), {V.INVALID_TRANSITION}),
    "decompose-on-answer": (
        (gather(), extract(), synth(), reason(ReasonMethod.DECOMPOSE)),
        {V.INVALID_TRANSITION},
    ),
    "parallel-gather-twice": (
        (gather(mode=GatherMode.PARALLEL_2), gather(mode=GatherMode.PARALLEL_4)),
        {V.INVALID_TRANSITION},
    ),
    # unbounded retry chains: verifiers never stack
    "stacked-fact-verifiers": (
        (gather(), extract(), verify(), verify(on_failure=FailureStrategy.RETRY_2), synth()),
        {V.PLACEMENT},
    ),
    "stacked-answer-verifiers": ((direct(), verify(), verify()), {V.PLACEMENT}),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_compositions_are_rejected_with_their_reason(name):
    stages, expected = INVALID[name]
    genome = Genome.from_stages(stages)
    assert codes(FULL.validate(genome)) == expected
    # the same reason through a contract that supports every kind (admission == grammar here)
    assert expected <= codes(ConstraintChecker().check(genome, snapshot_contract()))
    with pytest.raises(CompileError):
        compile_genome(genome)


@pytest.mark.parametrize(
    "name", sorted(n for n, (_, c) in INVALID.items() if V.NO_ANSWER_TERMINAL not in c)
)
def test_generators_can_never_produce_an_invalid_composition(name):
    """Rules are monotone: the first offending stage is never offered as a successor."""
    stages, _ = INVALID[name]
    i = next(
        i
        for i in range(len(stages))
        if FULL.validate(Genome.from_stages(stages[: i + 1]), complete=False)
    )
    prefix = Genome.from_stages(stages[:i])
    assert stages[i] not in FULL.valid_successor_specs(prefix)
    assert stages[i] not in ConstraintChecker().admissible_successors(prefix, snapshot_contract())


def test_missing_producer_messages_name_the_options():
    [empty] = [v for v in FULL.validate(Genome()) if v.code is V.MISSING_REQUIRED_STAGE]
    assert "GATHER -> EXTRACT -> SYNTHESIZE" in empty.message and "DIRECT" in empty.message
    legacy = [v.message for v in Grammar().validate(Genome()) if v.code is V.MISSING_REQUIRED_STAGE]
    assert legacy == [f"required stage {k} is missing" for k in ("GATHER", "EXTRACT", "SYNTHESIZE")]
    no_retrieval = workflow_grammar(ticket_contract())
    [only] = [
        v.message for v in no_retrieval.validate(Genome()) if v.code is V.MISSING_REQUIRED_STAGE
    ]
    assert only == "required stage DIRECT is missing"


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "GATHER", "source": "fetch", "mode": "parallel-8"},  # unbounded fan-out width
        {"kind": "VERIFY", "method": "schema_check", "on_failure": "retry-forever"},
        {"kind": "VERIFY", "method": "schema_check", "on_failure": "retry-3"},
        {"kind": "REASON", "method": "decompose", "depth": 5},  # no free decomposition depth
        {"kind": "DIRECT", "method": "answer", "prompt": "anything"},
        {"kind": "CONFIDENCE_GATE", "min_support": "support-1"},
        {"kind": "CODE", "source": "print(1)"},  # no arbitrary-code stage
        {"kind": "PLANNER", "goal": "anything"},  # no unbounded planner stage
    ],
)
def test_unbounded_or_free_form_stages_are_not_representable(raw):
    with pytest.raises(ValidationError):
        Genome.model_validate({"stages": [raw]})


def test_hard_constraints_on_parallel_and_retries_still_apply_to_the_new_vocabulary():
    checker, contract = ConstraintChecker(), snapshot_contract()
    jev4 = Genome.of(gather(JEV, GatherMode.PARALLEL_4), extract(), synth(), gate())
    assert codes(checker.check(jev4, contract)) == {V.JEV_PARALLEL_4}
    jev_regather = Genome.of(
        gather(JEV), extract(), synth(), verify(on_failure=FailureStrategy.REGATHER), gate()
    )
    assert codes(checker.check(jev_regather, contract)) == {V.JEV_REGATHER}
    sc = Genome.of(direct(), verify(VerifyMethod.SELF_CONSISTENCY))
    assert codes(checker.check(sc, contract)) == set()
    assert codes(ConstraintChecker(config=RUNTIME).check(sc, contract)) == {V.RUNTIME_UNAVAILABLE}


# == 4. the contract selects the vocabulary ======================================================
def test_contract_vocabulary_comes_from_the_contract_not_a_legacy_label():
    k = StageKind
    # no context columns: only what a row's inputs support (DIRECT + VERIFY), DIRECT-capable
    assert ticket_contract().workflow.stages == (k.VERIFY, k.DIRECT)
    assert workflow_grammar(ticket_contract()).valid_successors(Genome()) == (k.DIRECT,)
    # a context column: the full retrieval vocabulary, including the confidence gate
    assert qa_contract().workflow.stages == ALL_STAGE_KINDS
    assert workflow_grammar(qa_contract()).version.startswith("grammar/2[")
    # the frozen benchmark gets grammar/1 only because the adapter's contracts say so
    for contract in legacy_contracts().values():
        assert contract.workflow.stages == LEGACY_STAGE_KINDS
        assert workflow_grammar(contract).version == GRAMMAR_VERSION
    # an omitted vocabulary is normalized, so it hashes like the explicit one
    assert qa_contract().contract_hash == qa_contract(stages=ALL_STAGE_KINDS).contract_hash
    assert qa_contract(stages=tuple(reversed(ALL_STAGE_KINDS))).workflow.stages == ALL_STAGE_KINDS
    # and the vocabulary is part of identity
    assert qa_contract().contract_hash != qa_contract(stages=LEGACY_STAGE_KINDS).contract_hash


@pytest.mark.parametrize(
    ("make", "stages", "match"),
    [
        (ticket_contract, (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE), "context"),
        (ticket_contract, (StageKind.DIRECT, StageKind.CONFIDENCE_GATE), "context"),
        (qa_contract, (StageKind.VERIFY,), "cannot produce an answer"),
        (qa_contract, (StageKind.DIRECT, StageKind.FILTER), "only occur on the GATHER"),
        (qa_contract, (StageKind.GATHER, StageKind.EXTRACT, StageKind.DIRECT), "only occur"),
    ],
)
def test_an_impossible_vocabulary_is_refused_with_the_contract(make, stages, match):
    with pytest.raises(ValueError, match=match):  # ContractError, raised during validation
        make(stages=stages)


def test_a_vocabulary_cannot_repeat_a_stage():
    with pytest.raises(ValidationError):
        WorkflowSpec(stages=(StageKind.DIRECT, StageKind.DIRECT))


def test_the_contract_workflow_policy_changes_the_admissible_genomes():
    checker = ConstraintChecker(config=RUNTIME)
    retrieval_only = (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE)

    def first(contract):
        return {s.kind for s in checker.admissible_successors(Genome(), contract)}

    assert first(ticket_contract()) == {StageKind.DIRECT}
    assert first(qa_contract(stages=retrieval_only)) == {StageKind.GATHER}
    assert first(qa_contract()) == {StageKind.GATHER, StageKind.DIRECT}
    # the gate is offered after SYNTHESIZE only when the contract's vocabulary includes it
    answered = Genome.of(gather(), extract(), synth())
    no_gate = tuple(k for k in ALL_STAGE_KINDS if k is not StageKind.CONFIDENCE_GATE)
    gate_kinds = [
        {s.kind for s in checker.admissible_successors(answered, c)}
        for c in (qa_contract(), qa_contract(stages=no_gate))
    ]
    assert gate_kinds == [
        {StageKind.VERIFY, StageKind.CONFIDENCE_GATE},
        {StageKind.VERIFY},
    ]
    sizes = {
        name: workflow_grammar(c).language_size()
        for name, c in {
            "full": qa_contract(),
            "no_gate": qa_contract(stages=no_gate),
            "direct": ticket_contract(),
        }.items()
    }
    retrieval = sizes["no_gate"] - sizes["direct"]
    assert sizes["full"] == sizes["direct"] + 3 * retrieval  # gate: none | 50 | 100
    # and the same genome is admitted or refused depending only on the contract
    genome = Genome.of(direct())
    assert checker.is_valid(genome, qa_contract())
    assert codes(checker.check(genome, qa_contract(stages=retrieval_only))) == {V.STAGE_UNSUPPORTED}


@pytest.mark.parametrize(
    ("contract", "stages", "expected"),
    [
        (make_contract(), (direct(),), {V.STAGE_UNSUPPORTED}),  # legacy vocabulary
        (make_contract(), (gather(), extract(), synth(), gate()), {V.STAGE_UNSUPPORTED}),
        (ticket_contract(), (gather(), extract(), synth()), {V.STAGE_UNSUPPORTED}),
        (qa_contract(), (gather(GatherSource.API), extract(), synth()), {V.SOURCE_NOT_ALLOWED}),
        (
            snapshot_contract(allowed_sources=(JEV,), interaction_required=True),
            (direct(),),
            {V.INTERACTION_REQUIRES_JEV},
        ),
    ],
)
def test_stages_unsupported_by_the_contract_are_rejected(contract, stages, expected):
    checker = ConstraintChecker()
    genome = Genome.from_stages(stages)
    assert codes(checker.check(genome, contract)) == expected
    assert not checker.is_valid(genome, contract, complete=False)


def test_unsupported_sources_stay_unsupported():
    # an uploaded dataset can only be read with fetch; jev (web/interactive) is never executable
    with pytest.raises(ValueError, match="cannot be gathered"):
        qa_contract(allowed_sources=(GatherSource.FETCH, JEV))
    checker = ConstraintChecker(config=RUNTIME)
    proposals = MMASACO().propose(40, SearchContext(snapshot_contract(), checker, seed=3))
    assert not any(s.kind == "GATHER" and s.source is JEV for g in proposals for s in g.stages)


# == 5. the search space is finite, exactly counted, and generators stay inside it ===============
def test_language_sizes_and_lengths_are_finite_and_pinned():
    # Counting memoizes on grammar states; it would recurse forever on a cyclic grammar.
    assert Grammar().language_size() == 340_200  # legacy: unchanged
    assert FULL.language_size() == 1_020_610  # = 3 x legacy (gate: none|50|100) + 10 DIRECT
    assert workflow_grammar(ticket_contract()).language_size() == 10
    assert Grammar().max_genome_length() == 8
    assert FULL.max_genome_length() == MAX_GENOME_STAGES == 9
    assert MAX_STEPS == MAX_GENOME_STAGES + 1


@pytest.mark.parametrize(
    "kinds",
    [
        (StageKind.DIRECT, StageKind.VERIFY),
        (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE, StageKind.CONFIDENCE_GATE),
        (
            StageKind.GATHER,
            StageKind.FILTER,
            StageKind.EXTRACT,
            StageKind.REASON,
            StageKind.SYNTHESIZE,
            StageKind.DIRECT,
        ),
    ],
)
def test_grammar_enumeration_is_exact_and_every_generated_genome_compiles(kinds):
    grammar = Grammar(kinds)
    genomes = list(grammar.enumerate())
    assert len(genomes) == grammar.language_size()
    assert len({g.genome_hash for g in genomes}) == len(genomes)
    for g in genomes:
        assert grammar.validate(g) == ()
        assert len(g) <= MAX_GENOME_STAGES
        compile_genome(g)


def _representative(space: tuple[Genome, ...]) -> list[Genome]:
    """The first genome using each configured stage, plus a deterministic stride sample."""
    chosen: dict[str, Genome] = {}
    for g in space:
        for s in g.stages:
            chosen.setdefault(s.model_dump_json(), g)
    picked = {g.genome_hash: g for g in [*chosen.values(), *space[::97]]}
    return list(picked.values())


def test_every_admissible_genome_for_an_uploaded_dataset_passes_admission():
    space = qa_space()
    assert len(space) == 3 * 2_754 + 6  # 2,754 = the legacy one-source space (one verifier)
    assert len({g.genome_hash for g in space}) == len(space)
    checker, contract = ConstraintChecker(config=ONE_VERIFIER), qa_contract()
    covered = set()
    for g in space:
        assert checker.check(g, contract) == ()
        covered |= {c for s in g.stages for c in capabilities(s)}
    assert covered == set(Capability)  # every capability is reachable for this dataset
    for g in _representative(space):  # (the execution tests below run these same genomes)
        compile_genome(g)
    direct_only = list(ConstraintChecker(config=RUNTIME).enumerate_admissible(ticket_contract()))
    assert len(direct_only) == 6 and all(g.stages[0].kind == "DIRECT" for g in direct_only)


@pytest.mark.parametrize("make", [RandomSearch, MMASACO])
def test_optimizer_proposals_under_a_contract_stay_in_the_admissible_space(make):
    context = SearchContext(
        contract=qa_contract(), checker=ConstraintChecker(config=ONE_VERIFIER), seed=7
    )
    proposals = make().propose(60, context)
    ensure_admissible(proposals, context)
    assert {g.genome_hash for g in proposals} <= {g.genome_hash for g in qa_space()}


def test_bounded_generation_terminates_even_when_the_space_is_tiny():
    checker = ConstraintChecker(config=RUNTIME)
    context = SearchContext(contract=ticket_contract(), checker=checker, seed=1)
    proposals = MMASACO().propose(50, context)  # only 6 genomes exist: proposes them, then stops
    assert 1 <= len(proposals) <= 6
    rng = random.Random(0)
    for _ in range(200):
        g = construct_genome(context, rng, lambda _p, options: [1.0] * len(options))
        assert g is not None and len(g) <= 2


def test_retries_are_bounded_for_every_admissible_genome():
    m, contract = StaticCostModel(), qa_contract()
    assert max(m.estimate(g, contract).max_retries for g in qa_space()) == 2  # retry-2 x 1
    assert max(len(g) for g in qa_space()) <= MAX_GENOME_STAGES


# == 6. legacy benchmarks: unchanged, and only through the adapter ===============================
# Measured with the PRE-#36 code (main @ 845d79b, RuntimeTask path) and with this code (contract
# path): identical sets. A-001 allows fetch + jev; the runner refuses jev.
LEGACY_A001_ONE_VERIFIER = (
    2_754,
    "d49e4ad4f1e542f9940eb0a81ac5d90b8e5b1f7567fb46e10bf571d6f11df332",
)
LEGACY_API_ONE_VERIFIER = (
    2_754,
    "0d1411e2918606fe1201ce0afb26c4c40b4807d2a2f066891272703e226edae4",
)


def _space_id(checker: ConstraintChecker, contract: TaskContract) -> tuple[int, str]:
    hashes = sorted(g.genome_hash for g in checker.enumerate_admissible(contract))
    return len(hashes), canonical_hash(hashes)


def test_legacy_benchmark_grammar_reaches_search_only_through_the_adapter():
    task = legacy_execution_tasks()["A-001"]
    assert task.contract.workflow.stages == LEGACY_STAGE_KINDS
    assert _space_id(ConstraintChecker(config=ONE_VERIFIER), task.contract) == (
        LEGACY_A001_ONE_VERIFIER
    )
    runner = WorkflowRunner(model=None, benchmark_hash="b")
    assert runner.versions(task).grammar_version == "grammar/1"
    assert runner.versions(qa_task()).grammar_version == workflow_grammar(qa_contract()).version
    # generic code never branches on the benchmark: no task class / legacy evaluator in grammar
    for module in ("core/grammar.py", "core/constraints.py", "core/cost_model.py"):
        text = (ROOT / module).read_text(encoding="utf-8")
        assert "TaskClass" not in text and "LEGACY_FIELD_MATCH" not in text
        assert "task_class" not in text
    from core import task_contract

    for fn in (
        task_contract.supported_stage_kinds,
        task_contract.WorkflowSpec,
        task_contract._check_workflow,
        task_contract.workflow_grammar,
    ):
        source = inspect.getsource(fn)
        assert "LEGACY" not in source and "task_class" not in source and "TaskClass" not in source


def test_legacy_grammar_identity_and_language_are_unchanged():
    assert Grammar().version == GRAMMAR_VERSION == "grammar/1"
    assert Grammar().kinds == LEGACY_STAGE_KINDS
    contract = make_contract(allowed_sources=(GatherSource.API,))  # legacy-vocabulary contract
    assert _space_id(ConstraintChecker(config=ONE_VERIFIER), contract) == LEGACY_API_ONE_VERIFIER


@pytest.mark.parametrize("name", sorted(MVP_GENOMES))
def test_mvp_genomes_are_admissible_everywhere_and_compile_to_the_same_dag(name):
    genome = MVP_GENOMES[name]
    for contract in (make_contract(), snapshot_contract(), qa_contract()):
        assert ConstraintChecker(config=RUNTIME).check(genome, contract) == ()
    assert compile_genome(genome) == compile_genome(genome, Grammar())


def test_workflow_memory_refuses_results_from_another_vocabulary():
    """Warm-start memory is keyed on grammar/1; extended-vocabulary results fail closed."""
    from memory.warm_start import IncompatibleMemory, update_from_experiment

    run_versions = WorkflowRunner(model=None, benchmark_hash="b").versions(qa_task())
    results = {
        "optimizer": MMASACO.name,
        "suite": "rivers",
        "versions": [
            {"run_versions": run_versions.model_dump(mode="json"), "evaluator_version": "e"}
        ],
    }
    with pytest.raises(IncompatibleMemory, match="grammar/2"):
        update_from_experiment(results, MMASACO())


# == 7. no RuntimeTask / TaskSpec authority comes back ===========================================
LEGACY_AUTHORITY = {"RuntimeTask", "TaskSpec", "TaskClass", "GroundTruth"}


@pytest.mark.parametrize(
    "module",
    [
        "core/grammar.py",
        "core/constraints.py",
        "core/cost_model.py",
        "core/task_contract.py",
        "runtime/runner.py",
        "runtime/executors/gate.py",
        "runtime/executors/gemma_stages.py",
        "optimizers/construct.py",
    ],
)
def test_the_grammar_path_imports_no_legacy_authority(module):
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not imported & LEGACY_AUTHORITY


# == 8. rejection happens at admission, before any model call ====================================
@pytest.mark.parametrize(
    ("task", "stages", "expected"),
    [
        (ExecutionTask(contract=make_contract(), example=ExampleInput(
            row_id="r", values={"question": "q?", "snapshot_id": "snap-001"})),
         (direct(),), V.STAGE_UNSUPPORTED),  # legacy vocabulary
        (qa_task(), INVALID["evidence-check-without-pages"][0], V.UNSATISFIED_DEPENDENCY),
        (qa_task(), INVALID["second-gate"][0], V.AFTER_TERMINAL),
        (qa_task(), (gather(), extract()), V.MISSING_REQUIRED_STAGE),
        (qa_task(), INVALID["decompose-twice"][0], V.PLACEMENT),
        (ticket_task(), (gather(), extract(), synth()), V.STAGE_UNSUPPORTED),
        (qa_task(qa_contract(stages=(StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE))),
         (direct(),), V.STAGE_UNSUPPORTED),  # the contract's policy, not the runner, decides
    ],
)  # fmt: skip
def test_invalid_genomes_are_rejected_before_any_model_execution(task, stages, expected):
    model = RowModel("answer", "Seine")
    runner = WorkflowRunner(model=model, benchmark_hash="inline")
    with pytest.raises(InadmissibleGenome) as err:
        runner.run_sync(Genome.from_stages(stages), task)
    assert expected in {v.code for v in err.value.violations}
    assert model.requests == []  # nothing reached the model


# == 9. admissible genomes execute against uploaded datasets (#37 runtime) =======================
def test_every_direct_workflow_executes_on_a_dataset_without_context():
    contract = ticket_contract()
    task = ticket_task(contract)
    for genome in ConstraintChecker(config=RUNTIME).enumerate_admissible(contract):
        model = TicketModel()
        result = run_row(genome, task, model)
        assert result.failure is None, (genome.canonical_json(), result.failure)
        assert result.answer.values == {"label": "billing"}
        assert result.budget_usage.tool_calls == 0 and len(model.requests) == 1
        assert "I was charged twice" in model.requests[0].input_text
        assert contract.instructions in model.requests[0].input_text


def test_a_no_context_contract_drives_a_direct_search_end_to_end():
    """ContractSuite -> run_search -> WorkflowRunner (MAF), on a dataset no retrieval can read."""
    pytest.importorskip("agent_framework")
    contract = ticket_contract()
    splits = DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("t1", "t2", "t3")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("t4",)),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=("t5",)),
        ),
    )
    suite, references = contract_suite(contract, splits, TICKET_DATA)
    model = TicketModel()
    runner = WorkflowRunner(model=model, benchmark_hash="inline")
    executed: list[str] = []
    proposed: list[Genome] = []

    def run_workflow(genome, task, trial, seed):
        executed.append(task.id)
        proposed.append(genome)
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    train, val = search_tasks(suite)
    evaluate = make_evaluate_fn(run_workflow, ContractEvaluator(references), (*train, *val))
    config = ExperimentConfig(budget=4, batch_size=2, trials=1)
    result = run_search(MMASACO(), evaluate, suite, config, seed=0, checker=runner.checker)

    assert 1 <= result["workflow_evaluations"] <= 4  # only 6 genomes exist; ACO dedupes
    assert result["train_pass_rate"] == 1.0 and result["champion"]["feasible"]
    assert proposed and all(g.stages[0].kind == "DIRECT" for g in proposed)
    # #37 split invariants: the final-test row never ran; only optimization rows may feed back
    assert "t5" not in executed and set(executed) <= {"t1", "t2", "t3", "t4"}
    assert {t.id for t in train} == {"t1", "t2", "t3"} and {t.id for t in val} == {"t4"}
    suite.splits.check_feedback(["t1", "t2", "t3"])
    for row in ("t4", "t5"):  # validation selects, test only reports
        with pytest.raises(SplitAccessError):
            suite.splits.check_feedback([row])
    run = runner.run_sync(Genome.of(direct()), train[0])
    assert run.key.contract_hash == contract.contract_hash
    assert run.key.versions.grammar_version == workflow_grammar(contract).version


class RiverModel:
    """Scripted backend that reads "The <River> flows through" from the gathered passage only.
    Without a passage (DIRECT) it has nothing to read and answers ""."""

    model_hash = "river-model"

    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        import re

        self.requests.append(request)
        text = request.input_text
        if request.prompt_template_id.startswith("extract"):
            m = re.search(r"The (\w+) flows through \w+", text.split("Pages:")[-1])
            facts = [{"field": "answer", "value": m[1], "page_id": "passage", "quote": m[0]}]
            body = {"facts": facts if m else []}
        elif request.prompt_template_id.startswith("direct"):
            body = {"answer": {"answer": ""}}
        else:
            m = re.search(r'\] answer = "([^"]*)"', text)
            body = {"answer": {"answer": m[1] if m else ""}, "citations": {"answer": [0]}}
        return GenerationResponse(
            text=json.dumps(body), parsed=body, prompt_tokens=30, completion_tokens=5,
            model_hash=self.model_hash,
        )  # fmt: skip


def test_uploaded_dataset_flows_from_grammar_to_evaluationspec_dispatcher_to_feedback():
    """TaskContract -> workflow.stages grammar -> execution (#37 runtime, MAF) -> EvaluationSpec
    dispatcher (#38) -> deterministic Evaluation (+ EvaluatorRecord) -> optimizer feedback."""
    pytest.importorskip("agent_framework")
    from core.evaluation_spec import EVALUATOR_VERSIONS
    from core.results import Verdict

    contract = qa_contract()
    splits = DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("q1",)),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("q2",)),
        ),
    )
    suite, references = contract_suite(contract, splits, QA_DATA)
    # targets stay with the evaluator: execution sees inputs + context only
    assert all(set(t.example.values) == {"question", "passage"} for t in suite.tasks)
    assert "Seine" not in repr(references)

    # 1. the grammar comes from workflow.stages (DIRECT + gate available for this dataset)
    grammar = workflow_grammar(contract)
    assert {StageKind.DIRECT, StageKind.CONFIDENCE_GATE} <= set(grammar.kinds)
    model = RiverModel()
    runner = WorkflowRunner(
        model=model, benchmark_hash="inline", checker=ConstraintChecker(config=RUNTIME)
    )
    evaluator = ContractEvaluator(references)
    q1 = next(t for t in suite.tasks if t.id == "q1")

    def judged(genome):
        return evaluator.evaluate_run(q1, runner.run_sync(genome, q1))

    # 2-4. execution -> EvaluationSpec dispatcher -> deterministic Evaluation with its record
    gated = judged(Genome.of(gather(), extract(), synth(SynthesizeMethod.CITE_EVIDENCE), gate()))
    record = gated.evaluation.evaluator
    assert gated.evaluation.verdict is Verdict.PASS and gated.execution.answer.values == {
        "answer": "Seine"
    }
    assert (record.kind, record.version) == (
        "exact_match",
        EVALUATOR_VERSIONS[contract.evaluation.evaluator],
    )
    assert record.spec_hash == contract.evaluation.identity_hash
    assert record.ok and record.passed and record.quality == 1.0
    assert gated.execution.key.versions.grammar_version == grammar.version
    # a DIRECT answer without the passage is a deterministic FAIL, not an evaluator failure
    direct_run = judged(Genome.of(direct()))
    assert direct_run.evaluation.verdict is Verdict.FAIL
    assert direct_run.evaluation.evaluator.ok and direct_run.evaluation.evaluator.passed is False
    # a confidence-gate abstention (LOW_CONFIDENCE) is judged the same way
    abstain = evaluator.evaluate_run(
        q1, run_row(Genome.of(gather(), extract(), synth(), gate()), q1, RowModel(
            "answer", "Seine", quote="not in the passage"))
    )  # fmt: skip
    assert abstain.execution.failure.kind is FailureKind.LOW_CONFIDENCE
    assert abstain.evaluation.verdict is Verdict.FAIL and abstain.evaluation.evaluator.ok

    # 5. optimizer feedback: a search observes only evaluator-judged runs on optimization rows
    observed = []

    class Recording(MMASACO):
        def observe(self, results):
            observed.extend(results)
            super().observe(results)

    def run_workflow(genome, task, trial, seed):
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    train, val = search_tasks(suite)
    evaluate = make_evaluate_fn(run_workflow, evaluator, (*train, *val))
    result = run_search(
        Recording(), evaluate, suite, ExperimentConfig(budget=6, batch_size=2, trials=1), seed=0,
        checker=runner.checker,
    )  # fmt: skip
    assert result["workflow_evaluations"] == 6 and observed
    assert {r.execution.task_id for r in observed} == {"q1"}  # validation never feeds back
    assert all(r.evaluation.evaluator.kind == "exact_match" for r in observed)
    assert all(
        r.evaluation.evaluator.spec_hash == contract.evaluation.identity_hash for r in observed
    )
    assert result["champion"]["validation_pass_rate"] == 1.0  # Tiber read from q2's own passage


def test_an_unrunnable_evaluation_spec_fails_closed_before_the_grammar_runs_anything():
    """No expected values for a row: refused by the dispatcher preflight, zero model calls."""
    task = ticket_task()

    def run_workflow(*_):  # the runtime (hence the model) must never be reached
        raise AssertionError("execution must not be reached")

    with pytest.raises(ContractError, match="no expected values"):
        make_evaluate_fn(run_workflow, ContractEvaluator({}), (task,))


def test_admissible_workflows_execute_on_a_dataset_with_context():
    contract = qa_contract()
    task = qa_task(contract)
    sample = _representative(qa_space())
    assert len(sample) > 80
    for genome in sample:
        result = run_row(genome, task, RowModel("answer", "Seine"))
        assert result.failure is None, (genome.canonical_json(), result.failure)
        assert result.answer.values == {"answer": "Seine"}
        retrieval = genome.stages[0].kind == "GATHER"
        assert result.budget_usage.tool_calls == (1 if retrieval else 0)  # one context column
        if genome.stages[-1].kind == "CONFIDENCE_GATE":
            [fe] = result.evidence
            assert fe.spans[0].page_id == "passage"


def test_a_gate_abstains_when_the_answer_is_not_supported():
    unsupported = RowModel("answer", "Seine", quote="not in the passage")
    result = run_row(Genome.of(gather(), extract(), synth(), gate()), qa_task(), unsupported)
    assert result.failure.kind is FailureKind.LOW_CONFIDENCE
    assert result.failure.stage_index == 3


def test_context_workflows_run_end_to_end_through_maf():
    pytest.importorskip("agent_framework")
    runner = WorkflowRunner(
        model=RowModel("answer", "Seine"),
        benchmark_hash="inline",
        checker=ConstraintChecker(config=RUNTIME),
    )
    task = qa_task()
    for genome in (Genome.of(direct(), verify()), Genome.of(*VALID["verified-then-gated"])):
        result = runner.run_sync(genome, task)
        assert result.failure is None and result.answer.values == {"answer": "Seine"}
        assert result.key.versions.grammar_version == workflow_grammar(task.contract).version


# == 10. the two new executors ===================================================================
def _gate_input(answer: Answer, min_support: SupportThreshold, pages: tuple[Page, ...]):
    return ExecutorInput(stage_index=3, stage=gate(min_support), payload=answer, source_pages=pages)


def test_confidence_gate_measures_evidence_support():
    page = Page(page_id="p", source_ref="r", content="The Seine flows through Paris.")
    start = page.content.index("Seine")
    schema = _schema(river=FieldType.STRING, city=FieldType.STRING)
    half = Answer(
        values={"river": "Seine", "city": "Paris"},
        evidence=(FieldEvidence(field="river", spans=(page.span(start, start + 5),)),),
    )
    originals = {"p": page}
    assert answer_support(half, schema, originals) == 0.5
    assert answer_support(Answer(values={"river": "Seine"}), schema, originals) == 0.0
    ctx = make_ctx(ExecutionTask(contract=make_contract(answer_schema=schema), example=ExampleInput(
        row_id="r", values={"question": "q?", "snapshot_id": "snap-001"})))  # fmt: skip
    gate_exec = ConfidenceGateExecutor()
    passed = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.HALF, (page,)), ctx))
    assert passed.payload == half
    refused = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.ALL, (page,)), ctx))
    assert refused.failure.kind is FailureKind.LOW_CONFIDENCE
    no_pages = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.HALF, ()), ctx))
    assert no_pages.failure.kind is FailureKind.LOW_CONFIDENCE  # nothing verifies without pages


def test_direct_executor_fails_closed_and_keeps_only_schema_fields():
    task = ticket_task()
    inp = ExecutorInput(stage_index=0, stage=direct(), payload=task)
    no_model = asyncio.run(DirectExecutor().run(inp, make_ctx(task)))
    assert no_model.failure.kind is FailureKind.MODEL_ERROR

    class Model(RowModel):
        def __init__(self, body):
            super().__init__("label", "x")
            self.body = body

        async def generate(self, request):
            self.requests.append(request)
            return GenerationResponse(
                text=json.dumps(self.body),
                parsed=self.body,
                prompt_tokens=1,
                completion_tokens=1,
                model_hash="m",
            )

    bad = asyncio.run(DirectExecutor().run(inp, make_ctx(task, Model({"label": "billing"}))))
    assert bad.failure.kind is FailureKind.SCHEMA_INVALID
    extra = Model({"answer": {"label": "billing", "secret": 1}})
    out = asyncio.run(DirectExecutor().run(inp, make_ctx(task, extra)))
    assert out.payload == Answer(values={"label": "billing"})
    assert extra.requests[0].prompt_template_id == "direct.answer"
    assert task.inputs_text in extra.requests[0].input_text
