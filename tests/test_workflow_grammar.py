"""Generic workflow grammar (issue #21): the typed stage vocabulary, admission invariants, the
bounded search space, compilation, and execution against uploaded-dataset rows.

The model backend is a scripted test double; nothing here calls a real model.
"""

from __future__ import annotations

import asyncio
import json
import random
from functools import cache

import pytest
from pydantic import ValidationError

from benchmarks.legacy_adapter import legacy_contracts
from compiler.dag import CompileError, compile_genome
from core.canonical import canonical_hash
from core.constraints import ConstraintChecker, ConstraintConfig
from core.cost_model import CostTable, StaticCostModel
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
from core.task_contract import TaskContract, workflow_grammar, workflow_stage_kinds
from core.task_spec import RuntimeTask, TaskClass
from core.violations import ViolationCode as V
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
from tests.conftest import extract, flt, gather, make_caps, make_runtime_task, reason, synth, verify
from tests.contract_helpers import classification_contract, qa_contract
from tests.runtime_helpers import drive, make_ctx

FULL = Grammar(ALL_STAGE_KINDS)
JEV = GatherSource.JEV
# What ``WorkflowRunner`` admits by default: jev and self_consistency are not executable.
RUNTIME = ConstraintConfig(
    unavailable_sources=(GatherSource.JEV,),
    unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,),
)


def direct(method=DirectMethod.ANSWER) -> DirectStage:
    return DirectStage(method=method)


def gate(min_support=SupportThreshold.HALF) -> ConfidenceGateStage:
    return ConfidenceGateStage(min_support=min_support)


def codes(violations) -> set[V]:
    return {v.code for v in violations}


# -- uploaded-dataset fixtures --------------------------------------------------------------------
QA_ROW = {
    "qid": "q1",
    "question": "Which river flows through Paris?",
    "passage": "Paris is the capital of France.\n\nThe Seine flows through Paris.\n\n"
    "Bananas grow in warm climates.",
}
QA_QUOTE = "The Seine flows through Paris"
TICKET_ROW = {"id": "t1", "text": "I was charged twice for my plan this month."}


class RowPages:
    """A dataset row's context columns served as frozen pages (the ``fetch`` source)."""

    def __init__(self, contract: TaskContract, rows: list[dict[str, str]]) -> None:
        ds = contract.dataset
        self.dataset_id = ds.dataset_id
        self._rows = {r[ds.id_column]: {c: r[c] for c in ds.context_columns} for r in rows}

    def list_page_ids(self, snapshot_id: str) -> list[str]:
        return sorted(self._rows[snapshot_id])

    def read_page(self, snapshot_id: str, page_id: str) -> Page:
        return Page(
            page_id=page_id,
            source_ref=f"dataset://{self.dataset_id}/{snapshot_id}/{page_id}",
            content=self._rows[snapshot_id][page_id],
        )


def row_task(contract: TaskContract, row: dict[str, str]) -> RuntimeTask:
    """The runtime view of one row (what a TaskContract-driven caller would hand the runtime)."""
    inputs = "\n".join(f"{c}: {row[c]}" for c in contract.dataset.input_columns)
    return RuntimeTask(
        id=row[contract.dataset.id_column],
        task_class=TaskClass.A,  # legacy label required by RuntimeTask; not used by the grammar
        question=f"{contract.instructions}\n{inputs}",
        answer_schema=contract.output_schema,
        caps=make_caps(),
        allowed_sources=(GatherSource.FETCH,),
        snapshot_id=row[contract.dataset.id_column],
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


def qa_model() -> RowModel:
    return RowModel("answer", "Seine")


def run_row(genome: Genome, task: RuntimeTask, pages, model):
    """Compile and execute one genome through the real executors (no MAF needed)."""
    dag = compile_genome(genome)
    runner = StageRunner(dag, default_executors(pages=pages), make_ctx(task, model))
    return asyncio.run(drive(dag, runner, task))


@cache
def qa_space() -> tuple[Genome, ...]:
    """Every admissible genome for a QA dataset with a context column, on the shipped runtime,
    with one active verifier (keeps the exhaustive tests fast; two verifiers is 49,092)."""
    checker = ConstraintChecker(
        grammar=workflow_grammar(qa_contract()),
        config=RUNTIME.model_copy(update={"max_active_verifiers": 1}),
    )
    return tuple(checker.enumerate_admissible(row_task(qa_contract(), QA_ROW)))


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


def test_cost_and_budget_implications_respect_the_vocabulary(task):
    m = StaticCostModel(CostTable(proven_lower_bound=True))
    for spec in (s for k in StageKind for s in all_stage_specs(k)):
        m.estimate(Genome.of(spec), task)  # every configuration is costed
    assert m.estimate(Genome.of(direct()), task).tool_calls == 0
    gated = Genome.of(gather(), extract(), synth(), gate())
    assert m.estimate(gated, task).tokens == m.estimate(Genome.of(*gated.stages[:3]), task).tokens
    # the empty prefix's cheapest completion depends on which producers the task supports
    legacy = m.estimate(Genome(), task)
    assert (legacy.tokens, legacy.tool_calls) == (1500 + 800, 2)  # unchanged legacy bound
    generic = m.estimate(Genome(), task, kinds=ALL_STAGE_KINDS)
    assert (generic.tokens, generic.tool_calls) == (800, 0)  # DIRECT(answer)
    # a token cap that rules out retrieval leaves exactly the cheap DIRECT configuration
    tight = make_runtime_task(caps=make_caps(tokens=1000))
    proven = StaticCostModel(CostTable(proven_lower_bound=True))
    assert ConstraintChecker(cost_model=proven).admissible_successors(Genome(), tight) == ()
    qa = ConstraintChecker(grammar=workflow_grammar(qa_contract()), cost_model=proven)
    assert qa.admissible_successors(Genome(), tight) == (direct(),)


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
def test_representative_compositions_are_admissible_and_compile(name, task):
    genome = Genome.from_stages(VALID[name])
    assert FULL.validate(genome) == ()
    assert ConstraintChecker(grammar=FULL).check(genome, task) == ()
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
    assert codes(ConstraintChecker(grammar=FULL).check(longest)) == {V.TOO_MANY_VERIFIERS}


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


def test_missing_producer_messages_name_the_options():
    [empty] = [v for v in FULL.validate(Genome()) if v.code is V.MISSING_REQUIRED_STAGE]
    assert "GATHER -> EXTRACT -> SYNTHESIZE" in empty.message and "DIRECT" in empty.message
    legacy = [v.message for v in Grammar().validate(Genome()) if v.code is V.MISSING_REQUIRED_STAGE]
    assert legacy == [f"required stage {k} is missing" for k in ("GATHER", "EXTRACT", "SYNTHESIZE")]
    no_retrieval = workflow_grammar(classification_contract())
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
    checker = ConstraintChecker(grammar=FULL)
    jev4 = Genome.of(gather(JEV, GatherMode.PARALLEL_4), extract(), synth(), gate())
    assert codes(checker.check(jev4)) == {V.JEV_PARALLEL_4}
    jev_regather = Genome.of(
        gather(JEV), extract(), synth(), verify(on_failure=FailureStrategy.REGATHER), gate()
    )
    assert codes(checker.check(jev_regather)) == {V.JEV_REGATHER}
    sc = verify(VerifyMethod.SELF_CONSISTENCY)
    assert codes(checker.check(Genome.of(direct(), sc))) == set()
    assert codes(
        ConstraintChecker(grammar=FULL, config=RUNTIME).check(Genome.of(direct(), sc))
    ) == {V.RUNTIME_UNAVAILABLE}


@pytest.mark.parametrize(
    ("grammar", "stages", "task_kw", "expected"),
    [
        (Grammar(), (direct(),), {}, {V.STAGE_UNSUPPORTED}),
        (Grammar(), (gather(), extract(), synth(), gate()), {}, {V.STAGE_UNSUPPORTED}),
        (
            workflow_grammar(classification_contract()),
            (gather(), extract(), synth()),
            {},
            {V.STAGE_UNSUPPORTED},
        ),
        (
            workflow_grammar(qa_contract()),
            (gather(GatherSource.API), extract(), synth()),
            {"allowed_sources": (GatherSource.FETCH,)},
            {V.SOURCE_NOT_ALLOWED},
        ),
        (
            FULL,
            (direct(),),
            {"allowed_sources": (JEV,), "interaction_required": True},
            {V.INTERACTION_REQUIRES_JEV},
        ),
    ],
)
def test_stages_unsupported_by_the_task_or_contract_are_rejected(
    grammar, stages, task_kw, expected
):
    checker = ConstraintChecker(grammar=grammar)
    genome = Genome.from_stages(stages)
    assert codes(checker.check(genome, make_runtime_task(**task_kw))) == expected
    assert not checker.is_valid(genome, make_runtime_task(**task_kw), complete=False)


def test_contract_vocabularies():
    k = StageKind
    assert workflow_stage_kinds(classification_contract()) == (k.VERIFY, k.DIRECT)
    assert workflow_stage_kinds(qa_contract()) == ALL_STAGE_KINDS
    assert workflow_grammar(qa_contract()).version.startswith("grammar/2[")
    for contract in legacy_contracts().values():  # the frozen benchmark keeps grammar/1
        assert workflow_stage_kinds(contract) == LEGACY_STAGE_KINDS
        assert workflow_grammar(contract).version == GRAMMAR_VERSION


# == 4. the search space is finite, exactly counted, and generators stay inside it ===============
def test_language_sizes_and_lengths_are_finite_and_pinned():
    # Counting memoizes on grammar states; it would recurse forever on a cyclic grammar.
    assert Grammar().language_size() == 340_200  # legacy: unchanged
    assert FULL.language_size() == 1_020_610  # = 3 x legacy (gate: none|50|100) + 10 DIRECT
    assert workflow_grammar(classification_contract()).language_size() == 10
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


def test_every_admissible_genome_for_an_uploaded_dataset_passes_admission():
    space = qa_space()
    assert len(space) == 3 * 2_754 + 6  # 2,754 = legacy fetch-only space (one verifier)
    assert len({g.genome_hash for g in space}) == len(space)
    checker = ConstraintChecker(grammar=workflow_grammar(qa_contract()), config=RUNTIME)
    task = row_task(qa_contract(), QA_ROW)
    covered = set()
    for g in space:
        assert checker.check(g, task) == ()
        covered |= {c for s in g.stages for c in capabilities(s)}
    assert covered == set(Capability)  # every capability is reachable for this dataset
    for g in _representative(space):  # (the execution tests below run these same genomes)
        compile_genome(g)
    no_context = ConstraintChecker(
        grammar=workflow_grammar(classification_contract()), config=RUNTIME
    )
    direct_only = list(
        no_context.enumerate_admissible(row_task(classification_contract(), TICKET_ROW))
    )
    assert len(direct_only) == 6 and all(g.stages[0].kind == "DIRECT" for g in direct_only)


@pytest.mark.parametrize("make", [RandomSearch, MMASACO])
def test_optimizer_proposals_under_a_contract_grammar_stay_in_the_admissible_space(make):
    checker = ConstraintChecker(
        grammar=workflow_grammar(qa_contract()),
        config=RUNTIME.model_copy(update={"max_active_verifiers": 1}),
    )
    context = SearchContext(task=row_task(qa_contract(), QA_ROW), checker=checker, seed=7)
    proposals = make().propose(60, context)
    ensure_admissible(proposals, context)
    space = {g.genome_hash for g in qa_space()}
    assert {g.genome_hash for g in proposals} <= space


def test_bounded_generation_terminates_even_when_the_space_is_tiny():
    checker = ConstraintChecker(grammar=workflow_grammar(classification_contract()), config=RUNTIME)
    context = SearchContext(
        task=row_task(classification_contract(), TICKET_ROW), checker=checker, seed=1
    )
    proposals = MMASACO().propose(50, context)  # only 6 genomes exist: proposes them, then stops
    assert 1 <= len(proposals) <= 6
    rng = random.Random(0)
    for _ in range(200):
        g = construct_genome(context, rng, lambda _p, options: [1.0] * len(options))
        assert g is not None and len(g) <= 2


def test_retries_are_bounded_for_every_admissible_genome():
    m = StaticCostModel()
    task = row_task(qa_contract(), QA_ROW)
    assert max(m.estimate(g, task).max_retries for g in qa_space()) == 2  # retry-2 x 1 verifier
    assert max(len(g) for g in qa_space()) <= MAX_GENOME_STAGES


# == 5. existing benchmark paths are preserved ===================================================
# Measured with the PRE-CHANGE code (main @ 845d79b) and with this code: identical.
LEGACY_API_ONE_VERIFIER = (
    2_754,
    "0d1411e2918606fe1201ce0afb26c4c40b4807d2a2f066891272703e226edae4",
)


def test_legacy_grammar_identity_and_language_are_unchanged():
    assert Grammar().version == GRAMMAR_VERSION == "grammar/1"
    assert Grammar().kinds == LEGACY_STAGE_KINDS
    runner = WorkflowRunner(model=None, benchmark_hash="b")
    assert runner.versions().grammar_version == "grammar/1"
    checker = ConstraintChecker(config=RUNTIME.model_copy(update={"max_active_verifiers": 1}))
    task = make_runtime_task(allowed_sources=(GatherSource.API,))
    hashes = sorted(g.genome_hash for g in checker.enumerate_admissible(task))
    assert (len(hashes), canonical_hash(hashes)) == LEGACY_API_ONE_VERIFIER


@pytest.mark.parametrize("name", sorted(MVP_GENOMES))
def test_mvp_genomes_are_admissible_everywhere_and_compile_to_the_same_dag(name, task):
    genome = MVP_GENOMES[name]
    for grammar in (Grammar(), FULL, workflow_grammar(qa_contract())):
        assert ConstraintChecker(grammar=grammar, config=RUNTIME).check(genome, task) == ()
    assert compile_genome(genome) == compile_genome(genome, Grammar())


# == 6. rejection happens at admission, before any model call ====================================
@pytest.mark.parametrize(
    ("grammar", "stages", "expected"),
    [
        (None, (direct(),), V.STAGE_UNSUPPORTED),  # default runner: legacy vocabulary
        (
            workflow_grammar(qa_contract()),
            INVALID["evidence-check-without-pages"][0],
            V.UNSATISFIED_DEPENDENCY,
        ),
        (workflow_grammar(qa_contract()), INVALID["second-gate"][0], V.AFTER_TERMINAL),
        (workflow_grammar(qa_contract()), (gather(), extract()), V.MISSING_REQUIRED_STAGE),
        (workflow_grammar(qa_contract()), INVALID["decompose-twice"][0], V.PLACEMENT),
        (
            workflow_grammar(classification_contract()),
            (gather(), extract(), synth()),
            V.STAGE_UNSUPPORTED,
        ),
    ],
)
def test_invalid_genomes_are_rejected_before_any_model_execution(grammar, stages, expected):
    model = qa_model()
    checker = ConstraintChecker(grammar=grammar, config=RUNTIME) if grammar else None
    runner = WorkflowRunner(
        model=model,
        benchmark_hash="b",
        pages=RowPages(qa_contract(), [QA_ROW]),
        checker=checker,
    )
    with pytest.raises(InadmissibleGenome) as err:
        runner.run_sync(Genome.from_stages(stages), row_task(qa_contract(), QA_ROW))
    assert expected in {v.code for v in err.value.violations}
    assert model.requests == []  # nothing reached the model


# == 7. admissible genomes execute against uploaded datasets =====================================
def test_every_direct_workflow_executes_on_a_dataset_without_context():
    contract = classification_contract()
    task = row_task(contract, TICKET_ROW)
    checker = ConstraintChecker(grammar=workflow_grammar(contract), config=RUNTIME)
    for genome in checker.enumerate_admissible(task):
        model = RowModel("label", "billing")
        result = run_row(genome, task, RowPages(contract, [TICKET_ROW]), model)
        assert result.failure is None, (genome.canonical_json(), result.failure)
        assert result.answer.values == {"label": "billing"}
        assert result.budget_usage.tool_calls == 0 and len(model.requests) == 1
        assert "I was charged twice" in model.requests[0].input_text


def _representative(space: tuple[Genome, ...]) -> list[Genome]:
    """The first genome using each configured stage, plus a deterministic stride sample."""
    chosen: dict[str, Genome] = {}
    for g in space:
        for s in g.stages:
            chosen.setdefault(s.model_dump_json(), g)
    picked = {g.genome_hash: g for g in [*chosen.values(), *space[::97]]}
    return list(picked.values())


def test_admissible_workflows_execute_on_a_dataset_with_context():
    contract = qa_contract()
    task = row_task(contract, QA_ROW)
    sample = _representative(qa_space())
    assert len(sample) > 80
    for genome in sample:
        result = run_row(genome, task, RowPages(contract, [QA_ROW]), qa_model())
        assert result.failure is None, (genome.canonical_json(), result.failure)
        assert result.answer.values == {"answer": "Seine"}
        retrieval = genome.stages[0].kind == "GATHER"
        assert result.budget_usage.tool_calls == (1 if retrieval else 0)  # one context column
        if genome.stages[-1].kind == "CONFIDENCE_GATE":
            [fe] = result.evidence
            assert fe.spans[0].page_id == "passage"


def test_a_gate_abstains_when_the_answer_is_not_supported():
    contract = qa_contract()
    unsupported = RowModel("answer", "Seine", quote="not in the passage")
    genome = Genome.of(gather(), extract(), synth(), gate())
    result = run_row(genome, row_task(contract, QA_ROW), RowPages(contract, [QA_ROW]), unsupported)
    assert result.failure.kind is FailureKind.LOW_CONFIDENCE
    assert result.failure.stage_index == 3


def test_contract_workflows_run_end_to_end_through_maf():
    pytest.importorskip("agent_framework")
    contract = qa_contract()
    checker = ConstraintChecker(grammar=workflow_grammar(contract), config=RUNTIME)
    runner = WorkflowRunner(
        model=qa_model(),
        benchmark_hash="b",
        pages=RowPages(contract, [QA_ROW]),
        checker=checker,
    )
    task = row_task(contract, QA_ROW)
    for genome in (Genome.of(direct(), verify()), Genome.of(*VALID["verified-then-gated"])):
        result = runner.run_sync(genome, task)
        assert result.failure is None and result.answer.values == {"answer": "Seine"}
        assert result.key.versions.grammar_version == checker.grammar.version


# == 8. the two new executors ====================================================================
def _gate_input(answer: Answer, min_support: SupportThreshold, pages: tuple[Page, ...]):
    return ExecutorInput(stage_index=3, stage=gate(min_support), payload=answer, source_pages=pages)


def test_confidence_gate_measures_evidence_support():
    from core.evidence import FieldEvidence
    from core.task_spec import AnswerField, AnswerSchema, FieldType

    page = Page(page_id="p", source_ref="r", content="The Seine flows through Paris.")
    start = page.content.index("Seine")
    schema = AnswerSchema(
        fields=(
            AnswerField(name="river", type=FieldType.STRING),
            AnswerField(name="city", type=FieldType.STRING),
        )
    )
    half = Answer(
        values={"river": "Seine", "city": "Paris"},
        evidence=(FieldEvidence(field="river", spans=(page.span(start, start + 5),)),),
    )
    originals = {"p": page}
    assert answer_support(half, schema, originals) == 0.5
    assert answer_support(Answer(values={"river": "Seine"}), schema, originals) == 0.0
    task = make_runtime_task(answer_schema=schema)
    ctx = make_ctx(task)
    gate_exec = ConfidenceGateExecutor()
    passed = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.HALF, (page,)), ctx))
    assert passed.payload == half
    refused = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.ALL, (page,)), ctx))
    assert refused.failure.kind is FailureKind.LOW_CONFIDENCE
    no_pages = asyncio.run(gate_exec.run(_gate_input(half, SupportThreshold.HALF, ()), ctx))
    assert no_pages.failure.kind is FailureKind.LOW_CONFIDENCE  # nothing verifies without pages


def test_direct_executor_fails_closed_and_keeps_only_schema_fields():
    task = row_task(classification_contract(), TICKET_ROW)
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
