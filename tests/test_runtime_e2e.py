"""End-to-end through the REAL Microsoft Agent Framework graph (skipped if MAF is missing:
``pip install -e ".[dev,maf]"``). Model backend is the scripted test double."""

import pytest

pytest.importorskip("agent_framework")

from compiler.dag import compile_genome  # noqa: E402
from compiler.maf_compiler import MAFCompiler  # noqa: E402
from core.constraints import ConstraintChecker  # noqa: E402
from core.cost_model import CostTable, StaticCostModel  # noqa: E402
from core.genome import Genome  # noqa: E402
from core.results import (  # noqa: E402
    BudgetCap,
    ExecutionResult,
    FailureKind,
    StageStatus,
)
from core.stages import GatherSource  # noqa: E402
from runtime.executors.registry import default_executors  # noqa: E402
from runtime.maf_nodes import StageNode  # noqa: E402
from runtime.mvp_genomes import GENOME_A, GENOME_B, GENOME_C, MVP_GENOMES  # noqa: E402
from runtime.runner import InadmissibleGenome, WorkflowRunner  # noqa: E402
from runtime.sources import DirectoryApiSource, DirectorySnapshotSource  # noqa: E402
from runtime.stage_runner import StageRunner  # noqa: E402
from tests.conftest import extract, gather, make_caps, make_task, synth  # noqa: E402
from tests.runtime_helpers import (  # noqa: E402
    QUOTE,
    FailingModel,
    ScriptedModel,
    make_ctx,
    write_snapshot,
)


@pytest.fixture
def root(tmp_path):
    write_snapshot(tmp_path)
    write_snapshot(tmp_path, "snap-big", extra_pages=3)
    return tmp_path


def runner_for(root, model):
    return WorkflowRunner(
        model=model,
        benchmark_hash="bench-1",
        pages=DirectorySnapshotSource(root),
        api=DirectoryApiSource(root),
    )


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_three_genomes_execute_end_to_end_through_the_same_entry_point(name, root):
    model = ScriptedModel()
    runner = runner_for(root, model)
    result = runner.run_sync(MVP_GENOMES[name], make_task(), trial=0, seed=11)

    assert isinstance(result, ExecutionResult)
    assert result.failure is None
    assert result.answer.values == {"capital": "Paris"}
    [fe] = result.evidence
    assert fe.field == "capital" and fe.spans[0].page_id == "p1"
    assert [t.status for t in result.stage_trace] == [StageStatus.OK] * len(MVP_GENOMES[name])
    assert result.budget_usage.tokens == 120 * len(model.requests)
    assert result.budget_usage.tool_calls == 3
    assert result.metrics.model_calls == len(model.requests) >= 2
    assert result.genome_hash == MVP_GENOMES[name].genome_hash
    assert result.key.versions.benchmark_hash == "bench-1"
    assert result.key.versions.model_hash == "scripted-model-v1"
    assert result.key.versions.compiler_version == MAFCompiler.version
    assert not {"verdict", "fitness"} & set(type(result).model_fields)


def test_execution_is_deterministic_for_identical_inputs(root):
    task = make_task()
    a = runner_for(root, ScriptedModel()).run_sync(GENOME_B, task, seed=5)
    b = runner_for(root, ScriptedModel()).run_sync(GENOME_B, task, seed=5)
    assert a.run_id == b.run_id
    assert a.answer == b.answer and a.evidence == b.evidence
    assert [(t.input_digest, t.output_digest) for t in a.stage_trace] == [
        (t.input_digest, t.output_digest) for t in b.stage_trace
    ]
    assert runner_for(root, ScriptedModel()).run_sync(GENOME_B, task, seed=6).run_id != a.run_id


def test_compilation_is_deterministic_and_yields_a_valid_maf_graph(root):
    compiler = MAFCompiler()
    assert compiler.to_dag(GENOME_C).dag_hash == compiler.to_dag(GENOME_C).dag_hash

    dag = compiler.to_dag(GENOME_C)
    task = make_task()
    stage_runner = StageRunner(dag, default_executors(), make_ctx(task))
    nodes = {n.node_id: StageNode(n, stage_runner, n.node_id == dag.end_node) for n in dag.nodes}
    workflow = compiler.build(dag, nodes)
    assert [e.id for e in workflow.get_executors_list()] == [n.node_id for n in dag.nodes]
    assert workflow.get_start_executor().id == dag.start_node
    assert [e.id for e in workflow.get_output_executors()] == [dag.end_node]
    # same genome -> same graph shape on a second build
    again = compiler.build(
        dag, {n.node_id: StageNode(n, stage_runner, n.node_id == dag.end_node) for n in dag.nodes}
    )
    assert [e.id for e in again.get_executors_list()] == [
        e.id for e in workflow.get_executors_list()
    ]


def test_executors_really_run_inside_the_maf_graph(root):
    model = ScriptedModel()
    result = runner_for(root, model).run_sync(GENOME_C, make_task())
    assert [t.kind.value for t in result.stage_trace] == [
        "GATHER", "EXTRACT", "VERIFY", "SYNTHESIZE", "VERIFY",
    ]  # fmt: skip
    assert [r.prompt_template_id for r in model.requests] == ["extract.direct", "synthesize.direct"]
    assert all(t.input_digest and t.output_digest for t in result.stage_trace)


def test_verify_retry_works_inside_the_maf_graph(root):
    model = ScriptedModel(extract_quotes=["not in any page", QUOTE])
    result = runner_for(root, model).run_sync(GENOME_C, make_task())
    assert result.failure is None and result.budget_usage.retries == 1
    assert [r.prompt_template_id for r in model.requests].count("extract.direct") == 2


def test_budget_breach_stops_the_maf_run_and_downstream_nodes_never_execute(root):
    model = ScriptedModel()
    task = make_task(snapshot_id="snap-big", caps=make_caps(tool_calls=3))
    result = runner_for(root, model).run_sync(GENOME_A, task)
    assert result.failure.kind is FailureKind.BUDGET_EXCEEDED
    assert result.failure.cap is BudgetCap.TOOL_CALLS
    assert model.requests == []  # EXTRACT/SYNTHESIZE never ran
    assert [t.kind.value for t in result.stage_trace] == ["GATHER"]


def test_provably_infeasible_genome_is_not_executed_but_reports_the_breach(root):
    model = ScriptedModel()
    runner = runner_for(root, model)
    runner.checker = ConstraintChecker(
        cost_model=StaticCostModel(CostTable(proven_lower_bound=True))
    )
    result = runner.run_sync(GENOME_A, make_task(caps=make_caps(tokens=1000)))
    assert (
        result.failure.kind is FailureKind.BUDGET_EXCEEDED
        and result.failure.cap is BudgetCap.TOKENS
    )
    assert result.stage_trace == () and model.requests == []


def test_a_failing_backend_stops_the_graph_midway_without_faking(root):
    result = runner_for(root, FailingModel()).run_sync(GENOME_A, make_task())
    assert result.failure.kind is FailureKind.MODEL_ERROR and result.answer is None
    assert [(t.kind.value, t.status.value) for t in result.stage_trace] == [
        ("GATHER", "ok"),
        ("EXTRACT", "failed"),
    ]


def test_no_model_configured_fails_loudly(root):
    result = runner_for(root, None).run_sync(GENOME_A, make_task())
    assert result.failure.kind is FailureKind.MODEL_ERROR
    assert result.key.versions.model_hash == "no-model"


def test_mock_api_source_works_through_the_graph(root):
    model = ScriptedModel(extract_quotes=['"capital":"Paris"'])
    genome = Genome.of(gather(GatherSource.API), extract(), synth())
    result = runner_for(root, model).run_sync(genome, make_task())
    assert result.failure is None
    assert result.evidence[0].spans[0].page_id == "api:country"
    assert result.budget_usage.tool_calls == 1


def test_structurally_invalid_genomes_are_refused_before_any_work(root):
    from core.stages import GatherMode

    bad = Genome.of(gather(GatherSource.JEV, GatherMode.PARALLEL_4), extract(), synth())
    model = ScriptedModel()
    with pytest.raises(InadmissibleGenome, match="jev_parallel_4"):
        runner_for(root, model).run_sync(bad, make_task())
    with pytest.raises(InadmissibleGenome):
        runner_for(root, model).run_sync(Genome.of(gather(), extract()), make_task())
    assert model.requests == []


def test_dag_matches_the_pure_compiler_output():
    assert (
        MAFCompiler().to_dag(GENOME_A).nodes
        == compile_genome(GENOME_A, version=MAFCompiler.version).nodes
    )
