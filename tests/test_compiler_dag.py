import os
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.dag import CompileError, DagEdge, DagError, DagNode, WorkflowDAG, compile_genome
from compiler.maf_compiler import MAFCompiler
from core.genome import Genome
from core.grammar import DataType
from tests.conftest import extract, gather, minimal_genome, reason, synth, verify

ROOT = Path(__file__).resolve().parent.parent


def full_genome() -> Genome:
    return Genome.of(gather(), extract(), verify(), reason(), synth(), verify())


def test_same_genome_compiles_to_the_same_dag():
    a, b = compile_genome(full_genome()), compile_genome(full_genome())
    assert a == b
    assert a.canonical_json() == b.canonical_json()
    assert a.dag_hash == b.dag_hash


def test_dag_hash_is_stable_across_processes():
    code = (
        "from compiler.dag import compile_genome;"
        "from tests.test_compiler_dag import full_genome;"
        "print(compile_genome(full_genome()).dag_hash)"
    )
    env = {
        "PYTHONPATH": str(ROOT),
        "PYTHONHASHSEED": "7",
        **{k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH", "TEMP", "TMP")},
    }
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == compile_genome(full_genome()).dag_hash


def test_different_genomes_compile_to_different_dags():
    assert compile_genome(minimal_genome()).dag_hash != compile_genome(full_genome()).dag_hash


def test_generated_dag_is_valid_typed_and_linear():
    dag = compile_genome(full_genome())
    dag.validate_graph()
    assert [n.stage_index for n in dag.nodes] == list(range(6))
    assert dag.nodes[0].input_type == DataType.TASK
    assert dag.nodes[-1].output_type == DataType.ANSWER
    assert dag.topological_order() == tuple(n.node_id for n in dag.nodes)
    assert dag.start_node == "00_gather" and dag.end_node == "05_verify"
    assert dag.genome_hash == full_genome().genome_hash


def test_incomplete_or_ill_typed_genomes_do_not_compile():
    with pytest.raises(CompileError):
        compile_genome(Genome.of(gather(), extract()))  # no Answer
    with pytest.raises(CompileError):
        compile_genome(Genome.of(gather(), synth()))  # Pages cannot feed SYNTHESIZE
    with pytest.raises(CompileError):
        compile_genome(Genome())


def _node(i, kind_stage, inp, out):
    return DagNode(
        node_id=f"n{i}", stage_index=i, stage=kind_stage, input_type=inp, output_type=out
    )


def _good_nodes():
    return [
        _node(0, gather(), DataType.TASK, DataType.PAGES),
        _node(1, extract(), DataType.PAGES, DataType.FACTS),
        _node(2, synth(), DataType.FACTS, DataType.ANSWER),
    ]


def _dag(nodes, edges):
    return WorkflowDAG(
        genome_hash="h", compiler_version="t", nodes=tuple(nodes), edges=tuple(edges)
    )


def test_hand_built_valid_dag_is_accepted():
    n = _good_nodes()
    dag = _dag(
        n,
        [
            DagEdge(source="n0", target="n1", data_type=DataType.PAGES),
            DagEdge(source="n1", target="n2", data_type=DataType.FACTS),
        ],
    )
    assert dag.start_node == "n0"


@pytest.mark.parametrize(
    ("edges", "why"),
    [
        (  # cycle
            [
                ("n0", "n1", DataType.PAGES),
                ("n1", "n2", DataType.FACTS),
                ("n2", "n1", DataType.ANSWER),
            ],
            "cycle",
        ),
        ([("n0", "n1", DataType.PAGES)], "sink"),  # n2 disconnected
        ([("n0", "n1", DataType.FACTS), ("n1", "n2", DataType.FACTS)], "mismatched"),  # wrong type
        ([("n0", "nX", DataType.PAGES), ("n1", "n2", DataType.FACTS)], "unknown"),
    ],
)
def test_invalid_graphs_are_rejected(edges, why):
    with pytest.raises(ValueError):
        _dag(_good_nodes(), [DagEdge(source=s, target=t, data_type=d) for s, t, d in edges])


def test_duplicate_node_ids_and_empty_dag_rejected():
    n = _good_nodes()
    with pytest.raises(ValueError):
        _dag([n[0], n[0]], [])
    with pytest.raises(ValueError):
        _dag([], [])


def test_dag_without_proper_source_or_sink_rejected():
    only_extract = [_node(0, extract(), DataType.PAGES, DataType.FACTS)]
    with pytest.raises(ValueError):
        _dag(only_extract, [])


def test_maf_compiler_delegates_to_the_pure_transformation():
    c = MAFCompiler()
    dag = c.to_dag(minimal_genome())
    assert dag.compiler_version == c.version
    assert dag == c.to_dag(minimal_genome())
    assert dag.nodes == compile_genome(minimal_genome(), version=c.version).nodes


def test_maf_build_requires_an_executor_for_every_node():
    c = MAFCompiler()
    with pytest.raises(CompileError):
        c.build(c.to_dag(minimal_genome()), {})


def test_dag_error_type_is_a_value_error():
    assert issubclass(DagError, ValueError)
