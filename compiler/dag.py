"""Framework-neutral workflow DAG + the pure Genome -> DAG transformation.

The DAG is the intermediate representation between a genome and any execution framework
(MAF today, LangGraph possibly later). Compilation is pure: same genome + same compiler
version => byte-identical DAG. Nothing here imports MAF.

Failure strategies (retry-N / regather) are node attributes, not back-edges, so the graph is
always acyclic; the runtime owns re-execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import networkx as nx
from pydantic import BaseModel, ConfigDict, model_validator

from core.canonical import canonical_hash, canonical_json
from core.genome import Genome
from core.grammar import SIGNATURES, START_TYPE, DataType, Grammar
from core.stages import ALL_STAGE_KINDS, StageKind, StageSpec

COMPILER_VERSION = "compiler/1"


class DagError(ValueError):
    pass


class CompileError(ValueError):
    pass


class DagNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    stage_index: int
    stage: StageSpec
    input_type: DataType
    output_type: DataType


class DagEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    data_type: DataType


class WorkflowDAG(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    genome_hash: str
    compiler_version: str
    nodes: tuple[DagNode, ...]
    edges: tuple[DagEdge, ...]

    @model_validator(mode="after")
    def _valid(self) -> WorkflowDAG:
        self.validate_graph()
        return self

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.node_id, node=n)
        for e in self.edges:
            g.add_edge(e.source, e.target, data_type=e.data_type)
        return g

    def validate_graph(self) -> None:
        ids = [n.node_id for n in self.nodes]
        if not ids:
            raise DagError("DAG has no nodes")
        if len(set(ids)) != len(ids):
            raise DagError("duplicate node ids")
        by_id = {n.node_id: n for n in self.nodes}
        for e in self.edges:
            if e.source not in by_id or e.target not in by_id:
                raise DagError(f"edge {e.source}->{e.target} references an unknown node")
            if not (by_id[e.source].output_type == e.data_type == by_id[e.target].input_type):
                raise DagError(f"edge {e.source}->{e.target} has mismatched data types")
        g = self.to_networkx()
        if not nx.is_directed_acyclic_graph(g):
            raise DagError("workflow graph contains a cycle")
        if not nx.is_weakly_connected(g):
            raise DagError("workflow graph is not connected")
        sources = [n for n, d in g.in_degree() if d == 0]
        sinks = [n for n, d in g.out_degree() if d == 0]
        if len(sources) != 1 or by_id[sources[0]].input_type != START_TYPE:
            raise DagError("workflow must have exactly one source consuming the Task")
        if len(sinks) != 1 or by_id[sinks[0]].output_type != DataType.ANSWER:
            raise DagError("workflow must have exactly one sink producing the Answer")

    @property
    def start_node(self) -> str:
        return next(n for n, d in self.to_networkx().in_degree() if d == 0)

    @property
    def end_node(self) -> str:
        return next(n for n, d in self.to_networkx().out_degree() if d == 0)

    def topological_order(self) -> tuple[str, ...]:
        return tuple(nx.lexicographical_topological_sort(self.to_networkx()))

    def canonical_json(self) -> str:
        return canonical_json(self)

    @property
    def dag_hash(self) -> str:
        return canonical_hash(self)


def compile_genome(
    genome: Genome, grammar: Grammar | None = None, version: str = COMPILER_VERSION
) -> WorkflowDAG:
    """Pure, deterministic Genome -> WorkflowDAG. Rejects grammar-invalid/incomplete genomes.

    The default grammar admits every stage kind: compiling checks structure (typing, placement,
    dependencies, terminal stages). Whether a task supports a kind is admission's job
    (``ConstraintChecker``), which runs before compilation.
    """
    violations = (grammar or Grammar(ALL_STAGE_KINDS)).validate(genome, complete=True)
    if violations:
        raise CompileError(f"genome is not compilable: {violations[0].message}")
    nodes: list[DagNode] = []
    current = START_TYPE
    for i, stage in enumerate(genome.stages):
        out = SIGNATURES[StageKind(stage.kind)][current]
        nodes.append(
            DagNode(
                node_id=f"{i:02d}_{stage.kind.lower()}",
                stage_index=i,
                stage=stage,
                input_type=current,
                output_type=out,
            )
        )
        current = out
    edges = tuple(
        DagEdge(source=a.node_id, target=b.node_id, data_type=a.output_type)
        for a, b in zip(nodes, nodes[1:], strict=False)
    )
    return WorkflowDAG(
        genome_hash=genome.genome_hash,
        compiler_version=version,
        nodes=tuple(nodes),
        edges=edges,
    )


class Compiler(Protocol):
    """Framework boundary: Genome -> DAG (pure) -> framework-specific executable workflow."""

    version: str

    def to_dag(self, genome: Genome) -> WorkflowDAG: ...

    def build(self, dag: WorkflowDAG, executors: Mapping[str, Any]) -> Any:
        """``executors`` maps ``node_id`` to a framework-native executor built by runtime/."""
        ...
