"""Microsoft Agent Framework (MAF) adapter - the ONLY module that references MAF.

Boundary: genome -> ``compile_genome`` (pure, framework-neutral DAG) -> ``MAFCompiler.build``
(DAG + MAF-native executors -> MAF workflow). Replacing MAF later (e.g. LangGraph) means
writing another ``Compiler``; Genome, optimizers, evaluator and benchmarks do not change.

MAF API surface used (verified against the official docs, python tab):
  https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/builder-and-execution
    ``from agent_framework import WorkflowBuilder``
    ``WorkflowBuilder(start_executor=..., output_from=[...])``, ``.add_edge(a, b)``, ``.build()``
  https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/executors
    executors subclass ``agent_framework.Executor`` and use ``@handler``.
Package: ``pip install agent-framework`` (optional extra ``maf``).

NOT done here (Track B): writing the MAF ``Executor`` subclasses that wrap
``runtime.executors.base.StageExecutor`` implementations, and running a built workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compiler.dag import COMPILER_VERSION, CompileError, WorkflowDAG, compile_genome
from core.genome import Genome
from core.grammar import Grammar

MAF_COMPILER_VERSION = f"maf/{COMPILER_VERSION}"


class MAFCompiler:
    version = MAF_COMPILER_VERSION

    def __init__(self, grammar: Grammar | None = None) -> None:
        self._grammar = grammar or Grammar()

    def to_dag(self, genome: Genome) -> WorkflowDAG:
        return compile_genome(genome, self._grammar, version=self.version)

    def build(self, dag: WorkflowDAG, executors: Mapping[str, Any]) -> Any:
        """Build a MAF workflow. ``executors[node_id]`` must be a MAF ``Executor`` instance."""
        missing = [n.node_id for n in dag.nodes if n.node_id not in executors]
        if missing:
            raise CompileError(f"no executor supplied for nodes: {missing}")
        try:
            from agent_framework import WorkflowBuilder  # lazy: MAF stays optional
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "agent-framework is not installed (pip install agent-framework)"
            ) from exc
        builder = WorkflowBuilder(
            start_executor=executors[dag.start_node],
            output_from=[executors[dag.end_node]],
        )
        for edge in dag.edges:
            builder.add_edge(executors[edge.source], executors[edge.target])
        return builder.build()
