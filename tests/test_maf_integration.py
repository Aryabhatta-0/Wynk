"""Runs only where ``agent-framework`` is installed (pip install agent-framework).

Checks that the documented MAF builder calls in compiler/maf_compiler.py really accept a
compiled DAG. Uses toy MAF executors; real stage executors are Track B.
"""

import pytest

pytest.importorskip("agent_framework")

from agent_framework import Executor, WorkflowContext, handler  # noqa: E402

from compiler.maf_compiler import MAFCompiler  # noqa: E402
from core.genome import Genome  # noqa: E402
from tests.conftest import extract, gather, synth, verify  # noqa: E402


class _Pass(Executor):
    @handler
    async def run(self, message: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.send_message(message)
        await ctx.yield_output(message)


def test_compiled_dag_builds_into_a_maf_workflow():
    compiler = MAFCompiler()
    dag = compiler.to_dag(Genome.of(gather(), extract(), verify(), synth()))
    executors = {n.node_id: _Pass(id=n.node_id) for n in dag.nodes}
    workflow = compiler.build(dag, executors)
    assert workflow is not None
