"""MAF adapter for the runtime: one thin MAF ``Executor`` per DAG node.

Deliberately NOT using ``from __future__ import annotations``: MAF reads the handler's type
annotations at class-definition time.

API used (official docs, python tab):
  https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/executors
    ``Executor`` subclass + ``@handler`` + ``WorkflowContext[SendType, OutputType]``,
    ``ctx.send_message(...)``, ``ctx.yield_output(...)``.
Nodes carry no logic: they hand the payload to ``StageRunner`` (framework-neutral) and forward
the result. A failed stage sends nothing, so downstream nodes simply never run.
"""

from dataclasses import dataclass
from typing import Any

from agent_framework import Executor, WorkflowContext, handler

from compiler.dag import DagNode
from runtime.stage_runner import StageRunner


@dataclass(frozen=True)
class Envelope:
    """Message passed along the MAF graph; ``payload`` is a core payload (Task/Pages/...)."""

    payload: Any


class StageNode(Executor):
    def __init__(self, node: DagNode, runner: StageRunner, is_end: bool) -> None:
        super().__init__(id=node.node_id)
        self._dag_node = node
        self._stage_runner = runner
        self._is_end = is_end

    @handler
    async def handle(self, message: Envelope, ctx: WorkflowContext[Envelope, Envelope]) -> None:
        out = await self._stage_runner.execute_node(self._dag_node, message.payload)
        if out is None:
            return
        if self._is_end:
            await ctx.yield_output(Envelope(out))
        else:
            await ctx.send_message(Envelope(out))
