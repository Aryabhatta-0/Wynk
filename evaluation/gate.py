"""Evaluator boundary (implementation: Track A).

The offline evaluator is the ONLY authority for PASS / FAIL / INFEASIBLE and search fitness.
It is deterministic and must never call an LLM. It is the only place (with benchmarks/ and
store/) that may hold a ``TaskSpec`` and therefore ground truth.
"""

from __future__ import annotations

from typing import Protocol

from core.results import Evaluation, ExecutionResult
from core.task_spec import TaskSpec


class Evaluator(Protocol):
    version: str

    def evaluate(self, task: TaskSpec, result: ExecutionResult) -> Evaluation:
        """Pure function of (task, result): same inputs -> same Evaluation.

        Rules fixed by the architecture contract:
          * a ``BUDGET_EXCEEDED`` failure (or usage above caps) => INFEASIBLE;
          * otherwise PASS iff every field matches ground truth under its matcher.
        """
        ...
