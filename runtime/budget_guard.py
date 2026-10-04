"""Deterministic budget enforcement. No LLM ever decides whether a cap was crossed.

Wrap every executor with ``runtime.executors.base.GuardedExecutor``; crossing any cap yields a
``BUDGET_EXCEEDED`` failure, which the evaluator maps to INFEASIBLE.
"""

from __future__ import annotations

from core.results import BudgetCap, BudgetUsage, usage_exceeds
from core.task_spec import Caps


class BudgetExceeded(RuntimeError):
    def __init__(self, caps_hit: tuple[BudgetCap, ...]) -> None:
        super().__init__("budget exceeded: " + ", ".join(c.value for c in caps_hit))
        self.caps_hit = caps_hit


class BudgetGuard:
    """Accumulates usage for ONE run and checks it against the task's caps."""

    def __init__(self, caps: Caps) -> None:
        self.caps = caps
        self._usage = BudgetUsage()

    @property
    def usage(self) -> BudgetUsage:
        return self._usage

    def charge(self, delta: BudgetUsage) -> tuple[BudgetCap, ...]:
        """Add ``delta`` to the running total; return the caps now exceeded (empty if none)."""
        self._usage = self._usage + delta
        return self.exceeded()

    def exceeded(self) -> tuple[BudgetCap, ...]:
        return usage_exceeds(self._usage, self.caps)

    def ensure_within(self) -> None:
        hit = self.exceeded()
        if hit:
            raise BudgetExceeded(hit)
