"""Experiment-level budget: what ONE strategy run (one strategy x one seed) may spend in total.

This is separate from ``runtime.budget_guard.BudgetGuard``, which caps a single workflow run with
the contract's per-run caps. The experiment budget is the comparison's fairness contract: every
strategy of an experiment gets an identical ``ExperimentBudget`` and its own ``BudgetLedger``,
and the ledger applies the same rules to all of them.

Unit. A *candidate evaluation* is one proposed workflow run on every optimization row x trials
(optimizer feedback) and every validation row x trials (selection). It is the only unit the
comparison counts in - never "ACO epochs" or "random samples".

Rules (deterministic, no strategy-specific branch):

  * before a candidate: it may start only if fewer than ``max_candidate_evaluations`` have been
    admitted AND every resource cap still has headroom (used < cap) - both provable up front;
  * before every workflow run: every resource cap must still have headroom;
  * after every run: the MEASURED usage of that run (``ExecutionResult.metrics`` /
    ``budget_usage``, priced by the pricing policy when there is one) is charged - never an
    optimizer or cost-model estimate. A run that pushes any counter strictly above its cap aborts
    the in-flight candidate: the spend is real and stays on the ledger, but the candidate is
    never shown to the optimizer, never eligible for selection and never on the learning curve,
    and the strategy stops. What a run will cost is only known after it ran, so the overshoot
    is bounded by one workflow run and no strategy can profit from it.

Monetary cost. The runtime has no authoritative price for anything, so this module never invents
one. A ``PricingPolicy`` - supplied by the caller, provider-neutral, part of the experiment
identity - converts measured usage into cost. Without one, cost is ``None`` (unknown, not zero)
and asking for ``max_cost`` fails closed before the experiment starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from core.canonical import canonical_hash
from core.results import EvaluatedRun

LEDGER_VERSION = "experiment-ledger/1"


class ExperimentBudgetError(ValueError):
    """The experiment budget cannot be enforced as requested (raised before any run)."""


class PricingError(ValueError):
    """The pricing policy could not price a measured run: fail closed, never charge zero."""


class ExperimentBudget(BaseModel):
    """Shared caps for one strategy run. ``None`` = not capped (still measured and reported)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_candidate_evaluations: PositiveInt
    max_model_calls: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    max_wall_time_s: PositiveFloat | None = None
    max_cost: PositiveFloat | None = None  # needs a PricingPolicy

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class MeasuredUsage:
    """What one workflow run measurably consumed, as reported by the runtime."""

    model_hash: str
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    tokens: int
    tool_calls: int
    retries: int
    wall_time_s: float

    @classmethod
    def of(cls, run: EvaluatedRun) -> MeasuredUsage:
        ex = run.execution
        return cls(
            model_hash=ex.key.versions.model_hash,
            model_calls=ex.metrics.model_calls,
            prompt_tokens=ex.metrics.prompt_tokens,
            completion_tokens=ex.metrics.completion_tokens,
            tokens=ex.budget_usage.tokens,
            tool_calls=ex.budget_usage.tool_calls,
            retries=ex.budget_usage.retries,
            wall_time_s=ex.budget_usage.wall_time_s,
        )


class PricingPolicy(Protocol):
    """Provider-neutral conversion of measured usage into cost (any currency unit).

    ``identity`` names the exact price table/version; it becomes part of the experiment identity,
    so results priced differently are never mistaken for each other. ``cost`` must be a pure
    function of the usage and must raise (e.g. ``PricingError``) for anything it cannot price.
    """

    @property
    def identity(self) -> str: ...

    def cost(self, usage: MeasuredUsage) -> float: ...


class Resource(StrEnum):
    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"
    WALL_TIME = "wall_time_s"
    COST = "cost"


class StopReason(StrEnum):
    CANDIDATE_EVALUATIONS = "candidate_evaluations"  # the shared candidate cap was reached
    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"
    WALL_TIME = "wall_time_s"
    COST = "cost"
    STRATEGY_EXHAUSTED = "strategy_exhausted"  # the strategy had nothing more to propose


_RESOURCE_STOP = {
    Resource.MODEL_CALLS: StopReason.MODEL_CALLS,
    Resource.TOKENS: StopReason.TOKENS,
    Resource.WALL_TIME: StopReason.WALL_TIME,
    Resource.COST: StopReason.COST,
}


def check_budget(budget: ExperimentBudget, pricing: PricingPolicy | None) -> None:
    """Fail closed before an experiment whose budget cannot be enforced."""
    if budget.max_cost is not None and pricing is None:
        raise ExperimentBudgetError(
            "max_cost needs a pricing policy: the runtime measures no monetary cost, and an "
            "unpriced cost cap could never be enforced"
        )
    if pricing is not None and not str(getattr(pricing, "identity", "") or ""):
        raise ExperimentBudgetError("a pricing policy must declare a non-empty identity")


class BudgetLedger:
    """Running totals of one strategy run against the shared ``ExperimentBudget``."""

    def __init__(self, budget: ExperimentBudget, pricing: PricingPolicy | None = None) -> None:
        check_budget(budget, pricing)
        self.budget = budget
        self.pricing = pricing
        self.candidate_evaluations = 0  # admitted (complete, within budget)
        self.aborted_candidates = 0
        self.runs = 0
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokens = 0
        self.tool_calls = 0
        self.retries = 0
        self.wall_time_s = 0.0
        self.cost: float | None = 0.0 if pricing is not None else None

    # -- caps ---------------------------------------------------------------------------------
    def _caps(self) -> dict[Resource, float | None]:
        b = self.budget
        return {
            Resource.MODEL_CALLS: b.max_model_calls,
            Resource.TOKENS: b.max_tokens,
            Resource.WALL_TIME: b.max_wall_time_s,
            Resource.COST: b.max_cost,
        }

    def _used(self, resource: Resource) -> float:
        if resource is Resource.COST:
            return self.cost or 0.0
        return getattr(self, resource.value)

    def exhausted(self) -> StopReason | None:
        """First resource whose cap leaves no headroom (used >= cap), if any."""
        for resource, cap in self._caps().items():
            if cap is not None and self._used(resource) >= cap:
                return _RESOURCE_STOP[resource]
        return None

    def exceeded(self) -> tuple[Resource, ...]:
        """Resources strictly above their cap (equal is allowed, as for per-run caps)."""
        return tuple(
            r for r, cap in self._caps().items() if cap is not None and self._used(r) > cap
        )

    def candidate_stop(self) -> StopReason | None:
        """Why no further candidate may start (``None``: it may)."""
        if self.candidate_evaluations >= self.budget.max_candidate_evaluations:
            return StopReason.CANDIDATE_EVALUATIONS
        return self.exhausted()

    def run_stop(self) -> StopReason | None:
        """Why no further workflow run may start (``None``: it may)."""
        return self.exhausted()

    # -- charging -----------------------------------------------------------------------------
    def price(self, usage: MeasuredUsage) -> float | None:
        if self.pricing is None:
            return None
        try:
            cost = float(self.pricing.cost(usage))
        except PricingError:
            raise
        except Exception as exc:  # a pricing policy that breaks is a pricing failure
            raise PricingError(f"pricing policy {self.pricing.identity!r} failed: {exc}") from exc
        if not math.isfinite(cost) or cost < 0:
            raise PricingError(f"pricing policy {self.pricing.identity!r} returned {cost!r}")
        return cost

    def charge(self, run: EvaluatedRun) -> dict[str, Any]:
        """Charge one MEASURED run; returns the charged amounts (the run's ledger entry)."""
        usage = MeasuredUsage.of(run)
        cost = self.price(usage)
        self.runs += 1
        self.model_calls += usage.model_calls
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.tokens += usage.tokens
        self.tool_calls += usage.tool_calls
        self.retries += usage.retries
        self.wall_time_s += usage.wall_time_s
        if cost is not None:
            self.cost = (self.cost or 0.0) + cost
        return {
            "model_calls": usage.model_calls,
            "tokens": usage.tokens,
            "wall_time_s": usage.wall_time_s,
            "cost": cost,
        }

    # -- reporting ----------------------------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        return {
            "candidate_evaluations": self.candidate_evaluations,
            "aborted_candidates": self.aborted_candidates,
            "workflow_runs": self.runs,
            "model_calls": self.model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "wall_time_s": self.wall_time_s,
            "cost": self.cost,
        }
