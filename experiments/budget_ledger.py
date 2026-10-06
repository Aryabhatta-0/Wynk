"""Experiment-level budget: what ONE strategy run (one strategy x one seed) may spend in total.

This is separate from ``runtime.budget_guard.BudgetGuard``, which caps a single workflow run with
the contract's per-run caps. The experiment budget is the comparison's fairness contract: every
strategy of an experiment gets an identical ``ExperimentBudget`` and its own ``BudgetLedger``,
and the ledger applies the same rules to all of them.

Unit. A *candidate evaluation* is one proposed workflow run on every optimization row x trials
(optimizer feedback) and every validation row x trials (selection). It is the only unit the
comparison counts in - never "ACO epochs" or "random samples".

Hard caps by reservation (deterministic, no strategy-specific branch):

  1. reserve   before a candidate starts, the ledger reserves its COMPLETE worst case:
               ``runs x per-run reservation``, where the per-run reservation comes from the
               authoritative ``TaskContract`` limits - ``maximum_model_calls``,
               ``maximum_tokens_per_example``, ``maximum_wall_time_s`` - and, for cost, from the
               pricing policy's upper-bound quote for exactly those limits. Never from an
               optimizer or cost-model estimate. If ``committed + reservation`` would exceed any
               cap, or the candidate cap is reached, the candidate does not start: ZERO rows run.
  2. execute   the candidate's runs execute; every run's MEASURED usage is recorded.
  3. settle    the measured total is committed and the unused reservation released. A measured
               total above its reservation means the authoritative limits did not hold: the
               ledger raises ``ReservationOverflow`` and the experiment fails closed.

So ``committed <= cap`` holds after every settlement, for model calls, tokens, execution time
and cost - final usage never exceeds a configured cap. A cap without an authoritative limit to
reserve against (e.g. ``max_model_calls`` while the contract sets no ``maximum_model_calls``) is
refused before the experiment starts.

Monetary cost. The runtime has no authoritative price for anything, so this module never invents
one. A ``PricingPolicy`` - supplied by the caller, provider-neutral, part of the experiment
identity - converts measured usage into cost and quotes a safe upper bound for a reservation.
Without one, cost is ``None`` (unknown, not zero) and asking for ``max_cost`` fails closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from core.canonical import canonical_hash
from core.results import EvaluatedRun
from core.task_contract import TaskContract

LEDGER_VERSION = "experiment-ledger/2"


class ExperimentBudgetError(ValueError):
    """The experiment budget cannot be enforced as requested (raised before any run)."""


class PricingError(ValueError):
    """The pricing policy could not price a run or quote a bound: fail closed, never 0."""


class ReservationOverflow(RuntimeError):
    """Measured usage exceeded its authoritative reservation: the experiment fails closed."""


class ExperimentBudget(BaseModel):
    """Shared caps for one strategy run. ``None`` = not capped (still measured and reported).

    ``max_wall_time_s`` caps EXECUTION time: the sum of the runtime-measured wall time of every
    workflow run (what ``maximum_wall_time_s`` bounds per run). End-to-end wall-clock time
    depends on concurrency and the machine; it is measured and reported, never a cap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_candidate_evaluations: PositiveInt
    max_model_calls: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    max_wall_time_s: PositiveFloat | None = None
    max_cost: PositiveFloat | None = None  # needs a PricingPolicy with an upper-bound quote

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

    ``identity`` names the exact price table/version; it becomes part of the experiment identity.
    ``cost`` prices one measured run. ``max_cost`` quotes a SAFE UPPER BOUND for any run of
    ``model_hash`` that uses at most ``model_calls`` calls and ``tokens`` tokens (however they
    split between prompt and completion). Both are pure and must raise for anything they cannot
    price.
    """

    @property
    def identity(self) -> str: ...

    def cost(self, usage: MeasuredUsage) -> float: ...

    def max_cost(self, model_hash: str, model_calls: int, tokens: int) -> float: ...


class Resource(StrEnum):
    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"
    WALL_TIME = "wall_time_s"
    COST = "cost"


class StopReason(StrEnum):
    CANDIDATE_EVALUATIONS = "candidate_evaluations"  # the shared candidate cap was reached
    MODEL_CALLS = "model_calls"  # the next complete candidate could not be reserved
    TOKENS = "tokens"
    WALL_TIME = "wall_time_s"
    COST = "cost"
    STRATEGY_EXHAUSTED = "strategy_exhausted"  # the strategy had nothing more to propose


RESOURCE_STOP = {
    Resource.MODEL_CALLS: StopReason.MODEL_CALLS,
    Resource.TOKENS: StopReason.TOKENS,
    Resource.WALL_TIME: StopReason.WALL_TIME,
    Resource.COST: StopReason.COST,
}


@dataclass(frozen=True)
class Reservation:
    """Upper bounds per resource (``None``: that resource is not capped, nothing reserved)."""

    model_calls: float | None = None
    tokens: float | None = None
    wall_time_s: float | None = None
    cost: float | None = None

    def get(self, resource: Resource) -> float | None:
        return getattr(self, resource.value)

    def times(self, n: int) -> Reservation:
        return Reservation(
            **{r.value: (None if (v := self.get(r)) is None else v * n) for r in Resource}
        )

    def as_dict(self) -> dict[str, float | None]:
        return {r.value: self.get(r) for r in Resource}


def run_reservation(
    budget: ExperimentBudget,
    contract: TaskContract,
    pricing: PricingPolicy | None,
    model_hash: str,
) -> Reservation:
    """The authoritative worst case of ONE workflow run, for every capped resource.

    Raises ``ExperimentBudgetError`` before anything runs if a capped resource has no
    authoritative per-run limit to reserve against.
    """
    check_budget(budget, pricing)
    limits = contract.constraints

    def need(cap: float | None, limit: float | None, name: str) -> float | None:
        if cap is None:
            return None
        if limit is None:
            raise ExperimentBudgetError(
                f"the experiment caps {name} but contract {contract.task_id} sets no per-run "
                f"limit to reserve against"
            )
        return float(limit)

    calls = need(budget.max_model_calls, limits.maximum_model_calls, "model calls")
    tokens = need(budget.max_tokens, limits.maximum_tokens_per_example, "tokens")
    wall = need(budget.max_wall_time_s, limits.maximum_wall_time_s, "execution time")
    cost = None
    if budget.max_cost is not None:
        assert pricing is not None  # check_budget
        if limits.maximum_model_calls is None or limits.maximum_tokens_per_example is None:
            raise ExperimentBudgetError(
                "a cost cap needs maximum_model_calls and maximum_tokens_per_example in the "
                "contract: the pricing quote must bound a run that uses at most those"
            )
        quote = getattr(pricing, "max_cost", None)
        if quote is None:
            raise ExperimentBudgetError(
                f"pricing policy {pricing.identity!r} gives no upper-bound quote (max_cost), so "
                "a cost cap cannot be reserved"
            )
        try:
            cost = float(
                quote(model_hash, limits.maximum_model_calls, limits.maximum_tokens_per_example)
            )
        except Exception as exc:
            raise ExperimentBudgetError(f"pricing policy cannot quote a run: {exc}") from exc
        if not math.isfinite(cost) or cost < 0:
            raise ExperimentBudgetError(f"pricing policy quoted {cost!r}")
    return Reservation(model_calls=calls, tokens=tokens, wall_time_s=wall, cost=cost)


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
    """Committed (measured) totals of one strategy run against the shared budget."""

    def __init__(
        self,
        budget: ExperimentBudget,
        pricing: PricingPolicy | None = None,
        per_run: Reservation | None = None,
    ) -> None:
        check_budget(budget, pricing)
        self.budget = budget
        self.pricing = pricing
        self.per_run = per_run or Reservation()
        for r in Resource:
            if self._cap(r) is not None and self.per_run.get(r) is None:
                raise ExperimentBudgetError(f"no per-run reservation for capped {r.value}")
        self.candidate_evaluations = 0
        self.runs = 0
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokens = 0
        self.tool_calls = 0
        self.retries = 0
        self.wall_time_s = 0.0
        self.cost: float | None = 0.0 if pricing is not None else None
        self.reserved: Reservation | None = None  # the open reservation, if a candidate runs
        self.peak_reserved: dict[str, float | None] = {}

    # -- caps ---------------------------------------------------------------------------------
    def _cap(self, resource: Resource) -> float | None:
        b = self.budget
        return {
            Resource.MODEL_CALLS: b.max_model_calls,
            Resource.TOKENS: b.max_tokens,
            Resource.WALL_TIME: b.max_wall_time_s,
            Resource.COST: b.max_cost,
        }[resource]

    def _committed(self, resource: Resource) -> float:
        if resource is Resource.COST:
            return self.cost or 0.0
        return getattr(self, resource.value)

    # -- reserve / settle ---------------------------------------------------------------------
    def can_reserve(self, runs: int) -> StopReason | None:
        """Why a complete candidate of ``runs`` workflow runs could not start (``None``: it can)."""
        if self.candidate_evaluations >= self.budget.max_candidate_evaluations:
            return StopReason.CANDIDATE_EVALUATIONS
        want = self.per_run.times(runs)
        for r in Resource:
            cap, need = self._cap(r), want.get(r)
            if cap is not None and need is not None and self._committed(r) + need > cap:
                return RESOURCE_STOP[r]
        return None

    def reserve(self, runs: int) -> StopReason | None:
        """Reserve a complete candidate of ``runs`` workflow runs, or say why it cannot start."""
        if self.reserved is not None:
            raise RuntimeError("a candidate is already reserved")
        stop = self.can_reserve(runs)
        if stop is None:
            self.reserved = self.per_run.times(runs)
        return stop

    def slack(self, spent: dict[str, float]) -> bool:
        """True if one more run still fits inside the open reservation (used for retries)."""
        assert self.reserved is not None
        for r in Resource:
            need, extra = self.reserved.get(r), self.per_run.get(r)
            if need is not None and extra is not None and spent.get(r.value, 0.0) + extra > need:
                return False
        return True

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

    def measure(self, run: EvaluatedRun) -> dict[str, Any]:
        """The run's measured, priced usage (its ledger entry); nothing is committed yet."""
        usage = MeasuredUsage.of(run)
        return {
            "model_calls": usage.model_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tokens": usage.tokens,
            "tool_calls": usage.tool_calls,
            "retries": usage.retries,
            "wall_time_s": usage.wall_time_s,
            "cost": self.price(usage),
        }

    def settle(self, entries: list[dict[str, Any]], *, admitted: bool) -> None:
        """Commit the measured usage of the reserved candidate and release the reservation.
        Raises ``ReservationOverflow`` if the measured total exceeds what was reserved."""
        reserved, self.reserved = self.reserved, None
        if reserved is None:
            raise RuntimeError("settle without a reservation")
        spent = {r.value: sum((e[r.value] or 0.0) for e in entries) for r in Resource}
        for r in Resource:
            need = reserved.get(r)
            if need is not None and spent[r.value] > need:
                raise ReservationOverflow(
                    f"measured {r.value} {spent[r.value]} exceeded its authoritative reservation "
                    f"{need}: the contract's per-run limits did not hold"
                )
        for key in ("model_calls", "prompt_tokens", "completion_tokens", "tokens"):
            setattr(self, key, getattr(self, key) + sum(e[key] for e in entries))
        self.tool_calls += sum(e["tool_calls"] for e in entries)
        self.retries += sum(e["retries"] for e in entries)
        self.wall_time_s += spent[Resource.WALL_TIME.value]
        if self.cost is not None:
            self.cost += spent[Resource.COST.value]
        self.runs += len(entries)
        if admitted:
            self.candidate_evaluations += 1
        for r in Resource:
            cap = self._cap(r)
            if cap is not None and self._committed(r) > cap:  # unreachable unless overflow
                raise ReservationOverflow(f"committed {r.value} exceeds its cap")

    # -- reporting ----------------------------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        return {
            "candidate_evaluations": self.candidate_evaluations,
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
