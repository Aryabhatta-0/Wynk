"""Deterministic pre-execution cost estimates.

Contract: ``CostModel.estimate(genome, contract)`` returns a *best-case lower bound* for tokens,
latency and tool calls (no retries happen), so ``estimate > cap`` PROVES the cap cannot be
met. For a partial genome the estimate also adds the cheapest completion of the still-missing
producer stages, so a prefix that can no longer fit is rejected early. An empty prefix may be
completed by either producer chain (GATHER -> EXTRACT -> SYNTHESIZE, or DIRECT when the contract's
``workflow.stages`` include it), so it takes the cheaper of the two per dimension. Retries are
reported
separately (``max_retries``, ``retry_risk``) and are never used to reject.

``StaticCostModel`` ships UNCALIBRATED placeholder numbers. They are configuration
(``CostTable``), meant to be replaced by values fitted from the run store later. These defaults
are advisory only; hard rejection requires a table explicitly marked ``proven_lower_bound``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from core.genome import Genome
from core.grammar import completions
from core.results import BudgetCap
from core.stages import GatherMode, StageKind, StageSpec, all_stage_specs
from core.task_spec import Caps

if TYPE_CHECKING:  # core.task_contract -> core.constraints -> this module
    from core.task_contract import TaskContract


class StageCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: NonNegativeInt = 0
    latency_s: NonNegativeFloat = 0.0
    tool_calls: NonNegativeInt = 0
    retry_risk: float = Field(default=0.0, ge=0.0, le=1.0)  # P(needs a retry), VERIFY only


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: NonNegativeInt
    latency_s: NonNegativeFloat
    tool_calls: NonNegativeInt
    max_retries: NonNegativeInt = 0
    retry_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    proven_lower_bound: bool = True


class CostModel(Protocol):
    version: str

    def estimate(self, genome: Genome, contract: TaskContract) -> CostEstimate: ...


def exceeded_caps(estimate: CostEstimate, caps: Caps) -> tuple[BudgetCap, ...]:
    """Caps provably exceeded by the best-case estimate. Retries are never provable here."""
    if not estimate.proven_lower_bound:
        return ()
    out: list[BudgetCap] = []
    if estimate.tokens > caps.tokens:
        out.append(BudgetCap.TOKENS)
    if estimate.latency_s > caps.wall_time_s:
        out.append(BudgetCap.WALL_TIME)
    if estimate.tool_calls > caps.tool_calls:
        out.append(BudgetCap.TOOL_CALLS)
    return tuple(out)


class CostTable(BaseModel):
    """Placeholder, uncalibrated. Key = ``KIND:option`` (GATHER uses the source)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "cost-table/0-uncalibrated"
    proven_lower_bound: bool = False
    stages: dict[str, StageCost] = Field(
        default_factory=lambda: {
            "GATHER:fetch": StageCost(latency_s=2.0, tool_calls=3),
            "GATHER:api": StageCost(latency_s=0.5, tool_calls=2),
            "GATHER:jev": StageCost(latency_s=20.0, tool_calls=10),
            "FILTER:keyword_chunk": StageCost(latency_s=0.05),
            "FILTER:section_select": StageCost(latency_s=0.05),
            "EXTRACT:direct": StageCost(tokens=1500, latency_s=3.0),
            "EXTRACT:schema_guided": StageCost(tokens=2000, latency_s=4.0),
            "EXTRACT:cot": StageCost(tokens=3500, latency_s=8.0),
            "REASON:single": StageCost(tokens=1000, latency_s=2.0),
            "REASON:decompose": StageCost(tokens=3000, latency_s=6.0),
            "VERIFY:schema_check": StageCost(latency_s=0.01, retry_risk=0.10),
            "VERIFY:evidence_span": StageCost(latency_s=0.05, retry_risk=0.15),
            "VERIFY:self_consistency": StageCost(tokens=4500, latency_s=9.0, retry_risk=0.20),
            "SYNTHESIZE:direct": StageCost(tokens=800, latency_s=2.0),
            "SYNTHESIZE:cite_evidence": StageCost(tokens=1200, latency_s=3.0),
            "DIRECT:answer": StageCost(tokens=800, latency_s=2.0),
            "DIRECT:cot": StageCost(tokens=1500, latency_s=4.0),
            "CONFIDENCE_GATE:support-50": StageCost(latency_s=0.01),
            "CONFIDENCE_GATE:support-100": StageCost(latency_s=0.01),
        }
    )
    gather_mode_latency_factor: dict[GatherMode, float] = Field(
        default_factory=lambda: {
            GatherMode.SEQUENTIAL: 1.0,
            GatherMode.PARALLEL_2: 0.5,
            GatherMode.PARALLEL_4: 0.25,
        }
    )


_RETRIES = {"retry-1": 1, "retry-2": 2, "regather": 1}


def _key(spec: StageSpec) -> str:
    if spec.kind == "GATHER":
        option = spec.source
    elif spec.kind == "CONFIDENCE_GATE":
        option = spec.min_support
    else:
        option = spec.method
    return f"{spec.kind}:{option.value}"


class StaticCostModel:
    def __init__(self, table: CostTable | None = None) -> None:
        self.table = table or CostTable()
        self.version = self.table.version

    def _stage_cost(self, spec: StageSpec) -> StageCost:
        cost = self.table.stages[_key(spec)]
        if spec.kind == "GATHER":
            factor = self.table.gather_mode_latency_factor[spec.mode]
            cost = cost.model_copy(update={"latency_s": cost.latency_s * factor})
        return cost

    def _cheapest(self, kinds: tuple[StageKind, ...]) -> tuple[int, float, int]:
        """Per-dimension minimum over all options of each kind: a lower bound on adding them."""
        tokens, latency, calls = 0, 0.0, 0
        for kind in kinds:
            options = [self._stage_cost(s) for s in all_stage_specs(kind)]
            tokens += min(c.tokens for c in options)
            latency += min(c.latency_s for c in options)
            calls += min(c.tool_calls for c in options)
        return tokens, latency, calls

    def estimate(self, genome: Genome, contract: TaskContract) -> CostEstimate:
        tokens, latency, calls = 0, 0.0, 0
        max_retries, no_retry = 0, 1.0
        for spec in genome.stages:
            c = self._stage_cost(spec)
            tokens += c.tokens
            latency += c.latency_s
            calls += c.tool_calls
            if spec.kind == "VERIFY":
                max_retries += _RETRIES[spec.on_failure.value]
                no_retry *= 1.0 - c.retry_risk
        # Per-dimension minimum over every completion the contract's vocabulary allows: still a
        # valid lower bound whichever completion is taken.
        rest = [self._cheapest(p) for p in completions(genome.stages, contract.workflow.stages)]
        tokens += min(c[0] for c in rest)
        latency += min(c[1] for c in rest)
        calls += min(c[2] for c in rest)
        return CostEstimate(
            tokens=tokens,
            latency_s=latency,
            tool_calls=calls,
            max_retries=max_retries,
            retry_risk=1.0 - no_retry,
            proven_lower_bound=self.table.proven_lower_bound,
        )
