"""Shared violation vocabulary for the grammar and the constraint layer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ViolationCode(StrEnum):
    # grammar (typing / structure)
    INVALID_TRANSITION = "invalid_transition"
    PLACEMENT = "placement"
    MISSING_REQUIRED_STAGE = "missing_required_stage"
    NO_ANSWER_TERMINAL = "no_answer_terminal"
    STAGE_UNSUPPORTED = "stage_unsupported"  # kind not in this grammar's (task's) vocabulary
    UNSATISFIED_DEPENDENCY = "unsatisfied_dependency"  # needs an upstream stage that is absent
    AFTER_TERMINAL = "after_terminal"  # a stage (incl. a second gate) after a terminal stage
    # hard constraints
    JEV_PARALLEL_4 = "jev_parallel_4"
    TOO_MANY_VERIFIERS = "too_many_verifiers"
    SELF_CONSISTENCY_REPEATED = "self_consistency_repeated"
    JEV_REGATHER = "jev_regather"
    # task-aware constraints
    SOURCE_NOT_ALLOWED = "source_not_allowed"
    INTERACTION_REQUIRES_JEV = "interaction_requires_jev"
    BUDGET_INFEASIBLE = "budget_infeasible"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    # measured hard limits (core.constraints.check_limits)
    LIMIT_VIOLATED = "limit_violated"
    METRIC_MISSING = "metric_missing"


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ViolationCode
    message: str
    stage_index: int | None = None
