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
    # hard constraints
    JEV_PARALLEL_4 = "jev_parallel_4"
    TOO_MANY_VERIFIERS = "too_many_verifiers"
    SELF_CONSISTENCY_REPEATED = "self_consistency_repeated"
    JEV_REGATHER = "jev_regather"
    # task-aware constraints
    SOURCE_NOT_ALLOWED = "source_not_allowed"
    INTERACTION_REQUIRES_JEV = "interaction_requires_jev"
    BUDGET_INFEASIBLE = "budget_infeasible"


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ViolationCode
    message: str
    stage_index: int | None = None
