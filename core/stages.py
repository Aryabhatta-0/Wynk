"""Typed stage specifications (the building blocks of a genome).

One frozen model per stage kind, joined by a union discriminated on ``kind``. Option
values are the exact strings used by the architecture (``parallel-2``, ``retry-1``, ...).

Every option is an enum, so each kind has a finite, enumerable set of configurations
(``all_stage_specs``). There is deliberately no free-form / arbitrary-code stage.
"""

from __future__ import annotations

import itertools
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StageKind(StrEnum):
    # Definition order is the grammar's deterministic successor order: append, never reorder.
    GATHER = "GATHER"
    FILTER = "FILTER"
    EXTRACT = "EXTRACT"
    REASON = "REASON"
    VERIFY = "VERIFY"
    SYNTHESIZE = "SYNTHESIZE"
    DIRECT = "DIRECT"
    CONFIDENCE_GATE = "CONFIDENCE_GATE"


# The frozen benchmark's vocabulary (grammar/1). The default grammar admits only these kinds.
LEGACY_STAGE_KINDS: tuple[StageKind, ...] = (
    StageKind.GATHER,
    StageKind.FILTER,
    StageKind.EXTRACT,
    StageKind.REASON,
    StageKind.VERIFY,
    StageKind.SYNTHESIZE,
)
ALL_STAGE_KINDS: tuple[StageKind, ...] = tuple(StageKind)


class GatherSource(StrEnum):
    FETCH = "fetch"
    API = "api"
    JEV = "jev"


class GatherMode(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL_2 = "parallel-2"
    PARALLEL_4 = "parallel-4"


class FilterMethod(StrEnum):
    KEYWORD_CHUNK = "keyword_chunk"
    SECTION_SELECT = "section_select"


class ExtractMethod(StrEnum):
    DIRECT = "direct"
    SCHEMA_GUIDED = "schema_guided"
    COT = "cot"


class ReasonMethod(StrEnum):
    SINGLE = "single"
    DECOMPOSE = "decompose"


class VerifyMethod(StrEnum):
    SCHEMA_CHECK = "schema_check"
    EVIDENCE_SPAN = "evidence_span"
    SELF_CONSISTENCY = "self_consistency"


class FailureStrategy(StrEnum):
    RETRY_1 = "retry-1"
    RETRY_2 = "retry-2"
    REGATHER = "regather"


class SynthesizeMethod(StrEnum):
    DIRECT = "direct"
    CITE_EVIDENCE = "cite_evidence"


class DirectMethod(StrEnum):
    ANSWER = "answer"
    COT = "cot"


class SupportThreshold(StrEnum):
    """Minimum fraction of answer fields that must be backed by verified evidence."""

    HALF = "support-50"
    ALL = "support-100"


SUPPORT_FRACTION: dict[SupportThreshold, float] = {
    SupportThreshold.HALF: 0.5,
    SupportThreshold.ALL: 1.0,
}


class _Stage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GatherStage(_Stage):
    """Task -> Pages. Required."""

    kind: Literal["GATHER"] = "GATHER"
    source: GatherSource
    mode: GatherMode


class FilterStage(_Stage):
    """Pages -> Pages. Optional."""

    kind: Literal["FILTER"] = "FILTER"
    method: FilterMethod


class ExtractStage(_Stage):
    """Pages -> Facts. Required."""

    kind: Literal["EXTRACT"] = "EXTRACT"
    method: ExtractMethod


class ReasonStage(_Stage):
    """Facts -> Facts. Optional."""

    kind: Literal["REASON"] = "REASON"
    method: ReasonMethod


class VerifyStage(_Stage):
    """Facts -> Facts or Answer -> Answer (decided by position). Optional.

    Runtime VERIFY never sees ground truth; it only checks runtime outputs/evidence.
    """

    kind: Literal["VERIFY"] = "VERIFY"
    method: VerifyMethod
    on_failure: FailureStrategy


class SynthesizeStage(_Stage):
    """Facts -> Answer. Required."""

    kind: Literal["SYNTHESIZE"] = "SYNTHESIZE"
    method: SynthesizeMethod


class DirectStage(_Stage):
    """Task -> Answer in one model call, without retrieval. Produces no evidence."""

    kind: Literal["DIRECT"] = "DIRECT"
    method: DirectMethod


class ConfidenceGateStage(_Stage):
    """Answer -> Answer. Terminal: passes the answer only if at least ``min_support`` of its
    fields are backed by evidence that verifies against the gathered pages (deterministic, no
    model, no ground truth); otherwise the run abstains."""

    kind: Literal["CONFIDENCE_GATE"] = "CONFIDENCE_GATE"
    min_support: SupportThreshold


StageSpec = Annotated[
    GatherStage
    | FilterStage
    | ExtractStage
    | ReasonStage
    | VerifyStage
    | SynthesizeStage
    | DirectStage
    | ConfidenceGateStage,
    Field(discriminator="kind"),
]

_SPEC_CLASSES: dict[StageKind, type[_Stage]] = {
    StageKind.GATHER: GatherStage,
    StageKind.FILTER: FilterStage,
    StageKind.EXTRACT: ExtractStage,
    StageKind.REASON: ReasonStage,
    StageKind.VERIFY: VerifyStage,
    StageKind.SYNTHESIZE: SynthesizeStage,
    StageKind.DIRECT: DirectStage,
    StageKind.CONFIDENCE_GATE: ConfidenceGateStage,
}


def all_stage_specs(kind: StageKind) -> tuple[_Stage, ...]:
    """Every concrete configuration of ``kind``, in deterministic (enum-definition) order."""
    cls = _SPEC_CLASSES[kind]
    names = [n for n in cls.model_fields if n != "kind"]
    domains = [list(_enum_type(cls, n)) for n in names]
    return tuple(
        cls(**dict(zip(names, combo, strict=True))) for combo in itertools.product(*domains)
    )


def _enum_type(cls: type[_Stage], field: str) -> type[StrEnum]:
    return cls.model_fields[field].annotation
