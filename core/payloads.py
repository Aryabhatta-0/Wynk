"""Data flowing between stages: Pages, Facts, Answer (the task is ``ExecutionTask``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from core.canonical import sha256_hex
from core.evidence import EvidenceSpan, FieldEvidence

if TYPE_CHECKING:  # core.run_contract imports core.results, which imports this module
    from core.run_contract import ExecutionTask


class Page(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1)
    source_ref: str  # URL / API endpoint / snapshot path - provenance only
    content: str

    @property
    def content_hash(self) -> str:
        return sha256_hex(self.content)

    def span(self, char_start: int, char_end: int) -> EvidenceSpan:
        return EvidenceSpan(
            page_id=self.page_id,
            char_start=char_start,
            char_end=char_end,
            content_hash=self.content_hash,
        )


class Pages(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: tuple[Page, ...] = ()


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    value: Any
    spans: tuple[EvidenceSpan, ...] = ()


class Facts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: tuple[Fact, ...] = ()


class Answer(BaseModel):
    """Candidate answer produced by the runtime. Correctness is judged only offline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: dict[str, Any]
    evidence: tuple[FieldEvidence, ...] = ()


# What an executor may consume / produce.
Payload: TypeAlias = "ExecutionTask | Pages | Facts | Answer"
