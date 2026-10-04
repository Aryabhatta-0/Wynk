"""Structured evidence. No loose citation strings anywhere in the framework."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceSpan(BaseModel):
    """A character range inside a gathered page, pinned to that page's content hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    content_hash: str  # sha256 hex of the page content the offsets refer to

    @field_validator("content_hash")
    @classmethod
    def _is_sha256(cls, v: str) -> str:
        if not _SHA256.match(v):
            raise ValueError("content_hash must be a lowercase sha256 hex digest")
        return v

    @model_validator(mode="after")
    def _non_empty_range(self) -> EvidenceSpan:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class FieldEvidence(BaseModel):
    """Evidence cited for one answer field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
