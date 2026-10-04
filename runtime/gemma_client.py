"""Thin model-client contract. No serving backend is wired in Phase 0.

Gemma may ONLY extract, reason, synthesize, or act as the self-consistency sampler. It never
judges correctness, picks workflows, or sees ground truth - ``ModelRole`` makes any other use
unrepresentable, and requests carry only strings/schemas.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from core.canonical import canonical_hash


class ModelRole(StrEnum):
    EXTRACT = "extract"
    REASON = "reason"
    SYNTHESIZE = "synthesize"
    SELF_CONSISTENCY = "self_consistency"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    prompt_template_id: str = Field(min_length=1)
    prompt_template_version: str = Field(min_length=1)
    input_text: str
    output_schema: dict[str, Any] | None = None  # JSON Schema for structured output
    seed: int
    max_tokens: int = Field(gt=0)

    def cache_key(self, model_hash: str) -> str:
        """Deterministic identity of (request, model): same key => same cacheable response."""
        return canonical_hash({"request": self.model_dump(mode="json"), "model_hash": model_hash})


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    parsed: dict[str, Any] | None = None  # set when output_schema was honoured
    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    model_hash: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(Protocol):
    model_hash: str

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...
