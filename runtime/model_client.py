"""The ONE provider-neutral model-client contract. Runtime and executors depend on this only.

A model may ONLY extract, reason, synthesize, or act as the self-consistency sampler. It never
judges correctness, picks workflows, or sees ground truth - ``ModelRole`` makes any other use
unrepresentable, and requests carry only strings/schemas.

Backends (e.g. ``runtime.backends.openai_compatible``) implement ``ModelClient``; the registry
binds one to its pinned entry with ``RegisteredModelClient``, so the hash every run reports
(``RunVersions.model_hash``) is the registry's ``ModelEntry.model_hash`` and nothing else.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from core.canonical import canonical_hash
from core.models import (
    DisallowedModelError,
    ModelCapability,
    ModelEntry,
    ModelIdentityError,
)


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
    attempts: int = Field(default=1, ge=1)
    backoff_time_s: NonNegativeFloat = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(Protocol):
    model_hash: str

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class ModelError(RuntimeError):
    """The backend failed or returned something unusable. Never papered over with fake output.

    ``attempts`` is how many times the model was actually invoked (0: it never was)."""

    def __init__(self, message: str, *, attempts: int = 0, backoff_time_s: float = 0.0) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.backoff_time_s = backoff_time_s


class ModelUnavailableError(ModelError):
    """No backend configured, or it cannot be reached."""


class ModelContractViolation(ModelError):
    """A request or response breaks the registered model's contract (context window,
    capability, identity). Raised instead of a usable response; never retried."""


def request_violation(entry: ModelEntry, request: GenerationRequest) -> str | None:
    """Why ``entry`` provably cannot serve ``request`` (``None``: nothing provable)."""
    if not entry.supports(ModelCapability.TEXT_GENERATION):
        return f"model {entry.name!r} does not declare text generation"
    window = entry.context_window
    if window is not None and request.max_tokens > window:
        return (
            f"max_tokens {request.max_tokens} exceeds model {entry.name!r} context window {window}"
        )
    return None


class RegisteredModelClient:
    """A backend bound to its pinned registry entry: entry -> client -> exact ``model_hash``.

    Construction fails closed if the entry is disabled or the backend's own identity differs
    from the entry's ``model_hash``. Each request is checked against the entry's declared
    capabilities BEFORE the backend is invoked, and each response must carry the pinned hash.
    """

    def __init__(self, entry: ModelEntry, backend: ModelClient) -> None:
        if not entry.enabled:
            raise DisallowedModelError(f"model {entry.name!r} is disabled")
        if backend.model_hash != entry.model_hash:
            raise ModelIdentityError(
                f"backend reports model_hash {backend.model_hash!r}, registry entry "
                f"{entry.name!r} pins {entry.model_hash!r}"
            )
        self.entry = entry
        self.backend = backend
        self.model_hash = entry.model_hash
        self.cacheable = bool(getattr(backend, "cacheable", entry.revision is not None))

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        violation = request_violation(self.entry, request)
        if violation is not None:
            raise ModelContractViolation(violation)
        response = await self.backend.generate(request)
        if response.model_hash != self.model_hash:
            raise ModelContractViolation(
                f"response from model {response.model_hash!r}, expected {self.model_hash!r}",
                attempts=response.attempts,
                backoff_time_s=response.backoff_time_s,
            )
        return response
