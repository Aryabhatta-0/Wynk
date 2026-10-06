"""Provider-neutral model registry: the ONE authority for which models exist, what they can do,
how they are called, what they cost, and the exact runtime identity each one reports.

    ModelRegistry   pinned ``ModelEntry`` per model (unique name, natural key and model_hash)
      -> AllowedModels   the set a task / experiment explicitly declares, checked against its
                         ``ModelRequirements`` when it is built (fail closed, before any call)
      -> ModelClient     ``runtime.model_client.RegisteredModelClient``: an adapter whose own
                         hash must equal the entry's ``model_hash``
      -> RunVersions.model_hash -> experiment identity (``ExperimentPlan.expected_model_hash``)

Every arrow is an equality check, so there is no second model identity that can silently
disagree. Pure data: no I/O beyond parsing, no provider SDKs, no network, and no prices -
``ModelPricing`` is data supplied by whoever writes the registry. Capabilities are only what the
entry declares; nothing is ever inferred from a model's name.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from core.canonical import canonical_hash
from core.experiment import ModelConfiguration

REGISTRY_SCHEMA_VERSION = "wynk-model-registry/1"
REGISTRY_PRICING_VERSION = "registry-pricing/1"


class ModelRegistryError(ValueError):
    """The registry or a model declaration is inconsistent (raised before any model call)."""


class UnknownModelError(ModelRegistryError):
    """No registry entry pins this model."""


class DisallowedModelError(ModelRegistryError):
    """The model is disabled or outside the declared allowed set."""


class UnsupportedCapabilityError(ModelRegistryError):
    """The model provably lacks a capability the task / workflow / request needs."""


class ModelIdentityError(ModelRegistryError):
    """Two declarations of the same model disagree (hash, provider, model id, settings)."""


class ModelCapability(StrEnum):
    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"  # honours a JSON Schema response format


class TimeoutPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_timeout_s: PositiveFloat = Field(default=120.0, allow_inf_nan=False)


class RetryPolicy(BaseModel):
    """Retries of ONE model call on transient failures (rate limits, 5xx, dropped connections).

    Attempt ``n`` (0-based) that fails transiently is retried while ``n < max_retries``, after
    ``retry_after`` if the backend sent one, else ``backoff_s * 2**n``; never longer than
    ``max_delay_s``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: NonNegativeInt = 3
    backoff_s: NonNegativeFloat = Field(default=2.0, allow_inf_nan=False)
    max_delay_s: PositiveFloat = Field(default=60.0, allow_inf_nan=False)

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def retries(self, attempt: int) -> bool:
        return attempt < self.max_retries

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        wait = self.backoff_s * 2**attempt if retry_after is None else retry_after
        return min(max(0.0, wait), self.max_delay_s)


class ModelPricing(BaseModel):
    """Prices of one model, per million tokens, in ``currency``. ``version`` names the price
    sheet; it is part of every pricing identity, so a price change is a different experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    prompt_per_million: NonNegativeFloat = Field(allow_inf_nan=False)
    completion_per_million: NonNegativeFloat = Field(allow_inf_nan=False)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_million
            + completion_tokens * self.completion_per_million
        ) / 1_000_000

    def max_cost(self, tokens: int) -> float:
        """Safe upper bound for ``tokens`` tokens split ANY way between prompt and completion."""
        return tokens * max(self.prompt_per_million, self.completion_per_million) / 1_000_000


class ModelEntry(BaseModel):
    """One pinned model. ``model_hash`` is the exact identity the runtime client reports.

    ``provider`` says who serves the model (the ``ModelConfiguration.provider`` of experiments);
    ``adapter`` says which ``ModelClient`` backend speaks its API. ``revision`` is optional - a
    gateway may not expose one - but the hash is always pinned. ``context_window = None`` means
    UNKNOWN: it never satisfies a context requirement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str | None = None
    endpoint: str | None = None
    model_hash: str = Field(min_length=1)
    context_window: PositiveInt | None = None
    capabilities: tuple[ModelCapability, ...] = ()
    timeout: TimeoutPolicy = TimeoutPolicy()
    retry: RetryPolicy = RetryPolicy()
    pricing: ModelPricing | None = None
    enabled: bool = True

    @field_validator("capabilities")
    @classmethod
    def _canonical_capabilities(cls, v: tuple[ModelCapability, ...]) -> tuple[ModelCapability, ...]:
        if len(set(v)) != len(v):
            raise ValueError("capabilities must be distinct")
        return tuple(sorted(v))

    @field_validator("revision", "endpoint")
    @classmethod
    def _no_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("use null, not a blank string")
        return v

    def supports(self, capability: ModelCapability) -> bool:
        return capability in self.capabilities

    @property
    def structured_output(self) -> bool:
        return self.supports(ModelCapability.STRUCTURED_OUTPUT)

    @property
    def natural_key(self) -> tuple[str, ...]:
        """What makes two entries the same model; two such entries must not coexist."""
        endpoint = (self.endpoint or "").rstrip("/")
        return (
            self.provider,
            self.adapter,
            self.model_id,
            self.revision or "",
            endpoint,
            str(self.structured_output),
        )

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def check_configuration(self, config: ModelConfiguration) -> None:
        """Fail closed unless an experiment's ``ModelConfiguration`` describes THIS entry."""
        problems = []
        if config.provider != self.provider:
            problems.append(f"provider {config.provider!r} != {self.provider!r}")
        if config.model != self.model_id:
            problems.append(f"model {config.model!r} != {self.model_id!r}")
        params = config.parameters
        if "structured_output" in params and bool(params["structured_output"]) != (
            self.structured_output
        ):
            problems.append(
                f"structured_output {params['structured_output']!r} != {self.structured_output}"
            )
        if "revision" in params and (params["revision"] or None) != self.revision:
            problems.append(f"revision {params['revision']!r} != {self.revision!r}")
        if problems:
            raise ModelIdentityError(
                f"model configuration disagrees with registry entry {self.name!r}: "
                + "; ".join(problems)
            )


class ModelRequirements(BaseModel):
    """What a task / workflow / experiment needs from any model it may run on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[ModelCapability, ...] = (ModelCapability.TEXT_GENERATION,)
    min_context_tokens: PositiveInt | None = None

    def unmet(self, entry: ModelEntry) -> tuple[str, ...]:
        out = [f"lacks {c.value}" for c in self.capabilities if not entry.supports(c)]
        need = self.min_context_tokens
        if need is not None:
            if entry.context_window is None:
                out.append(f"context window unknown, {need} tokens required")
            elif entry.context_window < need:
                out.append(f"context window {entry.context_window} < required {need}")
        return tuple(out)

    def check(self, entry: ModelEntry) -> None:
        unmet = self.unmet(entry)
        if unmet:
            raise UnsupportedCapabilityError(f"model {entry.name!r}: " + "; ".join(unmet))


@dataclass(frozen=True)
class AllowedModels:
    """The explicitly declared set of models a task / experiment may execute on.

    Built (usually by ``ModelRegistry.allow``) only from enabled entries that meet
    ``requirements``; ``admit`` is the gate a runtime applies BEFORE invoking a model.
    """

    models: tuple[ModelEntry, ...]
    requirements: ModelRequirements = ModelRequirements()

    def __post_init__(self) -> None:
        if not self.models:
            raise DisallowedModelError("the allowed model set must not be empty")
        hashes = [m.model_hash for m in self.models]
        if len(set(hashes)) != len(hashes):
            raise ModelIdentityError("the allowed model set lists one model_hash twice")
        for entry in self.models:
            if not entry.enabled:
                raise DisallowedModelError(f"model {entry.name!r} is disabled")
            self.requirements.check(entry)

    @property
    def hashes(self) -> tuple[str, ...]:
        return tuple(m.model_hash for m in self.models)

    def __contains__(self, model_hash: object) -> bool:
        return model_hash in self.hashes

    def admit(self, model_hash: str) -> ModelEntry:
        for entry in self.models:
            if entry.model_hash == model_hash:
                return entry
        raise DisallowedModelError(
            f"model {model_hash!r} is not in the allowed set {list(self.hashes)}"
        )


class ModelRegistry(BaseModel):
    """Immutable set of pinned models. Duplicate names, hashes or natural keys are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wynk-model-registry/1"] = REGISTRY_SCHEMA_VERSION
    entries: tuple[ModelEntry, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> ModelRegistry:
        for what, key in (
            ("name", lambda e: e.name),
            ("model_hash", lambda e: e.model_hash),
            (
                "model (provider, adapter, model_id, revision, endpoint, structured)",
                lambda e: e.natural_key,
            ),
        ):
            seen: dict[Any, str] = {}
            for e in self.entries:
                k = key(e)
                if k in seen:
                    raise ValueError(f"duplicate {what}: entries {seen[k]!r} and {e.name!r}")
                seen[k] = e.name
        return self

    @classmethod
    def load(cls, path: Path) -> ModelRegistry:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def get(self, name: str) -> ModelEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise UnknownModelError(f"no registry entry named {name!r}")

    def by_hash(self, model_hash: str) -> ModelEntry:
        for e in self.entries:
            if e.model_hash == model_hash:
                return e
        raise UnknownModelError(f"model_hash {model_hash!r} is not pinned by any registry entry")

    def resolve(self, ref: str) -> ModelEntry:
        """Entry by name or exact model_hash (unknown: fail closed)."""
        for e in self.entries:
            if ref in (e.name, e.model_hash):
                return e
        raise UnknownModelError(f"{ref!r} is neither a registry name nor a pinned model_hash")

    def allow(
        self, refs: Iterable[str], requirements: ModelRequirements | None = None
    ) -> AllowedModels:
        """The declared allowed set: every ref must resolve to an enabled, capable entry."""
        return AllowedModels(
            tuple(self.resolve(r) for r in refs), requirements or ModelRequirements()
        )


class RegistryPricing:
    """``experiments.budget_ledger.PricingPolicy`` backed by registry pricing metadata.

    Prices exactly the allowed models; any other ``model_hash`` raises (the ledger turns that into
    a ``PricingError``: fail closed, never 0). Its identity covers every price sheet it uses.
    Build it with ``registry_pricing`` so an unpriced model means "cost unknown", not an error.
    """

    def __init__(self, models: Iterable[ModelEntry]) -> None:
        self._pricing: dict[str, ModelPricing] = {}
        for entry in models:
            if entry.pricing is None:
                raise ModelRegistryError(f"model {entry.name!r} has no pricing")
            self._pricing[entry.model_hash] = entry.pricing
        if not self._pricing:
            raise ModelRegistryError("registry pricing needs at least one priced model")
        table = {h: p.model_dump(mode="json") for h, p in self._pricing.items()}
        self._identity = f"{REGISTRY_PRICING_VERSION}:{canonical_hash(table)}"

    @property
    def identity(self) -> str:
        return self._identity

    def pricing(self, model_hash: str) -> ModelPricing:
        try:
            return self._pricing[model_hash]
        except KeyError:
            raise KeyError(f"no registry pricing for model {model_hash!r}") from None

    def cost(self, usage: Any) -> float:
        """``usage``: ``MeasuredUsage`` (model_hash, prompt_tokens, completion_tokens)."""
        return _finite(
            self.pricing(usage.model_hash).cost(usage.prompt_tokens, usage.completion_tokens)
        )

    def max_cost(self, model_hash: str, model_calls: int, tokens: int) -> float:
        del model_calls  # registry prices are per token; there is no per-call fee
        return _finite(self.pricing(model_hash).max_cost(tokens))


def registry_pricing(models: AllowedModels | Iterable[ModelEntry]) -> RegistryPricing | None:
    """Pricing policy for the allowed models, or ``None`` (cost UNKNOWN) if any is unpriced."""
    entries = models.models if isinstance(models, AllowedModels) else tuple(models)
    if not entries or any(e.pricing is None for e in entries):
        return None
    return RegistryPricing(entries)


def _finite(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"price computed as {value!r}")
    return value
