"""Deprecated import path, kept so frozen experiment drivers keep importing what they ran with.

The provider-neutral contract lives in ``runtime.model_client``; the OpenAI-compatible HTTP
adapter in ``runtime.backends.openai_compatible``. New code imports from those.
"""

from __future__ import annotations

from runtime.backends.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    client_from_env,
)
from runtime.model_client import (
    GenerationRequest,
    GenerationResponse,
    ModelClient,
    ModelError,
    ModelRole,
    ModelUnavailableError,
)

GemmaConfig = OpenAICompatibleConfig  # legacy name

__all__ = [
    "GemmaConfig",
    "GenerationRequest",
    "GenerationResponse",
    "ModelClient",
    "ModelError",
    "ModelRole",
    "ModelUnavailableError",
    "OpenAICompatibleClient",
    "OpenAICompatibleConfig",
    "client_from_env",
]
