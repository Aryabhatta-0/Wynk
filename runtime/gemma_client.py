"""Thin model-client contract. No serving backend is wired in Phase 0.

Gemma may ONLY extract, reason, synthesize, or act as the self-consistency sampler. It never
judges correctness, picks workflows, or sees ground truth - ``ModelRole`` makes any other use
unrepresentable, and requests carry only strings/schemas.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

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
    attempts: int = Field(default=1, ge=1)
    backoff_time_s: NonNegativeFloat = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(Protocol):
    model_hash: str

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...


# --- concrete backend: any OpenAI-compatible /chat/completions endpoint --------------------


class ModelError(RuntimeError):
    """The backend failed or returned something unusable. Never papered over with fake output."""

    def __init__(self, message: str, *, attempts: int = 0, backoff_time_s: float = 0.0) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.backoff_time_s = backoff_time_s


class ModelUnavailableError(ModelError):
    """No backend configured, or it cannot be reached."""


@dataclass(frozen=True)
class GemmaConfig:
    """Connection settings. Env: GEMMA_BASE_URL, GEMMA_MODEL, GEMMA_API_KEY (optional),
    GEMMA_MODEL_REVISION (optional, part of the model hash), GEMMA_STRUCTURED=0 to disable
    ``response_format`` json_schema, GEMMA_TIMEOUT_S (default 120), GEMMA_MAX_RETRIES (default 3,
    retries on HTTP 429/5xx and connection errors with exponential backoff)."""

    base_url: str
    model: str
    api_key: str | None = None
    revision: str = ""
    structured: bool = True
    timeout_s: float = 120.0
    max_retries: int = 3
    backoff_s: float = 2.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GemmaConfig:
        env = os.environ if env is None else env
        missing = [k for k in ("GEMMA_BASE_URL", "GEMMA_MODEL") if not env.get(k)]
        if missing:
            raise ModelUnavailableError(
                "no Gemma backend configured: set "
                + ", ".join(missing)
                + " (OpenAI-compatible endpoint, e.g. a local server or a hosted gateway)"
            )
        return cls(
            base_url=env["GEMMA_BASE_URL"],
            model=env["GEMMA_MODEL"],
            api_key=env.get("GEMMA_API_KEY") or None,
            revision=env.get("GEMMA_MODEL_REVISION", ""),
            structured=env.get("GEMMA_STRUCTURED", "1") != "0",
            timeout_s=float(env.get("GEMMA_TIMEOUT_S", "120")),
            max_retries=int(env.get("GEMMA_MAX_RETRIES", "3")),
        )


class OpenAICompatibleClient:
    """``ModelClient`` over an OpenAI-compatible chat-completions API (stdlib only)."""

    def __init__(self, config: GemmaConfig) -> None:
        self.config = config
        self.model_hash = canonical_hash(
            {
                "model": config.model,
                "revision": config.revision,
                "endpoint": config.base_url.rstrip("/"),
                "structured": config.structured,
            }
        )
        self.cacheable = bool(config.revision)  # unpinned weights can change behind a model name

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        body, attempts, backoff_time_s = await asyncio.to_thread(self._post, self._payload(request))
        try:
            text = body["choices"][0]["message"]["content"] or ""
            usage = body["usage"]
            prompt_tokens, completion_tokens = (
                int(usage["prompt_tokens"]),
                int(usage["completion_tokens"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(
                f"unexpected backend response (token usage required): {exc}",
                attempts=attempts,
                backoff_time_s=backoff_time_s,
            ) from exc
        return GenerationResponse(
            text=text,
            parsed=_try_parse(text) if request.output_schema else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_hash=self.model_hash,
            attempts=attempts,
            backoff_time_s=backoff_time_s,
        )

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.input_text}],
            "temperature": 0,
            "seed": request.seed,
            "max_tokens": request.max_tokens,
        }
        if request.output_schema and self.config.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.output_schema},
            }
        return payload

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        """POST with retries on rate limits (429), server errors (5xx) and dropped connections,
        so concurrency does not turn transient backend hiccups into MODEL_ERROR runs."""
        attempt = 0
        backoff_time_s = 0.0
        while True:
            try:
                return self._post_once(payload), attempt + 1, backoff_time_s
            except _RetryableError as exc:
                if attempt >= self.config.max_retries:
                    exc.error.attempts = attempt + 1
                    exc.error.backoff_time_s = backoff_time_s
                    raise exc.error from exc.error.__cause__
                delay = exc.retry_after
                if delay is None:
                    delay = self.config.backoff_s * 2**attempt
                start = time.perf_counter()
                time.sleep(min(delay, 60.0))
                backoff_time_s += time.perf_counter() - start
                attempt += 1
            except ModelError as exc:
                exc.attempts = attempt + 1
                exc.backoff_time_s = backoff_time_s
                raise

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            error = ModelError(f"backend returned HTTP {exc.code}")
            error.__cause__ = exc
            if exc.code == 429 or exc.code >= 500:
                raise _RetryableError(error, _retry_after(exc)) from exc
            raise error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = ModelUnavailableError(f"backend unreachable: {exc}")
            error.__cause__ = exc
            raise _RetryableError(error) from exc
        except json.JSONDecodeError as exc:
            raise ModelError("backend returned invalid JSON") from exc


class _RetryableError(Exception):
    def __init__(self, error: ModelError, retry_after: float | None = None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return max(0.0, float(exc.headers.get("Retry-After", "")))
    except (TypeError, ValueError, AttributeError):
        return None


def _try_parse(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def client_from_env() -> OpenAICompatibleClient:
    """Raises ``ModelUnavailableError`` when no backend is configured."""
    return OpenAICompatibleClient(GemmaConfig.from_env())
