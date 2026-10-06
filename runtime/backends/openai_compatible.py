"""``ModelClient`` adapter for any OpenAI-compatible ``/chat/completions`` endpoint (stdlib only).

An adapter, not the abstraction: runtime code sees only ``runtime.model_client``. The adapter's
identity is ``openai_compatible_model_hash(model, revision, endpoint, structured)`` - unchanged
from the original client, so every frozen ``expected_model_hash`` still matches.
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
from typing import Any

from core.canonical import canonical_hash
from core.models import (
    ModelCapability,
    ModelEntry,
    ModelRegistryError,
    RetryPolicy,
    TimeoutPolicy,
)
from runtime.model_client import (
    GenerationRequest,
    GenerationResponse,
    ModelError,
    ModelUnavailableError,
    RegisteredModelClient,
)

ADAPTER = "openai_compatible"

# Each setting reads WYNK_MODEL_*, falling back to the legacy GEMMA_* name.
ENV = {
    "base_url": ("WYNK_MODEL_BASE_URL", "GEMMA_BASE_URL"),
    "model": ("WYNK_MODEL", "GEMMA_MODEL"),
    "api_key": ("WYNK_MODEL_API_KEY", "GEMMA_API_KEY"),
    "revision": ("WYNK_MODEL_REVISION", "GEMMA_MODEL_REVISION"),
    "structured": ("WYNK_MODEL_STRUCTURED", "GEMMA_STRUCTURED"),
    "timeout_s": ("WYNK_MODEL_TIMEOUT_S", "GEMMA_TIMEOUT_S"),
    "max_retries": ("WYNK_MODEL_MAX_RETRIES", "GEMMA_MAX_RETRIES"),
}


def openai_compatible_model_hash(model: str, revision: str, endpoint: str, structured: bool) -> str:
    """The exact runtime identity of an OpenAI-compatible model (pin this in the registry)."""
    return canonical_hash(
        {
            "model": model,
            "revision": revision,
            "endpoint": endpoint.rstrip("/"),
            "structured": structured,
        }
    )


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Connection settings. From the environment (``from_env``): WYNK_MODEL_BASE_URL,
    WYNK_MODEL, WYNK_MODEL_API_KEY (optional), WYNK_MODEL_REVISION (optional, part of the model
    hash), WYNK_MODEL_STRUCTURED=0 to disable ``response_format`` json_schema,
    WYNK_MODEL_TIMEOUT_S (default 120), WYNK_MODEL_MAX_RETRIES (default 3, retries on HTTP
    429/5xx and connection errors with exponential backoff). The legacy GEMMA_* names still work.
    From a registry entry: ``from_entry``."""

    base_url: str
    model: str
    api_key: str | None = None
    revision: str = ""
    structured: bool = True
    timeout_s: float = 120.0
    max_retries: int = 3
    backoff_s: float = 2.0
    max_delay_s: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OpenAICompatibleConfig:
        env = os.environ if env is None else env

        def get(key: str, default: str = "") -> str:
            for name in ENV[key]:
                if env.get(name):
                    return env[name]
            return default

        missing = [f"{ENV[k][0]} (or {ENV[k][1]})" for k in ("base_url", "model") if not get(k)]
        if missing:
            raise ModelUnavailableError(
                "no model backend configured: set "
                + ", ".join(missing)
                + " (OpenAI-compatible endpoint, e.g. a local server or a hosted gateway)"
            )
        return cls(
            base_url=get("base_url"),
            model=get("model"),
            api_key=get("api_key") or None,
            revision=get("revision"),
            structured=get("structured", "1") != "0",
            timeout_s=float(get("timeout_s", "120")),
            max_retries=int(get("max_retries", "3")),
        )

    @classmethod
    def from_entry(cls, entry: ModelEntry, api_key: str | None = None) -> OpenAICompatibleConfig:
        """Settings for a pinned registry entry (credentials are never part of the registry)."""
        if entry.adapter != ADAPTER:
            raise ModelRegistryError(f"entry {entry.name!r} uses adapter {entry.adapter!r}")
        if not entry.endpoint:
            raise ModelRegistryError(f"entry {entry.name!r} has no endpoint")
        return cls(
            base_url=entry.endpoint,
            model=entry.model_id,
            api_key=api_key,
            revision=entry.revision or "",
            structured=entry.structured_output,
            timeout_s=entry.timeout.request_timeout_s,
            max_retries=entry.retry.max_retries,
            backoff_s=entry.retry.backoff_s,
            max_delay_s=entry.retry.max_delay_s,
        )

    @property
    def model_hash(self) -> str:
        return openai_compatible_model_hash(
            self.model, self.revision, self.base_url, self.structured
        )

    @property
    def retry(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.max_retries, backoff_s=self.backoff_s, max_delay_s=self.max_delay_s
        )

    def entry(self, name: str | None = None, *, provider: str = ADAPTER) -> ModelEntry:
        """An UNREGISTERED entry describing exactly these settings, for ad-hoc runs whose
        identity is only recorded (never for experiments, which resolve through a registry).
        Capabilities come from the adapter protocol and the operator's ``structured`` setting;
        the context window is unknown."""
        caps = [ModelCapability.TEXT_GENERATION]
        if self.structured:
            caps.append(ModelCapability.STRUCTURED_OUTPUT)
        return ModelEntry(
            name=name or f"{self.model}@{self.base_url.rstrip('/')}",
            provider=provider,
            adapter=ADAPTER,
            model_id=self.model,
            revision=self.revision or None,
            endpoint=self.base_url,
            model_hash=self.model_hash,
            capabilities=tuple(caps),
            timeout=TimeoutPolicy(request_timeout_s=self.timeout_s),
            retry=self.retry,
        )


class OpenAICompatibleClient:
    """``ModelClient`` over an OpenAI-compatible chat-completions API (stdlib only)."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self.retry = config.retry
        self.model_hash = config.model_hash
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
                if not self.retry.retries(attempt):
                    exc.error.attempts = attempt + 1
                    exc.error.backoff_time_s = backoff_time_s
                    raise exc.error from exc.error.__cause__
                start = time.perf_counter()
                time.sleep(self.retry.delay(attempt, exc.retry_after))
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


def registered_client(entry: ModelEntry, api_key: str | None = None) -> RegisteredModelClient:
    """The registry-bound client for ``entry`` (fails closed if the hash does not reproduce)."""
    return RegisteredModelClient(
        entry, OpenAICompatibleClient(OpenAICompatibleConfig.from_entry(entry, api_key))
    )


def client_from_env(env: Mapping[str, str] | None = None) -> RegisteredModelClient:
    """Ad-hoc client from the environment. Raises ``ModelUnavailableError`` when no backend is
    configured."""
    config = OpenAICompatibleConfig.from_env(env)
    return RegisteredModelClient(config.entry(), OpenAICompatibleClient(config))
