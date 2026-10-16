"""Hosted JEV adapter: TypeSafe's ``systemone`` choice API behind ``runtime.sources.JevSource``.

The provider answers bounded choice questions only. Each decision asks two of them over the
observation Wynk built: the operation (OPEN / DONE / BLOCKED) and, for OPEN, which listed page.
An answer is accepted only if it is a well-formed probability distribution over exactly the
offered choices whose ``choice`` is its most probable option; anything else is a ``SourceError``
and no action is taken. ``parallel-2`` (``width`` 2) also opens the next most probable page when
the provider gives it a non-zero probability.

Configuration is explicit (``TypeSafeJevConfig.from_env`` returns ``None`` without a key, which
leaves JEV unavailable - it never falls back to another source). Every request has a bounded
timeout and is never retried. The API key is sent only in the ``Authorization`` header; it is not
part of ``repr``, ``identity``, or any error message. The JEV model's own token usage belongs to
the provider: Wynk charges each decision as one tool call, not as Wynk model tokens.

Sending a decision request sends the goal (the task question) and the text of the pages already
opened to the provider; configure JEV only for data that may leave the machine.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from runtime.sources import JevDecision, JevObservation, SourceError

ADAPTER = "typesafe-jev"
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
MAX_TIMEOUT_S = 120.0
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "api_key": ("WYNK_JEV_API_KEY", "TYPESAFE_API_KEY"),
    "model": ("WYNK_JEV_MODEL", "TYPESAFE_MODEL"),
    "endpoint": ("WYNK_JEV_ENDPOINT",),
    "timeout_s": ("WYNK_JEV_TIMEOUT_S",),
}
RULES = (
    "Choose only among the listed options. Open a page when its name suggests it may answer the "
    "goal and it is not open yet. Choose DONE once the open pages contain what the goal asks "
    "for. Choose BLOCKED when no listed page can help."
)

# (url, body, headers, timeout_s) -> response body. Injectable so tests never touch the network.
Transport = Callable[[str, bytes, Mapping[str, str], float], bytes]


def _urllib_transport(url: str, body: bytes, headers: Mapping[str, str], timeout_s: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (https enforced)
        return resp.read()


@dataclass(frozen=True)
class TypeSafeJevConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = 25.0
    max_page_chars: int = 6000  # text of the open pages shown to the provider, in total

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("JEV api_key is empty")
        if not self.model.strip():
            raise ValueError("JEV model is empty")
        parts = urlsplit(self.endpoint)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("JEV endpoint must be an https URL")
        if not 0 < self.timeout_s <= MAX_TIMEOUT_S:
            raise ValueError(f"JEV timeout_s must be in (0, {MAX_TIMEOUT_S}]")
        if self.max_page_chars < 0:
            raise ValueError("JEV max_page_chars must be >= 0")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> TypeSafeJevConfig | None:
        """The configured hosted JEV, or ``None`` when no API key is set (JEV unavailable)."""

        def get(key: str) -> str:
            return next((env[k].strip() for k in ENV_KEYS[key] if env.get(k, "").strip()), "")

        api_key = get("api_key")
        if not api_key:
            return None
        timeout = get("timeout_s")
        return cls(
            api_key=api_key,
            model=get("model") or DEFAULT_MODEL,
            endpoint=get("endpoint") or DEFAULT_ENDPOINT,
            timeout_s=float(timeout) if timeout else 25.0,
        )


class TypeSafeJevSource:
    """``JevSource`` backed by the hosted TypeSafe JEV model."""

    def __init__(self, config: TypeSafeJevConfig, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def __repr__(self) -> str:
        return f"TypeSafeJevSource(identity={self.identity!r})"

    @property
    def identity(self) -> str:
        parts = urlsplit(self.config.endpoint)
        return f"jev:{ADAPTER}/{self.config.model}@{parts.hostname}{parts.path}"

    def decide(self, observation: JevObservation) -> JevDecision:
        targets = {str(i + 1): e for i, e in enumerate(observation.elements)}
        operations = {
            "DONE": "The open pages already contain what the goal asks for.",
            "BLOCKED": "No listed page can help with the goal.",
        }
        if targets:
            operations = {"OPEN": "Open one of the listed pages to read it.", **operations}
        questions: dict[str, Any] = {
            "operation": {
                "type": "choice",
                "criteria": operations,
                "instructions": {"goal": observation.goal, "rules": RULES},
            }
        }
        if targets:
            questions["open_target"] = {
                "type": "choice",
                "criteria": {i: {"element": f"[{i}] {e.label}"} for i, e in targets.items()},
                "instructions": {"goal": observation.goal, "operation": "OPEN", "rules": RULES},
            }
        body = {
            "model": self.config.model,
            "state": {
                "page": {
                    "url": observation.surface,
                    "title": f"{len(observation.opened)} page(s) open",
                    "text": self._opened_text(observation),
                },
                "elements": [
                    {"index": i, "label": e.label, "operations": ["OPEN"]}
                    for i, e in targets.items()
                ],
                "recent_actions": [
                    {"action": "OPEN", "text": p.page_id} for p in observation.opened[-10:]
                ],
            },
            "questions": questions,
        }
        answers = self._post(body).get("answers")
        if not isinstance(answers, dict):
            raise SourceError("jev provider response has no answers")
        op = _valid_choice(answers.get("operation"), operations)
        confidence = float(op["confidence"])
        if op["choice"] == "DONE":
            return JevDecision(operation="done", confidence=confidence)
        if op["choice"] == "BLOCKED":
            return JevDecision(operation="blocked", confidence=confidence)
        target = _valid_choice(answers.get("open_target"), targets)
        probs: dict[str, float] = target["probabilities"]
        ranked = sorted(targets, key=lambda i: (-probs[i], int(i)))
        ranked.remove(target["choice"])
        picked = [target["choice"], *[i for i in ranked if probs[i] > 0]][: observation.width]
        return JevDecision(
            operation="open",
            targets=tuple(targets[i].element_id for i in picked),
            confidence=confidence,
        )

    def _opened_text(self, observation: JevObservation) -> str:
        text = "\n\n".join(f"[{p.page_id}]\n{p.content}" for p in observation.opened)
        return text[: self.config.max_page_chars]

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        try:
            raw = self._transport(
                self.config.endpoint, json.dumps(body).encode(), headers, self.config.timeout_s
            )
        except urllib.error.HTTPError as exc:
            raise SourceError(f"jev provider returned HTTP {exc.code}") from None
        except TimeoutError:
            raise SourceError(f"jev provider timed out after {self.config.timeout_s}s") from None
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            raise SourceError(f"jev provider unreachable ({type(exc).__name__})") from None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise SourceError("jev provider returned invalid JSON") from None
        if not isinstance(data, dict):
            raise SourceError("jev provider returned invalid JSON")
        return data


def _valid_choice(answer: object, ids: Mapping[str, object]) -> dict[str, Any]:
    """A choice answer over exactly ``ids``: finite probabilities in [0, 1] summing to 1, and the
    choice is (one of) the most probable options. Otherwise no action is taken."""
    if not isinstance(answer, dict):
        raise SourceError("jev provider returned an invalid choice; no action taken")
    try:
        probs = answer["probabilities"]
        numbers = [*probs.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probs) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probs.values()) - 1) < 0.02
            and probs[answer["choice"]] >= max(probs.values()) - 1e-6
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise SourceError("jev provider returned an invalid choice; no action taken")
    return answer


def jev_from_env(
    env: Mapping[str, str], transport: Transport | None = None
) -> TypeSafeJevSource | None:
    """The hosted ``JevSource`` configured in ``env``, or ``None`` (JEV unavailable)."""
    config = TypeSafeJevConfig.from_env(env)
    return None if config is None else TypeSafeJevSource(config, transport)
