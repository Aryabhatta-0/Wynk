"""Hosted websearch adapter: the Brave Search web API behind ``runtime.sources.WebSearchSource``.

One ``search`` is exactly one HTTPS ``GET`` to the configured endpoint with a fixed parameter
set - the query (``q``), the result count and the pinned filters in ``FIXED_PARAMS`` - and the
API key in the ``X-Subscription-Token`` header only. Nothing else about the task is sent. The
request has a bounded timeout (the smaller of the configured one and the run's remaining wall
time), is never retried, never falls back to another provider and never follows a redirect.

The response is validated strictly before any of it is used: a JSON object of ``type`` "search"
whose optional ``web.results`` is a list of objects with a string ``title`` and an absolute
http(s) ``url`` (``description``, when present, a string), no more of them than were requested.
Anything else is a ``SourceError`` and no page is produced. Titles / snippets are cut to the
``runtime.sources`` bounds. Returned URLs are recorded as page provenance and never opened.

Configuration is explicit (``BraveSearchConfig.from_env`` returns ``None`` without a key, which
leaves websearch unavailable). ``identity`` - recorded as ``RunVersions.evidence_source`` - names
the adapter, the endpoint and every parameter that shapes the results, so two configurations
that could return different results never share a run identity. Brave exposes no response
revision, so none is invented. The key is not part of ``repr``, ``identity`` or any error.

Brave's per-request price is not recorded as Wynk cost: Wynk's pricing path prices model tokens
only, so a websearch request is accounted as one tool call and no USD amount is invented.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

from pydantic import ValidationError

from core.evidence_authority import WEBSEARCH_IDENTITY_PREFIX
from runtime.sources import (
    MAX_SNIPPET_CHARS,
    MAX_WEBSEARCH_RESULTS,
    SourceError,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)

ADAPTER = "brave-search"
DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_TIMEOUT_S = 15.0
MAX_TIMEOUT_S = 60.0
DEFAULT_MAX_RESULTS = 5
MAX_RESPONSE_BYTES = 2_000_000
MAX_TITLE_CHARS = 300
# Pinned request parameters besides ``q`` / ``count``: part of ``identity``.
FIXED_PARAMS: tuple[tuple[str, str], ...] = (
    ("result_filter", "web"),
    ("safesearch", "strict"),
    ("spellcheck", "false"),
    ("text_decorations", "false"),
)
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "api_key": ("WYNK_WEBSEARCH_API_KEY", "BRAVE_SEARCH_API_KEY"),
    "endpoint": ("WYNK_WEBSEARCH_ENDPOINT",),
    "timeout_s": ("WYNK_WEBSEARCH_TIMEOUT_S",),
    "max_results": ("WYNK_WEBSEARCH_MAX_RESULTS",),
}

# (url, headers, timeout_s) -> response body. Injectable so tests never touch the network.
Transport = Callable[[str, Mapping[str, str], float], bytes]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None  # a redirect is an HTTPError: never followed (it could leave https)


def _urllib_transport(url: str, headers: Mapping[str, str], timeout_s: float) -> bytes:
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=dict(headers), method="GET")
    with opener.open(req, timeout=timeout_s) as resp:  # https enforced by the config
        body = resp.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise SourceError("websearch provider response is too large")
    return body


# A header value Wynk sends: visible ASCII only (no whitespace, CR/LF or control characters).
_HEADER_SAFE = re.compile(r"[\x21-\x7e]+")


def _canonical_endpoint(endpoint: str) -> str:
    """``host[:port]/path`` of ``endpoint``: lower-case host, the default https port omitted, so
    two endpoints share it only if they are the same service."""
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    host = f"[{host}]" if ":" in host else host  # an IPv6 literal keeps its brackets
    port = "" if parts.port in (None, 443) else f":{parts.port}"
    return f"{host}{port}{parts.path}"


class WebSearchConfigError(ValueError):
    """Websearch is configured in a way Wynk refuses to run (never contains the API key)."""


@dataclass(frozen=True)
class BraveSearchConfig:
    api_key: str = field(repr=False)
    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_results: int = DEFAULT_MAX_RESULTS

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise WebSearchConfigError("websearch api_key is empty")
        if not _HEADER_SAFE.fullmatch(self.api_key):  # never echoed: it is the secret
            raise WebSearchConfigError(
                "websearch api_key contains characters an HTTP header cannot carry"
            )
        if "?" in self.endpoint or "#" in self.endpoint:  # even an empty query or fragment
            raise WebSearchConfigError("websearch endpoint must not contain '?' or '#'")
        parts = urlsplit(self.endpoint)
        try:
            parts.port  # noqa: B018 - raises for a malformed port
        except ValueError:
            raise WebSearchConfigError("websearch endpoint has an invalid port") from None
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise WebSearchConfigError(
                "websearch endpoint must be an https URL without credentials, query or fragment"
            )
        if not 0 < self.timeout_s <= MAX_TIMEOUT_S:
            raise WebSearchConfigError(f"websearch timeout_s must be in (0, {MAX_TIMEOUT_S}]")
        if not 1 <= self.max_results <= MAX_WEBSEARCH_RESULTS:
            raise WebSearchConfigError(
                f"websearch max_results must be in [1, {MAX_WEBSEARCH_RESULTS}]"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BraveSearchConfig | None:
        """The configured websearch, or ``None`` when no API key is set (websearch unavailable)."""

        def get(key: str) -> str:
            return next((env[k].strip() for k in ENV_KEYS[key] if env.get(k, "").strip()), "")

        api_key = get("api_key")
        if not api_key:
            return None
        timeout, count = get("timeout_s"), get("max_results")
        try:
            timeout_s = float(timeout) if timeout else DEFAULT_TIMEOUT_S
        except ValueError:
            raise WebSearchConfigError("WYNK_WEBSEARCH_TIMEOUT_S must be a number") from None
        try:
            max_results = int(count) if count else DEFAULT_MAX_RESULTS
        except ValueError:
            raise WebSearchConfigError("WYNK_WEBSEARCH_MAX_RESULTS must be an integer") from None
        return cls(
            api_key=api_key,
            endpoint=get("endpoint") or DEFAULT_ENDPOINT,
            timeout_s=timeout_s,
            max_results=max_results,
        )


class BraveWebSearchSource:
    """``WebSearchSource`` backed by the hosted Brave Search web API."""

    def __init__(self, config: BraveSearchConfig, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def __repr__(self) -> str:
        return f"BraveWebSearchSource(identity={self.identity!r})"

    @property
    def max_results(self) -> int:
        return self.config.max_results

    @property
    def identity(self) -> str:
        params = urlencode(sorted((("count", str(self.config.max_results)), *FIXED_PARAMS)))
        endpoint = _canonical_endpoint(self.config.endpoint)
        return f"{WEBSEARCH_IDENTITY_PREFIX}{ADAPTER}@{endpoint}?{params}"

    def request_url(self, request: WebSearchRequest) -> str:
        if request.max_results != self.config.max_results:
            raise SourceError("websearch request does not match the configured result count")
        params = (("q", request.query), ("count", str(request.max_results)), *FIXED_PARAMS)
        # exactly the endpoint the identity names (canonical host, port and path)
        return f"https://{_canonical_endpoint(self.config.endpoint)}?{urlencode(params)}"

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        url = self.request_url(request)
        headers = {"Accept": "application/json", "X-Subscription-Token": self.config.api_key}
        timeout_s = min(self.config.timeout_s, request.timeout_s)
        try:
            raw = self._transport(url, headers, timeout_s)
        except urllib.error.HTTPError as exc:
            raise SourceError(f"websearch provider returned HTTP {exc.code}") from None
        except TimeoutError:
            raise SourceError(f"websearch provider timed out after {timeout_s:.1f}s") from None
        except SourceError:
            raise
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            raise SourceError(f"websearch provider unreachable ({type(exc).__name__})") from None
        except Exception as exc:  # anything else: report its type only, never its message
            raise SourceError(f"websearch request failed ({type(exc).__name__})") from None
        return parse_response(raw, request.max_results)


def parse_response(raw: bytes, max_results: int) -> WebSearchResponse:
    """Strictly validated results of one Brave web-search response body."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SourceError("websearch provider returned invalid JSON") from None
    if not isinstance(data, dict) or data.get("type") != "search":
        raise SourceError("websearch provider returned an unexpected response shape")
    web = data.get("web", {"results": []})  # Brave omits ``web`` when nothing matched
    items = web.get("results") if isinstance(web, dict) else None
    if not isinstance(items, list):
        raise SourceError("websearch provider returned an unexpected response shape")
    if len(items) > max_results:
        raise SourceError(
            f"websearch provider returned {len(items)} results; {max_results} were requested"
        )
    results = []
    for rank, item in enumerate(items, start=1):
        results.append(_result(item, rank))
    return WebSearchResponse(results=tuple(results))


def _result(item: Any, rank: int) -> WebSearchResult:
    if not isinstance(item, dict):
        raise SourceError("websearch provider returned a malformed result")
    url, title = item.get("url"), item.get("title")
    snippet = item.get("description", "")
    if not (isinstance(url, str) and isinstance(title, str) and isinstance(snippet, str)):
        raise SourceError("websearch provider returned a malformed result")
    try:
        return WebSearchResult(
            rank=rank,
            url=url,
            title=" ".join(title.split())[:MAX_TITLE_CHARS],
            snippet=" ".join(snippet.split())[:MAX_SNIPPET_CHARS],
        )
    except ValidationError:  # never echo the provider's payload
        raise SourceError("websearch provider returned a malformed result") from None


def websearch_from_env(
    env: Mapping[str, str], transport: Transport | None = None
) -> BraveWebSearchSource | None:
    """The hosted ``WebSearchSource`` configured in ``env``, or ``None`` (websearch unavailable)."""
    config = BraveSearchConfig.from_env(env)
    return None if config is None else BraveWebSearchSource(config, transport)
