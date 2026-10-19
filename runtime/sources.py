"""Where GATHER gets pages from. Local, frozen, offline by design for ``fetch`` / ``api``.

Track A owns the real snapshot/mirror/mock-API implementations; they only need to satisfy these
two small protocols. Reference implementations read a directory layout:

    <root>/<snapshot_id>/*.txt|*.md|*.html     frozen pages (page_id = file stem)
    <root>/<snapshot_id>/api/*.json            mock API endpoints (page_id = "api:<stem>")

``jev`` is a browsing agent over the same snapshot surface. Wynk owns the browser loop: it
observes the pages not yet opened, asks a ``JevSource`` for a bounded choice among exactly those
(open up to N of them, done, or blocked), validates the answer against the observation, and only
then opens the chosen pages through the ``PageSource``. A provider never returns code, selectors
or URLs that Wynk executes; anything outside the offered choices is rejected.

``websearch`` is opt-in LIVE web search (a contract must list it in ``allowed_sources``). Wynk
sends a ``WebSearchSource`` one bounded ``WebSearchRequest`` (the row's inputs as the query, a
result cap and a deadline) and gets back a ``WebSearchResponse``: ranked results with a title, a
snippet and their URL. ``search_pages`` turns exactly those results into pages (deterministic
page ids, the real URL as ``source_ref``). Nothing returned is executed and no returned URL is
ever opened: the snippets ARE the evidence. Web results change over time, so a websearch run is
never a frozen benchmark snapshot (see ``runtime.executors.gather``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from core.canonical import canonical_json, sha256_hex
from core.evidence_authority import check_provider_identity
from core.payloads import Page

PAGE_SUFFIXES = (".txt", ".md", ".html")


class SourceError(RuntimeError):
    pass


def _contained(root: Path, *parts: str) -> Path:
    base = root.resolve()
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base):
        raise SourceError("source path escapes its root")
    return path


class PageSource(Protocol):
    """Frozen pages ("fetch")."""

    def list_page_ids(self, snapshot_id: str) -> list[str]: ...

    def read_page(self, snapshot_id: str, page_id: str) -> Page: ...


class ApiSource(Protocol):
    """Mock API ("api"): each endpoint call returns one page-like record."""

    def list_endpoints(self, snapshot_id: str) -> list[str]: ...

    def call(self, snapshot_id: str, endpoint: str) -> Page: ...


class DirectorySnapshotSource:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, snapshot_id: str) -> Path:
        d = _contained(self.root, snapshot_id)
        if not d.is_dir():
            raise SourceError(f"unknown snapshot: {snapshot_id}")
        return d

    def list_page_ids(self, snapshot_id: str) -> list[str]:
        files = [f for f in self._dir(snapshot_id).iterdir() if f.suffix in PAGE_SUFFIXES]
        return sorted(f.stem for f in files)

    def read_page(self, snapshot_id: str, page_id: str) -> Page:
        for suffix in PAGE_SUFFIXES:
            f = _contained(self._dir(snapshot_id), f"{page_id}{suffix}")
            if f.is_file():
                return Page(
                    page_id=page_id,
                    source_ref=f"snapshot://{snapshot_id}/{f.name}",
                    content=f.read_text(encoding="utf-8"),
                )
        raise SourceError(f"page not found: {snapshot_id}/{page_id}")


class DirectoryApiSource:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, snapshot_id: str) -> Path:
        d = _contained(_contained(self.root, snapshot_id), "api")
        if not d.is_dir():
            raise SourceError(f"snapshot {snapshot_id} has no mock api")
        return d

    def list_endpoints(self, snapshot_id: str) -> list[str]:
        return sorted(f.stem for f in self._dir(snapshot_id).glob("*.json"))

    def call(self, snapshot_id: str, endpoint: str) -> Page:
        f = _contained(self._dir(snapshot_id), f"{endpoint}.json")
        if not f.is_file():
            raise SourceError(f"unknown endpoint: {endpoint}")
        record = json.loads(f.read_text(encoding="utf-8"))
        return Page(
            page_id=f"api:{endpoint}",
            source_ref=f"api://{snapshot_id}/{endpoint}",
            content=canonical_json(record),
        )


# -- JEV ----------------------------------------------------------------------------------------


class JevElement(BaseModel):
    """One observed control the agent may choose: a snapshot page it has not opened yet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    element_id: str = Field(min_length=1)  # the page_id it opens
    label: str


class JevObservation(BaseModel):
    """What the agent sees before one decision. ``width`` is how many elements it may open in
    this step (1 for ``sequential``, 2 for ``parallel-2``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str  # the task question; never a target value
    surface: str  # e.g. "snapshot://<snapshot_id>"
    elements: tuple[JevElement, ...]
    opened: tuple[Page, ...] = ()  # pages already opened, in the order they were opened
    width: PositiveInt = 1
    step: int = Field(default=0, ge=0)


class JevDecision(BaseModel):
    """One bounded choice: open 1..width of the observed elements, or stop (done / blocked)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["open", "done", "blocked"]
    targets: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _shape(self) -> JevDecision:
        if self.operation == "open" and not self.targets:
            raise ValueError("open needs at least one target")
        if self.operation != "open" and self.targets:
            raise ValueError(f"{self.operation} takes no targets")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must not repeat")
        return self


class JevSource(Protocol):
    """A JEV provider: picks the next bounded browser action. ``identity`` names the provider and
    model version; JEV runs record it so they are never confused with frozen fetch/API runs."""

    @property
    def identity(self) -> str: ...

    def decide(self, observation: JevObservation) -> JevDecision: ...


def checked_decision(decision: object, observation: JevObservation) -> JevDecision:
    """The provider's answer, accepted only if it is one of the choices ``observation`` offered.
    Raises ``SourceError`` (never executes anything) for any other answer."""
    if not isinstance(decision, JevDecision):
        raise SourceError("jev provider returned an unstructured decision")
    offered = {e.element_id for e in observation.elements}
    unknown = [t for t in decision.targets if t not in offered]
    if unknown:
        raise SourceError(f"jev chose elements it was not offered: {', '.join(sorted(unknown))}")
    if len(decision.targets) > observation.width:
        raise SourceError(
            f"jev chose {len(decision.targets)} elements; this step allows {observation.width}"
        )
    return decision


# -- websearch ----------------------------------------------------------------------------------
# Bounds every websearch request obeys, whatever the provider: the query a provider may receive,
# how many results one request may return, and how much snippet text becomes one page.
MAX_QUERY_CHARS = 400
MAX_WEBSEARCH_RESULTS = 10
MAX_SNIPPET_CHARS = 2000


def web_query(text: str) -> str:
    """The search query for ``text`` (a row's inputs): whitespace collapsed, cut to at most
    ``MAX_QUERY_CHARS`` at a word boundary. Deterministic; empty if ``text`` has no words."""
    words = " ".join(text.split())
    if len(words) <= MAX_QUERY_CHARS:
        return words
    cut = words[: MAX_QUERY_CHARS + 1]
    return cut[: cut.rfind(" ")] if " " in cut else cut[:MAX_QUERY_CHARS]


class WebSearchRequest(BaseModel):
    """One bounded search. ``timeout_s`` is the most wall time the provider may spend on it (the
    run's remaining wall-time budget, at most the provider's own timeout)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    max_results: int = Field(ge=1, le=MAX_WEBSEARCH_RESULTS)
    timeout_s: float = Field(gt=0, allow_inf_nan=False)


class WebSearchResult(BaseModel):
    """One ranked result as the provider returned it (validated, never followed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: PositiveInt  # 1 = the provider's first result
    url: str = Field(min_length=1, max_length=2048)
    title: str = Field(max_length=1000)
    snippet: str = Field(max_length=MAX_SNIPPET_CHARS)

    @field_validator("url")
    @classmethod
    def _web_url(cls, v: str) -> str:
        parts = urlsplit(v)
        if parts.scheme not in ("http", "https") or not parts.hostname or re.search(r"\s", v):
            raise ValueError("a search result URL must be an absolute http(s) URL")
        return v


class WebSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[WebSearchResult, ...] = ()

    @model_validator(mode="after")
    def _ranked(self) -> WebSearchResponse:
        if [r.rank for r in self.results] != list(range(1, len(self.results) + 1)):
            raise ValueError("search results must be ranked 1..n in order")
        return self


class WebSearchSource(Protocol):
    """A hosted web-search provider. ``identity`` names the provider and every configuration
    value that changes what a search returns (endpoint, result count, fixed parameters); a
    websearch run records it as ``RunVersions.evidence_source``. ``max_results`` is the result
    cap it is configured for. ``search`` performs exactly ONE request and never retries."""

    @property
    def identity(self) -> str: ...

    @property
    def max_results(self) -> int: ...

    def search(self, request: WebSearchRequest) -> WebSearchResponse: ...


class ProviderDrifted(SourceError):
    """A pinned provider now reports another identity: the call was refused, never made, so it
    is not charged."""


class PinnedJevSource:
    """A ``JevSource`` whose identity was read and validated ONCE, when it was attached. The
    runner records this identity and the executor uses this object, so the two can never
    disagree; every decision first re-checks the provider still reports it (``SourceError``,
    nothing requested, if not)."""

    def __init__(self, inner: JevSource) -> None:
        self._inner = inner
        self._identity = check_provider_identity(inner.identity, websearch=False)

    @property
    def identity(self) -> str:
        return self._identity

    def decide(self, observation: JevObservation) -> JevDecision:
        if self._inner.identity != self._identity:
            raise ProviderDrifted(DRIFTED.format(source="jev"))
        return self._inner.decide(observation)


class PinnedWebSearchSource:
    """A ``WebSearchSource`` whose identity was read and validated ONCE (it must be in the
    ``websearch:`` namespace the live-evidence exclusions key on). ``drifted`` lets the executor
    refuse before charging anything; ``search`` re-checks before the one request is sent."""

    def __init__(self, inner: WebSearchSource) -> None:
        self._inner = inner
        self._identity = check_provider_identity(inner.identity, websearch=True)

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def max_results(self) -> int:
        return self._inner.max_results

    def drifted(self) -> bool:
        return self._inner.identity != self._identity

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        if self.drifted():
            raise ProviderDrifted(DRIFTED.format(source="websearch"))
        return self._inner.search(request)


DRIFTED = "{source} provider identity changed after it was configured"


def pin_jev(source: JevSource | None) -> PinnedJevSource | None:
    if source is None or isinstance(source, PinnedJevSource):
        return source
    return PinnedJevSource(source)


def pin_websearch(source: WebSearchSource | None) -> PinnedWebSearchSource | None:
    if source is None or isinstance(source, PinnedWebSearchSource):
        return source
    return PinnedWebSearchSource(source)


def checked_response(response: object, request: WebSearchRequest) -> WebSearchResponse:
    """The provider's answer, accepted only if it is a structured response within the request's
    bounds. Raises ``SourceError`` (and uses none of it) otherwise."""
    if not isinstance(response, WebSearchResponse):
        raise SourceError("websearch provider returned an unstructured response")
    if len(response.results) > request.max_results:
        raise SourceError(
            f"websearch provider returned {len(response.results)} results; "
            f"{request.max_results} were requested"
        )
    return response


def search_pages(response: WebSearchResponse) -> tuple[Page, ...]:
    """One page per result, in rank order. ``page_id`` is ``web-<rank>-<url digest>`` (stable for
    the same result list), ``source_ref`` the real result URL, and the content its title and
    snippet - exactly what the provider returned, nothing fetched."""
    return tuple(
        Page(
            page_id=f"web-{r.rank:02d}-{sha256_hex(r.url)[:12]}",
            source_ref=r.url,
            content=f"{r.title}\n\n{r.snippet}" if r.title else r.snippet,
        )
        for r in response.results
    )
