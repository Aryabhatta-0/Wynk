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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from core.canonical import canonical_json
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
