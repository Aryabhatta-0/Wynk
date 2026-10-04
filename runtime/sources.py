"""Where GATHER gets pages from. Local, frozen, offline by design (MVP).

Track A owns the real snapshot/mirror/mock-API implementations; they only need to satisfy these
two small protocols. Reference implementations read a directory layout:

    <root>/<snapshot_id>/*.txt|*.md|*.html     frozen pages (page_id = file stem)
    <root>/<snapshot_id>/api/*.json            mock API endpoints (page_id = "api:<stem>")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from core.canonical import canonical_json
from core.payloads import Page

PAGE_SUFFIXES = (".txt", ".md", ".html")


class SourceError(RuntimeError):
    pass


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
        d = self.root / snapshot_id
        if not d.is_dir():
            raise SourceError(f"unknown snapshot: {snapshot_id}")
        return d

    def list_page_ids(self, snapshot_id: str) -> list[str]:
        files = [f for f in self._dir(snapshot_id).iterdir() if f.suffix in PAGE_SUFFIXES]
        return sorted(f.stem for f in files)

    def read_page(self, snapshot_id: str, page_id: str) -> Page:
        for suffix in PAGE_SUFFIXES:
            f = self._dir(snapshot_id) / f"{page_id}{suffix}"
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
        d = self.root / snapshot_id / "api"
        if not d.is_dir():
            raise SourceError(f"snapshot {snapshot_id} has no mock api")
        return d

    def list_endpoints(self, snapshot_id: str) -> list[str]:
        return sorted(f.stem for f in self._dir(snapshot_id).glob("*.json"))

    def call(self, snapshot_id: str, endpoint: str) -> Page:
        f = self._dir(snapshot_id) / f"{endpoint}.json"
        if not f.is_file():
            raise SourceError(f"unknown endpoint: {endpoint}")
        record = json.loads(f.read_text(encoding="utf-8"))
        return Page(
            page_id=f"api:{endpoint}",
            source_ref=f"api://{snapshot_id}/{endpoint}",
            content=canonical_json(record),
        )
