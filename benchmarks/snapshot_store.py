"""Read-only access to the frozen local snapshots (no network, no mutation).

Layout: ``snapshots/<snapshot_id>/records.json`` (the underlying structured data, served by
``mock_api``) and ``snapshots/<snapshot_id>/pages/<page_id>.txt`` (the static pages a gatherer
sees). Pages are read as raw UTF-8 bytes so content hashes never depend on the platform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.canonical import canonical_hash
from core.payloads import Page, Pages

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "snapshots"


class SnapshotStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SNAPSHOT_ROOT

    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted(p.name for p in self.root.iterdir() if p.is_dir()))

    def has_snapshot(self, snapshot_id: str) -> bool:
        return (self.root / snapshot_id).is_dir()

    def pages(self, snapshot_id: str) -> Pages:
        pages_dir = self.root / snapshot_id / "pages"
        if not pages_dir.is_dir():
            return Pages()
        return Pages(
            pages=tuple(
                Page(
                    page_id=f.stem,
                    source_ref=f"snapshot://{snapshot_id}/{f.stem}",
                    content=f.read_bytes().decode("utf-8"),
                )
                for f in sorted(pages_dir.glob("*.txt"))
            )
        )

    def page(self, snapshot_id: str, page_id: str) -> Page | None:
        for p in self.pages(snapshot_id).pages:
            if p.page_id == page_id:
                return p
        return None

    def records(self, snapshot_id: str) -> dict[str, list[dict[str, Any]]]:
        path = self.root / snapshot_id / "records.json"
        return json.loads(path.read_bytes().decode("utf-8"))["endpoints"]

    def snapshot_hash(self, snapshot_id: str) -> str:
        pages = {p.page_id: p.content_hash for p in self.pages(snapshot_id).pages}
        return canonical_hash(
            {"snapshot": snapshot_id, "pages": pages, "records": self.records(snapshot_id)}
        )
