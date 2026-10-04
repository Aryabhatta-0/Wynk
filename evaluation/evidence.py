"""Deterministic evidence verification against frozen snapshots."""

from __future__ import annotations

from typing import Protocol

from benchmarks.snapshot_store import SnapshotStore
from core.evidence import EvidenceSpan


class EvidenceVerifier(Protocol):
    def is_valid(self, span: EvidenceSpan, snapshot_id: str) -> bool:
        """True iff the span's page exists in the snapshot, its content hash matches, and the
        character range lies inside the page."""
        ...


class SnapshotEvidenceVerifier:
    """Checks spans against the local snapshot pages. Structural validity only (MVP): it does
    not judge whether the cited text *supports* the value - that would need semantics."""

    def __init__(self, store: SnapshotStore | None = None) -> None:
        self.store = store or SnapshotStore()
        self._cache: dict[str, dict[str, tuple[str, int]]] = {}

    def _index(self, snapshot_id: str) -> dict[str, tuple[str, int]]:
        if snapshot_id not in self._cache:
            self._cache[snapshot_id] = {
                p.page_id: (p.content_hash, len(p.content))
                for p in self.store.pages(snapshot_id).pages
            }
        return self._cache[snapshot_id]

    def is_valid(self, span: EvidenceSpan, snapshot_id: str) -> bool:
        entry = self._index(snapshot_id).get(span.page_id)
        if entry is None:
            return False
        content_hash, length = entry
        return span.content_hash == content_hash and span.char_end <= length
