"""Deterministic evidence verification against frozen snapshots (implementation: Track A)."""

from __future__ import annotations

from typing import Protocol

from core.evidence import EvidenceSpan


class EvidenceVerifier(Protocol):
    def is_valid(self, span: EvidenceSpan, snapshot_id: str) -> bool:
        """True iff the span's page exists in the snapshot, its content hash matches, and the
        character range lies inside the page."""
        ...
