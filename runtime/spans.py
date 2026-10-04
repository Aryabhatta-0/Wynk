"""Evidence-span helpers: locate quotes in pages and validate spans against original pages.

Models never emit character offsets. They quote text; the runtime finds the quote and builds
the ``EvidenceSpan`` itself (deterministic, pinned to the ORIGINAL page's content hash).

FILTER outputs chunk pages that keep the original ``page_id`` and record their offset in
``source_ref`` as ``<ref>#chars=<start>-<end>``; ``chunk_offset`` reads it back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from core.evidence import EvidenceSpan
from core.payloads import Page

_CHUNK_REF = re.compile(r"#chars=(\d+)-(\d+)$")


def chunk_ref(source_ref: str, start: int, end: int) -> str:
    return f"{source_ref}#chars={start}-{end}"


def chunk_offset(page: Page) -> int:
    m = _CHUNK_REF.search(page.source_ref)
    return int(m.group(1)) if m else 0


def locate_quote(
    quote: str,
    pages: Iterable[Page],
    originals: Mapping[str, Page],
    page_id: str | None = None,
) -> EvidenceSpan | None:
    """Find ``quote`` verbatim in ``pages`` (preferring ``page_id``); return a span against the
    original page, or None if it cannot be located."""
    if not quote.strip():
        return None
    candidates = list(pages)
    if page_id is not None:
        candidates.sort(key=lambda p: p.page_id != page_id)  # stable: preferred page first
    for page in candidates:
        idx = page.content.find(quote)
        original = originals.get(page.page_id)
        if idx < 0 or original is None:
            continue
        start = chunk_offset(page) + idx
        end = start + len(quote)
        if original.content[start:end] == quote:
            return original.span(start, end)
    return None


def span_text(span: EvidenceSpan, originals: Mapping[str, Page]) -> str | None:
    """Text a span points at, or None if the span is invalid for the original pages
    (unknown page, content-hash mismatch, or range outside the page)."""
    page = originals.get(span.page_id)
    if page is None or page.content_hash != span.content_hash:
        return None
    if span.char_end > len(page.content):
        return None
    return page.content[span.char_start : span.char_end]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).strip().casefold()


def supports(value: Any, text: str) -> bool:
    """Deterministic check that ``text`` contains ``value`` (strings/numbers/lists thereof).
    Values of other types (bool, ...) are not text-checkable and pass."""
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, list):
        return all(supports(v, text) for v in value)
    if isinstance(value, str | int | float):
        return _norm(str(value)) in _norm(text)
    return True
