"""FILTER executor: Pages -> Pages, deterministic keyword relevance, no model.

``keyword_chunk`` splits pages into paragraph chunks (long ones into fixed windows);
``section_select`` splits on headings. Chunks are scored by overlap with the question's
keywords; the best ``MAX_CHUNKS`` are kept in original order. If nothing matches, pages pass
through unchanged (never silently drop everything). Chunks keep the original ``page_id`` and
record their offset in ``source_ref`` so evidence spans still point at the original page.
"""

from __future__ import annotations

import re

from core.payloads import Page, Pages
from core.results import BudgetUsage
from core.stages import FilterMethod, StageKind
from runtime.executors.base import ExecutorInput, ExecutorOutput, RunContext, StageExecutor
from runtime.spans import chunk_ref

MAX_CHUNKS = 6
WINDOW = 1200
_STOP = frozenset(
    "the a an of in on at to is are was were be what which who whom whose when where why how "
    "for and or by with from as that this these those it its do does did has have had".split()
)
_PARAGRAPH = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^(#{1,6}\s|<h[1-6][\s>])", re.MULTILINE | re.IGNORECASE)


def keywords(*texts: str) -> frozenset[str]:
    words = (w for t in texts for w in re.findall(r"[a-z0-9]+", t.casefold()))
    return frozenset(w for w in words if len(w) > 2 and w not in _STOP)


def _ranges(content: str, method: FilterMethod) -> list[tuple[int, int]]:
    if method == FilterMethod.SECTION_SELECT:
        bounds = sorted({0, *(m.start() for m in _HEADING.finditer(content)), len(content)})
    else:
        bounds = sorted({0, *(m.end() for m in _PARAGRAPH.finditer(content)), len(content)})
    out: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        for s in range(a, b, WINDOW):  # cap chunk size
            out.append((s, min(s + WINDOW, b)))
    return [(a, b) for a, b in out if content[a:b].strip()]


class FilterExecutor(StageExecutor):
    kind = StageKind.FILTER

    async def run(self, inp: ExecutorInput, ctx: RunContext) -> ExecutorOutput:
        pages = inp.payload
        assert isinstance(pages, Pages)
        want = keywords(ctx.task.inputs_text, *(f.name for f in ctx.task.answer_schema.fields))
        scored: list[tuple[int, int, Page]] = []
        for pi, page in enumerate(pages.pages):
            for start, end in _ranges(page.content, inp.stage.method):
                text = page.content[start:end]
                score = len(want & keywords(text))
                if score:
                    chunk = Page(
                        page_id=page.page_id,
                        source_ref=chunk_ref(page.source_ref, start, end),
                        content=text,
                    )
                    scored.append((score, pi * 10**9 + start, chunk))
        if not scored:
            return ExecutorOutput(payload=pages, usage=BudgetUsage())
        best = sorted(scored, key=lambda t: (-t[0], t[1]))[:MAX_CHUNKS]
        kept = tuple(c for _, _, c in sorted(best, key=lambda t: t[1]))
        return ExecutorOutput(payload=Pages(pages=kept), usage=BudgetUsage())
