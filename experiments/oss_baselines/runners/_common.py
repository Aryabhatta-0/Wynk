"""Stdlib-only helpers shared by every runner (each runner executes inside its own venv).

Runner protocol: ``python <runner>.py <input.json> <output.json>``.

input.json  (written by the harness; contains NO ground truth):
    {"run_id", "task_prompt", "pages": {page_id: text}, "model": {...}, "task": {...}}
output.json:
    {"ok", "final", "agent_latency_s", "tool_calls": [...], "steps", "error", "framework"}

Every external framework gets the SAME two tools over the task's frozen pages, with the same
names, docstrings and behaviour (``list_pages`` / ``read_page``). That is the CONTROLLED_SOURCE
information universe Wynk's GATHER stage also reads from.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

LIST_PAGES_DOC = "List the ids of the source pages available for this task."
READ_PAGE_DOC = "Return the full text of one source page."
READ_PAGE_ARG_DOC = "Id of the page to read, as returned by list_pages."


class PageTools:
    """The page corpus for one task plus a log of every tool invocation."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = dict(pages)
        self.calls: list[dict[str, Any]] = []

    def list_pages(self) -> str:
        out = ", ".join(sorted(self.pages))
        self.calls.append({"tool": "list_pages", "args": {}, "chars": len(out)})
        return out

    def read_page(self, page_id: str) -> str:
        key = str(page_id).strip()
        out = self.pages.get(key)
        if out is None:
            out = f"No page with id {key!r}. Available pages: {', '.join(sorted(self.pages))}"
        self.calls.append({"tool": "read_page", "args": {"page_id": key}, "chars": len(out)})
        return out


def jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))


def main(framework: Callable[[], dict[str, str]], run: Callable[[dict, PageTools], Any]) -> None:
    """``run(inp, tools) -> (final, steps)``; times only the workflow, not interpreter imports."""
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, encoding="utf-8") as f:
        inp = json.load(f)
    tools = PageTools(inp["pages"])
    out: dict[str, Any] = {"framework": framework(), "ok": False}
    t0 = time.perf_counter()
    try:
        final, steps = run(inp, tools)
        out.update(ok=True, final=jsonable(final), steps=jsonable(steps))
    except BaseException as exc:  # noqa: BLE001 - every failure must be reported, not raised
        out["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:4000],
            "traceback": traceback.format_exc()[-8000:],
        }
    out["agent_latency_s"] = time.perf_counter() - t0
    out["tool_calls"] = tools.calls
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, default=str)


def dist_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)
