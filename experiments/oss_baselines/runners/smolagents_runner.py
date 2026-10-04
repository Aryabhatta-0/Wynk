"""Hugging Face smolagents - Starter Agent.

The README quickstart, unchanged in shape:

    model = ...
    agent = CodeAgent(tools=[WebSearchTool()], model=model)
    agent.run(query)

Differences from the quickstart (all documented in manifest.json):
  * model: ``OpenAIServerModel`` (the documented class for OpenAI-compatible servers) pointed at
    the shared Gemma endpoint via the metering proxy, temperature 0, max_tokens 1024;
  * tools: the shared ``list_pages`` / ``read_page`` page tools instead of ``WebSearchTool``
    (CONTROLLED_SOURCE: every system reads the same frozen pages).
Every other CodeAgent setting is the library default (max_steps, executor, prompts, imports).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

# framework imports happen before timing starts (not part of the workflow latency)
from smolagents import CodeAgent, OpenAIServerModel, tool  # noqa: E402


def framework() -> dict[str, str]:
    return {"name": "smolagents", "version": _common.dist_version("smolagents")}


def run(inp: dict, pages: _common.PageTools):
    m = inp["model"]

    @tool
    def list_pages() -> str:
        """List the ids of the source pages available for this task."""
        return pages.list_pages()

    @tool
    def read_page(page_id: str) -> str:
        """Return the full text of one source page.

        Args:
            page_id: Id of the page to read, as returned by list_pages.
        """
        return pages.read_page(page_id)

    model = OpenAIServerModel(
        model_id=m["model"],
        api_base=m["base_url"],
        api_key=m["api_key"],
        client_kwargs={"max_retries": m["max_retries"], "timeout": m["timeout_s"]},
        temperature=m["temperature"],
        max_tokens=m["max_tokens"],
    )
    agent = CodeAgent(tools=[list_pages, read_page], model=model)
    final = agent.run(inp["task_prompt"])
    steps = [s.dict() for s in agent.memory.steps]
    return final, {"n_steps": len(agent.memory.steps), "memory": steps}


if __name__ == "__main__":
    _common.main(framework, run)
