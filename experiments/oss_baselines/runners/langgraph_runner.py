"""LangGraph - Basic ReAct Agent (P2).

The prebuilt tool-calling ReAct agent from the LangGraph quickstart, unchanged in shape:

    agent = create_react_agent(model, tools=[...], prompt="You are a helpful assistant")
    agent.invoke({"messages": [{"role": "user", "content": query}]})

Graph: START -> LLM -> (tool call? -> tools -> LLM) -> END.

Differences from the quickstart (all documented in manifest.json):
  * model: ``ChatOpenAI`` pointed at the shared endpoint via the metering proxy, temperature 0,
    max_tokens 1024;
  * tools: the shared ``list_pages`` / ``read_page`` page tools.
Everything else is the library default (recursion limit, no checkpointer).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

# framework imports happen before timing starts (not part of the workflow latency)
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402


def framework() -> dict[str, str]:
    return {"name": "langgraph", "version": _common.dist_version("langgraph")}


def run(inp: dict, pages: _common.PageTools):
    m = inp["model"]

    def list_pages() -> str:
        """List the ids of the source pages available for this task."""
        return pages.list_pages()

    def read_page(page_id: str) -> str:
        """Return the full text of one source page. page_id: id of the page to read, as
        returned by list_pages."""
        return pages.read_page(page_id)

    model = ChatOpenAI(
        model=m["model"],
        base_url=m["base_url"],
        api_key=m["api_key"],
        temperature=m["temperature"],
        max_tokens=m["max_tokens"],
        timeout=m["timeout_s"],
        max_retries=m["max_retries"],
    )
    agent = create_react_agent(
        model, tools=[list_pages, read_page], prompt="You are a helpful assistant"
    )
    out = agent.invoke({"messages": [{"role": "user", "content": inp["task_prompt"]}]})
    messages = out["messages"]
    return messages[-1].content, {"messages": [msg.model_dump(mode="json") for msg in messages]}


if __name__ == "__main__":
    _common.main(framework, run)
