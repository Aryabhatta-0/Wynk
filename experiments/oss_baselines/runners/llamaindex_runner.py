"""LlamaIndex - Starter Agentic RAG (FunctionAgent, tool-based retrieval).

The official starter tutorial agent, unchanged in shape:

    agent = FunctionAgent(tools=[...], llm=..., system_prompt="You are a helpful assistant ...")
    response = await agent.run(user_msg)

Retrieval is agentic: the agent decides which source pages to retrieve through tools. There is
no vector index (the pages are a few hundred bytes each; whole pages fit in context).

Differences from the starter (all documented in manifest.json):
  * llm: ``OpenAILike`` (the documented class for OpenAI-compatible servers) pointed at the
    shared endpoint via the metering proxy, temperature 0, max_tokens 1024, function calling on,
    context_window = the model's real 262144 (the OpenAILike default of 3900 would truncate);
  * tools: the shared ``list_pages`` / ``read_page`` page tools;
  * system_prompt: starter wording with the capability changed to reading source pages.
Everything else is the library default (streaming, max iterations, memory).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

# framework imports happen before timing starts (not part of the workflow latency)
from llama_index.core.agent.workflow import FunctionAgent  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant that can read the source pages for a task."
CONTEXT_WINDOW = 262_144  # google/gemma-4-31b-it context_length reported by the backend


def framework() -> dict[str, str]:
    return {"name": "llama-index-core", "version": _common.dist_version("llama-index-core")}


def run(inp: dict, pages: _common.PageTools):
    m = inp["model"]

    def list_pages() -> str:
        """List the ids of the source pages available for this task."""
        return pages.list_pages()

    def read_page(page_id: str) -> str:
        """Return the full text of one source page. page_id: id of the page to read, as
        returned by list_pages."""
        return pages.read_page(page_id)

    llm = OpenAILike(
        model=m["model"],
        api_base=m["base_url"],
        api_key=m["api_key"],
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=CONTEXT_WINDOW,
        temperature=m["temperature"],
        max_tokens=m["max_tokens"],
        timeout=m["timeout_s"],
        max_retries=m["max_retries"],
    )
    agent = FunctionAgent(tools=[list_pages, read_page], llm=llm, system_prompt=SYSTEM_PROMPT)

    async def go():
        return await agent.run(user_msg=inp["task_prompt"])

    response = asyncio.run(go())
    return str(response), {
        "tool_calls": [
            {"tool": t.tool_name, "kwargs": t.tool_kwargs}
            for t in getattr(response, "tool_calls", []) or []
        ]
    }


if __name__ == "__main__":
    _common.main(framework, run)
