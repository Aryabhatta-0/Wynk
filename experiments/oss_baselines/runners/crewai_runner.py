"""CrewAI - Starter Research Workflow.

Built from the official ``crewai create crew`` template (crewai_cli/templates/crew): its
``researcher`` agent (role / goal / backstory verbatim, ``{topic}`` filled in) running its
``research_task`` in a sequential Crew. Only the research half of the template is used - the
template's second agent writes a markdown report file, which is not this task's output.

Differences from the template (all documented in manifest.json):
  * ``{topic}`` = "the provided source documents" (one generic value for every task);
  * task description = the shared task prompt; expected_output = the shared JSON answer format;
  * llm: ``LLM(model="openai/<gemma>")`` pointed at the shared endpoint via the metering proxy,
    temperature 0, max_tokens 1024;
  * tools: the shared ``list_pages`` / ``read_page`` page tools (template ships a placeholder
    ``MyCustomTool``; the generated project's README points at SerperDevTool for web search).
Everything else is the library default (no memory, no planning, no manager, default max_iter).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# framework imports happen before timing starts (not part of the workflow latency)
from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

TOPIC = "the provided source documents"
EXPECTED_OUTPUT = "ONLY the JSON object described in the task (answer + evidence), nothing else."


def framework() -> dict[str, str]:
    return {"name": "crewai", "version": _common.dist_version("crewai")}


def run(inp: dict, pages: _common.PageTools):
    m = inp["model"]

    @tool("list_pages")
    def list_pages() -> str:
        """List the ids of the source pages available for this task."""
        return pages.list_pages()

    @tool("read_page")
    def read_page(page_id: str) -> str:
        """Return the full text of one source page. page_id: id of the page to read, as
        returned by list_pages."""
        return pages.read_page(page_id)

    llm = LLM(
        model=f"openai/{m['model']}",
        base_url=m["base_url"],
        api_key=m["api_key"],
        temperature=m["temperature"],
        max_tokens=m["max_tokens"],
        timeout=m["timeout_s"],
        max_retries=m["max_retries"],
    )
    researcher = Agent(
        role=f"{TOPIC} Senior Data Researcher",
        goal=f"Uncover cutting-edge developments in {TOPIC}",
        backstory=(
            "You're a seasoned researcher with a knack for uncovering the latest "
            f"developments in {TOPIC}. Known for your ability to find the most relevant "
            "information and present it in a clear and concise manner."
        ),
        tools=[list_pages, read_page],
        llm=llm,
        verbose=True,
    )
    research_task = Task(
        description=inp["task_prompt"], expected_output=EXPECTED_OUTPUT, agent=researcher
    )
    crew = Crew(
        agents=[researcher], tasks=[research_task], process=Process.sequential, verbose=True
    )
    result = crew.kickoff()
    usage = getattr(result, "token_usage", None)
    return result.raw, {
        "tasks_output": [t.model_dump(mode="json") for t in result.tasks_output],
        "framework_token_usage": usage.model_dump(mode="json") if usage is not None else None,
    }


if __name__ == "__main__":
    _common.main(framework, run)
