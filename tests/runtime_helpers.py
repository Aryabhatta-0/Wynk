"""Test doubles/fixtures for runtime tests.

``ScriptedModel`` is a TEST DOUBLE implementing the ModelClient protocol with canned,
role-based JSON. Production code never contains canned model output.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.results import RunVersions
from core.run_contract import ExecutionTask
from core.stages import GatherSource
from runtime.budget_guard import BudgetGuard
from runtime.executors.base import RunContext
from runtime.gemma_client import GenerationRequest, GenerationResponse
from tests.conftest import make_task

PAGES = {
    "p1": (
        "# France\n\nParis is the capital of France. It lies on the Seine.\n\n"
        "The population is about 2,100,000 people.\n"
    ),
    "p2": "# Germany\n\nBerlin is the capital of Germany.\n",
    "p3": "Unrelated page about bananas and farming.\n",
}
QUOTE = "Paris is the capital of France"


def write_snapshot(root: Path, snapshot_id: str = "snap-001", extra_pages: int = 0) -> Path:
    d = root / snapshot_id
    (d / "api").mkdir(parents=True)
    for name, text in PAGES.items():
        (d / f"{name}.txt").write_text(text, encoding="utf-8")
    for i in range(extra_pages):
        (d / f"x{i}.txt").write_text(f"filler page {i}\n", encoding="utf-8")
    (d / "api" / "country.json").write_text(
        json.dumps({"country": "France", "capital": "Paris"}), encoding="utf-8"
    )
    return root


def versions() -> RunVersions:
    return RunVersions(
        model_hash="m",
        prompt_template_version="p",
        benchmark_hash="b",
        compiler_version="c",
        grammar_version="g",
    )


def make_ctx(task: ExecutionTask | None = None, model=None) -> RunContext:
    task = task or make_task()
    return RunContext(
        task=task, seed=1, trial=0, versions=versions(), guard=BudgetGuard(task.caps), model=model
    )


def all_sources_task(**kw) -> ExecutionTask:
    return make_task(allowed_sources=(GatherSource.FETCH, GatherSource.API, GatherSource.JEV), **kw)


class ScriptedModel:
    """Deterministic fake backend. ``extract_quotes`` is consumed one entry per EXTRACT call,
    which lets tests simulate a bad first attempt followed by a good retry."""

    model_hash = "scripted-model-v1"

    def __init__(self, extract_quotes: list[str] | None = None, answer: str = "Paris") -> None:
        self.extract_quotes = list(extract_quotes if extract_quotes is not None else [QUOTE])
        self.answer = answer
        self.requests: list[GenerationRequest] = []
        self.tokens = (100, 20)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        template = request.prompt_template_id
        if template.startswith("extract"):
            quote = (
                self.extract_quotes.pop(0)
                if len(self.extract_quotes) > 1
                else (self.extract_quotes[0])
            )
            body = {
                "facts": [
                    {"field": "capital", "value": self.answer, "page_id": "p1", "quote": quote}
                ]
            }
        elif template.startswith("reason"):
            body = {"facts": [{"field": "capital", "value": self.answer, "quote": QUOTE}]}
        else:
            body = {"answer": {"capital": self.answer}, "citations": {"capital": [0]}}
        text = json.dumps(body)
        return GenerationResponse(
            text=text,
            parsed=body,
            prompt_tokens=self.tokens[0],
            completion_tokens=self.tokens[1],
            model_hash=self.model_hash,
        )


class FailingModel:
    model_hash = "failing"

    async def generate(self, request):
        from runtime.gemma_client import ModelUnavailableError

        raise ModelUnavailableError("backend down")


def build_runner(genome, task, root: Path, model, key=None):
    """A StageRunner over the MVP executors (no MAF involved)."""
    from compiler.dag import compile_genome
    from runtime.executors.registry import default_executors
    from runtime.sources import DirectoryApiSource, DirectorySnapshotSource
    from runtime.stage_runner import StageRunner

    dag = compile_genome(genome)
    execs = default_executors(DirectorySnapshotSource(root), DirectoryApiSource(root))
    return dag, StageRunner(dag, execs, make_ctx(task, model))


async def drive(dag, runner, task):
    """Sequential stand-in for the MAF graph: feeds each node's output to the next."""
    payload = task
    for node in dag.nodes:
        payload = await runner.execute_node(node, payload)
        if payload is None:
            break
    from core.results import RunKey

    key = RunKey(
        genome_hash=dag.genome_hash,
        task_id=task.id,
        contract_hash=task.contract_hash,
        trial=0,
        seed=1,
        versions=versions(),
    )
    return runner.result(key)
