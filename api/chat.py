"""wynk chat backend: one question -> a small ant colony of real workflow runs on the model.

    python -m api.chat --env-file .env          # serves http://127.0.0.1:8787 (the UI proxies /api)

``POST /api/chat {"query": "..."}`` streams server-sent events in the shape the UI reads
(``ui/src/lib/chat.ts``):
route -> plan -> ants -> stage... -> score... -> pick -> answer... -> done.

* The question is routed to a fact sheet in the frozen benchmark library (company name match);
  anything else is refused without a model call.
* The facts to look for come from fixed keyword rules (the model never plans or judges).
* Ants: the best ant colony workflow recorded on fact sheets, plus explorers proposed by MMAS
  from the colony's pheromone (kept in memory; every winner's path is reinforced).
* Every ant runs through the real runtime (``WorkflowRunner`` stages on the model); stage progress
  is streamed as it happens.
* Results are checked without ground truth: did the run finish, does every cited span verify
  against the snapshot bytes, how many tokens. The best one is kept.
* The model (synthesize role) phrases the reply from the winner's values and quotes only.

``GET /api/health`` reports whether a model backend is configured. The key never leaves the
server and is never logged.

``/api/v1/...`` is the product API (projects, dataset upload / registration / splits), served by
``api.product`` and mounted here so the UI's ``/api`` proxy reaches it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from api import product
from benchmarks.legacy_adapter import legacy_adhoc_task
from benchmarks.loader import load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.genome import Genome
from core.results import ExecutionResult
from core.run_contract import ExecutionTask
from core.task_spec import AnswerField, AnswerSchema, RuntimeTask, TaskClass
from evaluation.evidence import SnapshotEvidenceVerifier
from experiments.real_runtime import build_runner
from optimizers.aco_mmas import MMASACO
from optimizers.base import SearchContext
from optimizers.construct import path_edges
from runtime.backends.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
from runtime.executors.base import RunContext
from runtime.model_client import GenerationRequest, ModelRole, RegisteredModelClient
from runtime.runner import InadmissibleGenome, WorkflowRunner
from runtime.stage_runner import StageRunner

ELITE_GENOME = Path("experiments/results/real-gemma4/A/best_aco_genome.json")
EXPLORERS = 2

# keyword -> answer field (name, type), checked in this order
FIELDS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(ceo|chief executive|led by|boss|head)\b", re.I), "ceo", "string"),
    (
        re.compile(r"\b(employees?|staff|headcount|people|workforce)\b", re.I),
        "employees",
        "integer",
    ),
    (re.compile(r"\b(revenue|sales|turnover|income)\b", re.I), "revenue", "number"),
    (re.compile(r"\b(founded|founding|established|started|year)\b", re.I), "founded", "integer"),
    (re.compile(r"\b(hq|headquarter\w*|based|located|city)\b", re.I), "hq", "string"),
    (re.compile(r"\b(products?|services?|sell|make|offer)\b", re.I), "products", "string_list"),
    (re.compile(r"\b(fiscal|financial year|fy)\b", re.I), "fy_end", "date"),
]
DEFAULT_FIELDS = [("ceo", "string"), ("founded", "integer"), ("hq", "string")]


def load_env_file(path: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


class Library:
    """The fact sheets a question can be answered from (benchmark class A snapshots)."""

    def __init__(self) -> None:
        self.specs = {
            i: s for i, s in load_task_specs().items() if s.runtime.task_class is TaskClass.A
        }
        self.store = SnapshotStore()
        self.sources: list[tuple[str, str, str]] = []  # (snapshot id, company, template task id)
        for task_id, spec in sorted(self.specs.items()):
            sid = spec.runtime.snapshot_id
            pages = self.store.pages(sid).pages
            name = pages[0].content.split(" - ")[0].strip() if pages else sid
            self.sources.append((sid, name, task_id))

    def route(self, query: str) -> tuple[str, str, str, str] | None:
        q = query.lower()
        best = None
        for sid, name, task_id in self.sources:
            words = [w for w in re.split(r"[\s&]+", name.lower()) if len(w) > 2]
            hits = [w for w in words if w in q]
            if hits and (best is None or len(hits) > len(best[3].split())):
                best = (sid, name, task_id, " ".join(hits))
        return best


def plan_fields(query: str) -> list[tuple[str, str]]:
    fields = [(name, kind) for rx, name, kind in FIELDS if rx.search(query)]
    return fields or DEFAULT_FIELDS


class Colony:
    """The chat's ant colony: MMAS pheromone shared across questions, warmed by the recorded
    best workflow and reinforced by every winner."""

    def __init__(self, elite: Genome | None) -> None:
        self.aco = MMASACO()
        self.elite = elite
        self.questions = 0
        self.lock = threading.Lock()
        if elite is not None:
            self.reinforce(elite, 1.0)

    def reinforce(self, genome: Genome, quality: float) -> None:
        cfg = self.aco.config
        self.aco.epoch += 1
        delta = max(0.1, min(1.0, quality)) * cfg.rho * cfg.tau_max
        for edge in path_edges(genome):
            value = min(cfg.tau_max, self.aco.pheromone(edge) + delta)
            self.aco._edges[edge] = (max(cfg.tau_min, value), self.aco.epoch)  # noqa: SLF001

    def trail(self, genome: Genome) -> float:
        edges = path_edges(genome)
        return sum(self.aco.pheromone(e) for e in edges) / len(edges)

    def ants(self, task: ExecutionTask, checker) -> list[tuple[str, Genome]]:
        with self.lock:
            self.questions += 1
            ctx = SearchContext(
                contract=task.contract, checker=checker, seed=self.questions, round=0
            )
            out: list[tuple[str, Genome]] = []
            if self.elite is not None and not checker.check(
                self.elite, task.contract, complete=True
            ):
                out.append(("elite", self.elite))
            seen = {g.genome_hash for _, g in out}
            for g in self.aco.propose(EXPLORERS + 2, ctx):
                if g.genome_hash not in seen and len(out) < EXPLORERS + 1:
                    out.append(("explorer", g))
                    seen.add(g.genome_hash)
            return out


def stage_view(genome: Genome) -> list[dict[str, Any]]:
    out = []
    for s in genome.stages:
        d = s.model_dump(mode="json")
        kind = d.pop("kind")
        order = ("source", "mode", "method", "on_failure", "min_support")
        opts = [str(d[k]).replace("_", " ") for k in order if k in d]
        out.append({"kind": kind, "options": opts})
    return out


class ProgressStageRunner(StageRunner):
    """The runtime's StageRunner, reporting each stage as it starts and ends."""

    def __init__(self, dag, executors, ctx, on_stage) -> None:
        super().__init__(dag, executors, ctx)
        self.on_stage = on_stage

    async def execute_node(self, node, payload):
        self.on_stage(node.stage_index, "running", 0, None)
        before = self.ctx.guard.usage.tokens
        out = await super().execute_node(node, payload)
        used = self.ctx.guard.usage.tokens - before
        if out is None:
            self.on_stage(
                node.stage_index, "failed", used, self.failure.message if self.failure else "failed"
            )
        else:
            self.on_stage(node.stage_index, "ok", used, None)
        return out


def run_with_progress(
    runner: WorkflowRunner, genome: Genome, task: ExecutionTask, on_stage
) -> ExecutionResult:
    """``WorkflowRunner.run``, with stage progress (same checks, same executors, same result)."""
    from core.violations import ViolationCode
    from runtime.budget_guard import BudgetGuard
    from runtime.maf_nodes import Envelope, StageNode

    key = runner.run_key(genome, task)
    violations = runner.checker.check(genome, task.contract, complete=True)
    if any(v.code != ViolationCode.BUDGET_INFEASIBLE for v in violations):
        raise InadmissibleGenome(tuple(violations))
    if violations:
        return runner._static_breach(key, genome, task)  # noqa: SLF001
    dag = runner.compiler.to_dag(genome)
    ctx = RunContext(
        task=task,
        seed=0,
        trial=0,
        versions=key.versions,
        guard=BudgetGuard(task.caps),
        model=runner.model,
    )
    stages = ProgressStageRunner(dag, runner._executors, ctx, on_stage)  # noqa: SLF001
    nodes = {n.node_id: StageNode(n, stages, is_end=n.node_id == dag.end_node) for n in dag.nodes}
    workflow = runner.compiler.build(dag, nodes)
    asyncio.run(workflow.run(Envelope(task)))
    return stages.result(key)


class Engine:
    def __init__(self, env: dict[str, str]) -> None:
        self.config = OpenAICompatibleConfig.from_env(env)
        self.model = RegisteredModelClient(self.config.entry(), OpenAICompatibleClient(self.config))
        self.library = Library()
        self.runner = build_runner(self.model, self.library.store)
        self.verifier = SnapshotEvidenceVerifier(self.library.store)
        elite = None
        if ELITE_GENOME.is_file():
            elite = Genome.model_validate(
                {"stages": json.loads(ELITE_GENOME.read_text())["stages"]}
            )
        self.colony = Colony(elite)
        self.count = 0

    def answer(self, query: str, emit) -> None:
        hit = self.library.route(query)
        if hit is None:
            names = ", ".join(n for _, n, _ in self.library.sources[:3])
            emit(
                {
                    "type": "refuse",
                    "message": "I could not match that to a company in my library, so I did "
                    f"not run a workflow. Try asking about {names} or another company "
                    "listed above.",
                }
            )
            return
        sid, name, template_id, matched = hit
        template = self.library.specs[template_id].runtime
        pages = [p.page_id for p in self.library.store.pages(sid).pages]
        emit(
            {
                "type": "route",
                "source": {"id": sid, "name": name},
                "matched": matched,
                "pages": pages,
            }
        )

        fields = plan_fields(query)
        emit({"type": "plan", "fields": [{"name": n, "type": t} for n, t in fields]})

        self.count += 1
        # The benchmark library is legacy data: the question becomes a contract at the adapter.
        task = legacy_adhoc_task(
            RuntimeTask(
                id=f"chat-{self.count}",
                task_class=TaskClass.A,
                question=query,
                answer_schema=AnswerSchema(
                    fields=tuple(AnswerField(name=n, type=t) for n, t in fields)
                ),
                caps=template.caps,
                allowed_sources=template.allowed_sources,
                snapshot_id=sid,
            ),
            self.library.store,
        )
        picked = self.colony.ants(task, self.runner.checker)
        ants = [
            {
                "id": i + 1,
                "role": role,
                "stages": stage_view(g),
                "trail": round(self.colony.trail(g), 2),
            }
            for i, (role, g) in enumerate(picked)
        ]
        emit({"type": "ants", "ants": ants})

        def run_ant(i: int, genome: Genome) -> ExecutionResult | None:
            ant = i + 1

            def on_stage(index, status, tokens, message):
                event = {
                    "type": "stage",
                    "ant": ant,
                    "index": index,
                    "status": status,
                    "tokens": tokens,
                }
                if message:
                    event["message"] = message
                emit(event)

            try:
                return run_with_progress(self.runner, genome, task, on_stage)
            except (
                Exception
            ) as exc:  # inadmissible or runtime error: this ant failed, the colony goes on
                emit(
                    {
                        "type": "stage",
                        "ant": ant,
                        "index": 0,
                        "status": "failed",
                        "message": str(exc)[:200],
                    }
                )
                return None

        with ThreadPoolExecutor(max_workers=len(picked)) as pool:
            results = list(pool.map(lambda a: run_ant(*a), enumerate(g for _, g in picked)))

        scored = []
        for i, ((_, genome), res) in enumerate(zip(picked, results, strict=True)):
            completed = bool(res and res.failure is None and res.answer)
            spans = [s for fe in (res.answer.evidence if completed else ()) for s in fe.spans]
            cited = {fe.field for fe in (res.answer.evidence if completed else ())}
            evidence = (
                completed
                and cited >= {n for n, _ in fields}
                and all(self.verifier.is_valid(s, sid) for s in spans)
            )
            tokens = res.budget_usage.tokens if res else 0
            if completed and evidence:
                score = 1.0 + 0.1 * max(0.0, 1 - tokens / task.caps.tokens)
            elif completed:
                score = 0.5 * len(res.answer.values) / len(fields)
            else:
                score = 0.0
            scored.append((score, -tokens, i, genome, res))
            emit(
                {
                    "type": "score",
                    "ant": i + 1,
                    "completed": completed,
                    "evidence": evidence,
                    "tokens": tokens,
                    "score": round(score, 3),
                }
            )

        score, _, best_i, genome, res = max(scored)
        if score <= 0 or res is None or res.answer is None:
            emit({"type": "pick", "ant": best_i + 1, "reason": "No ant finished with an answer."})
            emit(
                {
                    "type": "answer",
                    "delta": "None of the workflows produced an answer I could check, "
                    "so I will not guess. Try asking again.",
                }
            )
            emit({"type": "done", "quotes": []})
            return
        verified = score >= 1.0
        self.colony.reinforce(genome, score / 1.1)
        emit(
            {
                "type": "pick",
                "ant": best_i + 1,
                "reason": "Every quote verified on the page, at the lowest token cost."
                if verified
                else "The most complete answer, but its quotes could not all be verified.",
            }
        )

        quotes = []
        for fe in res.answer.evidence:
            for s in fe.spans:
                page = self.library.store.page(sid, s.page_id)
                if page and self.verifier.is_valid(s, sid):
                    text = page.content[s.char_start : s.char_end].strip()
                    if text and all(q["text"] != text for q in quotes):
                        quotes.append({"page": s.page_id, "text": text})

        reply = self.compose(query, name, res.answer.values, quotes, verified)
        for word in re.split(r"(?<=\s)", reply):
            if word:
                emit({"type": "answer", "delta": word})
        emit({"type": "done", "quotes": quotes})

    def compose(
        self, query: str, company: str, values: dict, quotes: list[dict], verified: bool
    ) -> str:
        facts = json.dumps(values, ensure_ascii=False)
        lines = (
            "\n".join(f'- "{q["text"]}" ({q["page"]})' for q in quotes) or "- (no verified quotes)"
        )
        prompt = (
            f"Question: {query}\n"
            f"Facts extracted from the {company} fact sheet: {facts}\n"
            f"Supporting quotes:\n{lines}\n\n"
            "Answer the question in one to three plain sentences using only these facts. "
            "Do not add anything that is not in the facts. Do not mention JSON or quotes."
        )
        try:
            res = asyncio.run(
                self.model.generate(
                    GenerationRequest(
                        role=ModelRole.SYNTHESIZE,
                        prompt_template_id="chat-reply",
                        prompt_template_version="chat-1",
                        input_text=prompt,
                        seed=0,
                        max_tokens=200,
                    )
                )
            )
            text = res.text.strip()
        except Exception:  # the facts still stand without the phrasing
            text = ""
        if not text:
            text = "; ".join(f"{k}: {v}" for k, v in values.items())
        if not verified:
            text += " (Not every quote behind this could be verified on the page.)"
        return text


def make_handler(engine: Engine | None, problem: str | None, api: product.ProductAPI | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if api is not None and product.is_product_path(self.path):
                product.respond(self, api)
            elif self.path == "/api/health":
                self._json(
                    200,
                    {
                        "ok": engine is not None,
                        "model": engine.config.model if engine else None,
                        "problem": problem,
                    },
                )
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if api is not None and product.is_product_path(self.path):
                product.respond(self, api)
                return
            if self.path != "/api/chat":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                query = str(json.loads(self.rfile.read(length) or b"{}").get("query", "")).strip()
            except json.JSONDecodeError:
                query = ""
            if not query:
                self._json(400, {"error": "query is required"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            lock = threading.Lock()

            def emit(event: dict) -> None:
                with lock:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

            try:
                if engine is None:
                    emit({"type": "error", "message": f"No model backend: {problem}"})
                else:
                    engine.answer(query, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser stopped the run
            except Exception as exc:  # report, never hang the stream
                try:
                    emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]})
                except OSError:
                    pass

        def log_message(self, *args) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="file with WYNK_MODEL_BASE_URL, WYNK_MODEL, WYNK_MODEL_API_KEY (or legacy GEMMA_*)",
    )
    p.add_argument("--port", type=int, default=8787)
    product.add_arguments(p)
    args = p.parse_args(argv)
    engine, problem = None, None
    try:
        engine = Engine(load_env_file(args.env_file))
    except Exception as exc:  # serve /api/health so the UI can say what is missing
        problem = str(exc)
    api = product.api_from_args(args)
    server = product.Server(("127.0.0.1", args.port), make_handler(engine, problem, api))
    model = engine.config.model if engine else f"no model: {problem}"
    print(f"wynk chat on http://127.0.0.1:{args.port} ({model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
