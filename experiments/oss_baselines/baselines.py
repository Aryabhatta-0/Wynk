"""External-baseline interface, shared task prompt, output normalisation and common evaluation.

    ExternalBaseline.run(task, run_index) -> BaselineResult

Every system - the frozen Wynk workflow and each OSS starter workflow - is a
``SubprocessBaseline``: a runner script executed in that system's own interpreter (separate venv
per framework), with the same input, the same per-run timeout, and model traffic through the same
``MeteringProxy``. The adapters only normalise input/output; they never change a workflow.

Output normalisation is deterministic and identical for all external systems: parse the final
JSON (``answer`` + per-field ``evidence`` quotes), then turn quotes into ``EvidenceSpan``s with
Wynk's own ``runtime.spans.locate_quote`` - the same function Wynk's runtime uses. The resulting
``ExecutionResult`` goes through the unchanged ``DeterministicEvaluator``.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from core.canonical import canonical_hash, sha256_hex
from core.evidence import EvidenceSpan, FieldEvidence
from core.payloads import Answer, Page
from core.results import (
    BudgetUsage,
    Evaluation,
    ExecutionMetrics,
    ExecutionResult,
    FailureInfo,
    FailureKind,
    RunKey,
    RunVersions,
    usage_exceeds,
)
from core.task_spec import Caps, RuntimeTask, TaskSpec
from evaluation.gate import DeterministicEvaluator
from evaluation.schema import validate_answer
from experiments.oss_baselines.proxy import MeteringProxy, summarize_calls
from runtime.spans import locate_quote, supports

HERE = Path(__file__).resolve().parent
RUNNERS = HERE / "runners"
REPO = HERE.parents[1]
# outside every Python package (tests scan package dirs); gitignored
VENVS = REPO / ".oss-venvs"

TASK_PROMPT_VERSION = "oss-task-prompt/2"
MODE = "CONTROLLED_SOURCE"
RUN_TIMEOUT_S = 360.0  # hang guard (2x the 180 s wall cap); applied identically to every system
UNCAPPED = Caps(tokens=10**12, wall_time_s=1e12, tool_calls=10**12, retries=10**12)


class RunStatus(StrEnum):
    SUCCESS = "SUCCESS"  # in-budget, schema-valid answer with valid evidence for every field
    TIMEOUT = "TIMEOUT"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


NO_ANSWER_STATUSES = {RunStatus.TIMEOUT, RunStatus.RUNTIME_ERROR, RunStatus.INVALID_OUTPUT}


@dataclass(frozen=True)
class ModelSettings:
    """Identical generation settings for every system (Wynk's own client: temperature 0,
    max_tokens <= 1024 per call, 120 s per-call timeout, 3 retries on 429/5xx)."""

    model: str
    revision: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 120.0
    max_retries: int = 3

    @property
    def model_hash(self) -> str:  # same identity Wynk's OpenAICompatibleClient reports
        return sha256_hex(f"{self.model}@{self.revision}")


@dataclass(frozen=True)
class SystemSpec:
    name: str  # short id
    label: str  # precise label used everywhere in results
    kind: str  # "wynk" | "external"
    runner: str  # file in runners/
    venv: str | None  # .oss-venvs/<venv>; None = the main Wynk interpreter
    priority: str = "P0"

    def python(self) -> str:
        import sys

        if self.venv is None:  # Wynk: the repo's own env (needs the [maf] extra)
            own = REPO / ".venv" / ("Scripts/python.exe" if _windows() else "bin/python")
            return str(own) if own.is_file() else sys.executable
        exe = VENVS / self.venv / ("Scripts/python.exe" if _windows() else "bin/python")
        return str(exe)


def _windows() -> bool:
    import os

    return os.name == "nt"


SYSTEMS: dict[str, SystemSpec] = {
    s.name: s
    for s in (
        SystemSpec(
            "smolagents",
            "smolagents Starter Agent (CodeAgent)",
            "external",
            "smolagents_runner.py",
            "smolagents",
        ),
        SystemSpec(
            "crewai",
            "CrewAI Starter Research Workflow (1-agent sequential crew)",
            "external",
            "crewai_runner.py",
            "crewai",
        ),
        SystemSpec(
            "llamaindex",
            "LlamaIndex Starter Agentic RAG (FunctionAgent, tool retrieval)",
            "external",
            "llamaindex_runner.py",
            "llamaindex",
        ),
        SystemSpec(
            "langgraph",
            "LangGraph Basic ReAct Agent (create_react_agent)",
            "external",
            "langgraph_runner.py",
            "langgraph",
            priority="P2",
        ),
        SystemSpec("wynk", "Wynk Optimized Workflow (frozen)", "wynk", "wynk_runner.py", None),
    )
}


# --- shared task prompt ------------------------------------------------------------------------


def _fields(task: RuntimeTask) -> str:
    # same rendering as Wynk's runtime prompts (runtime/prompts/templates.py::_fields)
    return "\n".join(
        f"- {f.name} ({f.type.value}{', required' if f.required else ''})"
        for f in task.answer_schema.fields
    )


def task_prompt(task: RuntimeTask) -> str:
    """The user query every external system receives. Carries exactly what Wynk's runtime
    prompts carry (question, answer fields + JSON types, verbatim-quote evidence requirement)."""
    return (
        f"{task.question}\n\n"
        "Use the available tools to read the source pages for this task (list_pages, "
        "read_page). Answer using only information stated in those pages.\n\n"
        f"Answer fields (use these JSON types):\n{_fields(task)}\n\n"
        "Final answer: return ONLY a JSON object of this form:\n"
        '{"answer": {<field>: <value>, ...}, "evidence": {<field>: [{"page_id": <page id>, '
        '"quote": <exact text copied from that page>}, ...], ...}}\n'
        "Give at least one evidence entry per answer field (several when a value combines "
        "several places on a page). Each quote must be copied verbatim from the page so it can "
        "be located."
    )


# --- normalisation -----------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _loads(text: str) -> Any:
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def _objects(text: str) -> list[str]:
    """Every balanced ``{...}`` substring (outermost first), ignoring braces inside strings."""
    out: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, quote, esc = 0, None, False
        for i in range(start, len(text)):
            c = text[i]
            if quote:
                esc = (c == "\\") and not esc
                if c == quote and not esc:
                    quote = None
                continue
            if c in "\"'":
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
    return out


def parse_final(final: Any, field_names: set[str]) -> dict[str, Any] | None:
    """Final output -> ``{"answer": {...}, "evidence": {...}}`` or None (INVALID_OUTPUT).

    Lenient and identical for every system: accepts a dict, JSON / Python-literal text, fenced
    code, JSON embedded in prose (last candidate wins), and a bare ``{field: value}`` object."""
    candidates: list[Any] = []
    if isinstance(final, dict):
        candidates.append(final)
    elif isinstance(final, str):
        text = final.strip()
        blobs = [text, *_FENCE.findall(text), *_objects(text)]
        candidates += [v for v in (_loads(b.strip()) for b in blobs) if isinstance(v, dict)]
    best = None
    for obj in candidates:
        if isinstance(obj.get("answer"), dict):
            best = obj
        elif best is None and field_names & set(obj):
            best = {"answer": {k: obj[k] for k in obj if k in field_names}}
    return best


def _quotes(entry: Any) -> list[tuple[str, str | None]]:
    if isinstance(entry, str):
        return [(entry, None)]
    if isinstance(entry, dict) and isinstance(entry.get("quote"), str):
        pid = entry.get("page_id")
        return [(entry["quote"], pid if isinstance(pid, str) else None)]
    if isinstance(entry, list):
        return [q for e in entry for q in _quotes(e)]
    return []


_ELLIPSIS = re.compile(r"\.\.\.|…")
_MIN_SEGMENT = 8  # non-whitespace chars; shorter excerpt fragments are not citations


def _ws_free(text: str) -> tuple[str, list[int]]:
    """``text`` without whitespace, plus each kept char's offset in ``text``."""
    idx = [i for i, c in enumerate(text) if not c.isspace()]
    return "".join(text[i] for i in idx), idx


def locate_citation(
    quote: str, pages: dict[str, Page], page_id: str | None, *, lenient: bool = True
) -> tuple[list[EvidenceSpan], str | None]:
    """Quote -> spans on the original pages, and how it matched (``None`` = not located).

    1. ``exact``: Wynk's own ``runtime.spans.locate_quote`` (what Wynk's runtime does).
    2. ``ws_insensitive``: the quote split on ellipses ("...") into excerpts; EVERY excerpt
       (>= 8 non-whitespace chars) must occur in one page ignoring whitespace only (re-flowed
       JSON / line breaks still count; any changed character does not). Each excerpt becomes one
       span over the original text. Deliberately more lenient than Wynk's runtime, so formatting
       never costs an external system its evidence."""
    span = locate_quote(quote, tuple(pages.values()), pages, page_id)
    if span is not None:
        return [span], "exact"
    if not lenient:  # sensitivity analysis: Wynk-runtime strictness only
        return [], None
    segments = [seg for seg in _ELLIPSIS.split(quote) if seg.strip()]
    if not segments:
        return [], None
    for page in sorted(pages.values(), key=lambda p: p.page_id != page_id):
        flat, idx = _ws_free(page.content)
        spans: list[EvidenceSpan] = []
        for seg in segments:
            needle, _ = _ws_free(seg)
            at = flat.find(needle) if len(needle) >= _MIN_SEGMENT else -1
            if at < 0:
                break
            spans.append(page.span(idx[at], idx[at + len(needle) - 1] + 1))
        else:
            return spans, "ws_insensitive"
    return [], None


def normalize_answer(
    parsed: dict[str, Any] | None,
    task: RuntimeTask,
    pages: dict[str, Page],
    *,
    lenient: bool = True,
) -> tuple[Answer | None, dict[str, Any]]:
    """Parsed final -> ``Answer`` (values restricted to schema fields, spans located from the
    cited quotes). Also returns the raw quotes per field, and how each matched, for the audit."""
    if parsed is None:
        return None, {}
    allowed = task.answer_schema.field_names
    values = {k: v for k, v in parsed["answer"].items() if k in allowed}
    raw_ev = parsed.get("evidence") or parsed.get("citations") or {}
    raw_ev = raw_ev if isinstance(raw_ev, dict) else {}
    evidence: list[FieldEvidence] = []
    quotes: dict[str, Any] = {}
    for name in values:
        spans: list[EvidenceSpan] = []
        quotes[name] = []
        for q, pid in _quotes(raw_ev.get(name)):
            found, how = locate_citation(q, pages, pid, lenient=lenient)
            quotes[name].append({"quote": q, "page_id": pid, "matched": how})
            spans += [s for s in found if s not in spans]
        if spans:
            evidence.append(FieldEvidence(field=name, spans=tuple(spans)))
    return Answer(values=values, evidence=tuple(evidence)), quotes


# --- results -----------------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """One execution of one system on one task (everything needed to audit it)."""

    system: str
    label: str
    task_id: str
    task_class: str
    run_index: int
    mode: str
    status: RunStatus
    latency_s: float
    execution: ExecutionResult
    telemetry: dict[str, Any]
    raw_final: Any = None
    quotes: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    framework: dict[str, Any] | None = None
    timed_out: bool = False
    raw_dir: str | None = None
    proxy_calls: list[dict[str, Any]] = field(default_factory=list)


class ExternalBaseline(Protocol):
    spec: SystemSpec

    def run(self, task: RuntimeTask, run_index: int) -> BaselineResult: ...


class SubprocessBaseline:
    """Runs ``spec.runner`` in ``spec.python()``; the adapter never changes the workflow."""

    def __init__(
        self,
        spec: SystemSpec,
        proxy: MeteringProxy,
        settings: ModelSettings,
        pages_for: Any,  # (snapshot_id) -> dict[page_id, Page]
        benchmark_hash: str,
        raw_root: Path,
        wynk: dict[str, Any] | None = None,
        timeout_s: float = RUN_TIMEOUT_S,
    ) -> None:
        self.spec = spec
        self.proxy = proxy
        self.settings = settings
        self.pages_for = pages_for
        self.benchmark_hash = benchmark_hash
        self.raw_root = raw_root
        self.wynk = wynk
        self.timeout_s = timeout_s

    def run_id(self, task: RuntimeTask, run_index: int) -> str:
        return f"{self.spec.name}.{task.id}.r{run_index}"

    def build_input(self, task: RuntimeTask, run_index: int, base_url: str) -> dict[str, Any]:
        s = self.settings
        pages = self.pages_for(task.snapshot_id)
        inp: dict[str, Any] = {
            "run_id": self.run_id(task, run_index),
            "task": task.model_dump(mode="json"),  # RuntimeTask only: no ground truth
            "task_prompt": task_prompt(task),
            "pages": {pid: p.content for pid, p in pages.items()},
            "model": {
                "base_url": base_url,
                "api_key": "via-metering-proxy",  # the proxy injects the real key
                "model": s.model,
                "temperature": s.temperature,
                "max_tokens": s.max_tokens,
                "timeout_s": s.timeout_s,
                "max_retries": s.max_retries,
            },
        }
        if self.wynk is not None:
            inp["wynk"] = {**self.wynk, "trial": run_index, "seed": run_index}
        return inp

    def run(self, task: RuntimeTask, run_index: int) -> BaselineResult:
        run_id = self.run_id(task, run_index)
        raw_dir = self.raw_root / run_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        in_path, out_path = raw_dir / "input.json", raw_dir / "output.json"
        out_path.unlink(missing_ok=True)
        in_path.write_text(
            json.dumps(self.build_input(task, run_index, self.proxy.base_url(run_id)), indent=1),
            encoding="utf-8",
        )
        cmd = [self.spec.python(), str(RUNNERS / self.spec.runner), str(in_path), str(out_path)]
        timed_out = False
        t0 = time.perf_counter()
        with (
            (raw_dir / "stdout.log").open("wb") as so,
            (raw_dir / "stderr.log").open("wb") as se,
        ):
            env = _child_env()
            try:
                subprocess.run(
                    cmd,
                    stdout=so,
                    stderr=se,
                    stdin=subprocess.DEVNULL,
                    timeout=self.timeout_s,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
        wall = time.perf_counter() - t0
        out: dict[str, Any] = {}
        if out_path.is_file():
            out = json.loads(out_path.read_text(encoding="utf-8"))
        elif not timed_out:
            tail = (raw_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
            out = {"ok": False, "error": {"type": "RunnerCrashed", "message": tail}}
        calls = self.proxy.calls(run_id)
        (raw_dir / "proxy_calls.json").write_text(
            json.dumps([c.as_dict() for c in calls], indent=1, default=str), encoding="utf-8"
        )
        latency = self.timeout_s if timed_out else float(out.get("agent_latency_s") or wall)
        return self._result(task, run_index, out, calls, latency, timed_out, raw_dir)

    # --- normalise one runner output into an ExecutionResult ----------------------------------
    def _result(self, task, run_index, out, calls, latency, timed_out, raw_dir) -> BaselineResult:
        pages = self.pages_for(task.snapshot_id)
        tel = summarize_calls(calls)
        error = out.get("error")
        quotes: dict[str, Any] = {}
        if self.spec.kind == "wynk":
            execution = self._wynk_execution(task, run_index, out, tel, latency, timed_out)
        else:
            parsed = parse_final(out.get("final"), set(task.answer_schema.field_names))
            answer, quotes = normalize_answer(parsed, task, pages) if out.get("ok") else (None, {})
            n_tools = len(out.get("tool_calls") or [])
            failure = None
            if timed_out:
                failure = FailureInfo(kind=FailureKind.EXECUTOR_ERROR, message="run timed out")
            elif error:
                failure = FailureInfo(
                    kind=FailureKind.EXECUTOR_ERROR,
                    message=f"{error.get('type')}: {error.get('message', '')}"[:2000],
                )
            elif answer is None:
                failure = FailureInfo(kind=FailureKind.NO_ANSWER, message="no parseable answer")
            execution = ExecutionResult(
                key=self._key(task, run_index),
                answer=answer,
                metrics=ExecutionMetrics(
                    model_calls=tel["llm_calls"],
                    prompt_tokens=tel["prompt_tokens"] or 0,
                    completion_tokens=tel["completion_tokens"] or 0,
                    pages_fetched=sum(
                        1 for c in out.get("tool_calls") or [] if c.get("tool") == "read_page"
                    ),
                ),
                budget_usage=BudgetUsage(
                    tokens=_tokens(tel),
                    wall_time_s=latency,
                    tool_calls=n_tools,
                ),
                failure=failure,
            )
        status = classify(execution, task, pages, timed_out=timed_out, errored=bool(error))
        telemetry = {
            **tel,
            "tool_calls": execution.budget_usage.tool_calls,
            "retries": execution.budget_usage.retries,
            "workflow_steps": _steps(self.spec.kind, out),
            "tokens_measured_by": "metering-proxy (backend-reported usage)",
        }
        return BaselineResult(
            system=self.spec.name,
            label=self.spec.label,
            task_id=task.id,
            task_class=task.task_class.value,
            run_index=run_index,
            mode=MODE,
            status=status,
            latency_s=latency,
            execution=execution,
            telemetry=telemetry,
            raw_final=out.get("final"),
            quotes=quotes,
            error=error,
            framework=out.get("framework"),
            timed_out=timed_out,
            raw_dir=str(raw_dir),
            proxy_calls=[
                {k: v for k, v in c.as_dict().items() if k not in ("request_body", "response_body")}
                for c in calls
            ],
        )

    def _key(self, task: RuntimeTask, run_index: int) -> RunKey:
        return RunKey(
            genome_hash=canonical_hash({"system": self.spec.name, "runner": self.spec.runner}),
            task_id=task.id,
            trial=run_index,
            seed=run_index,
            versions=RunVersions(
                model_hash=self.settings.model_hash,
                prompt_template_version=TASK_PROMPT_VERSION,
                benchmark_hash=self.benchmark_hash,
                compiler_version=f"external:{self.spec.name}",
                grammar_version="external",
            ),
        )

    def _wynk_execution(self, task, run_index, out, tel, latency, timed_out) -> ExecutionResult:
        """Wynk's own ExecutionResult, with tokens / wall time re-measured exactly like every
        external system (proxy usage, runner-timed latency) so caps are applied uniformly."""
        final = out.get("final") if out.get("ok") else None
        if isinstance(final, dict) and not timed_out:
            own = ExecutionResult.model_validate(final)
        else:
            err = out.get("error") or {}
            msg = "run timed out" if timed_out else f"{err.get('type')}: {err.get('message')}"
            own = ExecutionResult(
                key=self._key(task, run_index),
                failure=FailureInfo(kind=FailureKind.EXECUTOR_ERROR, message=msg[:2000]),
            )
        usage = own.budget_usage.model_copy(
            update={
                "tokens": _tokens(tel),
                "wall_time_s": latency,
            }
        )
        return own.model_copy(update={"budget_usage": usage})


def _tokens(tel: dict[str, Any]) -> int:
    """Tokens charged against the cap, identical for every system: the backend-reported total,
    or - when some call never returned usage - the sum over the calls that did (lower bound)."""
    total = tel["total_tokens"]
    return total if total is not None else tel["known_total_tokens"]


def _steps(kind: str, out: dict[str, Any]) -> int | None:
    steps = out.get("steps")
    if not isinstance(steps, dict):
        return None
    if kind == "wynk":
        return len(steps.get("stage_trace") or []) or None
    for k in ("n_steps",):
        if isinstance(steps.get(k), int):
            return steps[k]
    if isinstance(steps.get("messages"), list):
        return sum(1 for m in steps["messages"] if m.get("type") == "ai")
    return None


def _child_env() -> dict[str, str]:
    import os

    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}  # proxy holds keys
    env.update(
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        CREWAI_DISABLE_TELEMETRY="true",
        CREWAI_TRACING_ENABLED="false",
        OTEL_SDK_DISABLED="true",
        HF_HUB_DISABLE_TELEMETRY="1",
        ANONYMIZED_TELEMETRY="false",
        LANGCHAIN_TRACING_V2="false",
    )
    return env


# --- status + evaluation -----------------------------------------------------------------------


def classify(
    ex: ExecutionResult,
    task: RuntimeTask,
    pages: dict[str, Page],
    *,
    timed_out: bool = False,
    errored: bool = False,
) -> RunStatus:
    """Execution status, same rule for every system. Correctness is NOT decided here."""
    if timed_out:
        return RunStatus.TIMEOUT
    if usage_exceeds(ex.budget_usage, task.caps) or (
        ex.failure is not None and ex.failure.kind is FailureKind.BUDGET_EXCEEDED
    ):
        return RunStatus.BUDGET_EXCEEDED
    if errored or (
        ex.failure is not None
        and ex.failure.kind in (FailureKind.MODEL_ERROR, FailureKind.EXECUTOR_ERROR)
    ):
        return RunStatus.RUNTIME_ERROR
    if ex.answer is None or validate_answer(task.answer_schema, ex.answer.values):
        return RunStatus.INVALID_OUTPUT
    cited = {fe.field: fe.spans for fe in ex.answer.evidence}
    for f in task.answer_schema.field_names:
        spans = cited.get(f) or ()
        if not spans or not all(_span_valid(s, pages) for s in spans):
            return RunStatus.MISSING_EVIDENCE
    return RunStatus.SUCCESS


def _span_valid(span: EvidenceSpan, pages: dict[str, Page]) -> bool:
    page = pages.get(span.page_id)
    return (
        page is not None
        and page.content_hash == span.content_hash
        and span.char_end <= len(page.content)
    )


def uncapped(spec: TaskSpec, ex: ExecutionResult) -> tuple[TaskSpec, ExecutionResult]:
    """Same task with caps lifted (diagnostic: correctness independent of the budget)."""
    spec_u = spec.model_copy(update={"runtime": spec.runtime.model_copy(update={"caps": UNCAPPED})})
    if ex.failure is not None and ex.failure.kind is FailureKind.BUDGET_EXCEEDED:
        ex = ex.model_copy(update={"failure": None})
    return spec_u, ex


def evaluate(
    evaluator: DeterministicEvaluator, spec: TaskSpec, ex: ExecutionResult
) -> tuple[Evaluation, Evaluation]:
    """(official evaluation under the task's caps, diagnostic evaluation with caps lifted)."""
    return evaluator.evaluate(spec, ex), evaluator.evaluate(*uncapped(spec, ex))


def quote_support(ex: ExecutionResult, pages: dict[str, Page]) -> float | None:
    """Fraction of answered fields whose cited text literally contains the value (diagnostic for
    unsupported claims; derived values such as sums cannot be quoted, so read with care)."""
    if ex.answer is None or not ex.answer.values:
        return None
    cited = {fe.field: fe.spans for fe in ex.answer.evidence}
    ok = 0
    for name, value in ex.answer.values.items():
        texts = [
            pages[s.page_id].content[s.char_start : s.char_end]
            for s in cited.get(name, ())
            if _span_valid(s, pages)
        ]
        ok += any(supports(value, t) for t in texts)
    return ok / len(ex.answer.values)
