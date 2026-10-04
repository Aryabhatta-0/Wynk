"""Wynk frozen workflow vs. official OSS *starter* workflows, on the held-out TEST split.

    python -m experiments.oss_baselines.harness freeze      # persist Wynk's selected workflows
    python -m experiments.oss_baselines.harness manifest    # pinned versions + config
    python -m experiments.oss_baselines.harness smoke       # 1 task x 1 run x every system
    python -m experiments.oss_baselines.harness run --runs 3 --workers 8
    python -m experiments.oss_baselines.harness report --out <results dir>

Model backend from ``GEMMA_BASE_URL`` / ``GEMMA_MODEL`` / ``GEMMA_API_KEY`` (same as Wynk).
Every system's model traffic goes through one ``MeteringProxy``. Results (one JSON line per
execution, plus raw input/output/stdout/stderr/proxy calls per execution) go to
``experiments/results/oss_baselines/<name>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.heldout import HELDOUT_DIR, HELDOUT_SNAPSHOTS, SPLIT
from benchmarks.loader import benchmark_hash, load_splits, load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.genome import Genome
from core.payloads import Page
from core.results import ExecutionResult
from core.task_spec import TaskSpec
from evaluation.evidence import SnapshotEvidenceVerifier
from evaluation.gate import DeterministicEvaluator
from experiments.oss_baselines import report as rep
from experiments.oss_baselines.baselines import (
    HERE,
    MODE,
    RUN_TIMEOUT_S,
    SYSTEMS,
    TASK_PROMPT_VERSION,
    BaselineResult,
    ModelSettings,
    SubprocessBaseline,
    classify,
    evaluate,
    normalize_answer,
    parse_final,
    quote_support,
    task_prompt,
)
from experiments.oss_baselines.proxy import MeteringProxy

FROZEN = HERE / "frozen_wynk.json"
MANIFEST = HERE / "manifest.json"
RESULTS = Path("experiments/results/oss_baselines")
ACO_RESULTS = Path("experiments/results/real-gemma4")
ACO_SOURCE_COMMIT = "ff2358b"  # commit that produced experiments/results/real-gemma4


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- freeze ------------------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> None:
    """Persist the workflow Wynk SELECTED (search on train, incumbent chosen on validation).

    Selection rule = ``experiments/run_mvp.py::cmd_experiment``: the incumbent of the ACO seed with
    the best final validation fitness. Nothing here searches, updates pheromones or touches the
    test split. Refuses to overwrite an existing freeze."""
    if FROZEN.exists() and not args.force:
        raise SystemExit(f"{FROZEN} exists: the Wynk workflow is already frozen")
    classes: dict[str, Any] = {}
    for cls in ("A", "B"):
        d = ACO_RESULTS / cls
        results = json.loads((d / "results.json").read_text(encoding="utf-8"))
        saved = json.loads((d / "best_aco_genome.json").read_text(encoding="utf-8"))
        genome = Genome.from_stages(saved["stages"])
        aco = [r for r in results["runs"] if r["optimizer"] == "aco_mmas"]
        chosen = max(aco, key=lambda r: r["validation_fitness"])
        if chosen["best_genome_hash"] != genome.genome_hash:
            raise SystemExit(f"class {cls}: saved genome is not the selected ACO incumbent")
        classes[cls] = {
            "genome_hash": genome.genome_hash,
            "genome_stages": saved["stages"],
            "workflow": (d / "best_aco_workflow.txt").read_text(encoding="utf-8").strip(),
            "optimizer_run_id": f"{d.as_posix()}/results.json#optimizer=aco_mmas,seed="
            f"{chosen['seed']}",
            "optimizer_results_sha256": _sha256_file(d / "results.json"),
            "optimizer_config": results["config"],
            "workflow_evaluations": chosen["workflow_evaluations"],
            "validation_fitness": chosen["validation_fitness"],
            "validation_pass_rate": chosen["pass_rate"],
            "train_pass_rate": chosen["train_pass_rate"],
            "train_tasks": results["train_tasks"],
            "validation_tasks": results["validation_tasks"],
            "evaluator_at_selection": results["evaluator_version"],
            "search_benchmark_hash": results["benchmark_hash"],
            "model_hash_at_selection": results["model_hash"],
            "model_assignments": {
                s["kind"]: "google/gemma-4-31b-it (ZenMux)"
                for s in saved["stages"]
                if s["kind"] in ("EXTRACT", "REASON", "SYNTHESIZE")
            },
            "tool_configuration": {
                s["kind"]: {k: v for k, v in s.items() if k != "kind"}
                for s in saved["stages"]
                if s["kind"] == "GATHER"
            },
        }
    store = SnapshotStore(HELDOUT_SNAPSHOTS)
    frozen = {
        "schema": "wynk-frozen-workflow/1",
        "frozen_at": _now(),
        "source_commit": ACO_SOURCE_COMMIT,
        "selection_rule": "per task class: incumbent of the ACO (MMAS) seed with the best final "
        "validation fitness (experiments/run_mvp.py cmd_experiment); search used only the train "
        "split, selection only the validation split",
        "policy": "frozen before any external comparison: no pheromone updates, no optimisation "
        "against test tasks, no manual edits after observing results",
        "classes": classes,
        "heldout_test": {
            "benchmark_hash": benchmark_hash(HELDOUT_DIR, store),
            "task_ids": list(load_splits(HELDOUT_DIR)[SPLIT]),
        },
    }
    FROZEN.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
    print(f"froze {FROZEN} sha256={_sha256_file(FROZEN)}")
    for cls, c in classes.items():
        print(f"  {cls}: {c['genome_hash'][:12]} {c['workflow']}")


def load_frozen() -> dict[str, Any]:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    for cls, c in frozen["classes"].items():
        if Genome.from_stages(c["genome_stages"]).genome_hash != c["genome_hash"]:
            raise SystemExit(f"frozen genome for class {cls} does not match its hash")
    return frozen


# --- manifest ----------------------------------------------------------------------------------

EXTERNAL_META: dict[str, dict[str, Any]] = {
    "smolagents": {
        "framework": "Hugging Face smolagents",
        "repository": "https://github.com/huggingface/smolagents",
        "package": "smolagents[openai]==1.26.0",
        "tag": "v1.26.0",
        "commit": "12c1bc820eca50ace6f80a21d90426d41d74f845",
        "example": "README quickstart: CodeAgent(tools=[WebSearchTool()], model=model); "
        "agent.run(query)",
        "agent_type": "CodeAgent (single agent, writes Python actions)",
        "agents": 1,
        "configuration": "library defaults (max_steps=20, local executor, default prompts and "
        "authorized imports); OpenAIServerModel(model_id, api_base, temperature=0, "
        "max_tokens=1024, client_kwargs={max_retries: 3, timeout: 120})",
        "prompt_modifications": "none to the framework prompts; the query is the shared task "
        "prompt",
        "tool_modifications": "WebSearchTool replaced by the shared list_pages/read_page tools",
    },
    "crewai": {
        "framework": "CrewAI",
        "repository": "https://github.com/crewAIInc/crewAI",
        "package": "crewai==1.15.23",
        "tag": "1.15.23",
        "commit": "deaa71e168069a1d5307340172875def4330e75b",
        "example": "official `crewai create crew` template (crewai_cli/templates/crew): "
        "researcher agent + research_task, Process.sequential",
        "agent_type": "Crew with 1 agent (template 'researcher'), 1 task, sequential process",
        "agents": 1,
        "configuration": "library defaults (no memory, no planning, no manager, default "
        "max_iter); LLM(model='openai/<model>', base_url, temperature=0, max_tokens=1024, "
        "timeout=120, max_retries=3)",
        "prompt_modifications": "role/goal/backstory verbatim from the template with "
        "{topic}='the provided source documents'; task description = shared task prompt; "
        "expected_output = 'ONLY the JSON object described in the task (answer + evidence), "
        "nothing else.'; the template's second agent (reporting_analyst, writes report.md) is "
        "not used",
        "tool_modifications": "template placeholder MyCustomTool / SerperDevTool replaced by the "
        "shared list_pages/read_page tools",
    },
    "llamaindex": {
        "framework": "LlamaIndex",
        "repository": "https://github.com/run-llama/llama_index",
        "package": "llama-index-core==0.14.25, llama-index-llms-openai-like==0.8.1",
        "tag": "v0.14.25",
        "commit": "f12d46acab73f5b2243ef49c2f00101617b38ce4",
        "example": "starter tutorial: FunctionAgent(tools=[...], llm=..., system_prompt='You "
        "are a helpful assistant that can ...'); await agent.run(user_msg)",
        "agent_type": "FunctionAgent (single agent, native function calling); agentic "
        "retrieval through tools, no vector index",
        "agents": 1,
        "configuration": "library defaults (streaming, max iterations, memory); "
        "OpenAILike(model, api_base, is_chat_model=True, is_function_calling_model=True, "
        "context_window=262144, temperature=0, max_tokens=1024, timeout=120, max_retries=3)",
        "prompt_modifications": "system_prompt='You are a helpful assistant that can read the "
        "source pages for a task.' (starter wording, capability changed); user_msg = shared "
        "task prompt",
        "tool_modifications": "starter's example tools replaced by the shared "
        "list_pages/read_page tools",
    },
    "langgraph": {
        "framework": "LangGraph",
        "repository": "https://github.com/langchain-ai/langgraph",
        "package": "langgraph==1.2.12, langchain-openai==1.6.7",
        "tag": "1.2.12",
        "commit": "49cce0ca852be4cfb567a1cbe0e511ff325a1682",
        "example": "quickstart prebuilt agent: create_react_agent(model, tools=[...], "
        "prompt='You are a helpful assistant'); agent.invoke({'messages': [...]})",
        "agent_type": "prebuilt ReAct tool-calling graph: START -> LLM -> tools? -> LLM -> END",
        "agents": 1,
        "configuration": "library defaults (recursion_limit=25, no checkpointer); "
        "ChatOpenAI(model, base_url, temperature=0, max_tokens=1024, timeout=120, max_retries=3)",
        "prompt_modifications": "none (quickstart prompt); the user message is the shared "
        "task prompt",
        "tool_modifications": "quickstart get_weather tool replaced by the shared "
        "list_pages/read_page tools",
    },
}


def model_settings() -> ModelSettings:
    model = os.environ.get("GEMMA_MODEL")
    if not model or not os.environ.get("GEMMA_BASE_URL"):
        raise SystemExit("set GEMMA_BASE_URL / GEMMA_MODEL / GEMMA_API_KEY (see .env)")
    return ModelSettings(model=model, revision=os.environ.get("GEMMA_MODEL_REVISION", ""))


def build_manifest() -> dict[str, Any]:
    frozen = load_frozen()
    req = HERE / "requirements"
    systems: dict[str, Any] = {}
    for name, spec in SYSTEMS.items():
        entry: dict[str, Any] = {
            "label": spec.label,
            "priority": spec.priority,
            "runner": f"experiments/oss_baselines/runners/{spec.runner}",
            "runner_sha256": _sha256_file(HERE / "runners" / spec.runner),
        }
        if spec.kind == "external":
            entry.update(EXTERNAL_META[name])
            entry["requirements_lock"] = f"experiments/oss_baselines/requirements/{spec.venv}.txt"
            entry["requirements_lock_sha256"] = _sha256_file(req / f"{spec.venv}.txt")
            entry["install_command"] = (
                f"uv venv --python 3.13 .oss-venvs/{spec.venv} && "
                f"uv pip install --python .oss-venvs/{spec.venv} "
                f"-r experiments/oss_baselines/requirements/{spec.venv}.txt"
            )
            entry["tools"] = ["list_pages()", "read_page(page_id)"]
        else:
            entry.update(
                framework="Wynk (this repository)",
                workflows={c: v["workflow"] for c, v in frozen["classes"].items()},
                genome_hashes={c: v["genome_hash"] for c, v in frozen["classes"].items()},
                frozen_file_sha256=_sha256_file(FROZEN),
                configuration="unchanged Wynk runtime (WorkflowRunner -> MAF); "
                "OpenAICompatibleClient(temperature=0, max_tokens<=1024, timeout=120, "
                "max_retries=3)",
                tools=[
                    "GATHER fetch: every frozen page",
                    "GATHER api: mock JSON endpoint (byte-identical to the 'items' page)",
                ],
                install_command='pip install -e ".[dev,maf]"',
            )
        systems[name] = entry
    s = model_settings()
    return {
        "schema": "oss-baselines-manifest/1",
        "generated_at": _now(),
        "mode": MODE,
        "mode_note": "Wynk has no web-search tool, so NATIVE_SEARCH is not applicable: every "
        "system reads the same frozen pages of each held-out task.",
        "model": {
            "name": s.model,
            "provider": os.environ.get("GEMMA_BASE_URL"),
            "temperature": s.temperature,
            "max_tokens_per_call": s.max_tokens,
            "per_call_timeout_s": s.timeout_s,
            "client_retries": s.max_retries,
            "run_timeout_s": RUN_TIMEOUT_S,
        },
        "task_prompt_version": TASK_PROMPT_VERSION,
        "task_prompt_example": task_prompt(load_task_specs(HELDOUT_DIR)["TA-001"].runtime_view()),
        "heldout": frozen["heldout_test"],
        "evaluator": _evaluator().version + "+" + _evaluator().fitness_fn.version,
        "systems": systems,
    }


def cmd_manifest(_args: argparse.Namespace) -> None:
    m = build_manifest()
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}")


# --- running -----------------------------------------------------------------------------------


def _evaluator() -> DeterministicEvaluator:
    return DeterministicEvaluator(SnapshotEvidenceVerifier(SnapshotStore(HELDOUT_SNAPSHOTS)))


class Bench:
    def __init__(self, out: Path, systems: Sequence[str], timeout_s: float) -> None:
        self.out = out
        self.frozen = load_frozen()
        self.store = SnapshotStore(HELDOUT_SNAPSHOTS)
        self.specs: dict[str, TaskSpec] = load_task_specs(HELDOUT_DIR)
        self.bhash = benchmark_hash(HELDOUT_DIR, self.store)
        if self.bhash != self.frozen["heldout_test"]["benchmark_hash"]:
            raise SystemExit("held-out benchmark changed since the Wynk workflow was frozen")
        self.evaluator = _evaluator()
        self.settings = model_settings()
        self.proxy = MeteringProxy(
            os.environ["GEMMA_BASE_URL"], os.environ.get("GEMMA_API_KEY")
        ).start()
        self._pages: dict[str, dict[str, Page]] = {}
        self._lock = threading.Lock()
        self.baselines: dict[tuple[str, str], SubprocessBaseline] = {}
        for name in systems:
            spec = SYSTEMS[name]
            for cls in ("A", "B"):
                wynk = None
                if spec.kind == "wynk":
                    c = self.frozen["classes"][cls]
                    wynk = {"genome_hash": c["genome_hash"], "genome_stages": c["genome_stages"]}
                self.baselines[(name, cls)] = SubprocessBaseline(
                    spec,
                    self.proxy,
                    self.settings,
                    self.pages_for,
                    self.bhash,
                    out / "raw",
                    wynk=wynk,
                    timeout_s=timeout_s,
                )

    def pages_for(self, snapshot_id: str) -> dict[str, Page]:
        with self._lock:
            if snapshot_id not in self._pages:
                self._pages[snapshot_id] = {
                    p.page_id: p for p in self.store.pages(snapshot_id).pages
                }
            return self._pages[snapshot_id]

    def record(self, r: BaselineResult) -> dict[str, Any]:
        spec = self.specs[r.task_id]
        pages = self.pages_for(spec.runtime.snapshot_id)
        official, diag = evaluate(self.evaluator, spec, r.execution)
        ex = r.execution
        spans = []
        for fe in ex.evidence:
            for s in fe.spans:
                p = pages.get(s.page_id)
                spans.append(
                    {
                        "field": fe.field,
                        **s.model_dump(mode="json"),
                        "text": p.content[s.char_start : s.char_end] if p else None,
                    }
                )
        n = len(diag.field_results) or 1
        return {
            "system": r.system,
            "label": r.label,
            "task_id": r.task_id,
            "task_class": r.task_class,
            "run_index": r.run_index,
            "mode": r.mode,
            "status": r.status.value,
            "verdict": official.verdict.value,
            "fitness": official.fitness,
            "evaluator_version": official.evaluator_version,
            "field_results": [f.model_dump(mode="json") for f in official.field_results],
            "uncapped": {
                "verdict": diag.verdict.value,
                "all_fields_matched": bool(diag.field_results)
                and all(f.matched for f in diag.field_results),
                "field_accuracy": sum(f.matched for f in diag.field_results) / n,
                "evidence_valid_fraction": sum(bool(f.evidence_valid) for f in diag.field_results)
                / n,
                "field_results": [f.model_dump(mode="json") for f in diag.field_results],
            },
            "quote_supports_value_fraction": quote_support(ex, pages),
            "latency_s": round(r.latency_s, 3),
            "budget_usage": ex.budget_usage.model_dump(mode="json"),
            "caps": spec.caps.model_dump(mode="json"),
            "telemetry": r.telemetry,
            "answer": ex.answer.values if ex.answer else None,
            "evidence": spans,
            "quotes": r.quotes,
            "raw_final": r.raw_final if r.system != "wynk" else None,
            "failure": ex.failure.model_dump(mode="json") if ex.failure else None,
            "error": r.error,
            "framework": r.framework,
            "timed_out": r.timed_out,
            "task": spec.runtime.model_dump(mode="json"),
            "execution_result": ex.model_dump(mode="json"),
            "proxy_calls": r.proxy_calls,
            "raw_dir": r.raw_dir,
            "finished_at": _now(),
        }

    def run(self, jobs: list[tuple[str, str, int]], workers: int) -> list[dict[str, Any]]:
        runs_path = self.out / "runs.jsonl"
        done = set()
        if runs_path.is_file():  # crash recovery only: completed executions are never re-run
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                done.add((d["system"], d["task_id"], d["run_index"]))
        todo = [j for j in jobs if j not in done]
        print(
            f"{len(jobs)} executions ({len(jobs) - len(todo)} already recorded), {workers} workers",
            flush=True,
        )
        t0 = time.time()
        write_lock = threading.Lock()
        rows: list[dict[str, Any]] = []

        def one(job: tuple[str, str, int]) -> None:
            system, task_id, idx = job
            task = self.specs[task_id].runtime_view()
            r = self.baselines[(system, task.task_class.value)].run(task, idx)
            row = self.record(r)
            with write_lock:
                rows.append(row)
                with runs_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                print(
                    f"[{time.time() - t0:6.0f}s] {system:10s} {task_id} r{idx} "
                    f"{row['status']:16s} {row['verdict']:10s} fit {row['fitness']:+.3f} "
                    f"tok {row['telemetry']['total_tokens']} {row['latency_s']:.1f}s",
                    flush=True,
                )

        self.out.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
        return rows

    def close(self) -> None:
        self.proxy.stop()


def _jobs(systems: Sequence[str], task_ids: Sequence[str], runs: int) -> list[tuple[str, str, int]]:
    jobs = [(s, t, r) for r in range(runs) for t in task_ids for s in systems]
    random.Random(0).shuffle(jobs)  # interleave systems so backend load hits all of them equally
    return jobs


def _write_run_meta(out: Path, args: argparse.Namespace, systems, task_ids) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    meta = {
        "started_at": _now(),
        "command": args.cmd,
        "systems": list(systems),
        "task_ids": list(task_ids),
        "runs_per_task": getattr(args, "runs", 1),
        "workers": args.workers,
        "frozen_wynk_sha256": _sha256_file(FROZEN),
        "manifest": build_manifest(),
    }
    if meta_path.exists():  # resumed run: keep the original, append the resume
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.setdefault("resumed_at", []).append(meta["started_at"])
        meta = old
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


def _systems(arg: str | None) -> list[str]:
    names = arg.split(",") if arg else list(SYSTEMS)
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    return names


def cmd_smoke(args: argparse.Namespace) -> None:
    systems = _systems(args.systems)
    out = args.out or RESULTS / f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    task_ids = args.tasks.split(",")
    _write_run_meta(out, args, systems, task_ids)
    bench = Bench(out, systems, args.timeout)
    try:
        bench.run(_jobs(systems, task_ids, 1), args.workers)
    finally:
        bench.close()
    rep.write_report(out)


def cmd_run(args: argparse.Namespace) -> None:
    systems = _systems(args.systems)
    out = args.out or RESULTS / f"full-{datetime.now():%Y%m%d-%H%M%S}"
    task_ids = list(load_splits(HELDOUT_DIR)[SPLIT])
    _write_run_meta(out, args, systems, task_ids)
    bench = Bench(out, systems, args.timeout)
    try:
        bench.run(_jobs(systems, task_ids, args.runs), args.workers)
    finally:
        bench.close()
    rep.write_report(out)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    """Re-score RECORDED external outputs with Wynk-runtime citation strictness (exact quotes
    only). No new executions; Wynk rows are unchanged. Secondary analysis, not the headline."""
    specs = load_task_specs(HELDOUT_DIR)
    store = SnapshotStore(HELDOUT_SNAPSHOTS)
    evaluator = _evaluator()
    rows = rep.load_rows(args.out)
    out_rows = []
    for row in rows:
        row = dict(row)
        if row["system"] != "wynk" and row["execution_result"]["answer"] is not None:
            spec = specs[row["task_id"]]
            task = spec.runtime_view()
            pages = {p.page_id: p for p in store.pages(task.snapshot_id).pages}
            parsed = parse_final(row["raw_final"], set(task.answer_schema.field_names))
            answer, _ = normalize_answer(parsed, task, pages, lenient=False)
            ex = ExecutionResult.model_validate(row["execution_result"])
            ex = ex.model_copy(update={"answer": answer})
            official, diag = evaluate(evaluator, spec, ex)
            n = len(diag.field_results) or 1
            row.update(
                verdict=official.verdict.value,
                fitness=official.fitness,
                status=classify(ex, task, pages).value
                if row["status"] not in ("TIMEOUT", "RUNTIME_ERROR")
                else row["status"],
                uncapped={
                    **row["uncapped"],
                    "evidence_valid_fraction": sum(
                        bool(f.evidence_valid) for f in diag.field_results
                    )
                    / n,
                },
            )
        out_rows.append(row)
    summary = rep.summarize(out_rows)
    target = args.out / "sensitivity_exact_citations"
    target.mkdir(exist_ok=True)
    (target / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    md = rep.markdown(summary, None).replace(
        "# Wynk frozen workflow vs. OSS starter workflows",
        "# SENSITIVITY (secondary): external citations located exactly, as Wynk's runtime does",
    )
    (target / "report.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_report(args: argparse.Namespace) -> None:
    rep.write_report(args.out)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    sub.add_parser("manifest")
    for name, default_workers in (("smoke", 5), ("run", 8)):
        s = sub.add_parser(name)
        s.add_argument("--systems", default=None, help="comma list; default: all")
        s.add_argument("--workers", type=int, default=default_workers)
        s.add_argument("--timeout", type=float, default=RUN_TIMEOUT_S)
        s.add_argument("--out", type=Path, default=None)
        if name == "smoke":
            s.add_argument("--tasks", default="TA-001")
        else:
            s.add_argument("--runs", type=int, default=3)
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    sen = sub.add_parser("sensitivity")
    sen.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    {
        "freeze": cmd_freeze,
        "manifest": cmd_manifest,
        "smoke": cmd_smoke,
        "run": cmd_run,
        "report": cmd_report,
        "sensitivity": cmd_sensitivity,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
