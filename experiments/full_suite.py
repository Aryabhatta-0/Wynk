"""Full-benchmark evaluation (#13): ACO vs random search on EVERY suite of the frozen benchmark.

    python -m experiments.full_suite plan                      # protocol, task counts, run budget
    python -m experiments.full_suite run --out experiments/results/full-<tag>
    python -m experiments.full_suite run --synthetic           # labelled fake objective (no model)
    python -m experiments.full_suite report --out <dir>        # rebuild per-task + summary files

The canonical protocol lives in ``experiments/configs/full_suite.json`` (suites, optimizers,
seeds, candidate budget, trials); every value can be overridden on the command line, and the
effective protocol is written next to the results.

Three phases, three authorities (``core.dataset.ALLOWED_USES``):

1. SEARCH, per (suite, optimizer, seed): ``experiments.learning_curves.run_search`` on the suite's
   optimization rows (optimizer feedback) and validation rows (selection: incumbent curve, suite
   champion, per-task champions). Every optimizer of a suite gets the SAME candidate budget:
   ``genome_budget`` genomes x every optimization row x ``search_trials``. A held-out row reaching
   this phase is refused (``SplitAccessError``) before it runs.
2. FREEZE: every search record (champion, task champions, curve, every run) is written to
   ``search.json`` before any held-out row executes. Nothing after this point can reach an
   optimizer, a champion choice or workflow memory (this runner writes no memory at all).
3. FINAL, reporting only: each search's already-selected suite champion runs on every held-out
   test row of its suite (``final_trials`` trials). Results go to ``final.json`` and the reports.

A suite with no admissible workflow under the configured runtime (``Ajev`` without a JEV
provider) is recorded as ``skipped`` with the reason; its tasks still get per-task rows saying
so. Nothing is substituted for it and nothing is synthesized.

Outputs (``--out``): ``protocol.json``, ``search.json``, ``final.json``, ``evaluations.jsonl``,
``per_task.json`` / ``per_task.csv`` (one row per task x optimizer x seed), ``summary.json``,
``REPORT.md`` and ``learning_curve_<suite>.png``. Model runs are cached by run id in
``run_cache.jsonl`` / ``final_run_cache.jsonl``, so re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from benchmarks.legacy_adapter import legacy_suite
from benchmarks.loader import HELDOUT_DIR, suite_names
from core.canonical import canonical_hash
from core.constraints import ConstraintChecker
from core.dataset import SplitAccessError, SplitRole, SplitUse
from core.genome import Genome
from core.results import EvaluatedRun, Verdict
from core.run_contract import ContractSuite, ExecutionTask
from experiments.learning_curves import (
    EvaluateFn,
    ExperimentConfig,
    evaluate_with_retries,
    make_optimizer,
    run_search,
)
from experiments.task_champions import run_score

SCHEMA = "wynk-full-suite/1"
PROTOCOL_PATH = Path(__file__).resolve().parent / "configs" / "full_suite.json"
FINAL_TRIAL_OFFSET = 20_000  # final-run seeds never coincide with search / validation seeds


@dataclass(frozen=True)
class Protocol:
    """One benchmark protocol. ``genome_budget`` is the candidate-evaluation budget: genomes
    proposed and run on every optimization row x ``search_trials`` - identical for every
    optimizer of a suite."""

    name: str
    suites: tuple[str, ...]
    optimizers: tuple[str, ...]
    seeds: tuple[int, ...]
    genome_budget: int
    batch_size: int
    search_trials: int
    final_trials: int
    lcb_z: float
    workers: int
    max_concurrent_runs: int
    comparison: tuple[str, str] = ("aco_mmas", "random_search")
    notes: str = ""

    @classmethod
    def load(cls, path: Path = PROTOCOL_PATH) -> Protocol:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Protocol:
        p = cls(
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in raw.items()
                if k in cls.__dataclass_fields__
            }
        )
        p.validate()
        return p

    def validate(self) -> None:
        if not self.suites or not self.optimizers or not self.seeds:
            raise ValueError("a protocol needs suites, optimizers and seeds")
        if min(self.genome_budget, self.batch_size, self.search_trials, self.final_trials) < 1:
            raise ValueError("budgets and trials must be positive")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for c in self.comparison:
            if c not in self.optimizers:
                raise ValueError(f"comparison optimizer {c} is not run by the protocol")

    def config_for(self, suite: ContractSuite) -> ExperimentConfig:
        """The ``ExperimentConfig`` of every search of ``suite`` (same for all optimizers)."""
        n_train = len(suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK))
        return ExperimentConfig(
            budget=self.genome_budget * n_train * self.search_trials,
            batch_size=self.batch_size,
            trials=self.search_trials,
            lcb_z=self.lcb_z,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


# -- run records ----------------------------------------------------------------------------------
def gather_source(genome: Genome) -> str:
    g = next((s for s in genome.stages if s.kind == "GATHER"), None)
    return g.source.value if g is not None else "none"


def failure_kind(run: EvaluatedRun) -> str | None:
    """Why a run did not pass: the runtime failure kind, else ``infeasible`` / ``wrong_answer``."""
    if run.execution.failure is not None:
        return run.execution.failure.kind.value
    if run.evaluation.verdict is Verdict.INFEASIBLE:
        return "infeasible"
    if run.evaluation.verdict is Verdict.FAIL:
        return "wrong_answer"
    return None


def run_row(run: EvaluatedRun, genome: Genome, split: str) -> dict[str, Any]:
    """Plain-data record of one evaluated run (no answer values, no expected values)."""
    ex, ev = run.execution, run.evaluation
    return {
        "task_id": ex.key.task_id,
        "split": split,
        "genome_hash": ex.key.genome_hash,
        "trial": ex.key.trial,
        "exec_seed": ex.key.seed,
        "run_id": ex.key.run_id,
        "verdict": ev.verdict.value,
        "passed": ev.verdict is Verdict.PASS,
        "fitness": ev.fitness,
        "score": run_score(run),
        "tokens": ex.budget_usage.tokens,
        "model_calls": ex.metrics.model_calls,
        "tool_calls": ex.budget_usage.tool_calls,
        "retries": ex.budget_usage.retries,
        "wall_time_s": round(ex.budget_usage.wall_time_s, 3),
        "gather_source": gather_source(genome),
        "evidence_provider": ex.key.versions.evidence_source,
        "failure_kind": failure_kind(run),
        "failure_message": (ex.failure.message[:200] if ex.failure else None),
    }


class Recorder:
    """Records every run of one search (the FINAL attempt per genome x task x trial: backend
    retries supersede earlier attempts) and refuses held-out rows outright."""

    def __init__(
        self,
        suite: ContractSuite,
        label: dict[str, Any],
        log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.label = label
        self.log = log
        self.split_of: dict[str, str] = {}
        for t in suite.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK):
            self.split_of[t.id] = "train"
        for t in suite.tasks_for(SplitRole.VALIDATION, SplitUse.SELECTION):
            self.split_of[t.id] = "validation"
        self.genomes: dict[str, Genome] = {}
        self.runs: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.order: list[str] = []  # genome hashes in first-evaluation order (determinism check)
        self._lock = threading.Lock()

    def wrap(self, evaluate: EvaluateFn) -> EvaluateFn:
        def recorded(genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
            split = self.split_of.get(task.id)
            if split is None:  # held-out (or foreign) row: never executed during search
                raise SplitAccessError(f"row {task.id!r} may not run during search")
            run = evaluate(genome, task, trial, seed)
            row = run_row(run, genome, split)
            with self._lock:
                h = genome.genome_hash
                if h not in self.genomes:
                    self.genomes[h] = genome
                    self.order.append(h)
                self.runs[(h, task.id, trial)] = row
                if self.log is not None:
                    self.log({**self.label, **row})
            return run

        return recorded

    def rows(self) -> list[dict[str, Any]]:
        return [self.runs[k] for k in sorted(self.runs)]


def throttled(evaluate: EvaluateFn, gate: threading.Semaphore) -> EvaluateFn:
    """At most ``gate``'s capacity of workflow runs in flight across every concurrent search."""

    def run(*job: Any) -> EvaluatedRun:
        with gate:
            return evaluate(*job)

    return run


def unavailable_reason(suite: ContractSuite, checker: ConstraintChecker) -> str | None:
    """Why no workflow is admissible for ``suite`` under ``checker`` (``None``: some is)."""
    if next(checker.enumerate_admissible(suite.policy), None) is not None:
        return None
    workflow = suite.policy.workflow
    missing = sorted(
        s.value for s in workflow.allowed_sources if s in checker.config.unavailable_sources
    )
    if workflow.interaction_required and "jev" in missing:
        return (
            "interaction_required: GATHER jev needs a configured JEV provider "
            "(WYNK_JEV_API_KEY + WYNK_JEV_MODEL); none is configured, so no workflow is "
            "admissible and nothing is substituted"
        )
    return f"no admissible workflow under this runtime (unavailable sources: {missing or '-'})"


# -- phases ---------------------------------------------------------------------------------------
@dataclass
class SearchJob:
    suite: str
    optimizer: str
    seed: int
    config: ExperimentConfig
    status: str = "pending"  # completed | skipped | failed
    reason: str | None = None
    result: dict[str, Any] | None = None
    recorder: Recorder | None = field(default=None, repr=False)

    def record(self) -> dict[str, Any]:
        rec = self.recorder
        return {
            "suite": self.suite,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "status": self.status,
            "reason": self.reason,
            "config": asdict(self.config),
            "result": self.result,
            "genome_order": rec.order if rec else [],
            "genomes": {h: g.canonical() for h, g in sorted(rec.genomes.items())} if rec else {},
            "runs": rec.rows() if rec else [],
        }


def run_searches(
    protocol: Protocol,
    suites: Mapping[str, ContractSuite],
    evaluate: EvaluateFn,
    checker: ConstraintChecker,
    *,
    log: Callable[[dict[str, Any]], None] | None = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> list[SearchJob]:
    """Phase 1: every (suite, optimizer, seed) search, concurrently. Search order inside a search
    is seeded and submission-ordered (``run_search``), so concurrency never changes proposals."""
    gate = threading.Semaphore(protocol.max_concurrent_runs)
    shared = throttled(evaluate, gate)
    jobs: list[SearchJob] = []
    for name in protocol.suites:
        suite = suites[name]
        config = protocol.config_for(suite)
        reason = unavailable_reason(suite, checker)
        for opt in protocol.optimizers:
            for seed in protocol.seeds:
                job = SearchJob(name, opt, seed, config)
                if reason is not None:
                    job.status, job.reason = "skipped", reason
                jobs.append(job)
    t0 = time.time()

    def one(job: SearchJob) -> None:
        label = {"suite": job.suite, "optimizer": job.optimizer, "seed": job.seed}
        job.recorder = Recorder(suites[job.suite], label, log)
        try:
            job.result = run_search(
                make_optimizer(job.optimizer, job.config),
                job.recorder.wrap(shared),
                suites[job.suite],
                job.config,
                job.seed,
                checker=checker,
                workers=protocol.workers,
            )
            job.status = "completed"
            r = job.result
            progress(
                f"[{time.time() - t0:7.0f}s] {job.suite:6s} {job.optimizer:14s} seed {job.seed}: "
                f"{r['workflow_evaluations']} evals, val fitness {r['validation_fitness']}"
            )
        except Exception as exc:  # recorded, never silently dropped; a re-run resumes from cache
            job.status, job.reason = "failed", f"{type(exc).__name__}: {exc}"
            progress(f"[{time.time() - t0:7.0f}s] {job.suite} {job.optimizer} seed {job.seed}: "
                     f"FAILED {job.reason}")  # fmt: skip

    runnable = [j for j in jobs if j.status == "pending"]
    if runnable:
        with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
            list(pool.map(one, runnable))
    return jobs


def champion_genome(record: Mapping[str, Any]) -> Genome | None:
    result = record.get("result") or {}
    champ = result.get("champion")
    if record["status"] != "completed" or not champ:
        return None
    return Genome.from_canonical(record["genomes"][champ["genome_hash"]])


def run_final(
    protocol: Protocol,
    suites: Mapping[str, ContractSuite],
    frozen_searches: Sequence[Mapping[str, Any]],
    evaluate: EvaluateFn,
    *,
    progress: Callable[[str], None] = lambda _m: None,
) -> list[dict[str, Any]]:
    """Phase 3 (reporting only): each frozen search's suite champion on every held-out row of
    its suite. Reads the FROZEN records only; returns plain rows and touches nothing else."""
    gate = threading.Semaphore(protocol.max_concurrent_runs)
    shared = throttled(evaluate, gate)
    out: list[dict[str, Any]] = []
    work: list[tuple[dict[str, Any], Genome, ExecutionTask, int, int]] = []
    for rec in frozen_searches:
        genome = champion_genome(rec)
        label = {"suite": rec["suite"], "optimizer": rec["optimizer"], "seed": rec["seed"]}
        if genome is None:
            continue
        for task in suites[rec["suite"]].tasks_for(SplitRole.TEST, SplitUse.REPORTING):
            for trial in range(protocol.final_trials):
                seed = int(
                    canonical_hash(
                        [rec["seed"], genome.genome_hash, task.id, FINAL_TRIAL_OFFSET + trial]
                    )[:8],
                    16,
                )
                work.append((label, genome, task, trial, seed))
    progress(f"final: {len(work)} held-out runs")

    def one(item: tuple[dict[str, Any], Genome, ExecutionTask, int, int]) -> dict[str, Any]:
        label, genome, task, trial, seed = item
        try:
            run = evaluate_with_retries(shared, (genome, task, trial, seed))
        except Exception as exc:  # backend outage: recorded as unscored, never as a FAIL
            return {
                **label,
                "task_id": task.id,
                "split": "test",
                "genome_hash": genome.genome_hash,
                "trial": trial,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {**label, **run_row(run, genome, "test")}

    if work:
        with ThreadPoolExecutor(max_workers=max(1, protocol.max_concurrent_runs)) as pool:
            out = list(pool.map(one, work))
    return out


# -- artifacts ------------------------------------------------------------------------------------
def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n"


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(obj), encoding="utf-8")


def suites_for(protocol: Protocol) -> dict[str, ContractSuite]:
    known = suite_names()
    for name in protocol.suites:
        if name not in known:
            raise ValueError(f"unknown suite {name!r}; the benchmark has {known}")
    return {name: legacy_suite(name) for name in protocol.suites}


@dataclass(frozen=True)
class Backend:
    """Where runs come from: the real runtime + evaluator, or the labelled synthetic objective."""

    synthetic: bool
    search: EvaluateFn
    final: EvaluateFn
    checker: ConstraintChecker
    identity: dict[str, Any]


def synthetic_backend() -> Backend:
    from experiments.synthetic import SYNTHETIC_VERSION, synthetic_evaluate
    from runtime.runner import WorkflowRunner

    checker = WorkflowRunner(model=None, benchmark_hash="synthetic").checker  # no JEV either
    return Backend(
        synthetic=True,
        search=synthetic_evaluate,
        final=synthetic_evaluate,
        checker=checker,
        identity={"evaluator_version": SYNTHETIC_VERSION, "model_hash": "synthetic"},
    )


def real_backend(suites: Mapping[str, ContractSuite], out: Path) -> Backend:
    from benchmarks.legacy_adapter import legacy_evaluator
    from benchmarks.loader import BENCH_DIR, benchmark_hash
    from benchmarks.snapshot_store import SnapshotStore
    from experiments.real_runtime import RunCache, build_runner, real_evaluate_fn
    from runtime.backends.openai_compatible import client_from_env
    from runtime.backends.typesafe_jev import jev_from_env

    client = client_from_env()
    jev = jev_from_env(os.environ)  # None unless a hosted JEV is configured
    store = SnapshotStore(BENCH_DIR / "snapshots")
    held_store = SnapshotStore(HELDOUT_DIR / "snapshots")
    runner = build_runner(client, store, jev=jev)
    held_runner = build_runner(client, held_store, bench_dir=HELDOUT_DIR, jev=jev)
    search_tasks = [
        t
        for s in suites.values()
        for role, use in (
            (SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK),
            (SplitRole.VALIDATION, SplitUse.SELECTION),
        )
        for t in s.tasks_for(role, use)
    ]
    test_tasks = [
        t for s in suites.values() for t in s.tasks_for(SplitRole.TEST, SplitUse.REPORTING)
    ]
    evaluator = legacy_evaluator(BENCH_DIR)
    return Backend(
        synthetic=False,
        search=real_evaluate_fn(runner, evaluator, search_tasks, RunCache(out / "run_cache.jsonl")),
        final=real_evaluate_fn(
            held_runner,
            legacy_evaluator(HELDOUT_DIR),
            test_tasks,
            RunCache(out / "final_run_cache.jsonl"),
        ),
        checker=runner.checker,
        identity={
            "evaluator_version": evaluator.fitness_fn.version,
            "model_hash": runner.versions().model_hash,
            "jev_provider": jev.identity if jev is not None else None,
            "search_benchmark_hash": benchmark_hash(BENCH_DIR, store),
            "heldout_benchmark_hash": benchmark_hash(HELDOUT_DIR, held_store),
        },
    )


def run_protocol(
    protocol: Protocol,
    backend: Backend,
    out: Path,
    *,
    final: bool = True,
    plots: bool = True,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Search -> freeze -> final -> reports. Returns the summary."""
    from benchmarks.loader import BENCH_DIR, benchmark_hash
    from experiments import full_suite_report as report

    suites = suites_for(protocol)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA,
        "synthetic": backend.synthetic,
        "protocol": protocol.to_dict(),
        "benchmark_hash": benchmark_hash(BENCH_DIR),
        "heldout_benchmark_hash": benchmark_hash(HELDOUT_DIR),
        "suite_hashes": {n: s.identity_hash for n, s in suites.items()},
        "budgets": {n: asdict(protocol.config_for(s)) for n, s in suites.items()},
        **backend.identity,
    }
    write(out / "protocol.json", meta)

    log_path = out / "evaluations.jsonl"
    log_path.write_text("", encoding="utf-8")
    log_lock = threading.Lock()

    def log(row: dict[str, Any]) -> None:
        with log_lock, log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    t0 = time.time()
    jobs = run_searches(
        protocol, suites, backend.search, backend.checker, log=log, progress=progress
    )
    searches = [j.record() for j in jobs]
    write(out / "search.json", {"schema": SCHEMA, "searches": searches})  # FROZEN before final
    frozen = json.loads((out / "search.json").read_text(encoding="utf-8"))["searches"]
    finals = run_final(protocol, suites, frozen, backend.final, progress=progress) if final else []
    write(out / "final.json", {"schema": SCHEMA, "ran": final, "runs": finals})
    summary = report.write_reports(out, plots=plots)
    progress(f"done in {time.time() - t0:.0f}s -> {out}")
    return summary


def plan(protocol: Protocol) -> dict[str, Any]:
    """Counts and budgets the protocol implies (no model calls)."""
    suites = suites_for(protocol)
    rows = {}
    for name, s in suites.items():
        train = s.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK)
        val = s.tasks_for(SplitRole.VALIDATION, SplitUse.SELECTION)
        test = s.tasks_for(SplitRole.TEST, SplitUse.REPORTING)
        cfg = protocol.config_for(s)
        rows[name] = {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "search_budget_per_optimizer_seed": cfg.budget,
            "max_validation_runs_per_search": protocol.genome_budget * len(val) * cfg.trials,
            "final_runs_per_optimizer_seed": len(test) * protocol.final_trials,
            "interaction_required": s.policy.workflow.interaction_required,
        }
    searches = len(protocol.optimizers) * len(protocol.seeds)
    return {
        "suites": rows,
        "searches": searches * len(suites),
        "train_runs_upper_bound": searches
        * sum(r["search_budget_per_optimizer_seed"] for r in rows.values()),
        "final_runs": searches * sum(r["final_runs_per_optimizer_seed"] for r in rows.values()),
    }


def _protocol_from_args(args: argparse.Namespace) -> Protocol:
    p = Protocol.load(args.config)
    overrides = {
        "suites": tuple(args.suites) if args.suites else None,
        "seeds": tuple(range(args.seeds)) if args.seeds else None,
        "genome_budget": args.genome_budget,
        "search_trials": args.search_trials,
        "final_trials": args.final_trials,
        "workers": args.workers,
        "max_concurrent_runs": args.max_concurrent_runs,
    }
    p = replace(p, **{k: v for k, v in overrides.items() if v is not None})
    p.validate()
    return p


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, default=PROTOCOL_PATH)
        s.add_argument("--suites", nargs="+")
        s.add_argument("--seeds", type=int, help="use seeds 0..N-1")
        s.add_argument("--genome-budget", type=int)
        s.add_argument("--search-trials", type=int)
        s.add_argument("--final-trials", type=int)
        s.add_argument("--workers", type=int)
        s.add_argument("--max-concurrent-runs", type=int)
        if name == "run":
            s.add_argument("--out", type=Path)
            s.add_argument("--synthetic", action="store_true", help="labelled fake objective")
            s.add_argument("--no-final", action="store_true", help="skip held-out reporting")
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "report":
        from experiments import full_suite_report as report

        report.write_reports(args.out)
        print(f"rewrote reports in {args.out}")
        return
    protocol = _protocol_from_args(args)
    if args.cmd == "plan":
        print(dump({"protocol": protocol.to_dict(), **plan(protocol)}), end="")
        return
    default = "full-synthetic" if args.synthetic else f"full-{time.strftime('%Y%m%d-%H%M%S')}"
    out = args.out or Path("experiments/results") / default
    suites = suites_for(protocol)
    backend = synthetic_backend() if args.synthetic else real_backend(suites, out)
    run_protocol(protocol, backend, out, final=not args.no_final)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA",
    "Backend",
    "Protocol",
    "Recorder",
    "SearchJob",
    "failure_kind",
    "plan",
    "run_final",
    "run_protocol",
    "run_row",
    "run_searches",
]
