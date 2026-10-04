"""Learn from an experiment, and warm-start MMAS ACO from what was learned.

* ``update_from_experiment(results, optimizer)`` turns ONE real ``run_search`` result (measured
  by the deterministic evaluator) + the ``MMASACO`` that produced it into a ``WorkflowMemory``:
  top-3 workflows by validation fitness + the explored edge pheromones. No LLM is involved.
* ``aco_from_memory(memory, expected)`` returns a warm ``MMASACO`` if the memory is compatible,
  otherwise a normal cold one - and always says why.

The headline ACO-vs-random comparison (``experiments.learning_curves``) stays cold-start; this is
a separate, secondary experiment: ``python -m memory.warm_start --synthetic ...``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from core.canonical import canonical_hash
from core.genome import Genome
from core.grammar import GRAMMAR_VERSION
from core.task_spec import RuntimeTask
from experiments.learning_curves import EvaluateFn, ExperimentConfig, run_search
from memory.models import (
    MAX_WORKFLOWS,
    EdgePheromone,
    MemoryKey,
    WorkflowMemory,
    WorkflowRecord,
    genome_path,
    node_label,
)
from memory.store import WorkflowMemoryStore
from optimizers.aco_mmas import MMASACO, ACOConfig

Start = Literal["cold", "warm"]


class IncompatibleMemory(ValueError):
    pass


@dataclass(frozen=True)
class WarmStartReport:
    mode: Start  # what actually happened
    reasons: tuple[str, ...]  # why (always non-empty)
    edges_loaded: int = 0


# -- compatibility ---------------------------------------------------------------------------
def incompatibilities(memory: WorkflowMemory, expected: MemoryKey) -> tuple[str, ...]:
    """Every field of the memory key that differs from what the current search expects."""
    have, want = memory.key.model_dump(), expected.model_dump()
    return tuple(
        f"{field}: memory has {have[field]!r}, search expects {want[field]!r}"
        for field in sorted(want)
        if have[field] != want[field]
    )


def synthetic_key(task_class: str) -> MemoryKey:
    """Key of a search on the labelled SYNTHETIC objective (never compatible with real memory)."""
    from experiments.synthetic import SYNTHETIC_VERSION

    return MemoryKey(
        task_class=task_class,
        grammar_version=GRAMMAR_VERSION,
        optimizer_version=MMASACO.version,
        benchmark_hash="synthetic",
        model_hash="synthetic",
        evaluator_version=SYNTHETIC_VERSION,
        prompt_template_version="synthetic",
        compiler_version="synthetic",
        aco_config_hash=canonical_hash(asdict(ACOConfig())),
    )


# -- learn -----------------------------------------------------------------------------------
def _key_from_results(results: Mapping[str, Any], optimizer: MMASACO) -> MemoryKey:
    versions = results.get("versions") or []
    if len(versions) != 1:
        raise ValueError(f"need exactly one run/evaluator version in results, got {len(versions)}")
    v = versions[0]
    rv = v["run_versions"]
    if rv["grammar_version"] not in (GRAMMAR_VERSION, "synthetic"):
        raise IncompatibleMemory(
            f"results were produced under grammar {rv['grammar_version']!r}, "
            f"this code is {GRAMMAR_VERSION!r}"
        )
    return MemoryKey(
        task_class=results["task_class"],
        grammar_version=GRAMMAR_VERSION,
        optimizer_version=MMASACO.version,
        benchmark_hash=rv["benchmark_hash"],
        model_hash=rv["model_hash"],
        evaluator_version=v["evaluator_version"],
        prompt_template_version=rv["prompt_template_version"],
        compiler_version=rv["compiler_version"],
        aco_config_hash=canonical_hash(asdict(optimizer.config)),
    )


def _record(w: Mapping[str, Any]) -> WorkflowRecord:
    genome = Genome.model_validate({"stages": w["genome"]["stages"]})
    if genome.genome_hash != w["genome_hash"]:
        raise ValueError(f"workflow {w['genome_hash'][:12]} does not match its genome")
    return WorkflowRecord(
        genome_hash=w["genome_hash"],
        path=genome_path(genome),
        genome=genome,
        validation_fitness=w["validation_fitness"],
        pass_rate=w["validation_pass_rate"],
        mean_tokens=w["mean_tokens"],
        mean_wall_time_s=w["mean_wall_time_s"],
        validation_runs=w["validation_runs"],
    )


def top_workflows(records: Sequence[WorkflowRecord]) -> tuple[WorkflowRecord, ...]:
    """Best ``MAX_WORKFLOWS`` by validation fitness, then pass rate, then hash (deterministic).
    A later record for the same genome replaces an earlier one."""
    by_hash = {r.genome_hash: r for r in records}
    ranked = sorted(
        by_hash.values(), key=lambda r: (-r.validation_fitness, -r.pass_rate, r.genome_hash)
    )
    return tuple(ranked[:MAX_WORKFLOWS])


def update_from_experiment(
    results: Mapping[str, Any],
    optimizer: MMASACO,
    *,
    previous: WorkflowMemory | None = None,
    updated_at: str | None = None,
) -> WorkflowMemory:
    """Build memory from one ``run_search`` result of ``optimizer``.

    Pass ``previous`` only when ``optimizer`` was warm-started from it: its pheromones already
    carry the old state, and its workflows are merged (top 3 kept). A cold run starts afresh.
    """
    if results.get("optimizer") != MMASACO.name or not isinstance(optimizer, MMASACO):
        raise ValueError("memory is learned from an MMAS ACO run and its optimizer")
    key = _key_from_results(results, optimizer)
    if previous is not None and (bad := incompatibilities(previous, key)):
        raise IncompatibleMemory("refusing to merge: " + "; ".join(bad))
    synthetic = "synthetic" in key.evaluator_version
    pheromones = tuple(
        EdgePheromone(src=a, dst=b, tau=tau, label=f"{node_label(a)} -> {node_label(b)}")
        for (a, b), tau in optimizer.explored_pheromones().items()
    )
    old = previous.workflows if previous else ()
    return WorkflowMemory(
        key=key,
        source="synthetic" if synthetic else "real",
        updated_at=updated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        runs_merged=(previous.runs_merged if previous else 0) + 1,
        workflows=top_workflows([*old, *(_record(w) for w in results["workflows"])]),
        pheromones=pheromones,
        aco_epoch=optimizer.epoch,
    )


# -- warm start ------------------------------------------------------------------------------
def aco_from_memory(
    memory: WorkflowMemory | None, expected: MemoryKey, config: ACOConfig | None = None
) -> tuple[MMASACO, WarmStartReport]:
    """``MMASACO.from_memory`` equivalent. Warm only when compatible; otherwise cold + reasons."""
    config = config or ACOConfig()
    expected = expected.model_copy(update={"aco_config_hash": canonical_hash(asdict(config))})
    if memory is None:
        return MMASACO(config), WarmStartReport(
            "cold", (f"no memory for class {expected.task_class}",)
        )
    if not memory.key.prompt_template_version or not memory.key.compiler_version:
        return MMASACO(config), WarmStartReport("cold", ("memory lacks prompt/compiler identity",))
    if bad := incompatibilities(memory, expected):
        return MMASACO(config), WarmStartReport("cold", ("incompatible memory",) + bad)
    opt = MMASACO.from_pheromones(memory.pheromone_map(), config, epoch=memory.aco_epoch)
    return opt, WarmStartReport(
        "warm",
        (f"memory compatible ({memory.source}, {memory.runs_merged} run(s) merged)",),
        len(memory.pheromones),
    )


def run_aco(
    evaluate: EvaluateFn,
    train: Sequence[RuntimeTask],
    val: Sequence[RuntimeTask],
    *,
    start: Start,
    expected: MemoryKey,
    store: WorkflowMemoryStore,
    seed: int,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], MMASACO, WarmStartReport]:
    """One ACO search, cold or warm-started from ``store``."""
    if start == "cold":
        opt, report = (
            MMASACO(ACOConfig(lcb_z=config.lcb_z)),
            WarmStartReport("cold", ("cold start requested",)),
        )
    else:
        opt, report = aco_from_memory(
            store.load(expected.task_class), expected, ACOConfig(lcb_z=config.lcb_z)
        )
    return run_search(opt, evaluate, train, val, config, seed), opt, report


def _evals_to_reach(result: Mapping[str, Any], target: float) -> int | None:
    return next(
        (p["evaluations"] for p in result["curve"] if p["validation_fitness"] >= target), None
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Secondary demo: cold or warm ACO on the SYNTHETIC objective, then save memory.

    The real run needs Track A's runtime: call ``run_aco`` with ``make_evaluate_fn`` and the real
    ``MemoryKey`` (benchmark hash, model hash, evaluator version).
    """
    import argparse

    from benchmarks.loader import runtime_tasks
    from core.task_spec import TaskClass
    from experiments.synthetic import synthetic_evaluate
    from memory.store import DEFAULT_DIR
    from memory.summary import render

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--synthetic",
        action="store_true",
        required=True,
        help="required: only the labelled fake objective is wired here",
    )
    p.add_argument("--task-class", choices=["A", "B"], default="B")
    p.add_argument("--start", choices=["cold", "warm"], default="cold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument(
        "--compare",
        action="store_true",
        help="also run a cold search with the same seed (not saved) for comparison",
    )
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)

    cls = TaskClass(args.task_class)
    train, val = runtime_tasks("train", cls), runtime_tasks("validation", cls)
    store, key = WorkflowMemoryStore(args.dir), synthetic_key(cls.value)
    config = ExperimentConfig(budget=args.budget)
    kw = dict(expected=key, store=store, seed=args.seed, config=config)

    result, opt, report = run_aco(synthetic_evaluate, train, val, start=args.start, **kw)
    print(f"[SYNTHETIC objective - not a benchmark result] class {cls.value}, seed {args.seed}")
    print(
        f"start: {report.mode} ({'; '.join(report.reasons)}); edges loaded: {report.edges_loaded}"
    )
    print(
        f"{report.mode}: final validation fitness {result['validation_fitness']:.3f}, "
        f"pass rate {result['pass_rate']:.2f}, {result['workflow_evaluations']} evaluations"
    )
    if args.compare:
        cold, _, _ = run_aco(synthetic_evaluate, train, val, start="cold", **kw)
        target = min(result["validation_fitness"], cold["validation_fitness"])
        print(
            f"cold: final validation fitness {cold['validation_fitness']:.3f}, "
            f"pass rate {cold['pass_rate']:.2f}"
        )
        print(
            f"evaluations to reach {target:.3f}: {report.mode}={_evals_to_reach(result, target)}"
            f" cold={_evals_to_reach(cold, target)}"
        )
    if not args.no_save:
        # Merge only into the memory this run was warm-started from; a cold run replaces it.
        previous = store.load(cls.value) if report.mode == "warm" else None
        if previous is None and store.load(cls.value) is not None:
            print(f"{report.mode} start: replacing stored class {cls.value} memory (not merged)")
        memory = update_from_experiment(result, opt, previous=previous)
        print(f"saved {store.save(memory)}\n")
        print(render(memory))


if __name__ == "__main__":
    main()
