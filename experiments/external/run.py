"""Fixed vs random vs ACO on any registered external benchmark, with a REAL model backend.

    python -m experiments.external.run prepare  --benchmark mmlu-pro --protocol v1 --source F
    python -m experiments.external.run run      --benchmark mmlu-pro --protocol v1 --source F
                                                [--seeds 0 1] [--env-file .env]
    python -m experiments.external.run assemble --benchmark mmlu-pro --protocol v1 --source F

* ``prepare`` rebuilds the subset from the official file and checks it against the committed
  manifest; ``--freeze`` writes that manifest once, before any strategy has run.
* ``run`` executes ``run_strategy`` for each (seed, strategy) of the frozen protocol, writing
  each record to ``DIR/runs/<strategy>/seed-<n>.json`` (regenerable, never committed). It needs
  a configured OpenAI-compatible backend; without one it exits ``BLOCKED``. The client's model
  hash must equal the protocol's before anything runs, and the leakage gate must pass.
* ``assemble`` builds the canonical artifact (``experiment.json.gz``, deterministic bytes) and a
  SMALL ``summary.json`` (identity once, per-seed metrics, aggregates, curves), ``REPORT.md`` and
  ``learning_curves.png``. Nothing per-candidate is committed uncompressed.

Every command refuses a protocol whose canonical hash differs from its lock entry.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.external.adapter import (
    Benchmark,
    Prepared,
    check_isolation,
    check_manifest,
    frozen_manifest,
    load_protocol,
    manifest_path,
    plan_from,
    prepare,
)
from experiments.external.mmlu_pro import MMLU_PRO
from experiments.external.musique import MUSIQUE
from experiments.optimization_experiment import (
    SUMMARY_METRICS,
    ExperimentPlan,
    assemble,
    dumps,
    resource_curves,
    run_strategy,
    searchable_evaluator,
    summarize,
    write_compact,
)

BENCHMARKS: dict[str, Benchmark] = {b.key: b for b in (MUSIQUE, MMLU_PRO)}
BLOCKED = 3  # exit code: no real backend configured
SUMMARY_SCHEMA = "wynk-benchmark-summary/1"


def frozen(bench: Benchmark, source: Path, protocol: Mapping[str, Any]) -> Prepared:
    prepared = prepare(bench, source, protocol)
    if not manifest_path(bench).is_file():
        raise SystemExit(f"no frozen manifest for {bench.key}: run `prepare --freeze` first")
    check_manifest(prepared.manifest, frozen_manifest(bench))
    return prepared


def results_dir(bench: Benchmark, version: str) -> Path:
    return bench.results_dir / f"protocol-{version}"


def real_runner(env_file: Path | None):
    from api.chat import load_env_file
    from runtime.gemma_client import GemmaConfig, ModelUnavailableError, OpenAICompatibleClient
    from runtime.runner import WorkflowRunner

    try:
        config = GemmaConfig.from_env(load_env_file(env_file))
    except ModelUnavailableError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(BLOCKED) from exc
    client = OpenAICompatibleClient(config)
    return WorkflowRunner(model=client, benchmark_hash="inline"), client


# -- commands -----------------------------------------------------------------------------------
def cmd_prepare(bench: Benchmark, args) -> None:
    protocol = load_protocol(bench, args.protocol)
    p = prepare(bench, args.source, protocol)
    path = manifest_path(bench)
    if args.freeze:
        if path.is_file():
            raise SystemExit("manifest is already frozen; it is never rewritten")
        path.write_text(dumps(p.manifest), encoding="utf-8")
    else:
        check_manifest(p.manifest, frozen_manifest(bench))
    counts = {s.role.value: len(s.row_ids) for s in p.splits.splits}
    print(f"manifest {p.manifest['manifest_hash']} rows {p.contract.dataset.row_count} {counts}")


def cmd_run(bench: Benchmark, args) -> None:
    protocol = load_protocol(bench, args.protocol)
    p = frozen(bench, args.source, protocol)
    plan = plan_from(protocol, p.manifest["manifest_hash"])
    runner, client = real_runner(args.env_file)
    if client.model_hash != plan.expected_model_hash:  # bound before ANY call
        raise SystemExit(
            f"model hash {client.model_hash} != frozen {plan.expected_model_hash}: refusing to run"
        )

    def run_workflow(genome, task, trial, seed):
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    suite, evaluate, version = searchable_evaluator(p.contract, p.splits, p.data, run_workflow)
    check_isolation(bench, suite)  # leakage gate, before any model call
    for seed in args.seeds or list(plan.seeds):
        for strategy in plan.strategies:
            path = args.out / "runs" / strategy.value / f"seed-{seed}.json"
            if path.is_file():
                print(f"skip {path} (exists)")
                continue
            record = run_strategy(
                plan,
                strategy,
                seed,
                suite,
                evaluate,
                checker=runner.checker,
                evaluator_version=version,
                workers=protocol["execution"]["workers"],
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps({"evaluator_version": version, **record}), encoding="utf-8")
            champ = record["champion"]
            print(
                f"{strategy.value} seed {seed}: {record['usage']['candidate_evaluations']} "
                f"candidates, stop={record['stop_reason']}, champion "
                f"{champ['validation']['score_mean'] if champ else None}",
                flush=True,
            )


def build_artifact(bench: Benchmark, plan: ExperimentPlan, p: Prepared, runs_dir: Path) -> dict:
    """The canonical artifact from the per-run records in ``runs_dir``."""
    from experiments.contract_run import contract_suite

    records, versions = [], set()
    for strategy in plan.strategies:
        for seed in plan.seeds:
            path = runs_dir / strategy.value / f"seed-{seed}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            versions.add(rec.pop("evaluator_version"))
            records.append(rec)
    if len(versions) != 1:
        raise SystemExit(f"runs disagree on the evaluator version: {versions}")
    suite, _ = contract_suite(p.contract, p.splits, p.data)
    return assemble(
        plan,
        suite,
        records,
        evaluator_version=versions.pop(),
        synthetic=False,
        provenance=p.manifest,
    )


def cmd_assemble(bench: Benchmark, args) -> None:
    protocol = load_protocol(bench, args.protocol)
    p = frozen(bench, args.source, protocol)
    plan = plan_from(protocol, p.manifest["manifest_hash"])
    artifact = build_artifact(bench, plan, p, args.out / "runs")
    meta = write_compact(artifact, args.out)["artifact"]  # experiment.json.gz (+ sha256)
    summary = benchmark_summary(bench, args.protocol, artifact, meta)
    (args.out / "summary.json").write_text(compact_dumps(summary), encoding="utf-8")
    plot_curves(artifact, args.out / "learning_curves.png", curve_title(bench, args.protocol))
    text = report(bench, summary)
    (args.out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


# -- compact summary ----------------------------------------------------------------------------
CURVE_COLUMNS = (
    "evaluation",
    "cumulative_model_calls",
    "cumulative_tokens",
    "cumulative_execution_s",
    "cumulative_e2e_wall_s",
    "cumulative_cost",
    "best_so_far_score",
)


def workflow_label(genome: Mapping[str, Any]) -> str:
    """``DIRECT(cot) -> VERIFY(schema_check,retry-1)``: a genome's stages and options."""
    parts = []
    for stage in genome["stages"]:
        opts = ",".join(str(v) for k, v in stage.items() if k != "kind")
        parts.append(f"{stage['kind']}({opts})")
    return " -> ".join(parts)


def champion_optimization_score(run: Mapping[str, Any]) -> float | None:
    """Mean score of the champion on the OPTIMIZATION rows (all its evaluations). Selection
    uses validation rows only, so this is free of the selection's winner's curse."""
    champ = run["champion"]
    if champ is None:
        return None
    scores = [
        e["score"] or 0.0
        for c in run["candidates"]
        if c["genome_hash"] == champ["genome_hash"]
        for e in c["runs"]
        if e["split"] == "optimization" and e["verdict"] is not None
    ]
    return statistics.fmean(scores) if scores else None


def failure_counts(run: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in run["candidates"]:
        for e in c["runs"]:
            if e["failure"]:
                out[e["failure"]] = out.get(e["failure"], 0) + 1
    return dict(sorted(out.items()))


def benchmark_summary(
    bench: Benchmark, version: str, artifact: Mapping[str, Any], meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Small, reviewable summary: identity once, per-seed rows, aggregates and curves. The
    per-candidate detail lives only in ``experiment.json.gz`` (verified by ``meta``)."""
    agg = summarize(artifact["runs"])
    per_seed = []
    for strategy, v in agg["by_strategy"].items():
        runs = {r["seed"]: r for r in artifact["runs"] if r["strategy"] == strategy}
        for row in v["per_seed"]:
            run = runs[row["seed"]]
            champ = run["champion"]
            per_seed.append(
                {"strategy": strategy}
                | row
                | {
                    "champion_workflow": workflow_label(champ["genome"]) if champ else None,
                    "champion_optimization_score": champion_optimization_score(run),
                    "distinct_workflows": run["distinct_genomes"],
                    "workflow_runs": run["usage"]["workflow_runs"],
                    "failures": failure_counts(run),
                }
            )
    return {
        "schema": SUMMARY_SCHEMA,
        "benchmark": bench.key,
        "title": bench.title,
        "task_family": bench.task_family,
        "metric": bench.metric,
        "protocol": version,
        "protocol_id": artifact["identity"]["plan"].get("protocol_id"),
        "experiment_id": artifact["experiment_id"],
        "manifest_hash": (artifact.get("provenance") or {}).get("manifest_hash"),
        "artifact": dict(meta),
        "identity": artifact["identity"],
        "fairness": artifact["fairness"],
        "splits": artifact["splits"],
        "test_runs": artifact["test_runs"],
        "model_hashes": artifact["model_hashes"],
        "per_seed": per_seed,
        "aggregate": {
            s: {"seeds": v["seeds"], "metrics": v["metrics"]} for s, v in agg["by_strategy"].items()
        },
        "highest_mean_champion_validation_score": agg["highest_mean_champion_validation_score"],
        "curves": {
            "columns": list(CURVE_COLUMNS),
            "runs": {
                f"{r['strategy']}/seed-{r['seed']}": [
                    [p[c] for c in CURVE_COLUMNS] for p in r["curve"]
                ]
                for r in artifact["runs"]
            },
        },
    }


def compact_dumps(obj: Mapping[str, Any]) -> str:
    """JSON with one top-level key per line (and one entry per line for lists/maps of rows):
    valid, deterministic and a few dozen lines instead of thousands."""

    def one(v: Any) -> str:
        return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    lines = []
    for k in sorted(obj):
        v = obj[k]
        if isinstance(v, list) and v and isinstance(v[0], dict):
            inner = ",\n  ".join(one(x) for x in v)
            lines.append(f"{one(k)}:[\n  {inner}\n ]")
        elif isinstance(v, dict) and k in ("aggregate", "curves"):
            inner = ",\n  ".join(f"{one(kk)}:{one(v[kk])}" for kk in sorted(v))
            lines.append(f"{one(k)}:{{\n  {inner}\n }}")
        else:
            lines.append(f"{one(k)}:{one(v)}")
    return "{\n " + ",\n ".join(lines) + "\n}\n"


# -- report + curves ----------------------------------------------------------------------------
def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


PER_SEED_COLUMNS = (
    ("stop", "stop_reason", None),
    ("cand", "candidate_evaluations", None),
    ("val", "champion_validation_score", 3),
    ("opt (champion)", "champion_optimization_score", 3),
    ("calls", "model_calls", None),
    ("prompt tok", "prompt_tokens", None),
    ("compl tok", "completion_tokens", None),
    ("tokens", "tokens", None),
    ("tok/ex", "tokens_per_example", 0),
    ("lat mean", "mean_latency_s", 1),
    ("lat p50", "p50_latency_s", 1),
    ("lat p95", "p95_latency_s", 1),
    ("lat max", "max_latency_s", 1),
    ("exec s", "execution_s", 0),
    ("E2E s", "e2e_wall_s", 0),
    ("opt ovh s", "optimizer_overhead_s", 3),
    ("eval ovh s", "evaluator_overhead_s", 3),
    ("ex/s", "examples_per_s", 2),
    ("cost", "cost", 3),
)


def report(bench: Benchmark, summary: Mapping[str, Any]) -> str:
    prob = summary["identity"]["problem"]
    m = bench.metric
    lines = [
        f"# {bench.title} protocol-{summary['protocol']}: fixed vs random vs ACO (real model)",
        "",
        f"- task family: {bench.task_family}; metric: validation {m}",
        f"- experiment_id `{summary['experiment_id']}`; protocol_id `{summary['protocol_id']}`",
        f"- fixed baseline rule `{summary['fairness']['fixed_baseline_rule']}`",
        f"- model `{prob['model']['model']}` model_hash `{prob['model_hash']}`"
        f" prompts `{prob['prompt_template_version']}`",
        f"- manifest `{summary['manifest_hash']}` dataset `{prob['dataset_content_hash']}`"
        f" splits `{prob['splits_hash']}` contract `{prob['task_contract_hash']}`",
        f"- splits {summary['splits']}; test rows executed: {summary['test_runs']}",
        f"- evaluator `{prob['evaluator_run_version']}` config {prob['evaluator_config']}",
        "- latency = runtime-measured workflow time per run; exec s = summed workflow time;"
        " E2E s = strategy wall-clock (the 3 seeds ran as 3 concurrent processes); cost is n/a"
        " (no authoritative pricing)",
        f"- opt (champion) = the champion's {m} on the optimization rows, which champion"
        " selection never reads (no best-of-N selection bias)",
        f"- artifact `{summary['artifact']['file']}` sha256 `{summary['artifact']['gz_sha256']}`",
        "",
        "## Per seed",
        "",
        "| strategy | seed | champion workflow | "
        + " | ".join(c[0] for c in PER_SEED_COLUMNS)
        + " | failures |",
        "|" + "---|" * (len(PER_SEED_COLUMNS) + 4),
    ]
    for p in summary["per_seed"]:
        cells = [fmt(p.get(k), d if d is not None else 3) for _, k, d in PER_SEED_COLUMNS]
        lines.append(
            f"| {p['strategy']} | {p['seed']} | {p['champion_workflow']} | "
            + " | ".join(cells)
            + f" | {p['failures'] or '-'} |"
        )
    lines += [
        "",
        "## Aggregate over seeds (mean / std / min / max)",
        "",
        "| strategy | metric | mean | std | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, v in summary["aggregate"].items():
        for name in SUMMARY_METRICS:
            s = v["metrics"][name]
            lines.append(
                f"| {strategy} | {name} | {fmt(s['mean'])} | {fmt(s['std'])} |"
                f" {fmt(s['min'])} | {fmt(s['max'])} |"
            )
    lines += [
        "",
        "Highest mean champion validation "
        + m
        + ": "
        + ", ".join(summary["highest_mean_champion_validation_score"]),
    ]
    return "\n".join(lines) + "\n"


CURVE_AXES_TITLES = {
    "evaluation": "candidate evaluations",
    "cumulative_model_calls": "cumulative model calls",
    "cumulative_tokens": "cumulative tokens",
    "cumulative_execution_s": "cumulative execution time (s)",
    "cumulative_e2e_wall_s": "cumulative end-to-end wall time (s)",
    "cumulative_cost": "cumulative cost",
}
STRATEGY_COLORS = {"fixed": "#7f7f7f", "random": "#1f77b4", "aco": "#d62728"}


def curve_title(bench: Benchmark, version: str) -> str:
    return f"{bench.title} protocol-{version}: fixed vs random vs ACO, validation {bench.metric}"


def plot_curves(
    artifact: Mapping[str, Any], path: Path, title: str, ylabel: str = "best-so-far validation"
) -> None:
    """Best-so-far validation score against every resource axis, one line per (strategy, seed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (axis, name) in zip(axs.flat, CURVE_AXES_TITLES.items(), strict=True):
        drawn = False
        for run in artifact["runs"]:
            pts = resource_curves(run).get(axis)
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            ax.step(
                xs,
                ys,
                where="post",
                color=STRATEGY_COLORS[run["strategy"]],
                alpha=0.7,
                marker="o",
                label=f"{run['strategy']} s{run['seed']}",
            )
            drawn = True
        ax.set_title(name if drawn else f"{name} (not available)")
        ax.set_xlabel(name)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run", "assemble"):
        sp = sub.add_parser(name)
        sp.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
        sp.add_argument("--protocol", required=True)
        sp.add_argument("--source", type=Path, required=True)
        if name == "prepare":
            sp.add_argument("--freeze", action="store_true")
        if name == "run":
            sp.add_argument("--env-file", type=Path, default=Path(".env"))
            sp.add_argument("--seeds", type=int, nargs="*")
        if name in ("run", "assemble"):
            sp.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    bench = BENCHMARKS[args.benchmark]
    if getattr(args, "out", "unset") is None:
        args.out = results_dir(bench, args.protocol)
    {"prepare": cmd_prepare, "run": cmd_run, "assemble": cmd_assemble}[args.cmd](bench, args)


if __name__ == "__main__":
    main()
