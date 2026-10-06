"""Canonical cross-benchmark matrix: fixed vs random vs ACO, from committed artifacts only.

    python -m experiments.external.matrix          # writes experiments/results/benchmark-matrix/

Every number is recomputed from a committed ``experiment.json.gz`` (both SHA-256 digests
verified by ``load_compact``); nothing is rerun and nothing is copied by hand. MuSiQue enters as
immutable prior evidence (protocol-v2, PR #39). The winner rule is the one pre-registered in the
MMLU-Pro protocol: the highest mean champion validation score over seeds (ties listed jointly).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.external.adapter import Benchmark, build_contract, load_protocol
from experiments.external.mmlu_pro import MMLU_PRO
from experiments.external.musique import MUSIQUE
from experiments.external.run import champion_optimization_score, fmt, workflow_label
from experiments.optimization_experiment import load_compact, summarize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "benchmark-matrix"
MATRIX_SCHEMA = "wynk-benchmark-matrix/1"
STRATEGIES = ("fixed", "random", "aco")


@dataclass(frozen=True)
class Entry:
    bench: Benchmark
    protocol: str
    status: str

    @property
    def directory(self) -> Path:
        return self.bench.results_dir / f"protocol-{self.protocol}"


ENTRIES = (
    Entry(MUSIQUE, "v2", "prior evidence (PR #39), immutable: not rerun"),
    Entry(MMLU_PRO, "v1", "this PR"),
)


def admissible_workflows(bench: Benchmark, protocol: Mapping[str, Any]) -> int:
    """How many workflows the runtime admits for the benchmark's contract. Admission reads the
    contract's shape and limits only, so a one-row stand-in dataset gives the exact count."""
    from runtime.runner import WorkflowRunner

    row = {c: "x" for c in bench.columns.all}
    contract = build_contract(bench, (json.dumps(row) + "\n").encode(), protocol)
    checker = WorkflowRunner(model=None, benchmark_hash="count").checker
    return sum(1 for _ in checker.enumerate_admissible(contract))


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def entry_row(entry: Entry, *, count_space: bool = True) -> dict[str, Any]:
    summary, artifact = load_compact(entry.directory)  # verifies both recorded digests
    protocol = load_protocol(entry.bench, entry.protocol)
    by = summarize(artifact["runs"])["by_strategy"]
    strategies: dict[str, Any] = {}
    for s in STRATEGIES:
        m = by[s]["metrics"]
        runs = [r for r in artifact["runs"] if r["strategy"] == s]
        strategies[s] = {
            "validation_mean": m["champion_validation_score"]["mean"],
            "validation_std": m["champion_validation_score"]["std"],
            "validation_per_seed": [p["champion_validation_score"] for p in by[s]["per_seed"]],
            "champion_optimization_mean": _mean([champion_optimization_score(r) for r in runs]),
            "champion_workflows": sorted(
                {workflow_label(r["champion"]["genome"]) for r in runs if r["champion"]}
            ),
            "candidates": m["candidate_evaluations"]["mean"],
            "distinct_workflows": _mean([r["distinct_genomes"] for r in runs]),
            "model_calls": m["model_calls"]["mean"],
            "tokens": m["tokens"]["mean"],
            "tokens_per_example": m["tokens_per_example"]["mean"],
            "execution_s": m["execution_s"]["mean"],
            "e2e_wall_s": m["e2e_wall_s"]["mean"],
            "p95_latency_s": m["p95_latency_s"]["mean"],
            "optimizer_overhead_s": m["optimizer_overhead_s"]["mean"],
            "cost": m["cost"]["mean"],
        }
    means = {s: v["validation_mean"] for s, v in strategies.items()}
    top = max(means.values())
    fixed = strategies["fixed"]
    return {
        "benchmark": entry.bench.key,
        "title": entry.bench.title,
        "task_family": entry.bench.task_family,
        "metric": entry.bench.metric,
        "protocol": entry.protocol,
        "status": entry.status,
        "experiment_id": artifact["experiment_id"],
        "artifact_gz_sha256": summary["artifact"]["gz_sha256"],
        "fixed_rule": artifact["fairness"]["fixed_baseline_rule"],
        "model_hash": artifact["model_hashes"],
        "splits": artifact["splits"],
        "test_runs": artifact["test_runs"],
        "seeds": len(by["fixed"]["seeds"]),
        "candidate_budget": protocol["budget"]["max_candidate_evaluations"],
        "context_columns": bool(entry.bench.columns.context),
        "admissible_workflows": (
            admissible_workflows(entry.bench, protocol) if count_space else None
        ),
        "strategies": strategies,
        # ties = equal up to float rounding (real differences are >= 1/rows, far above 1e-9)
        "winner": sorted(s for s, v in means.items() if math.isclose(v, top, abs_tol=1e-9)),
        "search_vs_fixed": {
            s: {
                "validation_delta": strategies[s]["validation_mean"] - fixed["validation_mean"],
                "token_ratio": strategies[s]["tokens"] / fixed["tokens"],
                "call_ratio": strategies[s]["model_calls"] / fixed["model_calls"],
                "e2e_ratio": strategies[s]["e2e_wall_s"] / fixed["e2e_wall_s"],
            }
            for s in ("random", "aco")
        },
        "aco_minus_random": strategies["aco"]["validation_mean"]
        - strategies["random"]["validation_mean"],
    }


def build_matrix(*, count_space: bool = True) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "winner_rule": "highest mean champion validation score over seeds (ties listed jointly)",
        "rows": [entry_row(e, count_space=count_space) for e in ENTRIES],
    }


def _sfr(row: Mapping[str, Any], key: str, digits: int = 0) -> str:
    s = row["strategies"]
    return " / ".join(fmt(float(s[x][key]), digits) for x in STRATEGIES)


def _score(v: Mapping[str, Any]) -> str:
    return f"{v['validation_mean']:.3f} ± {v['validation_std']:.3f}"


def matrix_table(matrix: Mapping[str, Any]) -> str:
    lines = [
        "| Benchmark | Task family | Fixed | Random | ACO | Calls (F / R / A) |"
        " Tokens (F / R / A) | E2E s (F / R / A) | Winner |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in matrix["rows"]:
        s = r["strategies"]
        lines.append(
            f"| {r['title']} ({r['metric']}, protocol-{r['protocol']}) | {r['task_family']} |"
            f" {_score(s['fixed'])} | {_score(s['random'])} | {_score(s['aco'])} |"
            f" {_sfr(r, 'model_calls')} | {_sfr(r, 'tokens')} | {_sfr(r, 'e2e_wall_s')} |"
            f" {', '.join(r['winner'])} |"
        )
    return "\n".join(lines)


def detail_table(matrix: Mapping[str, Any]) -> str:
    lines = [
        "| Benchmark | context columns | admissible workflows | candidate budget |"
        " champion on optimization rows (F / R / A) | tokens/example (F / R / A) |"
        " exec s (F / R / A) | random-fixed | ACO-fixed | ACO-random |"
        " search tokens x fixed (R / A) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in matrix["rows"]:
        sv = r["search_vs_fixed"]
        opt = " / ".join(fmt(r["strategies"][x]["champion_optimization_mean"]) for x in STRATEGIES)
        lines.append(
            f"| {r['title']} | {'yes' if r['context_columns'] else 'no'} |"
            f" {fmt(r['admissible_workflows'])} | {r['candidate_budget']} | {opt} |"
            f" {_sfr(r, 'tokens_per_example')} | {_sfr(r, 'execution_s')} |"
            f" {sv['random']['validation_delta']:+.3f} | {sv['aco']['validation_delta']:+.3f} |"
            f" {r['aco_minus_random']:+.3f} |"
            f" {sv['random']['token_ratio']:.1f}x / {sv['aco']['token_ratio']:.1f}x |"
        )
    return "\n".join(lines)


def write(matrix: Mapping[str, Any], out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(
        json.dumps(matrix, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "TABLES.md").write_text(
        "<!-- generated by `python -m experiments.external.matrix`; do not edit -->\n\n"
        "## Benchmark matrix\n\n"
        + matrix_table(matrix)
        + "\n\nScores: mean ± sample std of the champion's validation score over seeds."
        " Calls / tokens / E2E: mean per strategy run (one seed). Cost: not reported (no"
        " authoritative pricing in the runtime).\n\n## Search characteristics\n\n"
        + detail_table(matrix)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    matrix = build_matrix()
    write(matrix, args.out)
    print(matrix_table(matrix))
    print()
    print(detail_table(matrix))


if __name__ == "__main__":
    main()
