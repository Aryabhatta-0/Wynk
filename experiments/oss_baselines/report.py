"""Aggregate ``runs.jsonl`` into the comparison table. Pure function of the recorded executions.

Metrics are fixed in advance (README "Metrics"); none is chosen after looking at results:

  success_rate          evaluator verdict PASS (fields match + valid evidence + within caps)
  fitness               evaluator fitness (fitness/mvp-2; INFEASIBLE = -1)          PRIMARY
  answer_correct_rate   all fields match ground truth, caps ignored                 diagnostic
  field_accuracy        fraction of fields matching, caps ignored                    diagnostic
  evidence_valid_rate   fraction of fields citing >=1 valid span, caps ignored
  constraint_pass_rate  run within every task cap (tokens, wall time, tool calls, retries)
  failure_rate          no usable answer: TIMEOUT / RUNTIME_ERROR / INVALID_OUTPUT
  latency, tokens, cost, llm_calls, tool_calls   per execution (metering proxy / runner clock)

Best external baseline = the external system with the highest mean fitness over all tasks.
Relative changes are only shown where the baseline value is > 0 (a ratio against 0 or a negative
fitness is meaningless); absolute differences are always shown.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

NO_ANSWER = {"TIMEOUT", "RUNTIME_ERROR", "INVALID_OUTPUT"}
OVER_CAPS = {"BUDGET_EXCEEDED", "TIMEOUT"}
ORDER = ["smolagents", "crewai", "llamaindex", "langgraph", "wynk"]
BOOTSTRAP = 2000


def load_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "runs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _per_task(rows: list[dict[str, Any]], key) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(key(r))
    return {t: sum(v) / len(v) for t, v in by.items()}


def bootstrap_ci(per_task: dict[str, float], seed: int = 0) -> tuple[float, float] | None:
    """95% percentile CI of the mean over tasks (tasks resampled; runs averaged per task)."""
    vals = list(per_task.values())
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tel = [r["telemetry"] for r in rows]
    statuses: dict[str, int] = defaultdict(int)
    for r in rows:
        statuses[r["status"]] += 1
    fit_task = _per_task(rows, lambda r: r["fitness"])
    pass_by_task: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        pass_by_task[r["task_id"]].append(r["verdict"] == "PASS")
    lat = [r["latency_s"] for r in rows]
    return {
        "executions": n,
        "tasks": len(fit_task),
        "success_rate": sum(r["verdict"] == "PASS" for r in rows) / n,
        "fitness": sum(r["fitness"] for r in rows) / n,
        "fitness_ci95": bootstrap_ci(fit_task),
        "answer_correct_rate": sum(r["uncapped"]["all_fields_matched"] for r in rows) / n,
        "field_accuracy": sum(r["uncapped"]["field_accuracy"] for r in rows) / n,
        "evidence_valid_rate": sum(r["uncapped"]["evidence_valid_fraction"] for r in rows) / n,
        "quote_supports_value": _mean([r["quote_supports_value_fraction"] for r in rows]),
        "constraint_pass_rate": sum(r["status"] not in OVER_CAPS for r in rows) / n,
        "failure_rate": sum(r["status"] in NO_ANSWER for r in rows) / n,
        "all_runs_pass_task_rate": sum(all(v) for v in pass_by_task.values()) / len(pass_by_task),
        "latency_mean_s": sum(lat) / n,
        "latency_median_s": statistics.median(lat),
        "tokens_mean": _mean([t["total_tokens"] for t in tel]),
        "tokens_missing": sum(t["total_tokens"] is None for t in tel),
        "cost_mean_usd": _mean([t["cost_usd"] for t in tel]),
        "cost_missing": sum(t["cost_usd"] is None for t in tel),
        "llm_calls_mean": _mean([t["llm_calls"] for t in tel]),
        "tool_calls_mean": _mean([t["tool_calls"] for t in tel]),
        "status_counts": dict(sorted(statuses.items())),
    }


HIGHER = [
    "fitness",
    "success_rate",
    "answer_correct_rate",
    "evidence_valid_rate",
    "constraint_pass_rate",
]
LOWER = ["failure_rate", "latency_mean_s", "tokens_mean", "cost_mean_usd"]


def compare(wynk: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in HIGHER + LOWER:
        w, b = wynk.get(m), base.get(m)
        if w is None or b is None:
            out[m] = {"wynk": w, "baseline": b, "diff": None, "relative": None}
            continue
        rel = None
        if b > 0:
            rel = (w - b) / b if m in HIGHER else (b - w) / b  # LOWER: positive = reduction
        out[m] = {
            "wynk": w,
            "baseline": b,
            "diff": w - b,
            "relative": rel,
            "relative_meaning": (
                "change, higher is better" if m in HIGHER else "reduction, positive = Wynk lower"
            ),
        }
    return out


def paired_fitness_diff(rows_w, rows_b) -> dict[str, Any]:
    fw, fb = _per_task(rows_w, lambda r: r["fitness"]), _per_task(rows_b, lambda r: r["fitness"])
    common = sorted(set(fw) & set(fb))
    diffs = {t: fw[t] - fb[t] for t in common}
    return {
        "tasks": len(common),
        "mean_diff": sum(diffs.values()) / len(diffs) if diffs else None,
        "ci95": bootstrap_ci(diffs, seed=1),
        "tasks_wynk_better": sum(d > 0 for d in diffs.values()),
        "tasks_tied": sum(d == 0 for d in diffs.values()),
        "tasks_wynk_worse": sum(d < 0 for d in diffs.values()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems = [s for s in ORDER if any(r["system"] == s for r in rows)]
    labels = {r["system"]: r["label"] for r in rows}
    scopes = {
        "all": rows,
        "A": [r for r in rows if r["task_class"] == "A"],
        "B": [r for r in rows if r["task_class"] == "B"],
    }
    tables = {
        scope: {
            s: aggregate([r for r in rs if r["system"] == s])
            for s in systems
            if any(r["system"] == s for r in rs)
        }
        for scope, rs in scopes.items()
        if rs
    }
    summary: dict[str, Any] = {"systems": systems, "labels": labels, "tables": tables}
    externals = [s for s in systems if s != "wynk"]
    if "wynk" in systems and externals:
        best = max(externals, key=lambda s: tables["all"][s]["fitness"])
        summary["best_external_baseline"] = best
        summary["comparisons"] = {
            scope: {s: compare(t["wynk"], t[s]) for s in externals if s in t and "wynk" in t}
            for scope, t in tables.items()
        }
        summary["paired_fitness"] = {
            s: paired_fitness_diff(
                [r for r in rows if r["system"] == "wynk"], [r for r in rows if r["system"] == s]
            )
            for s in externals
        }
    return summary


def _f(x: Any, pct: bool = False, nd: int = 3) -> str:
    if x is None:
        return "null"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def markdown(summary: dict[str, Any], meta: dict[str, Any] | None) -> str:
    labels = summary["labels"]
    lines = ["# Wynk frozen workflow vs. OSS starter workflows", ""]
    if meta:
        lines += [
            "- mode: CONTROLLED_SOURCE (same frozen pages for every system)",
            f"- model: {meta['manifest']['model']['name']} (temperature 0, max_tokens 1024/call)",
            f"- tasks: held-out TEST split ({len(meta['task_ids'])} tasks), runs per task: "
            f"{meta.get('runs_per_task')}",
            f"- evaluator: {meta['manifest']['evaluator']}",
            "",
        ]
    for scope, table in summary["tables"].items():
        title = {
            "all": "All held-out tasks",
            "A": "Class A (fact sheets)",
            "B": "Class B (inventory API)",
        }[scope]
        lines += [
            f"## {title}",
            "",
            "| Workflow | n | Success | Fitness (95% CI) | Answer correct* | Evidence valid* "
            "| Within caps | Failure | Latency mean (s) | Tokens mean | Cost mean (USD) "
            "| LLM calls | Tool calls |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s, a in table.items():
            ci = a["fitness_ci95"]
            ci_s = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            cost = None if a["cost_mean_usd"] is None else f"{a['cost_mean_usd']:.6f}"
            lines.append(
                f"| {labels[s]} | {a['executions']} | {_f(a['success_rate'], True)} "
                f"| {a['fitness']:.3f}{ci_s} | {_f(a['answer_correct_rate'], True)} "
                f"| {_f(a['evidence_valid_rate'], True)} | {_f(a['constraint_pass_rate'], True)} "
                f"| {_f(a['failure_rate'], True)} | {a['latency_mean_s']:.1f} "
                f"| {_f(a['tokens_mean'], nd=0)} | {cost or 'null'} "
                f"| {_f(a['llm_calls_mean'], nd=1)} | {_f(a['tool_calls_mean'], nd=1)} |"
            )
        lines += [
            "",
            "Status counts: "
            + "; ".join(f"{labels[s]}: {a['status_counts']}" for s, a in table.items()),
            "",
        ]
    lines += ["\\* caps ignored (diagnostic); every other column is under the task's own caps.", ""]
    if "best_external_baseline" in summary:
        best = summary["best_external_baseline"]
        c = summary["comparisons"]["all"][best]
        lines += [
            "## Wynk vs. strongest starter baseline",
            "",
            f"Strongest external starter workflow (highest mean fitness): **{labels[best]}**",
            "",
            "| Metric | Starter | Wynk | Diff | Relative |",
            "|---|---|---|---|---|",
        ]
        for m, v in c.items():
            rel = (
                "n/a"
                if v["relative"] is None
                else f"{100 * v['relative']:+.1f}% ({v['relative_meaning']})"
            )
            lines.append(
                f"| {m} | {_f(v['baseline'])} | {_f(v['wynk'])} | {_f(v['diff'])} | {rel} |"
            )
        lines += [
            "",
            "## Paired per-task fitness difference (Wynk - starter)",
            "",
            "| Starter | Tasks | Mean diff | 95% CI | Wynk better / tied / worse |",
            "|---|---|---|---|---|",
        ]
        for s, p in summary["paired_fitness"].items():
            ci = p["ci95"]
            lines.append(
                f"| {labels[s]} | {p['tasks']} | {_f(p['mean_diff'])} "
                f"| {'n/a' if not ci else f'[{ci[0]:.3f}, {ci[1]:.3f}]'} "
                f"| {p['tasks_wynk_better']} / {p['tasks_tied']} / {p['tasks_wynk_worse']} |"
            )
    return "\n".join(lines) + "\n"


def write_report(out: Path) -> dict[str, Any]:
    rows = load_rows(out)
    summary = summarize(rows)
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    md = markdown(summary, meta)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return summary
