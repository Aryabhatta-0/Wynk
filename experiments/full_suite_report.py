"""Reports for ``experiments.full_suite``: per-task rows, seed-level statistics, ACO vs random.

Everything here is recomputed from the run directory (``protocol.json``, ``search.json``,
``final.json``), so ``python -m experiments.full_suite report --out <dir>`` rebuilds it offline.

Statistics are seed-level. Within one (optimizer, seed), a group's value is the mean over its
tasks; across seeds we report the mean, the SD and a 95% Student-t CI. ACO vs random is paired
by seed (same seed index, same tasks), with an exact two-sided sign-flip permutation test on the
per-seed differences. With n seeds the smallest attainable p is 2 / 2^n (0.0625 for 5 seeds), so
a protocol with few seeds can never be ``significant`` at 0.05 - which is reported, not hidden.
Per-task wins / losses / ties are listed task by task and every ACO loss is listed by name, so
a regression on one task stays visible even when the mean improves. A task-level exact sign
test over wins vs losses is reported as a second, separate test.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.legacy_adapter import legacy_suite
from benchmarks.loader import BENCH_DIR, HELDOUT_DIR, load_catalog, load_task_specs
from core.dataset import SplitRole, SplitUse

SPLITS = ("train", "validation", "test")
METRICS = (
    "pass_rate",
    "fitness_mean",
    "score_mean",
    "tokens_mean",
    "model_calls_mean",
    "tool_calls_mean",
    "wall_time_s_mean",
)
TIE_TOL = 1e-9
ALPHA = 0.05
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042)  # fmt: skip


# -- statistics -----------------------------------------------------------------------------------
def t95(df: int) -> float:
    return _T95[df - 1] if 1 <= df <= len(_T95) else 1.96


def describe(values: Sequence[float]) -> dict[str, Any]:
    """Mean, SD and 95% t-CI of seed-level observations (CI is ``None`` below two seeds)."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None, "values": []}
    mean = statistics.fmean(xs)
    if n < 2:
        return {"n": 1, "mean": mean, "sd": None, "ci95": None, "values": xs}
    sd = statistics.stdev(xs)
    half = t95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half], "values": xs}


def sign_flip_p(diffs: Sequence[float]) -> float | None:
    """Exact two-sided paired permutation p-value (mean difference, all 2^n sign flips)."""
    d = [float(x) for x in diffs]
    if not d:
        return None
    if len(d) > 16:
        raise ValueError("exact sign-flip test is enumerated; at most 16 paired seeds")
    observed = abs(statistics.fmean(d))
    if observed <= TIE_TOL:
        return 1.0
    hits = sum(
        abs(statistics.fmean(s * x for s, x in zip(signs, d, strict=True))) >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=len(d))
    )
    return hits / 2 ** len(d)


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test of wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def min_attainable_p(n: int) -> float | None:
    return 2 / 2**n if n else None


# -- per-task rows --------------------------------------------------------------------------------
def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def stage_path(canonical: Mapping[str, Any] | None) -> str | None:
    if not canonical:
        return None
    parts = []
    for s in canonical["stages"]:
        detail = s.get("source") or s.get("method") or s.get("mode") or ""
        parts.append(f"{s['kind']}({detail})" if detail else s["kind"])
    return " > ".join(parts)


def task_row(base: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One per-task record from the selected genome's runs on that task."""
    kinds = Counter(r["failure_kind"] for r in runs if r["failure_kind"])
    passes = sum(r["passed"] for r in runs)
    row = {
        **base,
        "trials": len(runs),
        "passes": passes,
        "pass_rate": passes / len(runs) if runs else None,
        "passed": (passes / len(runs) >= 0.5) if runs else None,
        "fitness_mean": _mean(r["fitness"] for r in runs),
        "score_mean": _mean(r["score"] for r in runs),
        "tokens_mean": _mean(r["tokens"] for r in runs),
        "model_calls_mean": _mean(r["model_calls"] for r in runs),
        "tool_calls_mean": _mean(r["tool_calls"] for r in runs),
        "wall_time_s_mean": _mean(r["wall_time_s"] for r in runs),
        "evidence_source": ",".join(sorted({r["gather_source"] for r in runs})) or None,
        "evidence_provider": ",".join(sorted({r["evidence_provider"] or "-" for r in runs}))
        or None,
        "failure_kinds": dict(sorted(kinds.items())),
        "runs": [
            {k: r[k] for k in ("trial", "run_id", "verdict", "fitness", "score", "failure_kind")}
            for r in sorted(runs, key=lambda r: r["trial"])
        ],
    }
    return row


def split_ids(suite_names: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    out = {}
    for name in suite_names:
        s = legacy_suite(name)
        out[name] = {
            "train": [
                t.id for t in s.tasks_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK)
            ],
            "validation": [t.id for t in s.tasks_for(SplitRole.VALIDATION, SplitUse.SELECTION)],
            "test": [t.id for t in s.tasks_for(SplitRole.TEST, SplitUse.REPORTING)],
        }
    return out


def per_task_rows(
    searches: Sequence[Mapping[str, Any]],
    finals: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Mapping[str, Sequence[str]]],
    meta: Mapping[str, Mapping[str, str]],
    classes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Every suite task x optimizer x seed - completed, skipped or failed alike."""
    final_by: dict[tuple, list[Mapping[str, Any]]] = {}
    for r in finals:
        final_by.setdefault((r["suite"], r["optimizer"], r["seed"], r["task_id"]), []).append(r)
    rows = []
    for rec in searches:
        result = rec.get("result") or {}
        champ = result.get("champion") or {}
        champ_hash = champ.get("genome_hash")
        task_champs = {c["task_id"]: c["genome_hash"] for c in result.get("task_champions", [])}
        genome = rec["genomes"].get(champ_hash) if champ_hash else None
        trials = rec["config"]["trials"]
        for split in SPLITS:
            for tid in splits[rec["suite"]][split]:
                base = {
                    "suite": rec["suite"],
                    "task_id": tid,
                    "task_class": classes[tid],
                    "shape": meta[tid]["shape"],
                    "split": split,
                    "optimizer": rec["optimizer"],
                    "seed": rec["seed"],
                    "status": rec["status"],
                    "reason": rec["reason"],
                    "search_trials": trials,
                    "selected_genome_hash": champ_hash,
                    "selected_genome_path": stage_path(genome),
                    "champion_feasible": champ.get("feasible"),
                    "task_champion_genome_hash": task_champs.get(tid),
                }
                if split == "test":
                    got = final_by.get((rec["suite"], rec["optimizer"], rec["seed"], tid), [])
                    runs = [r for r in got if "error" not in r]
                    base["unscored_runs"] = len(got) - len(runs)  # backend outages, not FAILs
                else:
                    runs = [
                        r
                        for r in rec["runs"]
                        if r["genome_hash"] == champ_hash and r["task_id"] == tid
                    ]
                rows.append(task_row(base, runs))
    return rows


# -- aggregates -----------------------------------------------------------------------------------
def _group_keys(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("overall", "all"),
        ("class", row["task_class"]),
        ("suite", row["suite"]),
        ("shape", row["shape"]),
    ]


def seed_means(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, dict[int, dict[str, float]]]:
    """(optimizer, split, group kind, group) -> seed -> metric -> mean over that group's tasks."""
    acc: dict[tuple, dict[int, dict[str, list[float]]]] = {}
    for r in rows:
        if r["status"] != "completed" or not r["trials"]:
            continue
        for kind, group in _group_keys(r):
            per_seed = acc.setdefault((r["optimizer"], r["split"], kind, group), {})
            cell = per_seed.setdefault(r["seed"], {m: [] for m in METRICS})
            for m in METRICS:
                if r[m] is not None:
                    cell[m].append(r[m])
    return {
        key: {
            seed: {m: statistics.fmean(v) for m, v in cell.items() if v}
            for seed, cell in sorted(per_seed.items())
        }
        for key, per_seed in acc.items()
    }


def aggregates(means: Mapping[tuple, Mapping[int, Mapping[str, float]]]) -> list[dict[str, Any]]:
    out = []
    for (opt, split, kind, group), per_seed in sorted(means.items()):
        out.append(
            {
                "optimizer": opt,
                "split": split,
                "group_kind": kind,
                "group": group,
                "seeds": sorted(per_seed),
                **{
                    m: describe([per_seed[s][m] for s in sorted(per_seed) if m in per_seed[s]])
                    for m in METRICS
                },
            }
        )
    return out


def paired(
    means: Mapping[tuple, Mapping[int, Mapping[str, float]]], a: str, b: str
) -> list[dict[str, Any]]:
    """``a - b`` per seed (seeds both completed), for every split x group x metric."""
    out = []
    keys = sorted({k[1:] for k in means if k[0] in (a, b)})
    for split, kind, group in keys:
        ma, mb = means.get((a, split, kind, group), {}), means.get((b, split, kind, group), {})
        seeds = sorted(set(ma) & set(mb))
        if not seeds:
            continue
        rec: dict[str, Any] = {"split": split, "group_kind": kind, "group": group, "seeds": seeds}
        for m in METRICS:
            diffs = [ma[s][m] - mb[s][m] for s in seeds if m in ma[s] and m in mb[s]]
            p = sign_flip_p(diffs)
            rec[m] = {
                **describe(diffs),
                "p_sign_flip": p,
                "min_attainable_p": min_attainable_p(len(diffs)),
                "significant": p is not None and p < ALPHA,
            }
        out.append(rec)
    return out


def per_task_comparison(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Per split, per task: mean pass rate over seeds (then mean score) of ``a`` vs ``b``."""
    cells: dict[tuple, dict[str, list[Mapping[str, Any]]]] = {}
    for r in rows:
        if r["status"] == "completed" and r["trials"] and r["optimizer"] in (a, b):
            cells.setdefault((r["split"], r["task_id"]), {}).setdefault(r["optimizer"], []).append(
                r
            )
    out: dict[str, Any] = {}
    for split in SPLITS:
        tasks = []
        for (sp, tid), by in sorted(cells.items()):
            if sp != split or a not in by or b not in by:
                continue
            ra, rb = by[a], by[b]
            pa, pb = _mean(r["pass_rate"] for r in ra), _mean(r["pass_rate"] for r in rb)
            sa, sb = _mean(r["score_mean"] for r in ra), _mean(r["score_mean"] for r in rb)
            if abs(pa - pb) > TIE_TOL:
                outcome = "win" if pa > pb else "loss"
            elif sa is not None and sb is not None and abs(sa - sb) > TIE_TOL:
                outcome = "win" if sa > sb else "loss"
            else:
                outcome = "tie"
            tasks.append(
                {
                    "task_id": tid,
                    "suite": ra[0]["suite"],
                    "task_class": ra[0]["task_class"],
                    "shape": ra[0]["shape"],
                    "outcome": outcome,
                    f"{a}_pass_rate": pa,
                    f"{b}_pass_rate": pb,
                    f"{a}_score": sa,
                    f"{b}_score": sb,
                    "pass_rate_delta": pa - pb,
                    "seeds": len(ra),
                    "rescued": pa >= 0.5 > pb,
                    "broken": pb >= 0.5 > pa,
                }
            )
        c = Counter(t["outcome"] for t in tasks)
        out[split] = {
            "wins": c["win"],
            "losses": c["loss"],
            "ties": c["tie"],
            "p_sign_test": sign_test_p(c["win"], c["loss"]),
            "rescued": [t["task_id"] for t in tasks if t["rescued"]],
            "broken": [t["task_id"] for t in tasks if t["broken"]],
            "regressions": [t for t in tasks if t["outcome"] == "loss"],
            "tasks": tasks,
        }
    return out


def champions(searches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in searches:
        result = rec.get("result") or {}
        champ = result.get("champion") or {}
        h = champ.get("genome_hash")
        out.append(
            {
                "suite": rec["suite"],
                "optimizer": rec["optimizer"],
                "seed": rec["seed"],
                "status": rec["status"],
                "reason": rec["reason"],
                "budget": rec["config"]["budget"],
                "workflow_evaluations": result.get("workflow_evaluations"),
                "distinct_genomes": len(rec["genome_order"]),
                "suite_champion": h,
                "suite_champion_path": stage_path(rec["genomes"].get(h)) if h else None,
                "champion_feasible": champ.get("feasible"),
                "champion_validation_fitness": champ.get("validation_fitness"),
                "champion_validation_pass_rate": champ.get("validation_pass_rate"),
                "task_champions": {
                    c["task_id"]: c["genome_hash"] for c in result.get("task_champions", [])
                },
            }
        )
    return out


def summarize(
    protocol: Mapping[str, Any],
    searches: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    final_ran: bool,
) -> dict[str, Any]:
    a, b = protocol["protocol"]["comparison"]
    means = seed_means(rows)
    status = Counter((s["suite"], s["status"]) for s in searches)
    return {
        "schema": protocol["schema"],
        "synthetic": protocol["synthetic"],
        "comparison": {"a": a, "b": b, "difference": f"{a} - {b}"},
        "final_ran": final_ran,
        "searches": {f"{k[0]}:{k[1]}": v for k, v in sorted(status.items())},
        "skipped_suites": sorted(
            {(s["suite"], s["reason"]) for s in searches if s["status"] == "skipped"}
        ),
        "failed_searches": [
            {k: s[k] for k in ("suite", "optimizer", "seed", "reason")}
            for s in searches
            if s["status"] == "failed"
        ],
        "aggregates": aggregates(means),
        "paired": paired(means, a, b),
        "per_task": per_task_comparison(rows, a, b),
        "champions": champions(searches),
        "notes": {
            "statistics": "seed-level means; 95% Student-t CI; exact sign-flip permutation p on "
            "per-seed paired differences; significant only if p < 0.05",
            "per_task_outcome": "mean pass rate over seeds, tie broken by mean evaluator score",
            "passed": "a per-task row passes when at least half of its trials pass",
            "rescued": f"{a} mean pass rate >= 0.5 while {b} < 0.5 (broken: the reverse)",
        },
    }


# -- rendering ------------------------------------------------------------------------------------
def _f(x: Any, nd: int = 3, pct: bool = False) -> str:
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(d: Mapping[str, Any], nd: int = 3, pct: bool = False) -> str:
    if d["mean"] is None:
        return "-"
    s = _f(d["mean"], nd, pct)
    if d["ci95"]:
        lo, hi = d["ci95"]
        s += f" [{_f(lo, nd, pct)}, {_f(hi, nd, pct)}]"
    return s


def _agg(summary: Mapping[str, Any], opt: str, split: str, kind: str, group: str):
    for r in summary["aggregates"]:
        if (r["optimizer"], r["split"], r["group_kind"], r["group"]) == (opt, split, kind, group):
            return r
    return None


def _pair(summary: Mapping[str, Any], split: str, kind: str, group: str):
    for r in summary["paired"]:
        if (r["split"], r["group_kind"], r["group"]) == (split, kind, group):
            return r
    return None


def markdown(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    a, b = summary["comparison"]["a"], summary["comparison"]["b"]
    p = protocol["protocol"]
    lines = ["# Full-suite benchmark: ACO vs random search", ""]
    if summary["synthetic"]:
        lines += [
            "> **SYNTHETIC RUN.** Every number below comes from the labelled fake objective "
            "(`experiments.synthetic`), not from a model. It checks the pipeline; it is not a "
            "result.",
            "",
        ]
    lines += [
        f"- Protocol `{p['name']}`: suites {', '.join(p['suites'])}; optimizers "
        f"{', '.join(p['optimizers'])}; seeds {p['seeds']}; genome budget {p['genome_budget']} x "
        f"optimization rows x {p['search_trials']} search trial(s); batch {p['batch_size']}; "
        f"final trials {p['final_trials']}.",
        f"- Search benchmark `{protocol['benchmark_hash'][:16]}`, held-out "
        f"`{protocol['heldout_benchmark_hash'][:16]}`, model `{protocol.get('model_hash', '-')}`.",
        f"- Evidence provider for JEV: `{protocol.get('jev_provider') or 'none configured'}`.",
        "",
        "## Searches",
        "",
        "| suite | budget (train evals) | "
        + " | ".join(f"{o} evals (per seed)" for o in p["optimizers"])
        + " | status |",
        "|---|---|" + "---|" * len(p["optimizers"]) + "---|",
    ]
    by_suite: dict[str, list[Mapping[str, Any]]] = {}
    for c in summary["champions"]:
        by_suite.setdefault(c["suite"], []).append(c)
    for suite in p["suites"]:
        cs = by_suite.get(suite, [])
        cells = []
        for o in p["optimizers"]:
            ev = [c["workflow_evaluations"] for c in cs if c["optimizer"] == o]
            cells.append(", ".join("-" if e is None else str(e) for e in ev))
        st = Counter(c["status"] for c in cs)
        lines.append(
            f"| {suite} | {cs[0]['budget'] if cs else '-'} | "
            + " | ".join(cells)
            + f" | {', '.join(f'{k} {v}' for k, v in sorted(st.items()))} |"
        )
    for suite, reason in summary["skipped_suites"]:
        lines += ["", f"**{suite} not run:** {reason}."]
    for f in summary["failed_searches"]:
        lines += ["", f"**Failed:** {f['suite']} {f['optimizer']} seed {f['seed']}: {f['reason']}"]

    if summary.get("final_unscored_runs"):
        lines += [
            "",
            f"**{summary['final_unscored_runs']} held-out run(s) unscored** (backend unavailable "
            "after retries); they are excluded from the held-out rows, not counted as failures.",
        ]

    split_title = {
        "train": "Optimization rows (optimizer feedback)",
        "validation": "Validation rows (selection)",
        "test": "Held-out test rows (reporting only)",
    }
    for split in ("validation", "test", "train"):
        if split == "test" and not summary["final_ran"]:
            lines += ["", f"## {split_title[split]}", "", "Final evaluation was not run."]
            continue
        lines += [
            "",
            f"## {split_title[split]}",
            "",
            "Seed-level mean [95% t-CI]. Delta = "
            f"{a} - {b}, paired by seed; p = exact sign-flip permutation test.",
            "",
            f"| group | {a} pass rate | {b} pass rate | delta pass rate | p | "
            f"{a} score | {b} score | {a} tokens | {b} tokens | {a} tool calls | {b} tool calls |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        groups = [("overall", "all")]
        for kind in ("class", "suite", "shape"):
            groups += sorted(
                {(r["group_kind"], r["group"]) for r in summary["aggregates"]
                 if r["split"] == split and r["group_kind"] == kind}
            )  # fmt: skip
        for kind, group in groups:
            ra, rb = _agg(summary, a, split, kind, group), _agg(summary, b, split, kind, group)
            if ra is None and rb is None:
                continue
            pr = _pair(summary, split, kind, group)
            empty = {"mean": None, "ci95": None}
            lines.append(
                f"| {kind}: {group} | {_ci(ra['pass_rate'] if ra else empty, pct=True)} | "
                f"{_ci(rb['pass_rate'] if rb else empty, pct=True)} | "
                f"{_ci(pr['pass_rate'], pct=True) if pr else '-'} | "
                f"{_f(pr['pass_rate']['p_sign_flip']) if pr else '-'} | "
                f"{_f(ra['score_mean']['mean'] if ra else None)} | "
                f"{_f(rb['score_mean']['mean'] if rb else None)} | "
                f"{_f(ra['tokens_mean']['mean'] if ra else None, 0)} | "
                f"{_f(rb['tokens_mean']['mean'] if rb else None, 0)} | "
                f"{_f(ra['tool_calls_mean']['mean'] if ra else None, 2)} | "
                f"{_f(rb['tool_calls_mean']['mean'] if rb else None, 2)} |"
            )
        pr = _pair(summary, split, "overall", "all")
        if pr:
            sig = [m for m in METRICS if pr[m]["significant"]]
            lines += [
                "",
                f"Seeds paired: {len(pr['seeds'])}; smallest attainable p: "
                f"{_f(pr['pass_rate']['min_attainable_p'])}. Significant at 0.05: "
                f"{', '.join(sig) if sig else 'none'}.",
            ]
        pt = summary["per_task"][split]
        lines += [
            "",
            f"Per task ({a} vs {b}): **{pt['wins']} wins / {pt['losses']} losses / {pt['ties']} "
            f"ties** (exact sign test p = {_f(pt['p_sign_test'])}). Rescued by {a}: "
            f"{', '.join(pt['rescued']) or 'none'}. Broken by {a}: "
            f"{', '.join(pt['broken']) or 'none'}.",
        ]
        if pt["regressions"]:
            lines += [
                "",
                f"Tasks where {a} is worse than {b}:",
                "",
                f"| task | suite | shape | {a} pass rate | {b} pass rate | {a} score | {b} score |",
                "|---|---|---|---|---|---|---|",
            ]
            for t in pt["regressions"]:
                lines.append(
                    f"| {t['task_id']} | {t['suite']} | {t['shape']} | "
                    f"{_f(t[f'{a}_pass_rate'], pct=True)} | {_f(t[f'{b}_pass_rate'], pct=True)} | "
                    f"{_f(t[f'{a}_score'])} | {_f(t[f'{b}_score'])} |"
                )

    lines += ["", "## Seed variance (overall pass rate per seed)", ""]
    lines += ["| optimizer | split | per-seed pass rate | SD |", "|---|---|---|---|"]
    for r in summary["aggregates"]:
        if r["group_kind"] == "overall":
            vals = ", ".join(f"{100 * v:.1f}%" for v in r["pass_rate"]["values"])
            lines.append(
                f"| {r['optimizer']} | {r['split']} | {vals} | {_f(r['pass_rate']['sd'])} |"
            )

    lines += [
        "",
        "## Suite champions",
        "",
        "| suite | optimizer | seed | champion | path | feasible | val pass rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary["champions"]:
        if c["status"] != "completed":
            continue
        lines.append(
            f"| {c['suite']} | {c['optimizer']} | {c['seed']} | "
            f"`{(c['suite_champion'] or '-')[:12]}` "
            f"| {c['suite_champion_path'] or '-'} | {c['champion_feasible']} | "
            f"{_f(c['champion_validation_pass_rate'], pct=True)} |"
        )
    lines += [
        "",
        "Per-task rows (with per-task champions) are in `per_task.json` / `per_task.csv`; every "
        "run is in `search.json` / `final.json`; learning curves are `learning_curve_<suite>.png`.",
        "",
    ]
    return "\n".join(lines)


CSV_FIELDS = (
    "suite", "task_id", "task_class", "shape", "split", "optimizer", "seed", "status",
    "search_trials", "selected_genome_hash", "selected_genome_path", "champion_feasible",
    "task_champion_genome_hash", "trials", "passes", "pass_rate", "passed", "fitness_mean",
    "score_mean", "tokens_mean", "model_calls_mean", "tool_calls_mean", "wall_time_s_mean",
    "evidence_source", "evidence_provider", "failure_kinds", "reason",
)  # fmt: skip


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "failure_kinds": json.dumps(r["failure_kinds"], sort_keys=True)})


def plot_curves(searches: Sequence[Mapping[str, Any]], out: Path, synthetic: bool) -> list[str]:
    try:
        from experiments.report import plot_learning_curves
    except ImportError:  # pragma: no cover - matplotlib is a declared dependency
        return []
    written = []
    for suite in sorted({s["suite"] for s in searches}):
        done = [s["result"] for s in searches if s["suite"] == suite and s["status"] == "completed"]
        if not done or not any(r["curve"] for r in done):
            continue
        path = out / f"learning_curve_{suite}.png"
        plot_learning_curves({"runs": done, "suite": suite, "synthetic": synthetic}, path)
        written.append(path.name)
    return written


def write_reports(out: Path, *, plots: bool = True) -> dict[str, Any]:
    """(Re)write ``per_task.json``/``.csv``, ``summary.json``, ``REPORT.md`` and the plots."""
    from experiments.full_suite import write

    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    searches = json.loads((out / "search.json").read_text(encoding="utf-8"))["searches"]
    final = json.loads((out / "final.json").read_text(encoding="utf-8"))
    suites = protocol["protocol"]["suites"]
    meta = {
        tid: {"suite": m.suite, "shape": m.shape}
        for tid, m in {**load_catalog(), **load_catalog(HELDOUT_DIR)}.items()
    }
    splits = split_ids(suites)
    classes = {
        tid: spec.runtime.task_class.value
        for d in (BENCH_DIR, HELDOUT_DIR)
        for tid, spec in load_task_specs(d).items()
    }
    rows = per_task_rows(searches, final["runs"], splits, meta, classes)
    write(out / "per_task.json", {"schema": protocol["schema"], "rows": rows})
    write_csv(rows, out / "per_task.csv")
    summary = summarize(protocol, searches, rows, final["ran"])
    summary["final_unscored_runs"] = sum("error" in r for r in final["runs"])
    summary["learning_curves"] = plot_curves(searches, out, protocol["synthetic"]) if plots else []
    write(out / "summary.json", summary)
    (out / "REPORT.md").write_text(markdown(summary, protocol), encoding="utf-8")
    return summary
