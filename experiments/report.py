"""Experiment reporting: aggregate learning curves over seeds and plot ACO vs random."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Categorical slots 1 and 2 of the validated reference palette (light mode, surface #fcfcfb).
COLORS = {"aco_mmas": "#2a78d6", "random_search": "#eb6834"}
LABELS = {"aco_mmas": "ACO (MMAS)", "random_search": "Random search"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def aggregate_curves(
    results: Mapping[str, Any], y: str = "validation_fitness"
) -> dict[str, dict[str, list[float]]]:
    """Per optimizer: ``{"x", "mean", "se", "n"}`` over seeds, as a step function of evaluations.

    A seed's value at ``x`` is its last recorded point with ``evaluations <= x``; seeds that have
    not yet recorded a point at ``x`` are skipped for that ``x``.
    """
    by_opt: dict[str, list[list[dict[str, Any]]]] = {}
    for run in results["runs"]:
        by_opt.setdefault(run["optimizer"], []).append(run["curve"])
    out: dict[str, dict[str, list[float]]] = {}
    for name, curves in sorted(by_opt.items()):
        xs = sorted({p["evaluations"] for c in curves for p in c})
        mean, se, n = [], [], []
        for x in xs:
            vals = [
                [p[y] for p in c if p["evaluations"] <= x][-1]
                for c in curves
                if any(p["evaluations"] <= x for p in c)
            ]
            mean.append(statistics.fmean(vals))
            se.append(statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            n.append(len(vals))
        out[name] = {"x": [float(x) for x in xs], "mean": mean, "se": se, "n": n}
    return out


def plot_learning_curves(results: Mapping[str, Any], path: Path) -> None:
    """x = workflow evaluations, y = validation fitness of the best-so-far (train-selected)
    workflow; line = mean over seeds, band = +/- 1 standard error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate_curves(results)
    n_seeds = len({r["seed"] for r in results["runs"]})
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for name, c in agg.items():
        color = COLORS.get(name, INK_2)
        lo = [m - s for m, s in zip(c["mean"], c["se"], strict=True)]
        hi = [m + s for m, s in zip(c["mean"], c["se"], strict=True)]
        ax.fill_between(c["x"], lo, hi, step="post", color=color, alpha=0.15, linewidth=0)
        ax.step(
            c["x"], c["mean"], where="post", color=color, linewidth=2, label=LABELS.get(name, name)
        )
    ax.set_xlabel("Training workflow evaluations", color=INK_2)
    ax.set_ylabel("validation fitness of best-so-far workflow", color=INK_2)
    tag = "SYNTHETIC objective - not a benchmark result" if results["synthetic"] else "real runtime"
    ax.set_title(
        f"ACO vs random search, suite {results['suite']}  [{tag}]",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.text(0.01, 0.01, f"mean +/- 1 SE over {n_seeds} seeds", fontsize=8, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.margins(x=0.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
