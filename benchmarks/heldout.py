"""Held-out TEST split for the external-baseline comparison (``experiments/oss_baselines``).

    python -m benchmarks.heldout        # regenerates benchmarks/heldout/ (committed, hash-pinned)

The frozen MVP benchmark only has ``train`` (optimizer search) and ``validation`` (incumbent
selection) splits, so neither may be used to report a final comparison. This module renders a
disjoint TEST set from the SAME task-class definitions as ``benchmarks.build`` (same page
templates, same 8 Class A question templates, same 8 Class B question/truth functions, same caps
and matchers) over NEW fictional entities: 8 new companies and 8 new inventory seeds.

No optimizer, pheromone store or incumbent selection ever reads this directory. It lives outside
``benchmarks/snapshots`` so the frozen MVP ``benchmark_hash`` is unchanged.

Layout mirrors the main benchmark (``tasks.json``, ``splits.json``, ``snapshots/<id>/...``), so
``SnapshotStore(HELDOUT_SNAPSHOTS)`` / ``load_task_specs(HELDOUT_DIR)`` work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.build import (
    A_FIELDS,
    A_TASKS,
    B_TASKS,
    CAPS_A,
    CAPS_B,
    _a_pages,
    _dump,
    _items,
    _spec,
    _write,
)
from benchmarks.mock_api import render_response
from core.stages import GatherSource
from core.task_spec import TaskClass, TaskSpec

HELDOUT_DIR = Path(__file__).resolve().parent / "heldout"
HELDOUT_SNAPSHOTS = HELDOUT_DIR / "snapshots"
SPLIT = "test"

# New fictional companies (disjoint from benchmarks.build.COMPANIES); same attribute shape.
TEST_COMPANIES: list[dict[str, Any]] = [
    {
        "name": "Veldkamp Instruments",
        "founded": 1978,
        "hq": "Eindhoven",
        "ceo": "Ruben Smits",
        "employees": 2890,
        "revenue": 433.7,
        "prev_revenue": 401.2,
        "fy_end": "2024-03-31",
        "products": ["oscilloscopes", "probes", "analyzers"],
        "office": "Leuven",
        "acquired": 2002,
    },
    {
        "name": "Kestrel Analytics",
        "founded": 2011,
        "hq": "Dublin",
        "ceo": "Aoife Brennan",
        "employees": 520,
        "revenue": 61.3,
        "prev_revenue": 48.9,
        "fy_end": "2024-12-31",
        "products": ["dashboards", "forecasting"],
        "office": "Galway",
        "acquired": 2019,
    },
    {
        "name": "Marisol Ceramics",
        "founded": 1946,
        "hq": "Valencia",
        "ceo": "Lucia Navarro",
        "employees": 7120,
        "revenue": 1548.2,
        "prev_revenue": 1502.6,
        "fy_end": "2024-06-30",
        "products": ["tiles", "sinks", "glazes", "bricks"],
        "office": "Zaragoza",
        "acquired": 1993,
    },
    {
        "name": "Fjordline Energy",
        "founded": 1972,
        "hq": "Bergen",
        "ceo": "Ingrid Solberg",
        "employees": 3340,
        "revenue": 2087.5,
        "prev_revenue": 2154.1,
        "fy_end": "2024-09-30",
        "products": ["hydropower", "grid services"],
        "office": "Stavanger",
        "acquired": 2007,
    },
    {
        "name": "Pellucid Glassworks",
        "founded": 1999,
        "hq": "Krakow",
        "ceo": "Tomasz Wrona",
        "employees": 910,
        "revenue": 132.4,
        "prev_revenue": 120.8,
        "fy_end": "2024-01-31",
        "products": ["panels", "mirrors", "lenses"],
        "office": "Wroclaw",
        "acquired": 2013,
    },
    {
        "name": "Ashgrove Mills",
        "founded": 1908,
        "hq": "Manchester",
        "ceo": "Eleanor Pryce",
        "employees": 6480,
        "revenue": 718.9,
        "prev_revenue": 731.5,
        "fy_end": "2024-04-30",
        "products": ["flour", "oats"],
        "office": "Hull",
        "acquired": 1984,
    },
    {
        "name": "Lumen Ledger",
        "founded": 2017,
        "hq": "Antwerp",
        "ceo": "Bram Peeters",
        "employees": 168,
        "revenue": 18.6,
        "prev_revenue": 9.7,
        "fy_end": "2024-12-31",
        "products": ["bookkeeping", "tax filing", "audits"],
        "office": "Bruges",
        "acquired": 2022,
    },
    {
        "name": "Corvina Marine",
        "founded": 2006,
        "hq": "Genoa",
        "ceo": "Matteo Ricci",
        "employees": 1730,
        "revenue": 356.1,
        "prev_revenue": 318.4,
        "fy_end": "2024-10-31",
        "products": ["hulls", "propellers"],
        "office": "Livorno",
        "acquired": 2016,
    },
]

# New inventory seeds (the MVP benchmark used 1..8); paired with B_TASKS question n.
TEST_B_SEEDS = (21, 22, 23, 24, 25, 26, 27, 28)


def build(bench_dir: Path = HELDOUT_DIR) -> list[TaskSpec]:
    root = bench_dir / "snapshots"
    specs: list[TaskSpec] = []
    a_pairs = zip(A_TASKS, TEST_COMPANIES, strict=True)
    for n, ((_idx, asked, question), c) in enumerate(a_pairs, start=1):
        snap = f"TA-{n:03d}"
        for page_id, text in _a_pages(c).items():
            _write(root / snap / "pages" / f"{page_id}.txt", text)
        _write(root / snap / "records.json", _dump({"endpoints": {"company": [c]}}))
        fields = {f: (*A_FIELDS[f], c[f]) for f in asked}
        specs.append(
            _spec(
                snap,
                TaskClass.A,
                f"Per the {c['name']} fact sheet: {question}",
                snap,
                (GatherSource.FETCH, GatherSource.JEV),
                CAPS_A,
                fields,
            )
        )
    b_pairs = zip(B_TASKS, TEST_B_SEEDS, strict=True)
    for n, ((_seed, question, truth), seed) in enumerate(b_pairs, start=1):
        snap = f"TB-{n:03d}"
        items = _items(seed)
        _write(root / snap / "pages" / "items.txt", render_response("items", items))
        _write(root / snap / "records.json", _dump({"endpoints": {"items": items}}))
        specs.append(
            _spec(
                snap,
                TaskClass.B,
                question,
                snap,
                (GatherSource.FETCH, GatherSource.API),
                CAPS_B,
                truth(items),
            )
        )
    _write(
        bench_dir / "tasks.json",
        _dump({"schema_version": 1, "tasks": [s.model_dump(mode="json") for s in specs]}),
    )
    _write(
        bench_dir / "splits.json",
        _dump(
            {
                "schema_version": 1,
                "method": "held-out TEST split for external comparison; never used for search",
                "splits": {SPLIT: sorted(s.id for s in specs)},
            }
        ),
    )
    return specs


if __name__ == "__main__":
    build()
