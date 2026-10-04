"""Regenerates the frozen MVP benchmark (snapshots, tasks.json, splits.json).

Run once: ``python -m benchmarks.build``. The OUTPUT is the frozen artifact (committed and pinned
by a golden ``benchmark_hash`` in tests); this script documents how it was derived and computes
exact ground truth from the same records the pages/API are rendered from. Fictional data only -
no network, no real-world facts that could drift.

Class A: extract facts from static fact-sheet pages (with distractor numbers/dates).
Class B: filter/aggregate over a JSON API endpoint (``mock_api``); truth computed from records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.mock_api import render_response
from benchmarks.snapshot_store import SNAPSHOT_ROOT
from core.stages import GatherSource
from core.task_spec import (
    AnswerField,
    AnswerSchema,
    Caps,
    FieldType,
    GroundTruth,
    MatcherConfig,
    MatcherKind,
    RuntimeTask,
    TaskClass,
    TaskSpec,
)

BENCH_DIR = Path(__file__).resolve().parent
SPLIT_METHOD = (
    "per class: order by sha256('wynk-split-v1:'+task_id); last N_VALIDATION -> validation"
)
N_VALIDATION = 3

# field name -> (answer type, matcher config, truth value)
Fields = dict[str, tuple[FieldType, MatcherConfig, Any]]

NORM = MatcherConfig(kind=MatcherKind.NORMALIZED_TEXT)
EXACT = MatcherConfig(kind=MatcherKind.EXACT)
SET = MatcherConfig(kind=MatcherKind.SET_EQUAL)
DATE = MatcherConfig(kind=MatcherKind.DATE)
INT0 = MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0)
CENTS = MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0.01)

# --- Class A: company fact sheets ----------------------------------------------------------------
COMPANIES: list[dict[str, Any]] = [
    {
        "name": "Zephyr Dynamics",
        "founded": 1987,
        "hq": "Lisbon",
        "ceo": "Marta Oliveira",
        "employees": 4210,
        "revenue": 812.4,
        "prev_revenue": 770.9,
        "fy_end": "2024-03-31",
        "products": ["turbines", "rotors", "sensors"],
        "office": "Porto",
        "acquired": 2011,
    },
    {
        "name": "Quillon Labs",
        "founded": 2003,
        "hq": "Tallinn",
        "ceo": "Jaan Kask",
        "employees": 380,
        "revenue": 45.2,
        "prev_revenue": 31.8,
        "fy_end": "2024-12-31",
        "products": ["compilers", "linters"],
        "office": "Riga",
        "acquired": 2016,
    },
    {
        "name": "Brasswick Foods",
        "founded": 1952,
        "hq": "Leeds",
        "ceo": "Harriet Moss",
        "employees": 12950,
        "revenue": 2310.0,
        "prev_revenue": 2288.5,
        "fy_end": "2024-06-30",
        "products": ["biscuits", "cereal", "sauces", "tea"],
        "office": "Cork",
        "acquired": 1999,
    },
    {
        "name": "Nordhavn Shipping",
        "founded": 1969,
        "hq": "Aarhus",
        "ceo": "Lars Vinter",
        "employees": 2760,
        "revenue": 1195.7,
        "prev_revenue": 1240.3,
        "fy_end": "2024-09-30",
        "products": ["freight", "port services"],
        "office": "Gdansk",
        "acquired": 2005,
    },
    {
        "name": "Ostrava Optics",
        "founded": 1994,
        "hq": "Brno",
        "ceo": "Petra Novak",
        "employees": 640,
        "revenue": 98.6,
        "prev_revenue": 91.1,
        "fy_end": "2024-01-31",
        "products": ["lenses", "microscopes", "cameras"],
        "office": "Vienna",
        "acquired": 2008,
    },
    {
        "name": "Calder & Finch",
        "founded": 1921,
        "hq": "Glasgow",
        "ceo": "Duncan Reid",
        "employees": 5330,
        "revenue": 640.3,
        "prev_revenue": 655.0,
        "fy_end": "2024-04-30",
        "products": ["textiles", "uniforms"],
        "office": "Belfast",
        "acquired": 1987,
    },
    {
        "name": "Helio Pay",
        "founded": 2015,
        "hq": "Utrecht",
        "ceo": "Sanne de Wit",
        "employees": 215,
        "revenue": 22.9,
        "prev_revenue": 11.4,
        "fy_end": "2024-12-31",
        "products": ["payments", "invoicing", "payroll"],
        "office": "Ghent",
        "acquired": 2021,
    },
    {
        "name": "Tamarind Robotics",
        "founded": 2009,
        "hq": "Bologna",
        "ceo": "Giulia Ferrante",
        "employees": 1480,
        "revenue": 301.8,
        "prev_revenue": 260.2,
        "fy_end": "2024-10-31",
        "products": ["arms", "grippers"],
        "office": "Turin",
        "acquired": 2018,
    },
]

A_FIELDS: dict[str, tuple[FieldType, MatcherConfig]] = {
    "hq": (FieldType.STRING, NORM),
    "ceo": (FieldType.STRING, NORM),
    "founded": (FieldType.INTEGER, EXACT),
    "employees": (FieldType.INTEGER, INT0),
    "revenue": (FieldType.NUMBER, MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0.05)),
    "fy_end": (FieldType.DATE, DATE),
    "products": (FieldType.STRING_LIST, SET),
}

# (company index, asked fields, question)
A_TASKS: list[tuple[int, list[str], str]] = [
    (
        0,
        ["hq", "founded"],
        "which city is the headquarters, and in what year was the company founded?",
    ),
    (1, ["ceo", "employees"], "who is the CEO and how many employees does the company have?"),
    (
        2,
        ["revenue", "fy_end"],
        "what is the latest annual revenue (USD millions) and the fiscal year end date?",
    ),
    (3, ["products", "hq"], "list the company's products/services and the headquarters city."),
    (4, ["founded", "ceo", "employees"], "founding year, CEO name and employee count?"),
    (5, ["fy_end", "revenue"], "fiscal year end date and latest annual revenue (USD millions)?"),
    (6, ["products", "ceo"], "which products does the company sell and who is the CEO?"),
    (
        7,
        ["hq", "revenue", "founded"],
        "headquarters city, latest annual revenue (USD millions) and founding year?",
    ),
]


def _a_pages(c: dict[str, Any]) -> dict[str, str]:
    return {
        "overview": (
            f"{c['name']} - Company Overview\n"
            f"{c['name']} was founded in {c['founded']} and is headquartered in {c['hq']}.\n"
            f"The company is led by chief executive officer {c['ceo']}.\n"
            f"Core products and services: {', '.join(c['products'])}.\n"
            f"A regional office is maintained in {c['office']}.\n"
        ),
        "financials": (
            f"{c['name']} - Financial Highlights\n"
            f"Latest annual revenue: USD {c['revenue']} million "
            f"(prior year: USD {c['prev_revenue']} million).\n"
            f"Total employees: {c['employees']}.\n"
            f"The fiscal year ends on {c['fy_end']}.\n"
        ),
        "news": (
            f"{c['name']} - News Archive\n"
            f"In {c['acquired']}, {c['name']} completed an acquisition that expanded its "
            f"{c['office']} site.\n"
            f"Industry analysts expect sector growth of {c['founded'] % 17 + 3} percent "
            "next year.\n"
            f"The {c['office']} office hosted {c['employees'] // 40} visitors at its open day.\n"
        ),
    }


# --- Class B: inventory API ------------------------------------------------------------------
_NAMES = [
    "Anvil",
    "Bolt",
    "Clamp",
    "Drill",
    "Easel",
    "Funnel",
    "Gasket",
    "Hinge",
    "Ingot",
    "Jig",
    "Kettle",
    "Lathe",
    "Mallet",
    "Nozzle",
    "Oilcan",
    "Pulley",
    "Quill",
    "Rivet",
    "Spanner",
    "Tongs",
]
_CATEGORIES = ["tools", "fasteners", "plumbing", "paint"]
_WAREHOUSES = ["north", "south", "east"]


class _Lcg:
    """Tiny deterministic generator (no dependence on ``random`` implementation details)."""

    def __init__(self, seed: int) -> None:
        self.x = (seed * 2654435761 + 12345) % 2**31

    def next(self, n: int) -> int:
        self.x = (1103515245 * self.x + 12345) % 2**31
        return (self.x >> 8) % n


def _items(seed: int, n: int = 14) -> list[dict[str, Any]]:
    rng = _Lcg(seed)
    names = list(_NAMES)
    items = []
    for i in range(n):
        name = names.pop(rng.next(len(names)))
        items.append(
            {
                "sku": f"SKU-{100 + seed * 17 + i * 7:04d}",
                "name": f"{name} {rng.next(90) + 10}",
                "category": _CATEGORIES[rng.next(len(_CATEGORIES))],
                "warehouse": _WAREHOUSES[rng.next(len(_WAREHOUSES))],
                "stock": rng.next(400),
                "unit_price": round(rng.next(9000) / 100 + 1.5, 2),
                "restock_date": f"2024-{rng.next(12) + 1:02d}-{rng.next(28) + 1:02d}",
            }
        )
    return items


def _top(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    best = max(r[key] for r in rows)
    winners = [r for r in rows if r[key] == best]
    assert len(winners) == 1, f"tie on {key}: ground truth would be ambiguous"
    return winners[0]


def _in(rows: list[dict[str, Any]], **eq: str) -> list[dict[str, Any]]:
    return [r for r in rows if all(r[k] == v for k, v in eq.items())]


def _b_total_stock(i):
    return {
        "total_stock": (FieldType.INTEGER, INT0, sum(r["stock"] for r in _in(i, category="tools")))
    }


def _b_low_stock(i):
    names = [r["name"] for r in _in(i, warehouse="north") if r["stock"] < 150]
    return {"item_names": (FieldType.STRING_LIST, SET, names)}


def _b_priciest(i):
    top = _top(i, "unit_price")
    return {
        "item": (FieldType.STRING, NORM, top["name"]),
        "price": (FieldType.NUMBER, CENTS, top["unit_price"]),
    }


def _b_earliest(i):
    date = min(r["restock_date"] for r in _in(i, category="plumbing"))
    return {"earliest_restock": (FieldType.DATE, DATE, date)}


def _b_avg_price(i):
    rows = _in(i, category="fasteners")
    return {
        "avg_price": (
            FieldType.NUMBER,
            CENTS,
            round(sum(r["unit_price"] for r in rows) / len(rows), 2),
        )
    }


def _b_count(i):
    n = sum(1 for r in _in(i, warehouse="south") if r["unit_price"] > 40)
    return {"count": (FieldType.INTEGER, INT0, n)}


def _b_largest_stock(i):
    top = _top(i, "stock")
    return {
        "sku": (FieldType.STRING, EXACT, top["sku"]),
        "warehouse": (FieldType.STRING, NORM, top["warehouse"]),
    }


def _b_value(i):
    value = round(sum(r["stock"] * r["unit_price"] for r in _in(i, category="paint")), 2)
    return {"inventory_value": (FieldType.NUMBER, CENTS, value)}


B_TASKS: list[tuple[int, str, Callable[[list[dict[str, Any]]], Fields]]] = [
    (1, "What is the total stock across all items in category 'tools'?", _b_total_stock),
    (2, "List the names of items in the 'north' warehouse with stock below 150.", _b_low_stock),
    (3, "Which item has the highest unit price, and what is that price?", _b_priciest),
    (4, "What is the earliest restock date among items in category 'plumbing'?", _b_earliest),
    (
        5,
        "What is the average unit price of items in category 'fasteners' (2 decimals)?",
        _b_avg_price,
    ),
    (6, "How many items in the 'south' warehouse have a unit price above 40?", _b_count),
    (7, "Give the SKU of the item with the largest stock and its warehouse.", _b_largest_stock),
    (
        8,
        "What is the total inventory value (stock x unit price) of category 'paint', "
        "rounded to 2 decimals?",
        _b_value,
    ),
]

CAPS_A = Caps(tokens=9000, wall_time_s=180.0, tool_calls=12, retries=2)
CAPS_B = Caps(tokens=7000, wall_time_s=180.0, tool_calls=10, retries=1)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def _spec(task_id, task_class, question, snapshot_id, sources, caps, fields: Fields) -> TaskSpec:
    return TaskSpec(
        runtime=RuntimeTask(
            id=task_id,
            task_class=task_class,
            question=question,
            answer_schema=AnswerSchema(
                fields=tuple(AnswerField(name=n, type=t) for n, (t, _m, _v) in fields.items())
            ),
            caps=caps,
            allowed_sources=sources,
            snapshot_id=snapshot_id,
        ),
        ground_truth=GroundTruth(values={n: v for n, (_t, _m, v) in fields.items()}),
        matchers={n: m for n, (_t, m, _v) in fields.items()},
    )


def split_order(task_id: str) -> str:
    return hashlib.sha256(f"wynk-split-v1:{task_id}".encode()).hexdigest()


def compute_splits(task_ids_by_class: dict[str, list[str]]) -> dict[str, list[str]]:
    train: list[str] = []
    validation: list[str] = []
    for _cls, ids in sorted(task_ids_by_class.items()):
        ordered = sorted(ids, key=split_order)
        train += ordered[:-N_VALIDATION]
        validation += ordered[-N_VALIDATION:]
    return {"train": sorted(train), "validation": sorted(validation)}


def build(root: Path = SNAPSHOT_ROOT, bench_dir: Path = BENCH_DIR) -> list[TaskSpec]:
    specs: list[TaskSpec] = []

    for n, (idx, asked, question) in enumerate(A_TASKS, start=1):
        c = COMPANIES[idx]
        snap = f"A-{n:03d}"
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

    for n, (seed, question, truth) in enumerate(B_TASKS, start=1):
        snap = f"B-{n:03d}"
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
    by_class: dict[str, list[str]] = {}
    for s in specs:
        by_class.setdefault(s.runtime.task_class.value, []).append(s.id)
    _write(
        bench_dir / "splits.json",
        _dump({"schema_version": 1, "method": SPLIT_METHOD, "splits": compute_splits(by_class)}),
    )
    return specs


if __name__ == "__main__":
    build()
