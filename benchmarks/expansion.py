"""The expanded benchmark (#13): new task shapes over new fictional entities.

``benchmarks.build`` (search: optimization + validation) and ``benchmarks.heldout`` (test) call
``search_tasks`` / ``heldout_tasks`` here after rendering the original MVP tasks, which stay
byte-identical. Everything is derived offline from deterministic generators: no network, no live
JEV response, no real-world facts that could drift. Ground truth is computed from the same records
the pages / API responses are rendered from.

Suites (one shared search policy each; ``ContractSuite`` admits one genome for all its rows):

    A      fact sheets, 3-4 pages            fetch|jev   9k tokens, 12 tools   (original caps)
    Along  dossiers, 12 pages, 3-5 fields    fetch|jev   16k tokens, 20 tools  (long chains)
    Ajev   investor portals, 18 pages        jev only    12k tokens, 10 tools  interaction_required
    B      one inventory endpoint            fetch|api   7k tokens, 10 tools   (original caps)
    Bjoin  four related endpoints            fetch|api   14k tokens, 12 tools  (joins, chains)

Why ``Ajev`` genuinely requires interaction: a portal has 18 pages and a run may spend at most 10
tool calls, so reading the whole surface (fetch: one call per page) can never fit the contract.
The task is "navigate the portal": only an agent that chooses which pages to open can answer it.
The contract says so (``interaction_required``), and without a configured JEV provider these
tasks have no admissible workflow - they are reported as not run, never answered another way.

Search and held-out use disjoint entities AND disjoint question templates of the same shapes,
so held-out measures generalization to new entities and new phrasings, not memorized prompts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from benchmarks.mock_api import render_response
from core.stages import GatherSource
from core.task_spec import Caps, FieldType, MatcherConfig, MatcherKind, TaskClass

# field name -> (answer type, matcher config, truth value); same shape as benchmarks.build.Fields
Fields = dict[str, tuple[FieldType, MatcherConfig, Any]]

NORM = MatcherConfig(kind=MatcherKind.NORMALIZED_TEXT)
SET = MatcherConfig(kind=MatcherKind.SET_EQUAL)
DATE = MatcherConfig(kind=MatcherKind.DATE)
INT0 = MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0)
CENTS = MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0.01)
TENTH = MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, abs_tol=0.05)

FETCH_JEV = (GatherSource.FETCH, GatherSource.JEV)
FETCH_API = (GatherSource.FETCH, GatherSource.API)
JEV_ONLY = (GatherSource.JEV,)

CAPS_A_LONG = Caps(tokens=16000, wall_time_s=240.0, tool_calls=20, retries=2)
CAPS_A_JEV = Caps(tokens=12000, wall_time_s=240.0, tool_calls=10, retries=1)
CAPS_B_JOIN = Caps(tokens=14000, wall_time_s=240.0, tool_calls=12, retries=1)


@dataclass(frozen=True)
class Suite:
    name: str  # alphanumeric: also a workflow-memory file name (memory.store)
    task_class: TaskClass
    description: str


SUITES: dict[str, Suite] = {
    s.name: s
    for s in (
        Suite("A", TaskClass.A, "fact-sheet lookups over 3-4 pages (original MVP policy)"),
        Suite("Along", TaskClass.A, "multi-document dossiers: 3-5 fields, cross-document chains"),
        Suite("Ajev", TaskClass.A, "investor portals larger than the tool budget: JEV required"),
        Suite("B", TaskClass.B, "filter/aggregate/sort over one inventory endpoint"),
        Suite("Bjoin", TaskClass.B, "joins and dependency chains across four endpoints"),
    )
}


@dataclass(frozen=True)
class NewTask:
    """One generated task: everything ``benchmarks.build._spec`` needs plus catalog metadata."""

    task_id: str
    suite: str
    shape: str
    question: str
    sources: tuple[GatherSource, ...]
    caps: Caps
    fields: Fields
    pages: dict[str, str]
    records: dict[str, list[dict[str, Any]]]
    interaction_required: bool = False


class Ambiguous(ValueError):
    """The generated data would make the ground truth ambiguous (a tie, an empty answer)."""


class _Rng:
    """Deterministic LCG (same recurrence as ``benchmarks.build._Lcg``; no ``random`` module)."""

    def __init__(self, seed: int) -> None:
        self.x = (seed * 2654435761 + 12345) % 2**31

    def below(self, n: int) -> int:
        self.x = (1103515245 * self.x + 12345) % 2**31
        return (self.x >> 8) % n

    def between(self, lo: int, hi: int) -> int:  # inclusive
        return lo + self.below(hi - lo + 1)

    def pick(self, pool: list[Any]) -> Any:
        return pool[self.below(len(pool))]

    def sample(self, pool: list[Any], k: int) -> list[Any]:
        left = list(pool)
        return [left.pop(self.below(len(left))) for _ in range(k)]


# -- fictional name banks ---------------------------------------------------------------------
_PREFIX = [
    "Arden", "Belmar", "Cobalt", "Dunmore", "Elstree", "Farrow", "Glenmark", "Harlow", "Ivel",
    "Juniper", "Kelvar", "Larkspur", "Merrow", "Northgate", "Orvik", "Pemberly", "Quarrel",
    "Rosmer", "Sablewood", "Tidewell", "Umberly", "Vantor", "Wexley", "Yarrow", "Zelmar",
    "Ashby", "Brisco", "Corran", "Delvin", "Emberly", "Fennick", "Galloway",
]  # fmt: skip
_SUFFIX = [
    "Systems", "Industries", "Holdings", "Logistics", "Foods", "Materials", "Analytics",
    "Engineering", "Textiles", "Energy", "Pharma", "Instruments", "Marine", "Software",
    "Chemicals", "Optics", "Robotics", "Packaging", "Ceramics", "Mobility",
]  # fmt: skip
_FIRST = [
    "Amara", "Bastian", "Celine", "Dmitri", "Elif", "Fintan", "Greta", "Hugo", "Ilse", "Jonas",
    "Katya", "Lorenzo", "Maren", "Nils", "Odile", "Pavel", "Quinn", "Rosa", "Stellan", "Tilde",
    "Ulrich", "Vera", "Wim", "Xenia", "Yusuf", "Zora", "Anders", "Brigid", "Cosimo", "Dagny",
]  # fmt: skip
_LAST = [
    "Albrecht", "Bergstrom", "Castell", "Duarte", "Eklund", "Fontaine", "Gallo", "Haugen",
    "Iversen", "Jansen", "Kowalczyk", "Lindqvist", "Moreau", "Nygaard", "Orsini", "Petrov",
    "Quaranta", "Rasmussen", "Saarinen", "Tamm", "Urbano", "Vasquez", "Wojcik", "Ystad",
    "Zeller", "Aalto", "Brandt", "Cerny", "Delacroix", "Esposito",
]  # fmt: skip
_CITIES = [
    "Lisbon", "Porto", "Tallinn", "Riga", "Leeds", "Cork", "Aarhus", "Gdansk", "Brno", "Vienna",
    "Glasgow", "Belfast", "Utrecht", "Ghent", "Bologna", "Turin", "Lyon", "Basel", "Graz",
    "Malmo", "Tampere", "Bilbao", "Seville", "Poznan", "Cluj", "Ljubljana", "Zagreb", "Kaunas",
    "Bremen", "Leipzig", "Nantes", "Bordeaux", "Aberdeen", "Galway", "Rotterdam", "Liege",
]  # fmt: skip
_EXCHANGES = [
    "Euronext Lisbon", "Nasdaq Tallinn", "London Stock Exchange", "Xetra", "SIX Swiss Exchange",
    "Nasdaq Copenhagen", "Borsa Italiana", "Warsaw Stock Exchange", "Vienna Stock Exchange",
]  # fmt: skip
_AUDITORS = [
    "Halden & Rowe", "Mirant Partners", "Calloway Audit", "Brennick LLP", "Osterfeld Assurance",
    "Tarrant Kemp", "Vossberg & Lyle", "Quayle Assurance",
]  # fmt: skip
_PRODUCTS = [
    "turbines", "sensors", "valves", "pumps", "adhesives", "coatings", "cables", "drones",
    "scanners", "filters", "bearings", "batteries", "kiosks", "routers", "lenses", "resins",
]  # fmt: skip
_SEGMENTS = [
    "Industrial", "Consumer", "Healthcare", "Aerospace", "Energy Services", "Digital",
    "Logistics", "Agriculture", "Infrastructure", "Retail Solutions", "Defense", "Water",
]  # fmt: skip
_FY_ENDS = ["2024-01-31", "2024-03-31", "2024-04-30", "2024-06-30", "2024-09-30", "2024-12-31"]


def _bijective(index: int, n: int, step: int) -> int:
    """index -> a distinct slot of ``range(n)`` (``step`` is coprime with ``n``)."""
    if index >= n:
        raise ValueError(f"name bank exhausted ({index} >= {n})")
    return (index * step + 11) % n


def _company_name(index: int) -> str:
    k = _bijective(index, len(_PREFIX) * len(_SUFFIX), 389)
    return f"{_PREFIX[k // len(_SUFFIX)]} {_SUFFIX[k % len(_SUFFIX)]}"


def _person(index: int) -> str:
    n = len(
        _FIRST
    )  # == len(_LAST): (first, last) = (k mod n, (k div n + 7 k) mod n) is a bijection
    k = _bijective(index, n * n, 433)
    first = k % n
    return f"{_FIRST[first]} {_LAST[(k // n + 7 * first) % n]}"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Company/person bank ranges: search entities draw from [0, 300) / [0, 450), held-out from
# [300, 640) / [450, 900): disjoint by construction.
COMPANY_SLOTS, PERSON_SLOTS = 6, 10
HELDOUT_COMPANY_BASE, HELDOUT_PERSON_BASE = 300, 450


def dossier(
    entity: int,
    *,
    heldout: bool,
    acquisition_years: tuple[int, int] = (1995, 2023),
    variant: int = 0,
):
    """One fictional company with everything any Class A page/template may need. ``variant``
    re-draws the numbers (never the names) when a variant's truth would be ambiguous."""
    cbase = (HELDOUT_COMPANY_BASE if heldout else 0) + entity * COMPANY_SLOTS
    pbase = (HELDOUT_PERSON_BASE if heldout else 0) + entity * PERSON_SLOTS
    r = _Rng(7001 + entity + (5000 if heldout else 0) + 100_000 * variant)
    founded = r.between(1905, 2008)
    ceo_since = r.between(max(founded + 6, 2012), 2023)
    revenue = round(r.between(400, 300000) / 10, 1)
    shares = r.sample([22, 27, 31, 36, 41, 47, 52, 58], 3)
    shares[2] = 100 - shares[0] - shares[1]
    if shares[2] <= 0 or len(set(shares)) < 3:
        shares = [50, 30, 20]
    seg_names = r.sample(_SEGMENTS, 3)
    seg_rev = [round(revenue * s / 100, 1) for s in shares[:2]]
    seg_rev.append(round(revenue - sum(seg_rev), 1))
    cities = r.sample(_CITIES, 9)
    hq, planned_hq, target_hq, prev_hq, peer_hq = cities[:5]
    sub_cities = cities[5:8]
    sub_years = sorted(r.sample(list(range(founded + 1, 2023)), 3))
    order = r.sample([0, 1, 2], 3)  # which subsidiary gets which founding year
    subs = [
        {
            "name": _company_name(cbase + 1 + i),
            "city": sub_cities[i],
            "founded": sub_years[order[i]],
            "md": _person(pbase + 7 + i),
            "staff": r.between(40, 900),
        }
        for i in range(3)
    ]
    office_sub = r.below(3)
    employees = r.between(150, 16000)
    div_month = r.between(3, 11)
    return {
        "name": _company_name(cbase),
        "founded": founded,
        "hq": hq,
        "planned_hq": planned_hq,
        "office": subs[office_sub]["city"],
        "office_subsidiary": office_sub,
        "exchange": r.pick(_EXCHANGES),
        "ticker": "".join(w[0] for w in _company_name(cbase).split()) + "XYZQ"[r.below(4)],
        "products": r.sample(_PRODUCTS, r.between(2, 4)),
        "ceo": _person(pbase),
        "ceo_since": ceo_since,
        "former_ceo": _person(pbase + 1),
        "former_ceo_from": ceo_since - r.between(4, 12),
        "cfo": _person(pbase + 2),
        "chair": _person(pbase + 3),
        "segment_heads": [_person(pbase + 4 + i) for i in range(3)],
        "employees": employees,
        "employee_target": employees + r.between(200, 2500),
        "revenue": revenue,
        "prev_revenue": round(revenue * r.between(78, 99) / 100, 1),
        "forecast_revenue": round(revenue * r.between(104, 125) / 100, 1),
        "fy_end": r.pick(_FY_ENDS),
        "auditor": r.pick(_AUDITORS),
        "segments": [{"name": n, "revenue": v} for n, v in zip(seg_names, seg_rev, strict=True)],
        "subsidiaries": subs,
        "acquired_company": _company_name(cbase + 4),
        "acquisition_year": r.between(max(founded + 1, acquisition_years[0]), acquisition_years[1]),
        "acquisition_price": round(r.between(50, 90000) / 10, 1),
        "target_founded": r.between(1950, 2010),
        "target_hq": target_hq,
        "prev_employer": _company_name(cbase + 5),
        "prev_employer_hq": prev_hq,
        "peer": _person(pbase + 6).split()[1] + " Group",
        "peer_hq": peer_hq,
        "dividend_amount": round(r.between(5, 250) / 100, 2),
        "dividend_date": f"2024-{div_month:02d}-{r.between(1, 28):02d}",
        "prev_dividend_amount": round(r.between(5, 250) / 100, 2),
        "prev_dividend_date": f"2023-{r.between(3, 11):02d}-{r.between(1, 28):02d}",
        "growth": r.between(2, 14),
    }


# -- Class A page renderers ---------------------------------------------------------------------
def _p_overview(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Company Overview\n"
        f"{c['name']} was founded in {c['founded']} and is headquartered in {c['hq']}.\n"
        f"Shares trade on the {c['exchange']} under the ticker {c['ticker']}.\n"
        f"Core products and services: {', '.join(c['products'])}.\n"
        f"A regional office is maintained in {c['office']}.\n"
    )


def _p_leadership(c: dict[str, Any], *, heads: bool) -> str:
    text = (
        f"{c['name']} - Leadership\n"
        f"Chief executive officer: {c['ceo']} (since {c['ceo_since']}). Before joining, "
        f"{c['ceo']} was chief executive of {c['prev_employer']}.\n"
        f"{c['former_ceo']} served as chief executive officer from {c['former_ceo_from']} "
        f"until {c['ceo_since']}.\n"
        f"Chief financial officer: {c['cfo']}.\n"
        f"Chair of the board: {c['chair']}.\n"
    )
    if heads:
        pairs = zip(c["segments"], c["segment_heads"], strict=True)
        text += "Segment heads: " + "; ".join(f"{s['name']} - {h}" for s, h in pairs) + ".\n"
    return text


def _p_financials(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Financial Highlights\n"
        f"Latest annual revenue: USD {c['revenue']:.1f} million "
        f"(prior year: USD {c['prev_revenue']:.1f} million).\n"
        f"Management forecasts revenue of USD {c['forecast_revenue']:.1f} million for the "
        "coming year.\n"
        f"Total employees: {c['employees']}.\n"
        f"The fiscal year ends on {c['fy_end']}.\n"
        f"The financial statements are audited by {c['auditor']}.\n"
    )


def _p_press(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Press Releases\n"
        f"{c['name']} announced plans to relocate its headquarters to {c['planned_hq']} by 2027.\n"
        f"The company is targeting a workforce of {c['employee_target']} employees within three "
        "years.\n"
        f"Former chief executive {c['former_ceo']} joined the advisory council of "
        f"{c['peer']}.\n"
        f"Analysts expect sector growth of {c['growth']} percent next year.\n"
    )


def _p_segments(c: dict[str, Any]) -> str:
    rows = "".join(f"- {s['name']}: {s['revenue']:.1f}\n" for s in c["segments"])
    return (
        f"{c['name']} - Business Segments\n"
        "Revenue by segment for the latest fiscal year (USD million):\n" + rows
    )


def _p_governance(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Governance\n"
        f"The board is chaired by {c['chair']}.\n"
        f"External auditor: {c['auditor']}.\n"
        f"The audit committee met {c['growth'] + 3} times during the fiscal year ending "
        f"{c['fy_end']}.\n"
    )


def _p_subsidiary(c: dict[str, Any], s: dict[str, Any]) -> str:
    return (
        f"{s['name']} - Subsidiary Profile\n"
        f"{s['name']} is a wholly owned subsidiary of {c['name']}.\n"
        f"It was founded in {s['founded']} and is based in {s['city']}.\n"
        f"Managing director: {s['md']}.\n"
        f"It employs {s['staff']} people.\n"
    )


def _p_acquisition(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Acquisition Announcement\n"
        f"In {c['acquisition_year']}, {c['name']} completed the acquisition of "
        f"{c['acquired_company']} for USD {c['acquisition_price']:.1f} million.\n"
        f"The business was integrated into the {c['segments'][0]['name']} segment.\n"
    )


def _p_target(c: dict[str, Any]) -> str:
    return (
        f"{c['acquired_company']} - Company Profile\n"
        f"{c['acquired_company']} was founded in {c['target_founded']} and was originally "
        f"headquartered in {c['target_hq']}.\n"
        f"Following its acquisition by {c['name']} in {c['acquisition_year']}, its operations "
        f"moved to {c['hq']}.\n"
    )


def _p_peers(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Industry Peers\n"
        f"{c['prev_employer']} is headquartered in {c['prev_employer_hq']} and competes with "
        f"{c['name']} in the {c['segments'][1]['name']} segment.\n"
        f"{c['peer']} is headquartered in {c['peer_hq']}.\n"
    )


def _p_dividends(c: dict[str, Any]) -> str:
    return (
        f"{c['name']} - Dividends\n"
        f"Most recent dividend: USD {c['dividend_amount']:.2f} per share, paid on "
        f"{c['dividend_date']}.\n"
        f"Previous dividend: USD {c['prev_dividend_amount']:.2f} per share, paid on "
        f"{c['prev_dividend_date']}.\n"
    )


def std_pages(c: dict[str, Any]) -> dict[str, str]:
    return {
        "overview": _p_overview(c),
        "leadership": _p_leadership(c, heads=False),
        "financials": _p_financials(c),
        "press": _p_press(c),
    }


def long_pages(c: dict[str, Any]) -> dict[str, str]:
    pages = {
        "overview": _p_overview(c),
        "leadership": _p_leadership(c, heads=True),
        "financials": _p_financials(c),
        "segments": _p_segments(c),
        "governance": _p_governance(c),
        "acquisition": _p_acquisition(c),
        f"profile-{_slug(c['acquired_company'])}": _p_target(c),
        "peers": _p_peers(c),
        "press": _p_press(c),
    }
    for s in c["subsidiaries"]:
        pages[f"subsidiary-{_slug(s['name'])}"] = _p_subsidiary(c, s)
    return pages


PORTAL_YEARS = (2021, 2022, 2023, 2024)


def portal_pages(c: dict[str, Any]) -> dict[str, str]:
    """An 18-page investor portal (larger than ``CAPS_A_JEV.tool_calls``)."""
    name = c["name"]
    pages = {
        "about": _p_overview(c).replace("Company Overview", "About", 1),
        "leadership": _p_leadership(c, heads=True),
        "financials": _p_financials(c),
        "segments": _p_segments(c),
        "governance": _p_governance(c),
        "dividends": _p_dividends(c),
        "careers": (
            f"{name} - Careers\nWe are hiring across {len(c['subsidiaries']) + 1} sites, "
            f"including {c['planned_hq']} and {c['office']}.\nOur goal is a workforce of "
            f"{c['employee_target']} people.\n"
        ),
        "sustainability": (
            f"{name} - Sustainability\nWe cut emissions by {c['growth'] + 9} percent since "
            f"{c['former_ceo_from']}.\nThe sustainability report is reviewed by "
            f"{_AUDITORS[(_AUDITORS.index(c['auditor']) + 3) % len(_AUDITORS)]}.\n"
        ),
        "faq": (
            f"{name} - Investor FAQ\nQ: When does the fiscal year end? A: See the financial "
            "highlights page.\nQ: Who audits the accounts? A: See the governance page.\n"
            "Q: Where are dividends announced? A: On the dividends page.\n"
        ),
        "contact": (
            f"{name} - Contact\nInvestor relations, {c['hq']}.\nPress office, {c['planned_hq']}.\n"
        ),
    }
    for year in PORTAL_YEARS:
        if year == c["acquisition_year"]:
            body = (
                f"In {year}, {name} completed the acquisition of {c['acquired_company']} "
                f"(founded in {c['target_hq']}) for USD {c['acquisition_price']:.1f} million.\n"
            )
        else:
            body = (
                f"In {year}, {name} opened a facility in "
                f"{c['subsidiaries'][year % 3]['city']} and reported "
                f"{c['growth'] + year % 7} percent order growth.\n"
            )
        pages[f"press-{year}"] = f"{name} - Press {year}\n{body}"
    for s in c["subsidiaries"]:
        pages[f"subsidiary-{_slug(s['name'])}"] = _p_subsidiary(c, s)
    pages["index"] = f"{name} - Investor Portal\nSections: " + ", ".join(sorted(pages)) + ".\n"
    return pages


# -- Class A question templates ---------------------------------------------------------------
def _sub(c: dict[str, Any], which: str) -> dict[str, Any]:
    key = min if which == "oldest" else max
    return key(c["subsidiaries"], key=lambda s: s["founded"])


def _seg(c: dict[str, Any], which: str) -> tuple[dict[str, Any], str]:
    key = max if which == "largest" else min
    i = key(range(3), key=lambda k: c["segments"][k]["revenue"])
    revenues = [s["revenue"] for s in c["segments"]]
    if len(set(revenues)) < 3:
        raise Ambiguous("tied segment revenues")
    return c["segments"][i], c["segment_heads"][i]


A_FIELD: dict[str, tuple[FieldType, MatcherConfig, Callable[[dict[str, Any]], Any]]] = {
    "hq": (FieldType.STRING, NORM, lambda c: c["hq"]),
    "founded": (FieldType.INTEGER, INT0, lambda c: c["founded"]),
    "ceo": (FieldType.STRING, NORM, lambda c: c["ceo"]),
    "former_ceo": (FieldType.STRING, NORM, lambda c: c["former_ceo"]),
    "ceo_since": (FieldType.INTEGER, INT0, lambda c: c["ceo_since"]),
    "cfo": (FieldType.STRING, NORM, lambda c: c["cfo"]),
    "chair": (FieldType.STRING, NORM, lambda c: c["chair"]),
    "employees": (FieldType.INTEGER, INT0, lambda c: c["employees"]),
    "revenue": (FieldType.NUMBER, TENTH, lambda c: c["revenue"]),
    "prev_revenue": (FieldType.NUMBER, TENTH, lambda c: c["prev_revenue"]),
    "fy_end": (FieldType.DATE, DATE, lambda c: c["fy_end"]),
    "products": (FieldType.STRING_LIST, SET, lambda c: list(c["products"])),
    "ticker": (FieldType.STRING, NORM, lambda c: c["ticker"]),
    "exchange": (FieldType.STRING, NORM, lambda c: c["exchange"]),
    "auditor": (FieldType.STRING, NORM, lambda c: c["auditor"]),
    "acquired_company": (FieldType.STRING, NORM, lambda c: c["acquired_company"]),
    "acquisition_year": (FieldType.INTEGER, INT0, lambda c: c["acquisition_year"]),
    "acquisition_price": (FieldType.NUMBER, TENTH, lambda c: c["acquisition_price"]),
    "target_original_hq": (FieldType.STRING, NORM, lambda c: c["target_hq"]),
    "ceo_prev_employer": (FieldType.STRING, NORM, lambda c: c["prev_employer"]),
    "prev_employer_hq": (FieldType.STRING, NORM, lambda c: c["prev_employer_hq"]),
    "dividend_amount": (FieldType.NUMBER, CENTS, lambda c: c["dividend_amount"]),
    "dividend_date": (FieldType.DATE, DATE, lambda c: c["dividend_date"]),
    "oldest_subsidiary": (FieldType.STRING, NORM, lambda c: _sub(c, "oldest")["name"]),
    "oldest_subsidiary_city": (FieldType.STRING, NORM, lambda c: _sub(c, "oldest")["city"]),
    "oldest_subsidiary_founded": (FieldType.INTEGER, INT0, lambda c: _sub(c, "oldest")["founded"]),
    "youngest_subsidiary": (FieldType.STRING, NORM, lambda c: _sub(c, "youngest")["name"]),
    "youngest_subsidiary_city": (FieldType.STRING, NORM, lambda c: _sub(c, "youngest")["city"]),
    "youngest_subsidiary_founded": (
        FieldType.INTEGER,
        INT0,
        lambda c: _sub(c, "youngest")["founded"],
    ),
    "office_subsidiary": (
        FieldType.STRING,
        NORM,
        lambda c: c["subsidiaries"][c["office_subsidiary"]]["name"],
    ),
    "office_subsidiary_md": (
        FieldType.STRING,
        NORM,
        lambda c: c["subsidiaries"][c["office_subsidiary"]]["md"],
    ),
    "office_subsidiary_founded": (
        FieldType.INTEGER,
        INT0,
        lambda c: c["subsidiaries"][c["office_subsidiary"]]["founded"],
    ),
    "largest_segment": (FieldType.STRING, NORM, lambda c: _seg(c, "largest")[0]["name"]),
    "largest_segment_revenue": (
        FieldType.NUMBER,
        TENTH,
        lambda c: _seg(c, "largest")[0]["revenue"],
    ),
    "largest_segment_head": (FieldType.STRING, NORM, lambda c: _seg(c, "largest")[1]),
    "smallest_segment": (FieldType.STRING, NORM, lambda c: _seg(c, "smallest")[0]["name"]),
    "smallest_segment_revenue": (
        FieldType.NUMBER,
        TENTH,
        lambda c: _seg(c, "smallest")[0]["revenue"],
    ),
    "smallest_segment_head": (FieldType.STRING, NORM, lambda c: _seg(c, "smallest")[1]),
}

# (shape, asked fields, question). The question names the company via "{name}".
ATemplate = tuple[str, list[str], str]

A_STD_SEARCH: list[ATemplate] = [
    ("single_page_lookup", ["ticker", "exchange"],
     "on which exchange do {name} shares trade, and under what ticker?"),
    ("single_page_lookup", ["cfo", "chair"],
     "who is the chief financial officer of {name}, and who chairs its board?"),
    ("distractor", ["ceo", "former_ceo"],
     "who is the current chief executive of {name}, and who held the role immediately before?"),
    ("distractor", ["revenue", "employees"],
     "what was {name}'s latest reported annual revenue (USD millions, not a forecast) and its "
     "current number of employees (not a target)?"),
    ("distractor", ["hq", "prev_revenue"],
     "where is {name} headquartered today (not a planned relocation), and what was its "
     "prior-year revenue in USD millions?"),
    ("multi_page", ["auditor", "chair", "founded"],
     "who audits {name}, who chairs its board, and in what year was it founded?"),
    ("multi_page", ["ceo", "fy_end", "products"],
     "who leads {name} as CEO, when does its fiscal year end, and what products does it sell?"),
    ("distractor", ["employees", "hq", "cfo"],
     "how many people does {name} employ today, in which city is its current headquarters, and "
     "who is its CFO?"),
]  # fmt: skip
A_STD_HELDOUT: list[ATemplate] = [
    ("single_page_lookup", ["auditor", "fy_end"],
     "which firm audits {name}'s financial statements, and on what date does its fiscal year "
     "end?"),
    ("distractor", ["ceo", "revenue"],
     "name {name}'s chief executive today and its most recent actual annual revenue in USD "
     "millions (ignore forecasts)."),
    ("multi_page", ["exchange", "cfo", "employees"],
     "which stock exchange lists {name}, who is the finance chief, and what is the headcount?"),
    ("distractor", ["former_ceo", "hq", "prev_revenue"],
     "who was {name}'s previous chief executive, where is the company headquartered now, and "
     "what revenue (USD millions) did it report for the prior year?"),
]  # fmt: skip
A_LONG_SEARCH: list[ATemplate] = [
    ("cross_document_chain", ["oldest_subsidiary", "oldest_subsidiary_city",
                              "oldest_subsidiary_founded"],
     "which of {name}'s subsidiaries was founded first, where is it based, and in what year was "
     "it founded?"),
    ("cross_document_chain", ["largest_segment", "largest_segment_revenue",
                              "largest_segment_head"],
     "which of {name}'s business segments had the highest revenue last fiscal year, what was "
     "that revenue (USD millions), and who heads that segment?"),
    ("cross_document_chain", ["acquired_company", "acquisition_price", "target_original_hq"],
     "which company did {name} acquire, how much did it pay (USD millions), and where was the "
     "acquired company originally headquartered?"),
    ("wide_schema", ["hq", "ceo", "auditor", "fy_end", "employees"],
     "give {name}'s headquarters city, chief executive, auditor, fiscal year end date and total "
     "employee count."),
    ("wide_schema", ["ticker", "exchange", "cfo", "revenue"],
     "give {name}'s ticker symbol, listing exchange, chief financial officer and latest annual "
     "revenue in USD millions."),
    ("cross_document_chain", ["office_subsidiary", "office_subsidiary_md",
                              "office_subsidiary_founded"],
     "{name} keeps a regional office in the same city as one of its subsidiaries: which "
     "subsidiary is that, who is its managing director, and when was it founded?"),
    ("cross_document_chain", ["smallest_segment", "smallest_segment_revenue",
                              "smallest_segment_head", "fy_end"],
     "which of {name}'s segments had the lowest revenue, how much was it (USD millions), who "
     "heads it, and when does the fiscal year end?"),
    ("cross_document_chain", ["ceo", "ceo_since", "ceo_prev_employer", "prev_employer_hq"],
     "who is {name}'s chief executive, since what year, which company did they lead before, and "
     "where is that company headquartered?"),
]  # fmt: skip
A_LONG_HELDOUT: list[ATemplate] = [
    ("cross_document_chain", ["youngest_subsidiary", "youngest_subsidiary_city",
                              "youngest_subsidiary_founded"],
     "which is {name}'s most recently founded subsidiary, in which city is it based, and what "
     "year was it founded?"),
    ("wide_schema", ["founded", "chair", "cfo", "auditor", "revenue"],
     "report {name}'s founding year, board chair, chief financial officer, external auditor and "
     "latest annual revenue (USD millions)."),
    ("cross_document_chain", ["acquired_company", "acquisition_year", "target_original_hq",
                              "acquisition_price"],
     "name the company {name} acquired, the year of the deal, the acquired company's original "
     "headquarters city and the price paid in USD millions."),
    ("cross_document_chain", ["largest_segment", "largest_segment_head",
                              "largest_segment_revenue", "fy_end"],
     "identify {name}'s top segment by revenue, the person heading it, that segment's revenue "
     "(USD millions) and the fiscal year end date."),
]  # fmt: skip
A_JEV_SEARCH: list[ATemplate] = [
    ("interactive_navigation", ["cfo", "dividend_date"],
     "using the {name} investor portal, find the chief financial officer and the payment date "
     "of the most recent dividend."),
    ("interactive_navigation", ["acquired_company", "acquisition_year"],
     "browse the {name} investor portal's press releases: which company did {name} acquire, "
     "and in which year?"),
    ("interactive_navigation", ["oldest_subsidiary", "oldest_subsidiary_city"],
     "navigate the {name} investor portal to find its oldest subsidiary and the city it is "
     "based in."),
    ("interactive_navigation", ["auditor", "chair", "fy_end"],
     "on the {name} investor portal, find the external auditor, the board chair and the fiscal "
     "year end date."),
    ("interactive_navigation", ["largest_segment", "largest_segment_revenue"],
     "using the {name} investor portal, which business segment earned the most revenue last "
     "year, and how much (USD millions)?"),
    ("interactive_navigation", ["dividend_amount", "ceo"],
     "on the {name} investor portal, find the most recent dividend per share (USD) and the "
     "current chief executive."),
]  # fmt: skip
A_JEV_HELDOUT: list[ATemplate] = [
    ("interactive_navigation", ["cfo", "auditor"],
     "navigate the {name} investor portal to find the finance chief and the firm that audits "
     "the accounts."),
    ("interactive_navigation", ["youngest_subsidiary", "youngest_subsidiary_founded"],
     "using the {name} investor portal, which subsidiary was founded most recently, and in what "
     "year?"),
    ("interactive_navigation", ["acquisition_price", "target_original_hq"],
     "browse the {name} investor portal: how much (USD millions) did {name} pay for its "
     "acquisition, and where was the acquired company founded?"),
    ("interactive_navigation", ["dividend_amount", "dividend_date", "chair"],
     "on the {name} investor portal, find the latest dividend per share (USD), its payment date "
     "and the chair of the board."),
]  # fmt: skip


def _a_fields(c: dict[str, Any], asked: list[str]) -> Fields:
    return {f: (A_FIELD[f][0], A_FIELD[f][1], A_FIELD[f][2](c)) for f in asked}


def _a_tasks(
    templates: list[ATemplate],
    prefix: str,
    suite: str,
    *,
    heldout: bool,
    entity_base: int,
    pages: Callable[[dict[str, Any]], dict[str, str]],
    sources: tuple[GatherSource, ...],
    caps: Caps,
    first: int = 1,
    interaction: bool = False,
    acquisition_years: tuple[int, int] = (1995, 2023),
) -> list[NewTask]:
    out = []
    for n, (shape, asked, question) in enumerate(templates, start=first):
        entity = entity_base + n - first
        variant = 0
        while True:  # deterministic: the first variant whose truth is unambiguous
            c = dossier(
                entity, heldout=heldout, acquisition_years=acquisition_years, variant=variant
            )
            try:
                fields = _a_fields(c, asked)
                break
            except Ambiguous:
                variant += 1
        name = c["name"]
        possessive = f"{name}'" if name.endswith("s") else f"{name}'s"
        q = question.format(name=name).replace(f"{name}'s", possessive)
        out.append(
            NewTask(
                task_id=f"{prefix}-{n:03d}",
                suite=suite,
                shape=shape,
                question=q[0].upper() + q[1:],
                sources=sources,
                caps=caps,
                fields=fields,
                pages=pages(c),
                records={"company": [c]},
                interaction_required=interaction,
            )
        )
    return out


# -- Class B: one inventory endpoint ------------------------------------------------------------
def _only(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any], best=max):
    target = best(key(r) for r in rows)
    winners = [r for r in rows if key(r) == target]
    if len(winners) != 1:
        raise Ambiguous("tie")
    return winners[0]


def _top_k(rows, key, k: int, *, reverse: bool = True) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=key, reverse=reverse)
    if len(ranked) <= k or key(ranked[k - 1]) == key(ranked[k]):
        raise Ambiguous("tie at the top-k boundary")
    return ranked[:k]


def _group(rows, by: str, value: Callable[[list[dict[str, Any]]], Any], best=max):
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[by], []).append(r)
    scores = {g: value(rs) for g, rs in groups.items()}
    target = best(scores.values())
    winners = [g for g, s in scores.items() if s == target]
    if len(winners) != 1:
        raise Ambiguous("tied groups")
    return winners[0], target


def _q2(d: str) -> bool:
    return "2024-04-01" <= d <= "2024-06-30"


BTemplate = Callable[[list[dict[str, Any]]], tuple[str, str, Fields]]


def _b1(i):
    top = _top_k(i, lambda r: r["stock"], 3)
    return (
        "top_k_sort",
        "List the names of the 3 items with the highest stock.",
        {"item_names": (FieldType.STRING_LIST, SET, [r["name"] for r in top])},
    )


def _b2(i):
    n = sum(1 for r in i if _q2(r["restock_date"]))
    return (
        "date_aggregate",
        "How many items have a restock date in the second quarter of 2024 (2024-04-01 to "
        "2024-06-30 inclusive)?",
        {"count": (FieldType.INTEGER, INT0, n)},
    )


def _b3(i):
    wh, total = _group(i, "warehouse", lambda rs: sum(r["stock"] for r in rs))
    return (
        "group_by",
        "Which warehouse holds the largest total stock, and what is that total?",
        {
            "warehouse": (FieldType.STRING, NORM, wh),
            "total_stock": (FieldType.INTEGER, INT0, total),
        },
    )


def _b4(i):
    rows = [r for r in i if r["warehouse"] == "east" and r["unit_price"] < 30]
    if not rows:
        raise Ambiguous("empty filter")
    return (
        "filter_aggregate",
        "What is the total stock of items in the 'east' warehouse with a unit price below 30?",
        {"total_stock": (FieldType.INTEGER, INT0, sum(r["stock"] for r in rows))},
    )


def _b5(i):
    top = _only([r for r in i if r["warehouse"] == "south"], lambda r: r["restock_date"])
    return (
        "date_aggregate",
        "What is the latest restock date among items in the 'south' warehouse, and which item "
        "is restocked on that date?",
        {
            "latest_restock": (FieldType.DATE, DATE, top["restock_date"]),
            "item": (FieldType.STRING, NORM, top["name"]),
        },
    )


def _b6(i):
    cat, n = _group(i, "category", len)
    return (
        "group_by",
        "Which category has the most items, and how many items does it have?",
        {"category": (FieldType.STRING, NORM, cat), "count": (FieldType.INTEGER, INT0, n)},
    )


def _b7(i):
    prices = [r["unit_price"] for r in i if r["category"] == "tools"]
    if len(prices) < 2:
        raise Ambiguous("too few tools")
    return (
        "filter_aggregate",
        "What is the difference between the highest and the lowest unit price among items in "
        "category 'tools' (2 decimals)?",
        {"price_range": (FieldType.NUMBER, CENTS, round(max(prices) - min(prices), 2))},
    )


def _b8(i):
    skus = [r["sku"] for r in i if r["category"] == "paint" and r["stock"] > 100]
    if not skus:
        raise Ambiguous("empty filter")
    return (
        "filter_list",
        "List the SKUs of items in category 'paint' with stock above 100.",
        {"skus": (FieldType.STRING_LIST, SET, skus)},
    )


def _bh1(i):
    low = _top_k(i, lambda r: r["unit_price"], 2, reverse=False)
    return (
        "top_k_sort",
        "Which 2 items have the lowest unit prices? Give their names.",
        {"item_names": (FieldType.STRING_LIST, SET, [r["name"] for r in low])},
    )


def _bh2(i):
    n = sum(1 for r in i if r["restock_date"] >= "2024-07-01")
    return (
        "date_aggregate",
        "How many items are due for restocking on or after 2024-07-01?",
        {"count": (FieldType.INTEGER, INT0, n)},
    )


def _bh3(i):
    cat, avg = _group(
        i, "category", lambda rs: round(sum(r["unit_price"] for r in rs) / len(rs), 2)
    )
    return (
        "group_by",
        "Which category has the highest average unit price, and what is that average (2 decimals)?",
        {"category": (FieldType.STRING, NORM, cat), "avg_price": (FieldType.NUMBER, CENTS, avg)},
    )


def _bh4(i):
    rows = [r for r in i if r["warehouse"] == "north"]
    if not rows:
        raise Ambiguous("empty filter")
    return (
        "filter_aggregate",
        "What is the average stock per item in the 'north' warehouse (2 decimals)?",
        {
            "avg_stock": (
                FieldType.NUMBER,
                CENTS,
                round(sum(r["stock"] for r in rows) / len(rows), 2),
            )
        },
    )


def _on_items(template: BTemplate) -> Callable[[dict[str, list[dict[str, Any]]]], Any]:
    return lambda endpoints: template(endpoints["items"])


B_STD_SEARCH: list[BTemplate] = [_b1, _b2, _b3, _b4, _b5, _b6, _b7, _b8]
B_STD_HELDOUT: list[BTemplate] = [_bh1, _bh2, _bh3, _bh4]


# -- Class B: four related endpoints (retail operations) ---------------------------------------
_SUPPLIERS = [
    "Halvorsen Metals", "Petrakis Supply", "Oakridge Components", "Lumio Parts",
    "Brightwater Trading", "Kessler Werk", "Montalvo Industrial", "Nordlys Supply",
    "Castellan Tools", "Fairhaven Wholesale", "Ironbark Imports", "Saltmarsh Goods",
]  # fmt: skip
_COUNTRIES = ["Germany", "Norway", "Spain", "Poland", "Italy", "Ireland"]
_REGIONS = ["north", "south", "east", "west"]
_ITEM_NAMES = [
    "Anvil", "Bolt", "Clamp", "Drill", "Easel", "Funnel", "Gasket", "Hinge", "Ingot", "Jig",
    "Kettle", "Lathe", "Mallet", "Nozzle", "Oilcan", "Pulley", "Quill", "Rivet", "Spanner",
    "Tongs",
]  # fmt: skip
_CATEGORIES = ["tools", "fasteners", "plumbing", "paint"]


def retail(seed: int) -> dict[str, list[dict[str, Any]]]:
    r = _Rng(seed)
    suppliers = [
        {
            "supplier_id": f"SUP-{k + 1:02d}",
            "name": name,
            "country": r.pick(_COUNTRIES[:4]),
            "rating": r.between(20, 50) / 10,
        }
        for k, name in enumerate(r.sample(_SUPPLIERS, 5))
    ]
    stores = [
        {"store_id": f"ST-{k + 1}", "city": city, "region": r.pick(_REGIONS)}
        for k, city in enumerate(r.sample(_CITIES, 4))
    ]
    items = [
        {
            "sku": f"SKU-{2000 + seed * 13 + k * 7}",
            "name": f"{name} {r.between(10, 99)}",
            "category": r.pick(_CATEGORIES),
            "supplier_id": suppliers[r.below(5)]["supplier_id"],
            "unit_price": round(r.between(150, 9000) / 100, 2),
        }
        for k, name in enumerate(r.sample(_ITEM_NAMES, 12))
    ]
    orders = [
        {
            "order_id": f"ORD-{1000 + k}",
            "sku": items[r.below(12)]["sku"],
            "store_id": stores[r.below(4)]["store_id"],
            "quantity": r.between(1, 60),
            "order_date": f"2024-{r.between(1, 6):02d}-{r.between(1, 28):02d}",
        }
        for k in range(24)
    ]
    return {"items": items, "orders": orders, "stores": stores, "suppliers": suppliers}


class _Retail:
    """Joined views over one retail snapshot (ground-truth side only)."""

    def __init__(self, d: dict[str, list[dict[str, Any]]]) -> None:
        self.d = d
        self.item = {i["sku"]: i for i in d["items"]}
        self.supplier = {s["supplier_id"]: s for s in d["suppliers"]}
        self.store = {s["store_id"]: s for s in d["stores"]}

    def orders(self) -> list[dict[str, Any]]:
        """Each order joined with its item, supplier and store."""
        out = []
        for o in self.d["orders"]:
            item = self.item[o["sku"]]
            out.append(
                {
                    **o,
                    "item": item,
                    "supplier": self.supplier[item["supplier_id"]],
                    "store": self.store[o["store_id"]],
                    "revenue": o["quantity"] * item["unit_price"],
                }
            )
        return out


def _j1(d):
    v = _Retail(d)
    top = _only(d["items"], lambda i: i["unit_price"])
    s = v.supplier[top["supplier_id"]]
    return (
        "multi_endpoint_join",
        "Which supplier supplies the most expensive item, and in which country is that supplier "
        "based?",
        {
            "supplier": (FieldType.STRING, NORM, s["name"]),
            "country": (FieldType.STRING, NORM, s["country"]),
        },
    )


def _j2(d):
    v = _Retail(d)
    country = d["suppliers"][0]["country"]
    qty = sum(o["quantity"] for o in v.orders() if o["supplier"]["country"] == country)
    if qty == 0:
        raise Ambiguous("no orders")
    return (
        "join_aggregate",
        f"What is the total quantity ordered of items supplied by suppliers based in {country}?",
        {"total_quantity": (FieldType.INTEGER, INT0, qty)},
    )


def _j3(d):
    v = _Retail(d)
    top = _only(v.orders(), lambda o: o["quantity"])
    return (
        "dependency_chain",
        "Which city's store placed the single largest order by quantity, and which supplier "
        "supplied the item in that order?",
        {
            "city": (FieldType.STRING, NORM, top["store"]["city"]),
            "supplier": (FieldType.STRING, NORM, top["supplier"]["name"]),
        },
    )


def _j4(d):
    v = _Retail(d)
    cat = d["items"][0]["category"]
    n = sum(
        1 for o in v.orders() if o["order_date"][:7] == "2024-03" and o["item"]["category"] == cat
    )
    return (
        "date_aggregate",
        f"How many orders were placed in March 2024 for items in category '{cat}'?",
        {"count": (FieldType.INTEGER, INT0, n)},
    )


def _per_supplier_qty(v: _Retail) -> list[dict[str, Any]]:
    totals: dict[str, int] = {}
    for o in v.orders():
        totals[o["supplier"]["name"]] = totals.get(o["supplier"]["name"], 0) + o["quantity"]
    return [{"name": k, "qty": q} for k, q in totals.items()]


def _j5(d):
    top = _top_k(_per_supplier_qty(_Retail(d)), lambda s: s["qty"], 3)
    return (
        "top_k_sort",
        "Name the 3 suppliers whose items were ordered in the largest total quantity.",
        {"suppliers": (FieldType.STRING_LIST, SET, [s["name"] for s in top])},
    )


def _j6(d):
    v = _Retail(d)
    region = d["stores"][0]["region"]
    total = round(sum(o["revenue"] for o in v.orders() if o["store"]["region"] == region), 2)
    return (
        "join_aggregate",
        f"What is the total order revenue (quantity x unit price) of stores in region "
        f"'{region}', rounded to 2 decimals?",
        {"revenue": (FieldType.NUMBER, CENTS, total)},
    )


def _j7(d):
    v = _Retail(d)
    rows = [{"region": o["store"]["region"]} for o in v.orders()]
    region, n = _group(rows, "region", len)
    return (
        "group_by",
        "Which store region placed the most orders, and how many orders was that?",
        {"region": (FieldType.STRING, NORM, region), "count": (FieldType.INTEGER, INT0, n)},
    )


def _j8(d):
    v = _Retail(d)
    name = d["suppliers"][1]["name"]
    dates = [o["order_date"] for o in v.orders() if o["supplier"]["name"] == name]
    if not dates:
        raise Ambiguous("no orders")
    return (
        "date_aggregate",
        f"What is the earliest order date for items supplied by {name}?",
        {"earliest_order": (FieldType.DATE, DATE, min(dates))},
    )


def _j9(d):
    v = _Retail(d)
    city = d["stores"][1]["city"]
    sups = {
        o["supplier"]["supplier_id"]: o["supplier"]
        for o in v.orders()
        if o["store"]["city"] == city
    }
    if not sups:
        raise Ambiguous("no orders")
    avg = round(sum(s["rating"] for s in sups.values()) / len(sups), 2)
    return (
        "join_aggregate",
        f"What is the average rating of the distinct suppliers whose items were ordered by the "
        f"store in {city} (2 decimals)?",
        {"avg_rating": (FieldType.NUMBER, CENTS, avg)},
    )


def _j10(d):
    ordered = {o["sku"] for o in d["orders"]}
    names = [i["name"] for i in d["items"] if i["sku"] not in ordered]
    if not names:
        raise Ambiguous("every item was ordered")
    return (
        "anti_join",
        "List the names of items that were never ordered.",
        {"item_names": (FieldType.STRING_LIST, SET, names)},
    )


def _j11(d):
    v = _Retail(d)
    skus: dict[str, set[str]] = {}
    for o in v.orders():
        skus.setdefault(o["store"]["city"], set()).add(o["sku"])
    rows = [{"city": c, "n": len(s)} for c, s in skus.items()]
    top = _only(rows, lambda r: r["n"])
    return (
        "group_by",
        "Which store (by city) ordered the largest number of distinct SKUs, and how many?",
        {
            "city": (FieldType.STRING, NORM, top["city"]),
            "distinct_skus": (FieldType.INTEGER, INT0, top["n"]),
        },
    )


def _j12(d):
    v = _Retail(d)
    s = d["suppliers"][2]
    n_items = sum(1 for i in d["items"] if i["supplier_id"] == s["supplier_id"])
    if n_items == 0:
        raise Ambiguous("supplier without items")
    revenue = round(
        sum(o["revenue"] for o in v.orders() if o["supplier"]["supplier_id"] == s["supplier_id"]), 2
    )
    return (
        "join_aggregate",
        f"For supplier {s['name']}: how many items does it supply, and what is the total order "
        "revenue (quantity x unit price) of those items, rounded to 2 decimals?",
        {
            "item_count": (FieldType.INTEGER, INT0, n_items),
            "revenue": (FieldType.NUMBER, CENTS, revenue),
        },
    )


def _j13(d):
    v = _Retail(d)
    region = d["stores"][2]["region"]
    rows = [
        {"name": o["supplier"]["name"], "q": o["quantity"], "s": o["supplier"]}
        for o in v.orders()
        if o["store"]["region"] == region
    ]
    if not rows:
        raise Ambiguous("no orders")
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["name"]] = totals.get(row["name"], 0) + row["q"]
    best = _only([{"name": k, "q": q} for k, q in totals.items()], lambda r: r["q"])
    rating = next(s["rating"] for s in d["suppliers"] if s["name"] == best["name"])
    return (
        "dependency_chain",
        f"Among orders from stores in region '{region}', which supplier's items were ordered in "
        "the largest total quantity, and what is that supplier's rating?",
        {
            "supplier": (FieldType.STRING, NORM, best["name"]),
            "rating": (FieldType.NUMBER, CENTS, rating),
        },
    )


def _j14(d):
    v = _Retail(d)
    price = 40
    qty = sum(
        o["quantity"]
        for o in v.orders()
        if o["order_date"] <= "2024-03-31" and o["item"]["unit_price"] > price
    )
    if qty == 0:
        raise Ambiguous("empty")
    return (
        "date_aggregate",
        f"What total quantity was ordered in the first quarter of 2024 (January to March) for "
        f"items with a unit price above {price}?",
        {"total_quantity": (FieldType.INTEGER, INT0, qty)},
    )


def _jh1(d):
    v = _Retail(d)
    low = _only(d["items"], lambda i: i["unit_price"], best=min)
    s = v.supplier[low["supplier_id"]]
    return (
        "multi_endpoint_join",
        "Who supplies the cheapest item, and what is that supplier's rating?",
        {
            "supplier": (FieldType.STRING, NORM, s["name"]),
            "rating": (FieldType.NUMBER, CENTS, s["rating"]),
        },
    )


def _jh2(d):
    v = _Retail(d)
    region = d["stores"][3]["region"]
    qty = sum(o["quantity"] for o in v.orders() if o["store"]["region"] == region)
    if qty == 0:
        raise Ambiguous("no orders")
    return (
        "join_aggregate",
        f"How many units in total were ordered by stores in region '{region}'?",
        {"total_quantity": (FieldType.INTEGER, INT0, qty)},
    )


def _jh3(d):
    v = _Retail(d)
    totals: dict[str, float] = {}
    for o in v.orders():
        totals[o["sku"]] = totals.get(o["sku"], 0.0) + o["revenue"]
    best = _only([{"sku": k, "r": round(x, 2)} for k, x in totals.items()], lambda r: r["r"])
    item = v.item[best["sku"]]
    return (
        "dependency_chain",
        "Which item generated the highest total order revenue (quantity x unit price), and which "
        "supplier supplies it?",
        {
            "item": (FieldType.STRING, NORM, item["name"]),
            "supplier": (FieldType.STRING, NORM, v.supplier[item["supplier_id"]]["name"]),
        },
    )


def _jh4(d):
    v = _Retail(d)
    city = d["stores"][0]["city"]
    rows = [o for o in v.orders() if o["store"]["city"] == city]
    if not rows:
        raise Ambiguous("no orders")
    last = _only(rows, lambda o: o["order_date"])
    return (
        "date_aggregate",
        f"What is the most recent order date at the store in {city}, and which item was ordered "
        "then?",
        {
            "latest_order": (FieldType.DATE, DATE, last["order_date"]),
            "item": (FieldType.STRING, NORM, last["item"]["name"]),
        },
    )


def _jh5(d):
    v = _Retail(d)
    totals: dict[str, float] = {}
    for o in v.orders():
        totals[o["store"]["city"]] = totals.get(o["store"]["city"], 0.0) + o["revenue"]
    top = _top_k([{"city": k, "r": round(x, 2)} for k, x in totals.items()], lambda r: r["r"], 2)
    return (
        "top_k_sort",
        "Which 2 store cities have the highest total order revenue (quantity x unit price)?",
        {"cities": (FieldType.STRING_LIST, SET, [r["city"] for r in top])},
    )


def _jh6(d):
    v = _Retail(d)
    rows = [{"country": o["supplier"]["country"]} for o in v.orders()]
    country, n = _group(rows, "country", len)
    return (
        "group_by",
        "Suppliers from which country received the most orders (counting orders, not units), "
        "and how many?",
        {"country": (FieldType.STRING, NORM, country), "count": (FieldType.INTEGER, INT0, n)},
    )


def _jh7(d):
    v = _Retail(d)
    cat = d["items"][1]["category"]
    q = [o["quantity"] for o in v.orders() if o["item"]["category"] == cat]
    if not q:
        raise Ambiguous("no orders")
    return (
        "join_aggregate",
        f"What is the average quantity per order for items in category '{cat}' (2 decimals)?",
        {"avg_quantity": (FieldType.NUMBER, CENTS, round(sum(q) / len(q), 2))},
    )


def _jh8(d):
    v = _Retail(d)
    active = {o["supplier"]["name"] for o in v.orders() if o["order_date"][:7] == "2024-02"}
    names = [s["name"] for s in d["suppliers"] if s["name"] not in active]
    if not names or len(names) == len(d["suppliers"]):
        raise Ambiguous("trivial")
    return (
        "anti_join",
        "Which suppliers had none of their items ordered in February 2024?",
        {"suppliers": (FieldType.STRING_LIST, SET, names)},
    )


B_JOIN_SEARCH: list[BTemplate] = [_j1, _j2, _j3, _j4, _j5, _j6, _j7, _j8, _j9, _j10, _j11, _j12,
                                  _j13, _j14]  # fmt: skip
B_JOIN_HELDOUT: list[BTemplate] = [_jh1, _jh2, _jh3, _jh4, _jh5, _jh6, _jh7, _jh8]


def _b_tasks(
    templates: list[Callable[[dict[str, list[dict[str, Any]]]], Any]],
    prefix: str,
    suite: str,
    *,
    seed_base: int,
    data: Callable[[int], dict[str, list[dict[str, Any]]]],
    caps: Caps,
    first: int = 1,
) -> list[NewTask]:
    out = []
    for n, template in enumerate(templates, start=first):
        seed = seed_base + n
        while True:  # deterministic: the first seed whose truth is unambiguous
            endpoints = data(seed)
            try:
                shape, question, fields = template(endpoints)
                break
            except Ambiguous:
                seed += 1000
        out.append(
            NewTask(
                task_id=f"{prefix}-{n:03d}",
                suite=suite,
                shape=shape,
                question=question,
                sources=FETCH_API,
                caps=caps,
                fields=fields,
                # Class B pages ARE the unfiltered API responses (benchmarks.mock_api)
                pages={ep: render_response(ep, rows) for ep, rows in sorted(endpoints.items())},
                records=endpoints,
            )
        )
    return out


def _inventory(seed: int) -> dict[str, list[dict[str, Any]]]:
    from benchmarks.build import _items  # the MVP generator: same row shape as B-001..B-008

    return {"items": _items(seed, n=18)}


# -- public entry points ------------------------------------------------------------------------
def search_tasks(caps_a: Caps, caps_b: Caps) -> list[NewTask]:
    """New optimization/validation tasks. Suites A / B continue the MVP numbering (A-009,
    B-009, ...); the new suites have their own prefixes (AL-, AJ-, BJ-)."""
    return [
        *_a_tasks(A_STD_SEARCH, "A", "A", heldout=False, entity_base=0, pages=std_pages,
                  sources=FETCH_JEV, caps=caps_a, first=9),
        *_a_tasks(A_LONG_SEARCH, "AL", "Along", heldout=False, entity_base=10, pages=long_pages,
                  sources=FETCH_JEV, caps=CAPS_A_LONG),
        *_a_tasks(A_JEV_SEARCH, "AJ", "Ajev", heldout=False, entity_base=20, pages=portal_pages,
                  sources=JEV_ONLY, caps=CAPS_A_JEV, interaction=True,
                  acquisition_years=(PORTAL_YEARS[0], PORTAL_YEARS[-1])),
        *_b_tasks([_on_items(t) for t in B_STD_SEARCH], "B", "B", seed_base=40,
                  data=_inventory, caps=caps_b, first=9),
        *_b_tasks(B_JOIN_SEARCH, "BJ", "Bjoin", seed_base=100, data=retail, caps=CAPS_B_JOIN),
    ]  # fmt: skip


def heldout_tasks(caps_a: Caps, caps_b: Caps) -> list[NewTask]:
    """New held-out TEST tasks: new entities, new templates of the same shapes."""
    return [
        *_a_tasks(A_STD_HELDOUT, "TA", "A", heldout=True, entity_base=0, pages=std_pages,
                  sources=FETCH_JEV, caps=caps_a, first=9),
        *_a_tasks(A_LONG_HELDOUT, "TAL", "Along", heldout=True, entity_base=10,
                  pages=long_pages, sources=FETCH_JEV, caps=CAPS_A_LONG),
        *_a_tasks(A_JEV_HELDOUT, "TAJ", "Ajev", heldout=True, entity_base=20, pages=portal_pages,
                  sources=JEV_ONLY, caps=CAPS_A_JEV, interaction=True,
                  acquisition_years=(PORTAL_YEARS[0], PORTAL_YEARS[-1])),
        *_b_tasks([_on_items(t) for t in B_STD_HELDOUT], "TB", "B", seed_base=60,
                  data=_inventory, caps=caps_b, first=9),
        *_b_tasks(B_JOIN_HELDOUT, "TBJ", "Bjoin", seed_base=200, data=retail, caps=CAPS_B_JOIN),
    ]  # fmt: skip


# Shapes of the original MVP tasks (B is fixed by its truth function; A by where fields live).
MVP_B_SHAPES = {
    "_b_total_stock": "filter_aggregate",
    "_b_low_stock": "filter_list",
    "_b_priciest": "top_k_sort",
    "_b_earliest": "date_aggregate",
    "_b_avg_price": "filter_aggregate",
    "_b_count": "filter_aggregate",
    "_b_largest_stock": "top_k_sort",
    "_b_value": "filter_aggregate",
}
MVP_A_PAGE_OF = {
    "hq": "overview",
    "founded": "overview",
    "ceo": "overview",
    "products": "overview",
    "employees": "financials",
    "revenue": "financials",
    "fy_end": "financials",
}


def mvp_a_shape(asked: list[str]) -> str:
    return "single_page_lookup" if len({MVP_A_PAGE_OF[f] for f in asked}) == 1 else "multi_page"
