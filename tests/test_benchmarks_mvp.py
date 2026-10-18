"""Frozen benchmark (search + held-out): shape, splits, catalog, fixtures, ground truth, hashes.

Structural validation of every task (#13): ids, snapshots, answer schemas vs ground truth and
matchers, split membership, search/held-out disjointness, JEV interaction contracts, constraint
admissibility and deterministic regeneration. No model, no network.
"""

import json
from datetime import date

import pytest

from benchmarks import build, expansion, heldout
from benchmarks.build import compute_splits, n_validation
from benchmarks.legacy_adapter import legacy_contracts, legacy_suite
from benchmarks.loader import (
    HELDOUT_DIR,
    benchmark_hash,
    load_catalog,
    load_splits,
    load_task_specs,
    runtime_tasks,
    split_specs,
    suite_names,
)
from benchmarks.mock_api import MockAPI, UnknownEndpoint
from benchmarks.snapshot_store import SnapshotStore
from core.constraints import ConstraintChecker, ConstraintConfig
from core.dataset import SplitRole
from core.genome import Genome
from core.stages import GatherSource
from core.task_spec import FieldType, RuntimeTask, TaskClass
from core.violations import ViolationCode
from evaluation.matchers import DefaultMatcher

# Pins of the committed benchmark bytes. A change means the frozen benchmark changed.
# #13: 16 -> 60 search tasks (suites A, Along, Ajev, B, Bjoin) and a catalog.json
GOLDEN_BENCHMARK_HASH = "5e48e773aa5a94a5cbaae247cf270b58ce8f3277b4cfeb4aedc86749b65680d1"
# #13: 16 -> 40 held-out tasks; the original 16 keep their subset hash (OSS comparison)
GOLDEN_HELDOUT_FULL_HASH = "1f283f76999ab4e08a34d87b716667deb1a1bf03bae0d7c5bd9b84d0ea5c4358"

STORE = SnapshotStore()
HSTORE = SnapshotStore(HELDOUT_DIR / "snapshots")
SPECS = load_task_specs()
HSPECS = load_task_specs(HELDOUT_DIR)
CATALOG = load_catalog()
HCATALOG = load_catalog(HELDOUT_DIR)
ALL = [(s, STORE, CATALOG) for s in SPECS.values()] + [
    (s, HSTORE, HCATALOG) for s in HSPECS.values()
]
SUITES = ("A", "Ajev", "Along", "B", "Bjoin")

# What a value of each answer type must look like in the ground truth.
_TYPE_OK = {
    FieldType.STRING: lambda v: isinstance(v, str) and v.strip() != "",
    FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
    FieldType.NUMBER: lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    FieldType.BOOLEAN: lambda v: isinstance(v, bool),
    FieldType.DATE: lambda v: isinstance(v, str) and date.fromisoformat(v) is not None,
    FieldType.STRING_LIST: lambda v: (
        isinstance(v, list) and v and all(isinstance(x, str) for x in v)
    ),
}


def _ids(rows):
    return [s.id for s, _, _ in rows]


# -- size, balance, catalog ---------------------------------------------------------------------
def test_benchmark_has_at_least_80_tasks_balanced_between_class_a_and_b():
    classes = [s.runtime.task_class for s, _, _ in ALL]
    assert len(ALL) >= 80
    assert TaskClass.C not in classes
    a, b = classes.count(TaskClass.A), classes.count(TaskClass.B)
    assert abs(a - b) <= 0.1 * len(ALL), (a, b)


def test_all_task_ids_are_unique_across_search_and_heldout():
    ids = _ids(ALL)
    assert len(ids) == len(set(ids))
    assert not set(SPECS) & set(HSPECS)


def test_catalog_covers_exactly_the_tasks_and_names_known_suites():
    assert set(CATALOG) == set(SPECS) and set(HCATALOG) == set(HSPECS)
    assert suite_names() == SUITES
    for spec, _, catalog in ALL:
        meta = catalog[spec.id]
        assert meta.suite in expansion.SUITES and meta.shape and meta.shape != "unknown"
        assert expansion.SUITES[meta.suite].task_class is spec.runtime.task_class


def test_every_suite_has_optimization_validation_and_heldout_rows():
    splits = load_splits()
    for name in SUITES:
        train = [i for i in splits["train"] if CATALOG[i].suite == name]
        val = [i for i in splits["validation"] if CATALOG[i].suite == name]
        test = [i for i in HSPECS if HCATALOG[i].suite == name]
        assert len(train) >= 4 and len(val) >= 2 and len(test) >= 4, name


def test_tasks_of_a_suite_share_one_search_policy():
    for name in SUITES:
        rts = [s.runtime for s, _, cat in ALL if cat[s.id].suite == name]
        ref = rts[0]
        for rt in rts:
            assert (rt.caps, rt.allowed_sources, rt.interaction_required, rt.task_class) == (
                ref.caps,
                ref.allowed_sources,
                ref.interaction_required,
                ref.task_class,
            )
        suite = legacy_suite(name)  # ContractSuite re-checks the shared policy itself
        assert suite.name == name


def test_suites_cover_the_required_task_shapes():
    shapes = {(cat[s.id].suite, cat[s.id].shape) for s, _, cat in ALL}
    want_a = {"single_page_lookup", "multi_page", "distractor", "cross_document_chain",
              "wide_schema", "interactive_navigation"}  # fmt: skip
    want_b = {"filter_aggregate", "filter_list", "top_k_sort", "date_aggregate", "group_by",
              "multi_endpoint_join", "join_aggregate", "dependency_chain", "anti_join"}  # fmt: skip
    assert want_a <= {sh for su, sh in shapes if su.startswith("A")}
    assert want_b <= {sh for su, sh in shapes if su.startswith("B")}
    assert max(len(s.runtime.answer_schema.fields) for s, _, _ in ALL) == 5


def test_no_task_template_is_a_renamed_copy():
    for specs in (SPECS, HSPECS):  # within one directory every question is distinct
        questions = [s.runtime.question for s in specs.values()]
        assert len(questions) == len(set(questions))
    # #13 held-out tasks use their own templates: none repeats a search question. (The original
    # TB-001..TB-008 re-ask B-001..B-008 over new inventories, as frozen for the OSS comparison.)
    search = {s.runtime.question for s in SPECS.values()}
    new_heldout = [s for s in HSPECS.values() if s.id not in heldout.OSS_TEST_IDS]
    assert not {s.runtime.question for s in new_heldout} & search


# -- fixtures and ground truth ------------------------------------------------------------------
def test_every_task_has_its_own_snapshot_with_pages_and_records():
    for spec, store, _ in ALL:
        sid = spec.runtime.snapshot_id
        assert sid == spec.id
        assert store.has_snapshot(sid), sid
        assert store.pages(sid).pages, sid
        assert store.records(sid), sid
    assert set(STORE.snapshot_ids()) == set(SPECS)
    assert set(HSTORE.snapshot_ids()) == set(HSPECS)


def test_answer_schema_ground_truth_and_matchers_agree():
    matcher = DefaultMatcher()
    for spec, _, _ in ALL:
        fields = {f.name: f for f in spec.runtime.answer_schema.fields}
        assert set(fields) == set(spec.ground_truth.values) == set(spec.matchers)
        for name, f in fields.items():
            value = spec.ground_truth.values[name]
            assert _TYPE_OK[f.type](value), (spec.id, name, value)
            # the truth must satisfy its own matcher (an unmatchable truth is a broken task)
            assert matcher.matches(value, value, spec.matchers[name]), (spec.id, name)


def test_class_a_ground_truth_is_literally_present_on_the_snapshot_pages():
    for spec, store, _ in ALL:
        if spec.runtime.task_class is not TaskClass.A:
            continue
        text = " ".join(p.content for p in store.pages(spec.runtime.snapshot_id).pages)
        for field, value in spec.ground_truth.values.items():
            for item in value if isinstance(value, list) else [value]:
                assert str(item) in text, (spec.id, field, item)


def test_class_b_pages_are_the_unfiltered_api_response_of_every_endpoint():
    for spec, store, _ in ALL:
        if spec.runtime.task_class is not TaskClass.B:
            continue
        sid = spec.runtime.snapshot_id
        api = MockAPI(store)
        endpoints = store.records(sid)
        assert sorted(p.page_id for p in store.pages(sid).pages) == sorted(endpoints)
        for endpoint in endpoints:
            assert api.get(sid, endpoint) == store.page(sid, endpoint).content


def test_bjoin_tasks_need_more_than_one_endpoint():
    for spec, store, cat in ALL:
        if cat[spec.id].suite == "Bjoin":
            assert set(store.records(spec.id)) == {"items", "orders", "stores", "suppliers"}


def test_mock_api_filters_come_from_the_same_records_and_are_deterministic():
    api = MockAPI(STORE)
    records = STORE.records("B-001")["items"]
    body = json.loads(api.get("B-001", "items", category="tools"))
    assert body["results"] == [r for r in records if r["category"] == "tools"]
    assert api.get("B-001", "items", category="tools") == api.get(
        "B-001", "items", category="tools"
    )
    with pytest.raises(UnknownEndpoint):
        api.get("B-001", "nope")


def test_class_b_ground_truth_is_exact_over_the_api_records():
    records = STORE.records("B-001")["items"]
    assert SPECS["B-001"].ground_truth.values["total_stock"] == sum(
        r["stock"] for r in records if r["category"] == "tools"
    )
    retail = STORE.records("BJ-010")
    ordered = {o["sku"] for o in retail["orders"]}
    never = [i["name"] for i in retail["items"] if i["sku"] not in ordered]
    assert sorted(SPECS["BJ-010"].ground_truth.values["item_names"]) == sorted(never)


# -- splits and leakage -------------------------------------------------------------------------
def test_every_task_is_in_exactly_one_split():
    main, held = load_splits(), load_splits(HELDOUT_DIR)
    assert set(main) == {"train", "validation"} and set(held) == {"test"}
    rows = [i for ids in (*main.values(), *held.values()) for i in ids]
    assert len(rows) == len(set(rows))
    assert set(main["train"]) | set(main["validation"]) == set(SPECS)
    assert set(held["test"]) == set(HSPECS)


def test_splits_match_the_documented_per_suite_rule():
    splits = load_splits()
    by_suite: dict[str, list[str]] = {}
    for tid, meta in CATALOG.items():
        by_suite.setdefault(meta.suite, []).append(tid)
    assert compute_splits(by_suite) == {k: sorted(v) for k, v in splits.items()}
    for name, ids in by_suite.items():
        val = [i for i in splits["validation"] if i in ids]
        assert len(val) == n_validation(len(ids)), name


def test_search_and_heldout_share_no_rows_snapshots_or_entities():
    assert not set(STORE.snapshot_ids()) & set(HSTORE.snapshot_ids())

    def companies(specs, store):
        out = set()
        for s in specs.values():
            for rec in store.records(s.id).get("company", []):
                out |= {rec["name"], *(sub["name"] for sub in rec.get("subsidiaries", []))}
        return out

    assert not companies(SPECS, STORE) & companies(HSPECS, HSTORE)
    for name in SUITES:
        suite = legacy_suite(name)
        test_rows = set(suite.splits.split(SplitRole.TEST).row_ids)
        search_rows = {*suite.splits.split(SplitRole.OPTIMIZATION).row_ids,
                       *suite.splits.split(SplitRole.VALIDATION).row_ids}  # fmt: skip
        assert test_rows and search_rows and not test_rows & search_rows
        assert test_rows <= set(HSPECS) and search_rows <= set(SPECS)


def test_runtime_view_contains_no_ground_truth_and_is_a_runtime_task():
    for split in ("train", "validation"):
        for t in runtime_tasks(split):
            assert isinstance(t, RuntimeTask)
            assert "ground_truth" not in t.model_dump_json()


def test_split_specs_are_sorted_and_class_filtered():
    for cls in (TaskClass.A, TaskClass.B):
        specs = split_specs("train", cls)
        assert [s.id for s in specs] == sorted(s.id for s in specs)
        assert all(s.runtime.task_class is cls for s in specs)


# -- JEV / interaction contracts ----------------------------------------------------------------
def test_interaction_required_tasks_are_jev_only_portals_too_large_to_fetch():
    jev = [(s, st) for s, st, cat in ALL if s.runtime.interaction_required]
    assert len(jev) >= 8
    for spec, store in jev:
        rt = spec.runtime
        assert rt.allowed_sources == (GatherSource.JEV,)
        # the reason interaction is required: reading every page cannot fit the tool budget
        assert len(store.pages(rt.snapshot_id).pages) > rt.caps.tool_calls
    for spec, store, _ in ALL:  # every other task's pages can all be read within its budget
        if not spec.runtime.interaction_required:
            assert len(store.pages(spec.id).pages) <= spec.runtime.caps.tool_calls, spec.id


def test_interaction_required_contracts_admit_only_jev_gathers():
    checker = ConstraintChecker()  # JEV available: the contract alone decides
    contracts = {**legacy_contracts(), **legacy_contracts(HELDOUT_DIR)}
    stages = [
        {"kind": "GATHER", "source": "fetch", "mode": "sequential"},
        {"kind": "EXTRACT", "method": "schema_guided"},
        {"kind": "SYNTHESIZE", "method": "cite_evidence"},
    ]
    fetch = Genome.from_stages(stages)
    jev = Genome.from_stages([{**stages[0], "source": "jev"}, *stages[1:]])
    for tid, contract in contracts.items():
        if contract.workflow.interaction_required:
            codes = {v.code for v in checker.check(fetch, contract)}
            assert ViolationCode.INTERACTION_REQUIRES_JEV in codes, tid
            assert checker.is_valid(jev, contract), tid


def test_without_a_jev_provider_interaction_tasks_have_no_admissible_workflow():
    checker = ConstraintChecker(config=ConstraintConfig(unavailable_sources=(GatherSource.JEV,)))
    policy = legacy_suite("Ajev").policy
    assert next(checker.enumerate_admissible(policy), None) is None
    # and the frozen fetch/API suites stay runnable with no JEV and no network
    for name in ("A", "Along", "B", "Bjoin"):
        assert next(checker.enumerate_admissible(legacy_suite(name).policy), None) is not None


def test_every_task_contract_admits_at_least_one_workflow():
    checker = ConstraintChecker()
    for tid, contract in {**legacy_contracts(), **legacy_contracts(HELDOUT_DIR)}.items():
        assert next(checker.enumerate_admissible(contract), None) is not None, tid


# -- hashes -------------------------------------------------------------------------------------
def test_benchmark_hashes_are_frozen():
    assert benchmark_hash() == GOLDEN_BENCHMARK_HASH
    assert benchmark_hash(HELDOUT_DIR, HSTORE) == GOLDEN_HELDOUT_FULL_HASH


def test_benchmark_regenerates_byte_identically(tmp_path):
    build.build(tmp_path / "snapshots", tmp_path)
    assert benchmark_hash(tmp_path) == GOLDEN_BENCHMARK_HASH
    for name in ("tasks.json", "splits.json", "catalog.json"):
        assert (tmp_path / name).read_bytes() == (build.BENCH_DIR / name).read_bytes()
    held = tmp_path / "held"
    heldout.build(held)
    assert benchmark_hash(held) == GOLDEN_HELDOUT_FULL_HASH


def test_subset_hash_ignores_other_tasks_and_rejects_unknown_ids():
    a = benchmark_hash(HELDOUT_DIR, task_ids=heldout.OSS_TEST_IDS)
    assert a == benchmark_hash(HELDOUT_DIR, task_ids=reversed(heldout.OSS_TEST_IDS))
    assert a != benchmark_hash(HELDOUT_DIR)
    with pytest.raises(KeyError):
        benchmark_hash(HELDOUT_DIR, task_ids=["NOPE-1"])


def test_legacy_suite_subset_keeps_each_row_in_its_split():
    from core.dataset import SplitRole, SplitUse

    mvp_b = (*(f"B-{i:03d}" for i in range(1, 9)), *(f"TB-{i:03d}" for i in range(1, 9)))
    suite = legacy_suite("B", task_ids=mvp_b)
    ids = lambda role, use: [t.id for t in suite.tasks_for(role, use)]  # noqa: E731
    assert ids(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK) == [
        "B-002", "B-003", "B-004", "B-005", "B-007"
    ]  # fmt: skip
    assert ids(SplitRole.VALIDATION, SplitUse.SELECTION) == ["B-001", "B-006", "B-008"]
    assert ids(SplitRole.TEST, SplitUse.REPORTING) == [f"TB-{i:03d}" for i in range(1, 9)]
    with pytest.raises(KeyError):
        legacy_suite("B", task_ids=["A-001"])
