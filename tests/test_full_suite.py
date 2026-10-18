"""Full-suite runner (#13): budgets, split authority, determinism, per-task completeness, stats.

Everything runs on the labelled synthetic objective - no model, no network.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.legacy_adapter import legacy_suite
from benchmarks.loader import suite_names
from core.dataset import SplitAccessError, SplitRole, SplitUse
from core.genome import Genome
from experiments import full_suite as fs
from experiments import full_suite_report as rep
from experiments.synthetic import synthetic_evaluate

SMALL = fs.Protocol(
    name="test",
    suites=("A", "Ajev", "Bjoin"),
    optimizers=("aco_mmas", "random_search"),
    seeds=(0, 1),
    genome_budget=3,
    batch_size=2,
    search_trials=1,
    final_trials=2,
    lcb_z=1.0,
    workers=2,
    max_concurrent_runs=4,
)


class Spy:
    """Synthetic evaluate that records which task ids ran, phase by phase."""

    def __init__(self) -> None:
        self.phase = "search"
        self.seen: dict[str, list[str]] = {"search": [], "final": []}
        self._lock = threading.Lock()

    def __call__(self, genome, task, trial, seed):
        with self._lock:
            self.seen[self.phase].append(task.id)
        return synthetic_evaluate(genome, task, trial, seed)


def _ids(suite: str, role: SplitRole, use: SplitUse) -> set[str]:
    return {t.id for t in legacy_suite(suite).tasks_for(role, use)}


def _run(tmp_path: Path, protocol: fs.Protocol = SMALL, *, final: bool = True):
    spy = Spy()
    base = fs.synthetic_backend()
    backend = replace(base, search=spy, final=spy)

    def progress(msg: str) -> None:
        if msg.startswith("final:"):
            spy.phase = "final"

    summary = fs.run_protocol(
        protocol, backend, tmp_path, final=final, plots=False, progress=progress
    )
    return spy, summary


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    spy, summary = _run(out)
    return out, spy, summary


def _load(out: Path, name: str):
    return json.loads((out / name).read_text(encoding="utf-8"))


# -- protocol ---------------------------------------------------------------------------------
def test_canonical_protocol_covers_the_whole_benchmark():
    p = fs.Protocol.load()
    assert set(p.suites) == set(suite_names())
    assert {"aco_mmas", "random_search"} <= set(p.optimizers)
    assert set(p.comparison) == {"aco_mmas", "random_search"}
    assert len(p.seeds) >= 3 and p.search_trials >= 1 and p.final_trials >= 1


def test_protocol_rejects_inconsistent_settings():
    with pytest.raises(ValueError):
        replace(SMALL, seeds=(0, 0)).validate()
    with pytest.raises(ValueError):
        replace(SMALL, genome_budget=0).validate()
    with pytest.raises(ValueError):
        replace(SMALL, optimizers=("aco_mmas",)).validate()


# -- budgets ----------------------------------------------------------------------------------
def test_every_optimizer_gets_the_same_candidate_budget(small_run):
    out, _, _ = small_run
    searches = _load(out, "search.json")["searches"]
    for suite in ("A", "Bjoin"):
        n_train = len(_ids(suite, SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK))
        done = [s for s in searches if s["suite"] == suite]
        assert {s["status"] for s in done} == {"completed"}
        budgets = {s["config"]["budget"] for s in done}
        assert budgets == {SMALL.genome_budget * n_train * SMALL.search_trials}
        for s in done:
            assert s["result"]["workflow_evaluations"] == s["config"]["budget"]
            train_runs = [r for r in s["runs"] if r["split"] == "train"]
            assert len(train_runs) == s["config"]["budget"]


# -- split authority ----------------------------------------------------------------------------
def test_no_heldout_row_runs_during_search(small_run):
    _, spy, _ = small_run
    held = set().union(*(_ids(s, SplitRole.TEST, SplitUse.REPORTING) for s in SMALL.suites))
    assert spy.seen["search"] and not held & set(spy.seen["search"])
    assert spy.seen["final"] and set(spy.seen["final"]) <= held


def test_recorder_refuses_a_heldout_row():
    suite = legacy_suite("A")
    rec = fs.Recorder(suite, {})
    test_task = suite.tasks_for(SplitRole.TEST, SplitUse.REPORTING)[0]
    genome = next(fs.synthetic_backend().checker.enumerate_admissible(suite.policy))
    with pytest.raises(SplitAccessError):
        rec.wrap(synthetic_evaluate)(genome, test_task, 0, 0)
    assert not rec.runs


def test_final_never_changes_search_results_or_champions(small_run, tmp_path):
    out, _, _ = small_run
    _run(tmp_path, final=False)
    assert _load(tmp_path, "search.json") == _load(out, "search.json")
    assert _load(tmp_path, "final.json") == {"schema": fs.SCHEMA, "ran": False, "runs": []}


def test_final_runs_each_frozen_champion_on_every_heldout_row(small_run):
    out, _, _ = small_run
    searches = _load(out, "search.json")["searches"]
    finals = _load(out, "final.json")["runs"]
    for s in searches:
        if s["status"] != "completed":
            continue
        mine = [
            r
            for r in finals
            if (r["suite"], r["optimizer"], r["seed"]) == (s["suite"], s["optimizer"], s["seed"])
        ]
        assert {r["genome_hash"] for r in mine} == {s["result"]["champion"]["genome_hash"]}
        held = _ids(s["suite"], SplitRole.TEST, SplitUse.REPORTING)
        assert sorted(r["task_id"] for r in mine) == sorted(t for t in held for _ in range(2))


def test_suite_without_admissible_workflow_is_skipped_not_substituted(small_run):
    out, spy, _ = small_run
    ajev = [s for s in _load(out, "search.json")["searches"] if s["suite"] == "Ajev"]
    assert ajev and all(s["status"] == "skipped" and "JEV" in s["reason"] for s in ajev)
    assert all(not s["runs"] and s["result"] is None for s in ajev)
    assert not any(t.startswith(("AJ-", "TAJ-")) for t in spy.seen["search"] + spy.seen["final"])


# -- determinism ----------------------------------------------------------------------------------
def test_same_seed_gives_the_same_candidate_ordering():
    suites = fs.suites_for(SMALL)
    checker = fs.synthetic_backend().checker
    first = fs.run_searches(SMALL, suites, synthetic_evaluate, checker)
    again = fs.run_searches(SMALL, suites, synthetic_evaluate, checker)
    assert [j.recorder.order for j in first if j.recorder] == [
        j.recorder.order for j in again if j.recorder
    ]
    assert [j.result for j in first] == [j.result for j in again]


def test_reports_are_deterministic(small_run):
    out, _, summary = small_run
    again = rep.write_reports(out, plots=False)
    assert json.dumps(again, sort_keys=True) == json.dumps(summary, sort_keys=True)


# -- per-task outputs -----------------------------------------------------------------------------
REQUIRED = (
    "suite", "task_id", "task_class", "shape", "split", "optimizer", "seed", "status",
    "search_trials", "selected_genome_hash", "trials", "passed", "pass_rate", "fitness_mean",
    "score_mean", "tokens_mean", "model_calls_mean", "tool_calls_mean", "wall_time_s_mean",
    "evidence_source", "evidence_provider", "failure_kinds",
)  # fmt: skip


def test_per_task_rows_cover_every_task_optimizer_and_seed(small_run):
    out, _, _ = small_run
    rows = _load(out, "per_task.json")["rows"]
    expected = set()
    for suite in SMALL.suites:
        for split, (role, use) in {
            "train": (SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK),
            "validation": (SplitRole.VALIDATION, SplitUse.SELECTION),
            "test": (SplitRole.TEST, SplitUse.REPORTING),
        }.items():
            for tid in _ids(suite, role, use):
                for opt in SMALL.optimizers:
                    for seed in SMALL.seeds:
                        expected.add((tid, split, opt, seed))
    got = [(r["task_id"], r["split"], r["optimizer"], r["seed"]) for r in rows]
    assert len(got) == len(set(got)) and set(got) == expected
    for r in rows:
        assert all(k in r for k in REQUIRED)
        if r["status"] == "completed":
            assert r["trials"] >= 1 and r["selected_genome_hash"] and r["evidence_source"]
            assert r["trials"] == len(r["runs"])
        else:
            assert r["trials"] == 0 and r["reason"]
    csv_lines = (out / "per_task.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == len(rows) + 1
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC RUN" in report and "Ajev not run" in report


def test_validation_rows_carry_their_task_champion(small_run):
    out, _, _ = small_run
    rows = _load(out, "per_task.json")["rows"]
    for r in rows:
        if r["status"] == "completed":
            assert (r["task_champion_genome_hash"] is not None) == (r["split"] == "validation")


def test_protocol_records_identity_and_synthetic_flag(small_run):
    out, _, _ = small_run
    meta = _load(out, "protocol.json")
    assert meta["synthetic"] is True and meta["schema"] == fs.SCHEMA
    assert meta["benchmark_hash"] and meta["heldout_benchmark_hash"]
    assert set(meta["budgets"]) == set(SMALL.suites)


# -- statistics -----------------------------------------------------------------------------------
def test_seed_level_statistics():
    d = rep.describe([1.0, 2.0, 3.0])
    assert d["mean"] == 2.0 and d["sd"] == 1.0
    assert d["ci95"] == pytest.approx([2.0 - 4.303 / 3**0.5, 2.0 + 4.303 / 3**0.5])
    assert rep.describe([0.5])["ci95"] is None
    assert rep.sign_flip_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert rep.sign_flip_p([0.0, 0.0]) == 1.0
    assert rep.sign_flip_p([1, -1]) == 1.0
    assert rep.sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert rep.sign_test_p(0, 0) is None
    assert rep.min_attainable_p(5) == pytest.approx(0.0625)


def _row(tid, opt, seed, pass_rate, score):
    return {
        "status": "completed", "trials": 2, "split": "test", "task_id": tid, "optimizer": opt,
        "seed": seed, "suite": "A", "task_class": "A", "shape": "x", "pass_rate": pass_rate,
        "score_mean": score,
    }  # fmt: skip


def test_a_per_task_regression_stays_visible_when_the_mean_improves():
    rows = []
    for seed in (0, 1, 2):
        rows += [_row("T1", "aco", seed, 1.0, 1.0), _row("T1", "rnd", seed, 0.0, 0.2)]
        rows += [_row("T2", "aco", seed, 1.0, 1.0), _row("T2", "rnd", seed, 0.0, 0.1)]
        rows += [_row("T3", "aco", seed, 0.0, 0.3), _row("T3", "rnd", seed, 1.0, 1.0)]
        rows += [_row("T4", "aco", seed, 0.5, 0.6), _row("T4", "rnd", seed, 0.5, 0.6)]
    cmp = rep.per_task_comparison(rows, "aco", "rnd")["test"]
    assert (cmp["wins"], cmp["losses"], cmp["ties"]) == (2, 1, 1)
    assert cmp["rescued"] == ["T1", "T2"] and cmp["broken"] == ["T3"]
    assert [t["task_id"] for t in cmp["regressions"]] == ["T3"]


def test_unknown_suite_is_refused():
    with pytest.raises(ValueError):
        fs.suites_for(replace(SMALL, suites=("Nope",)))


def test_champion_genome_round_trips(small_run):
    out, _, _ = small_run
    for s in _load(out, "search.json")["searches"]:
        g = fs.champion_genome(s)
        if s["status"] == "completed":
            assert isinstance(g, Genome)
            assert g.genome_hash == s["result"]["champion"]["genome_hash"]
        else:
            assert g is None


def test_final_records_a_backend_outage_as_unscored_not_failed(small_run):
    out, _, _ = small_run
    frozen = _load(out, "search.json")["searches"]

    def down(genome, task, trial, seed):
        raise RuntimeError("model unavailable after 3 workflow attempts")

    rows = fs.run_final(SMALL, fs.suites_for(SMALL), frozen, down)
    assert rows and all("error" in r and "verdict" not in r for r in rows)
