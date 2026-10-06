"""The reusable external-benchmark adapter (Issue #29): MMLU-Pro + MuSiQue behind one contract.

Fixture rows follow each official schema but are invented, with marker strings in every gold
annotation so a leak into execution is detectable. Tests that need an official file run only
where it has been downloaded (``data/external``); it is never committed. Tests over committed
results run on whatever result directories exist. Model backends here are TEST DOUBLES.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import inspect
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.canonical import canonical_hash
from core.dataset import SplitPlan, SplitRole, seeded_splits
from core.task_contract import ContractError, TaskType
from experiments import musique as m
from experiments.contract_run import contract_suite
from experiments.external.adapter import (
    Benchmark,
    LeakageError,
    SourceMismatch,
    check_isolation,
    check_manifest,
    check_sanitized,
    dataset_bytes,
    frozen_manifest,
    load_protocol,
    load_source,
    lock_path,
    manifest_path,
    plan_from,
    prepare,
    prepare_rows,
    protocol_path,
    stratified_sample,
)
from experiments.external.mmlu_pro import LETTERS, MMLU_PRO, render_options
from experiments.external.musique import MUSIQUE
from experiments.external.run import (
    BLOCKED,
    benchmark_summary,
    champion_optimization_score,
    compact_dumps,
    failure_counts,
    real_runner,
    workflow_label,
)
from experiments.optimization_experiment import (
    ExperimentError,
    Strategy,
    load_compact,
    make_strategy,
    run_strategy,
    searchable_evaluator,
    summarize,
    write_compact,
)
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import SearchContext
from optimizers.fixed_baseline import (
    COT_BASELINE,
    FIXED_RULES,
    RETRIEVAL_BASELINE,
    ReasoningBaseline,
    context_aware_workflow,
    reasoning_workflow,
)
from runtime.gemma_client import GenerationResponse
from tests.test_musique import ROWS as MUSIQUE_ROWS
from tests.test_musique_protocols import RUNTIME_CHECKER

ROOT = Path(__file__).resolve().parent.parent
MMLU_OFFICIAL = ROOT / "data" / "external" / "mmlu_pro" / "test-00000-of-00001.parquet"
MUSIQUE_OFFICIAL = ROOT / "data" / "external" / "musique" / "musique_ans_v1.0_dev.jsonl"
MMLU_RESULTS = MMLU_PRO.results_dir / "protocol-v1"
MATRIX_DIR = ROOT / "experiments" / "results" / "benchmark-matrix"
HAVE_RESULTS = (MMLU_RESULTS / "summary.json").is_file()
MARKERS = ("RATIONALE-MARKER", "SRC-MARKER", "CATEGORY-MARKER")
CATEGORIES = ("biology", "law", "math")


def mmlu_row(qid: int, category: str = "math", answer: str = "C", n: int = 10) -> dict:
    return {
        "question_id": qid,
        "question": f"Which option is right for question {qid}?",
        "options": [f"option {LETTERS[i]} of {qid}" for i in range(n)],
        "answer": answer,
        "answer_index": LETTERS.index(answer),
        "cot_content": f"RATIONALE-MARKER the answer is ({answer})",
        "category": category,
        "src": f"SRC-MARKER-{category}",
    }


MMLU_ROWS = [
    mmlu_row(100 * (i + 1) + j, c, LETTERS[(i + j) % 10])
    for i, c in enumerate(CATEGORIES)
    for j in range(4)
]


def small_protocol(per: int = 2) -> dict:
    p = load_protocol(MMLU_PRO, "v1")
    return p | {"sampling": {"seed": 2026, "per_category": per}}


def kinds(genome) -> list[str]:
    return [s.kind for s in genome.stages]


# -- pinned source ------------------------------------------------------------------------------
def test_mmlu_pro_source_is_pinned_to_an_exact_revision_and_file_hash():
    s = MMLU_PRO.source
    assert re.fullmatch(r"[0-9a-f]{40}", s["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", s["file_sha256"])
    assert s["revision"] in s["file_url"] and s["file_url"].endswith(s["file"])
    assert s["rows"] == 12032
    assert frozen_manifest(MMLU_PRO)["source"] == s
    assert frozen_manifest(MMLU_PRO)["preprocessing"]["hash"] == MMLU_PRO.preprocessing_hash


@pytest.mark.parametrize("bench", [MMLU_PRO, MUSIQUE])
def test_other_bytes_than_the_pinned_official_file_are_refused(bench, tmp_path):
    fake = tmp_path / "fake"
    fake.write_bytes(b"not the official file\n")
    with pytest.raises(SourceMismatch, match="sha256"):
        load_source(bench, fake)


def test_the_official_mmlu_pro_file_rebuilds_the_frozen_subset_exactly():
    if not MMLU_OFFICIAL.is_file():
        pytest.skip("official MMLU-Pro file not downloaded (data/external/mmlu_pro)")
    pytest.importorskip("pyarrow")
    p = prepare(MMLU_PRO, MMLU_OFFICIAL, load_protocol(MMLU_PRO, "v1"))
    check_manifest(p.manifest, frozen_manifest(MMLU_PRO))
    assert p.contract.dataset.row_count == 210


def test_the_official_musique_file_rebuilds_its_frozen_manifest_through_the_generic_adapter():
    if not MUSIQUE_OFFICIAL.is_file():
        pytest.skip("official MuSiQue file not downloaded (data/external/musique)")
    for version in ("v1", "v2"):
        p = prepare(MUSIQUE, MUSIQUE_OFFICIAL, load_protocol(MUSIQUE, version))
        check_manifest(p.manifest, frozen_manifest(MUSIQUE))


# -- deterministic subset -----------------------------------------------------------------------
def test_sampling_is_equal_per_stratum_reproducible_and_reads_ids_and_strata_only():
    keyed = [(MMLU_PRO.row_id(r), MMLU_PRO.stratum(r)) for r in MMLU_ROWS]
    a = stratified_sample(keyed, rule="r/1", seed=7, per_stratum=2)
    assert a == stratified_sample(list(reversed(keyed)), rule="r/1", seed=7, per_stratum=2)
    assert list(a) == sorted(CATEGORIES) and all(len(v) == 2 for v in a.values())
    # answers, rationales and origins do not move the sample
    flipped = [
        r | {"answer": "J", "answer_index": 9, "cot_content": "x", "src": "y"} for r in MMLU_ROWS
    ]
    assert (
        prepare_rows(MMLU_PRO, flipped, small_protocol()).manifest["selected_ids"]
        == (prepare_rows(MMLU_PRO, MMLU_ROWS, small_protocol()).manifest["selected_ids"])
    )
    with pytest.raises(ValueError, match="need 5"):
        stratified_sample(keyed, rule="r/1", seed=7, per_stratum=5)


def test_the_committed_mmlu_pro_manifest_is_self_consistent_and_its_split_reproduces():
    frozen = frozen_manifest(MMLU_PRO)
    body = {k: v for k, v in frozen.items() if k != "manifest_hash"}
    assert frozen["manifest_hash"] == canonical_hash(body)
    selected = frozen["selected_ids"]
    assert len(selected) == 14 and all(len(ids) == 15 for ids in selected.values())
    ids = sorted(i for group in selected.values() for i in group)
    assert len(set(ids)) == 210 == frozen["dataset"]["rows"]
    splits = seeded_splits(
        frozen["dataset"]["identity_hash"], tuple(ids), SplitPlan(**frozen["splits"]["plan"])
    )
    assert splits.identity_hash == frozen["splits"]["identity_hash"]
    assert {s.role.value: list(s.row_ids) for s in splits.splits} == frozen["splits"]["rows"]
    assert {k: len(v) for k, v in frozen["splits"]["rows"].items()} == {
        "optimization": 84,
        "validation": 63,
        "test": 63,
    }
    # the per-category draw is exactly the hash order over the official ids
    assert frozen["sampling"] == {
        "rule": "wynk-mmlu-pro-sample/1",
        "seed": 2026,
        "per_category": 15,
        "strata": sorted(selected),
        "selection_inputs": "official ids + the dataset-native stratum only",
    }


# -- frozen protocol ----------------------------------------------------------------------------
def test_protocols_are_refused_unless_they_match_their_lock(tmp_path, monkeypatch):
    protocol = load_protocol(MMLU_PRO, "v1")
    lock = json.loads(lock_path(MMLU_PRO).read_text("utf-8"))
    assert lock["protocols"]["v1"]["canonical_hash"] == canonical_hash(protocol)
    frozen_dir = tmp_path / "frozen"
    shutil.copytree(MMLU_PRO.frozen_dir, frozen_dir)
    tampered = protocol | {"seeds": [0, 1, 2, 3]}
    (frozen_dir / "protocol-v1.json").write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(type(MMLU_PRO), "frozen_dir", frozen_dir)
    with pytest.raises(SourceMismatch, match="frozen hash"):
        load_protocol(MMLU_PRO, "v1")


def test_the_mmlu_pro_protocol_freezes_the_comparison():
    p = load_protocol(MMLU_PRO, "v1")
    assert p["fixed_baseline_rule"] == "fixed_reasoning/1" in FIXED_RULES
    assert p["strategies"] == ["fixed", "random", "aco"] and p["seeds"] == [0, 1, 2]
    assert p["contract"]["evaluation"]["evaluator"] == "classification_accuracy"
    assert p["contract"]["evaluation"]["config"]["labels"] == list(LETTERS)
    # same model, exact model hash, prompts, per-run limits, candidates and trials as MuSiQue v2
    v2 = load_protocol(MUSIQUE, "v2")
    for key in ("model", "expected_model_hash", "expected_prompt_version", "trials", "seeds"):
        assert p[key] == v2[key]
    assert p["contract"]["constraints"] == v2["contract"]["constraints"]
    assert (
        p["budget"]["max_candidate_evaluations"] == v2["budget"]["max_candidate_evaluations"] == 6
    )
    assert p["search_implementations"]["aco"].startswith("aco_mmas / aco_mmas/1")


def _git(*args: str) -> str | None:
    exe = shutil.which("git")
    if exe is None:
        return None
    out = subprocess.run([exe, *args], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() if out.returncode == 0 else None


def test_the_mmlu_pro_protocol_and_manifest_were_committed_before_its_results():
    if not HAVE_RESULTS:
        pytest.skip("MMLU-Pro results not committed yet")
    if _git("rev-parse", "--is-shallow-repository") != "false":
        pytest.skip("full git history unavailable (shallow clone)")
    results = _git(
        "log", "--diff-filter=A", "--format=%H", "--", str(MMLU_RESULTS / "summary.json")
    )
    if not results:
        pytest.skip("results not committed yet")
    results_sha = results.splitlines()[-1]
    for frozen in (protocol_path(MMLU_PRO, "v1"), manifest_path(MMLU_PRO), lock_path(MMLU_PRO)):
        added = _git("log", "--diff-filter=A", "--format=%H", "--", str(frozen))
        assert added, f"{frozen.name} is not committed"
        frozen_sha = added.splitlines()[-1]
        assert frozen_sha != results_sha
        assert _git("merge-base", "--is-ancestor", frozen_sha, results_sha) is not None
        # and never modified afterwards
        assert len(_git("log", "--format=%H", "--", str(frozen)).splitlines()) == 1


# -- leakage ------------------------------------------------------------------------------------
def test_sanitize_keeps_only_id_question_options_and_target():
    clean = MMLU_PRO.sanitize(mmlu_row(5, answer="D", n=4))
    assert clean == {
        "id": "5",
        "question": "Which option is right for question 5?",
        "options": "A. option A of 5\nB. option B of 5\nC. option C of 5\nD. option D of 5",
        "answer": "D",
    }
    check_sanitized(MMLU_PRO, clean)
    with pytest.raises(ValueError, match="not an option"):
        MMLU_PRO.sanitize(mmlu_row(5, answer="D", n=4) | {"answer": "J"})
    with pytest.raises(ValueError):
        render_options(["only one"])


@pytest.mark.parametrize("extra", [{"cot_content": "x"}, {"category": "math"}, {"hint": "x"}])
def test_a_sanitized_row_with_anything_extra_is_refused(extra):
    clean = MMLU_PRO.sanitize(mmlu_row(5))
    with pytest.raises(LeakageError):
        check_sanitized(MMLU_PRO, clean | extra)


def _suite(rows=MMLU_ROWS, protocol=None):
    p = prepare_rows(MMLU_PRO, rows, protocol or small_protocol())
    suite, refs = contract_suite(p.contract, p.splits, p.data)
    return p, suite, refs


def test_gold_annotations_and_targets_never_reach_execution_tasks():
    p, suite, refs = _suite()
    check_isolation(MMLU_PRO, suite)
    for task in suite.tasks:
        assert set(task.example.values) == {"question", "options"}
        blob = json.dumps(task.example.values) + task.question
        assert not any(marker in blob for marker in MARKERS)
        assert "answer_index" not in blob and "category" not in blob
        assert refs.expected(task.id) == {"answer": json.loads(_row(p, task.id))["answer"]}
    leaky = suite.tasks[0].model_copy(
        update={"example": suite.tasks[0].example.model_copy(update={"values": {"answer": "C"}})}
    )
    with pytest.raises(LeakageError):
        check_isolation(MMLU_PRO, suite.model_copy(update={"tasks": (leaky, *suite.tasks[1:])}))


def _row(p, rid: str) -> str:
    return next(line for line in p.data.decode().splitlines() if json.loads(line)["id"] == rid)


def test_test_rows_cannot_even_be_judged():
    p, suite, _ = _suite()
    _, evaluate, _ = searchable_evaluator(p.contract, p.splits, p.data, lambda *a: None)
    test_ids = set(p.splits.split(SplitRole.TEST).row_ids)
    assert test_ids
    task = next(t for t in suite.tasks if t.id in test_ids)
    with pytest.raises(ContractError, match="not one this evaluator was bound to"):
        evaluate(COT_BASELINE, task, 0, 0)


# -- the shared contract serves MuSiQue too -------------------------------------------------------
def test_the_generic_adapter_reproduces_the_musique_pipeline_exactly():
    protocol = load_protocol(MUSIQUE, "v2") | {"sampling": {"seed": 2026, "per_hop": 2}}
    generic = prepare_rows(MUSIQUE, MUSIQUE_ROWS, protocol)
    selected = m.sample_ids((r["id"] for r in MUSIQUE_ROWS), seed=2026, per_hop=2)
    ids = sorted(i for g in selected.values() for i in g)
    data = m.dataset_bytes(MUSIQUE_ROWS, ids)
    contract = m.musique_contract(data, protocol)
    splits = m.musique_splits(contract, ids, SplitPlan(**protocol["split_plan"]))
    assert generic.data == data
    assert generic.contract.model_dump(mode="json") == contract.model_dump(mode="json")
    assert generic.splits == splits
    assert generic.manifest == m.manifest(
        selected, protocol=protocol, contract=contract, splits=splits
    )
    # and its frozen manifest has exactly the generic manifest's layout
    assert set(frozen_manifest(MUSIQUE)) == set(generic.manifest)
    assert dataset_bytes(MUSIQUE, MUSIQUE_ROWS, ids) == data


def test_musique_gold_annotations_are_dropped_by_the_shared_gate():
    for row in MUSIQUE_ROWS:
        clean = MUSIQUE.sanitize(row)
        check_sanitized(MUSIQUE, clean)
        assert not any(mk in json.dumps(clean) for mk in ("ALIAS", "DECOMP", "INTERMEDIATE"))


@pytest.mark.parametrize("bench", [MMLU_PRO, MUSIQUE])
def test_every_benchmark_defines_only_the_benchmark_specific_hooks(bench):
    overridden = {
        name
        for name, value in vars(type(bench)).items()
        if callable(value) and not name.startswith("_")
    }
    assert overridden <= {"parse", "row_id", "stratum", "strata", "sanitize", "sampling_manifest"}
    assert isinstance(bench, Benchmark) and bench.frozen_dir.is_dir()
    assert set(bench.columns.targets).isdisjoint(bench.columns.execution)


# -- the strong fixed rule ------------------------------------------------------------------------
def test_the_reasoning_rule_is_decided_by_the_dataset_shape_only():
    p, suite, _ = _suite()
    mmlu = p.contract
    assert mmlu.task_type is TaskType.CLASSIFICATION and not mmlu.dataset.context_columns
    assert reasoning_workflow(mmlu, RUNTIME_CHECKER) == COT_BASELINE
    musique = prepare_rows(
        MUSIQUE,
        MUSIQUE_ROWS,
        load_protocol(MUSIQUE, "v2") | {"sampling": {"seed": 1, "per_hop": 1}},
    ).contract
    assert reasoning_workflow(musique, RUNTIME_CHECKER) == RETRIEVAL_BASELINE
    assert reasoning_workflow(musique, RUNTIME_CHECKER) == context_aware_workflow(
        musique, RUNTIME_CHECKER
    )
    # other rows, other answers, other content: same choice
    others = [mmlu_row(9000 + 10 * k + i, c, "J") for i in (1, 2) for k, c in enumerate(CATEGORIES)]
    other = _suite(others)[0].contract
    assert other.contract_hash != mmlu.contract_hash
    assert reasoning_workflow(other, RUNTIME_CHECKER) == COT_BASELINE
    assert list(inspect.signature(reasoning_workflow).parameters) == ["contract", "checker"]
    src = inspect.getsource(reasoning_workflow)
    assert not any(w in src for w in ("score", "evaluat", "target", "row", "label"))


def test_the_reasoning_baseline_proposes_once_and_never_learns():
    p, suite, _ = _suite()
    opt = make_strategy(Strategy.FIXED, plan_from(load_protocol(MMLU_PRO, "v1"), "m" * 64))
    assert isinstance(opt, ReasoningBaseline)
    ctx = SearchContext(contract=p.contract, checker=RUNTIME_CHECKER, seed=0, round=0)
    assert opt.propose(2, ctx) == [COT_BASELINE] and opt.propose(2, ctx) == []


def test_the_mmlu_pro_contract_admits_exactly_six_workflows():
    p, _, _ = _suite()
    space = list(RUNTIME_CHECKER.enumerate_admissible(p.contract))
    assert len(space) == load_protocol(MMLU_PRO, "v1")["search_space"]["admissible_workflows"]
    assert COT_BASELINE in space
    assert {tuple(kinds(g)) for g in space} == {("DIRECT",), ("DIRECT", "VERIFY")}


# -- ACO maths unchanged ------------------------------------------------------------------------
PINNED_SOURCES = {
    "optimizers/aco_mmas.py": "59ca711462dd7f837782ceaf8237cbafc6064efa95f07d10723711e1b92958b3",
    "optimizers/construct.py": "0e4c63069d153e856bbee7597187428a88239b312bf0b1f2449cb9d37f138c85",
    "optimizers/random_search.py": (
        "10fdad4b07f2ca0d85816f61b38d1a0ba0c0b2a2de85e9bc17349c1ae01a6378"
    ),
}


@pytest.mark.parametrize("path", sorted(PINNED_SOURCES))
def test_search_implementations_are_byte_for_byte_the_ones_musique_ran(path):
    data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == PINNED_SOURCES[path], (
        f"{path} changed: the benchmark matrix compares the unchanged implementations"
    )


def test_aco_config_and_version_are_unchanged():
    assert dataclasses.asdict(ACOConfig()) == {
        "alpha": 1.0,
        "rho": 0.30,
        "tau_max": 1.0,
        "tau_min": 0.05,
        "global_best_period": 5,
        "lcb_z": 1.0,
    }
    aco = make_strategy(Strategy.ACO, plan_from(load_protocol(MMLU_PRO, "v1"), "m" * 64))
    assert isinstance(aco, MMASACO) and aco.version == "aco_mmas/1"
    assert dataclasses.asdict(aco.config) == dataclasses.asdict(ACOConfig())


# -- end to end through the real runtime with a scripted model -----------------------------------
class LetterModel:
    """TEST DOUBLE: answers "B" (DIRECT answer) or "C" (DIRECT cot) and records every prompt."""

    model_hash = "letter-model/1"

    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        letter = "C" if request.prompt_template_id == "direct.cot" else "B"
        body = {"reasoning": "scripted", "answer": {"answer": letter}}
        return GenerationResponse(
            text=json.dumps(body),
            parsed=body,
            prompt_tokens=40,
            completion_tokens=8,
            model_hash=self.model_hash,
        )


def test_all_three_strategies_run_the_mmlu_pro_adapter_under_identical_budgets():
    pytest.importorskip("agent_framework")
    from runtime.runner import WorkflowRunner

    p, suite_, _ = _suite()
    plan = plan_from(load_protocol(MMLU_PRO, "v1"), p.manifest["manifest_hash"]).model_copy(
        update={"expected_model_hash": LetterModel.model_hash}
    )
    model = LetterModel()
    runner = WorkflowRunner(model=model, benchmark_hash="inline")
    executed: list[str] = []

    def run_workflow(genome, task, trial, seed):
        executed.append(task.id)
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    suite, evaluate, version = searchable_evaluator(p.contract, p.splits, p.data, run_workflow)
    check_isolation(MMLU_PRO, suite)
    records = [
        run_strategy(plan, s, 0, suite, evaluate, checker=runner.checker, evaluator_version=version)
        for s in plan.strategies
    ]
    test_rows = set(p.splits.split(SplitRole.TEST).row_ids)
    assert executed and not test_rows & set(executed)
    assert all(not any(mk in r.input_text for mk in MARKERS) for r in model.requests)
    assert {json.dumps(r["identity"]["protocol"], sort_keys=True) for r in records} == {
        json.dumps(plan.protocol(), sort_keys=True)
    }
    assert len({r["identity"]["problem_id"] for r in records}) == 1
    assert len({json.dumps(r["reservation_per_run"]) for r in records}) == 1
    assert len({json.dumps(r["budget"]) for r in records}) == 1
    used = {r["strategy"]: r["usage"]["candidate_evaluations"] for r in records}
    assert used == {"fixed": 1, "random": 6, "aco": 6}
    by = {r["strategy"]: r for r in records}
    assert by["random"]["distinct_genomes"] == 6  # the whole space: random search is exhaustive
    assert by["fixed"]["champion"]["genome"] == COT_BASELINE.canonical()
    for r in records:
        assert r["split_usage"]["test_runs"] == 0
        assert r["model_hashes"] == [LetterModel.model_hash]


# -- metric / resource aggregation ----------------------------------------------------------------
def _fake_record(strategy: str, seed: int, opt_scores, val_score: float) -> dict:
    champ = {"genome_hash": "g1", "genome": COT_BASELINE.canonical()}
    runs = [
        {"split": "optimization", "verdict": "pass", "score": s, "failure": None}
        for s in opt_scores
    ] + [{"split": "validation", "verdict": None, "score": None, "failure": "model_error"}]
    return {
        "strategy": strategy,
        "seed": seed,
        "champion": champ | {"validation": {"score_mean": val_score}},
        "candidates": [
            {"genome_hash": "g1", "runs": runs},
            {
                "genome_hash": "g2",
                "runs": [
                    {
                        "split": "optimization",
                        "verdict": "fail",
                        "score": 0.0,
                        "failure": "schema_invalid",
                    }
                ],
            },
        ],
    }


def test_champion_optimization_score_and_failures_are_counted_exactly():
    rec = _fake_record("random", 0, [1.0, 0.0, 1.0, 1.0], 0.5)
    assert champion_optimization_score(rec) == pytest.approx(0.75)  # model errors excluded
    assert failure_counts(rec) == {"model_error": 1, "schema_invalid": 1}
    assert champion_optimization_score(rec | {"champion": None}) is None
    assert workflow_label(COT_BASELINE.canonical()) == "DIRECT(cot)"


def test_compact_summary_is_small_valid_json_that_round_trips():
    obj = {
        "per_seed": [{"a": 1, "b": [1, 2]}, {"a": 2}],
        "aggregate": {"x": {"m": 1}},
        "curves": {"columns": ["e"], "runs": {"fixed/seed-0": [[1, 0.5]]}},
        "z": "é",
    }
    text = compact_dumps(obj)
    assert json.loads(text) == obj and text == compact_dumps(json.loads(text))
    assert len(text.splitlines()) == 14  # one line per key, map entry or row


# -- committed results ----------------------------------------------------------------------------
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="MMLU-Pro results not committed yet")


@needs_results
def test_the_committed_artifact_verifies_and_its_summary_regenerates_exactly():
    summary, artifact = load_compact(MMLU_RESULTS)  # both SHA-256 digests checked
    assert summary == json.loads(
        compact_dumps(benchmark_summary(MMLU_PRO, "v1", artifact, summary["artifact"]))
    )
    assert (MMLU_RESULTS / "summary.json").read_text("utf-8") == compact_dumps(summary)
    assert artifact["experiment_id"] == canonical_hash(artifact["identity"])
    plan = plan_from(load_protocol(MMLU_PRO, "v1"), frozen_manifest(MMLU_PRO)["manifest_hash"])
    assert artifact["identity"]["plan"] == plan.identity_dump()
    assert artifact["model_hashes"] == [plan.expected_model_hash]
    assert artifact["synthetic"] is False
    assert {(r["strategy"], r["seed"]) for r in artifact["runs"]} == {
        (s.value, seed) for s in plan.strategies for seed in plan.seeds
    }
    for r in artifact["runs"]:
        assert r["identity"]["protocol"] == plan.protocol()
        if r["strategy"] == "fixed":
            assert r["champion"]["genome"] == COT_BASELINE.canonical()
    # the compressed bytes are what write_compact produces: deterministic
    again = write_compact(artifact, MMLU_RESULTS.parent / ".regen-check")
    try:
        assert again["artifact"] == summary["artifact"]
    finally:
        shutil.rmtree(MMLU_RESULTS.parent / ".regen-check")
    assert summary["aggregate"] == {
        s: {"seeds": v["seeds"], "metrics": v["metrics"]}
        for s, v in summarize(artifact["runs"])["by_strategy"].items()
    }


@needs_results
def test_the_committed_runs_never_touched_the_test_split():
    _, artifact = load_compact(MMLU_RESULTS)
    test_rows = set(frozen_manifest(MMLU_PRO)["splits"]["rows"]["test"])
    assert len(test_rows) == 63 and artifact["test_runs"] == 0
    for run in artifact["runs"]:
        rows = {e["row_id"] for c in run["candidates"] for e in c["runs"]}
        assert rows and not rows & test_rows


@needs_results
def test_committed_results_stay_small():
    files = {p.name: p.stat().st_size for p in MMLU_RESULTS.iterdir() if p.is_file()}
    assert set(files) == {
        "summary.json",
        "experiment.json.gz",
        "REPORT.md",
        "NOTES.md",
        "learning_curves.png",
    }
    assert len((MMLU_RESULTS / "summary.json").read_text("utf-8").splitlines()) < 100


@needs_results
def test_a_tampered_artifact_is_refused(tmp_path):
    for name in ("summary.json", "experiment.json.gz"):
        shutil.copy(MMLU_RESULTS / name, tmp_path / name)
    (tmp_path / "experiment.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(ExperimentError, match="sha256"):
        load_compact(tmp_path)


@needs_results
def test_the_benchmark_matrix_regenerates_from_the_committed_artifacts():
    from experiments.external.matrix import ENTRIES, build_matrix, matrix_table

    committed = json.loads((MATRIX_DIR / "matrix.json").read_text("utf-8"))
    fresh = build_matrix(count_space=False)
    for row, frozen in zip(fresh["rows"], committed["rows"], strict=True):
        assert row == frozen | {"admissible_workflows": None}
    assert [r["benchmark"] for r in committed["rows"]] == [e.bench.key for e in ENTRIES]
    mmlu = next(r for r in committed["rows"] if r["benchmark"] == "mmlu-pro")
    assert mmlu["admissible_workflows"] == 6
    assert matrix_table(committed) in (MATRIX_DIR / "TABLES.md").read_text("utf-8")
    # MuSiQue is prior evidence: the matrix reads its committed protocol-v2 artifact unchanged
    musique = next(r for r in committed["rows"] if r["benchmark"] == "musique")
    v2 = json.loads((MUSIQUE.results_dir / "protocol-v2" / "summary.json").read_text("utf-8"))
    assert musique["artifact_gz_sha256"] == v2["artifact"]["gz_sha256"]
    assert musique["experiment_id"] == v2["experiment_id"]


def test_without_a_real_backend_the_run_is_blocked(monkeypatch, tmp_path):
    for k in ("GEMMA_BASE_URL", "GEMMA_MODEL", "GEMMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as exc:
        real_runner(tmp_path / "missing.env")
    assert exc.value.code == BLOCKED
