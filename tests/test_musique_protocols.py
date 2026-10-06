"""MuSiQue protocol-v1 vs protocol-v2: frozen protocols, the fixed-baseline rules and the
committed (compressed) result artifacts.

Results-dependent tests run on whichever protocol result directories are committed; they never
need the official dataset file or a model backend.
"""

from __future__ import annotations

import gzip
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.canonical import canonical_hash
from core.constraints import ConstraintChecker, ConstraintConfig, ConstraintLimits
from core.stages import GatherSource, StageKind, VerifyMethod
from core.task_contract import ContractError, WorkflowSpec
from experiments.budget_ledger import ExperimentBudget
from experiments.musique import dataset_bytes, musique_contract
from experiments.musique_run import (
    LOCK,
    MANIFEST,
    PROTOCOLS,
    RESULTS,
    load_protocol,
    plan_from,
    protocol_hash,
)
from experiments.optimization_experiment import (
    CURVE_AXES,
    ExperimentError,
    ExperimentPlan,
    Strategy,
    compact_summary,
    load_compact,
    make_strategy,
    resource_curves,
    summarize,
    write_compact,
)
from optimizers.base import SearchContext
from optimizers.fixed_baseline import (
    DIRECT_BASELINE,
    RETRIEVAL_BASELINE,
    ContextAwareBaseline,
    FixedBaseline,
    context_aware_workflow,
)
from tests.contract_helpers import classification_contract
from tests.test_contract_runtime import DATA, FULL_CAPS, qa_contract
from tests.test_musique import ROWS

V1_CANONICAL_HASH = "00536ebc02dee55db5909bbd32f55bb01703f07dfb64792f293c8a332cc54422"
DOC_ONLY_KEYS = {
    "protocol_version",
    "note",
    "supersedes",
    "fixed_baseline_rule",
    "fixed_baseline_definition",
    "search_implementations",
}
RUNTIME_CHECKER = ConstraintChecker(  # what WorkflowRunner admits by default
    config=ConstraintConfig(
        unavailable_sources=(GatherSource.JEV,),
        unavailable_verifiers=(VerifyMethod.SELF_CONSISTENCY,),
    )
)
RESULT_DIRS = {v: RESULTS / f"protocol-{v}" for v in PROTOCOLS}
COMMITTED = [v for v, d in RESULT_DIRS.items() if (d / "summary.json").is_file()]


def lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def manifest_hash() -> str:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["manifest_hash"]


def kinds(genome) -> list[str]:
    return [s.kind for s in genome.stages]


# -- the context-aware fixed rule (fixed_context/1) --------------------------------------------
def test_a_dataset_with_context_gets_the_plain_retrieval_baseline():
    contract = qa_contract()  # the passage is a context column
    genome = context_aware_workflow(contract, ConstraintChecker())
    assert genome == RETRIEVAL_BASELINE
    assert kinds(genome) == ["GATHER", "EXTRACT", "SYNTHESIZE"]
    assert genome.stages[0].source == "fetch" and genome.stages[0].mode == "sequential"
    assert genome.stages[1].method == "direct" and genome.stages[2].method == "direct"
    assert not {"VERIFY", "REASON", "FILTER", "CONFIDENCE_GATE"} & set(kinds(genome))
    musique = musique_contract(dataset_bytes(ROWS, [r["id"] for r in ROWS]), load_protocol("v2"))
    assert context_aware_workflow(musique, RUNTIME_CHECKER) == RETRIEVAL_BASELINE


def test_a_dataset_without_context_gets_direct():
    contract = classification_contract(constraints=ConstraintLimits(**FULL_CAPS))
    assert not contract.dataset.context_columns
    assert context_aware_workflow(contract, ConstraintChecker()) == DIRECT_BASELINE


def test_the_fixed_rule_cannot_see_labels_rows_or_results():
    # its only inputs are the contract and the shared checker - no rows, targets or scores
    params = list(inspect.signature(context_aware_workflow).parameters)
    assert params == ["contract", "checker"]
    # different data and different labels, same shape -> the same baseline
    a = qa_contract()
    b = qa_contract(data=DATA.replace(b"Paris", b"Lyon "))
    assert a.dataset.content_hash != b.dataset.content_hash
    assert context_aware_workflow(a, ConstraintChecker()) == context_aware_workflow(
        b, ConstraintChecker()
    )
    # observing results changes nothing: it proposes its one workflow, then nothing
    opt = ContextAwareBaseline()
    ctx = SearchContext(contract=a, checker=ConstraintChecker(), seed=0)
    opt.observe([])
    assert opt.propose(3, ctx) == [RETRIEVAL_BASELINE] and opt.propose(3, ctx) == []


def test_the_fixed_rule_fails_closed_instead_of_falling_back():
    no_gather = qa_contract(workflow=WorkflowSpec(stages=(StageKind.DIRECT,)))
    assert no_gather.dataset.context_columns
    with pytest.raises(ContractError, match="fixed_context/1"):
        context_aware_workflow(no_gather, ConstraintChecker())


def test_the_plan_selects_the_pre_registered_fixed_rule():
    base = {
        "model": {"provider": "p", "model": "m"},
        "expected_model_hash": "h",
        "budget": ExperimentBudget(max_candidate_evaluations=1),
        "seeds": (0,),
    }
    v1_like = ExperimentPlan(**base)
    v2_like = ExperimentPlan(**base, fixed_rule="fixed_context/1", protocol_id="x")
    assert type(make_strategy(Strategy.FIXED, v1_like)) is FixedBaseline
    assert type(make_strategy(Strategy.FIXED, v2_like)) is ContextAwareBaseline
    # random / ACO are the same implementations under both rules
    for s in (Strategy.RANDOM, Strategy.ACO):
        assert type(make_strategy(s, v1_like)) is type(make_strategy(s, v2_like))
    with pytest.raises(ValueError, match="unknown fixed-baseline rule"):
        ExperimentPlan(**base, fixed_rule="best_on_validation/1")
    # unset optional fields never enter a v1 identity; set ones always enter a v2 identity
    assert "fixed_rule" not in v1_like.identity_dump() and "protocol_id" not in v1_like.protocol()
    assert v2_like.protocol()["protocol_id"] == "x"
    assert v2_like.identity_dump()["fixed_rule"] == "fixed_context/1"


# -- frozen protocols ---------------------------------------------------------------------------
def test_protocol_v1_is_unchanged():
    v1 = load_protocol("v1")
    assert protocol_hash(v1) == V1_CANONICAL_HASH == lock()["protocols"]["v1"]["canonical_hash"]
    plan = plan_from(v1, manifest_hash())
    assert plan.fixed_rule is None and plan.protocol_id is None  # v1's identity as it ran
    assert plan.fixed_baseline_rule == "fixed_shortest/1"


def test_protocol_v2_is_frozen_and_differs_from_v1_only_in_the_fixed_rule():
    v1, v2 = load_protocol("v1"), load_protocol("v2")
    assert protocol_hash(v2) == lock()["protocols"]["v2"]["canonical_hash"]
    changed = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
    assert changed <= DOC_ONLY_KEYS
    for key in (
        "sampling",
        "split_plan",
        "contract",
        "model",
        "expected_model_hash",
        "expected_prompt_version",
        "budget",
        "seeds",
        "strategies",
        "trials",
        "batch_size",
        "lcb_z",
        "execution",
    ):
        assert v1[key] == v2[key], key
    plan = plan_from(v2, manifest_hash())
    assert plan.fixed_rule == "fixed_context/1"
    assert plan.protocol_id == lock()["protocols"]["v2"]["canonical_hash"]
    assert plan.seeds == (0, 1, 2) and plan.dataset_manifest_hash == manifest_hash()


def test_v2_search_runs_get_new_identities_but_v1_identities_do_not_move():
    v1 = plan_from(load_protocol("v1"), manifest_hash())
    v2 = plan_from(load_protocol("v2"), manifest_hash())
    assert v1.protocol() != v2.protocol()  # every v2 run_id differs, random and ACO included
    assert v1.identity_dump() == v1.model_dump(mode="json", exclude={"fixed_rule", "protocol_id"})


# -- compressed artifacts -----------------------------------------------------------------------
def test_compact_artifacts_round_trip_and_detect_tampering(tmp_path):
    artifact = {
        "schema": "s",
        "synthetic": True,
        "experiment_id": "e",
        "identity": {"plan": {}},
        "provenance": {"manifest_hash": "m"},
        "fairness": {},
        "splits": {},
        "test_runs": 0,
        "model_hashes": [],
        "run_versions": [],
        "runs": [{"strategy": "fixed", "seed": 0, "candidates": [{"big": "x" * 1000}]}],
        "summary": {},
    }
    summary = write_compact(artifact, tmp_path)
    assert "candidates" not in summary["runs"][0]  # bulk stays only in the .gz
    loaded_summary, loaded = load_compact(tmp_path)
    assert loaded == artifact and loaded_summary == summary
    gz = tmp_path / "experiment.json.gz"
    assert write_compact(artifact, tmp_path / "again")["artifact"] == summary["artifact"]
    gz.write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(ExperimentError, match="sha256"):
        load_compact(tmp_path)


@pytest.mark.parametrize("version", COMMITTED)
def test_committed_artifact_verifies_and_matches_its_summary(version):
    summary, artifact = load_compact(RESULT_DIRS[version])  # both SHA-256 digests checked
    assert summary == compact_summary(artifact) | {"artifact": summary["artifact"]}
    assert artifact["experiment_id"] == canonical_hash(artifact["identity"])
    plan = plan_from(load_protocol(version), manifest_hash())
    assert artifact["identity"]["plan"] == plan.identity_dump()
    assert artifact["identity"]["problem"]["dataset_manifest_hash"] == manifest_hash()
    assert artifact["model_hashes"] == [plan.expected_model_hash]
    assert {(r["strategy"], r["seed"]) for r in artifact["runs"]} == {
        (s.value, seed) for s in plan.strategies for seed in plan.seeds
    }
    crosscheck = json.loads((RESULT_DIRS[version] / "official_crosscheck.json").read_text("utf-8"))
    for run in artifact["runs"]:
        assert set(resource_curves(run)) >= set(CURVE_AXES) - {"cumulative_cost"}
        if run["champion"] is not None:  # official answer F1 agrees with Wynk token F1
            got = crosscheck[f"{run['strategy']}/seed-{run['seed']}"]
            assert got["wynk_token_f1"] == pytest.approx(
                run["champion"]["validation"]["score_mean"]
            )
            if "answer_f1" in got:
                assert got["answer_f1"] == pytest.approx(got["wynk_token_f1"], abs=0.0015)


@pytest.mark.parametrize("version", COMMITTED)
def test_committed_runs_never_touched_the_test_split(version):
    _, artifact = load_compact(RESULT_DIRS[version])
    test_rows = set(json.loads(MANIFEST.read_text("utf-8"))["splits"]["rows"]["test"])
    assert len(test_rows) == 18 and artifact["test_runs"] == 0
    for run in artifact["runs"]:
        assert run["split_usage"]["test_runs"] == 0
        rows = {e["row_id"] for c in run["candidates"] for e in c["runs"]}
        assert rows and not rows & test_rows


def test_protocol_v1_result_is_kept_with_its_zero_f1_fixed_baseline():
    if "v1" not in COMMITTED:
        pytest.skip("protocol-v1 results not present")
    _, artifact = load_compact(RESULT_DIRS["v1"])
    assert artifact["fairness"]["fixed_baseline_rule"] == "fixed_shortest/1"
    for run in artifact["runs"]:
        if run["strategy"] == "fixed":
            assert [s["kind"] for s in run["champion"]["genome"]["stages"]] == ["DIRECT"]
            assert run["champion"]["validation"]["score_mean"] == 0.0


def test_protocol_v2_strategies_share_one_identity_except_the_strategy():
    if "v2" not in COMMITTED:
        pytest.skip("protocol-v2 results not committed yet")
    summary, artifact = load_compact(RESULT_DIRS["v2"])
    plan = plan_from(load_protocol("v2"), manifest_hash())
    assert (
        artifact["identity"]["plan"]["protocol_id"] == lock()["protocols"]["v2"]["canonical_hash"]
    )
    assert artifact["fairness"]["fixed_baseline_rule"] == "fixed_context/1"
    assert len({r["identity"]["problem_id"] for r in artifact["runs"]}) == 1
    assert {json.dumps(r["identity"]["protocol"], sort_keys=True) for r in artifact["runs"]} == {
        json.dumps(plan.protocol(), sort_keys=True)
    }
    for r in artifact["runs"]:
        assert set(r["identity"]) - {
            "strategy",
            "optimizer",
            "optimizer_version",
            "seed",
            "run_id",
        } == {"problem_id", "protocol", "budget_hash"}
        if r["strategy"] == "fixed":
            assert r["champion"]["genome"] == RETRIEVAL_BASELINE.canonical()
    assert summary["summary"] == summarize(artifact["runs"])
    # v1 results were not reused: no v2 run_id equals a v1 run_id
    if "v1" in COMMITTED:
        _, v1 = load_compact(RESULT_DIRS["v1"])
        assert not {r["run_id"] for r in v1["runs"]} & {r["run_id"] for r in artifact["runs"]}


def _git(*args: str) -> str | None:
    exe = shutil.which("git")
    if exe is None:
        return None
    out = subprocess.run([exe, *args], capture_output=True, text=True, cwd=LOCK.parent)
    return out.stdout.strip() if out.returncode == 0 else None


def test_protocol_v2_was_committed_before_its_results():
    if "v2" not in COMMITTED:
        pytest.skip("protocol-v2 results not committed yet")
    frozen = _git("log", "--diff-filter=A", "--format=%H", "--", "protocol-v2.json")
    results = _git(
        "log", "--diff-filter=A", "--format=%H", "--", str(RESULT_DIRS["v2"] / "summary.json")
    )
    if not frozen or not results:
        pytest.skip("git history unavailable (e.g. a shallow clone)")
    frozen_sha, results_sha = frozen.splitlines()[-1], results.splitlines()[-1]
    assert frozen_sha != results_sha
    assert _git("merge-base", "--is-ancestor", frozen_sha, results_sha) is not None


@pytest.mark.parametrize("version", sorted(PROTOCOLS))
def test_protocol_files_are_plain_json_with_a_lock_entry(version):
    assert PROTOCOLS[version].name == lock()["protocols"][version]["file"]
    assert isinstance(json.loads(Path(PROTOCOLS[version]).read_text("utf-8")), dict)
