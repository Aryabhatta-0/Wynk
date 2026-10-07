"""Canonical experiment provenance + reproducible result artifacts (Issue #26).

dataset -> experiment -> strategy runs -> selected challenger -> promotion evidence, through ONE
artifact: the unchanged #23 result body plus an immutable ProvenanceRecord, linked by sha256,
finalized exactly once by the #24 job and cited by the #25 promotion decision.

Every model backend here is a TEST DOUBLE (the synthetic objective with injected, deterministic
usage); a "crash" is ``Crash``, a ``BaseException`` nothing in the engine catches.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from api.product import ProductAPI
from core.canonical import canonical_hash, canonical_json, sha256_hex
from core.models import AllowedModels, ModelCapability, ModelEntry
from core.provenance import PROVENANCE_SCHEMA, ProvenanceMismatch, ProvenanceRecord
from core.task_contract import workflow_grammar
from experiments.artifacts import (
    ExperimentArtifacts,
    ReproductionMismatch,
    TraceNotFound,
    compare,
    replay,
)
from experiments.jobs import (
    JOB_SCHEMA,
    ArtifactCorrupted,
    ArtifactNotFinalized,
    ArtifactUnavailable,
    ExperimentJobDefinition,
    JobBinding,
    load_canonical,
)
from experiments.optimization_experiment import (
    ARTIFACT_SCHEMA,
    RUNNER_VERSION,
    Strategy,
    load_compact,
    write_compact,
)
from experiments.promotion import (
    DECISION_SCHEMA,
    LEGACY_DECISION_SCHEMA,
    EvidenceMismatch,
    NotPromotable,
    compatibility_identity,
    decision_record,
)
from experiments.provenance import (
    CANONICAL_SCHEMA,
    CLOCK,
    MEASURED,
    SCIENTIFIC,
    ArtifactIntegrityError,
    CanonicalArtifact,
    check_provenance,
    field_class,
    leaves,
    load_frozen,
    recompute_declared,
    scientific,
    trace,
    verify_artifact,
)
from experiments.synthetic import SYNTHETIC_VERSION
from optimizers.aco_mmas import ACOConfig
from store.datasets import IntegrityViolation
from store.jobs import SQLiteJobStore
from tests.test_champion_promotion import EVERYTHING, Rigged, World
from tests.test_champion_promotion import definition as promotion_definition
from tests.test_durable_jobs import Backend, Clock, Crash, Process, Runtime
from tests.test_durable_jobs import definition as job_definition
from tests.test_optimization_experiment import LinearPricing

ROOT = Path(__file__).resolve().parent.parent
FROZEN = {
    "musique/protocol-v1": ROOT / "experiments/results/musique/protocol-v1",
    "musique/protocol-v2": ROOT / "experiments/results/musique/protocol-v2",
    "mmlu-pro/protocol-v1": ROOT / "experiments/results/mmlu-pro/protocol-v1",
}
# The explicit migration of each committed result is deterministic: same directory, same id.
FROZEN_ARTIFACT_IDS = {
    "musique/protocol-v1": "a66096725977c5019d54ab992a66148b5b6cde9548a3399ba8ce7a3172e1edbf",
    "musique/protocol-v2": "7df6b7d1cc95cba8bd206daaa623a7a605513bdad5aec3df704cd586977677c6",
    "mmlu-pro/protocol-v1": "0bdcdcce8b60c03fe135492ba366e2be5e403d1b8066e5c2a8e34b56d9f9239c",
}

# TEST DOUBLE registry entry pinning the synthetic model every test plan expects.
ENTRY = ModelEntry(
    name="scripted",
    provider="test-double",
    adapter="test-double",
    model_id="scripted",
    model_hash="synthetic",
    capabilities=(ModelCapability.TEXT_GENERATION,),
)


class RegistryRuntime(Runtime):
    """TEST DOUBLE runtime that, like ``registry_runtime``, binds the pinned registry entry."""

    def bind(self, definition: ExperimentJobDefinition) -> JobBinding:
        return dataclasses.replace(super().bind(definition), models=AllowedModels((ENTRY,)))


def completed(tmp_path, d=None, *, backend=None, pricing=None, store_cls=SQLiteJobStore):
    backend = backend or Backend()
    proc = Process(tmp_path / "jobs.sqlite3", RegistryRuntime(backend, pricing), Clock(), store_cls)
    job_id = proc.jobs.create(d or job_definition()).job_id
    assert proc.run() != "crashed"
    return proc, job_id, backend


def tamper(path: Path, statement: str, *args: Any) -> None:
    """Edit the database file behind the store's back (an attacker drops the triggers first)."""
    conn = sqlite3.connect(path)
    for trigger in (
        "experiment_body_write_once",
        "experiment_artifacts_no_update",
        "experiment_artifacts_no_delete",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute(statement, args)
    conn.commit()
    conn.close()


def rows(path: Path, statement: str, *args: Any) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(statement, args).fetchall()
    finally:
        conn.close()


def to_path(body: dict, parts: tuple) -> str:
    """The stable field path of a leaf (``strategy.<s>.seed.<n>``, candidates and curve points
    by evaluation number, per-seed summaries by seed)."""
    keyed = {
        "candidates": ("candidate", "evaluation"),
        "curve": ("curve", "evaluation"),
        "per_seed": ("per_seed", "seed"),
        "best_so_far_curve": ("best_so_far_curve", "evaluation"),
    }
    out: list[str] = []
    node: Any = body
    i = 0
    if parts[:1] == ("runs",):
        run = body["runs"][parts[1]]
        out, node, i = ["strategy", run["strategy"], "seed", str(run["seed"])], run, 2
    while i < len(parts):
        p = parts[i]
        if p in keyed and i + 1 < len(parts) and isinstance(node.get(p), list):
            alias, key = keyed[p]
            node = node[p][parts[i + 1]]
            out += [alias, str(node[key])]
            i += 2
            continue
        out.append(str(p))
        node = node[p]
        i += 1
    return ".".join(out)


def call(api: ProductAPI, path: str):
    res = api.handle("GET", f"/api/v1/{path}", {}, io.BytesIO(b""))
    return res.status, res.body


def forge(art: CanonicalArtifact, mutate) -> tuple[dict, str]:
    """A self-consistent forgery: the provenance is edited and EVERY hash recomputed, so only
    the cross-identity checks can catch it."""
    env = copy.deepcopy(art.envelope)
    mutate(env["provenance"])
    env["provenance_id"] = canonical_hash(env["provenance"])
    return env, canonical_hash(env)


# == 1. provenance: every authority, copied from the authority =====================================
def test_provenance_records_every_required_authority(tmp_path):
    d = job_definition()
    proc, job_id, _ = completed(tmp_path, d, pricing=LinearPricing())
    art = proc.jobs.canonical(job_id)
    p, body = art.provenance, art.body
    c, plan = d.contract, d.plan
    identity = json.loads(proc.store.get_job(job_id).identity_json)

    assert art.envelope["schema"] == CANONICAL_SCHEMA and p.schema_ == PROVENANCE_SCHEMA
    assert body["schema"] == ARTIFACT_SCHEMA  # the #23 body, not a new result format
    # dataset id / version / content hash
    assert (p.dataset.dataset_id, p.dataset.dataset_version, p.dataset.content_hash) == (
        c.dataset.dataset_id,
        c.dataset.dataset_version,
        c.dataset.content_hash,
    )
    assert p.dataset.identity_hash == c.dataset.identity_hash
    # split hash + config
    assert p.splits.splits_hash == d.splits.identity_hash
    assert (
        p.splits.method == d.splits.method.value and p.splits.dataset_hash == d.splits.dataset_hash
    )
    assert p.splits.rows == {"optimization": 3, "validation": 2, "test": 1}
    # TaskContract + contract hash
    assert p.contract.contract == c.model_dump(mode="json")
    assert p.contract.contract_hash == c.contract_hash == identity["contract_hash"]
    # grammar version + hash
    assert p.grammar.version == workflow_grammar(c).version
    assert p.grammar.stage_kinds == tuple(k.value for k in c.workflow.stages)
    assert p.grammar.grammar_hash == canonical_hash(
        {"version": p.grammar.version, "stage_kinds": list(p.grammar.stage_kinds)}
    )
    # evaluator kind / version / spec hash (+ the run version every EvaluatedRun reported)
    assert p.evaluator.kind == c.evaluation.evaluator.value
    assert p.evaluator.spec_version == c.evaluation.evaluator_version
    assert p.evaluator.spec_hash == c.evaluation.identity_hash
    assert p.evaluator.run_version == SYNTHETIC_VERSION == identity["evaluator_version"]
    # model registry entry + exact model hash + prompt version
    assert p.model.registry_entry == ENTRY.model_dump(mode="json")
    assert p.model.registry_entry_hash == ENTRY.identity_hash
    assert p.model.model_hash == plan.expected_model_hash == ENTRY.model_hash
    assert body["model_hashes"] == [p.model.model_hash]
    assert p.model.configuration == plan.model.model_dump(mode="json")
    assert p.model.prompt_template_version == plan.expected_prompt_version
    # experiment budget + pricing identity
    assert p.budget.budget == plan.budget.model_dump(mode="json")
    assert p.budget.budget_hash == plan.budget.identity_hash == identity["budget_hash"]
    assert p.budget.pricing == LinearPricing.identity == identity["pricing"]
    assert p.budget.protocol == plan.protocol()
    # optimizer name / version / config, strategy + seed, candidate + workflow hashes
    assert [(r.strategy, r.seed, r.run_id) for r in p.runs] == [
        (u["strategy"], u["seed"], u["run_id"]) for u in identity["units"]
    ]
    for r, run in zip(p.runs, body["runs"], strict=True):
        assert (r.optimizer, r.optimizer_version) == (run["optimizer"], run["optimizer_version"])
        assert r.candidate_order_hash == canonical_hash(run["evaluated_genome_hashes"])
        assert r.selected_genome_hash == run["champion"]["genome_hash"]
        assert r.candidates == len(run["candidates"])
    aco = p.run("aco", 0)
    assert aco.optimizer_config == dataclasses.asdict(ACOConfig(lcb_z=plan.lcb_z))
    assert p.run("fixed", 0).optimizer_config is None
    # runtime / code / protocol versions
    assert p.versions.runner_version == RUNNER_VERSION and p.versions.job_schema == JOB_SCHEMA
    assert list(p.versions.run_versions) == body["run_versions"]
    assert p.definition_hash == d.definition_hash
    assert (p.experiment_id, p.problem_id) == (identity["experiment_id"], identity["problem_id"])
    assert p.migration is None
    # deterministic, and nothing clock-derived inside it
    assert p.provenance_id == canonical_hash(art.envelope["provenance"])
    assert not any(field_class(parts) == CLOCK for parts, _ in leaves(p.dump()))


def test_provenance_is_deterministic_across_two_runs_of_one_experiment(tmp_path):
    a, ja, _ = completed(tmp_path / "a")
    b, jb, _ = completed(tmp_path / "b")
    pa, pb = a.jobs.canonical(ja), b.jobs.canonical(jb)
    assert pa.provenance_id == pb.provenance_id
    # same science, different clocks: the exact bodies may differ, their scientific hash does not
    ea, eb = pa.envelope["experiment"], pb.envelope["experiment"]
    assert ea["scientific_sha256"] == eb["scientific_sha256"]


FORGERIES = {
    "dataset version": lambda p: p["dataset"].update(dataset_version=99),
    "dataset content hash": lambda p: p["dataset"].update(content_hash="0" * 64),
    "split hash": lambda p: p["splits"].update(splits_hash="f" * 64),
    "split rows": lambda p: p["splits"]["rows"].update(test=7),
    "contract hash": lambda p: p["contract"].update(contract_hash="a" * 64),
    "contract body": lambda p: p["contract"]["contract"].update(instructions="something else"),
    "grammar version": lambda p: p["grammar"].update(version="grammar/9"),
    "evaluator spec version": lambda p: p["evaluator"].update(spec_version="token_f1/9"),
    "evaluator run version": lambda p: p["evaluator"].update(run_version="other/1"),
    "evaluator spec hash": lambda p: p["evaluator"].update(spec_hash="b" * 64),
    "model hash": lambda p: p["model"].update(model_hash="another-model"),
    "registry entry": lambda p: p["model"]["registry_entry"].update(model_id="another"),
    "registry entry hash": lambda p: p["model"].update(registry_entry_hash="c" * 64),
    "prompt version": lambda p: p["model"].update(prompt_template_version="prompts/9"),
    "optimizer version": lambda p: p["runs"][-1].update(optimizer_version="aco_mmas/9"),
    "optimizer config": lambda p: p["runs"][-1]["optimizer_config"].update(rho=0.9),
    "strategy seed": lambda p: p["runs"][0].update(seed=99),
    "run id": lambda p: p["runs"][0].update(run_id="d" * 64),
    "candidate order": lambda p: p["runs"][0].update(candidate_order_hash="e" * 64),
    "selected workflow": lambda p: p["runs"][0].update(selected_genome_hash="f" * 64),
    "budget": lambda p: p["budget"]["budget"].update(max_candidate_evaluations=99),
    "pricing": lambda p: p["budget"].update(pricing="free/1"),
    "runner version": lambda p: p["versions"].update(runner_version="optimization-runner/9"),
    "run versions": lambda p: p["versions"]["run_versions"][0].update(compiler_version="x"),
    "synthetic flag": lambda p: p.update(synthetic=False),
    "definition": lambda p: p.update(definition_hash="9" * 64),
}


@pytest.mark.parametrize("what", sorted(FORGERIES))
def test_identities_cannot_disagree_even_when_every_hash_is_recomputed(tmp_path, what):
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    identity = json.loads(proc.store.get_job(job_id).identity_json)
    env, artifact_id = forge(art, FORGERIES[what])
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(env, art.body, artifact_id=artifact_id, identity=identity)
    # and through the store: the forged envelope (consistent hashes) is refused on every read
    tamper(
        proc.store.path,
        "UPDATE experiment_artifacts SET envelope_json=?, artifact_id=?, provenance_id=? "
        "WHERE job_id=?",
        canonical_json(env),
        artifact_id,
        env["provenance_id"],
        job_id,
    )
    with pytest.raises(ArtifactCorrupted):
        proc.jobs.artifact(job_id)


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.update(run_id="d" * 64),  # not the hash of its run identity
        lambda r: r.update(optimizer_version="aco_mmas/9"),  # not what the strategy builds
    ],
)
def test_strategy_run_identities_must_rehash_even_when_body_and_provenance_agree(tmp_path, edit):
    """A forger who edits the body AND the provenance the same way: they agree with each other,
    but the run identity no longer re-hashes / no longer names the strategy's optimizer."""
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    body = copy.deepcopy(art.body)
    edit(body["runs"][-1])
    body["runs"][-1]["identity"].update(
        {k: body["runs"][-1][k] for k in ("optimizer_version",)}
        | {"run_id": body["runs"][-1]["run_id"]}
    )
    prov = art.provenance.dump()
    last = prov["runs"][-1]
    last.update(run_id=body["runs"][-1]["run_id"])
    last.update(optimizer_version=body["runs"][-1]["optimizer_version"])
    with pytest.raises(ProvenanceMismatch, match=r"run_id does not hash|optimizer identity"):
        check_provenance(ProvenanceRecord.model_validate(prov), body)


def test_a_real_experiment_must_pin_its_registry_entry(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    record = art.provenance.model_copy(
        update={
            "synthetic": False,
            "model": art.provenance.model.model_copy(
                update={"registry_entry": None, "registry_entry_hash": None}
            ),
        }
    )
    body = {**art.body, "synthetic": False}
    with pytest.raises(ProvenanceMismatch, match="registry entry"):
        check_provenance(record, body)


def test_a_job_identity_that_disagrees_with_the_provenance_fails_closed(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    job = proc.store.get_job(job_id)
    row = proc.store.artifact(job_id)
    identity = json.loads(job.identity_json)
    for key, value in [("model_hash", "x"), ("pricing", "x"), ("evaluator_version", "x")]:
        with pytest.raises(ArtifactIntegrityError, match=f"job identity {key}"):
            verify_artifact(row.envelope_json, job.artifact_json, identity={**identity, key: value})


# == 2. traceability ===============================================================================
def test_every_summary_metric_resolves_to_its_artifact_sources(tmp_path):
    proc, job_id, _ = completed(tmp_path, pricing=LinearPricing())
    art = proc.jobs.canonical(job_id)
    body = art.body
    derived = 0
    for parts, value in leaves(body["summary"]):
        if parts[-1] == "seeds" or (len(parts) > 2 and parts[2] == "seeds"):
            continue
        full = ("summary", *parts)
        if "highest_mean_champion_validation_score" in full:
            continue
        path = to_path(body, full)
        t = trace(art, path)
        assert t["value"] == value, path
        assert t["pointer"] == "/" + "/".join(str(p) for p in full)
        assert t["artifact_id"] == art.artifact_id and t["provenance_id"] == art.provenance_id
        d = t["derivation"]
        assert d is not None, f"{path} has no source mapping"
        assert d["consistent"], f"{path}: {value} != {d['recomputed']} from {d['sources']}"
        for src in d["sources"]:  # every source is itself a stored field
            trace(art, src)
        derived += 1
    assert derived > 200


def test_every_strategy_run_number_resolves_and_derived_ones_recompute(tmp_path):
    proc, job_id, _ = completed(tmp_path, pricing=LinearPricing())
    art = proc.jobs.canonical(job_id)
    body = art.body
    checked = derived = 0
    for parts, value in leaves(body):
        if parts[0] != "runs" or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        path = to_path(body, parts)
        t = trace(art, path)
        assert t["value"] == value and t["run"]["run_id"] == body["runs"][parts[1]]["run_id"]
        if t["derivation"] is not None:
            assert t["derivation"]["consistent"], path
            derived += 1
        checked += 1
    assert checked > 1000 and derived > 100


def test_the_issue_example_traces_to_artifact_field_and_provenance(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    api = ExperimentArtifacts(proc.jobs)
    t = api.trace(job_id, "strategy.aco.seed.1.usage.tokens")
    art = proc.jobs.canonical(job_id)
    i = next(i for i, r in enumerate(art.body["runs"]) if (r["strategy"], r["seed"]) == ("aco", 1))
    assert t.pointer == f"/runs/{i}/usage/tokens"
    assert t.value == art.body["runs"][i]["usage"]["tokens"]
    assert (t.artifact_id, t.provenance_id) == (art.artifact_id, art.provenance_id)
    assert t.experiment_sha256 == sha256_hex(proc.store.get_job(job_id).artifact_json)
    assert t.field_class == SCIENTIFIC and t.run["run_id"] == art.body["runs"][i]["run_id"]
    assert t.derivation["rule"] == "sum/1" and t.derivation["consistent"]


def test_a_measured_workflow_run_traces_to_its_stored_write_ahead_attempt(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    t = trace(art, "strategy.aco.seed.1.candidate.2.runs.3.score")
    attempt_id = t["evidence"]["attempt_id"]
    stored = rows(
        proc.store.path,
        "SELECT result_json, state FROM experiment_attempts WHERE attempt_id=?",
        attempt_id,
    )
    assert stored and stored[0][1] == "COMPLETED"
    run = json.loads(stored[0][0])
    assert run["execution"]["key"]["task_id"] == t["evidence"]["row_id"]
    assert run["execution"]["key"]["seed"] == t["evidence"]["run_seed"]
    assert trace(art, "strategy.aco.seed.1.candidate.2.runs.3.wall_time_s")["field_class"] == (
        MEASURED
    )


def test_learning_curves_are_traceable_point_by_point(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    for run in art.body["runs"]:
        base = f"strategy.{run['strategy']}.seed.{run['seed']}"
        for point in run["curve"]:
            e = point["evaluation"]
            for key in ("cumulative_tokens", "cumulative_model_calls", "score", "max_score_so_far"):
                t = trace(art, f"{base}.curve.{e}.{key}")
                assert t["value"] == point[key] and t["derivation"]["consistent"]
            assert (
                trace(art, f"{base}.curve.{e}.best_so_far_score")["value"]
                == (point["best_so_far_score"])
            )
            assert trace(art, f"{base}.curve.{e}.cumulative_e2e_wall_s")["field_class"] == CLOCK
            assert trace(art, f"{base}.curve.{e}.cumulative_execution_s")["field_class"] == MEASURED
    for strategy, s in art.body["summary"]["by_strategy"].items():
        for point in s["best_so_far_curve"]:
            path = f"summary.by_strategy.{strategy}.best_so_far_curve.{point['evaluation']}"
            t = trace(art, f"{path}.best_so_far_score.mean")
            assert t["derivation"]["consistent"] and t["derivation"]["sources"]


def test_selected_workflows_are_traceable(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    for run, r in zip(art.body["runs"], art.provenance.runs, strict=True):
        base = f"strategy.{run['strategy']}.seed.{run['seed']}"
        t = trace(art, f"{base}.champion.genome_hash")
        assert t["value"] == r.selected_genome_hash and t["derivation"]["consistent"]
        assert trace(art, f"{base}.champion.validation.score_mean")["derivation"]["consistent"]


def test_job_view_numbers_carry_their_artifact_source(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    view = proc.jobs.get(job_id)
    svc = ExperimentArtifacts(proc.jobs)
    for unit in view.units:
        assert unit.sources is not None
        for field_name, path in unit.sources.items():
            assert svc.trace(job_id, path).value == json.loads(
                json.dumps(getattr(unit, field_name))
            ), (field_name, path)


def test_an_unknown_field_path_names_nothing(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    svc = ExperimentArtifacts(proc.jobs)
    for path in (
        "strategy.aco.seed.7.usage.tokens",
        "strategy.aco.seed.1.usage.total_tokens",
        "strategy.aco.seed.1.candidate.99",
        "summary.by_strategy.nope",
        "strategy.aco",
        "promotion.challenger",
    ):
        with pytest.raises(TraceNotFound):
            svc.trace(job_id, path)


# == 3. integrity: tampering fails closed, nothing is repaired =====================================
def test_a_tampered_result_body_fails_closed(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    body = json.loads(proc.store.get_job(job_id).artifact_json)
    body["runs"][0]["usage"]["tokens"] += 1
    tamper(
        proc.store.path,
        "UPDATE experiment_jobs SET artifact_json=? WHERE job_id=?",
        canonical_json(body),
        job_id,
    )
    with pytest.raises(ArtifactCorrupted, match="sha256 link"):
        proc.jobs.artifact(job_id)
    api = ProductAPI(None, proc.jobs, None)  # type: ignore[arg-type]
    for endpoint in ("artifact", "provenance", "verify", "trace?path=experiment_id", "reproduce"):
        status, out = call(api, f"experiments/{job_id}/{endpoint}")
        assert (status, out["error"]["code"]) == (500, "artifact_integrity_error"), endpoint
    assert proc.store.get_job(job_id).artifact_json == canonical_json(body)  # never repaired


def test_a_tampered_envelope_fails_closed(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    env = json.loads(proc.store.artifact(job_id).envelope_json)
    env["provenance"]["model"]["prompt_template_version"] = "prompts/edited"
    tamper(
        proc.store.path,
        "UPDATE experiment_artifacts SET envelope_json=? WHERE job_id=?",
        canonical_json(env),
        job_id,
    )
    with pytest.raises(ArtifactCorrupted, match="artifact_id"):
        proc.jobs.canonical(job_id)


def test_a_non_canonical_body_fails_closed(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    text = proc.store.get_job(job_id).artifact_json
    tamper(
        proc.store.path,
        "UPDATE experiment_jobs SET artifact_json=? WHERE job_id=?",
        json.dumps(json.loads(text), indent=1),  # same content, other bytes
        job_id,
    )
    with pytest.raises(ArtifactCorrupted, match="canonical form"):
        proc.jobs.canonical(job_id)


def test_a_rehashed_forged_body_still_fails_on_its_evidence(tmp_path):
    """A forger who rewrites the body AND every hash (body sha256, scientific sha256, artifact
    id) passes the hash checks - and is caught by the evidence: the declared metrics no longer
    re-derive, and the stored attempts no longer replay to it."""
    proc, job_id, _ = completed(tmp_path)
    art = proc.jobs.canonical(job_id)
    body = copy.deepcopy(art.body)
    body["runs"][-1]["candidates"][0]["runs"][-1]["score"] = 1.0  # a better-looking run
    env = copy.deepcopy(art.envelope)
    text = canonical_json(body)
    env["experiment"].update(
        sha256=sha256_hex(text),
        bytes=len(text),
        scientific_sha256=canonical_hash(scientific(body)),
    )
    tamper(
        proc.store.path,
        "UPDATE experiment_jobs SET artifact_json=? WHERE job_id=?",
        text,
        job_id,
    )
    tamper(
        proc.store.path,
        "UPDATE experiment_artifacts SET envelope_json=?, artifact_id=?, experiment_sha256=? "
        "WHERE job_id=?",
        canonical_json(env),
        canonical_hash(env),
        env["experiment"]["sha256"],
        job_id,
    )
    proc.jobs.canonical(job_id)  # every hash agrees ...
    svc = ExperimentArtifacts(proc.jobs)
    with pytest.raises(ArtifactCorrupted, match="declared metrics"):
        svc.verify(job_id)  # ... but the declared metrics do not follow from the runs
    with pytest.raises(ReproductionMismatch):
        svc.reproduce(job_id)  # ... and the stored attempts do not replay to it


def test_stored_artifacts_are_immutable_in_the_database(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    path = proc.store.path
    conn = sqlite3.connect(path)
    try:
        for statement in (
            "UPDATE experiment_artifacts SET artifact_id='x'",
            "DELETE FROM experiment_artifacts",
            "UPDATE experiment_jobs SET artifact_json='{}'",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)
    finally:
        conn.close()


# == 4. durable finalization: exactly once, idempotent across crashes ==============================
def test_finalization_writes_body_and_envelope_in_one_transaction(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    job, row = proc.store.get_job(job_id), proc.store.artifact(job_id)
    assert row.experiment_sha256 == sha256_hex(job.artifact_json)
    assert rows(proc.store.path, "SELECT COUNT(*) FROM experiment_artifacts") == [(1,)]
    assert row.artifact_id == proc.jobs.artifact(job_id).artifact_id


def test_a_crash_before_finalization_finalizes_once_on_recovery(tmp_path):
    class DiesFinalizing(SQLiteJobStore):
        def put_artifact(self, *args, **kwargs):
            raise Crash

    backend, path = Backend(), tmp_path / "jobs.sqlite3"
    first = Process(path, RegistryRuntime(backend), Clock(), DiesFinalizing, "first")
    job_id = first.jobs.create(job_definition()).job_id
    assert first.run() == "crashed"
    assert (
        first.store.artifact(job_id) is None and first.store.get_job(job_id).artifact_json is None
    )
    calls = len(backend.calls)

    second = Process(path, RegistryRuntime(backend), Clock(), name="second")
    second.worker.recover()
    second.worker.recover()  # a second recovery (or worker) changes nothing
    assert len(backend.calls) == calls  # finalization never calls the model
    assert rows(path, "SELECT COUNT(*) FROM experiment_artifacts") == [(1,)]
    second.jobs.canonical(job_id)


def test_a_crash_after_finalization_never_creates_a_second_artifact(tmp_path):
    class DiesAfterCommit(SQLiteJobStore):
        def put_artifact(self, *args, **kwargs):
            super().put_artifact(*args, **kwargs)
            raise Crash

    backend, path = Backend(), tmp_path / "jobs.sqlite3"
    first = Process(path, RegistryRuntime(backend), Clock(), DiesAfterCommit, "first")
    job_id = first.jobs.create(job_definition()).job_id
    assert first.run() == "crashed"
    finalized = first.store.artifact(job_id)
    assert finalized is not None

    second = Process(path, RegistryRuntime(backend), Clock(), name="second")
    second.worker.recover()
    second.worker._assemble(job_id)  # retried finalization: a no-op
    assert rows(path, "SELECT artifact_id FROM experiment_artifacts") == [(finalized.artifact_id,)]
    assert second.jobs.canonical(job_id).artifact_id == finalized.artifact_id


def test_a_duplicate_or_conflicting_finalization_is_refused(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    job, row = proc.store.get_job(job_id), proc.store.artifact(job_id)
    from store.jobs import NewArtifact

    same = NewArtifact(
        row.artifact_id,
        row.schema,
        row.experiment_id,
        row.provenance_id,
        row.experiment_sha256,
        row.envelope_json,
    )
    assert proc.store.put_artifact(job_id, job.artifact_json, same) is False  # idempotent
    other_env = json.loads(row.envelope_json) | {"parent": {"kind": "job", "job_id": "other"}}
    other = dataclasses.replace(
        same, artifact_id=canonical_hash(other_env), envelope_json=canonical_json(other_env)
    )
    with pytest.raises(IntegrityViolation, match="refusing a second one"):
        proc.store.put_artifact(job_id, job.artifact_json, other)
    with pytest.raises(IntegrityViolation, match="does not link"):
        proc.store.put_artifact(job_id, job.artifact_json + " ", same)
    assert rows(proc.store.path, "SELECT artifact_id FROM experiment_artifacts") == [
        (row.artifact_id,)
    ]


def test_a_job_completed_before_provenance_is_explicitly_finalized_on_recovery(tmp_path):
    proc, job_id, backend = completed(tmp_path)
    before = proc.store.artifact(job_id)
    tamper(proc.store.path, "DELETE FROM experiment_artifacts WHERE job_id=?", job_id)  # pre-#26
    assert proc.store.jobs_awaiting_envelope() == [job_id]
    with pytest.raises(ArtifactNotFinalized):
        proc.jobs.artifact(job_id)
    api = ProductAPI(None, proc.jobs, None)  # type: ignore[arg-type]
    assert (
        call(api, f"experiments/{job_id}/artifact")[1]["error"]["code"] == "artifact_not_finalized"
    )
    calls = len(backend.calls)
    restarted = Process(proc.store.path, RegistryRuntime(backend), Clock(), name="restarted")
    restarted.worker.recover()
    assert len(backend.calls) == calls
    assert restarted.store.artifact(job_id).artifact_id == before.artifact_id  # deterministic
    restarted.jobs.canonical(job_id)


def test_an_unfinished_job_has_no_artifact(tmp_path):
    proc = Process(tmp_path / "jobs.sqlite3", RegistryRuntime(Backend()), Clock())
    job_id = proc.jobs.create(job_definition()).job_id
    with pytest.raises(ArtifactUnavailable):
        load_canonical(proc.store, job_id)
    api = ProductAPI(None, proc.jobs, None)  # type: ignore[arg-type]
    for endpoint in ("provenance", "verify", "trace?path=experiment_id", "reproduce"):
        status, out = call(api, f"experiments/{job_id}/{endpoint}")
        assert (status, out["error"]["code"]) == (409, "job_not_completed"), endpoint


# == 5. reproduction ===============================================================================
def test_replay_from_stored_evidence_reproduces_every_declared_metric(tmp_path):
    proc, job_id, backend = completed(tmp_path, pricing=LinearPricing())
    calls = len(backend.calls)
    report = ExperimentArtifacts(proc.jobs).reproduce(job_id)
    assert len(backend.calls) == calls  # no model call
    assert report.reproduced and report.model_calls == 0 and report.mode == "stored_evidence"
    assert report.fields[SCIENTIFIC]["equal"] and report.fields[MEASURED]["equal"]
    assert report.fields[CLOCK]["compared"] == 0
    assert report.declared_metric_checks > 100
    art = proc.jobs.canonical(job_id)
    assert [(u["strategy"], u["seed"], u["selected_genome_hash"]) for u in report.units] == [
        (r.strategy, r.seed, r.selected_genome_hash) for r in art.provenance.runs
    ]


def test_observational_timing_is_excluded_from_deterministic_equality(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    stored = proc.jobs.canonical(job_id).body
    clocked = copy.deepcopy(stored)
    for run in clocked["runs"]:
        run["timing"]["e2e_wall_s"] += 1000.0
        for point in run["curve"]:
            point["cumulative_e2e_wall_s"] += 1000.0
    result = compare(stored, clocked, evidence=True)  # clock telemetry: never compared
    assert result[CLOCK]["differing"] > 0 and result[SCIENTIFIC]["equal"]
    remeasured = copy.deepcopy(stored)
    remeasured["runs"][0]["candidates"][0]["runs"][0]["wall_time_s"] += 0.25
    compare(stored, remeasured, evidence=False)  # re-execution: measured time is not compared
    with pytest.raises(ReproductionMismatch, match="measured telemetry"):
        compare(stored, remeasured, evidence=True)  # but it must replay from stored evidence
    changed = copy.deepcopy(stored)
    changed["runs"][0]["usage"]["tokens"] += 1
    with pytest.raises(ReproductionMismatch, match="scientific"):
        compare(stored, changed, evidence=False)


def test_reexecution_on_a_deterministic_backend_reproduces_the_scientific_fields(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    fresh = Backend()  # same deterministic test double, a new "process"
    art, rep = replay(proc.store, job_id, RegistryRuntime(Backend()), reexecute=fresh)
    assert fresh.calls and rep.model_calls == len(fresh.calls)
    fields = compare(art.body, rep.body, evidence=False)
    assert fields[SCIENTIFIC]["equal"] and fields[MEASURED]["equal"] is None


def test_missing_or_edited_evidence_fails_reproduction_closed(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    svc = ExperimentArtifacts(proc.jobs)
    (attempt_id, result_json), *_ = rows(
        proc.store.path,
        "SELECT attempt_id, result_json FROM experiment_attempts WHERE strategy='aco' "
        "ORDER BY seq DESC",
    )
    run = json.loads(result_json)
    run["evaluation"]["fitness"] = 0.123456
    tamper(
        proc.store.path,
        "UPDATE experiment_attempts SET result_json=? WHERE attempt_id=?",
        json.dumps(run),
        attempt_id,
    )
    with pytest.raises(ReproductionMismatch):
        svc.reproduce(job_id)
    tamper(proc.store.path, "DELETE FROM experiment_attempts WHERE attempt_id=?", attempt_id)
    with pytest.raises(ReproductionMismatch, match="no stored result"):
        svc.reproduce(job_id)


def test_reproduction_needs_a_runtime_that_reproduces_the_job_identity(tmp_path):
    proc, job_id, _ = completed(tmp_path)
    # another registry entry for the same model hash: the provenance pinned a different one
    other = ENTRY.model_copy(update={"name": "renamed"})

    class Renamed(Runtime):
        def bind(self, definition):
            return dataclasses.replace(super().bind(definition), models=AllowedModels((other,)))

    with pytest.raises(ReproductionMismatch, match="registry entry"):
        replay(proc.store, job_id, Renamed(Backend()))
    api = ProductAPI(None, Process(proc.store.path, None, Clock()).jobs, None)  # type: ignore
    status, out = call(api, f"experiments/{job_id}/reproduce")
    assert (status, out["error"]["code"]) == (503, "experiment_backend_unavailable")


# == 6. promotion cites the artifact and its provenance ===========================================
def test_promotion_cites_the_artifact_and_shares_its_provenance(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(promotion_definition())
    view = w.promote(job)
    art = w.jobs.canonical(job)
    record, p = view.record, art.provenance
    assert record["schema"] == DECISION_SCHEMA
    assert record["artifact"] == art.ref() == view.artifact
    ids = record["identities"]
    assert ids["task_contract_hash"] == p.contract.contract_hash
    assert ids["splits_hash"] == p.splits.splits_hash
    assert ids["dataset_content_hash"] == p.dataset.content_hash
    assert ids["model_hash"] == p.model.model_hash
    assert ids["evaluator_version"] == p.evaluator.run_version
    assert ids["grammar_version"] == p.grammar.version
    assert ids["run_versions"] == list(p.versions.run_versions)
    assert record["experiment"]["experiment_id"] == p.experiment_id
    champion = w.promotions.current(view.lineage_id).record
    assert champion["provenance"]["artifact_id"] == art.artifact_id
    assert champion["provenance"]["provenance_id"] == art.provenance_id
    # the compatibility identity read from the provenance IS the recorded one (same compat_hash,
    # so lineages from before provenance artifacts stay comparable)
    d = ExperimentJobDefinition.model_validate_json(w.jobs.store.get_job(job).definition_json)
    assert compatibility_identity(d, art.body, p) == compatibility_identity(d, art.body)
    w.promotions.verify(view.promotion_id)
    verified = ExperimentArtifacts(w.jobs, w.store).verify(job)
    assert verified.promotion["cites_artifact"] and verified.promotion["promotion_id"] == (
        view.promotion_id
    )


def test_the_selected_challenger_traces_to_its_strategy_run(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(promotion_definition())
    view = w.promote(job)
    svc = ExperimentArtifacts(w.jobs, w.store)
    t = svc.trace(job, "promotion.challenger.genome_hash")
    assert t.document == "promotion" and t.value == view.record["challenger"]["genome_hash"]
    assert t.artifact_id == w.jobs.canonical(job).artifact_id
    (link,) = t.links
    assert link["consistent"]
    assert svc.trace(job, f"{link['path']}.genome_hash").value == t.value
    assert svc.provenance(job).promotion_id == view.promotion_id


def test_a_promotion_refuses_an_unfinalized_or_tampered_artifact(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(promotion_definition())
    path = w.jobs.store.path
    envelope = rows(path, "SELECT envelope_json FROM experiment_artifacts")[0][0]
    tamper(path, "DELETE FROM experiment_artifacts")
    with pytest.raises(NotPromotable, match="not finalized"):
        w.promote(job)
    assert w.backend.on_test() == []  # the test split was never opened
    tamper(path, "UPDATE experiment_jobs SET artifact_json=artifact_json || ' '")
    tamper(
        path,
        "INSERT INTO experiment_artifacts VALUES (?,?,?,?,?,?,?,?)",
        job,
        canonical_hash(json.loads(envelope)),
        CANONICAL_SCHEMA,
        json.loads(envelope)["experiment_id"],
        json.loads(envelope)["provenance_id"],
        json.loads(envelope)["experiment"]["sha256"],
        envelope,
        "t",
    )
    with pytest.raises(EvidenceMismatch):
        w.promote(job)
    assert w.backend.on_test() == []


def test_a_decision_from_before_provenance_keeps_its_schema(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(promotion_definition())
    view = w.promote(job)
    row = w.store.get(view.promotion_id)
    pinned = json.loads(row.selection_json)
    legacy = {k: v for k, v in pinned.items() if k != "artifact"}
    d = ExperimentJobDefinition.model_validate_json(w.jobs.store.get_job(job).definition_json)
    identity = json.loads(w.jobs.store.get_job(job).identity_json)
    old = decision_record(
        promotion=row,
        definition=d,
        identity=identity,
        pinned=legacy,
        heldout=view.record["heldout"],
    )
    assert old["schema"] == LEGACY_DECISION_SCHEMA and "artifact" not in old
    assert old["identities"]["decision_schema"] == LEGACY_DECISION_SCHEMA


def test_product_api_exposes_provenance_trace_verify_and_reproduce(tmp_path):
    w = World(tmp_path, Rigged(EVERYTHING))
    job = w.experiment(promotion_definition(strategies=(Strategy.FIXED, Strategy.ACO)))
    w.promote(job)
    api = ProductAPI(None, w.jobs, w.promotions)  # type: ignore[arg-type]
    art = w.jobs.canonical(job)

    status, out = call(api, f"experiments/{job}/artifact")
    assert status == 200 and out["artifact_id"] == art.artifact_id
    assert out["artifact_schema"] == CANONICAL_SCHEMA and out["artifact"] == art.body
    status, out = call(api, f"experiments/{job}/provenance")
    assert status == 200 and ProvenanceRecord.model_validate(out["provenance"]) == art.provenance
    assert out["promotion_id"] is not None
    status, out = call(api, f"experiments/{job}/trace?path=strategy.aco.seed.0.usage.tokens")
    assert (
        status == 200 and out["derivation"]["consistent"] and out["artifact_id"] == art.artifact_id
    )
    status, out = call(api, f"experiments/{job}/trace?path=promotion.decision")
    assert status == 200 and out["document"] == "promotion"
    status, out = call(api, f"experiments/{job}/verify")
    assert status == 200 and out["verified"] and out["promotion"]["cites_artifact"]
    status, out = call(api, f"experiments/{job}/reproduce")
    assert status == 200 and out["reproduced"] and out["model_calls"] == 0

    assert call(api, f"experiments/{job}/trace?path=strategy.x.seed.0")[1]["error"]["code"] == (
        "trace_path_not_found"
    )
    for bad in ("", "a..b", "a/b", "../x"):
        status, out = call(api, f"experiments/{job}/trace?path={bad}")
        assert (status, out["error"]["code"]) == (400, "invalid_request"), bad
    assert call(api, f"experiments/{job}/trace")[0] == 400
    assert call(api, f"experiments/{job}/verify?x=1")[1]["error"]["code"] == "unknown_parameter"
    assert call(api, "experiments/j-nope/verify")[1]["error"]["code"] == "job_not_found"


# == 7. frozen MuSiQue / MMLU-Pro results: explicitly migrated, still verified =====================
@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen_results_are_explicitly_migrated_and_verify(name):
    art = load_frozen(FROZEN[name])
    assert art.artifact_id == FROZEN_ARTIFACT_IDS[name]  # deterministic migration
    p = art.provenance
    assert p.migration is not None and p.migration.method == "legacy-identity/1"
    assert "contract.contract" in p.migration.missing and p.contract.contract is None
    _, body = load_compact(FROZEN[name])
    assert art.body == body  # the result itself is untouched
    assert p.contract.contract_hash == body["identity"]["problem"]["task_contract_hash"]
    assert p.model.model_hash == body["identity"]["problem"]["model_hash"]
    assert [(r.strategy, r.seed, r.run_id) for r in p.runs] == [
        (r["strategy"], r["seed"], r["run_id"]) for r in body["runs"]
    ]
    assert recompute_declared(art.body) > 1000
    t = trace(art, "summary.by_strategy.aco.metrics.champion_validation_score.mean")
    assert t["derivation"]["consistent"] and t["artifact_id"] == art.artifact_id
    assert (
        art.envelope["experiment"]["gzip_sha256"]
        == json.loads((FROZEN[name] / "summary.json").read_text(encoding="utf-8"))["artifact"][
            "gz_sha256"
        ]
    )


def test_a_tampered_frozen_result_fails_closed(tmp_path):
    _, body = load_compact(FROZEN["musique/protocol-v2"])
    write_compact(body, tmp_path)
    load_frozen(tmp_path)
    body["runs"][0]["champion"]["validation"]["score_mean"] = 0.99
    write_compact(body, tmp_path)  # digests rewritten: a "consistent" forgery
    with pytest.raises(ArtifactIntegrityError, match="declared metrics"):
        recompute_declared(load_frozen(tmp_path).body)
    gz = tmp_path / "experiment.json.gz"
    data = bytearray(gz.read_bytes())
    data[len(data) // 2] ^= 0xFF  # one corrupted byte in the compressed stream
    gz.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        load_frozen(tmp_path)
