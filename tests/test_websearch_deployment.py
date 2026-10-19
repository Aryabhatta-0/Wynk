"""Selected live-search workflows deploy through the real runtime, with offline providers."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace

import pytest

from core.canonical import canonical_json
from core.constraints import ConstraintChecker
from core.dataset import SplitRole, SplitUse
from core.genome import Genome
from core.models import AllowedModels
from core.results import RunVersions, Verdict
from core.run_contract import ExampleInput, ExecutionTask
from core.stages import GatherSource, StageKind
from core.task_contract import WorkflowSpec
from experiments.deployment import (
    ModelBindingFailed,
    NotPublishable,
    bind_version,
    version_id_for,
    workflow_version_document,
)
from experiments.jobs import (
    DatasetRuntime,
    HeldoutBinding,
    JobBinding,
    WorkflowBackend,
    load_canonical,
)
from experiments.optimization_experiment import Strategy
from experiments.promotion import HELDOUT_TRIAL_OFFSET, PromotionFailed
from experiments.synthetic import SYNTHETIC_VERSION
from runtime.model_client import RegisteredModelClient
from runtime.sources import (
    DirectorySnapshotSource,
    WebSearchResponse,
    WebSearchResult,
    pin_websearch,
)
from store.deployments import VersionState
from tests.conftest import extract, gather, make_suite, synth
from tests.runtime_helpers import ScriptedJev
from tests.test_champion_inference import (
    ENTRY,
    INPUTS,
    LINEAGE,
    Env,
    ModelRuntime,
    PassageDouble,
    Stamped,
    definition,
    maf,
    real_versions,
)
from tests.test_champion_promotion import EVERYTHING, Rigged
from tests.test_contract_runtime import CAPITALS, DATA
from tests.test_runtime_websearch import WS_B, WS_ID, FakeWebSearch

WS = GatherSource.WEBSEARCH
STAGES = (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE)


class CapitalSearch(FakeWebSearch):
    """Local provider answering the countries in the fixture; responses may change live."""

    def search(self, request):
        self.requests.append(request)
        country, city = next(
            ((country, city) for _, country, city in CAPITALS if country in request.query),
            ("Peru", "Lima"),
        )
        return WebSearchResponse(
            results=(
                WebSearchResult(
                    rank=1,
                    url="https://example.org/capital",
                    title=country,
                    snippet=f"{self.city or city} is the capital of {country}.",
                ),
            )
        )

    city = None


def search_definition(sources=(WS,), strategies=(Strategy.FIXED,)):
    d = definition(strategies)
    contract = d.contract.model_copy(
        update={"workflow": WorkflowSpec(stages=STAGES, allowed_sources=sources)}
    )
    return d.model_copy(update={"contract": contract})


def world(root):
    """Actual execution for optimization, held-out promotion, and production inference."""
    from runtime.runner import WorkflowRunner

    env, provider = Env(root), CapitalSearch()

    def backend(d):
        runner = WorkflowRunner(
            model=RegisteredModelClient(ENTRY, env.model),
            benchmark_hash="inline",
            allowed_models=AllowedModels((ENTRY,)),
            websearch=provider,
        )
        return WorkflowBackend(
            run_workflow=lambda g, t, tr, s: runner.run_sync(g, t, trial=tr, seed=s),
            checker=runner.checker,
            models=runner.allowed_models,
            versions=runner.versions,
            evidence_provider=runner.evidence_authority(d.contract),
        )

    runtime = DatasetRuntime(data=lambda _: DATA, backend=backend, synthetic=True)
    env.runtime = env.jobs.runtime = env.worker.runtime = env.promotions.runtime = runtime
    return env, provider


def use_provider(env, provider):
    env.deployments.runtime = replace(env.inference_runtime, websearch=provider)


@pytest.fixture
def published(tmp_path):
    pytest.importorskip("agent_framework")
    env, provider = world(tmp_path)
    champion = env.champion(search_definition())
    version = env.publish(champion)
    provider.requests.clear()
    env.model.requests.clear()
    return env, provider, version


def test_selected_websearch_champion_pins_its_exact_provider(published):
    env, _, version = published
    doc = env.deployments.get(version).document
    assert doc["versions"]["evidence_source"] == WS_ID
    assert env.deployments.publish(doc["champion"]["champion_id"])[0].version_id == version


def test_websearch_champion_promotes_publishes_stages_and_invokes_live(published, monkeypatch):
    from runtime.runner import WorkflowRunner

    env, provider, version = published
    runs = []
    real_run = WorkflowRunner.run_sync

    def observe(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        runs.append(result)
        return result

    monkeypatch.setattr(WorkflowRunner, "run_sync", observe)
    use_provider(env, provider)
    env.deploy(version)
    assert provider.requests == [] and env.model.requests == []
    assert env.deployments.get(version).state is VersionState.PRODUCTION
    first = env.deployments.invoke_production(LINEAGE, INPUTS)
    (request,) = provider.requests
    assert request.query == INPUTS["question"] and request.max_results == 5
    provider.city = "Cusco"  # same provider identity, a new live result for identical inputs
    second = env.deployments.invoke_production(LINEAGE, INPUTS)
    assert first.output == {"answer": "Lima"}
    assert second.output == {"answer": "Cusco"}
    assert first.inference_id != second.inference_id
    assert len(provider.requests) == 2  # every invocation searches; no cache or replay
    assert env.model.requests
    for response, run in zip((first, second), runs, strict=True):
        stored = env.deployments.inference(response.inference_id)
        assert stored.status == "SUCCEEDED"
        assert stored.versions["evidence_source"] == WS_ID
        assert stored.versions == env.deployments.get(version).document["versions"]
        assert response.versions == stored.versions == run.key.versions.model_dump(mode="json")
        assert response.run_id == stored.run_id == run.key.run_id
        assert run.failure is None and run.budget_usage.tool_calls == 1


@pytest.mark.parametrize("identity", [None, WS_B])
def test_missing_or_different_inference_provider_fails_before_requests(published, identity):
    env, trained, version = published
    provider = FakeWebSearch(identity=identity) if identity else None
    if provider is not None:
        use_provider(env, provider)
    doc = env.deployments.get(version).document
    with pytest.raises(ModelBindingFailed, match="websearch") as exc:
        bind_version(doc, env.deployments.runtime)
    assert exc.value.details["reason"] == (
        "websearch_identity_mismatch" if identity else "websearch_unavailable"
    )
    with pytest.raises(ModelBindingFailed):
        env.deployments.stage(version, 0)
    assert env.revision() == 0
    assert env.deployments.get(version).state is VersionState.CREATED
    assert trained.requests == [] and env.model.requests == []
    assert env.clients == 0
    assert provider is None or provider.requests == []


@pytest.mark.parametrize("drift", ["identity", "max_results"])
def test_provider_drift_after_binding_refuses_search_without_charging(published, drift):
    env, provider, version = published
    use_provider(env, provider)
    bound = bind_version(env.deployments.get(version).document, env.deployments.runtime)
    if drift == "identity":
        provider._identity = WS_B
    else:
        provider._max_results += 1
    task = ExecutionTask(
        contract=bound.contract, example=ExampleInput(row_id="inference", values=INPUTS)
    )
    result = bound.runner.run_sync(bound.genome, task)
    assert result.failure is not None
    assert result.budget_usage.tool_calls == 0
    assert provider.requests == [] and env.model.requests == []
    assert result.key.versions.evidence_source == WS_ID


def publication_args(env, version):
    doc = env.deployments.get(version).document
    champ = env.promotions.store.champion(doc["champion"]["champion_id"])
    verified = env.promotions.verify(champ.promotion_id)
    job, art = load_canonical(env.jobs.store, verified.job_id)
    return dict(
        champion_id=champ.champion_id,
        champion_version=champ.version,
        lineage_id=champ.lineage_id,
        compat_hash=champ.compat_hash,
        record_json=champ.record_json,
        record=json.loads(champ.record_json),
        decision=verified.record,
        definition=search_definition(),
        artifact_ref=art.ref(),
        body=copy.deepcopy(art.body),
        provenance=art.provenance,
    )


@pytest.mark.parametrize(
    "damage", ["missing", "ambiguous", "unauthorized", "wrong_kind", "legacy_mixed"]
)
def test_publication_refuses_an_unprovable_selected_source(published, damage):
    env, _, version = published
    kw = publication_args(env, version)
    run = kw["body"]["runs"][0]
    selected = run["candidates"][0]
    if damage == "missing":
        run["candidates"] = []
    elif damage == "ambiguous":
        other = copy.deepcopy(selected)
        other["run_versions"]["evidence_source"] = WS_B
        run["run_versions"].append(other["run_versions"])
        run["candidates"].append(other)
    elif damage == "legacy_mixed":
        other_versions = selected.pop("run_versions").copy()
        other_versions.pop("evidence_source")
        run["run_versions"].append(other_versions)
    elif damage == "wrong_kind":
        selected["run_versions"].pop("evidence_source")
    else:
        prov = kw["provenance"]
        kw["provenance"] = prov.model_copy(
            update={"versions": prov.versions.model_copy(update={"evidence_provider": WS_B})}
        )
    message = "outside the verified provider authority" if damage == "unauthorized" else None
    with pytest.raises(NotPublishable, match=message):
        workflow_version_document(**kw)


def mixed_world(tmp_path, preferred, *, swap_heldout_source=False, include_direct=False):
    """Rigged scores select either kind; both sources occur in the canonical artifact."""
    backend = Stamped(lambda g, row: True)
    evaluate = backend._evaluate

    def scored(g, t, tr, seed):
        # The legacy synthetic objective only knows frozen source names; retain its metering.
        source = next((s.source for s in g.stages if s.kind == StageKind.GATHER), None)
        run = evaluate(g if source is None else Genome.of(gather(), *g.stages[1:]), t, tr, seed)
        identity = WS_ID if source is WS else None
        if swap_heldout_source and tr >= HELDOUT_TRIAL_OFFSET:
            identity = None if identity else WS_ID
        versions = RunVersions.model_validate(real_versions(t.contract)).model_copy(
            update={"evidence_source": identity}
        )
        key = run.execution.key.model_copy(
            update={"versions": versions, "genome_hash": g.genome_hash}
        )
        return run.model_copy(
            update={
                "execution": run.execution.model_copy(update={"key": key}),
                "evaluation": run.evaluation.model_copy(
                    update={
                        "fitness": float(source is preferred),
                        "verdict": Verdict.PASS if source is preferred else Verdict.FAIL,
                    }
                ),
            }
        )

    backend._evaluate = scored
    env = Env(tmp_path, backend)
    bind = env.runtime.bind
    env.runtime.bind = lambda d: replace(bind(d), evidence_provider=WS_ID)
    d = search_definition((GatherSource.FETCH, WS), (Strategy.RANDOM,))
    if include_direct:
        d = d.model_copy(
            update={
                "contract": d.contract.model_copy(
                    update={
                        "workflow": WorkflowSpec(
                            stages=(*STAGES, StageKind.DIRECT),
                            allowed_sources=(GatherSource.FETCH, WS),
                        )
                    }
                )
            }
        )
    d = d.model_copy(
        update={
            "plan": d.plan.model_copy(
                update={
                    "budget": d.plan.budget.model_copy(update={"max_candidate_evaluations": 24})
                }
            )
        }
    )
    return env, d


@maf
@pytest.mark.parametrize("preferred", [GatherSource.FETCH, WS])
def test_mixed_search_pins_only_the_selected_genomes_source(tmp_path, preferred):
    env, d = mixed_world(tmp_path, preferred)
    version = env.publish(env.champion(d))
    doc = env.deployments.get(version).document
    _, art = load_canonical(env.jobs.store, doc["provenance"]["job_id"])
    assert {v.get("evidence_source") for v in art.body["run_versions"]} == {None, WS_ID}
    assert doc["versions"].get("evidence_source") == (WS_ID if preferred is WS else None)
    if preferred is GatherSource.FETCH:
        env.deploy(version)  # inference runtime has no provider, despite opted-in grammar
        assert env.deployments.invoke_production(LINEAGE, INPUTS).output == {"answer": "Lima"}


@maf
@pytest.mark.parametrize("preferred", [GatherSource.FETCH, WS])
def test_promotion_refuses_another_candidates_source_on_heldout(tmp_path, preferred):
    env, d = mixed_world(tmp_path, preferred, swap_heldout_source=True)
    job_id = env.experiment(d)
    with pytest.raises(PromotionFailed, match="different.*versions"):
        env.promotions.promote(job_id)


@maf
@pytest.mark.parametrize("path", ["fetch", "direct"])
def test_legacy_non_websearch_document_and_content_address_are_unchanged(tmp_path, path):
    env = Env(tmp_path)
    d = definition((Strategy.FIXED,))
    if path == "direct":
        d = d.model_copy(
            update={
                "contract": d.contract.model_copy(
                    update={"workflow": WorkflowSpec(stages=(StageKind.DIRECT,))}
                )
            }
        )
        env.model = DirectCapitalModel()
    version = env.publish(env.champion(d))
    kw = publication_args(env, version)
    kw["definition"] = d
    original = env.deployments.get(version).document
    assert "evidence_source" not in original["versions"]
    for run in kw["body"]["runs"]:
        for candidate in run["candidates"]:
            candidate.pop("run_versions")  # historical artifact: one identity for the whole run
    legacy = workflow_version_document(**kw)
    assert canonical_json(legacy) == canonical_json(original)
    assert version_id_for(legacy) == version
    env.deploy(version)
    assert env.deployments.invoke_production(LINEAGE, INPUTS).status == "SUCCEEDED"


@pytest.mark.parametrize("drift", ["identity", "max_results"])
def test_production_invoke_refuses_provider_drift_before_binding_or_requests(published, drift):
    env, provider, version = published
    # The product configures a pinned provider once; invocation must not re-pin changed state.
    use_provider(env, pin_websearch(provider))
    env.deploy(version)
    env.clients = 0
    if drift == "identity":
        provider._identity = WS_B
    else:
        provider._max_results = 3
    with pytest.raises(ModelBindingFailed) as exc:
        env.deployments.invoke_production(LINEAGE, INPUTS)
    assert exc.value.details["reason"] == "websearch_provider_drifted"
    assert provider.requests == env.model.requests == []
    assert env.clients == 0 and env.inference_records() == 0


class DirectCapitalModel(PassageDouble):
    """TEST DOUBLE: the passage fake plus a scripted answer for the DIRECT path."""

    async def generate(self, request):
        response = await super().generate(request)
        if request.prompt_template_id.startswith("direct"):
            body = {"answer": {"answer": "Lima"}}
            return response.model_copy(update={"parsed": body, "text": json.dumps(body)})
        return response


@maf
def test_direct_winner_of_mixed_websearch_contract_deploys_without_provider(tmp_path):
    env, d = mixed_world(tmp_path, None, include_direct=True)
    env.model = DirectCapitalModel()
    version = env.publish(env.champion(d))
    doc = env.deployments.get(version).document
    _, art = load_canonical(env.jobs.store, doc["provenance"]["job_id"])
    assert {v.get("evidence_source") for v in art.body["run_versions"]} == {None, WS_ID}
    assert doc["workflow"]["genome"]["stages"][0]["kind"] == "DIRECT"
    assert "evidence_source" not in doc["versions"]
    assert env.deployments.runtime.websearch is None
    env.deploy(version)
    response = env.deployments.invoke_production(LINEAGE, INPUTS)
    assert response.status == "SUCCEEDED" and response.output == {"answer": "Lima"}
    assert "evidence_source" not in response.versions
    assert env.model.requests


@maf
@pytest.mark.parametrize("damage", ["missing", "ambiguous", "wrong", "unlisted"])
def test_publish_service_refuses_tampered_selected_candidate_versions(
    tmp_path, monkeypatch, damage
):
    from experiments import deployment

    env, d = mixed_world(tmp_path, WS)
    champion = env.champion(d)
    record = json.loads(env.promotions.store.champion(champion).record_json)
    job, art = load_canonical(env.jobs.store, record["provenance"]["job_id"])
    body = copy.deepcopy(art.body)
    run = body["runs"][0]
    assert {v.get("evidence_source") for v in run["run_versions"]} == {None, WS_ID}
    selected = [c for c in run["candidates"] if c["genome_hash"] == record["genome_hash"]]
    assert selected and all(c["run_versions"]["evidence_source"] == WS_ID for c in selected)
    if damage == "missing":
        for candidate in selected:
            candidate.pop("run_versions")
    elif damage == "ambiguous":
        other = copy.deepcopy(selected[0])
        other["run_versions"].pop("evidence_source")
        run["candidates"].append(other)
    else:
        for candidate in selected:
            if damage == "wrong":
                candidate["run_versions"].pop("evidence_source")  # a fetch candidate's identity
            else:
                candidate["run_versions"]["evidence_source"] = WS_B
    # Keep promotion verification real. Inject damage only at the deployment service's
    # verified-artifact boundary, so failure must come from its publication validation.
    monkeypatch.setattr(deployment, "load_canonical", lambda *a: (job, replace(art, body=body)))
    with pytest.raises(NotPublishable, match="(RunVersions|evidence source)"):
        env.deployments.publish(champion)
    assert env.deployments.list().versions == [] and env.model.requests == []


@maf
def test_a_promoted_jev_winner_remains_non_publishable(tmp_path):
    from runtime.runner import WorkflowRunner

    suite = make_suite(
        ("q1", "q2", "q3"),
        ("q4", "q5"),
        ("q6",),
        allowed_sources=(GatherSource.FETCH, GatherSource.JEV),
    )
    jev = ScriptedJev()
    runner = WorkflowRunner(
        model=RegisteredModelClient(ENTRY, PassageDouble()),
        benchmark_hash="inline",
        jev=jev,
        pages=DirectorySnapshotSource(tmp_path),
    )
    backend = Rigged(EVERYTHING)
    evaluate = backend._evaluate

    def scored(g, t, tr, seed):
        run = evaluate(g, t, tr, seed)
        is_jev = any(s.kind == StageKind.GATHER and s.source is GatherSource.JEV for s in g.stages)
        key = run.execution.key.model_copy(update={"versions": runner.versions(t, g)})
        return run.model_copy(
            update={
                "execution": run.execution.model_copy(update={"key": key}),
                "evaluation": run.evaluation.model_copy(
                    update={
                        "fitness": float(is_jev),
                        "verdict": Verdict.PASS if is_jev else Verdict.FAIL,
                    }
                ),
            }
        )

    backend._evaluate = scored

    class SnapshotRuntime(ModelRuntime):
        def bind(self, definition):
            return JobBinding(
                suite=suite,
                evaluate=self.backend,
                checker=ConstraintChecker(),
                evaluator_version=SYNTHETIC_VERSION,
                models=AllowedModels((ENTRY,)),
                evidence_provider=jev.identity,
            )

        def bind_heldout(self, definition):
            return HeldoutBinding(
                suite.tasks_for(SplitRole.TEST, SplitUse.PROMOTION_GATE),
                self.backend,
                SYNTHETIC_VERSION,
            )

    env = Env(tmp_path, backend)
    env.runtime = env.jobs.runtime = env.worker.runtime = env.promotions.runtime = SnapshotRuntime(
        backend
    )
    d = definition((Strategy.RANDOM,)).model_copy(
        update={"contract": suite.policy, "splits": suite.splits}
    )
    champion = env.champion(d)
    record = json.loads(env.promotions.store.champion(champion).record_json)
    genome = Genome.from_canonical(record["genome"])
    assert any(s.kind == StageKind.GATHER and s.source is GatherSource.JEV for s in genome.stages)
    with pytest.raises(NotPublishable, match="JEV workflows are not deployable"):
        env.deployments.publish(champion)
    assert env.deployments.list().versions == [] and jev.observations == env.model.requests == []


@maf
def test_api_configures_one_provider_for_both_jobs_and_inference(tmp_path, monkeypatch):
    from api import product
    from core.models import ModelRegistry
    from runtime.backends import brave_search

    registry_path = tmp_path / "models.json"
    registry_path.write_text(ModelRegistry(entries=(ENTRY,)).model_dump_json(), encoding="utf-8")
    provider = CapitalSearch()
    configured = []

    def configure(env):
        configured.append(True)
        return provider

    monkeypatch.setattr(brave_search, "websearch_from_env", configure)
    monkeypatch.setattr(product, "start_worker", lambda api: None)
    # Capture the runtimes at the application boundary, without starting server resources.
    monkeypatch.setattr(
        product, "build_api", lambda root, limit, jobs, inference, quotas: (jobs, inference)
    )
    monkeypatch.delenv("WYNK_BOOTSTRAP_API_KEY", raising=False)
    jobs, inference = product.api_from_args(
        argparse.Namespace(model_registry=registry_path, data_dir=tmp_path, max_upload_mb=1)
    )
    assert configured == [True]
    assert inference.websearch is provider
    env = Env(tmp_path / "world")
    monkeypatch.setattr(
        "runtime.backends.client_for", lambda entry, key: RegisteredModelClient(entry, env.model)
    )
    service = type("DatasetBytes", (), {"dataset_bytes": staticmethod(lambda _: DATA)})()
    backend = jobs(service).backend(search_definition())
    task = ExecutionTask(
        contract=search_definition().contract,
        example=ExampleInput(row_id="api", values=INPUTS),
    )
    genome = Genome.of(gather(WS), extract(), synth())
    result = backend.run_workflow(genome, task, 0, 0)
    assert result.failure is None and len(provider.requests) == 1
    assert result.key.versions.evidence_source == WS_ID
