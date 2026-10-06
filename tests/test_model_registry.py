"""Provider-neutral model registry + ModelClient (#27).

registry entry -> RegisteredModelClient -> exact model_hash -> RunVersions / experiment identity.
Every model here is a TEST DOUBLE; every price is invented. No external model is contacted.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from core.experiment import ModelConfiguration
from core.models import (
    AllowedModels,
    DisallowedModelError,
    ModelCapability,
    ModelEntry,
    ModelIdentityError,
    ModelPricing,
    ModelRegistry,
    ModelRequirements,
    RegistryPricing,
    RetryPolicy,
    TimeoutPolicy,
    UnknownModelError,
    UnsupportedCapabilityError,
    registry_pricing,
)
from experiments.budget_ledger import (
    BudgetLedger,
    ExperimentBudget,
    ExperimentBudgetError,
    MeasuredUsage,
    PricingError,
    check_budget,
    run_reservation,
)
from experiments.optimization_experiment import (
    ExperimentPlan,
    bind_model,
    optimize_uploaded_dataset,
    run_optimization_experiment,
)
from runtime.backends import client_for
from runtime.backends.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    client_from_env,
    openai_compatible_model_hash,
    registered_client,
)
from runtime.model_client import (
    GenerationRequest,
    GenerationResponse,
    ModelContractViolation,
    ModelError,
    ModelRole,
    RegisteredModelClient,
)
from runtime.runner import WorkflowRunner

ROOT = Path(__file__).resolve().parent.parent
TEXT, STRUCTURED = ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT

# invented price sheet (not any provider's real prices)
PRICE = ModelPricing(
    version="test-sheet/1", currency="TEST", prompt_per_million=2.0, completion_per_million=6.0
)


def entry(name: str = "model-a", model_hash: str = "hash-a", **kw: Any) -> ModelEntry:
    base: dict[str, Any] = {
        "name": name,
        "provider": "test-provider",
        "adapter": "test-double",
        "model_id": name,
        "model_hash": model_hash,
        "context_window": 32_768,
        "capabilities": (TEXT, STRUCTURED),
    }
    return ModelEntry(**{**base, **kw})


def registry(*entries: ModelEntry) -> ModelRegistry:
    return ModelRegistry(entries=entries or (entry(), entry("model-b", "hash-b")))


def request(**kw: Any) -> GenerationRequest:
    base: dict[str, Any] = {
        "role": ModelRole.EXTRACT,
        "prompt_template_id": "extract/direct",
        "prompt_template_version": "v1",
        "input_text": "hello",
        "output_schema": {"type": "object"},
        "seed": 7,
        "max_tokens": 64,
    }
    return GenerationRequest(**{**base, **kw})


class Recording:
    """TEST DOUBLE backend: records every request; replies with fixed usage."""

    def __init__(self, model_hash: str = "hash-a", reply_hash: str | None = None) -> None:
        self.model_hash = model_hash
        self.reply_hash = reply_hash or model_hash
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        self.requests.append(req)
        return GenerationResponse(
            text="{}",
            parsed={},
            prompt_tokens=70,
            completion_tokens=30,
            model_hash=self.reply_hash,
            attempts=2,
            backoff_time_s=0.25,
        )


# == registry ===================================================================================
def test_registry_looks_models_up_by_name_or_exact_hash():
    reg = registry()
    a = reg.get("model-a")
    assert reg.by_hash("hash-a") is a
    assert reg.resolve("model-a") is a and reg.resolve("hash-a") is a
    assert reg.get("model-b").model_hash == "hash-b"


@pytest.mark.parametrize(
    "duplicate",
    [
        entry("model-a", "hash-other", model_id="x"),  # same name
        entry("model-c", "hash-a", model_id="y"),  # same model_hash
        entry("model-c", "hash-c", model_id="model-a"),  # same model, different claimed hash
    ],
    ids=["name", "model_hash", "natural-key"],
)
def test_duplicate_identities_are_rejected(duplicate):
    with pytest.raises(ValidationError, match="duplicate"):
        ModelRegistry(entries=(entry(), duplicate))


def test_unknown_models_fail_closed():
    reg = registry()
    for lookup in (reg.get, reg.by_hash, reg.resolve):
        with pytest.raises(UnknownModelError):
            lookup("nobody")
    with pytest.raises(UnknownModelError):
        reg.allow(["hash-a", "nobody"])


def test_disabled_models_cannot_be_allowed_or_bound():
    off = entry(enabled=False)
    with pytest.raises(DisallowedModelError):
        registry(off).allow(["model-a"])
    with pytest.raises(DisallowedModelError):
        RegisteredModelClient(off, Recording())


def test_registry_is_immutable_and_round_trips_through_json(tmp_path):
    reg = registry(entry(pricing=PRICE, revision="rev-1", endpoint="http://h/v1"))
    path = tmp_path / "models.json"
    path.write_text(reg.model_dump_json(), encoding="utf-8")
    loaded = ModelRegistry.load(path)
    assert loaded == reg and loaded.identity_hash == reg.identity_hash
    with pytest.raises(ValidationError):
        loaded.entries[0].model_hash = "tampered"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate({"entries": [], "surprise": 1})
    with pytest.raises(ValidationError):
        entry(revision="  ")  # unpinned is null, never a blank string


def test_capabilities_are_declared_never_inferred_from_names():
    named = entry("json-structured-128k-model", capabilities=(), context_window=None)
    assert not named.supports(TEXT) and not named.structured_output
    assert named.context_window is None  # unknown, not guessed
    adhoc = OpenAICompatibleConfig(base_url="http://h/v1", model="any-32k", structured=False)
    e = adhoc.entry()
    assert e.capabilities == (TEXT,) and e.context_window is None


# == allowed models =============================================================================
def test_allowed_set_admits_only_declared_models():
    allowed = registry().allow(["model-a"])
    assert allowed.admit("hash-a").name == "model-a"
    assert "hash-a" in allowed and "hash-b" not in allowed
    with pytest.raises(DisallowedModelError):
        allowed.admit("hash-b")
    with pytest.raises(DisallowedModelError):
        AllowedModels(())
    with pytest.raises(ModelIdentityError):
        AllowedModels((entry(), entry()))


def test_runner_refuses_a_client_outside_the_allowed_set_before_any_call():
    outsider = Recording("hash-b")
    with pytest.raises(DisallowedModelError):
        WorkflowRunner(
            model=outsider, benchmark_hash="b", allowed_models=registry().allow(["model-a"])
        )
    assert outsider.requests == []


def test_runner_rechecks_the_allowed_set_on_every_run_before_any_call():
    from runtime.mvp_genomes import GENOME_A
    from tests.conftest import make_task

    model = Recording("hash-a")
    runner = WorkflowRunner(
        model=model, benchmark_hash="b", allowed_models=registry().allow(["model-a"])
    )
    runner.allowed_models = registry().allow(["model-b"])  # the declaration changed
    with pytest.raises(DisallowedModelError):
        runner.run_sync(GENOME_A, make_task())
    assert model.requests == []


def test_runner_rejects_a_client_bound_to_a_different_entry_for_the_same_hash():
    client = RegisteredModelClient(entry("model-a"), Recording("hash-a"))
    other = AllowedModels((entry("model-a-copy", model_id="other"),))  # same hash, other entry
    with pytest.raises(ModelIdentityError):
        WorkflowRunner(model=client, benchmark_hash="b", allowed_models=other)


def test_a_single_allowed_model_runs_exactly_as_without_a_declaration(tmp_path):
    pytest.importorskip("agent_framework")
    from runtime.mvp_genomes import GENOME_A
    from runtime.sources import DirectoryApiSource, DirectorySnapshotSource
    from tests.conftest import make_task
    from tests.runtime_helpers import ScriptedModel, write_snapshot

    write_snapshot(tmp_path)
    task = make_task()
    pinned = entry("scripted", ScriptedModel.model_hash)

    def run(**kw):
        runner = WorkflowRunner(
            benchmark_hash="b",
            pages=DirectorySnapshotSource(tmp_path),
            api=DirectoryApiSource(tmp_path),
            **kw,
        )
        result = runner.run_sync(GENOME_A, task, seed=3)
        return _without_wall_time(result.model_dump(mode="json")), result

    plain, _ = run(model=ScriptedModel())
    bound, result = run(
        model=RegisteredModelClient(pinned, ScriptedModel()),
        allowed_models=AllowedModels((pinned,)),
    )
    assert bound == plain
    assert result.failure is None
    assert result.key.versions.model_hash == pinned.model_hash


def _without_wall_time(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _without_wall_time(v) for k, v in obj.items() if k != "wall_time_s"}
    if isinstance(obj, list):
        return [_without_wall_time(v) for v in obj]
    return obj


@pytest.mark.parametrize(
    ("declared", "allowed_requirements"),
    [
        (entry(capabilities=(STRUCTURED,)), ModelRequirements(capabilities=())),
        (entry(context_window=512), ModelRequirements()),
    ],
    ids=["no-text-generation", "context-window-below-stage-output"],
)
def test_runner_rejects_workflows_the_model_provably_cannot_serve(declared, allowed_requirements):
    from runtime.mvp_genomes import GENOME_A
    from tests.conftest import make_task

    backend = Recording(declared.model_hash)
    runner = WorkflowRunner(
        model=backend,
        benchmark_hash="b",
        allowed_models=AllowedModels((declared,), allowed_requirements),
    )
    with pytest.raises(UnsupportedCapabilityError):
        runner.run_sync(GENOME_A, make_task())
    assert backend.requests == []


# == exact model_hash binding ===================================================================
def test_registered_client_requires_the_backend_to_report_the_pinned_hash():
    with pytest.raises(ModelIdentityError, match="pins 'hash-a'"):
        RegisteredModelClient(entry(), Recording("hash-b"))
    bound = RegisteredModelClient(entry(), Recording("hash-a"))
    assert bound.model_hash == "hash-a" and bound.entry.name == "model-a"


def test_a_response_from_another_model_fails_closed():
    backend = Recording("hash-a", reply_hash="hash-b")
    with pytest.raises(ModelContractViolation, match="hash-b") as exc:
        asyncio.run(RegisteredModelClient(entry(), backend).generate(request()))
    assert exc.value.attempts == 2  # the calls that did happen are still accounted


def _plan(model_hash: str = "hash-a", model: ModelConfiguration | None = None) -> ExperimentPlan:
    return ExperimentPlan(
        model=model or ModelConfiguration(provider="test-provider", model="model-a"),
        expected_model_hash=model_hash,
        budget=ExperimentBudget(max_candidate_evaluations=2),
        seeds=(0,),
    )


def test_experiment_declaring_model_a_refuses_runtime_model_b():
    reg = registry()
    assert bind_model(_plan(), reg, Recording("hash-a")).hashes == ("hash-a",)
    with pytest.raises(ModelIdentityError, match="hash-b"):
        bind_model(_plan(), reg, Recording("hash-b"))


def test_experiment_model_must_be_pinned_and_described_consistently():
    reg = registry()
    with pytest.raises(UnknownModelError):
        bind_model(_plan("unpinned-hash"), reg)
    for wrong in (
        ModelConfiguration(provider="someone-else", model="model-a"),
        ModelConfiguration(provider="test-provider", model="model-b"),
        ModelConfiguration(
            provider="test-provider", model="model-a", parameters={"structured_output": False}
        ),
    ):
        with pytest.raises(ModelIdentityError):
            bind_model(_plan(model=wrong), reg)


def test_experiment_refuses_a_plan_outside_its_declared_models_before_any_run():
    from tests.test_optimization_experiment import metered, suite_and_contract

    suite, _ = suite_and_contract()
    calls: list[str] = []
    with pytest.raises(DisallowedModelError):
        run_optimization_experiment(
            _plan("synthetic"),
            suite,
            metered(calls=calls),
            checker=None,  # never reached
            evaluator_version="x",
            synthetic=True,
            models=registry().allow(["model-a"]),
        )
    assert calls == []


def test_uploaded_dataset_experiment_through_the_registry(tmp_path):
    pytest.importorskip("agent_framework")
    from core.dataset import SplitRole
    from tests.test_contract_runtime import PassageModel
    from tests.test_optimization_experiment import UPLOAD, uploaded_contract, uploaded_splits

    pinned = entry("passage", PassageModel.model_hash, model_id="passage", pricing=PRICE)
    reg = registry(pinned)
    contract = uploaded_contract()
    splits = uploaded_splits(contract)
    p = ExperimentPlan(
        model=ModelConfiguration(provider="test-provider", model="passage"),
        expected_model_hash=PassageModel.model_hash,
        budget=ExperimentBudget(max_candidate_evaluations=2, max_tokens=200_000),
        seeds=(0,),
    )

    def experiment(pricing):
        client = RegisteredModelClient(pinned, PassageModel())
        models = bind_model(p, reg, client)
        runner = WorkflowRunner(model=client, benchmark_hash="inline", allowed_models=models)
        artifact = optimize_uploaded_dataset(
            contract,
            splits,
            UPLOAD,
            lambda g, t, tr, s: runner.run_sync(g, t, trial=tr, seed=s),
            p,
            checker=runner.checker,
            synthetic=True,
            pricing=pricing(models),
            models=models,
        )
        return artifact, client

    artifact, client = experiment(registry_pricing)
    assert artifact["model_hashes"] == [pinned.model_hash]
    assert artifact["identity"]["problem"]["pricing"].startswith("registry-pricing/1:")
    for run in artifact["runs"]:
        usage = run["usage"]
        expected = PRICE.cost(usage["prompt_tokens"], usage["completion_tokens"])
        assert usage["cost"] == pytest.approx(expected) and usage["cost"] > 0
    assert client.backend.requests
    assert not set(splits.split(SplitRole.TEST).row_ids) & {
        e["row_id"] for r in artifact["runs"] for c in r["candidates"] for e in c["runs"]
    }

    unpriced, _ = experiment(lambda models: None)  # no prices: cost unknown, never 0
    assert unpriced["identity"]["problem"]["pricing"] is None
    assert all(r["usage"]["cost"] is None for r in unpriced["runs"])


# == capabilities / context =====================================================================
def test_requirements_reject_unsupported_capabilities_and_unknown_context():
    needs_structured = ModelRequirements(capabilities=(TEXT, STRUCTURED))
    reg = registry(entry(capabilities=(TEXT,)), entry("model-b", "hash-b", context_window=None))
    with pytest.raises(UnsupportedCapabilityError, match="structured_output"):
        reg.allow(["model-a"], needs_structured)
    with pytest.raises(UnsupportedCapabilityError, match="unknown"):
        reg.allow(["model-b"], ModelRequirements(min_context_tokens=4096))
    with pytest.raises(UnsupportedCapabilityError, match="32768 < required"):
        registry().allow(["model-a"], ModelRequirements(min_context_tokens=65_536))
    assert registry().allow(["model-a"], needs_structured).hashes == ("hash-a",)


def test_requests_beyond_the_context_window_never_reach_the_backend():
    backend = Recording()
    client = RegisteredModelClient(entry(context_window=100), backend)
    with pytest.raises(ModelContractViolation, match="context window") as exc:
        asyncio.run(client.generate(request(max_tokens=101)))
    assert exc.value.attempts == 0 and backend.requests == []
    asyncio.run(client.generate(request(max_tokens=100)))
    assert len(backend.requests) == 1


def test_model_stage_charges_no_call_for_a_refused_request():
    from core.stages import DirectMethod, DirectStage
    from runtime.executors.base import ExecutorInput
    from runtime.executors.model_stages import DirectExecutor
    from tests.conftest import make_task
    from tests.runtime_helpers import make_ctx

    backend = Recording()
    client = RegisteredModelClient(entry(context_window=8), backend)
    task = make_task()
    out = asyncio.run(
        DirectExecutor().run(
            ExecutorInput(
                stage_index=0, stage=DirectStage(method=DirectMethod.ANSWER), payload=task
            ),
            make_ctx(task, client),
        )
    )
    assert out.failure is not None and "context window" in out.failure.message
    assert out.metrics.model_calls == 0 and out.usage.tokens == 0 and backend.requests == []


def test_adapter_sends_a_response_format_only_for_structured_entries():
    def payload(caps):
        e = entry(adapter="openai_compatible", endpoint="http://h/v1", capabilities=caps)
        cfg = OpenAICompatibleConfig.from_entry(e)
        return OpenAICompatibleClient(cfg)._payload(request())  # noqa: SLF001

    assert "response_format" in payload((TEXT, STRUCTURED))
    assert "response_format" not in payload((TEXT,))


# == retry / timeout ============================================================================
def test_retry_policy_is_exponential_honours_retry_after_and_is_capped():
    policy = RetryPolicy(max_retries=3, backoff_s=2.0, max_delay_s=10.0)
    assert [policy.delay(n) for n in range(4)] == [2.0, 4.0, 8.0, 10.0]
    assert policy.delay(0, retry_after=5.0) == 5.0 and policy.delay(0, retry_after=99) == 10.0
    assert [policy.retries(n) for n in range(4)] == [True, True, True, False]
    assert policy.max_attempts == 4
    with pytest.raises(ValidationError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValidationError):
        TimeoutPolicy(request_timeout_s=0)


def test_adapter_applies_the_entrys_timeout_and_retry_policy(monkeypatch):
    from runtime.backends import openai_compatible

    pinned = entry(
        adapter="openai_compatible",
        endpoint="http://h/v1",
        model_hash=openai_compatible_model_hash("model-a", "", "http://h/v1", True),
        timeout=TimeoutPolicy(request_timeout_s=7.5),
        retry=RetryPolicy(max_retries=2, backoff_s=0.5, max_delay_s=0.75),
    )
    timeouts: list[float] = []
    sleeps: list[float] = []

    def unavailable(req, timeout):
        timeouts.append(timeout)
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(openai_compatible.time, "sleep", sleeps.append)
    client = client_for(pinned)
    with pytest.raises(ModelError, match="HTTP 503") as exc:
        asyncio.run(client.generate(request()))
    assert timeouts == [7.5, 7.5, 7.5]  # 1 call + max_retries
    assert sleeps == [0.5, 0.75]  # backoff, capped by max_delay_s
    assert exc.value.attempts == 3


# == token accounting ===========================================================================
def test_registered_client_passes_usage_attempts_and_backoff_through():
    backend = Recording()
    resp = asyncio.run(RegisteredModelClient(entry(), backend).generate(request()))
    assert (resp.prompt_tokens, resp.completion_tokens, resp.total_tokens) == (70, 30, 100)
    assert (resp.attempts, resp.backoff_time_s, resp.model_hash) == (2, 0.25, "hash-a")
    assert backend.requests == [request()]  # seed, schema and max_tokens forwarded unchanged


# == pricing ====================================================================================
def usage(model_hash: str = "hash-a", prompt: int = 700, completion: int = 300) -> MeasuredUsage:
    return MeasuredUsage(
        model_hash=model_hash,
        model_calls=2,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokens=prompt + completion,
        tool_calls=0,
        retries=0,
        wall_time_s=1.0,
    )


def test_registry_pricing_prices_measured_usage():
    pricing = registry_pricing(registry(entry(pricing=PRICE)).allow(["model-a"]))
    assert isinstance(pricing, RegistryPricing)
    assert pricing.cost(usage()) == pytest.approx((700 * 2.0 + 300 * 6.0) / 1e6)
    with pytest.raises(KeyError):
        pricing.cost(usage("hash-b"))  # a model it was not built for is never priced at 0


@pytest.mark.parametrize("tokens", [0, 1, 999, 40_000])
def test_the_quote_bounds_every_prompt_completion_split(tokens):
    pricing = RegistryPricing([entry(pricing=PRICE)])
    quote = pricing.max_cost("hash-a", model_calls=3, tokens=tokens)
    for prompt in range(0, tokens + 1, max(1, tokens // 50)):
        assert pricing.cost(usage(prompt=prompt, completion=tokens - prompt)) <= quote + 1e-12
    assert quote == pytest.approx(tokens * 6.0 / 1e6)  # tight: all-completion reaches it


def test_unpriced_models_mean_cost_unknown_never_zero():
    reg = registry(entry(pricing=PRICE), entry("model-b", "hash-b"))
    assert registry_pricing(reg.allow(["model-b"])) is None
    assert registry_pricing(reg.allow(["model-a", "model-b"])) is None  # one unknown: unknown
    budget = ExperimentBudget(max_candidate_evaluations=1)
    assert BudgetLedger(budget, None).cost is None
    with pytest.raises(ExperimentBudgetError, match="pricing policy"):
        check_budget(ExperimentBudget(max_candidate_evaluations=1, max_cost=1.0), None)
    # pricing is optional: an unpriced model is still perfectly usable
    assert reg.allow(["model-b"]).admit("hash-b").pricing is None


def test_pricing_identity_names_the_exact_price_sheets():
    a = RegistryPricing([entry(pricing=PRICE)])
    b = RegistryPricing([entry(pricing=PRICE.model_copy(update={"version": "test-sheet/2"}))])
    c = RegistryPricing([entry(pricing=PRICE.model_copy(update={"prompt_per_million": 3.0}))])
    assert len({a.identity, b.identity, c.identity}) == 3
    assert a.identity == RegistryPricing([entry(pricing=PRICE)]).identity
    for bad in ({"prompt_per_million": -1}, {"completion_per_million": math.inf}, {"version": ""}):
        with pytest.raises(ValidationError):
            ModelPricing(**{**PRICE.model_dump(), **bad})


def test_registry_pricing_reserves_a_safe_cost_bound_and_fails_closed_on_strangers():
    from tests.test_optimization_experiment import limited_contract

    pricing = RegistryPricing([entry(pricing=PRICE)])
    contract = limited_contract()
    budget = ExperimentBudget(max_candidate_evaluations=1, max_cost=10.0)
    reservation = run_reservation(budget, contract, pricing, "hash-a")
    tokens = contract.constraints.maximum_tokens_per_example
    assert reservation.cost == pytest.approx(tokens * 6.0 / 1e6)
    with pytest.raises(ExperimentBudgetError, match="cannot quote"):
        run_reservation(budget, contract, pricing, "hash-b")

    ledger = BudgetLedger(budget, pricing, reservation)
    with pytest.raises(PricingError):  # a measured run of a model it has no price for
        ledger.price(usage("hash-b"))


# == compatibility ==============================================================================
def test_frozen_experiment_hash_reproduces_through_a_registry_entry():
    protocol = json.loads(
        (ROOT / "experiments/musique_frozen/protocol-v2.json").read_text(encoding="utf-8")
    )
    model = ModelConfiguration(**protocol["model"])
    pinned = ModelEntry(
        name="frozen-protocol-model",
        provider=model.provider,
        adapter="openai_compatible",
        model_id=model.model,
        endpoint="https://zenmux.ai/api/v1",
        model_hash=protocol["expected_model_hash"],
        capabilities=(TEXT, STRUCTURED),
    )
    client = registered_client(pinned)  # constructing never contacts the backend
    assert client.model_hash == protocol["expected_model_hash"]
    pinned.check_configuration(model)
    plan = _plan(protocol["expected_model_hash"], model)
    assert bind_model(plan, ModelRegistry(entries=(pinned,)), client).hashes == (
        protocol["expected_model_hash"],
    )


def test_legacy_import_path_and_env_names_still_work():
    from runtime import gemma_client, model_client
    from runtime.backends import openai_compatible

    assert gemma_client.GemmaConfig is openai_compatible.OpenAICompatibleConfig
    assert gemma_client.OpenAICompatibleClient is openai_compatible.OpenAICompatibleClient
    for name in ("GenerationRequest", "GenerationResponse", "ModelRole", "ModelError"):
        assert getattr(gemma_client, name) is getattr(model_client, name)
    legacy = {"GEMMA_BASE_URL": "http://h/v1", "GEMMA_MODEL": "m", "GEMMA_MAX_RETRIES": "5"}
    cfg = OpenAICompatibleConfig.from_env(legacy)
    assert (cfg.base_url, cfg.model, cfg.max_retries) == ("http://h/v1", "m", 5)
    neutral = OpenAICompatibleConfig.from_env({**legacy, "WYNK_MODEL": "n"})
    assert neutral.model == "n"  # the provider-neutral name wins
    client = client_from_env(legacy)
    assert isinstance(client, RegisteredModelClient)
    assert client.model_hash == OpenAICompatibleClient(cfg).model_hash


def test_adapters_refuse_entries_they_cannot_serve():
    from core.models import ModelRegistryError

    with pytest.raises(ModelRegistryError, match="no adapter"):
        client_for(entry(adapter="carrier-pigeon"))
    with pytest.raises(ModelRegistryError, match="no endpoint"):
        client_for(entry(adapter="openai_compatible"))
    with pytest.raises(ModelIdentityError):  # pinned hash does not reproduce
        client_for(entry(adapter="openai_compatible", endpoint="http://h/v1"))


# == provider-neutral boundaries ================================================================
NEUTRAL = sorted(
    {
        *(ROOT / "optimizers").glob("*.py"),
        *(ROOT / "core").glob("*.py"),
        *(ROOT / "compiler").glob("*.py"),
        *(ROOT / "evaluation").glob("*.py"),
        *(ROOT / "runtime" / "executors").glob("*.py"),
        *(ROOT / "runtime" / "prompts").glob("*.py"),
        *(
            ROOT / "runtime" / name
            for name in (
                "runner.py",
                "stage_runner.py",
                "maf_nodes.py",
                "budget_guard.py",
                "model_client.py",
            )
        ),
        ROOT / "experiments" / "optimization_experiment.py",
        ROOT / "experiments" / "budget_ledger.py",
    }
)
PROVIDER_MODULES = ("runtime.backends", "runtime.gemma_client", "urllib", "http", "openai")


@pytest.mark.parametrize("path", NEUTRAL, ids=lambda p: str(p.relative_to(ROOT)))
def test_core_search_and_runtime_paths_import_no_provider_code(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        else:
            continue
        for m in modules:
            assert not any(m == p or m.startswith(p + ".") for p in PROVIDER_MODULES), (
                f"{path.name} imports {m}"
            )
    assert "gemma" not in path.read_text(encoding="utf-8").lower()
