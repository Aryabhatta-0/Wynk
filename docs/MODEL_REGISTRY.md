# Model registry and ModelClient

One authority decides which models exist, what they can do, how they are called, what they cost
and which exact runtime identity each one reports: the model registry (`core/models.py`). The
runtime, executors and optimizers depend only on the provider-neutral `ModelClient` contract
(`runtime/model_client.py`). Provider code lives in adapters (`runtime/backends/`), and only
entry points (CLIs, the API) import them.

```
ModelRegistry ── pinned ModelEntry (unique name, natural key, model_hash)
   │  registry.allow([...], ModelRequirements)  /  bind_model(plan, registry, client)
   ▼
AllowedModels ── the explicitly declared set; requirements checked when it is built
   │  WorkflowRunner(allowed_models=...)  → admit(client.model_hash) before any call
   ▼
RegisteredModelClient(entry, backend) ── backend.model_hash must == entry.model_hash
   │  every request checked against the entry; every response must carry the pinned hash
   ▼
RunVersions.model_hash ──► ExperimentPlan.expected_model_hash ──► experiment identity
```

Every arrow is an equality check that fails closed, so no second model identity can disagree
silently. `ModelConfiguration` (hashed into `ExperimentIdentity`) is checked against the entry by
`ModelEntry.check_configuration`: same provider, same model id, and the same `structured_output`
or `revision` if the configuration states one.

## Registry schema (`wynk-model-registry/1`)

```json
{
  "schema_version": "wynk-model-registry/1",
  "entries": [
    {
      "name": "example-model@example-gateway",
      "provider": "example-gateway",
      "adapter": "openai_compatible",
      "model_id": "vendor/example-model",
      "revision": null,
      "endpoint": "https://gateway.example/v1",
      "model_hash": "<openai_compatible_model_hash(model_id, revision or '', endpoint, structured)>",
      "context_window": 131072,
      "capabilities": ["structured_output", "text_generation"],
      "timeout": {"request_timeout_s": 120.0},
      "retry": {"max_retries": 3, "backoff_s": 2.0, "max_delay_s": 60.0},
      "pricing": null,
      "enabled": true
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | registry key (unique) |
| `provider` | who serves the model; equals the experiment's `ModelConfiguration.provider` |
| `adapter` | which backend speaks its API (`openai_compatible`) |
| `model_id`, `revision` | what is requested. `revision` is optional because a gateway may not expose one. |
| `endpoint` | base URL. Credentials are never part of the registry. |
| `model_hash` | the exact identity the runtime client reports (**pinned**). `client_for(entry)` fails if the adapter does not reproduce it. |
| `context_window` | tokens; `null` = **unknown**, which never satisfies a context requirement |
| `capabilities` | `text_generation`, `structured_output`. Only what the entry declares; never inferred from a name. |
| `timeout`, `retry` | per-call timeout; transient-failure retries (exponential backoff, honours `Retry-After`, capped) |
| `pricing` | optional `{version, currency, prompt_per_million, completion_per_million}` |
| `enabled` | a disabled entry can be neither allowed nor bound |

Registries reject duplicate names, duplicate `model_hash`es and two entries for the same model
(provider, adapter, model id, revision, endpoint, structured). An unknown hash or name raises
`UnknownModelError`.

## Allowed models

A task or experiment declares the models it may run on as an `AllowedModels` set. The set is built
by `ModelRegistry.allow(refs, requirements)`, or by `bind_model(plan, registry, client)` for an
experiment, whose set is exactly its `expected_model_hash`. Once a runner has an allowed set, it
refuses each of the following with an exception, before compiling the workflow or invoking a
model:

* a client whose `model_hash` is outside the set (`DisallowedModelError`)
* a registered client bound to a different entry for the same hash (`ModelIdentityError`)
* a workflow with model stages on a model that declares no text generation, or whose known
  context window is smaller than one stage call's output budget (`UnsupportedCapabilityError`)

At request level, `RegisteredModelClient` refuses a request whose `max_tokens` exceeds the
context window. The backend is never invoked, so the refusal counts zero model calls.

With one allowed model, behaviour is exactly the #23 single-model binding: the same runs, the same
`RunVersions`, the same `expected_model_hash` checks. The optimizer does not choose models. The
registry provides the authority boundary that a later model-search change would use.

## Pricing

`RegistryPricing` implements the experiment ledger's `PricingPolicy` from registry metadata:

* `cost(usage)` = prompt tokens × prompt price + completion tokens × completion price
* `max_cost(model_hash, calls, tokens)` = tokens × max(prompt, completion price): a safe upper
  bound for any prompt/completion split, used to reserve a hard cost cap
* `identity` = `registry-pricing/1:<hash of every price sheet used>`, part of the experiment
  identity

`registry_pricing(models)` returns `None` when any allowed model is unpriced. Cost is then
**unknown** (`None`), never zero, and a `max_cost` cap fails closed. Pricing is never required
for a model to be usable. No prices are hard-coded anywhere in the runtime.

## Compatibility

* `OpenAICompatibleClient` keeps its identity formula, so frozen experiment hashes (for example,
  MuSiQue protocol v1/v2) still match. `tests/test_model_registry.py` reproduces protocol v2's hash
  through a registry entry.
* `runtime/gemma_client.py` is a deprecated re-export (`GemmaConfig` = `OpenAICompatibleConfig`),
  and each `WYNK_MODEL_*` env name falls back to its legacy `GEMMA_*` name.
* `runtime/executors/gemma_stages.py` is now `runtime/executors/model_stages.py`.
