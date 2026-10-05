# wynk UI

The product shell for dataset-driven workflow optimization:

**Project → Dataset → Configure → Optimize → Compare → Champion**

```bash
cd ui
npm install
npm run dev        # http://localhost:5173, mock data by default
npm run check      # typecheck + lint + unit/component tests + build
npm run test:e2e   # Playwright flows in your installed Chrome (PW_CHANNEL=chromium for the bundled one)
```

## Screens

| Route | Screen |
|---|---|
| `/projects` | projects, create one |
| `/projects/:id/datasets` | datasets with mapping status |
| `/projects/:id/datasets/new` | pick a CSV or JSONL file; preview, schema, column roles (input, target, context, row id) |
| `/projects/:id/datasets/:datasetId` | edit the mapping, preview rows |
| `/projects/:id/experiments/new` | task type, instructions, evaluator, **hard constraints** (must hold) vs **optimization preference** (objective, balanced weights), models, budget, splits |
| `/projects/:id/experiments/:experimentId` | run status, best quality/cost/latency, learning curve, champion graph, search (ACO) summary, candidate table |
| `/projects/:id/experiments/:experimentId/results` | fixed baseline vs random search vs Wynk on optimization, validation and held-out test; champion; *Why this workflow?* |
| `/projects/:id/workflows` | validated workflows and champions |
| `/playground` | the original ask-a-question chat demo (`?demo` replays it without a backend) |

Workflow states follow the split policy in `core/dataset.py`: **Candidate** (measured on the
optimization split, the only split that feeds the search) → **Validated** (re-measured on validation,
which may select and promote but never feeds the search) → **Champion** (the validated workflow the
objective ranks first among those meeting every hard constraint). Held-out test plays no part in any
state: it is measured once, after the champion is fixed, and only reported. Deployment is shown only
as *coming soon*.

## Data boundary

Screens talk to one typed adapter, `api()` from `src/api`. They never import mock data, build URLs or
read backend JSON.

- `src/api/types.ts` – the UI's view models. They carry the meaning of the merged contracts but are
  shaped for screens; they are **not** the Python models copied into TypeScript.
- `src/api/contract/` – the one translation layer to the merged contracts:
  - `wire.ts` – JSON shapes of `DatasetSpec`, `SplitPlan`, `EvaluationSpec`, `ObjectiveSpec`,
    `ConstraintLimits`, `CandidateMeasurements` and `TaskContract`. Only adapters and the mapping use it.
  - `mapping.ts` – view model → contract (`toTaskContract`, `toDatasetSpec`, `toSplitPlan`, …) and
    contract measurements → view model (`fromMeasurements`).
  - `rules.ts` – contract rules the forms need before anything is sent (column roles and types,
    task type ↔ evaluator, split uses, split sizing, the quality floor for cost/latency objectives).
  - `semantics.ts` – `check_limits` and `TaskContract.rank`, mirrored so mocked runs rank the same way.
- `src/api/client.ts` – the `WynkApi` interface every implementation satisfies.
- `src/api/mock/` – the default: in-memory fixtures and a seeded, simulated search. Invented numbers,
  nothing persisted, resets on reload. Fixtures are valid contract instances, and every new
  experiment is built into a `TaskContract` through the mapping layer. The UI always shows a
  **Mock data** badge in this mode.
- `src/api/live.ts` – `?api=live` or `VITE_WYNK_API=live`. The product API (endpoints) does not exist
  yet, so every call fails with `not_connected`; nothing fakes success. A real adapter will use
  `src/api/contract/mapping.ts`.

Where presentation differs from the contract, the conversion happens in one place:

| UI | Contract |
|---|---|
| objective quality / cost / latency / balanced | `maximize_quality` / `minimize_cost` / `minimize_latency` / `balanced` |
| balanced weights in whole % (sum 100) + scales | `weights` summing to 1, `scales` for penalized metrics only |
| split percentages + seed; optimization = the rest | `SplitPlan` basis points + seed |
| cost shown per 1,000 examples | `maximum_cost_per_example`, `mean_cost_per_example` (USD per example) |
| latency in seconds (mean and p95) | `maximum_mean_latency_s`, `maximum_p95_latency_s` |
| models the search may use | model configuration (`ModelConfiguration`), not a constraint |
| column roles input / target / context / row id | `input_columns`, `target_columns`, `context_columns`, `id_column` |

Not exposed yet: per-example run caps (`maximum_model_calls`, `maximum_tool_calls`, `maximum_retries`,
`maximum_wall_time_s`) are sent as no limit; `legacy_field_match` and the `wynk_snapshot` format belong
to the frozen benchmark. The search budget (candidates, generations, spend, duration) is a UI setting
with no backend contract yet.

Mock knobs (query string): `mockRunMs` (length of a new run), `mockLatency` (response delay),
`mockFail=listProjects,getResults` or `*` (inject failures to see error states).

Dataset files are parsed in the browser for preview only (`src/lib/dataset.ts`): column types,
nullability and the sha256 content hash follow the dataset contract. Nothing is uploaded. Parquet is
recognised but is not a contract format yet, so it is not previewed.

*Why this workflow?* (`src/lib/why.ts`) only restates measurements: quality, cost and latency differences
against the baselines on one split, constraint checks and how the champion was selected. It makes no
causal claims, and a metric that was not measured produces no statement.

## Design

White ground, one magenta accent (`src/styles.css` tokens), Geist / Geist Mono / Bricolage Grotesque.
Product screens are dense tables and panels: no blobs, glows or decorative motion. State colours
(good / bad / warn) always come with an icon and a label. Chart series: Wynk solid magenta, random
search dashed blue (a validated, colour-blind-safe pair with the dash as secondary encoding), fixed
baseline dotted grey.

The chat demo's pieces (ghost, completion bar, ant cards) are unchanged under `/playground`;
`WorkflowGraph` is shared and now also draws a static champion workflow.
