# wynk UI

The product shell for dataset-driven workflow optimization:

**Project → Dataset → Configure → Optimize → Compare → Champion**

The UI talks to the real Wynk product API by default. Projects, datasets, uploads, registration
and splits are real; **experiments are not**: there is no experiment backend yet (Issues #20–#23),
so in live mode the experiment screens say so, and only the mock demo simulates them.

```bash
# backend (repo root): chat + product API v1 on :8787, which the dev server proxies /api to
# #32: tenant routes need an API key; register one for a dev workspace (stored as a hash)
export WYNK_BOOTSTRAP_API_KEY=wynk_sk_00000000000000de_$(python -c "print('D'*43)")
python -m api.chat --env-file .env          # or: python -m api.product --port 8787

cd ui
npm install
WYNK_API_KEY=$WYNK_BOOTSTRAP_API_KEY npm run dev  # http://localhost:5173, live product API;
                   # the dev proxy adds the key server-side (never in the browser)
                   # http://localhost:5173/projects?api=mock  for the mock demo (or VITE_WYNK_API=mock)
npm run check      # typecheck + lint + unit/component tests + build
npm run test:e2e   # Playwright: starts `python -m api.product` on a fresh data dir + the dev server
                   # (WYNK_PYTHON=path/to/python; PW_CHANNEL=chromium for Playwright's bundled browser)
```

## Screens

| Route | Screen |
|---|---|
| `/projects` | projects, create one |
| `/projects/:id/datasets` | registered datasets: format, rows, columns, latest version, row-id source |
| `/projects/:id/datasets/new` | **Upload → Inspect → Map columns → Register**: the file's bytes go to the API, which infers types, nullability, row count, content hash and a preview; you choose roles (input, target, context, row id) and register. `?upload=` keeps the upload across reloads |
| `/projects/:id/datasets/:datasetId` | **Create splits**, then the dataset: versions (`?version=`), content and identity hashes, row-id source, durable splits with server-computed sizes, roles (changing them registers a new version), preview |
| `/projects/:id/experiments/new` | *(mock only)* task type, instructions, evaluator, **hard constraints** (must hold) vs **optimization preference** (objective, balanced weights), models, budget, splits |
| `/projects/:id/experiments/:experimentId` | *(mock only)* run status, best quality/cost/latency, learning curve, champion graph, search (ACO) summary, candidate table |
| `/projects/:id/experiments/:experimentId/results` | *(mock only)* fixed baseline vs random search vs Wynk on optimization, validation and held-out test; champion; *Why this workflow?* |
| `/projects/:id/workflows` | *(mock only)* validated workflows and champions |
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
    `ConstraintLimits`, `CandidateMeasurements`, `TaskContract`, and the product API v1 records and
    request bodies (`ProjectRecord`, `UploadRecord`, `RegisterDataset`, `DatasetVersionRecord`,
    `DatasetView`, `SplitsRecord`, the error envelope). Only adapters and the mapping use it.
  - `decode.ts` – checks every product API response against those shapes (exact field sets, as the
    Python models are `extra="forbid"`); drift fails as `contract_mismatch` instead of `undefined`.
  - `mapping.ts` – view model ↔ contract (`toTaskContract`, `toDatasetSpec`, `toRegisterDataset`,
    `splitPlanWire`, `fromUpload`, `fromDatasetVersion`, `fromSplitsRecord`, `fromMeasurements`, …).
    Values the server computed (types, nullability, counts, hashes, versions, split sizes) are
    copied, never recomputed.
  - `fixtures/product-api.v1.json` – real request/response pairs regenerated from `api/product.py`
    by `tests/test_product_api_ui_fixtures.py` (which fails if the API's responses change). The live
    adapter's tests replay them, so the UI is pinned to what the backend really sends.
  - `rules.ts` – contract rules the forms need before anything is sent (column roles and types,
    task type ↔ evaluator, split uses, split sizing, the quality floor for cost/latency objectives).
  - `semantics.ts` – `check_limits` and `TaskContract.rank`, mirrored so mocked runs rank the same way.
- `src/api/client.ts` – the `WynkApi` interface every implementation satisfies, and `ApiError`:
  the server's stable `code` (`api.product.ERROR_STATUS`), its `message`, `status`, `details`, plus a
  short `title` for people. Screens show all of them (`ApiErrorState`), so a friendly heading never
  hides the real reason.
- `src/api/live.ts` – **the default.** One method per product API route under `/api/v1`
  (`VITE_WYNK_API_BASE` overrides the base): projects, upload, inspect, register, datasets and
  versions, splits. Experiment methods reject with `experiment_backend_unavailable` without making a
  request. A failure is shown as a failure: there is no fallback to mock data.
- `src/api/mock/` – only with `?api=mock` or `VITE_WYNK_API=mock`, for automated tests and demos.
  Loaded lazily, so live mode never downloads it. In-memory fixtures, the same dataset lifecycle and
  error codes as the product API (`inspect.ts` stands in for the server's parser), and a seeded,
  simulated search. Invented numbers, nothing persisted, resets on reload. The UI shows a **Mock
  data** badge, and every experiment screen says the run is simulated.

Experiments (configure, run, results, workflows) have no backend yet: running them on uploaded
datasets is blocked on Issues #20–#23. With the live API those screens explain that and link the
issues; they never show a simulated run as if it were real.

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

In live mode the browser never parses a dataset: it sends the file's bytes with the `format` taken
from the extension, and the server decides (Parquet or anything else is answered with
`unsupported_format`, which is shown). The dev server's `/api` proxy lets a browser read an upload
refused before its body was read (e.g. `payload_too_large`) instead of reporting a network error; see
`vite.config.ts`.

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
