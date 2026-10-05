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
| `/projects/:id/datasets/new` | pick a CSV / JSONL / Parquet file; preview, schema, column roles (input, target, context) |
| `/projects/:id/datasets/:datasetId` | edit the mapping, preview rows |
| `/projects/:id/experiments/new` | task type, **hard constraints** (must hold) vs **optimization preference** (objective), allowed models, budget, splits |
| `/projects/:id/experiments/:experimentId` | run status, best quality/cost/latency, learning curve, champion graph, search (ACO) summary, candidate table |
| `/projects/:id/experiments/:experimentId/results` | fixed baseline vs random search vs Wynk on optimization, validation and held-out test; champion; *Why this workflow?* |
| `/projects/:id/workflows` | validated workflows and champions |
| `/playground` | the original ask-a-question chat demo (`?demo` replays it without a backend) |

Workflow states: **Candidate** (measured on the optimization split) → **Validated** (re-measured on
validation) → **Champion** (best validated workflow meeting every hard constraint). Deployment is
shown only as *coming soon*.

## Data boundary

Screens talk to one typed adapter, `api()` from `src/api`. They never import mock data or build URLs.

- `src/api/types.ts` – the UI's view models. They are **not** the backend contract; a real adapter maps the
  backend's responses onto them.
- `src/api/client.ts` – the `WynkApi` interface every implementation satisfies.
- `src/api/mock/` – the default: in-memory fixtures and a seeded, simulated search. Invented numbers,
  nothing persisted, resets on reload. The UI always shows a **Mock data** badge in this mode.
- `src/api/live.ts` – `?api=live` or `VITE_WYNK_API=live`. The product API does not exist yet, so every
  call fails with `not_connected`; nothing fakes success. Replace this file when the contract is final.

Mock knobs (query string): `mockRunMs` (length of a new run), `mockLatency` (response delay),
`mockFail=listProjects,getResults` or `*` (inject failures to see error states).

Dataset files are parsed in the browser for preview only (`src/lib/dataset.ts`); nothing is uploaded.
Parquet is recognised but needs the backend ingestion service, so it is not previewed yet.

*Why this workflow?* (`src/lib/why.ts`) only restates measurements: quality, cost and latency differences
against the baselines on one split, constraint checks and how the champion was selected. It makes no
causal claims.

## Design

White ground, one magenta accent (`src/styles.css` tokens), Geist / Geist Mono / Bricolage Grotesque.
Product screens are dense tables and panels: no blobs, glows or decorative motion. State colours
(good / bad / warn) always come with an icon and a label. Chart series: Wynk solid magenta, random
search dashed blue (a validated, colour-blind-safe pair with the dash as secondary encoding), fixed
baseline dotted grey.

The chat demo's pieces (ghost, completion bar, ant cards) are unchanged under `/playground`;
`WorkflowGraph` is shared and now also draws a static champion workflow.
