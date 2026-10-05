import type { WynkApi } from "./client";
import { createLiveApi } from "./live";
import type { MockOptions } from "./mock/mockApi";
import type { Dataset, DatasetVersion } from "./types";

export { ApiError, ERROR_COPY, errorMessage, type ApiErrorKind, type WynkApi } from "./client";
export { EXPERIMENT_BACKEND_MISSING } from "./live";
export type * from "./types";

/*
  Adapter selection. The real product API is the default; mock data is only ever used when asked
  for, and a live failure is shown as a failure, never answered from mock data.
    (default)                         live product API at /api/v1 (VITE_WYNK_API_BASE overrides)
    ?api=mock  or  VITE_WYNK_API=mock  mock adapter, for automated tests and demos
  Mock-only knobs:
    ?mockRunMs=4000        length of a new optimization run
    ?mockLatency=0         artificial response delay
    ?mockFail=listProjects comma-separated methods that fail ("*" for all)
*/
export type ApiMode = "live" | "mock";

export function selectMode(query: string, env: string | undefined): ApiMode {
  // an empty value (e.g. VITE_WYNK_API= in a .env file) means "not set"
  const requested = new URLSearchParams(query).get("api") || env || "live";
  if (requested === "live" || requested === "mock") return requested;
  throw new Error(`Unknown API mode "${requested}". Use api=live (the default) or api=mock.`);
}

const METHODS = [
  "listProjects",
  "getProject",
  "createProject",
  "uploadDataset",
  "getUpload",
  "registerDataset",
  "listDatasets",
  "getDataset",
  "getDatasetVersion",
  "createSplits",
  "listSplits",
  "getSplits",
  "listModels",
  "listExperiments",
  "getExperiment",
  "createExperiment",
  "cancelExperiment",
  "getResults",
  "listWorkflows",
] as const satisfies readonly Exclude<keyof WynkApi, "mode" | "experiments">[];

/** The mock adapter, loaded on first use: in live mode its code and fixtures are never fetched. */
function lazyMockApi(options: MockOptions): WynkApi {
  let loaded: Promise<WynkApi> | null = null;
  const load = () => (loaded ??= import("./mock/mockApi").then((m) => m.createMockApi(options)));
  const adapter: Record<string, unknown> = { mode: "mock", experiments: "simulated" };
  for (const name of METHODS)
    adapter[name] = async (...args: unknown[]) => {
      const mock = await load();
      return (mock[name] as (...a: unknown[]) => Promise<unknown>)(...args);
    };
  return adapter as unknown as WynkApi;
}

function fromEnvironment(): WynkApi {
  const query = window.location.search;
  if (selectMode(query, import.meta.env.VITE_WYNK_API) === "live") return createLiveApi({ baseUrl: import.meta.env.VITE_WYNK_API_BASE });
  const params = new URLSearchParams(query);
  const num = (key: string) => {
    const v = params.get(key);
    return v === null || v === "" || Number.isNaN(Number(v)) ? undefined : Number(v);
  };
  return lazyMockApi({
    runMs: num("mockRunMs"),
    latencyMs: num("mockLatency"),
    failOn: params.get("mockFail")?.split(",").filter(Boolean),
  });
}

let instance: WynkApi | null = null;

/** The one adapter every screen talks to. */
export function api(): WynkApi {
  instance ??= fromEnvironment();
  return instance;
}

/** Tests swap in their own adapter. */
export function setApi(next: WynkApi | null) {
  instance = next;
}

/** The version a dataset reports as its latest. */
export function latestVersion(d: Dataset): DatasetVersion {
  return d.versions.find((v) => v.version === d.latestVersion) ?? d.versions[d.versions.length - 1];
}
