import type { WynkApi } from "./client";
import { createLiveApi } from "./live";
import { createMockApi } from "./mock/mockApi";

export { ApiError, errorMessage, type WynkApi } from "./client";
export type * from "./types";

/*
  Adapter selection. Mock is the default until the product API exists.
    ?api=live  or  VITE_WYNK_API=live    use the (not yet available) product API
  Mock-only knobs, for demos and tests:
    ?mockRunMs=4000        length of a new optimization run
    ?mockLatency=0         artificial response delay
    ?mockFail=listProjects comma-separated methods that fail ("*" for all)
*/
function fromEnvironment(): WynkApi {
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("api") ?? import.meta.env.VITE_WYNK_API ?? "mock";
  if (mode === "live") return createLiveApi();
  const num = (key: string) => {
    const v = params.get(key);
    return v === null || v === "" || Number.isNaN(Number(v)) ? undefined : Number(v);
  };
  return createMockApi({
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
