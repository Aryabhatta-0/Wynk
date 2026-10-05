import { ApiError, type WynkApi } from "./client";
import * as decode from "./contract/decode";
import { WireShapeError } from "./contract/decode";
import {
  ContractMappingError,
  fromDatasetVersion,
  fromDatasetView,
  fromProject,
  fromSplitsRecord,
  fromUpload,
  toRegisterDataset,
  splitPlanWire,
} from "./contract/mapping";
import type { NewProjectWire } from "./contract/wire";

/*
  The product API v1 adapter (api/product.py). Every method is one documented route:

    listProjects        GET  /projects                         createProject  POST /projects
    getProject          GET  /projects/{id}
    uploadDataset       POST /projects/{id}/uploads?format=&filename=   (raw file bytes)
    getUpload           GET  /uploads/{id}                     registerDataset POST /uploads/{id}/register
    listDatasets        GET  /projects/{id}/datasets           getDataset     GET  /datasets/{id}
    getDatasetVersion   GET  /datasets/{id}/versions/{n}
    createSplits        POST /datasets/{id}/versions/{n}/splits
    listSplits          GET  /datasets/{id}/versions/{n}/splits
    getSplits           GET  /datasets/{id}/versions/{n}/splits/{hash}

  Responses are checked against the wire shapes (`contract/decode.ts`) and mapped to view models
  (`contract/mapping.ts`). Errors keep the server's stable code and message. Nothing here falls
  back to mock data, and the experiment methods fail without a request: there is no experiment
  backend yet (Issues #20–#23).
*/

export const DEFAULT_BASE_URL = "/api/v1";

export const EXPERIMENT_BACKEND_MISSING =
  "Wynk's product API does not run experiments yet: real optimization on uploaded datasets is blocked on Issues #20–#23. " +
  "Open the demo with ?api=mock to explore a simulated run.";

export interface LiveOptions {
  /** where the product API is mounted; default `/api/v1` (the dev server proxies `/api`) */
  baseUrl?: string;
  /** injectable for tests */
  fetch?: typeof fetch;
}

type Body = { json: unknown } | { bytes: Blob } | undefined;

export function createLiveApi(options: LiveOptions = {}): WynkApi {
  const base = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  const doFetch = options.fetch ?? ((input: RequestInfo | URL, init?: RequestInit) => globalThis.fetch(input, init));

  async function call<T>(method: "GET" | "POST", path: string, body: Body, read: (raw: unknown) => T): Promise<{ value: T; status: number }> {
    const init: RequestInit = { method, headers: { Accept: "application/json" } };
    if (body && "json" in body) {
      init.headers = { ...init.headers, "Content-Type": "application/json" };
      init.body = JSON.stringify(body.json);
    } else if (body) {
      // raw bytes, never a form: the server refuses multipart bodies
      init.headers = { ...init.headers, "Content-Type": "application/octet-stream" };
      init.body = body.bytes;
    }
    const url = `${base}${path}`;
    let res: Response;
    try {
      res = await doFetch(url, init);
    } catch (err) {
      throw new ApiError("network_error", `${method} ${url} failed: ${err instanceof Error ? err.message : String(err)}. Is the Wynk API running?`);
    }
    let raw: unknown = undefined;
    const text = await res.text().catch(() => "");
    try {
      raw = text ? JSON.parse(text) : undefined;
    } catch {
      raw = undefined;
    }
    if (!res.ok) {
      const envelope = decode.errorResponse(raw);
      if (envelope) throw new ApiError(envelope.error.code, envelope.error.message, { status: res.status, details: envelope.error.details });
      throw new ApiError("bad_response", `${method} ${url} answered HTTP ${res.status} without a Wynk API error body. Is the Wynk API running behind this address?`, {
        status: res.status,
      });
    }
    if (raw === undefined)
      throw new ApiError("bad_response", `${method} ${url} answered HTTP ${res.status} without a JSON body.`, { status: res.status });
    try {
      return { value: read(raw), status: res.status };
    } catch (err) {
      if (err instanceof WireShapeError || err instanceof ContractMappingError)
        throw new ApiError("contract_mismatch", `${method} ${url}: ${err.message}`, { status: res.status });
      throw err;
    }
  }

  const get = async <T>(path: string, read: (raw: unknown) => T) => (await call("GET", path, undefined, read)).value;
  /** 201: created now; 200: the identical resource already existed */
  const create = async <T>(path: string, body: Body, read: (raw: unknown) => T) => {
    const { value, status } = await call("POST", path, body, read);
    return { value, created: status === 201 };
  };
  const seg = encodeURIComponent;
  const versionPath = (datasetId: string, version: number) => `/datasets/${seg(datasetId)}/versions/${version}`;
  const noExperiments = (): Promise<never> => Promise.reject(new ApiError("experiment_backend_unavailable", EXPERIMENT_BACKEND_MISSING));

  return {
    mode: "live",
    experiments: "unavailable",

    listProjects: () => get("/projects", (r) => decode.projectList(r).projects.map(fromProject)),
    getProject: (id) => get(`/projects/${seg(id)}`, (r) => fromProject(decode.project(r))),
    createProject: async (input) => {
      const body: NewProjectWire = { name: input.name.trim(), description: input.description.trim() };
      return (await create("/projects", { json: body }, (r) => fromProject(decode.project(r)))).value;
    },

    uploadDataset: (projectId, file) => {
      const query = new URLSearchParams({ format: file.format, filename: file.name });
      return create(`/projects/${seg(projectId)}/uploads?${query}`, { bytes: file.bytes }, (r) => fromUpload(decode.upload(r)));
    },
    getUpload: (uploadId) => get(`/uploads/${seg(uploadId)}`, (r) => fromUpload(decode.upload(r))),
    registerDataset: (uploadId, input) =>
      create(`/uploads/${seg(uploadId)}/register`, { json: toRegisterDataset(input) }, (r) => fromDatasetVersion(decode.datasetVersion(r))),
    listDatasets: (projectId) => get(`/projects/${seg(projectId)}/datasets`, (r) => decode.datasetList(r).datasets.map(fromDatasetView)),
    getDataset: (datasetId) => get(`/datasets/${seg(datasetId)}`, (r) => fromDatasetView(decode.datasetView(r))),
    getDatasetVersion: (datasetId, version) => get(versionPath(datasetId, version), (r) => fromDatasetVersion(decode.datasetVersion(r))),
    createSplits: (datasetId, version, plan) =>
      create(`${versionPath(datasetId, version)}/splits`, { json: splitPlanWire(plan) }, (r) => fromSplitsRecord(decode.splitsRecord(r))),
    listSplits: (datasetId, version) => get(`${versionPath(datasetId, version)}/splits`, (r) => decode.splitsList(r).splits.map(fromSplitsRecord)),
    getSplits: (datasetId, version, splitsHash) =>
      get(`${versionPath(datasetId, version)}/splits/${seg(splitsHash)}`, (r) => fromSplitsRecord(decode.splitsRecord(r))),

    listModels: noExperiments,
    listExperiments: noExperiments,
    getExperiment: noExperiments,
    createExperiment: noExperiments,
    cancelExperiment: noExperiments,
    getResults: noExperiments,
    listWorkflows: noExperiments,
  };
}
