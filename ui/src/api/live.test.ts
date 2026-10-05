import { describe, expect, it, vi } from "vitest";
import { FIXTURE, body, exchange, replay } from "@/test/productApi";
import { ApiError, ERROR_COPY } from "./client";
import { fromDatasetVersion, toDatasetSpec } from "./contract/mapping";
import { createLiveApi } from "./live";
import type { ColumnMapping, DatasetFile } from "./types";

/*
  The live adapter against the product API's real exchanges: every request must be the one the
  Python API was called with, and every response is one it really sent.
*/
const live = (...names: string[]) => {
  const r = replay(...names);
  return { api: createLiveApi({ fetch: r.fetch }), ...r };
};

const fileFrom = (name: string, format = "csv"): DatasetFile => {
  const req = exchange(name).request.body as { text?: string; size?: number };
  const content = req.text ?? "x".repeat(req.size ?? 0);
  const fileName = new URLSearchParams(exchange(name).request.path.split("?")[1]).get("filename")!;
  return { name: fileName, format, bytes: new Blob([content]) };
};

const PID = body("create_project").project_id as string;
const UID = body("upload_csv").upload_id as string;
const V1: ColumnMapping = { input: ["subject", "body"], target: ["queue"], context: ["customer_tier"], id: "ticket_id" };
const V2: ColumnMapping = { input: ["body"], target: ["queue"], context: ["subject"], id: null };

describe("live adapter: projects", () => {
  it("creates, lists and gets projects; counts are unknown, not invented", async () => {
    const { api, remaining } = live("create_project", "list_projects", "get_project");
    const created = await api.createProject({ name: "Support triage", description: "Route tickets to queues" });
    const expected = {
      id: PID,
      name: "Support triage",
      description: "Route tickets to queues",
      createdAt: body("create_project").created_at,
      datasetCount: null,
      experimentCount: null,
    };
    expect(created).toEqual(expected);
    expect(await api.listProjects()).toEqual([expected]);
    expect(await api.getProject(PID)).toEqual(expected);
    expect(remaining).toEqual([]);
  });
});

describe("live adapter: upload and inspection", () => {
  it("sends the raw bytes and shows the server's inspection as reported", async () => {
    const { api, calls } = live("upload_csv", "upload_csv_again", "get_upload");
    const first = await api.uploadDataset(PID, fileFrom("upload_csv"));
    const wire = body("upload_csv");
    expect(first.created).toBe(true);
    expect(first.value).toEqual({
      id: wire.upload_id,
      projectId: PID,
      fileName: "tickets.csv",
      format: "csv",
      contentHash: wire.content_hash,
      sizeBytes: wire.size_bytes,
      rowCount: 8,
      columns: wire.columns.map((c: { name: string; type: string; nullable: boolean; null_count: number }) => ({
        name: c.name,
        type: c.type,
        nullable: c.nullable,
        nullCount: c.null_count,
      })),
      preview: wire.preview,
      parserVersion: wire.parser_version,
      createdAt: wire.created_at,
    });
    // the server, not the browser, saw the empty cell: customer_tier is nullable with one missing value
    expect(first.value.columns.find((c) => c.name === "customer_tier")).toEqual({ name: "customer_tier", type: "string", nullable: true, nullCount: 1 });
    // the content hash is sha256 of the exact bytes sent
    const sent = new TextEncoder().encode((exchange("upload_csv").request.body as { text: string }).text);
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", sent))].map((b) => b.toString(16).padStart(2, "0")).join("");
    expect(first.value.contentHash).toBe(digest);
    expect(new Headers(calls[0].init?.headers).get("Content-Type")).toBe("application/octet-stream");

    const again = await api.uploadDataset(PID, fileFrom("upload_csv"));
    expect(again.created).toBe(false);
    expect(again.value).toEqual(first.value);
    expect(await api.getUpload(UID)).toEqual(first.value);
  });

  it("reports JSONL types the server inferred, including json and string_list columns", async () => {
    const { api } = live("upload_jsonl");
    const { value } = await api.uploadDataset(PID, fileFrom("upload_jsonl", "jsonl"));
    expect(value.columns.map((c) => [c.name, c.type, c.nullable, c.nullCount])).toEqual(
      body("upload_jsonl").columns.map((c: { name: string; type: string; nullable: boolean; null_count: number }) => [
        c.name,
        c.type,
        c.nullable,
        c.null_count,
      ]),
    );
    expect(value.columns.find((c) => c.name === "tags")?.type).toBe("string_list");
    expect(value.columns.find((c) => c.name === "doc")?.type).toBe("json");
  });
});

describe("live adapter: registration, datasets and versions", () => {
  it("sends RegisterDataset exactly as the API expects and maps the version record", async () => {
    const { api, remaining } = live("register_v1", "register_v1_again", "register_v2", "list_datasets", "get_dataset", "get_version");
    const v1 = await api.registerDataset(UID, { datasetId: "support-tickets", name: "Support tickets", mapping: V1 });
    const wire = body("register_v1");
    expect(v1.created).toBe(true);
    expect(v1.value).toEqual({
      datasetId: "support-tickets",
      projectId: PID,
      uploadId: UID,
      version: 1,
      name: "Support tickets",
      format: "csv",
      contentHash: body("upload_csv").content_hash,
      identityHash: wire.identity_hash,
      rowCount: 8,
      columns: wire.spec.columns,
      mapping: V1,
      rowIdSource: "column",
      rowIdScheme: null,
      rowIdsHash: wire.row_ids_hash,
      createdAt: wire.created_at,
    });
    expect((await api.registerDataset(UID, { datasetId: "support-tickets", name: "Support tickets", mapping: V1 })).created).toBe(false);

    // no id column: row ids are generated by the server; a new version of the same dataset
    const v2 = await api.registerDataset(UID, { datasetId: "support-tickets", name: "Support tickets", mapping: V2 });
    expect(v2).toMatchObject({ created: true, value: { version: 2, rowIdSource: "generated", rowIdScheme: body("register_v2").row_id_scheme, mapping: V2 } });
    expect(v2.value.identityHash).not.toBe(v1.value.identityHash);

    const [listed] = await api.listDatasets(PID);
    expect(listed).toMatchObject({ id: "support-tickets", projectId: PID, name: "Support tickets", latestVersion: 2 });
    expect(listed.versions.map((v) => v.version)).toEqual([1, 2]);
    expect(listed.versions[0]).toEqual(v1.value);
    expect(await api.getDataset("support-tickets")).toEqual(listed);
    expect(await api.getDatasetVersion("support-tickets", 1)).toEqual(v1.value);
    expect(remaining).toEqual([]);
  });

  it("maps a version back to the very DatasetSpec the server stored (metadata aside)", () => {
    for (const name of ["register_v1", "register_v2"]) {
      const wire = body(name);
      expect(toDatasetSpec(fromDatasetVersion(wire))).toEqual({ ...wire.spec, metadata: {} });
    }
  });
});

describe("live adapter: splits", () => {
  it("creates, lists and gets deterministic splits with the server's sizes", async () => {
    const { api, remaining } = live("create_splits", "create_splits_again", "list_splits", "get_splits");
    const plan = { validationPct: 25, testPct: 25, seed: 7 };
    const created = await api.createSplits("support-tickets", 1, plan);
    const wire = body("create_splits");
    expect(created.created).toBe(true);
    expect(created.value).toEqual({
      datasetId: "support-tickets",
      version: 1,
      splitsHash: wire.splits_hash,
      datasetHash: body("register_v1").identity_hash,
      method: "seeded_hash/1",
      plan,
      sizes: { optimization: wire.sizes.optimization, validation: wire.sizes.validation, test: wire.sizes.test },
      createdAt: wire.created_at,
    });
    const again = await api.createSplits("support-tickets", 1, plan);
    expect(again).toEqual({ value: created.value, created: false });
    expect(await api.listSplits("support-tickets", 1)).toEqual([created.value]);
    expect(await api.getSplits("support-tickets", 1, wire.splits_hash)).toEqual(created.value);
    expect(remaining).toEqual([]);
  });
});

describe("live adapter: backend errors surface with their stable code and real reason", () => {
  const register = (mapping: Partial<ColumnMapping>, datasetId = "support-tickets") => ({
    datasetId,
    name: "Support tickets",
    mapping: { ...V1, ...mapping },
  });
  const cases: [string, (api: ReturnType<typeof createLiveApi>) => Promise<unknown>][] = [
    ["error_project_not_found", (a) => a.getProject("p-missing")],
    ["error_dataset_not_found", (a) => a.getDataset("missing")],
    ["error_splits_not_found", (a) => a.getSplits("support-tickets", 1, "0".repeat(64))],
    ["error_malformed_csv", (a) => a.uploadDataset(PID, fileFrom("error_malformed_csv"))],
    ["error_malformed_jsonl", (a) => a.uploadDataset(PID, fileFrom("error_malformed_jsonl", "jsonl"))],
    ["error_unsupported_format", (a) => a.uploadDataset(PID, fileFrom("error_unsupported_format", "parquet"))],
    ["error_empty_file", (a) => a.uploadDataset(PID, fileFrom("error_empty_file"))],
    ["error_invalid_column_name", (a) => a.uploadDataset(PID, fileFrom("error_invalid_column_name"))],
    ["error_duplicate_column", (a) => a.uploadDataset(PID, fileFrom("error_duplicate_column"))],
    ["error_payload_too_large", (a) => a.uploadDataset(PID, fileFrom("error_payload_too_large"))],
    ["error_unknown_column", (a) => a.registerDataset(UID, register({ input: ["subject", "missing"] }))],
    ["error_invalid_mapping", (a) => a.registerDataset(UID, register({ input: ["subject", "queue"] }))],
    ["error_invalid_id_column", (a) => a.registerDataset(UID, register({ context: [], id: "customer_tier" }))],
    ["error_invalid_request", (a) => a.registerDataset(UID, register({}, "Bad Id"))],
    [
      "error_json_column_role",
      (a) =>
        a.registerDataset(body("upload_jsonl").upload_id, {
          datasetId: "faq",
          name: "FAQ",
          mapping: { input: ["doc"], target: ["answer"], context: [], id: null },
        }),
    ],
    [
      "error_duplicate_row_id",
      (a) =>
        a.registerDataset(body("upload_duplicate_ids").upload_id, {
          datasetId: "dupes",
          name: "Dupes",
          mapping: { input: ["subject"], target: ["queue"], context: [], id: "ticket_id" },
        }),
    ],
    ["error_dataset_conflict", (a) => a.registerDataset(body("upload_other").upload_id, register({}))],
    ["error_split_plan", (a) => a.createSplits("support-tickets", 1, { validationPct: 50, testPct: 50, seed: 7 })],
    ["error_storage", (a) => a.getUpload(body("upload_duplicate_ids").upload_id)],
  ];

  it.each(cases)("%s", async (name, call) => {
    const { api } = live(name);
    const { code, message, details } = body(name).error;
    const err = await call(api).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(ApiError);
    const e = err as ApiError;
    expect({ code: e.code, message: e.message, status: e.status, details: e.details }).toEqual({
      code,
      message,
      status: exchange(name).response.status,
      details,
    });
    // a specific heading for every product API code, never a generic one
    expect(e.title).toBe(ERROR_COPY[code].title);
  });

  it("has a heading and kind for every stable code the product API defines", () => {
    expect(Object.keys(FIXTURE.error_status).filter((code) => !ERROR_COPY[code])).toEqual([]);
  });

  it("names the network failure when the API cannot be reached", async () => {
    const api = createLiveApi({ fetch: () => Promise.reject(new TypeError("Failed to fetch")) });
    await expect(api.listProjects()).rejects.toMatchObject({ code: "network_error", kind: "unavailable", message: expect.stringContaining("Failed to fetch") });
  });

  it("refuses a response that is not a product API error body (e.g. a proxy page) without guessing", async () => {
    const api = createLiveApi({ fetch: async () => new Response("<html>Bad gateway</html>", { status: 502 }) });
    await expect(api.listProjects()).rejects.toMatchObject({ code: "bad_response", status: 502 });
  });

  it("fails loudly when a response drifts from the contract", async () => {
    const withExtra = { ...body("get_project"), owner: "x" };
    const missing = { ...body("get_upload") };
    delete missing.row_count;
    const answer = (b: unknown) => createLiveApi({ fetch: async () => new Response(JSON.stringify(b), { status: 200 }) });
    await expect(answer(withExtra).getProject(PID)).rejects.toMatchObject({ code: "contract_mismatch", message: expect.stringMatching(/unexpected owner/) });
    await expect(answer(missing).getUpload(UID)).rejects.toMatchObject({ code: "contract_mismatch", message: expect.stringMatching(/missing row_count/) });
  });
});

describe("live adapter: experiments", () => {
  it("has no experiment backend: every experiment call fails without a request", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>();
    const api = createLiveApi({ fetch });
    expect(api.experiments).toBe("unavailable");
    for (const call of [
      () => api.listModels(),
      () => api.listExperiments(PID),
      () => api.getExperiment("e"),
      () => api.cancelExperiment("e"),
      () => api.getResults("e"),
      () => api.listWorkflows(PID),
    ])
      await expect(call()).rejects.toMatchObject({ code: "experiment_backend_unavailable", kind: "not_implemented", message: expect.stringMatching(/#20–#23/) });
    expect(fetch).not.toHaveBeenCalled();
  });
});
