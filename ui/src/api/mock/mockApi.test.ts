import { describe, expect, it } from "vitest";
import { ApiError } from "../client";
import type { ExperimentConfig } from "../types";
import { createMockApi } from "./mockApi";

function clock(start = Date.UTC(2026, 9, 5)) {
  let t = start;
  return { now: () => t, advance: (ms: number) => (t += ms) };
}

const config = (datasetId: string): ExperimentConfig => ({
  name: "Run",
  datasetId,
  taskType: "classification",
  instructions: "Label each row.",
  evaluation: { evaluator: "classification_accuracy", labels: ["billing", "technical", "account", "shipping"], caseSensitive: false },
  constraints: {
    minQuality: 0.8,
    maxCostPerExample: null,
    maxMeanLatencyS: null,
    maxP95LatencyS: null,
    maxTokensPerExample: null,
    maxWorkflowSteps: null,
  },
  preferences: { objective: "quality", balanced: null },
  models: ["gemma-3-27b-it", "gemma-4-31b-it"],
  budget: { maxCandidates: 40, maxGenerations: 6, maxSpendUsd: 30, maxDurationMin: 120 },
  splits: { validationPct: 20, testPct: 20, seed: 0 },
});

describe("mock adapter", () => {
  it("runs project → dataset → experiment → results, unlocking held-out test only at the end", async () => {
    const c = clock();
    const api = createMockApi({ latencyMs: 0, runMs: 10_000, now: c.now });
    const p = await api.createProject({ name: "Churn", description: "" });
    const csv = ["row_id,text,label", ...Array.from({ length: 500 }, (_, i) => `${i},text ${i},${i % 2 ? "billing" : "other"}`)].join("\n");
    const { value: upload, created } = await api.uploadDataset(p.id, { name: "rows.csv", format: "csv", bytes: new Blob([csv]) });
    expect(created).toBe(true);
    expect(upload.rowCount).toBe(500);
    expect(upload.contentHash).toMatch(/^[0-9a-f]{64}$/);
    expect(upload.columns.map((col) => col.type)).toEqual(["integer", "string", "string"]);
    const { value: d } = await api.registerDataset(upload.id, {
      datasetId: "rows",
      name: "rows",
      mapping: { input: ["text"], target: ["label"], context: [], id: "row_id" },
    });
    expect(d.version).toBe(1);
    expect((await api.getProject(p.id)).datasetCount).toBe(1);

    const e = await api.createExperiment(p.id, {
      ...config(d.datasetId),
      evaluation: { evaluator: "classification_accuracy", labels: ["billing", "other"], caseSensitive: false },
    });
    expect(e.status).toBe("queued");
    expect(e.identityHash).toMatch(/^[0-9a-f]{64}$/);
    expect((await api.getResults(e.id)).testLocked).toBe(true);

    c.advance(4000);
    const running = await api.getExperiment(e.id);
    expect(running.status).toBe("running");
    expect(running.championId).toBeNull();
    const mid = await api.getResults(e.id);
    expect(mid.testLocked).toBe(true);
    expect(mid.methods.every((m) => m.splits.test === undefined)).toBe(true);

    c.advance(10_000);
    const done = await api.getExperiment(e.id);
    expect(done.status).toBe("completed");
    expect(done.championId).not.toBeNull();
    const final = await api.getResults(e.id);
    expect(final.testLocked).toBe(false);
    expect(final.methods.map((m) => m.method)).toEqual(["fixed_baseline", "random_search", "wynk_aco"]);
    final.methods.forEach((m) => expect(m.splits.test?.measurement.n).toBe(final.splitSizes.test));
    // held-out test is reported, and never folded into any candidate's state
    expect(done.candidates.every((cand) => !("test" in cand))).toBe(true);
    expect(final.splitSizes).toEqual({ optimization: 300, validation: 100, test: 100 });

    const workflows = await api.listWorkflows(p.id);
    expect(workflows.filter((w) => w.candidate.state === "champion")).toHaveLength(1);
  });

  it("refuses an invalid configuration instead of pretending", async () => {
    const api = createMockApi({ latencyMs: 0 });
    await expect(api.createExperiment("p-support", { ...config("d-tickets"), splits: { validationPct: 50, testPct: 45, seed: 0 } })).rejects.toThrow(
      /at least 10%/,
    );
    await expect(
      api.createExperiment("p-support", {
        ...config("d-tickets"),
        constraints: { ...config("d").constraints, minQuality: null },
        preferences: { objective: "cost", balanced: null },
      }),
    ).rejects.toThrow(/needs a minimum quality/);
    await expect(api.createExperiment("p-invoices", { ...config("d-invoices") })).rejects.toThrow(/exactly one text target/);
    await expect(api.getExperiment("nope")).rejects.toMatchObject({ code: "not_found" });
  });

  it("cancels a running experiment and keeps its partial results without a champion", async () => {
    const c = clock();
    const api = createMockApi({ latencyMs: 0, runMs: 10_000, now: c.now });
    const e = await api.createExperiment("p-support", config("d-tickets"));
    c.advance(5000);
    const cancelled = await api.cancelExperiment(e.id);
    expect(cancelled.status).toBe("cancelled");
    c.advance(60_000);
    const later = await api.getExperiment(e.id);
    expect(later.status).toBe("cancelled");
    expect(later.championId).toBeNull();
    expect(later.progress.evaluated).toBe(cancelled.progress.evaluated);
    await expect(api.cancelExperiment(e.id)).rejects.toThrow(/Only a queued or running/);
  });

  it("serves the seeded failed experiment as failed, with its reason", async () => {
    const api = createMockApi({ latencyMs: 0 });
    const e = await api.getExperiment("e-invoices-1");
    expect(e.status).toBe("failed");
    expect(e.failure).toMatch(/HTTP 503/);
    expect(e.championId).toBeNull();
  });

  it("registers a new version when an upload is registered with different roles, and reuses an identical one", async () => {
    const api = createMockApi({ latencyMs: 0 });
    const v1 = (await api.getDataset("d-faq")).versions[0];
    const roles = { input: ["question"], target: ["answer"], context: ["passage"], id: "faq_id" };
    const v2 = await api.registerDataset(v1.uploadId, { datasetId: "d-faq", name: v1.name, mapping: roles });
    expect(v2).toMatchObject({ created: true, value: { version: 2, rowIdSource: "column" } });
    const again = await api.registerDataset(v1.uploadId, { datasetId: "d-faq", name: v1.name, mapping: roles });
    expect(again).toMatchObject({ created: false, value: { version: 2 } });
    expect((await api.getDataset("d-faq")).latestVersion).toBe(2);
  });

  it("uses the product API's error codes for the dataset flow", async () => {
    const api = createMockApi({ latencyMs: 0 });
    const file = (text: string, format = "csv") => ({ name: `f.${format}`, format, bytes: new Blob([text]) });
    await expect(api.uploadDataset("p-support", file("a\n1\n", "parquet"))).rejects.toMatchObject({ code: "unsupported_format" });
    await expect(api.uploadDataset("p-support", file("a,b\n1\n"))).rejects.toMatchObject({ code: "malformed_csv" });
    await expect(api.uploadDataset("p-nope", file("a\n1\n"))).rejects.toMatchObject({ code: "project_not_found" });
    const { value: u } = await api.uploadDataset("p-support", file("id,text,label\n1,a,x\n1,b,y\n"));
    const reg = (mapping: Partial<{ input: string[]; target: string[]; context: string[]; id: string | null }>, datasetId = "new-set") =>
      api.registerDataset(u.id, { datasetId, name: "n", mapping: { input: ["text"], target: ["label"], context: [], id: null, ...mapping } });
    await expect(reg({ input: ["nope"] })).rejects.toMatchObject({ code: "unknown_column" });
    await expect(reg({ input: ["text", "label"] })).rejects.toMatchObject({ code: "invalid_mapping" });
    await expect(reg({ id: "id" })).rejects.toMatchObject({ code: "duplicate_row_id" });
    await expect(reg({}, "Bad Id")).rejects.toMatchObject({ code: "invalid_request" });
    await expect(reg({}, "d-invoices")).rejects.toMatchObject({ code: "dataset_conflict" });
    const { value: v } = await reg({});
    expect(v.rowIdSource).toBe("generated");
    await expect(api.createSplits("new-set", 1, { validationPct: 50, testPct: 50, seed: 0 })).rejects.toMatchObject({ code: "invalid_request" });
    const splits = await api.createSplits("new-set", 1, { validationPct: 0, testPct: 50, seed: 0 });
    expect(splits.value.sizes).toEqual({ optimization: 1, validation: 0, test: 1 });
    expect((await api.createSplits("new-set", 1, { validationPct: 0, testPct: 50, seed: 0 })).created).toBe(false);
    await expect(api.getSplits("new-set", 1, "0".repeat(64))).rejects.toMatchObject({ code: "splits_not_found" });
  });

  it("serves fixtures that are valid contract instances", async () => {
    const api = createMockApi({ latencyMs: 0 });
    for (const p of await api.listProjects())
      for (const d of await api.listDatasets(p.id))
        for (const v of d.versions) {
          expect(["csv", "jsonl"]).toContain(v.format);
          expect(v.contentHash).toMatch(/^[0-9a-f]{64}$/);
          v.columns.forEach((c) => expect(c.name).toMatch(/^[A-Za-z_][A-Za-z0-9_]{0,63}$/));
        }
  });

  it("injects failures on request, for error states", async () => {
    const api = createMockApi({ latencyMs: 0, failOn: ["listProjects"] });
    await expect(api.listProjects()).rejects.toBeInstanceOf(ApiError);
    await expect(api.listModels()).resolves.toHaveLength(4);
  });

  it("returns copies, so screens cannot mutate adapter state", async () => {
    const api = createMockApi({ latencyMs: 0 });
    const [first] = await api.listProjects();
    first.name = "changed";
    expect((await api.listProjects())[0].name).not.toBe("changed");
  });
});
