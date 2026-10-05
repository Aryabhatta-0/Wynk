import { describe, expect, it } from "vitest";
import { ApiError } from "../client";
import { createLiveApi } from "../live";
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
  constraints: { minQuality: 0.8, maxCostPer1k: null, maxLatencyP95Ms: null, allowedModels: ["gemma-3-27b-it", "gemma-4-31b-it"] },
  preferences: { objective: "quality" },
  budget: { maxCandidates: 40, maxGenerations: 6, maxSpendUsd: 30, maxDurationMin: 120 },
  splits: { optimization: 60, validation: 20, test: 20 },
});

describe("mock adapter", () => {
  it("runs project → dataset → experiment → results, unlocking held-out test only at the end", async () => {
    const c = clock();
    const api = createMockApi({ latencyMs: 0, runMs: 10_000, now: c.now });
    const p = await api.createProject({ name: "Churn", description: "" });
    const d = await api.createDataset(p.id, {
      name: "rows",
      format: "csv",
      fileName: "rows.csv",
      sizeBytes: 100,
      rowCount: 500,
      columns: [
        { name: "text", type: "string", missing: 0, distinct: 2 },
        { name: "label", type: "string", missing: 0, distinct: 2 },
      ],
      preview: [{ text: "a", label: "x" }],
      mapping: { input: ["text"], target: "label", context: [] },
    });
    expect(d.status).toBe("ready");
    expect((await api.getProject(p.id)).datasetCount).toBe(1);

    const e = await api.createExperiment(p.id, config(d.id));
    expect(e.status).toBe("queued");
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
    final.methods.forEach((m) => expect(m.splits.test?.n).toBe(final.splitSizes.test));
    expect(final.splitSizes).toEqual({ optimization: 300, validation: 100, test: 100 });

    const workflows = await api.listWorkflows(p.id);
    expect(workflows.filter((w) => w.candidate.state === "champion")).toHaveLength(1);
  });

  it("refuses an invalid configuration or an unmapped dataset instead of pretending", async () => {
    const api = createMockApi({ latencyMs: 0 });
    await expect(
      api.createExperiment("p-support", { ...config("d-tickets"), splits: { optimization: 50, validation: 20, test: 20 } }),
    ).rejects.toThrow("Splits must add up to 100%.");
    await expect(api.createExperiment("p-support", config("d-faq"))).rejects.toThrow(/column mapping/);
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

describe("live adapter", () => {
  it("fails every call as not connected rather than inventing an API", async () => {
    const api = createLiveApi();
    expect(api.mode).toBe("live");
    await expect(api.listProjects()).rejects.toMatchObject({ code: "not_connected" });
    await expect(api.createExperiment("p", config("d"))).rejects.toMatchObject({ code: "not_connected" });
  });
});
