import { afterEach, describe, expect, it, vi } from "vitest";
import { api, selectMode, setApi } from "./index";
import { createMockApi } from "./mock/mockApi";

describe("adapter selection", () => {
  afterEach(() => {
    setApi(null);
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    window.history.replaceState(null, "", "/");
  });

  it("uses the real product API unless mock mode is asked for", () => {
    expect(selectMode("", undefined)).toBe("live");
    expect(selectMode("?mockLatency=0", undefined)).toBe("live"); // mock knobs alone never switch modes
    expect(selectMode("?api=mock", undefined)).toBe("mock");
    expect(selectMode("", "mock")).toBe("mock");
    expect(selectMode("?api=live", "mock")).toBe("live");
    expect(selectMode("?api=", "")).toBe("live");
  });

  it("refuses an unknown mode instead of picking one silently", () => {
    expect(() => selectMode("?api=demo", undefined)).toThrow(/Unknown API mode "demo"/);
    expect(() => selectMode("", "staging")).toThrow(/Unknown API mode/);
  });

  it("loads the mock lazily when ?api=mock is set, and it never touches the network", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>();
    vi.stubGlobal("fetch", fetch);
    window.history.replaceState(null, "", "/projects?api=mock&mockLatency=0");
    expect(api().mode).toBe("mock");
    expect(api().experiments).toBe("simulated");
    const projects = await api().listProjects();
    expect(projects.map((p) => p.id)).toEqual(["p-support", "p-invoices"]);
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("mock mode stays isolated", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("runs the whole dataset flow and a simulated experiment without a single request", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>();
    vi.stubGlobal("fetch", fetch);
    const mock = createMockApi({ latencyMs: 0 });
    const p = await mock.createProject({ name: "Offline", description: "" });
    const csv = "id,text,label\n1,a,x\n2,b,y\n3,c,x\n4,d,y\n";
    const { value: upload } = await mock.uploadDataset(p.id, { name: "rows.csv", format: "csv", bytes: new Blob([csv]) });
    const { value: v } = await mock.registerDataset(upload.id, {
      datasetId: "rows",
      name: "Rows",
      mapping: { input: ["text"], target: ["label"], context: [], id: "id" },
    });
    await mock.createSplits("rows", v.version, { validationPct: 25, testPct: 25, seed: 1 });
    await mock.listExperiments("p-support");
    expect(fetch).not.toHaveBeenCalled();
  });
});
