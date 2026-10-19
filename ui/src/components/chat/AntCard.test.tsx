import { describe, expect, it } from "vitest";
import { antLabel, optionsLabel } from "./AntCard";

describe("antLabel (#10: which recorded workflow an elite ant is)", () => {
  it("names the routed template task for a task elite", () => {
    expect(antLabel({ role: "elite", elite: "task", elite_task: "A-001" })).toBe("Task elite: best validated workflow for A-001");
  });

  it("falls back to the suite elite, including for older servers that send no elite kind", () => {
    expect(antLabel({ role: "elite", elite: "suite" })).toBe("Suite elite: best recorded workflow");
    expect(antLabel({ role: "elite" })).toBe("Suite elite: best recorded workflow");
  });

  it("labels explorers as before", () => {
    expect(antLabel({ role: "explorer" })).toBe("Explorer: sampled from the trail");
  });
});

describe("optionsLabel (#12: a GATHER stage names its evidence source)", () => {
  it("shows live web search distinctly from the frozen sources", () => {
    const gather = { kind: "GATHER" as const, options: ["websearch", "sequential"] };
    expect(optionsLabel(gather, "websearch")).toBe("live web search, sequential");
    expect(optionsLabel({ kind: "GATHER", options: ["fetch", "parallel-4"] }, "fetch")).toBe("frozen pages, parallel-4");
    expect(optionsLabel({ kind: "GATHER", options: ["jev", "sequential"] }, "jev")).toBe("JEV browsing, sequential");
    expect(optionsLabel({ kind: "EXTRACT", options: ["schema_guided"] }, undefined)).toBe("schema_guided");
  });
});
