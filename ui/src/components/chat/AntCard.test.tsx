import { describe, expect, it } from "vitest";
import { antLabel } from "./AntCard";

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
