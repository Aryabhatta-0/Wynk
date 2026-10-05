import { describe, expect, it } from "vitest";
import type { Measurement, MethodResult, ResultsComparison } from "@/api";
import { whyFacts } from "./why";

const m = (quality: number, costPer1k: number, latencyP95Ms: number): Measurement => ({ quality, costPer1k, latencyP95Ms, n: 100 });

const method = (id: MethodResult["method"], splits: MethodResult["splits"], ok = true): MethodResult => ({
  method: id,
  stages: [{ kind: "EXTRACT", options: ["direct"] }],
  model: "m",
  evaluated: id === "fixed_baseline" ? 1 : 48,
  splits,
  constraints: [
    { key: "minQuality", ok, observed: 0.9, limit: 0.85 },
    { key: "allowedModels", ok: true, observed: "m", limit: "m" },
  ],
});

const results = (testLocked: boolean, wynkOk = true): ResultsComparison => ({
  experimentId: "e",
  splitSizes: { optimization: 600, validation: 200, test: 200 },
  testLocked,
  methods: [
    method("fixed_baseline", {
      optimization: m(0.8, 0.2, 2000),
      validation: m(0.79, 0.2, 2000),
      ...(testLocked ? {} : { test: m(0.78, 0.2, 2000) }),
    }),
    method("random_search", {
      optimization: m(0.86, 0.3, 3000),
      validation: m(0.85, 0.3, 3000),
      ...(testLocked ? {} : { test: m(0.84, 0.3, 3000) }),
    }),
    method(
      "wynk_aco",
      { optimization: m(0.92, 0.3, 1800), validation: m(0.9, 0.3, 1800), ...(testLocked ? {} : { test: m(0.9, 0.3, 1800) }) },
      wynkOk,
    ),
  ],
});

describe("whyFacts", () => {
  it("states held-out differences against both baselines, with signs and tones from the numbers", () => {
    const facts = whyFacts({ results: results(false), taskType: "classification", objective: "quality", validatedCount: 5 });
    const byId = Object.fromEntries(facts.map((f) => [f.id, f]));
    expect(byId["quality-fixed_baseline"].text).toBe("Held-out test accuracy: 90.0% vs 78.0% for fixed baseline (+12.0 pts).");
    expect(byId["quality-fixed_baseline"].tone).toBe("better");
    expect(byId["quality-random_search"].text).toContain("(+6.0 pts)");
    // more expensive than the baseline is reported as worse, not hidden
    expect(byId.cost.text).toBe("Cost: $0.300 per 1,000 examples vs $0.200 for the fixed baseline (+50%).");
    expect(byId.cost.tone).toBe("worse");
    expect(byId.latency.tone).toBe("better");
    expect(byId.constraints).toEqual({ id: "constraints", text: "Met all 2 hard constraints on the held-out test split.", tone: "better" });
    expect(byId.selection.text).toBe("Selected from 5 validated workflows by the quality objective, after 48 candidates were evaluated.");
    expect(byId.generalization.text).toBe("Accuracy went from 92.0% on the optimization split to 90.0% on the held-out test split (−2.0 pts).");
  });

  it("falls back to validation while the test split is locked", () => {
    const facts = whyFacts({ results: results(true), taskType: "structured_extraction", objective: "cost", validatedCount: 1 });
    expect(facts.find((f) => f.id === "quality-fixed_baseline")!.text).toMatch(/^Validation field match: 90.0% vs 79.0%/);
    expect(facts.find((f) => f.id === "constraints")!.text).toContain("validation split");
    expect(facts.find((f) => f.id === "selection")!.text).toContain("1 validated workflow by the cost objective");
  });

  it("reports broken constraints instead of claiming success", () => {
    const facts = whyFacts({ results: results(false, false), taskType: "classification", objective: "quality", validatedCount: 5 });
    expect(facts.find((f) => f.id === "constraints")).toEqual({
      id: "constraints",
      text: "Broke minimum quality on the held-out test split.",
      tone: "worse",
    });
  });

  it("says nothing when there is no Wynk result", () => {
    const r = results(false);
    r.methods = r.methods.filter((x) => x.method !== "wynk_aco");
    expect(whyFacts({ results: r, taskType: "classification", objective: "quality", validatedCount: 0 })).toEqual([]);
  });

  it("never uses causal wording", () => {
    const text = whyFacts({ results: results(false), taskType: "question_answering", objective: "balanced", validatedCount: 3 })
      .map((f) => f.text)
      .join(" ");
    expect(text).not.toMatch(/\b(because|due to|thanks to|caused|helps)\b/i);
  });
});
