import { describe, expect, it } from "vitest";
import type { ConstraintCheck, EvaluationConfig, Measurement, MethodResult, ResultsComparison } from "@/api";
import { whyFacts } from "./why";

const m = (quality: number, costPerExample: number, meanLatencyS: number): Measurement => ({
  quality,
  costPerExample,
  meanLatencyS,
  p95LatencyS: meanLatencyS * 1.4,
  meanTokensPerExample: 900,
  maxTokensPerExample: 1500,
  workflowSteps: 3,
  n: 100,
});

const checks = (ok: boolean): ConstraintCheck[] => [
  { key: "minQuality", ok, observed: ok ? 0.9 : 0.8, limit: 0.85 },
  { key: "maxP95LatencyS", ok: true, observed: 2.5, limit: 5 },
];
const split = (x: Measurement, ok = true) => ({ measurement: x, constraints: checks(ok) });

const method = (id: MethodResult["method"], splits: MethodResult["splits"]): MethodResult => ({
  method: id,
  stages: [{ kind: "EXTRACT", options: ["direct"] }],
  model: "m",
  evaluated: id === "fixed_baseline" ? 1 : 48,
  splits,
});

const results = (testLocked: boolean, wynkOkOnTest = true): ResultsComparison => ({
  experimentId: "e",
  splitSizes: { optimization: 600, validation: 200, test: 200 },
  testLocked,
  methods: [
    method("fixed_baseline", {
      optimization: split(m(0.8, 0.0002, 2.4)),
      validation: split(m(0.79, 0.0002, 2.4)),
      ...(testLocked ? {} : { test: split(m(0.78, 0.0002, 2.4)) }),
    }),
    method("random_search", {
      optimization: split(m(0.86, 0.0003, 3)),
      validation: split(m(0.85, 0.0003, 3)),
      ...(testLocked ? {} : { test: split(m(0.84, 0.0003, 3)) }),
    }),
    method("wynk_aco", {
      optimization: split(m(0.92, 0.0003, 1.8)),
      validation: split(m(0.9, 0.0003, 1.8)),
      ...(testLocked ? {} : { test: split(m(0.9, 0.0003, 1.8), wynkOkOnTest) }),
    }),
  ],
});

const accuracy: EvaluationConfig = { evaluator: "classification_accuracy", labels: ["a", "b"], caseSensitive: false };

describe("whyFacts", () => {
  it("states held-out differences against both baselines, with signs and tones from the numbers", () => {
    const facts = whyFacts({ results: results(false), evaluation: accuracy, objective: "quality", validatedCount: 5 });
    const byId = Object.fromEntries(facts.map((f) => [f.id, f]));
    expect(byId["quality-fixed_baseline"].text).toBe("Held-out test accuracy: 90.0% vs 78.0% for fixed baseline (+12.0 pts).");
    expect(byId["quality-fixed_baseline"].tone).toBe("better");
    expect(byId["quality-random_search"].text).toContain("(+6.0 pts)");
    // more expensive than the baseline is reported as worse, not hidden
    expect(byId.cost.text).toBe("Cost: $0.300 / 1k examples vs $0.200 / 1k for the fixed baseline (+50%).");
    expect(byId.cost.tone).toBe("worse");
    expect(byId.latency.text).toBe("Mean latency: 1.8 s vs 2.4 s for the fixed baseline (−25%).");
    expect(byId.latency.tone).toBe("better");
    expect(byId.constraints).toEqual({ id: "constraints", text: "Met all 2 hard constraints on the held-out test split.", tone: "better" });
    expect(byId.selection.text).toBe(
      "Selected on the validation split from 5 validated workflows by the quality objective, after 48 candidates were evaluated.",
    );
    expect(byId.generalization.text).toBe("Accuracy went from 92.0% on the optimization split to 90.0% on the held-out test split (−2.0 pts).");
  });

  it("falls back to validation while the test split is locked, and names the evaluator's metric", () => {
    const facts = whyFacts({
      results: results(true),
      evaluation: { evaluator: "exact_match", caseSensitive: false, normalizeWhitespace: true },
      objective: "cost",
      validatedCount: 1,
    });
    expect(facts.find((f) => f.id === "quality-fixed_baseline")!.text).toMatch(/^Validation exact match: 90.0% vs 79.0%/);
    expect(facts.find((f) => f.id === "constraints")!.text).toContain("validation split");
    expect(facts.find((f) => f.id === "selection")!.text).toContain("1 validated workflow by the cost objective");
  });

  it("reports a constraint the champion broke on held-out test instead of claiming success", () => {
    const facts = whyFacts({ results: results(false, false), evaluation: accuracy, objective: "quality", validatedCount: 5 });
    expect(facts.find((f) => f.id === "constraints")).toEqual({
      id: "constraints",
      text: "Broke minimum quality on the held-out test split.",
      tone: "worse",
    });
  });

  it("says nothing about a metric that was not measured", () => {
    const r = results(false);
    const wynk = r.methods.find((x) => x.method === "wynk_aco")!;
    wynk.splits.test!.measurement = { ...wynk.splits.test!.measurement, costPerExample: null, meanLatencyS: null };
    const ids = whyFacts({ results: r, evaluation: accuracy, objective: "quality", validatedCount: 5 }).map((f) => f.id);
    expect(ids).not.toContain("cost");
    expect(ids).not.toContain("latency");
  });

  it("says nothing when there is no Wynk result", () => {
    const r = results(false);
    r.methods = r.methods.filter((x) => x.method !== "wynk_aco");
    expect(whyFacts({ results: r, evaluation: accuracy, objective: "quality", validatedCount: 0 })).toEqual([]);
  });

  it("never uses causal wording", () => {
    const text = whyFacts({
      results: results(false),
      evaluation: { evaluator: "token_f1", passThreshold: 0.8 },
      objective: "balanced",
      validatedCount: 3,
    })
      .map((f) => f.text)
      .join(" ");
    expect(text).not.toMatch(/\b(because|due to|thanks to|caused|helps)\b/i);
  });
});
