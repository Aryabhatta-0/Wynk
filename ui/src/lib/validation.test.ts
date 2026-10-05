import { describe, expect, it } from "vitest";
import type { ExperimentConfig } from "@/api";
import { validateConfig, validateMapping } from "./validation";

describe("validateMapping", () => {
  const cols = ["a", "b", "c"];
  it("accepts inputs plus one target", () => {
    expect(validateMapping({ input: ["a"], target: "b", context: ["c"] }, cols)).toEqual([]);
  });
  it("requires an input and a target", () => {
    expect(validateMapping({ input: [], target: null, context: [] }, cols)).toEqual([
      "Choose at least one input column.",
      "Choose the target column the workflow must produce.",
    ]);
  });
  it("rejects overlaps and unknown columns", () => {
    expect(validateMapping({ input: ["a", "b"], target: "b", context: ["a"] }, cols)).toEqual([
      "The target column cannot also be an input or context column.",
      "a cannot be both input and context.",
    ]);
    expect(validateMapping({ input: ["z"], target: "a", context: [] }, cols)[0]).toBe("Unknown column: z.");
  });
});

const valid: ExperimentConfig = {
  name: "x",
  datasetId: "d",
  taskType: "classification",
  constraints: { minQuality: 0.8, maxCostPer1k: null, maxLatencyP95Ms: 5000, allowedModels: ["m"] },
  preferences: { objective: "quality" },
  budget: { maxCandidates: 40, maxGenerations: 5, maxSpendUsd: 10, maxDurationMin: 30 },
  splits: { optimization: 60, validation: 20, test: 20 },
};

describe("validateConfig", () => {
  it("accepts a complete configuration with optional limits left empty", () => {
    expect(validateConfig(valid)).toEqual({});
  });

  it("flags each broken field", () => {
    const e = validateConfig({
      ...valid,
      name: " ",
      constraints: { minQuality: 1.5, maxCostPer1k: -1, maxLatencyP95Ms: 0, allowedModels: [] },
      budget: { maxCandidates: 4, maxGenerations: 6, maxSpendUsd: 0, maxDurationMin: Number.NaN },
      splits: { optimization: 70, validation: 20, test: 20 },
    });
    expect(Object.keys(e).sort()).toEqual(
      ["allowedModels", "maxCostPer1k", "maxDurationMin", "maxGenerations", "maxLatencyP95Ms", "maxSpendUsd", "minQuality", "name", "splits"].sort(),
    );
    expect(e.maxGenerations).toMatch(/cannot exceed candidates/);
    expect(e.splits).toBe("Splits must add up to 100%.");
  });

  it("requires every split to keep some rows", () => {
    expect(validateConfig({ ...valid, splits: { optimization: 98, validation: 2, test: 0 } }).splits).toMatch(/at least 5%/);
  });
});
