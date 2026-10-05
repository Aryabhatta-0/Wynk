import { describe, expect, it } from "vitest";
import type { Dataset, DatasetColumn, ExperimentConfig } from "@/api";
import { validateConfig, validateMapping } from "./validation";

const col = (name: string, type: DatasetColumn["type"] = "string", nullable = false): DatasetColumn => ({
  name,
  type,
  nullable,
  missing: nullable ? 1 : 0,
  distinct: 3,
});

describe("validateMapping (DatasetSpec role rules)", () => {
  const cols = [col("id"), col("text"), col("label"), col("notes", "string", true), col("meta", "json"), col("bad name")];

  it("accepts inputs, targets, context and an id", () => {
    expect(validateMapping({ input: ["text"], target: ["label"], context: ["notes"], id: "id" }, cols)).toEqual([]);
  });

  it("requires an input, a target and (to split) an id", () => {
    expect(validateMapping({ input: [], target: [], context: [], id: null }, cols)).toEqual([
      "Choose at least one input column.",
      "Choose at least one target column for the workflow to produce.",
      "Choose an id column: rows are split into optimization, validation and test by their id.",
    ]);
  });

  it("keeps roles disjoint", () => {
    expect(validateMapping({ input: ["text", "label"], target: ["label"], context: [], id: "id" }, cols)).toContain("label can have only one role.");
  });

  it("refuses nested JSON, non-identifier names, unknown columns and a nullable id", () => {
    const problems = validateMapping({ input: ["meta", "bad name", "ghost"], target: ["label"], context: [], id: "notes" }, cols);
    expect(problems).toContain("Unknown column: ghost.");
    expect(problems.some((p) => p.startsWith("Rename bad name"))).toBe(true);
    expect(problems).toContain("meta holds nested JSON, which cannot be mapped. Split it into one column per field.");
    expect(problems).toContain("The id column must be a string or integer column with a value in every row.");
  });
});

const dataset = (targets: DatasetColumn[]): Dataset => ({
  id: "d-1",
  projectId: "p",
  name: "d",
  format: "csv",
  fileName: "d.csv",
  sizeBytes: 10,
  contentHash: "a".repeat(64),
  version: 1,
  rowCount: 100,
  createdAt: "2026-01-01T00:00:00Z",
  status: "ready",
  columns: [col("id"), col("text"), ...targets],
  preview: [],
  mapping: { input: ["text"], target: targets.map((t) => t.name), context: [], id: "id" },
});

const valid: ExperimentConfig = {
  name: "x",
  datasetId: "d-1",
  taskType: "classification",
  instructions: "Label the text.",
  evaluation: { evaluator: "classification_accuracy", labels: ["a", "b"], caseSensitive: false },
  constraints: {
    minQuality: 0.8,
    maxCostPerExample: null,
    maxMeanLatencyS: null,
    maxP95LatencyS: 5,
    maxTokensPerExample: null,
    maxWorkflowSteps: null,
  },
  preferences: { objective: "quality", balanced: null },
  models: ["m"],
  budget: { maxCandidates: 40, maxGenerations: 5, maxSpendUsd: 10, maxDurationMin: 30 },
  splits: { validationPct: 20, testPct: 20, seed: 0 },
};

describe("validateConfig (TaskContract, ObjectiveSpec, ConstraintLimits, SplitPlan rules)", () => {
  const ds = dataset([col("label")]);

  it("accepts a complete configuration with optional limits left empty", () => {
    expect(validateConfig(valid, ds)).toEqual({});
  });

  it("checks the target shape and evaluator of each task type", () => {
    expect(validateConfig({ ...valid, taskType: "classification" }, dataset([col("a"), col("b")])).taskType).toMatch(/exactly one text target/);
    expect(validateConfig({ ...valid, taskType: "question_answering" }, dataset([col("flag", "boolean")])).taskType).toMatch(/text or numeric/);
    expect(
      validateConfig(
        { ...valid, taskType: "question_answering", evaluation: { evaluator: "token_f1", passThreshold: 0.8 } },
        dataset([col("n", "number")]),
      ).evaluation,
    ).toMatch(/cannot judge/);
    expect(
      validateConfig(
        { ...valid, taskType: "structured_extraction", evaluation: { evaluator: "numeric_tolerance", absoluteTolerance: 0, relativeTolerance: 0 } },
        dataset([col("total", "number"), col("vendor")]),
      ).evaluation,
    ).toMatch(/cannot judge/);
  });

  it("checks evaluator options", () => {
    expect(
      validateConfig({ ...valid, evaluation: { evaluator: "classification_accuracy", labels: ["a"], caseSensitive: false } }, ds).evaluation,
    ).toBe("List at least two labels.");
    expect(
      validateConfig({ ...valid, evaluation: { evaluator: "classification_accuracy", labels: ["a", "a"], caseSensitive: false } }, ds).evaluation,
    ).toBe("Labels must be unique.");
  });

  it("requires instructions and at least one model", () => {
    const e = validateConfig({ ...valid, instructions: " ", models: [] }, ds);
    expect(e.instructions).toBeDefined();
    expect(e.models).toBe("Allow at least one model.");
  });

  it("requires a quality floor for minimizing cost or latency", () => {
    const noFloor = { ...valid.constraints, minQuality: null };
    expect(validateConfig({ ...valid, constraints: noFloor, preferences: { objective: "cost", balanced: null } }, ds).minQuality).toMatch(
      /needs a minimum quality/,
    );
    expect(validateConfig({ ...valid, constraints: noFloor, preferences: { objective: "latency", balanced: null } }, ds).minQuality).toMatch(
      /needs a minimum quality/,
    );
    expect(validateConfig({ ...valid, constraints: noFloor }, ds).minQuality).toBeUndefined();
  });

  it("checks balanced weights: whole %, sum 100, positive quality, a scale per penalized metric", () => {
    const balanced = (b: Partial<NonNullable<ExperimentConfig["preferences"]["balanced"]>>) =>
      validateConfig(
        {
          ...valid,
          preferences: {
            objective: "balanced",
            balanced: { quality: 70, cost: 20, latency: 10, tokens: 0, costScale: 0.0005, latencyScale: 2, tokensScale: null, ...b },
          },
        },
        ds,
      ).objective;
    expect(balanced({})).toBeUndefined();
    expect(balanced({ quality: 60 })).toBe("Weights must add up to 100%.");
    expect(balanced({ quality: 0, cost: 90 })).toBe("Quality needs a weight above 0.");
    expect(balanced({ latencyScale: null })).toMatch(/needs a scale/);
    expect(balanced({ tokens: 10, quality: 60, tokensScale: null })).toMatch(/needs a scale/);
  });

  it("checks limit ranges in contract units", () => {
    const e = validateConfig(
      {
        ...valid,
        constraints: {
          minQuality: 1.5,
          maxCostPerExample: -1,
          maxMeanLatencyS: 0,
          maxP95LatencyS: -2,
          maxTokensPerExample: 1.5,
          maxWorkflowSteps: 0,
        },
      },
      ds,
    );
    expect(Object.keys(e).sort()).toEqual([
      "maxCostPerExample",
      "maxMeanLatencyS",
      "maxP95LatencyS",
      "maxTokensPerExample",
      "maxWorkflowSteps",
      "minQuality",
    ]);
    // a zero cost ceiling is legal (NonNegativeFloat); zero latency is not (PositiveFloat)
    expect(validateConfig({ ...valid, constraints: { ...valid.constraints, maxCostPerExample: 0 } }, ds).maxCostPerExample).toBeUndefined();
  });

  it("checks the split plan and the search budget", () => {
    expect(validateConfig({ ...valid, splits: { validationPct: 2, testPct: 20, seed: 0 } }, ds).splits).toMatch(/5% or more/);
    expect(validateConfig({ ...valid, splits: { validationPct: 50, testPct: 45, seed: 0 } }, ds).splits).toMatch(/at least 10%/);
    expect(validateConfig({ ...valid, splits: { validationPct: 20, testPct: 20, seed: -1 } }, ds).splits).toMatch(/seed/);
    expect(validateConfig({ ...valid, budget: { ...valid.budget, maxGenerations: 50 } }, ds).maxGenerations).toMatch(/cannot exceed candidates/);
  });
});
