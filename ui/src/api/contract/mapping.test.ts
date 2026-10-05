import { describe, expect, it } from "vitest";
import { DATASETS, EXPERIMENTS } from "../mock/fixtures";
import type { Dataset, OptimizationPreferences } from "../types";
import {
  ContractMappingError,
  fromMeasurements,
  toConstraintLimits,
  toDatasetSpec,
  toEvaluationSpec,
  toObjectiveSpec,
  toSplitPlan,
  toTaskContract,
} from "./mapping";
import { SPLIT_USES, evaluatorsFor, splitSizes } from "./rules";

const dataset = (id: string) => structuredClone(DATASETS.find((d) => d.id === id)!) as Dataset;

describe("toDatasetSpec (core/dataset.py DatasetSpec)", () => {
  it("maps roles, types, nullability and identity fields", () => {
    const spec = toDatasetSpec(dataset("d-tickets"));
    expect(spec).toMatchObject({
      schema_version: "datasetspec/1",
      dataset_id: "d-tickets",
      dataset_version: 1,
      format: "csv",
      id_column: "ticket_id",
      input_columns: ["subject", "body"],
      target_columns: ["queue"],
      context_columns: ["customer_tier"],
      row_count: 2400,
    });
    expect(spec.content_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(spec.columns.find((c) => c.name === "customer_tier")).toEqual({ name: "customer_tier", type: "string", nullable: true });
  });

  it("refuses a dataset the contract would reject", () => {
    const d = dataset("d-tickets");
    expect(() => toDatasetSpec({ ...d, contentHash: null })).toThrow(ContractMappingError);
    expect(() => toDatasetSpec({ ...d, mapping: { ...d.mapping, target: [] } })).toThrow(/target/);
  });
});

describe("toSplitPlan (core/dataset.py SplitPlan)", () => {
  it("turns percentages into basis points; optimization gets the rest", () => {
    expect(toSplitPlan({ validationPct: 20, testPct: 15, seed: 7 })).toEqual({ seed: 7, validation_bps: 2000, test_bps: 1500 });
    expect(splitSizes(10, { validationPct: 25, testPct: 25, seed: 0 })).toEqual({ optimization: 6, validation: 2, test: 2 });
    expect(() => toSplitPlan({ validationPct: 60, testPct: 40, seed: 0 })).toThrow(/leave rows/);
  });

  it("mirrors ALLOWED_USES: only optimization feeds the optimizer, test is reporting only", () => {
    expect(SPLIT_USES.optimization).toContain("optimizer_feedback");
    expect(SPLIT_USES.validation).not.toContain("optimizer_feedback");
    expect(SPLIT_USES.test).toEqual(["reporting"]);
  });
});

describe("toEvaluationSpec (core/evaluation_spec.py)", () => {
  it("maps each evaluator's options to its strict config and lets the backend pin the version", () => {
    expect(toEvaluationSpec({ evaluator: "classification_accuracy", labels: ["a", "b"], caseSensitive: false })).toEqual({
      schema_version: "evaluationspec/1",
      evaluator: "classification_accuracy",
      evaluator_version: "",
      config: { labels: ["a", "b"], case_sensitive: false },
    });
    expect(toEvaluationSpec({ evaluator: "token_f1", passThreshold: 0.8 }).config).toEqual({ pass_threshold: 0.8 });
    expect(toEvaluationSpec({ evaluator: "numeric_tolerance", absoluteTolerance: 0.5, relativeTolerance: 0.01 }).config).toEqual({
      absolute_tolerance: 0.5,
      relative_tolerance: 0.01,
    });
    expect(toEvaluationSpec({ evaluator: "json_schema_validity" }).config).toEqual({});
  });

  it("offers only the evaluators the task contract accepts", () => {
    expect(evaluatorsFor("classification", ["string"])).toEqual(["classification_accuracy", "exact_match"]);
    expect(evaluatorsFor("question_answering", ["string"])).toEqual(["token_f1", "exact_match"]);
    expect(evaluatorsFor("question_answering", ["number"])).toEqual(["numeric_tolerance", "exact_match"]);
    expect(evaluatorsFor("structured_extraction", ["string", "number"])).toEqual(["exact_match", "json_schema_validity"]);
    expect(evaluatorsFor("structured_extraction", ["integer", "number"])).toContain("numeric_tolerance");
  });
});

describe("toObjectiveSpec (core/objective.py)", () => {
  const p = (o: OptimizationPreferences) => toObjectiveSpec(o);

  it("maps UI objective names to contract modes, with no weights outside balanced", () => {
    expect(p({ objective: "quality", balanced: null })).toEqual({
      schema_version: "objectivespec/1",
      mode: "maximize_quality",
      weights: {},
      scales: {},
    });
    expect(p({ objective: "cost", balanced: null }).mode).toBe("minimize_cost");
    expect(p({ objective: "latency", balanced: null }).mode).toBe("minimize_latency");
  });

  it("turns balanced percentages into weights summing to 1, with scales only for penalized metrics", () => {
    const spec = p({
      objective: "balanced",
      balanced: { quality: 70, cost: 20, latency: 10, tokens: 0, costScale: 0.0005, latencyScale: 2, tokensScale: 900 },
    });
    expect(spec.mode).toBe("balanced");
    expect(spec.weights).toEqual({ quality: 0.7, cost: 0.2, latency: 0.1 });
    expect(Math.abs(Object.values(spec.weights).reduce((s, w) => s + (w ?? 0), 0) - 1)).toBeLessThan(1e-9);
    expect(spec.scales).toEqual({ cost: 0.0005, latency: 2 }); // tokens has no weight, so no scale
    expect(() =>
      p({
        objective: "balanced",
        balanced: { quality: 80, cost: 20, latency: 0, tokens: 0, costScale: null, latencyScale: null, tokensScale: null },
      }),
    ).toThrow(/positive scale/);
  });
});

describe("toConstraintLimits (core/constraints.py ConstraintLimits)", () => {
  it("keeps contract names and units; per-example run caps the UI does not expose are no limit", () => {
    expect(
      toConstraintLimits({
        minQuality: 0.8,
        maxCostPerExample: 0.0004,
        maxMeanLatencyS: 1.5,
        maxP95LatencyS: 4,
        maxTokensPerExample: 3000,
        maxWorkflowSteps: 5,
      }),
    ).toEqual({
      minimum_quality: 0.8,
      maximum_cost_per_example: 0.0004,
      maximum_mean_latency_s: 1.5,
      maximum_p95_latency_s: 4,
      maximum_tokens_per_example: 3000,
      maximum_workflow_steps: 5,
      maximum_model_calls: null,
      maximum_tool_calls: null,
      maximum_retries: null,
      maximum_wall_time_s: null,
    });
  });
});

describe("toTaskContract (core/task_contract.py)", () => {
  it("builds schemas from roles: inputs + context in, targets out, nullable columns optional", () => {
    const seed = EXPERIMENTS.find((e) => e.id === "e-triage-1")!;
    const c = toTaskContract("e-triage-1", seed.config, dataset("d-tickets"));
    expect(c.task_type).toBe("classification");
    expect(c.input_schema.fields).toEqual([
      { name: "subject", type: "string", required: true },
      { name: "body", type: "string", required: true },
      { name: "customer_tier", type: "string", required: false },
    ]);
    expect(c.output_schema.fields).toEqual([{ name: "queue", type: "string", required: true }]);
    expect(c.evaluation.evaluator).toBe("classification_accuracy");
    expect(c.constraints.maximum_p95_latency_s).toBe(9);
  });

  it("maps a multi-field extraction with typed targets", () => {
    const seed = EXPERIMENTS.find((e) => e.id === "e-invoices-1")!;
    const c = toTaskContract("e-invoices-1", seed.config, dataset("d-invoices"));
    expect(c.output_schema.fields.map((f) => [f.name, f.type])).toEqual([
      ["vendor", "string"],
      ["total", "number"],
      ["currency", "string"],
      ["due_date", "date"],
    ]);
    expect(c.objective.mode).toBe("minimize_cost");
    expect(c.constraints.minimum_quality).toBe(0.8);
  });
});

describe("fromMeasurements (core/objective.py CandidateMeasurements)", () => {
  it("keeps null as not measured", () => {
    const m = fromMeasurements(
      {
        quality: 0.9,
        mean_cost_per_example: null,
        mean_latency_s: 1.2,
        p95_latency_s: 2,
        mean_tokens_per_example: null,
        max_tokens_per_example: 1800,
        workflow_steps: 3,
        max_model_calls_per_example: null,
        max_tool_calls_per_example: null,
        max_retries_per_example: null,
        max_wall_time_s_per_example: null,
      },
      120,
    );
    expect(m).toEqual({
      quality: 0.9,
      costPerExample: null,
      meanLatencyS: 1.2,
      p95LatencyS: 2,
      meanTokensPerExample: null,
      maxTokensPerExample: 1800,
      workflowSteps: 3,
      n: 120,
    });
  });
});
