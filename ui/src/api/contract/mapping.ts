/*
  The one translation layer between UI view models and the backend contract shapes.

  UI presentation               contract
  ----------------------------  ----------------------------------------------
  objective "quality"           ObjectiveSpec.mode "maximize_quality" (etc.)
  balanced weights in %         weights summing to 1, scales for penalized metrics only
  split percentages + seed      SplitPlan basis points (optimization gets the rest)
  camelCase limits, seconds     ConstraintLimits snake_case, same units
  column roles                  DatasetSpec input / target / context / id columns
  task + evaluator options      TaskContract input/output schemas + EvaluationSpec

  An adapter for the real product API calls these; screens never do.
*/
import type {
  BalancedWeights,
  Dataset,
  EvaluationConfig,
  ExperimentConfig,
  HardConstraints,
  Measurement,
  Objective,
  OptimizationPreferences,
  SplitPlan,
} from "../types";
import type {
  CandidateMeasurementsWire,
  ConstraintLimitsWire,
  DatasetSpecWire,
  EvaluationSpecWire,
  FieldWire,
  MetricWire,
  ObjectiveModeWire,
  ObjectiveSpecWire,
  SplitPlanWire,
  TaskContractWire,
} from "./wire";
import { BPS, mappingProblems } from "./rules";

export class ContractMappingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContractMappingError";
  }
}

const SLUG = /^[a-z0-9][a-z0-9_.-]{0,127}$/;
const SHA256 = /^[0-9a-f]{64}$/;

export const OBJECTIVE_MODE: Record<Objective, ObjectiveModeWire> = {
  quality: "maximize_quality",
  cost: "minimize_cost",
  latency: "minimize_latency",
  balanced: "balanced",
};

export function toDatasetSpec(d: Dataset): DatasetSpecWire {
  if (!SLUG.test(d.id)) throw new ContractMappingError(`Dataset id ${d.id} is not a valid slug.`);
  if (!d.contentHash || !SHA256.test(d.contentHash)) throw new ContractMappingError("The dataset has no content hash.");
  const problems = mappingProblems(d.mapping, d.columns);
  if (problems.length) throw new ContractMappingError(problems[0]);
  return {
    schema_version: "datasetspec/1",
    dataset_id: d.id,
    dataset_version: d.version,
    name: d.name,
    content_hash: d.contentHash,
    format: d.format,
    columns: d.columns.map((c) => ({ name: c.name, type: c.type, nullable: c.nullable })),
    id_column: d.mapping.id,
    input_columns: d.mapping.input,
    target_columns: d.mapping.target,
    context_columns: d.mapping.context,
    row_count: d.rowCount,
    metadata: {},
  };
}

export function toSplitPlan(s: SplitPlan): SplitPlanWire {
  const validation_bps = Math.round(s.validationPct * 100);
  const test_bps = Math.round(s.testPct * 100);
  if (validation_bps < 0 || test_bps < 0 || validation_bps + test_bps >= BPS)
    throw new ContractMappingError("Validation and test must leave rows for optimization.");
  return { seed: s.seed, validation_bps, test_bps };
}

export function toEvaluationSpec(e: EvaluationConfig): EvaluationSpecWire {
  const base = { schema_version: "evaluationspec/1" as const, evaluator_version: "" };
  switch (e.evaluator) {
    case "classification_accuracy":
      return { ...base, evaluator: e.evaluator, config: { labels: e.labels, case_sensitive: e.caseSensitive } };
    case "exact_match":
      return { ...base, evaluator: e.evaluator, config: { case_sensitive: e.caseSensitive, normalize_whitespace: e.normalizeWhitespace } };
    case "token_f1":
      return { ...base, evaluator: e.evaluator, config: { pass_threshold: e.passThreshold } };
    case "json_schema_validity":
      return { ...base, evaluator: e.evaluator, config: {} };
    case "numeric_tolerance":
      return { ...base, evaluator: e.evaluator, config: { absolute_tolerance: e.absoluteTolerance, relative_tolerance: e.relativeTolerance } };
  }
}

export function toObjectiveSpec(p: OptimizationPreferences): ObjectiveSpecWire {
  const spec: ObjectiveSpecWire = { schema_version: "objectivespec/1", mode: OBJECTIVE_MODE[p.objective], weights: {}, scales: {} };
  if (p.objective !== "balanced") return spec;
  if (!p.balanced) throw new ContractMappingError("A balanced objective needs weights.");
  const b: BalancedWeights = p.balanced;
  const parts: [MetricWire, number, number | null][] = [
    ["quality", b.quality, null],
    ["cost", b.cost, b.costScale],
    ["latency", b.latency, b.latencyScale],
    ["tokens", b.tokens, b.tokensScale],
  ];
  for (const [metric, pct, scale] of parts) {
    if (pct <= 0) continue;
    spec.weights[metric] = pct / 100;
    if (metric !== "quality") {
      if (!scale || scale <= 0) throw new ContractMappingError(`The ${metric} weight needs a positive scale.`);
      spec.scales[metric] = scale;
    }
  }
  return spec;
}

/** Limits the UI does not expose yet (per-example run caps) are sent as "no limit". */
export function toConstraintLimits(c: HardConstraints): ConstraintLimitsWire {
  return {
    minimum_quality: c.minQuality,
    maximum_cost_per_example: c.maxCostPerExample,
    maximum_mean_latency_s: c.maxMeanLatencyS,
    maximum_p95_latency_s: c.maxP95LatencyS,
    maximum_tokens_per_example: c.maxTokensPerExample,
    maximum_workflow_steps: c.maxWorkflowSteps,
    maximum_model_calls: null,
    maximum_tool_calls: null,
    maximum_retries: null,
    maximum_wall_time_s: null,
  };
}

/**
 * The TaskContract an experiment configuration describes. Schemas follow the column roles:
 * inputs + context form the input schema, targets the output schema; a nullable column becomes an
 * optional field, because the contract refuses a required field over a nullable column.
 */
export function toTaskContract(taskId: string, config: ExperimentConfig, dataset: Dataset): TaskContractWire {
  if (!SLUG.test(taskId)) throw new ContractMappingError(`Task id ${taskId} is not a valid slug.`);
  const ds = toDatasetSpec(dataset);
  const field = (name: string): FieldWire => {
    const col = dataset.columns.find((c) => c.name === name)!;
    if (col.type === "json") throw new ContractMappingError(`${name} holds nested JSON and cannot be a schema field.`);
    return { name, type: col.type, required: !col.nullable };
  };
  return {
    schema_version: "taskcontract/1",
    task_id: taskId,
    contract_version: 1,
    task_type: config.taskType,
    instructions: config.instructions,
    input_schema: { fields: [...ds.input_columns, ...ds.context_columns].map(field) },
    output_schema: { fields: ds.target_columns.map(field) },
    dataset: ds,
    evaluation: toEvaluationSpec(config.evaluation),
    objective: toObjectiveSpec(config.preferences),
    constraints: toConstraintLimits(config.constraints),
  };
}

export function fromMeasurements(w: CandidateMeasurementsWire, n: number): Measurement {
  return {
    quality: w.quality,
    costPerExample: w.mean_cost_per_example,
    meanLatencyS: w.mean_latency_s,
    p95LatencyS: w.p95_latency_s,
    meanTokensPerExample: w.mean_tokens_per_example,
    maxTokensPerExample: w.max_tokens_per_example,
    workflowSteps: w.workflow_steps,
    n,
  };
}
