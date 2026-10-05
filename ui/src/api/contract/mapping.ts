/*
  The one translation layer between UI view models and the backend contract shapes.

  UI presentation               contract
  ----------------------------  ----------------------------------------------
  objective "quality"           ObjectiveSpec.mode "maximize_quality" (etc.)
  balanced weights in %         weights summing to 1, scales for penalized metrics only
  split percentages + seed      SplitPlan basis points (optimization gets the rest)
  camelCase limits, seconds     ConstraintLimits snake_case, same units
  column roles                  DatasetSpec input / target / context / id columns
                                (RegisterDataset row_ids: "column" iff an id column is chosen)
  task + evaluator options      TaskContract input/output schemas + EvaluationSpec
  product API records           Project, DatasetUpload, DatasetVersion, Dataset, DatasetSplitSet

  Adapters call these; screens never do. Values the server computed (types, nullability, row
  counts, hashes, versions, split sizes) are copied as they are, never recomputed here.
*/
import type {
  BalancedWeights,
  ColumnMapping,
  Dataset,
  DatasetSplitSet,
  DatasetUpload,
  DatasetVersion,
  EvaluationConfig,
  ExperimentConfig,
  HardConstraints,
  Measurement,
  Objective,
  OptimizationPreferences,
  Project,
  RegisterDataset,
  SplitId,
  SplitPlan,
} from "../types";
import type {
  CandidateMeasurementsWire,
  ConstraintLimitsWire,
  DatasetSpecWire,
  DatasetVersionRecordWire,
  DatasetViewWire,
  EvaluationSpecWire,
  FieldWire,
  MetricWire,
  ObjectiveModeWire,
  ObjectiveSpecWire,
  ProjectRecordWire,
  RegisterDatasetWire,
  SplitPlanWire,
  SplitsRecordWire,
  TaskContractWire,
  UploadRecordWire,
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

/** The DatasetSpec of a registered version: the inverse of `fromDatasetVersion` (metadata aside). */
export function toDatasetSpec(d: DatasetVersion): DatasetSpecWire {
  if (!SLUG.test(d.datasetId)) throw new ContractMappingError(`Dataset id ${d.datasetId} is not a valid slug.`);
  if (!SHA256.test(d.contentHash)) throw new ContractMappingError("The dataset has no content hash.");
  const problems = mappingProblems(d.mapping, d.columns);
  if (problems.length) throw new ContractMappingError(problems[0]);
  return {
    schema_version: "datasetspec/1",
    dataset_id: d.datasetId,
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

/** Percentages -> basis points, unchecked: the server validates the plan and says why it refuses one. */
export function splitPlanWire(s: SplitPlan): SplitPlanWire {
  return { seed: s.seed, validation_bps: Math.round(s.validationPct * 100), test_bps: Math.round(s.testPct * 100) };
}

export function toSplitPlan(s: SplitPlan): SplitPlanWire {
  const plan = splitPlanWire(s);
  if (plan.validation_bps < 0 || plan.test_bps < 0 || plan.validation_bps + plan.test_bps >= BPS)
    throw new ContractMappingError("Validation and test must leave rows for optimization.");
  return plan;
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
export function toTaskContract(taskId: string, config: ExperimentConfig, dataset: DatasetVersion): TaskContractWire {
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

/* ---------------------------------------------------------------- product API v1 */

export function fromProject(w: ProjectRecordWire): Project {
  // the project record carries no counts; the UI shows "—" rather than inventing them
  return { id: w.project_id, name: w.name, description: w.description, createdAt: w.created_at, datasetCount: null, experimentCount: null };
}

const uploadFormat = (f: UploadRecordWire["format"]): "csv" | "jsonl" => {
  if (f === "csv" || f === "jsonl") return f;
  throw new ContractMappingError(`An upload cannot have format ${f}.`);
};

export function fromUpload(w: UploadRecordWire): DatasetUpload {
  return {
    id: w.upload_id,
    projectId: w.project_id,
    fileName: w.filename,
    format: uploadFormat(w.format),
    contentHash: w.content_hash,
    sizeBytes: w.size_bytes,
    rowCount: w.row_count,
    columns: w.columns.map((c) => ({ name: c.name, type: c.type, nullable: c.nullable, nullCount: c.null_count })),
    preview: w.preview,
    parserVersion: w.parser_version,
    createdAt: w.created_at,
  };
}

/** Roles -> RegisterDataset. Row ids come from the id column when one is chosen, else the server generates them. */
export function toRegisterDataset(input: RegisterDataset): RegisterDatasetWire {
  const m: ColumnMapping = input.mapping;
  return {
    dataset_id: input.datasetId,
    name: input.name.trim(),
    input_columns: m.input,
    target_columns: m.target,
    context_columns: m.context,
    row_ids: m.id === null ? "generated" : "column",
    id_column: m.id,
  };
}

export function fromDatasetVersion(w: DatasetVersionRecordWire): DatasetVersion {
  const s = w.spec;
  if (w.identity_hash.length !== 64) throw new ContractMappingError("A dataset version has no identity hash.");
  return {
    datasetId: s.dataset_id,
    projectId: w.project_id,
    uploadId: w.upload_id,
    version: s.dataset_version,
    name: s.name,
    format: s.format,
    contentHash: s.content_hash,
    identityHash: w.identity_hash,
    rowCount: s.row_count,
    columns: s.columns.map((c) => ({ name: c.name, type: c.type, nullable: c.nullable })),
    mapping: { input: s.input_columns, target: s.target_columns, context: s.context_columns, id: s.id_column },
    rowIdSource: w.row_id_source,
    rowIdScheme: w.row_id_scheme,
    rowIdsHash: w.row_ids_hash,
    createdAt: w.created_at,
  };
}

export function fromDatasetView(w: DatasetViewWire): Dataset {
  const versions = w.versions.map(fromDatasetVersion).sort((a, b) => a.version - b.version);
  if (!versions.some((v) => v.version === w.latest_version))
    throw new ContractMappingError(`Dataset ${w.dataset_id} does not include its latest version ${w.latest_version}.`);
  return { id: w.dataset_id, projectId: w.project_id, name: w.name, latestVersion: w.latest_version, versions };
}

export function fromSplitPlan(w: SplitPlanWire): SplitPlan {
  return { validationPct: w.validation_bps / 100, testPct: w.test_bps / 100, seed: w.seed };
}

export function fromSplitsRecord(w: SplitsRecordWire): DatasetSplitSet {
  const sizes: Record<SplitId, number> = { optimization: 0, validation: 0, test: 0 };
  for (const [role, n] of Object.entries(w.sizes) as [SplitId, number][]) sizes[role] = n;
  return {
    datasetId: w.dataset_id,
    version: w.dataset_version,
    splitsHash: w.splits_hash,
    datasetHash: w.splits.dataset_hash,
    method: w.splits.method,
    plan: w.splits.plan === null ? null : fromSplitPlan(w.splits.plan),
    sizes,
    createdAt: w.created_at,
  };
}

