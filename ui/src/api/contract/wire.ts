/*
  JSON shapes of the merged backend contracts, as `model_dump(mode="json")` produces them.

  Only `src/api/contract/mapping.ts` and API adapters may import this file. Screens use the view
  models in `src/api/types.ts`. Field names and enum values here must match the Python models:

    DatasetSpec / SplitPlan        core/dataset.py
    EvaluationSpec                 core/evaluation_spec.py
    ObjectiveSpec / Measurements   core/objective.py
    ConstraintLimits               core/constraints.py
    TaskContract                   core/task_contract.py
*/

export type ColumnTypeWire = "string" | "integer" | "number" | "boolean" | "date" | "string_list" | "json";
export type FieldTypeWire = Exclude<ColumnTypeWire, "json">;

export interface ColumnSpecWire {
  name: string;
  type: ColumnTypeWire;
  nullable: boolean;
}

export interface DatasetSpecWire {
  schema_version: "datasetspec/1";
  dataset_id: string;
  dataset_version: number;
  name: string;
  content_hash: string;
  format: "csv" | "jsonl" | "wynk_snapshot";
  columns: ColumnSpecWire[];
  id_column: string | null;
  input_columns: string[];
  target_columns: string[];
  context_columns: string[];
  row_count: number;
  metadata: Record<string, string | number | boolean>;
}

export type SplitRoleWire = "optimization" | "validation" | "test";
export type SplitUseWire = "optimizer_feedback" | "selection" | "reporting";

export interface SplitPlanWire {
  seed: number;
  validation_bps: number;
  test_bps: number;
}

export type EvaluatorKindWire =
  "exact_match" | "classification_accuracy" | "token_f1" | "json_schema_validity" | "numeric_tolerance" | "legacy_field_match";

export interface EvaluationSpecWire {
  schema_version: "evaluationspec/1";
  evaluator: EvaluatorKindWire;
  /** empty: the backend pins the version it currently implements */
  evaluator_version: string;
  config: Record<string, unknown>;
}

export type ObjectiveModeWire = "maximize_quality" | "minimize_cost" | "minimize_latency" | "balanced";
export type MetricWire = "quality" | "cost" | "latency" | "tokens";

export interface ObjectiveSpecWire {
  schema_version: "objectivespec/1";
  mode: ObjectiveModeWire;
  weights: Partial<Record<MetricWire, number>>;
  scales: Partial<Record<MetricWire, number>>;
}

export interface ConstraintLimitsWire {
  minimum_quality: number | null;
  maximum_cost_per_example: number | null;
  maximum_mean_latency_s: number | null;
  maximum_p95_latency_s: number | null;
  maximum_tokens_per_example: number | null;
  maximum_workflow_steps: number | null;
  maximum_model_calls: number | null;
  maximum_tool_calls: number | null;
  maximum_retries: number | null;
  maximum_wall_time_s: number | null;
}

export interface CandidateMeasurementsWire {
  quality: number | null;
  mean_cost_per_example: number | null;
  mean_latency_s: number | null;
  p95_latency_s: number | null;
  mean_tokens_per_example: number | null;
  max_tokens_per_example: number | null;
  workflow_steps: number | null;
  max_model_calls_per_example: number | null;
  max_tool_calls_per_example: number | null;
  max_retries_per_example: number | null;
  max_wall_time_s_per_example: number | null;
}

export interface FieldWire {
  name: string;
  type: FieldTypeWire;
  required: boolean;
}

export interface TaskContractWire {
  schema_version: "taskcontract/1";
  task_id: string;
  contract_version: number;
  task_type: "classification" | "structured_extraction" | "question_answering";
  instructions: string;
  input_schema: { fields: FieldWire[] };
  output_schema: { fields: FieldWire[] };
  dataset: DatasetSpecWire;
  evaluation: EvaluationSpecWire;
  objective: ObjectiveSpecWire;
  constraints: ConstraintLimitsWire;
}
