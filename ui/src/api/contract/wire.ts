/*
  JSON shapes of the merged backend contracts, as `model_dump(mode="json")` produces them.

  Only `src/api/contract/mapping.ts` and API adapters may import this file. Screens use the view
  models in `src/api/types.ts`. Field names and enum values here must match the Python models:

    DatasetSpec / SplitPlan        core/dataset.py
    EvaluationSpec                 core/evaluation_spec.py
    ObjectiveSpec / Measurements   core/objective.py
    ConstraintLimits               core/constraints.py
    TaskContract                   core/task_contract.py

  and the product API v1 (api/product.py), whose records come from:

    ProjectRecord / UploadRecord / DatasetVersionRecord / SplitsRecord   store/datasets.py
    ColumnProfile                                                       ingestion/parse.py
    NewProject / RegisterDataset (request bodies)                       ingestion/service.py

  `fixtures/product-api.v1.json` holds real responses, regenerated from the Python API by
  tests/test_product_api_ui_fixtures.py; `decode.ts` checks every response against these shapes.
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

/* ---------------------------------------------------------------- product API v1 */

export type DatasetFormatWire = DatasetSpecWire["format"];

/** POST /projects body (ingestion/service.py NewProject) */
export interface NewProjectWire {
  name: string;
  description: string;
}

/** store/datasets.py ProjectRecord */
export interface ProjectRecordWire {
  project_id: string;
  name: string;
  description: string;
  created_at: string;
}

export interface ProjectListWire {
  projects: ProjectRecordWire[];
}

/** ingestion/parse.py ColumnProfile: a column as the server inspected it, over every row */
export interface ColumnProfileWire {
  name: string;
  type: ColumnTypeWire;
  nullable: boolean;
  null_count: number;
}

/** store/datasets.py UploadRecord. Every field except `filename` is computed from the bytes. */
export interface UploadRecordWire {
  upload_id: string;
  project_id: string;
  filename: string | null;
  format: DatasetFormatWire;
  content_hash: string;
  size_bytes: number;
  row_count: number;
  columns: ColumnProfileWire[];
  preview: Record<string, unknown>[];
  parser_version: string;
  created_at: string;
}

export type RowIdSourceWire = "column" | "generated";

/** POST /uploads/{id}/register body (ingestion/service.py RegisterDataset) */
export interface RegisterDatasetWire {
  dataset_id: string;
  name: string;
  input_columns: string[];
  target_columns: string[];
  context_columns: string[];
  row_ids: RowIdSourceWire;
  id_column: string | null;
}

/** store/datasets.py DatasetVersionRecord */
export interface DatasetVersionRecordWire {
  project_id: string;
  upload_id: string;
  spec: DatasetSpecWire;
  identity_hash: string;
  row_id_source: RowIdSourceWire;
  row_id_scheme: string | null;
  row_ids_hash: string;
  created_at: string;
}

/** api/product.py DatasetView */
export interface DatasetViewWire {
  dataset_id: string;
  project_id: string;
  name: string;
  latest_version: number;
  versions: DatasetVersionRecordWire[];
}

export interface DatasetListWire {
  datasets: DatasetViewWire[];
}

/** core/dataset.py DatasetSplit */
export interface DatasetSplitWire {
  split_id: string;
  role: SplitRoleWire;
  row_ids: string[];
}

export type SplitMethodWire = "seeded_hash/1" | "explicit";

/** core/dataset.py DatasetSplits */
export interface DatasetSplitsWire {
  schema_version: "datasetsplits/1";
  dataset_hash: string;
  method: SplitMethodWire;
  plan: SplitPlanWire | null;
  splits: DatasetSplitWire[];
}

/** store/datasets.py SplitsRecord. `sizes` has a key only for roles that received rows. */
export interface SplitsRecordWire {
  dataset_id: string;
  dataset_version: number;
  splits_hash: string;
  splits: DatasetSplitsWire;
  sizes: Partial<Record<SplitRoleWire, number>>;
  created_at: string;
}

export interface SplitsListWire {
  splits: SplitsRecordWire[];
}

/** api/product.py ErrorResponse. `code` is stable (api.product.ERROR_STATUS). */
export interface ErrorResponseWire {
  error: {
    code: string;
    message: string;
    details: Record<string, string | number | null>;
  };
}
