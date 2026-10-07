import type {
  Created,
  Dataset,
  DatasetFile,
  DatasetSplitSet,
  DatasetUpload,
  DatasetVersion,
  Experiment,
  ExperimentConfig,
  ID,
  ModelOption,
  NewProject,
  Project,
  RegisterDataset,
  ResultsComparison,
  SplitPlan,
  WorkflowSummary,
} from "./types";

/*
  The single boundary between the UI and Wynk's data. Screens call `api()` and nothing else;
  they never import mock data or build URLs.

  Two implementations exist:
    - live  (default): the product API v1 (`api/product.py`), `src/api/live.ts`
    - mock  (`?api=mock` or VITE_WYNK_API=mock): in-memory fixtures and a simulated search,
            `src/api/mock/`, for tests and demos only

  Projects and datasets mirror the product API one-to-one. Experiments have no backend yet
  (Issues #20–#23): the mock simulates them, the live adapter refuses them without a request.
*/
export interface WynkApi {
  readonly mode: "mock" | "live";
  /** "simulated": the mock invents runs. "unavailable": no experiment backend exists yet. */
  readonly experiments: "simulated" | "unavailable";

  listProjects(): Promise<Project[]>;
  getProject(id: ID): Promise<Project>;
  createProject(input: NewProject): Promise<Project>;

  /** Sends the file's bytes; the server inspects them (types, nulls, row count, hash, preview). */
  uploadDataset(projectId: ID, file: DatasetFile): Promise<Created<DatasetUpload>>;
  getUpload(uploadId: ID): Promise<DatasetUpload>;
  /** Registers an upload with column roles as a new version (or returns the identical one). */
  registerDataset(uploadId: ID, input: RegisterDataset): Promise<Created<DatasetVersion>>;
  listDatasets(projectId: ID): Promise<Dataset[]>;
  getDataset(datasetId: ID): Promise<Dataset>;
  getDatasetVersion(datasetId: ID, version: number): Promise<DatasetVersion>;
  createSplits(datasetId: ID, version: number, plan: SplitPlan): Promise<Created<DatasetSplitSet>>;
  listSplits(datasetId: ID, version: number): Promise<DatasetSplitSet[]>;
  getSplits(datasetId: ID, version: number, splitsHash: string): Promise<DatasetSplitSet>;

  // experiments: simulated in mock mode only, see `experiments`
  listModels(): Promise<ModelOption[]>;
  listExperiments(projectId: ID): Promise<Experiment[]>;
  getExperiment(id: ID): Promise<Experiment>;
  createExperiment(projectId: ID, config: ExperimentConfig): Promise<Experiment>;
  cancelExperiment(id: ID): Promise<Experiment>;
  getResults(experimentId: ID): Promise<ResultsComparison>;
  listWorkflows(projectId: ID): Promise<WorkflowSummary[]>;
}

/** What kind of failure an error is, for screens that react to it (retry, fix the file, ...). */
export type ApiErrorKind =
  | "not_found"
  | "invalid"
  | "conflict"
  | "too_large"
  | "unsupported"
  | "storage"
  | "server"
  | "unavailable"
  | "not_implemented"
  | "contract";

/**
 * Every stable error code the product API returns (`api.product.ERROR_STATUS`), plus the client's
 * own codes, with a short heading for people. The heading never replaces the reason: screens show
 * `title`, the server's `message` and the `code` together.
 */
export const ERROR_COPY: Record<string, { kind: ApiErrorKind; title: string }> = {
  invalid_request: { kind: "invalid", title: "The request was rejected" },
  invalid_json: { kind: "invalid", title: "The request was rejected" },
  unknown_parameter: { kind: "invalid", title: "The request was rejected" },
  not_found: { kind: "not_found", title: "Not found" },
  project_not_found: { kind: "not_found", title: "Project not found" },
  upload_not_found: { kind: "not_found", title: "Upload not found" },
  dataset_not_found: { kind: "not_found", title: "Dataset not found" },
  dataset_version_not_found: { kind: "not_found", title: "Dataset version not found" },
  splits_not_found: { kind: "not_found", title: "Splits not found" },
  method_not_allowed: { kind: "invalid", title: "The request was rejected" },
  dataset_conflict: { kind: "conflict", title: "This dataset id is already taken" },
  length_required: { kind: "invalid", title: "The upload was rejected" },
  payload_too_large: { kind: "too_large", title: "The file is too large" },
  unsupported_media_type: { kind: "unsupported", title: "The upload was rejected" },
  unsupported_format: { kind: "unsupported", title: "Unsupported file format" },
  empty_file: { kind: "invalid", title: "The file is empty" },
  invalid_encoding: { kind: "invalid", title: "The file is not valid UTF-8" },
  malformed_csv: { kind: "invalid", title: "The CSV file is malformed" },
  malformed_jsonl: { kind: "invalid", title: "The JSONL file is malformed" },
  no_rows: { kind: "invalid", title: "The file has no data rows" },
  too_many_columns: { kind: "invalid", title: "The file has too many columns" },
  invalid_column_name: { kind: "invalid", title: "A column name is not allowed" },
  duplicate_column: { kind: "invalid", title: "A column name appears twice" },
  unknown_column: { kind: "invalid", title: "The column mapping is invalid" },
  json_column_role: { kind: "invalid", title: "The column mapping is invalid" },
  invalid_mapping: { kind: "invalid", title: "The column mapping is invalid" },
  invalid_id_column: { kind: "invalid", title: "The id column cannot identify rows" },
  duplicate_row_id: { kind: "invalid", title: "Row ids are not unique" },
  invalid_split_plan: { kind: "invalid", title: "The split plan was rejected" },
  job_not_found: { kind: "not_found", title: "Experiment not found" },
  invalid_experiment: { kind: "invalid", title: "The experiment was rejected" },
  job_not_cancellable: { kind: "conflict", title: "This experiment has already finished" },
  job_not_resumable: { kind: "conflict", title: "This experiment cannot be resumed" },
  ambiguous_attempt: { kind: "conflict", title: "A model call's outcome is unknown" },
  job_not_completed: { kind: "conflict", title: "This experiment has no results yet" },
  experiment_not_promotable: { kind: "conflict", title: "This experiment cannot be promoted" },
  incompatible_incumbent: { kind: "conflict", title: "The current champion is not comparable" },
  promotion_in_progress: { kind: "conflict", title: "A promotion is already being evaluated" },
  promotion_ambiguous_attempt: { kind: "conflict", title: "A held-out model call's outcome is unknown" },
  promotion_failed: { kind: "conflict", title: "The held-out evaluation failed" },
  promotion_not_found: { kind: "not_found", title: "Promotion not found" },
  champion_not_found: { kind: "not_found", title: "No champion yet" },
  heldout_backend_unavailable: { kind: "unavailable", title: "Held-out evaluation is unavailable" },
  promotion_evidence_mismatch: { kind: "server", title: "Stored promotion evidence is inconsistent" },
  artifact_not_finalized: { kind: "conflict", title: "This experiment's artifact is not finalized yet" },
  trace_path_not_found: { kind: "not_found", title: "No such field in the experiment artifact" },
  artifact_integrity_error: { kind: "server", title: "The stored experiment artifact does not verify" },
  reproduction_mismatch: { kind: "server", title: "The stored evidence does not reproduce the artifact" },
  promotion_error: { kind: "server", title: "The promotion failed" },
  storage_error: { kind: "storage", title: "Stored data is unavailable" },
  internal_error: { kind: "server", title: "The Wynk API failed" },
  // client-side codes
  network_error: { kind: "unavailable", title: "Could not reach the Wynk API" },
  bad_response: { kind: "unavailable", title: "The Wynk API did not answer" },
  contract_mismatch: { kind: "contract", title: "Unexpected response from the Wynk API" },
  experiment_backend_unavailable: { kind: "not_implemented", title: "Experiments need a backend that does not exist yet" },
  mock_failure: { kind: "unavailable", title: "Injected mock failure" },
};

function kindForStatus(status: number | null): ApiErrorKind {
  if (status === 404) return "not_found";
  if (status === 409) return "conflict";
  if (status === 413) return "too_large";
  if (status !== null && status >= 400 && status < 500) return "invalid";
  return "server";
}

export class ApiError extends Error {
  /** stable: the product API's `error.code`, or a client code such as `network_error` */
  readonly code: string;
  readonly kind: ApiErrorKind;
  /** a heading for people; `message` keeps the real reason */
  readonly title: string;
  /** HTTP status, null when no response arrived */
  readonly status: number | null;
  readonly details: Record<string, string | number | null>;

  constructor(code: string, message: string, init: { status?: number | null; details?: Record<string, string | number | null> } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = init.status ?? null;
    this.details = init.details ?? {};
    const copy = ERROR_COPY[code];
    this.kind = copy?.kind ?? kindForStatus(this.status);
    this.title = copy?.title ?? "The Wynk API refused the request";
  }
}

/** One line for an error: heading, the real reason and the stable code. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return `${err.title}: ${err.message} [${err.code}]`;
  if (err instanceof Error) return err.message;
  return String(err);
}
