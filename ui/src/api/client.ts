import type {
  Dataset,
  DatasetDraft,
  ColumnMapping,
  Experiment,
  ExperimentConfig,
  ID,
  ModelOption,
  NewProject,
  Project,
  ResultsComparison,
  WorkflowSummary,
} from "./types";

/*
  The single boundary between the UI and Wynk's data. Screens call `api()` and nothing else;
  they never import mock data or build URLs.

  Two implementations exist:
    - mock  (default): in-memory fixtures and a simulated search, `src/api/mock/`
    - live  (`?api=live` or VITE_WYNK_API=live): the product API, which does not exist yet,
            so every call fails with `not_connected` instead of pretending to succeed.
  A real adapter replaces `live.ts` once the backend contract is final.
*/
export interface WynkApi {
  readonly mode: "mock" | "live";

  listProjects(): Promise<Project[]>;
  getProject(id: ID): Promise<Project>;
  createProject(input: NewProject): Promise<Project>;

  listDatasets(projectId: ID): Promise<Dataset[]>;
  getDataset(id: ID): Promise<Dataset>;
  /** Registers a dataset the client has parsed. A real backend receives the file itself. */
  createDataset(projectId: ID, draft: DatasetDraft): Promise<Dataset>;
  updateMapping(id: ID, mapping: ColumnMapping): Promise<Dataset>;

  listModels(): Promise<ModelOption[]>;

  listExperiments(projectId: ID): Promise<Experiment[]>;
  getExperiment(id: ID): Promise<Experiment>;
  createExperiment(projectId: ID, config: ExperimentConfig): Promise<Experiment>;
  cancelExperiment(id: ID): Promise<Experiment>;
  getResults(experimentId: ID): Promise<ResultsComparison>;

  listWorkflows(projectId: ID): Promise<WorkflowSummary[]>;
}

export type ApiErrorCode = "not_found" | "invalid" | "not_connected" | "unavailable";

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  constructor(code: ApiErrorCode, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
