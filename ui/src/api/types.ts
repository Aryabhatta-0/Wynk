/*
  The frontend's view of Wynk's product objects.

  These are UI view models, not the backend contract. The backend contract is being designed
  separately; when it lands, a real adapter maps its responses onto these types (see
  `client.ts`). Nothing outside `src/api/` may depend on how the data is fetched or where mock
  data comes from.
*/

export type ID = string;

/* ---------------------------------------------------------------- projects */

export interface Project {
  id: ID;
  name: string;
  description: string;
  createdAt: string; // ISO 8601
  datasetCount: number;
  experimentCount: number;
}

export interface NewProject {
  name: string;
  description: string;
}

/* ---------------------------------------------------------------- datasets */

export type DatasetFormat = "csv" | "jsonl" | "parquet";

export type ColumnType = "string" | "number" | "boolean" | "json" | "empty";

export interface DatasetColumn {
  name: string;
  type: ColumnType;
  /** rows in the preview with no value for this column */
  missing: number;
  /** distinct values in the preview (not the whole file) */
  distinct: number;
}

/** Which columns the workflow reads, which one it must produce, and which add context. */
export interface ColumnMapping {
  input: string[];
  target: string | null;
  context: string[];
}

export type DatasetRow = Record<string, unknown>;

/** A dataset as the client parsed it, before it is registered. */
export interface DatasetDraft {
  name: string;
  format: DatasetFormat;
  fileName: string;
  sizeBytes: number;
  /** rows counted in the file; null when only a sample was read */
  rowCount: number | null;
  columns: DatasetColumn[];
  preview: DatasetRow[];
  mapping: ColumnMapping;
}

export type DatasetStatus = "needs_mapping" | "ready";

export interface Dataset extends DatasetDraft {
  id: ID;
  projectId: ID;
  createdAt: string;
  status: DatasetStatus;
}

/* ---------------------------------------------------------------- configuration */

export type TaskType = "classification" | "structured_extraction" | "question_answering";

export type Objective = "quality" | "cost" | "latency" | "balanced";

export interface ModelOption {
  id: string;
  label: string;
  provider: string;
}

/** Must hold. A candidate that breaks one is infeasible and can never be champion. */
export interface HardConstraints {
  /** minimum quality on the validation split, 0..1 */
  minQuality: number | null;
  /** maximum USD per 1,000 examples */
  maxCostPer1k: number | null;
  /** maximum 95th-percentile latency per example, milliseconds */
  maxLatencyP95Ms: number | null;
  /** models a workflow may call; at least one */
  allowedModels: string[];
}

/** Steers the search among feasible candidates. Never overrides a hard constraint. */
export interface OptimizationPreferences {
  objective: Objective;
}

/** The search stops at whichever limit it reaches first. */
export interface SearchBudget {
  maxCandidates: number;
  maxGenerations: number;
  maxSpendUsd: number;
  maxDurationMin: number;
}

/** Percentages of the dataset; they sum to 100. */
export interface SplitPlan {
  optimization: number;
  validation: number;
  test: number;
}

export interface ExperimentConfig {
  name: string;
  datasetId: ID;
  taskType: TaskType;
  constraints: HardConstraints;
  preferences: OptimizationPreferences;
  budget: SearchBudget;
  splits: SplitPlan;
}

/* ---------------------------------------------------------------- workflows */

export type StageKind = "GATHER" | "FILTER" | "EXTRACT" | "REASON" | "VERIFY" | "SYNTHESIZE";

export interface WorkflowStage {
  kind: StageKind;
  options: string[];
}

/**
 * Candidate: evaluated on the optimization split.
 * Validated: re-evaluated on the validation split.
 * Champion:  the validated workflow chosen by the objective among those meeting every constraint.
 */
export type WorkflowState = "candidate" | "validated" | "champion";

/** One measurement of one workflow on one split. */
export interface Measurement {
  /** task metric, 0..1 (accuracy, field match or answer match, depending on the task) */
  quality: number;
  costPer1k: number;
  latencyP95Ms: number;
  /** examples measured */
  n: number;
}

export type ConstraintKey = "minQuality" | "maxCostPer1k" | "maxLatencyP95Ms" | "allowedModels";

export interface ConstraintCheck {
  key: ConstraintKey;
  ok: boolean;
  observed: number | string;
  limit: number | string;
}

export interface Candidate {
  id: ID;
  genomeHash: string;
  stages: WorkflowStage[];
  model: string;
  /** generation in which the search first proposed it */
  generation: number;
  state: WorkflowState;
  optimization: Measurement;
  validation: Measurement | null;
  /** checked on validation when measured there, otherwise on optimization */
  constraints: ConstraintCheck[];
  feasible: boolean;
}

/* ---------------------------------------------------------------- experiments */

export type ExperimentStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type ExperimentPhase = "searching" | "validating" | "testing" | "done";

/** Which budget limit ended the search. */
export type StopReason = "generations" | "candidates" | "spend" | "duration";

export interface SearchProgress {
  phase: ExperimentPhase;
  generation: number;
  evaluated: number;
  spendUsd: number;
  elapsedSec: number;
  /** set once the search phase is over */
  stopReason: StopReason | null;
}

/** Best-so-far value of the objective metric after `evaluated` candidates. */
export interface CurvePoint {
  evaluated: number;
  wynk: number | null;
  random: number | null;
}

export interface GenerationSummary {
  generation: number;
  /** candidates evaluated in this generation */
  evaluated: number;
  bestQuality: number;
  meanQuality: number;
  /**
   * Convergence, 0..1: for each stage decision (and the model), the share of this generation's
   * proposals that made the most common choice, averaged over decisions.
   */
  agreement: number;
  /** new candidates (never proposed before) in this generation */
  novel: number;
}

/** A transition between two stage choices the search keeps reinforcing. */
export interface EdgeSignal {
  from: string;
  to: string;
  /** relative to the strongest edge, 0..1 */
  strength: number;
  trend: "rising" | "steady" | "falling";
}

export interface Experiment {
  id: ID;
  projectId: ID;
  datasetId: ID;
  name: string;
  config: ExperimentConfig;
  status: ExperimentStatus;
  createdAt: string;
  finishedAt: string | null;
  progress: SearchProgress;
  /** fixed reference workflow, on the optimization split */
  baseline: Measurement | null;
  /** best feasible candidate so far by the objective, on the optimization split */
  bestId: ID | null;
  championId: ID | null;
  curve: CurvePoint[];
  generations: GenerationSummary[];
  edges: EdgeSignal[];
  candidates: Candidate[];
  /** set when status is failed */
  failure: string | null;
}

/* ---------------------------------------------------------------- results */

export type MethodId = "fixed_baseline" | "random_search" | "wynk_aco";

export type SplitId = "optimization" | "validation" | "test";

export interface MethodResult {
  method: MethodId;
  stages: WorkflowStage[];
  model: string;
  /** candidates the method evaluated; 1 for the fixed baseline */
  evaluated: number;
  splits: Partial<Record<SplitId, Measurement>>;
  /** checked on the held-out test split when it is unlocked, otherwise on validation */
  constraints: ConstraintCheck[];
}

export interface ResultsComparison {
  experimentId: ID;
  splitSizes: Record<SplitId, number>;
  methods: MethodResult[];
  /** the held-out test split is measured once, after the champion is fixed */
  testLocked: boolean;
}

export interface WorkflowSummary {
  candidate: Candidate;
  experimentId: ID;
  experimentName: string;
  datasetName: string;
  objective: Objective;
}
