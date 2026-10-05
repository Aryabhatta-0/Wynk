/*
  The frontend's view of Wynk's product objects.

  These are UI view models, not the backend contracts. They follow the meaning of the merged
  contracts (core/dataset.py, core/task_contract.py, core/evaluation_spec.py, core/objective.py,
  core/constraints.py, core/experiment.py) but are shaped for screens: camelCase, percentages
  where people type percentages, labels resolved. The translation to and from the contract
  shapes lives in one place, `src/api/contract/`. Nothing outside `src/api/` may depend on how
  the data is fetched, where mock data comes from, or what the backend's JSON looks like.
*/

export type ID = string;

/* ---------------------------------------------------------------- projects */

export interface Project {
  id: ID;
  name: string;
  description: string;
  createdAt: string; // ISO 8601
  /** null when the source does not report it (the product API's project record has no counts) */
  datasetCount: number | null;
  experimentCount: number | null;
}

export interface NewProject {
  name: string;
  description: string;
}

/* ---------------------------------------------------------------- datasets */

/**
 * Formats a dataset version can have. Uploads accept only csv and jsonl; `wynk_snapshot` is the
 * frozen benchmark layout and only appears on legacy datasets.
 */
export type DatasetFormat = "csv" | "jsonl" | "wynk_snapshot";

/** A column's value type. `json` is an opaque structured value and can take no role. */
export type ColumnType = "string" | "integer" | "number" | "boolean" | "date" | "string_list" | "json";

/** A column of a registered dataset version (DatasetSpec columns). */
export interface DatasetColumn {
  name: string;
  type: ColumnType;
  /** some row has no value */
  nullable: boolean;
}

/** A column as the server inspected the uploaded file, over every row. */
export interface InspectedColumn extends DatasetColumn {
  /** rows with no value for this column */
  nullCount: number;
}

/**
 * Column roles. They are disjoint: a column has at most one.
 * input   what a workflow reads (one or more)
 * target  what it must produce, used only to score it (one or more; one for classification and QA)
 * context optional supporting text a workflow may read
 * id      stable row id; required to split the dataset
 */
export interface ColumnMapping {
  input: string[];
  target: string[];
  context: string[];
  id: string | null;
}

export type DatasetRow = Record<string, unknown>;

/** A file to upload. The server decides whether `format` is supported and inspects the bytes. */
export interface DatasetFile {
  name: string;
  /** from the file extension; sent as-is, the server rejects anything it does not accept */
  format: string;
  bytes: Blob;
}

/**
 * An uploaded, inspected file (UploadRecord). Everything except `fileName` was computed by the
 * server from the bytes: hash, size, row count, column types, nullability and the preview.
 */
export interface DatasetUpload {
  id: ID;
  projectId: ID;
  fileName: string | null;
  format: DatasetFormat;
  /** sha256 of the exact uploaded bytes */
  contentHash: string;
  sizeBytes: number;
  rowCount: number;
  columns: InspectedColumn[];
  /** the first rows, long cells truncated by the server */
  preview: DatasetRow[];
  parserVersion: string;
  createdAt: string;
}

/** Column roles for registering an upload as a dataset version. */
export interface RegisterDataset {
  /** lowercase slug: letters, digits, `_`, `.`, `-`; owned by one project */
  datasetId: string;
  name: string;
  /** an id column gives row ids; without one the server generates them from row content */
  mapping: ColumnMapping;
}

/** How a version's rows are named: from its id column, or generated from row content. */
export type RowIdSource = "column" | "generated";

/** One registered, immutable dataset version (DatasetVersionRecord + its DatasetSpec). */
export interface DatasetVersion {
  datasetId: ID;
  projectId: ID;
  uploadId: ID;
  version: number;
  name: string;
  format: DatasetFormat;
  contentHash: string;
  /** DatasetSpec.identity_hash: changes iff content, columns, roles, row count, ids or version change */
  identityHash: string;
  rowCount: number;
  columns: DatasetColumn[];
  mapping: ColumnMapping;
  rowIdSource: RowIdSource;
  /** set iff row ids are generated */
  rowIdScheme: string | null;
  rowIdsHash: string;
  createdAt: string;
}

/** A dataset and all its versions, oldest first. */
export interface Dataset {
  id: ID;
  projectId: ID;
  /** the latest version's name */
  name: string;
  latestVersion: number;
  versions: DatasetVersion[];
}

/** Durable, deterministic splits of one dataset version (SplitsRecord). */
export interface DatasetSplitSet {
  datasetId: ID;
  version: number;
  /** DatasetSplits.identity_hash: same version + same plan gives the same hash */
  splitsHash: string;
  /** the identity hash of the version that was split */
  datasetHash: string;
  method: "seeded_hash/1" | "explicit";
  /** null for explicit splits */
  plan: SplitPlan | null;
  /** rows per split, as stored by the server; a role that received no rows has 0 */
  sizes: Record<SplitId, number>;
  createdAt: string;
}

/** A create call's result; `created` is false when the identical resource already existed. */
export interface Created<T> {
  value: T;
  created: boolean;
}

/* ---------------------------------------------------------------- configuration */

export type TaskType = "classification" | "structured_extraction" | "question_answering";

/**
 * How an output is judged. One evaluator per experiment, with the options that evaluator takes.
 * Quality is the mean per-example score it produces, 0..1.
 */
export type EvaluationConfig =
  | { evaluator: "classification_accuracy"; labels: string[]; caseSensitive: boolean }
  | { evaluator: "exact_match"; caseSensitive: boolean; normalizeWhitespace: boolean }
  | { evaluator: "token_f1"; passThreshold: number }
  | { evaluator: "json_schema_validity" }
  | { evaluator: "numeric_tolerance"; absoluteTolerance: number; relativeTolerance: number };

export type EvaluatorKind = EvaluationConfig["evaluator"];

export type Objective = "quality" | "cost" | "latency" | "balanced";

/**
 * Balanced objective: utility = wQuality·quality − Σ w·value / scale. Weights are percentages
 * summing to 100; quality keeps a positive weight; every penalized metric needs a scale (the value
 * that costs its full weight).
 */
export interface BalancedWeights {
  quality: number;
  cost: number;
  latency: number;
  tokens: number;
  /** USD per example that costs the full cost weight */
  costScale: number | null;
  /** mean seconds per example that cost the full latency weight */
  latencyScale: number | null;
  /** mean tokens per example that cost the full tokens weight */
  tokensScale: number | null;
}

/** Steers the search among feasible candidates. Never overrides a hard constraint. */
export interface OptimizationPreferences {
  objective: Objective;
  /** required when objective is balanced, otherwise null */
  balanced: BalancedWeights | null;
}

export interface ModelOption {
  id: string;
  label: string;
  provider: string;
}

/**
 * Hard limits on measured behaviour. A candidate that breaks one, or lacks the measurement one
 * needs, is infeasible and can never be champion. `null` = no limit. Equal to a limit is allowed.
 */
export interface HardConstraints {
  /** minimum quality, 0..1 */
  minQuality: number | null;
  /** maximum mean USD per example */
  maxCostPerExample: number | null;
  /** maximum mean seconds per example */
  maxMeanLatencyS: number | null;
  /** maximum 95th-percentile seconds per example */
  maxP95LatencyS: number | null;
  /** maximum tokens in any single example */
  maxTokensPerExample: number | null;
  /** maximum stages in the workflow */
  maxWorkflowSteps: number | null;
}

export type ConstraintKey = keyof HardConstraints;

/** The search stops at whichever limit it reaches first. Search settings only; not part of the task. */
export interface SearchBudget {
  maxCandidates: number;
  maxGenerations: number;
  maxSpendUsd: number;
  maxDurationMin: number;
}

/** Percentages (basis points / 100 in the contract); optimization gets the remaining rows. */
export interface SplitPlan {
  validationPct: number;
  testPct: number;
  /** seeds the reproducible, hash-based row assignment */
  seed: number;
}

export interface ExperimentConfig {
  name: string;
  datasetId: ID;
  taskType: TaskType;
  /** what the workflow should do with each row, in plain language */
  instructions: string;
  evaluation: EvaluationConfig;
  constraints: HardConstraints;
  preferences: OptimizationPreferences;
  /** models the search may use (the model configuration, not a constraint) */
  models: string[];
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
 * Candidate: measured on the optimization split (the only split that feeds the optimizer).
 * Validated: re-measured on the validation split, which may select and promote but never feeds the optimizer.
 * Champion:  the validated workflow the objective ranks first among those meeting every hard constraint.
 * The held-out test split plays no part in any of these states; it is reported only.
 */
export type WorkflowState = "candidate" | "validated" | "champion";

/** Aggregate measurements of one workflow on one split. `null` = not measured, never zero. */
export interface Measurement {
  /** mean per-example evaluator score, 0..1 */
  quality: number | null;
  /** mean USD per example */
  costPerExample: number | null;
  meanLatencyS: number | null;
  p95LatencyS: number | null;
  meanTokensPerExample: number | null;
  /** tokens in the worst single example */
  maxTokensPerExample: number | null;
  workflowSteps: number | null;
  /** examples measured */
  n: number;
}

export interface ConstraintCheck {
  key: ConstraintKey;
  ok: boolean;
  /** null when the measurement is missing, which fails the check */
  observed: number | null;
  limit: number;
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
  /** position among this experiment's candidates by hard limits, then the objective, on the optimization split (1 = best) */
  rank: number;
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

/** Best-so-far objective value (see `objectiveValue`) after `evaluated` candidates, optimization split. */
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
  /** identity of "this optimization problem, set up this way"; null until the backend assigns one */
  identityHash: string | null;
  progress: SearchProgress;
  /** fixed reference workflow, on the optimization split */
  baseline: Measurement | null;
  /** the baseline's objective value, on the same scale as `curve` */
  curveBaseline: number | null;
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

export interface SplitResult {
  measurement: Measurement;
  constraints: ConstraintCheck[];
}

export interface MethodResult {
  method: MethodId;
  stages: WorkflowStage[];
  model: string;
  /** candidates the method evaluated; 1 for the fixed baseline */
  evaluated: number;
  /** `test` is reporting only: it is filled after the champion is fixed and never feeds back */
  splits: Partial<Record<SplitId, SplitResult>>;
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
