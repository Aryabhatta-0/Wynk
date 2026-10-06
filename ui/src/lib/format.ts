import type {
  ConstraintKey,
  EvaluationConfig,
  EvaluatorKind,
  ExperimentStatus,
  MethodId,
  Objective,
  SplitId,
  StageKind,
  TaskType,
  WorkflowStage,
} from "@/api/types";

/** Shown for a measurement that is null: not measured, which is never the same as zero. */
export const NOT_MEASURED = "—";

const orDash =
  (f: (v: number) => string) =>
  (v: number | null): string =>
    v === null ? NOT_MEASURED : f(v);

export const pct = (q: number | null, digits = 1) => (q === null ? NOT_MEASURED : `${(q * 100).toFixed(digits)}%`);

export const usd = (v: number) => "$" + (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toFixed(3));

/**
 * Cost is measured per example (the contract's unit) but a single example costs fractions of a
 * cent, so screens show it per 1,000 examples. This is the only place that conversion happens.
 */
export const EXAMPLES_PER_COST_UNIT = 1000;
export const costPer1k = orDash((perExample: number) => `${usd(perExample * EXAMPLES_PER_COST_UNIT)} / 1k`);
export const costPer1kNumber = (perExample: number) => usd(perExample * EXAMPLES_PER_COST_UNIT);

/** Latency is measured in seconds per example. */
export const secs = orDash((s: number) => (s >= 1 ? `${s.toFixed(s >= 10 ? 0 : 1)} s` : `${Math.round(s * 1000)} ms`));

export const tokens = orDash((t: number) => `${Math.round(t).toLocaleString("en-US")} tok`);

export const int = (v: number) => v.toLocaleString("en-US");

export function duration(sec: number): string {
  if (sec < 60) return `${Math.round(sec)} s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m} min ${Math.round(sec % 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function date(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function relative(iso: string, now = Date.now()): string {
  const s = Math.round((now - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86_400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86_400);
  return d === 1 ? "yesterday" : `${d} days ago`;
}

/** A relative change, signed, e.g. "+12%" or "−8%". */
export function change(now: number, before: number): string {
  if (before === 0) return "n/a";
  const r = (now - before) / before;
  const v = Math.abs(r * 100);
  return `${r >= 0 ? "+" : "−"}${v < 10 ? v.toFixed(1) : v.toFixed(0)}%`;
}

/** An absolute change in quality, in percentage points. */
export function points(now: number, before: number): string {
  const d = (now - before) * 100;
  return `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} pts`;
}

export const TASK_LABEL: Record<TaskType, string> = {
  classification: "Classification",
  structured_extraction: "Structured extraction",
  question_answering: "Question answering",
};

export const EVALUATOR_LABEL: Record<EvaluatorKind, string> = {
  classification_accuracy: "Classification accuracy",
  exact_match: "Exact match",
  token_f1: "Token F1",
  json_schema_validity: "JSON schema validity",
  numeric_tolerance: "Numeric tolerance",
};

/** What "quality" means for an evaluator: the mean of its per-example score. */
export const EVALUATOR_METRIC: Record<EvaluatorKind, string> = {
  classification_accuracy: "accuracy",
  exact_match: "exact match",
  token_f1: "token F1",
  json_schema_validity: "schema validity",
  numeric_tolerance: "within tolerance",
};

export const metricName = (e: EvaluationConfig) => EVALUATOR_METRIC[e.evaluator];

export const OBJECTIVE_LABEL: Record<Objective, string> = {
  quality: "Quality",
  cost: "Cost",
  latency: "Latency",
  balanced: "Balanced",
};

export const METHOD_LABEL: Record<MethodId, string> = {
  fixed_baseline: "Fixed baseline",
  random_search: "Random search",
  wynk_aco: "Wynk (ACO)",
};

export const SPLIT_LABEL: Record<SplitId, string> = {
  optimization: "Optimization",
  validation: "Validation",
  test: "Held-out test",
};

export const STATUS_LABEL: Record<ExperimentStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export const CONSTRAINT_LABEL: Record<ConstraintKey, string> = {
  minQuality: "Minimum quality",
  maxCostPerExample: "Maximum cost",
  maxMeanLatencyS: "Maximum mean latency",
  maxP95LatencyS: "Maximum p95 latency",
  maxTokensPerExample: "Maximum tokens per example",
  maxWorkflowSteps: "Maximum workflow steps",
};

/** Short names for tight table cells. */
export const CONSTRAINT_SHORT: Record<ConstraintKey, string> = {
  minQuality: "quality",
  maxCostPerExample: "cost",
  maxMeanLatencyS: "mean latency",
  maxP95LatencyS: "p95",
  maxTokensPerExample: "tokens",
  maxWorkflowSteps: "steps",
};

export function constraintValue(key: ConstraintKey, v: number | null): string {
  if (v === null) return "not measured";
  if (key === "minQuality") return pct(v);
  if (key === "maxCostPerExample") return costPer1k(v);
  if (key === "maxMeanLatencyS" || key === "maxP95LatencyS") return secs(v);
  if (key === "maxTokensPerExample") return tokens(v);
  return String(v);
}

export const STAGE_LABEL: Record<StageKind, string> = {
  GATHER: "Gather",
  FILTER: "Filter",
  EXTRACT: "Extract",
  REASON: "Reason",
  VERIFY: "Verify",
  SYNTHESIZE: "Synthesize",
  DIRECT: "Direct",
  CONFIDENCE_GATE: "Confidence gate",
};

export const optionLabel = (o: string) => o.replaceAll("_", " ");

/** "Extract (schema guided) → Verify (schema check) → Synthesize (direct)" */
export function workflowText(stages: WorkflowStage[]): string {
  return stages.map((s) => `${STAGE_LABEL[s.kind]} (${s.options.map(optionLabel).join(", ")})`).join(" → ");
}
