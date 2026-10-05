import type { ConstraintKey, ExperimentStatus, MethodId, Objective, SplitId, StageKind, TaskType, WorkflowStage } from "@/api/types";

export const pct = (q: number, digits = 1) => `${(q * 100).toFixed(digits)}%`;

export const usd = (v: number) => (v >= 100 ? `$${v.toFixed(0)}` : v >= 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(3)}`);

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10_000 ? 0 : 1)} s` : `${Math.round(v)} ms`);

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

export const TASK_METRIC: Record<TaskType, string> = {
  classification: "accuracy",
  structured_extraction: "field match",
  question_answering: "answer match",
};

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
  maxCostPer1k: "Maximum cost",
  maxLatencyP95Ms: "Maximum p95 latency",
  allowedModels: "Allowed models",
};

export function constraintValue(key: ConstraintKey, v: number | string): string {
  if (typeof v === "string") return v;
  if (key === "minQuality") return pct(v);
  if (key === "maxCostPer1k") return `${usd(v)} / 1k`;
  if (key === "maxLatencyP95Ms") return ms(v);
  return String(v);
}

export const STAGE_LABEL: Record<StageKind, string> = {
  GATHER: "Gather",
  FILTER: "Filter",
  EXTRACT: "Extract",
  REASON: "Reason",
  VERIFY: "Verify",
  SYNTHESIZE: "Synthesize",
};

export const optionLabel = (o: string) => o.replaceAll("_", " ");

/** "Extract (schema guided) → Verify (schema check) → Synthesize (direct)" */
export function workflowText(stages: WorkflowStage[]): string {
  return stages.map((s) => `${STAGE_LABEL[s.kind]} (${s.options.map(optionLabel).join(", ")})`).join(" → ");
}
