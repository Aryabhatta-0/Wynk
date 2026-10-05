import type { ColumnMapping, ExperimentConfig } from "@/api/types";

/** Problems with a column mapping, in the order a person should fix them. Empty when valid. */
export function validateMapping(m: ColumnMapping, columns: string[]): string[] {
  const errors: string[] = [];
  const known = new Set(columns);
  const unknown = [...m.input, ...m.context, ...(m.target ? [m.target] : [])].filter((c) => !known.has(c));
  if (unknown.length) errors.push(`Unknown column${unknown.length > 1 ? "s" : ""}: ${unknown.join(", ")}.`);
  if (!m.input.length) errors.push("Choose at least one input column.");
  if (!m.target) errors.push("Choose the target column the workflow must produce.");
  if (m.target && (m.input.includes(m.target) || m.context.includes(m.target)))
    errors.push("The target column cannot also be an input or context column.");
  const both = m.input.filter((c) => m.context.includes(c));
  if (both.length) errors.push(`${both.join(", ")} cannot be both input and context.`);
  return errors;
}

export type ConfigField =
  | "name"
  | "datasetId"
  | "allowedModels"
  | "minQuality"
  | "maxCostPer1k"
  | "maxLatencyP95Ms"
  | "maxCandidates"
  | "maxGenerations"
  | "maxSpendUsd"
  | "maxDurationMin"
  | "splits";

export type ConfigErrors = Partial<Record<ConfigField, string>>;

const positive = (v: number) => Number.isFinite(v) && v > 0;
const wholePositive = (v: number) => Number.isInteger(v) && v > 0;

export function validateConfig(c: ExperimentConfig): ConfigErrors {
  const e: ConfigErrors = {};
  if (!c.name.trim()) e.name = "Name the experiment.";
  if (!c.datasetId) e.datasetId = "Choose a dataset that has a complete column mapping.";

  const k = c.constraints;
  if (!k.allowedModels.length) e.allowedModels = "Allow at least one model.";
  if (k.minQuality !== null && !(k.minQuality > 0 && k.minQuality <= 1)) e.minQuality = "Enter a quality between 0 and 1.";
  if (k.maxCostPer1k !== null && !positive(k.maxCostPer1k)) e.maxCostPer1k = "Enter a cost above 0.";
  if (k.maxLatencyP95Ms !== null && !positive(k.maxLatencyP95Ms)) e.maxLatencyP95Ms = "Enter a latency above 0.";

  const b = c.budget;
  if (!wholePositive(b.maxCandidates)) e.maxCandidates = "Enter a whole number of candidates above 0.";
  if (!wholePositive(b.maxGenerations)) e.maxGenerations = "Enter a whole number of generations above 0.";
  else if (wholePositive(b.maxCandidates) && b.maxGenerations > b.maxCandidates)
    e.maxGenerations = "Generations cannot exceed candidates: each generation evaluates at least one.";
  if (!positive(b.maxSpendUsd)) e.maxSpendUsd = "Enter a spend limit above 0.";
  if (!positive(b.maxDurationMin)) e.maxDurationMin = "Enter a duration above 0.";

  const s = c.splits;
  const parts = [s.optimization, s.validation, s.test];
  if (parts.some((p) => !Number.isInteger(p) || p < 5)) e.splits = "Each split needs at least 5% of the rows.";
  else if (parts.reduce((a, p) => a + p, 0) !== 100) e.splits = "Splits must add up to 100%.";
  return e;
}
