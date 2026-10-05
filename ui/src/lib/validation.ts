import type { ColumnMapping, Dataset, DatasetColumn, ExperimentConfig } from "@/api/types";
import { MAX_INSTRUCTIONS_CHARS, evaluatorsFor, mappingProblems, objectiveNeedsQualityFloor, targetProblems } from "@/api/contract/rules";

/** Problems with a column mapping, in the order a person should fix them. Empty when valid. */
export function validateMapping(m: ColumnMapping, columns: DatasetColumn[]): string[] {
  const problems = mappingProblems(m, columns);
  // the dataset contract makes the id optional, but an experiment splits its dataset and splitting needs one
  if (!m.id) problems.push("Choose an id column: rows are split into optimization, validation and test by their id.");
  return problems;
}

export type ConfigField =
  | "name"
  | "datasetId"
  | "taskType"
  | "instructions"
  | "evaluation"
  | "models"
  | "objective"
  | "minQuality"
  | "maxCostPerExample"
  | "maxMeanLatencyS"
  | "maxP95LatencyS"
  | "maxTokensPerExample"
  | "maxWorkflowSteps"
  | "maxCandidates"
  | "maxGenerations"
  | "maxSpendUsd"
  | "maxDurationMin"
  | "splits";

export type ConfigErrors = Partial<Record<ConfigField, string>>;

const positive = (v: number) => Number.isFinite(v) && v > 0;
const nonNegative = (v: number) => Number.isFinite(v) && v >= 0;
const wholePositive = (v: number) => Number.isInteger(v) && v > 0;

/** Every rule the task contract and the search settings impose, checked before anything is sent. */
export function validateConfig(c: ExperimentConfig, dataset?: Dataset): ConfigErrors {
  const e: ConfigErrors = {};
  if (!c.name.trim()) e.name = "Name the experiment.";
  if (!c.datasetId) e.datasetId = "Choose a dataset that has a complete column mapping.";
  if (!c.instructions.trim()) e.instructions = "Describe what the workflow should do with each row.";
  else if (c.instructions.length > MAX_INSTRUCTIONS_CHARS)
    e.instructions = `Keep instructions under ${MAX_INSTRUCTIONS_CHARS.toLocaleString("en-US")} characters.`;
  if (!c.models.length) e.models = "Allow at least one model.";

  // task type and evaluator against the dataset's target columns (core/task_contract.py)
  if (dataset) {
    const targetTypes = dataset.mapping.target.map((t) => dataset.columns.find((col) => col.name === t)?.type ?? "string");
    const shape = targetProblems(c.taskType, targetTypes);
    if (shape.length) e.taskType = shape[0];
    else if (!evaluatorsFor(c.taskType, targetTypes).includes(c.evaluation.evaluator))
      e.evaluation = "This evaluator cannot judge this task's output. Choose one of the listed evaluators.";
  }
  const ev = c.evaluation;
  if (!e.evaluation) {
    if (ev.evaluator === "classification_accuracy") {
      const labels = ev.labels.map((l) => l.trim());
      if (labels.length < 2 || labels.some((l) => !l)) e.evaluation = "List at least two labels.";
      else if (new Set(labels).size !== labels.length) e.evaluation = "Labels must be unique.";
    } else if (ev.evaluator === "token_f1" && !(ev.passThreshold > 0 && ev.passThreshold <= 1)) {
      e.evaluation = "The pass threshold must be above 0 and at most 1.";
    } else if (ev.evaluator === "numeric_tolerance" && !(nonNegative(ev.absoluteTolerance) && nonNegative(ev.relativeTolerance))) {
      e.evaluation = "Tolerances must be 0 or more.";
    }
  }

  // hard limits (core/constraints.py ConstraintLimits field rules)
  const k = c.constraints;
  if (k.minQuality !== null && !(k.minQuality >= 0 && k.minQuality <= 1)) e.minQuality = "Enter a quality between 0 and 100%.";
  if (k.maxCostPerExample !== null && !nonNegative(k.maxCostPerExample)) e.maxCostPerExample = "Enter a cost of 0 or more.";
  if (k.maxMeanLatencyS !== null && !positive(k.maxMeanLatencyS)) e.maxMeanLatencyS = "Enter a latency above 0.";
  if (k.maxP95LatencyS !== null && !positive(k.maxP95LatencyS)) e.maxP95LatencyS = "Enter a latency above 0.";
  if (k.maxTokensPerExample !== null && !wholePositive(k.maxTokensPerExample)) e.maxTokensPerExample = "Enter a whole number of tokens above 0.";
  if (k.maxWorkflowSteps !== null && !wholePositive(k.maxWorkflowSteps)) e.maxWorkflowSteps = "Enter a whole number of steps above 0.";

  // objective (core/objective.py ObjectiveSpec, core/task_contract.py _check_objective)
  const p = c.preferences;
  if (objectiveNeedsQualityFloor(p.objective) && k.minQuality === null && !e.minQuality)
    e.minQuality = `Minimizing ${p.objective} needs a minimum quality; otherwise a workflow that does nothing wins.`;
  if (p.objective === "balanced") {
    const b = p.balanced;
    if (!b) e.objective = "Set the balanced weights.";
    else {
      const weights = [b.quality, b.cost, b.latency, b.tokens];
      if (weights.some((w) => !Number.isInteger(w) || w < 0)) e.objective = "Weights are whole percentages of 0 or more.";
      else if (weights.reduce((s, w) => s + w, 0) !== 100) e.objective = "Weights must add up to 100%.";
      else if (b.quality <= 0) e.objective = "Quality needs a weight above 0.";
      else if (
        (
          [
            [b.cost, b.costScale],
            [b.latency, b.latencyScale],
            [b.tokens, b.tokensScale],
          ] as const
        ).some(([w, s]) => w > 0 && !(s !== null && positive(s)))
      )
        e.objective = "Every weighted cost, latency or token term needs a scale above 0.";
    }
  }

  // search settings (UI only; the backend has no budget contract yet)
  const b = c.budget;
  if (!wholePositive(b.maxCandidates)) e.maxCandidates = "Enter a whole number of candidates above 0.";
  if (!wholePositive(b.maxGenerations)) e.maxGenerations = "Enter a whole number of generations above 0.";
  else if (wholePositive(b.maxCandidates) && b.maxGenerations > b.maxCandidates)
    e.maxGenerations = "Generations cannot exceed candidates: each generation evaluates at least one.";
  if (!positive(b.maxSpendUsd)) e.maxSpendUsd = "Enter a spend limit above 0.";
  if (!positive(b.maxDurationMin)) e.maxDurationMin = "Enter a duration above 0.";

  // splits (core/dataset.py SplitPlan): validation and test must leave optimization rows; this
  // product also needs both, because champions are chosen on validation and reported on test
  const s = c.splits;
  if (![s.validationPct, s.testPct].every((v) => Number.isInteger(v) && v >= 5)) e.splits = "Validation and test each need a whole 5% or more.";
  else if (s.validationPct + s.testPct > 90) e.splits = "Leave at least 10% of rows for optimization.";
  else if (!Number.isInteger(s.seed) || s.seed < 0) e.splits = "The split seed is a whole number of 0 or more.";
  return e;
}
