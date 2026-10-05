/*
  Contract rules the UI needs before anything is sent: form validation and the mock adapter use
  them so the UI never offers a configuration the backend contracts would reject. Each rule names
  the Python rule it mirrors; the backend stays the authority and validates everything again.
*/
import type { ColumnMapping, ColumnType, DatasetColumn, EvaluatorKind, Objective, SplitId, SplitPlan, TaskType } from "../types";
import type { SplitUseWire } from "./wire";

/** core/dataset.py COLUMN_NAME: identifier-like, so a column maps 1:1 onto a schema field. */
export const COLUMN_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;

/** core/task_contract.py MAX_INSTRUCTIONS_CHARS */
export const MAX_INSTRUCTIONS_CHARS = 20_000;

/** core/dataset.py ALLOWED_USES: what each split's results may be used for. */
export const SPLIT_USES: Record<SplitId, readonly SplitUseWire[]> = {
  optimization: ["optimizer_feedback", "selection", "reporting"],
  validation: ["selection", "reporting"],
  test: ["reporting"],
};

/** core/dataset.py BPS and _assign: test and validation get floor(n·bps/10000) rows, optimization the rest. */
export const BPS = 10_000;

export function splitSizes(rows: number, plan: SplitPlan): Record<SplitId, number> {
  const test = Math.floor((rows * plan.testPct * 100) / BPS);
  const validation = Math.floor((rows * plan.validationPct * 100) / BPS);
  return { optimization: rows - test - validation, validation, test };
}

/** core/dataset.py: id_column must be a non-nullable string or integer column. */
export function canBeId(c: DatasetColumn): boolean {
  return !c.nullable && (c.type === "string" || c.type === "integer");
}

/** core/dataset.py ColumnType.field_type: json has no schema field type, so it can take no role. */
export const canTakeRole = (c: DatasetColumn): boolean => c.type !== "json" && COLUMN_NAME.test(c.name);

/** Mapping problems under the DatasetSpec rules, in the order a person should fix them. */
export function mappingProblems(m: ColumnMapping, columns: DatasetColumn[]): string[] {
  const out: string[] = [];
  const byName = new Map(columns.map((c) => [c.name, c]));
  const used = [...m.input, ...m.target, ...m.context, ...(m.id ? [m.id] : [])];
  const unknown = used.filter((c) => !byName.has(c));
  if (unknown.length) out.push(`Unknown column${unknown.length > 1 ? "s" : ""}: ${unknown.join(", ")}.`);
  const dupes = used.filter((c, i) => used.indexOf(c) !== i);
  if (dupes.length) out.push(`${[...new Set(dupes)].join(", ")} can have only one role.`);
  if (!m.input.length) out.push("Choose at least one input column.");
  if (!m.target.length) out.push("Choose at least one target column for the workflow to produce.");
  const badName = used.filter((c) => byName.has(c) && !COLUMN_NAME.test(c));
  if (badName.length)
    out.push(`Rename ${badName.join(", ")}: a mapped column name must start with a letter or _ and use only letters, digits and _.`);
  const json = used.filter((c) => byName.get(c)?.type === "json");
  if (json.length) out.push(`${json.join(", ")} holds nested JSON, which cannot be mapped. Split it into one column per field.`);
  const id = m.id ? byName.get(m.id) : undefined;
  if (id && !canBeId(id)) out.push("The id column must be a string or integer column with a value in every row.");
  return out;
}

/** core/task_contract.py _check_task_type: output shape each task type accepts. */
export function targetProblems(taskType: TaskType, targetTypes: ColumnType[]): string[] {
  if (taskType === "classification") {
    if (targetTypes.length !== 1 || targetTypes[0] !== "string") return ["Classification needs exactly one text target column."];
  } else if (taskType === "question_answering") {
    if (targetTypes.length !== 1 || !["string", "integer", "number"].includes(targetTypes[0]))
      return ["Question answering needs exactly one text or numeric target column."];
  } else if (!targetTypes.length) {
    return ["Structured extraction needs at least one target column."];
  }
  return [];
}

/** core/task_contract.py: evaluators that can judge each task type's output. */
export function evaluatorsFor(taskType: TaskType, targetTypes: ColumnType[]): EvaluatorKind[] {
  if (taskType === "classification") return ["classification_accuracy", "exact_match"];
  if (taskType === "question_answering") {
    if (targetTypes[0] === "string") return ["token_f1", "exact_match"];
    if (targetTypes[0] === "integer" || targetTypes[0] === "number") return ["numeric_tolerance", "exact_match"];
    return [];
  }
  const numeric = targetTypes.length > 0 && targetTypes.every((t) => t === "integer" || t === "number");
  return ["exact_match", "json_schema_validity", ...(numeric ? (["numeric_tolerance"] as const) : [])];
}

/** core/task_contract.py _check_objective: minimizing cost or latency needs a quality floor. */
export const objectiveNeedsQualityFloor = (o: Objective): boolean => o === "cost" || o === "latency";
