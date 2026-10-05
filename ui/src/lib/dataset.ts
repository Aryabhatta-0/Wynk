import type { ColumnMapping, DatasetColumn, DatasetRow, TaskType } from "@/api/types";
import { canBeId, canTakeRole } from "@/api/contract/rules";

/*
  Dataset helpers for screens: the upload format to request, and starting suggestions a person
  confirms. Nothing here reads a file or infers a schema: the server inspects every upload and its
  column types, nullability, row count and hash are shown as it reported them.
*/

/**
 * The `format` to send for a file name, from its extension. Sent as-is: the server accepts csv and
 * jsonl and answers anything else with `unsupported_format`, which the screen shows.
 */
export function uploadFormat(fileName: string): string {
  const ext = fileName.includes(".") ? fileName.toLowerCase().split(".").pop()! : "";
  return ext === "ndjson" ? "jsonl" : ext;
}

/** A dataset id (lowercase slug, core/dataset.py SLUG) suggested from a display name. */
export function slugFor(name: string): string {
  const slug = name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^[^a-z0-9]+|-+$/g, "")
    .slice(0, 64);
  return slug || "dataset";
}

/* ------------------------------------------------------------- suggestions */

const TARGET_NAMES = /^(label|labels|target|answer|answers|output|expected|gold|class|category|queue|intent|y)$/i;
const INPUT_NAMES = /(text|question|query|input|prompt|body|subject|document|content|message|title)/i;
const CONTEXT_NAMES = /(context|passage|source|evidence|tier|metadata|notes)/i;
const ID_NAMES = /(^id$|_id$|^uuid$|^key$)/i;

/** A starting mapping from column names and types. The person always confirms it. */
export function suggestMapping(columns: DatasetColumn[]): ColumnMapping {
  const usable = columns.filter(canTakeRole);
  const id = usable.find((c) => ID_NAMES.test(c.name) && canBeId(c))?.name ?? null;
  const names = usable.map((c) => c.name).filter((n) => n !== id && !ID_NAMES.test(n));
  const target = names.filter((n) => TARGET_NAMES.test(n)).slice(0, 1);
  const rest = names.filter((n) => !target.includes(n));
  const context = rest.filter((n) => CONTEXT_NAMES.test(n));
  let input = rest.filter((n) => INPUT_NAMES.test(n) && !context.includes(n));
  if (!input.length) input = rest.filter((n) => !context.includes(n)).slice(0, 1);
  return { input, target, context, id };
}

/** Task type suggested by the target columns and the preview rows. The person always confirms it. */
export function suggestTaskType(columns: DatasetColumn[], targets: string[], preview: DatasetRow[]): TaskType {
  if (targets.length !== 1) return "structured_extraction";
  const col = columns.find((c) => c.name === targets[0]);
  if (!col) return "question_answering";
  const distinct = labelsIn(preview, col.name).length;
  if (col.type === "string" && distinct >= 2 && distinct <= 20 && distinct <= Math.max(2, Math.floor(preview.length * 0.7))) return "classification";
  if (col.type === "string" || col.type === "integer" || col.type === "number") return "question_answering";
  return "structured_extraction";
}

const isMissing = (v: unknown) => v === null || v === undefined || v === "";

/** Distinct target labels seen in the given rows, most common first. */
export function labelsIn(rows: DatasetRow[], target: string): { label: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const r of rows) {
    const v = r[target];
    if (isMissing(v)) continue;
    const k = typeof v === "object" ? JSON.stringify(v) : String(v);
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return [...counts.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count);
}

export function cellText(v: unknown): string {
  if (isMissing(v)) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
