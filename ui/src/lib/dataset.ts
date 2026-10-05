import type { ColumnMapping, ColumnType, DatasetColumn, DatasetRow, TaskType } from "@/api/types";
import { canBeId, canTakeRole } from "@/api/contract/rules";

/*
  Client-side reading of a dataset file, for preview, schema and column roles only. Nothing is
  uploaded: the file never leaves the browser. Types, nullability and the content hash follow
  the dataset contract (core/dataset.py): column types string / integer / number / boolean / date /
  string_list / json, sha256 of the file bytes. Parquet is not a contract format yet, so it is
  recognised here only to say so.
*/

export const MAX_PREVIEW_BYTES = 20 * 1024 * 1024;
export const PREVIEW_ROWS = 25;

export type FileKind = "csv" | "jsonl" | "parquet";

export function detectFormat(fileName: string): FileKind | null {
  const ext = fileName.toLowerCase().split(".").pop();
  if (ext === "csv") return "csv";
  if (ext === "jsonl" || ext === "ndjson") return "jsonl";
  if (ext === "parquet") return "parquet";
  return null;
}

export interface ParsedDataset {
  rowCount: number;
  columns: DatasetColumn[];
  preview: DatasetRow[];
}

export class ParseError extends Error {}

/** Parses the whole file; types, nullability and counts are over every row, the preview is the first rows. */
export function parseDataset(text: string, format: "csv" | "jsonl"): ParsedDataset {
  const { names, rows } = format === "csv" ? readCsv(text) : readJsonl(text);
  if (!names.length) throw new ParseError("The file has no columns.");
  if (!rows.length) throw new ParseError("The file has a header but no rows.");
  return { rowCount: rows.length, columns: inferColumns(names, rows), preview: rows.slice(0, PREVIEW_ROWS) };
}

/** sha256 of the file bytes (the dataset contract's content_hash), or null without Web Crypto. */
export async function contentHash(bytes: ArrayBuffer): Promise<string | null> {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/* ------------------------------------------------------------- CSV (RFC 4180) */

export function parseCsvRecords(text: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;
  const src = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  while (i < src.length) {
    const ch = src[i];
    if (quoted) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
      } else field += ch;
      i++;
      continue;
    }
    if (ch === '"' && field === "") quoted = true;
    else if (ch === ",") {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      record.push(field);
      records.push(record);
      record = [];
      field = "";
      if (ch === "\r" && src[i + 1] === "\n") i++;
    } else field += ch;
    i++;
  }
  if (quoted) throw new ParseError("A quoted field is never closed.");
  if (field !== "" || record.length) {
    record.push(field);
    records.push(record);
  }
  return records.filter((r) => !(r.length === 1 && r[0] === ""));
}

function readCsv(text: string): { names: string[]; rows: DatasetRow[] } {
  const [header, ...body] = parseCsvRecords(text);
  if (!header) return { names: [], rows: [] };
  const names = header.map((h, i) => h.trim() || `column_${i + 1}`);
  const dupes = names.filter((n, i) => names.indexOf(n) !== i);
  if (dupes.length) throw new ParseError(`Duplicate column name: ${dupes[0]}.`);
  const rows = body.map((cells, r) => {
    if (cells.length > names.length) throw new ParseError(`Row ${r + 2} has ${cells.length} fields; the header has ${names.length}.`);
    return Object.fromEntries(names.map((n, i) => [n, cells[i] ?? ""]));
  });
  return { names, rows };
}

/* ------------------------------------------------------------- JSONL */

function readJsonl(text: string): { names: string[]; rows: DatasetRow[] } {
  const names: string[] = [];
  const seen = new Set<string>();
  const rows: DatasetRow[] = [];
  text.split(/\r?\n/).forEach((line, i) => {
    if (!line.trim()) return;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      throw new ParseError(`Line ${i + 1} is not valid JSON.`);
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new ParseError(`Line ${i + 1} is not a JSON object.`);
    for (const k of Object.keys(value)) {
      if (!seen.has(k)) {
        seen.add(k);
        names.push(k);
      }
    }
    rows.push(value as DatasetRow);
  });
  return { names, rows };
}

/* ------------------------------------------------------------- schema */

const isMissing = (v: unknown) => v === null || v === undefined || v === "";
const DATE = /^\d{4}-\d{2}-\d{2}$/;

/** The narrowest contract column type a single value fits. CSV cells are text, so they are read. */
function typeOf(v: unknown): ColumnType | null {
  if (isMissing(v)) return null;
  if (typeof v === "boolean") return "boolean";
  if (typeof v === "number") return Number.isInteger(v) ? "integer" : "number";
  if (Array.isArray(v)) return v.every((x) => typeof x === "string") ? "string_list" : "json";
  if (typeof v === "object") return "json";
  const s = String(v).trim();
  if (/^(true|false)$/i.test(s)) return "boolean";
  if (/^-?\d+$/.test(s)) return "integer";
  if (/^-?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(s)) return "number";
  if (DATE.test(s) && !Number.isNaN(Date.parse(s))) return "date";
  return "string";
}

function unify(types: Set<ColumnType>): ColumnType {
  if (types.size === 0) return "string";
  if (types.size === 1) return [...types][0];
  if (types.has("json")) return "json";
  if ([...types].every((t) => t === "integer" || t === "number")) return "number";
  return "string";
}

export function inferColumns(names: string[], rows: DatasetRow[]): DatasetColumn[] {
  return names.map((name) => {
    const values = rows.map((r) => r[name]);
    const missing = values.filter(isMissing).length;
    const types = new Set(values.map(typeOf).filter((t): t is ColumnType => t !== null));
    return {
      name,
      type: unify(types),
      nullable: missing > 0,
      missing,
      distinct: new Set(values.filter((v) => !isMissing(v)).map((v) => JSON.stringify(v))).size,
    };
  });
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

/** Task type suggested by the target columns. The person always confirms it. */
export function suggestTaskType(columns: DatasetColumn[], targets: string[], rowCount: number): TaskType {
  if (targets.length !== 1) return "structured_extraction";
  const col = columns.find((c) => c.name === targets[0]);
  if (!col) return "question_answering";
  if (col.type === "string" && col.distinct >= 2 && col.distinct <= 20 && col.distinct <= Math.max(2, Math.floor(rowCount * 0.7)))
    return "classification";
  if (col.type === "string" || col.type === "integer" || col.type === "number") return "question_answering";
  return "structured_extraction";
}

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
