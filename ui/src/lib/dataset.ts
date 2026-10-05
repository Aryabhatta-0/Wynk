import type { ColumnMapping, ColumnType, DatasetColumn, DatasetFormat, DatasetRow, TaskType } from "@/api/types";

/*
  Client-side reading of a dataset file, for preview and column mapping only. Nothing is
  uploaded: the file never leaves the browser. Parquet is binary and is read by Wynk's
  ingestion service once it exists, so here it is recognised but not previewed.
*/

export const MAX_PREVIEW_BYTES = 20 * 1024 * 1024;
export const PREVIEW_ROWS = 25;

export function detectFormat(fileName: string): DatasetFormat | null {
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

export function parseDataset(text: string, format: "csv" | "jsonl"): ParsedDataset {
  const { names, rows } = format === "csv" ? readCsv(text) : readJsonl(text);
  if (!names.length) throw new ParseError("The file has no columns.");
  if (!rows.length) throw new ParseError("The file has a header but no rows.");
  const preview = rows.slice(0, PREVIEW_ROWS);
  return { rowCount: rows.length, columns: inferColumns(names, preview), preview };
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

function typeOf(v: unknown): ColumnType {
  if (isMissing(v)) return "empty";
  if (typeof v === "boolean") return "boolean";
  if (typeof v === "number") return "number";
  if (typeof v === "object") return "json";
  const s = String(v).trim();
  if (/^(true|false)$/i.test(s)) return "boolean";
  if (/^-?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(s)) return "number";
  return "string";
}

export function inferColumns(names: string[], rows: DatasetRow[]): DatasetColumn[] {
  return names.map((name) => {
    const values = rows.map((r) => r[name]);
    const types = new Set(values.map(typeOf).filter((t) => t !== "empty"));
    const type: ColumnType = types.size === 0 ? "empty" : types.size === 1 ? [...types][0] : types.has("json") ? "json" : "string";
    return {
      name,
      type,
      missing: values.filter(isMissing).length,
      distinct: new Set(values.filter((v) => !isMissing(v)).map((v) => JSON.stringify(v))).size,
    };
  });
}

/* ------------------------------------------------------------- suggestions */

const TARGET_NAMES = /^(label|labels|target|answer|answers|output|expected|gold|class|category|queue|intent|fields|y)$/i;
const INPUT_NAMES = /(text|question|query|input|prompt|body|subject|document|content|message|title)/i;
const CONTEXT_NAMES = /(context|passage|source|evidence|tier|metadata|notes)/i;
const ID_NAMES = /(^id$|_id$|^uuid$|^key$)/i;

/** A starting mapping from column names. The person always confirms it. */
export function suggestMapping(columns: DatasetColumn[]): ColumnMapping {
  const names = columns.map((c) => c.name);
  const target = names.find((n) => TARGET_NAMES.test(n)) ?? null;
  const rest = names.filter((n) => n !== target && !ID_NAMES.test(n));
  const context = rest.filter((n) => CONTEXT_NAMES.test(n));
  let input = rest.filter((n) => INPUT_NAMES.test(n) && !context.includes(n));
  if (!input.length) input = rest.filter((n) => !context.includes(n)).slice(0, 1);
  return { input, target, context };
}

/** Task type suggested by the target column. The person always confirms it. */
export function suggestTaskType(columns: DatasetColumn[], target: string | null, previewRows: number): TaskType {
  const col = columns.find((c) => c.name === target);
  if (!col) return "question_answering";
  if (col.type === "json") return "structured_extraction";
  if (col.type === "boolean" || (col.distinct <= Math.max(2, Math.floor(previewRows * 0.7)) && col.distinct <= 20)) return "classification";
  return "question_answering";
}

/** Distinct target labels seen in the preview, most common first. */
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

/** Field names of a JSON target column across the preview. */
export function fieldsIn(rows: DatasetRow[], target: string): string[] {
  const out: string[] = [];
  for (const r of rows) {
    const v = r[target];
    const obj = typeof v === "string" ? safeJson(v) : v;
    if (obj && typeof obj === "object" && !Array.isArray(obj)) for (const k of Object.keys(obj)) if (!out.includes(k)) out.push(k);
  }
  return out;
}

function safeJson(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

export function cellText(v: unknown): string {
  if (isMissing(v)) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
