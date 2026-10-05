/*
  MOCK ONLY. Stands in for the server's upload inspection (ingestion/parse.py) so the mock adapter
  can accept a file. The live adapter never uses this: in live mode the product API parses the
  bytes and every type, null count, row count and hash comes from the server.

  Rules follow ingestion/parse.py closely enough for demos and tests, with the same error codes:
  strict UTF-8, CSV header + equal field counts (an empty cell is null), JSONL one object per line,
  identifier column names, no duplicate columns, at least one row.
*/
import { ApiError } from "../client";
import { COLUMN_NAME } from "../contract/rules";
import type { ColumnType, DatasetRow, InspectedColumn } from "../types";

export const PREVIEW_ROWS = 20;
export const PREVIEW_CELL_CHARS = 200;

export interface Inspected {
  columns: InspectedColumn[];
  rows: DatasetRow[];
  preview: DatasetRow[];
}

const reject = (code: string, message: string, details: Record<string, string | number | null> = {}): never => {
  throw new ApiError(code, message, { details });
};

export function inspect(bytes: Uint8Array, format: "csv" | "jsonl"): Inspected {
  if (!bytes.length) reject("empty_file", "the file is empty");
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return reject("invalid_encoding", "the file is not valid UTF-8");
  }
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  if (text.includes("\0")) reject("invalid_encoding", "the file contains NUL characters");
  if (!text.trim()) reject("empty_file", "the file holds no data");
  const { names, rows } = format === "csv" ? readCsv(text) : readJsonl(text);
  checkColumns(names);
  if (!rows.length) reject("no_rows", "the file has no data rows");
  const columns = names.map((name): InspectedColumn => {
    const present = rows.map((r) => r[name]).filter((v) => v !== null && v !== undefined);
    const nullCount = rows.length - present.length;
    return { name, type: present.length ? infer(present, format) : "string", nullable: nullCount > 0, nullCount };
  });
  const cell = (v: unknown) => {
    const s = typeof v === "string" ? v : v !== null && typeof v === "object" ? JSON.stringify(v) : null;
    return s !== null && s.length > PREVIEW_CELL_CHARS ? `${s.slice(0, PREVIEW_CELL_CHARS)}…` : v;
  };
  const preview = rows.slice(0, PREVIEW_ROWS).map((r) => Object.fromEntries(names.map((n) => [n, cell(r[n])])));
  return { columns, rows, preview };
}

function checkColumns(names: string[]) {
  const seen = new Set<string>();
  for (const name of names) {
    if (!COLUMN_NAME.test(name))
      reject("invalid_column_name", `column name '${name}' must start with a letter or _ and use only letters, digits and _ (at most 64 characters)`, {
        column: name,
      });
    if (seen.has(name)) reject("duplicate_column", `column '${name}' appears twice`, { column: name });
    seen.add(name);
  }
}

/* ------------------------------------------------------------- CSV (RFC 4180) */

export function parseCsvRecords(text: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;
  let touched = false; // the record has content; a blank line is an empty record, as in Python's csv
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch !== "\n" && ch !== "\r") touched = true;
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
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
      if (touched) record.push(field);
      records.push(record);
      record = [];
      field = "";
      touched = false;
      if (ch === "\r" && text[i + 1] === "\n") i++;
    } else field += ch;
    i++;
  }
  if (quoted) reject("malformed_csv", "a quoted field is never closed");
  if (field !== "" || record.length) {
    record.push(field);
    records.push(record);
  }
  return records;
}

function readCsv(text: string): { names: string[]; rows: DatasetRow[] } {
  const [header, ...body] = parseCsvRecords(text);
  if (!header?.length) return reject("malformed_csv", "the first line must be a header", { line: 1 });
  const rows = body.map((cells, r) => {
    if (cells.length !== header.length)
      reject("malformed_csv", `line ${r + 2} has ${cells.length} fields, the header has ${header.length}`, { line: r + 2 });
    return Object.fromEntries(header.map((n, i) => [n, cells[i] === "" ? null : cells[i]]));
  });
  return { names: header, rows };
}

/* ------------------------------------------------------------- JSONL */

function readJsonl(text: string): { names: string[]; rows: DatasetRow[] } {
  const lines = text.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  const names: string[] = [];
  const objects: Record<string, unknown>[] = [];
  lines.forEach((raw, i) => {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (!line.trim()) reject("malformed_jsonl", `line ${i + 1} is blank`, { line: i + 1 });
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (err) {
      reject("malformed_jsonl", `line ${i + 1}: ${err instanceof Error ? err.message : "invalid JSON"}`, { line: i + 1 });
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) reject("malformed_jsonl", `line ${i + 1} is not a JSON object`, { line: i + 1 });
    for (const k of Object.keys(value as object)) if (!names.includes(k)) names.push(k);
    objects.push(value as Record<string, unknown>);
  });
  return { names, rows: objects.map((o) => Object.fromEntries(names.map((n) => [n, o[n] ?? null]))) };
}

/* ------------------------------------------------------------- types */

const INTEGER = /^(0|-?[1-9][0-9]*)$/;
const NUMBER = /^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const isDate = (s: string) => DATE.test(s) && !Number.isNaN(Date.parse(s));

function infer(values: unknown[], format: "csv" | "jsonl"): ColumnType {
  const all = (pred: (v: unknown) => boolean) => values.every(pred);
  if (format === "csv") {
    const s = values as string[];
    if (s.every((v) => /^(true|false)$/i.test(v))) return "boolean";
    if (s.every((v) => INTEGER.test(v))) return "integer";
    if (s.every((v) => NUMBER.test(v))) return "number";
    if (s.every(isDate)) return "date";
    return "string";
  }
  if (all((v) => typeof v === "boolean")) return "boolean";
  if (all((v) => Number.isInteger(v))) return "integer";
  if (all((v) => typeof v === "number")) return "number";
  if (all((v) => typeof v === "string")) return all((v) => isDate(v as string)) ? "date" : "string";
  if (all((v) => Array.isArray(v) && v.every((x) => typeof x === "string"))) return "string_list";
  return "json";
}
