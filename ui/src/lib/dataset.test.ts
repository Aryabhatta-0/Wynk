import { describe, expect, it } from "vitest";
import {
  ParseError,
  detectFormat,
  fieldsIn,
  inferColumns,
  labelsIn,
  parseCsvRecords,
  parseDataset,
  suggestMapping,
  suggestTaskType,
} from "./dataset";

describe("detectFormat", () => {
  it("recognises the three supported formats by extension", () => {
    expect(detectFormat("a.CSV")).toBe("csv");
    expect(detectFormat("a.jsonl")).toBe("jsonl");
    expect(detectFormat("a.ndjson")).toBe("jsonl");
    expect(detectFormat("a.parquet")).toBe("parquet");
    expect(detectFormat("a.xlsx")).toBeNull();
  });
});

describe("parseCsvRecords", () => {
  it("handles quotes, escaped quotes, commas and newlines inside fields, CRLF and a BOM", () => {
    const text = '﻿a,b\r\n"x, y","say ""hi"""\r\n"line1\nline2",2\r\n';
    expect(parseCsvRecords(text)).toEqual([
      ["a", "b"],
      ["x, y", 'say "hi"'],
      ["line1\nline2", "2"],
    ]);
  });

  it("keeps a final row without a trailing newline and skips blank lines", () => {
    expect(parseCsvRecords("a\n\n1")).toEqual([["a"], ["1"]]);
  });

  it("rejects an unclosed quote", () => {
    expect(() => parseCsvRecords('a\n"open')).toThrow(ParseError);
  });
});

describe("parseDataset", () => {
  it("reads a CSV into rows, a count and inferred columns", () => {
    const parsed = parseDataset("text,label,score\nhello,pos,1.5\nbye,neg,\n", "csv");
    expect(parsed.rowCount).toBe(2);
    expect(parsed.preview[0]).toEqual({ text: "hello", label: "pos", score: "1.5" });
    expect(parsed.columns).toEqual([
      { name: "text", type: "string", missing: 0, distinct: 2 },
      { name: "label", type: "string", missing: 0, distinct: 2 },
      { name: "score", type: "number", missing: 1, distinct: 1 },
    ]);
  });

  it("rejects duplicate CSV headers and rows wider than the header", () => {
    expect(() => parseDataset("a,a\n1,2", "csv")).toThrow(/Duplicate column/);
    expect(() => parseDataset("a,b\n1,2,3", "csv")).toThrow(/Row 2 has 3 fields/);
  });

  it("reads JSONL, taking the union of keys in order of appearance", () => {
    const parsed = parseDataset('{"q":"a","fields":{"x":1}}\n\n{"q":"b","extra":true}\n', "jsonl");
    expect(parsed.rowCount).toBe(2);
    expect(parsed.columns.map((c) => [c.name, c.type])).toEqual([
      ["q", "string"],
      ["fields", "json"],
      ["extra", "boolean"],
    ]);
  });

  it("names the JSONL line that is broken", () => {
    expect(() => parseDataset('{"a":1}\nnot json', "jsonl")).toThrow("Line 2 is not valid JSON.");
    expect(() => parseDataset("[1,2]", "jsonl")).toThrow("Line 1 is not a JSON object.");
  });

  it("rejects a header with no rows", () => {
    expect(() => parseDataset("a,b\n", "csv")).toThrow(/no rows/);
  });
});

describe("suggestions", () => {
  const columns = inferColumns(
    ["ticket_id", "subject", "body", "customer_tier", "queue"],
    [
      { ticket_id: "1", subject: "s", body: "b", customer_tier: "pro", queue: "billing" },
      { ticket_id: "2", subject: "s2", body: "b2", customer_tier: "free", queue: "billing" },
      { ticket_id: "3", subject: "s3", body: "b3", customer_tier: "pro", queue: "technical" },
    ],
  );

  it("suggests target, inputs and context from column names, skipping ids", () => {
    expect(suggestMapping(columns)).toEqual({ input: ["subject", "body"], target: "queue", context: ["customer_tier"] });
  });

  it("suggests classification for a low-cardinality target and extraction for JSON", () => {
    expect(suggestTaskType(columns, "queue", 3)).toBe("classification");
    const json = inferColumns(["doc", "fields"], [{ doc: "x", fields: { a: 1 } }]);
    expect(suggestTaskType(json, "fields", 1)).toBe("structured_extraction");
    const free = inferColumns(
      ["q", "answer"],
      Array.from({ length: 10 }, (_, i) => ({ q: `q${i}`, answer: `a${i}` })),
    );
    expect(suggestTaskType(free, "answer", 10)).toBe("question_answering");
  });

  it("lists labels by frequency and the fields of JSON targets", () => {
    expect(labelsIn([{ y: "a" }, { y: "b" }, { y: "a" }, { y: "" }], "y")).toEqual([
      { label: "a", count: 2 },
      { label: "b", count: 1 },
    ]);
    expect(fieldsIn([{ t: { a: 1 } }, { t: '{"b":2}' }, { t: "nope" }], "t")).toEqual(["a", "b"]);
  });
});
