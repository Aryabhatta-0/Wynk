import { describe, expect, it } from "vitest";
import {
  ParseError,
  contentHash,
  detectFormat,
  inferColumns,
  labelsIn,
  parseCsvRecords,
  parseDataset,
  suggestMapping,
  suggestTaskType,
} from "./dataset";

describe("detectFormat", () => {
  it("recognises CSV and JSONL, and Parquet only to say it is not a contract format", () => {
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
      { name: "text", type: "string", nullable: false, missing: 0, distinct: 2 },
      { name: "label", type: "string", nullable: false, missing: 0, distinct: 2 },
      { name: "score", type: "number", nullable: true, missing: 1, distinct: 1 },
    ]);
  });

  it("infers the dataset contract's column types over every row, not just the preview", () => {
    const rows = [
      "n,int,day,flag",
      ...Array.from({ length: 40 }, (_, i) => `${i},${i},2026-01-${String((i % 28) + 1).padStart(2, "0")},true`),
      "x,4.5,soon,false",
    ];
    const cols = parseDataset(rows.join("\n"), "csv").columns;
    // a string in row 41 makes n a string column even though the preview shows only integers
    expect(cols.map((c) => [c.name, c.type])).toEqual([
      ["n", "string"],
      ["int", "number"],
      ["day", "string"],
      ["flag", "boolean"],
    ]);
    const typed = parseDataset(
      '{"id":1,"day":"2026-03-01","tags":["a","b"],"meta":{"k":1}}\n{"id":2,"day":"2026-03-02","tags":[],"meta":[1]}',
      "jsonl",
    ).columns;
    expect(typed.map((c) => [c.name, c.type])).toEqual([
      ["id", "integer"],
      ["day", "date"],
      ["tags", "string_list"],
      ["meta", "json"],
    ]);
  });

  it("computes the content hash as sha256 of the file bytes", async () => {
    // sha256("abc")
    expect(await contentHash(new TextEncoder().encode("abc").buffer as ArrayBuffer)).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
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

  it("suggests target, inputs, context and the id column from names and types", () => {
    expect(suggestMapping(columns)).toEqual({ input: ["subject", "body"], target: ["queue"], context: ["customer_tier"], id: "ticket_id" });
  });

  it("never suggests a role for nested JSON or non-identifier columns", () => {
    const cols = inferColumns(["text", "fields", "bad name"], [{ text: "t", fields: { a: 1 }, "bad name": "x" }]);
    const m = suggestMapping(cols);
    expect([...m.input, ...m.target, ...m.context]).toEqual(["text"]);
  });

  it("suggests a task type from the target columns", () => {
    expect(suggestTaskType(columns, ["queue"], 3)).toBe("classification");
    const free = inferColumns(
      ["q", "answer"],
      Array.from({ length: 10 }, (_, i) => ({ q: `q${i}`, answer: `a${i}` })),
    );
    expect(suggestTaskType(free, ["answer"], 10)).toBe("question_answering");
    const num = inferColumns(["q", "n"], [{ q: "a", n: 1.5 }]);
    expect(suggestTaskType(num, ["n"], 1)).toBe("question_answering");
    expect(suggestTaskType(columns, ["subject", "body"], 3)).toBe("structured_extraction");
  });

  it("lists labels by frequency", () => {
    expect(labelsIn([{ y: "a" }, { y: "b" }, { y: "a" }, { y: "" }], "y")).toEqual([
      { label: "a", count: 2 },
      { label: "b", count: 1 },
    ]);
  });
});
