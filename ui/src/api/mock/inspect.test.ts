import { describe, expect, it } from "vitest";
import { inspect, parseCsvRecords } from "./inspect";

const bytes = (s: string) => new TextEncoder().encode(s);
const codeOf = (fn: () => unknown) => {
  try {
    fn();
  } catch (err) {
    return (err as { code?: string }).code;
  }
  return null;
};

describe("mock inspection (stands in for ingestion/parse.py in mock mode only)", () => {
  it("handles quotes, escaped quotes, commas and newlines inside fields and CRLF", () => {
    expect(parseCsvRecords('a,b\r\n"x, y","say ""hi"""\r\n"line1\nline2",2\r\n')).toEqual([
      ["a", "b"],
      ["x, y", 'say "hi"'],
      ["line1\nline2", "2"],
    ]);
  });

  it("types columns over every row; an empty cell is null", () => {
    const rows = ["n,int,day,flag", ...Array.from({ length: 40 }, (_, i) => `${i},${i},2026-01-0${(i % 9) + 1},true`), "x,4.5,soon,"];
    const { columns } = inspect(bytes(rows.join("\n")), "csv");
    expect(columns).toEqual([
      { name: "n", type: "string", nullable: false, nullCount: 0 },
      { name: "int", type: "number", nullable: false, nullCount: 0 },
      { name: "day", type: "string", nullable: false, nullCount: 0 },
      { name: "flag", type: "boolean", nullable: true, nullCount: 1 },
    ]);
  });

  it("types JSONL values, with json and string_list columns", () => {
    const { columns, preview } = inspect(
      bytes('{"id":1,"day":"2026-03-01","tags":["a","b"],"meta":{"k":1}}\n{"id":2,"day":"2026-03-02","tags":[],"meta":[1]}\n'),
      "jsonl",
    );
    expect(columns.map((c) => [c.name, c.type])).toEqual([
      ["id", "integer"],
      ["day", "date"],
      ["tags", "string_list"],
      ["meta", "json"],
    ]);
    expect(preview).toHaveLength(2);
  });

  it("rejects what the server rejects, with the same codes", () => {
    expect(codeOf(() => inspect(bytes(""), "csv"))).toBe("empty_file");
    expect(codeOf(() => inspect(new Uint8Array([0xff, 0xfe]), "csv"))).toBe("invalid_encoding");
    expect(codeOf(() => inspect(bytes("a,b\n1\n"), "csv"))).toBe("malformed_csv");
    expect(codeOf(() => inspect(bytes("a\n\n1\n"), "csv"))).toBe("malformed_csv");
    expect(codeOf(() => inspect(bytes('a\n"open'), "csv"))).toBe("malformed_csv");
    expect(codeOf(() => inspect(bytes("a,a\n1,2\n"), "csv"))).toBe("duplicate_column");
    expect(codeOf(() => inspect(bytes("a b\n1\n"), "csv"))).toBe("invalid_column_name");
    expect(codeOf(() => inspect(bytes("a,b\n"), "csv"))).toBe("no_rows");
    expect(codeOf(() => inspect(bytes('{"a":1}\nnot json\n'), "jsonl"))).toBe("malformed_jsonl");
    expect(codeOf(() => inspect(bytes("[1,2]\n"), "jsonl"))).toBe("malformed_jsonl");
  });
});
