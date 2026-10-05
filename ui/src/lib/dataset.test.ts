import { describe, expect, it } from "vitest";
import type { DatasetColumn } from "@/api";
import { labelsIn, slugFor, suggestMapping, suggestTaskType, uploadFormat } from "./dataset";

const col = (name: string, type: DatasetColumn["type"] = "string", nullable = false): DatasetColumn => ({ name, type, nullable });

describe("uploadFormat", () => {
  it("names the format from the extension and leaves acceptance to the server", () => {
    expect(uploadFormat("a.CSV")).toBe("csv");
    expect(uploadFormat("a.jsonl")).toBe("jsonl");
    expect(uploadFormat("a.ndjson")).toBe("jsonl");
    // sent as-is: the server answers unsupported_format, and the screen shows that
    expect(uploadFormat("a.parquet")).toBe("parquet");
    expect(uploadFormat("README")).toBe("");
  });
});

describe("slugFor", () => {
  it("suggests a valid dataset id from a display name", () => {
    expect(slugFor("Support tickets, Q3")).toBe("support-tickets-q3");
    expect(slugFor("  ** ")).toBe("dataset");
    expect(slugFor("Ünïcode name")).toMatch(/^[a-z0-9][a-z0-9_.-]*$/);
  });
});

describe("suggestions", () => {
  const columns = [col("ticket_id"), col("subject"), col("body"), col("customer_tier", "string", true), col("queue")];
  const preview = [
    { ticket_id: "1", subject: "s", body: "b", customer_tier: "pro", queue: "billing" },
    { ticket_id: "2", subject: "s2", body: "b2", customer_tier: "free", queue: "billing" },
    { ticket_id: "3", subject: "s3", body: "b3", customer_tier: null, queue: "technical" },
  ];

  it("suggests target, inputs, context and the id column from names and types", () => {
    expect(suggestMapping(columns)).toEqual({ input: ["subject", "body"], target: ["queue"], context: ["customer_tier"], id: "ticket_id" });
  });

  it("never suggests a role for nested JSON or non-identifier columns, nor a nullable id", () => {
    const m = suggestMapping([col("text"), col("fields", "json"), col("bad name"), col("row_id", "string", true)]);
    expect([...m.input, ...m.target, ...m.context]).toEqual(["text"]);
    expect(m.id).toBeNull();
  });

  it("suggests a task type from the target column and the preview rows", () => {
    expect(suggestTaskType(columns, ["queue"], preview)).toBe("classification");
    const free = Array.from({ length: 10 }, (_, i) => ({ q: `q${i}`, answer: `a${i}` }));
    expect(suggestTaskType([col("q"), col("answer")], ["answer"], free)).toBe("question_answering");
    expect(suggestTaskType([col("q"), col("n", "number")], ["n"], [{ q: "a", n: 1.5 }])).toBe("question_answering");
    expect(suggestTaskType(columns, ["subject", "body"], preview)).toBe("structured_extraction");
  });

  it("lists labels by frequency", () => {
    expect(labelsIn([{ y: "a" }, { y: "b" }, { y: "a" }, { y: "" }], "y")).toEqual([
      { label: "a", count: 2 },
      { label: "b", count: 1 },
    ]);
  });
});
