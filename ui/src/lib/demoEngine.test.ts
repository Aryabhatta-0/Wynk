import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RunEvent } from "./chat";
import { jevGather, runQuery } from "./demoEngine";

const PAGES = {
  overview: "Overview\nThe company is led by chief executive officer Ada.\nIt was founded in 1990.",
  financials: "Financials\nTotal employees: 120.",
  news: "News\nNothing relevant.",
};

describe("jevGather (demo simulation of a bounded JEV gather)", () => {
  it("opens only what the fields need, then stops; each decision and page is a tool call", () => {
    // decision 1 opens overview (sequential); decision 2: done
    expect(jevGather(PAGES, [/chief executive/i], 1)).toEqual({ opened: ["overview"], toolCalls: 3 });
    // parallel-2: decision 1 opens financials + overview; decision 2: done
    expect(jevGather(PAGES, [/chief executive/i, /employees/i], 2)).toEqual({ opened: ["financials", "overview"], toolCalls: 4 });
  });

  it("is bounded by the pages there are, with no decision once every page is open", () => {
    const r = jevGather(PAGES, [/not on any page/i], 2);
    expect(r.opened).toEqual(["financials", "news", "overview"]);
    expect(r.toolCalls).toBe(2 + 3); // two decisions, three pages, no third decision
  });
});

describe("demo run", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("no longer fails the jev ant: its GATHER succeeds and reports tool calls", async () => {
    const events: RunEvent[] = [];
    const run = runQuery("Who is the CEO of Quillon Labs?", (e) => events.push(e), new AbortController().signal);
    await vi.runAllTimersAsync();
    await run;

    const ants = events.find((e) => e.type === "ants");
    if (ants?.type !== "ants") throw new Error("no ants event");
    const jevAnt = ants.ants.find((a) => a.stages.some((s) => s.kind === "GATHER" && s.options.includes("jev")));
    expect(jevAnt).toBeDefined();

    const stages = events.filter((e): e is Extract<RunEvent, { type: "stage" }> => e.type === "stage");
    expect(stages.filter((e) => e.status === "failed")).toEqual([]);
    expect(events.some((e) => "message" in e && /not available/.test(String(e.message)))).toBe(false);
    const gather = stages.find((e) => e.ant === jevAnt!.id && e.index === 0 && e.status === "ok");
    expect(gather?.tool_calls).toBeGreaterThan(0);

    const score = events.find((e) => e.type === "score" && e.ant === jevAnt!.id);
    expect(score).toMatchObject({ completed: true });
    expect(events.at(-1)?.type).toBe("done");
  });
});
