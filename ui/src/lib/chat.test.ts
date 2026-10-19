import { describe, expect, it } from "vitest";
import { applyEvent, evidenceSources, gatherSourceLabel, isLiveSource, newTurn, type Stage } from "./chat";

const plan = (source: string): Stage[] => [
  { kind: "GATHER", options: [source, "sequential"] },
  { kind: "EXTRACT", options: ["direct"] },
];

describe("gather sources (#12: live web search shown distinctly)", () => {
  it("names every source in plain words", () => {
    expect(gatherSourceLabel("fetch")).toBe("frozen pages");
    expect(gatherSourceLabel("api")).toBe("mock API");
    expect(gatherSourceLabel("jev")).toBe("JEV browsing");
    expect(gatherSourceLabel("websearch")).toBe("live web search");
    expect(gatherSourceLabel("other")).toBe("other");
  });

  it("only websearch is live", () => {
    expect(["fetch", "api", "jev", "websearch"].filter(isLiveSource)).toEqual(["websearch"]);
    expect(isLiveSource(undefined)).toBe(false);
  });

  it("groups ants by the source they read", () => {
    const ants = [
      { id: 1, stages: plan("fetch") },
      { id: 2, stages: plan("websearch") },
      { id: 3, stages: plan("fetch") },
    ];
    expect(evidenceSources(ants)).toEqual([
      { source: "fetch", label: "frozen pages", ants: [1, 3] },
      { source: "websearch", label: "live web search", ants: [2] },
    ]);
  });

  it("records the source a streamed stage event reports", () => {
    let turn = newTurn(1, "q");
    turn = applyEvent(turn, { type: "ants", ants: [{ id: 1, role: "explorer", trail: 0.5, stages: plan("jev") }] });
    expect(turn.ants[0].sources).toEqual(["jev", undefined]);
    turn = applyEvent(turn, { type: "stage", ant: 1, index: 0, status: "ok", tool_calls: 1, source: "websearch" });
    expect(turn.ants[0].sources[0]).toBe("websearch");
    expect(turn.ants[0].toolCalls[0]).toBe(1);
    expect(evidenceSources(turn.ants).map((r) => r.label)).toEqual(["live web search"]);
  });
});
