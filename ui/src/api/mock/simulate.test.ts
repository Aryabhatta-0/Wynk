import { describe, expect, it } from "vitest";
import { type SimInput, checkConstraints, simulate, snapshot } from "./simulate";

const input = (over: Partial<SimInput> = {}): SimInput => ({
  seed: "e-test",
  taskType: "classification",
  constraints: { minQuality: 0.8, maxCostPer1k: 0.6, maxLatencyP95Ms: 9000, allowedModels: ["gemma-3-27b-it", "gemma-4-31b-it"] },
  objective: "quality",
  budget: { maxCandidates: 64, maxGenerations: 8, maxSpendUsd: 50, maxDurationMin: 120 },
  splits: { optimization: 60, validation: 20, test: 20 },
  rowCount: 1000,
  ...over,
});

describe("simulate (mock search)", () => {
  it("is deterministic for a seed", () => {
    expect(simulate(input())).toEqual(simulate(input()));
    expect(simulate(input({ seed: "other" })).evaluated.map((e) => e.opt)).not.toEqual(simulate(input()).evaluated.map((e) => e.opt));
  });

  it("stays inside every budget limit", () => {
    for (const budget of [
      { maxCandidates: 20, maxGenerations: 50, maxSpendUsd: 50, maxDurationMin: 120 },
      { maxCandidates: 200, maxGenerations: 50, maxSpendUsd: 0.4, maxDurationMin: 120 },
      { maxCandidates: 200, maxGenerations: 3, maxSpendUsd: 50, maxDurationMin: 120 },
    ]) {
      const run = simulate(input({ budget }));
      expect(run.evaluated.length).toBeLessThanOrEqual(budget.maxCandidates);
      expect(run.generations.length).toBeLessThanOrEqual(budget.maxGenerations);
      expect(run.spend.at(-1) ?? 0).toBeLessThanOrEqual(budget.maxSpendUsd);
    }
    expect(simulate(input({ budget: { maxCandidates: 20, maxGenerations: 50, maxSpendUsd: 50, maxDurationMin: 120 } })).stopReason).toBe(
      "candidates",
    );
    expect(simulate(input({ budget: { maxCandidates: 200, maxGenerations: 50, maxSpendUsd: 0.4, maxDurationMin: 120 } })).stopReason).toBe("spend");
  });

  it("only crowns a validated candidate that meets every constraint and uses an allowed model", () => {
    const run = simulate(input());
    const final = snapshot(run, 1, "e");
    const champ = final.candidates.find((c) => c.id === final.championId)!;
    expect(champ.state).toBe("champion");
    expect(champ.validation).not.toBeNull();
    expect(checkConstraints(champ.validation!, champ.model, input().constraints).every((c) => c.ok)).toBe(true);
    expect(final.candidates.filter((c) => c.state === "champion")).toHaveLength(1);
  });

  it("finishes without a champion when no candidate can be feasible", () => {
    const run = simulate(input({ constraints: { ...input().constraints, minQuality: 0.999 } }));
    expect(run.championKey).toBeNull();
    expect(snapshot(run, 1, "e").championId).toBeNull();
  });

  it("reveals the run in order: search, then validation, then the champion", () => {
    const run = simulate(input());
    const early = snapshot(run, 0.3, "e");
    const validating = snapshot(run, 0.9, "e");
    const testing = snapshot(run, 0.96, "e");
    const done = snapshot(run, 1, "e");
    expect([early.phase, validating.phase, testing.phase, done.phase]).toEqual(["searching", "validating", "testing", "done"]);
    expect(early.evaluatedCount).toBeLessThan(done.evaluatedCount);
    expect(validating.candidates.some((c) => c.state !== "candidate")).toBe(false);
    expect(testing.candidates.some((c) => c.state === "validated")).toBe(true);
    expect(testing.championId).toBeNull();
    expect(done.championId).not.toBeNull();
  });

  it("plots a best-so-far curve that never gets worse", () => {
    const q = snapshot(simulate(input()), 1, "e")
      .curve.map((p) => p.wynk)
      .filter((v): v is number => v !== null);
    q.slice(1).forEach((v, i) => expect(v).toBeGreaterThanOrEqual(q[i]));
    const c = snapshot(simulate(input({ objective: "cost" })), 1, "e")
      .curve.map((p) => p.wynk)
      .filter((v): v is number => v !== null);
    c.slice(1).forEach((v, i) => expect(v).toBeLessThanOrEqual(c[i]));
  });

  it("reports relative edge strengths, never raw pheromone", () => {
    const edges = snapshot(simulate(input()), 1, "e").edges;
    expect(edges.length).toBeGreaterThan(0);
    expect(edges[0].strength).toBe(1);
    edges.forEach((e) => expect(e.strength).toBeGreaterThan(0));
    edges.forEach((e) => expect(e.strength).toBeLessThanOrEqual(1));
  });
});
