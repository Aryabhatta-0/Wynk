import { describe, expect, it } from "vitest";
import { splitSizes } from "../contract/rules";
import { checkLimits, compareKeys, rankKey } from "../contract/semantics";
import { type SimInput, simulate, snapshot } from "./simulate";

const input = (over: Partial<SimInput> = {}): SimInput => ({
  seed: "e-test",
  taskType: "classification",
  constraints: {
    minQuality: 0.8,
    maxCostPerExample: 0.0006,
    maxMeanLatencyS: null,
    maxP95LatencyS: 9,
    maxTokensPerExample: null,
    maxWorkflowSteps: null,
  },
  preferences: { objective: "quality", balanced: null },
  models: ["gemma-3-27b-it", "gemma-4-31b-it"],
  budget: { maxCandidates: 64, maxGenerations: 8, maxSpendUsd: 50, maxDurationMin: 120 },
  splits: { validationPct: 20, testPct: 20, seed: 0 },
  rowCount: 1000,
  ...over,
});

describe("simulate (mock search under the contract semantics)", () => {
  it("is deterministic for a seed", () => {
    expect(simulate(input())).toEqual(simulate(input()));
    expect(simulate(input({ seed: "other" })).evaluated.map((e) => e.opt)).not.toEqual(simulate(input()).evaluated.map((e) => e.opt));
  });

  it("sizes splits with the SplitPlan basis-point rule: optimization gets the rest", () => {
    expect(splitSizes(1001, { validationPct: 20, testPct: 15, seed: 0 })).toEqual({ optimization: 651, validation: 200, test: 150 });
    expect(simulate(input({ rowCount: 1001, splits: { validationPct: 20, testPct: 15, seed: 0 } })).splitSizes.optimization).toBe(651);
  });

  it("produces measurements in contract units", () => {
    const m = simulate(input()).evaluated[0].opt;
    expect(m.costPerExample).toBeGreaterThan(0);
    expect(m.costPerExample).toBeLessThan(0.01); // USD per example, not per 1k
    expect(m.meanLatencyS).toBeLessThan(m.p95LatencyS!);
    expect(m.maxTokensPerExample).toBeGreaterThanOrEqual(m.meanTokensPerExample!);
    expect(m.workflowSteps).toBeGreaterThanOrEqual(2);
  });

  it("stays inside every budget limit", () => {
    for (const budget of [
      { maxCandidates: 20, maxGenerations: 50, maxSpendUsd: 50, maxDurationMin: 120 },
      { maxCandidates: 200, maxGenerations: 50, maxSpendUsd: 0.1, maxDurationMin: 120 },
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
    expect(simulate(input({ budget: { maxCandidates: 200, maxGenerations: 50, maxSpendUsd: 0.1, maxDurationMin: 120 } })).stopReason).toBe("spend");
  });

  it("ranks like TaskContract.rank: every feasible candidate above every infeasible one", () => {
    const run = simulate(input());
    const { candidates } = snapshot(run, 1, "e");
    const byRank = [...candidates].sort((a, b) => a.rank - b.rank);
    const feasible = byRank.map((c) => checkLimits(input().constraints, c.optimization).every((k) => k.ok));
    // once an infeasible candidate appears in rank order, no feasible one follows
    expect(feasible.slice(feasible.indexOf(false) < 0 ? feasible.length : feasible.indexOf(false))).not.toContain(true);
    byRank.slice(1).forEach((c, i) => {
      const prev = byRank[i];
      expect(
        compareKeys(
          rankKey(prev.optimization, input().constraints, input().preferences),
          rankKey(c.optimization, input().constraints, input().preferences),
        ),
      ).toBeGreaterThanOrEqual(0);
    });
  });

  it("only crowns a validated candidate that meets every limit on validation", () => {
    const run = simulate(input());
    const final = snapshot(run, 1, "e");
    const champ = final.candidates.find((c) => c.id === final.championId)!;
    expect(champ.state).toBe("champion");
    expect(champ.validation).not.toBeNull();
    expect(checkLimits(input().constraints, champ.validation!).every((c) => c.ok)).toBe(true);
    expect(final.candidates.filter((c) => c.state === "champion")).toHaveLength(1);
  });

  it("finishes without a champion when no candidate can be feasible", () => {
    const run = simulate(input({ constraints: { ...input().constraints, minQuality: 0.999 } }));
    expect(run.championKey).toBeNull();
    expect(snapshot(run, 1, "e").championId).toBeNull();
  });

  it("follows the minimize_cost key: lowest mean cost first, then quality", () => {
    const prefs = { objective: "cost" as const, balanced: null };
    const { candidates } = snapshot(simulate(input({ preferences: prefs })), 1, "e");
    const feasible = candidates.filter((c) => checkLimits(input().constraints, c.optimization).every((k) => k.ok)).sort((a, b) => a.rank - b.rank);
    expect(feasible.length).toBeGreaterThan(1);
    feasible.slice(1).forEach((c, i) => expect(c.optimization.costPerExample!).toBeGreaterThanOrEqual(feasible[i].optimization.costPerExample!));
  });

  it("never measures or exposes the held-out test split in the search or candidate state", () => {
    const run = simulate(input());
    expect(Object.keys(run.baseline).sort()).toEqual(["optimization", "validation"]);
    for (const f of [0.3, 0.9, 0.96, 1]) {
      for (const c of snapshot(run, f, "e").candidates) {
        expect(Object.keys(c)).not.toContain("test");
        expect(c.validation === null || c.validation.n === run.splitSizes.validation).toBe(true);
        expect(c.optimization.n).toBe(run.splitSizes.optimization);
      }
    }
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
    const c = snapshot(simulate(input({ preferences: { objective: "cost", balanced: null } })), 1, "e")
      .curve.map((p) => p.wynk)
      .filter((v): v is number => v !== null);
    c.slice(1).forEach((v, i) => expect(v).toBeLessThanOrEqual(c[i]));
  });

  it("plots balanced utility for a balanced objective", () => {
    const balanced = { quality: 70, cost: 20, latency: 10, tokens: 0, costScale: 0.0005, latencyScale: 2, tokensScale: null };
    const { curve } = snapshot(simulate(input({ preferences: { objective: "balanced", balanced } })), 1, "e");
    const values = curve.map((p) => p.wynk).filter((v): v is number => v !== null);
    expect(values.length).toBeGreaterThan(0);
    values.forEach((v) => expect(v).toBeLessThan(0.7)); // at most 0.7 × quality, minus penalties
  });

  it("reports relative edge strengths, never raw pheromone", () => {
    const edges = snapshot(simulate(input()), 1, "e").edges;
    expect(edges.length).toBeGreaterThan(0);
    expect(edges[0].strength).toBe(1);
    edges.forEach((e) => expect(e.strength).toBeGreaterThan(0));
    edges.forEach((e) => expect(e.strength).toBeLessThanOrEqual(1));
  });
});
