/*
  How the backend contracts rank measured candidates, mirrored for the mock adapter so mocked
  runs obey the same semantics. Screens do not call this: they read `feasible`, `rank` and the
  constraint checks the adapter returns.

    check_limits          core/constraints.py   floors / ceilings; a missing measurement fails
    ObjectiveSpec.rank_key core/objective.py    higher is better
    CandidateRank.sort_key core/task_contract.py every feasible key outranks every infeasible one
*/
import type { ConstraintCheck, ConstraintKey, HardConstraints, Measurement, OptimizationPreferences } from "../types";

/** Each limit and the measurement it is compared with (core/constraints.py LIMIT_MEASUREMENTS). */
const LIMITS: Record<ConstraintKey, { field: keyof Measurement; floor: boolean }> = {
  minQuality: { field: "quality", floor: true },
  maxCostPerExample: { field: "costPerExample", floor: false },
  maxMeanLatencyS: { field: "meanLatencyS", floor: false },
  maxP95LatencyS: { field: "p95LatencyS", floor: false },
  maxTokensPerExample: { field: "maxTokensPerExample", floor: false },
  maxWorkflowSteps: { field: "workflowSteps", floor: false },
};

export function checkLimits(limits: HardConstraints, m: Measurement): ConstraintCheck[] {
  const out: ConstraintCheck[] = [];
  for (const key of Object.keys(LIMITS) as ConstraintKey[]) {
    const limit = limits[key];
    if (limit === null) continue;
    const { field, floor } = LIMITS[key];
    const observed = m[field];
    out.push({ key, limit, observed, ok: observed !== null && (floor ? observed >= limit : observed <= limit) });
  }
  return out;
}

/** The objective's sort key, higher is better; null when a metric it reads was not measured. */
export function objectiveKey(m: Measurement, p: OptimizationPreferences): number[] | null {
  const { quality, costPerExample: cost, meanLatencyS: latency, meanTokensPerExample: tokens } = m;
  if (quality === null) return null;
  switch (p.objective) {
    case "quality":
      return [quality];
    case "cost":
      return cost === null ? null : [-cost, quality];
    case "latency":
      return latency === null ? null : [-latency, quality];
    case "balanced": {
      const b = p.balanced;
      if (!b) return null;
      let utility = (b.quality / 100) * quality;
      const terms: [number, number | null, number | null][] = [
        [b.cost, cost, b.costScale],
        [b.latency, latency, b.latencyScale],
        [b.tokens, tokens, b.tokensScale],
      ];
      for (const [weight, value, scale] of terms) {
        if (weight <= 0) continue;
        if (value === null || !scale) return null;
        utility -= ((weight / 100) * value) / scale;
      }
      return [utility];
    }
  }
}

/** One number per candidate for the learning curve: the objective's primary value. */
export function objectiveValue(m: Measurement, p: OptimizationPreferences): number | null {
  if (p.objective === "cost") return m.costPerExample;
  if (p.objective === "latency") return m.meanLatencyS;
  return objectiveKey(m, p)?.[0] ?? null;
}

/**
 * core/task_contract.py rank(): hard limits first; an infeasible candidate's objective is never
 * computed. (The backend raises MissingMetric for a feasible candidate missing an objective
 * metric; the mock always measures them, and here such a candidate would simply rank last.)
 */
export function rankKey(m: Measurement, limits: HardConstraints, p: OptimizationPreferences): number[] {
  if (!checkLimits(limits, m).every((c) => c.ok)) return [0];
  const key = objectiveKey(m, p);
  return key === null ? [0] : [1, ...key];
}

/** Lexicographic comparison of sort keys: positive when `a` ranks above `b`. */
export function compareKeys(a: number[], b: number[]): number {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const d = (a[i] ?? -Infinity) - (b[i] ?? -Infinity);
    if (d !== 0) return d;
  }
  return 0;
}
