import type { MethodResult, Objective, ResultsComparison, SplitId, TaskType } from "@/api/types";
import { CONSTRAINT_LABEL, METHOD_LABEL, OBJECTIVE_LABEL, SPLIT_LABEL, TASK_METRIC, change, int, ms, pct, points, usd } from "./format";

export type FactTone = "better" | "worse" | "same" | "neutral";

export interface Fact {
  id: string;
  text: string;
  tone: FactTone;
}

export interface WhyInput {
  results: ResultsComparison;
  taskType: TaskType;
  objective: Objective;
  validatedCount: number;
}

/*
  "Why this workflow?" — only statements that restate measurements: differences against the
  baselines on one split, constraint checks, and how the workflow was selected. No causal
  claims about *why* a stage helps, because nothing measured says so.
*/
export function whyFacts({ results, taskType, objective, validatedCount }: WhyInput): Fact[] {
  const wynk = results.methods.find((m) => m.method === "wynk_aco");
  if (!wynk) return [];
  const split: SplitId = wynk.splits.test ? "test" : wynk.splits.validation ? "validation" : "optimization";
  const w = wynk.splits[split]!;
  const metric = TASK_METRIC[taskType];
  const facts: Fact[] = [];

  if (validatedCount > 0) {
    facts.push({
      id: "selection",
      text: `Selected from ${validatedCount} validated workflow${validatedCount === 1 ? "" : "s"} by the ${OBJECTIVE_LABEL[objective].toLowerCase()} objective, after ${int(wynk.evaluated)} candidates were evaluated.`,
      tone: "neutral",
    });
  }

  for (const id of ["fixed_baseline", "random_search"] as const) {
    const other = results.methods.find((m) => m.method === id)?.splits[split];
    if (!other) continue;
    facts.push({
      id: `quality-${id}`,
      text: `${SPLIT_LABEL[split]} ${metric}: ${pct(w.quality)} vs ${pct(other.quality)} for ${METHOD_LABEL[id].toLowerCase()} (${points(w.quality, other.quality)}).`,
      tone: tone(w.quality - other.quality, 0.0005),
    });
  }

  const base = results.methods.find((m) => m.method === "fixed_baseline")?.splits[split];
  if (base) {
    facts.push({
      id: "cost",
      text: `Cost: ${usd(w.costPer1k)} per 1,000 examples vs ${usd(base.costPer1k)} for the fixed baseline (${change(w.costPer1k, base.costPer1k)}).`,
      tone: tone(base.costPer1k - w.costPer1k, base.costPer1k * 0.005),
    });
    facts.push({
      id: "latency",
      text: `p95 latency: ${ms(w.latencyP95Ms)} vs ${ms(base.latencyP95Ms)} for the fixed baseline (${change(w.latencyP95Ms, base.latencyP95Ms)}).`,
      tone: tone(base.latencyP95Ms - w.latencyP95Ms, base.latencyP95Ms * 0.005),
    });
  }

  facts.push(constraintFact(wynk, split));

  const opt = wynk.splits.optimization;
  if (opt && split !== "optimization") {
    facts.push({
      id: "generalization",
      text: `${metric[0].toUpperCase()}${metric.slice(1)} went from ${pct(opt.quality)} on the optimization split to ${pct(w.quality)} on the ${SPLIT_LABEL[split].toLowerCase()} split (${points(w.quality, opt.quality)}).`,
      tone: "neutral",
    });
  }
  return facts;
}

function constraintFact(m: MethodResult, split: SplitId): Fact {
  const failed = m.constraints.filter((c) => !c.ok);
  const where = `on the ${SPLIT_LABEL[split].toLowerCase()} split`;
  if (!failed.length) {
    return { id: "constraints", text: `Met all ${m.constraints.length} hard constraints ${where}.`, tone: "better" };
  }
  return {
    id: "constraints",
    text: `Broke ${failed.map((c) => CONSTRAINT_LABEL[c.key].toLowerCase()).join(", ")} ${where}.`,
    tone: "worse",
  };
}

function tone(delta: number, epsilon: number): FactTone {
  if (Math.abs(delta) <= epsilon) return "same";
  return delta > 0 ? "better" : "worse";
}
