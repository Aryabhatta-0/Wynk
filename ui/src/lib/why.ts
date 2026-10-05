import type { ConstraintCheck, EvaluationConfig, Objective, ResultsComparison, SplitId } from "@/api/types";
import { CONSTRAINT_LABEL, METHOD_LABEL, OBJECTIVE_LABEL, SPLIT_LABEL, change, costPer1k, int, metricName, pct, points, secs } from "./format";

export type FactTone = "better" | "worse" | "same" | "neutral";

export interface Fact {
  id: string;
  text: string;
  tone: FactTone;
}

export interface WhyInput {
  results: ResultsComparison;
  evaluation: EvaluationConfig;
  objective: Objective;
  validatedCount: number;
}

/*
  "Why this workflow?" — only statements that restate measurements: differences against the
  baselines on one split, constraint checks, and how the workflow was selected. No causal
  claims about *why* a stage helps, because nothing measured says so. A metric that was not
  measured produces no statement rather than a guessed one.

  The champion was selected on validation. Held-out test numbers, when present, are reported
  only; they confirm or contradict the selection but never changed it.
*/
export function whyFacts({ results, evaluation, objective, validatedCount }: WhyInput): Fact[] {
  const wynk = results.methods.find((m) => m.method === "wynk_aco");
  if (!wynk) return [];
  const split: SplitId = wynk.splits.test ? "test" : wynk.splits.validation ? "validation" : "optimization";
  const w = wynk.splits[split]!.measurement;
  const metric = metricName(evaluation);
  const facts: Fact[] = [];

  if (validatedCount > 0) {
    facts.push({
      id: "selection",
      text: `Selected on the validation split from ${validatedCount} validated workflow${validatedCount === 1 ? "" : "s"} by the ${OBJECTIVE_LABEL[objective].toLowerCase()} objective, after ${int(wynk.evaluated)} candidates were evaluated.`,
      tone: "neutral",
    });
  }

  for (const id of ["fixed_baseline", "random_search"] as const) {
    const other = results.methods.find((m) => m.method === id)?.splits[split]?.measurement;
    if (!other || w.quality === null || other.quality === null) continue;
    facts.push({
      id: `quality-${id}`,
      text: `${SPLIT_LABEL[split]} ${metric}: ${pct(w.quality)} vs ${pct(other.quality)} for ${METHOD_LABEL[id].toLowerCase()} (${points(w.quality, other.quality)}).`,
      tone: tone(w.quality - other.quality, 0.0005),
    });
  }

  const base = results.methods.find((m) => m.method === "fixed_baseline")?.splits[split]?.measurement;
  if (base && w.costPerExample !== null && base.costPerExample !== null) {
    facts.push({
      id: "cost",
      text: `Cost: ${costPer1k(w.costPerExample)} examples vs ${costPer1k(base.costPerExample)} for the fixed baseline (${change(w.costPerExample, base.costPerExample)}).`,
      tone: tone(base.costPerExample - w.costPerExample, base.costPerExample * 0.005),
    });
  }
  if (base && w.meanLatencyS !== null && base.meanLatencyS !== null) {
    facts.push({
      id: "latency",
      text: `Mean latency: ${secs(w.meanLatencyS)} vs ${secs(base.meanLatencyS)} for the fixed baseline (${change(w.meanLatencyS, base.meanLatencyS)}).`,
      tone: tone(base.meanLatencyS - w.meanLatencyS, base.meanLatencyS * 0.005),
    });
  }

  facts.push(constraintFact(wynk.splits[split]!.constraints, split));

  const opt = wynk.splits.optimization?.measurement;
  if (opt && split !== "optimization" && opt.quality !== null && w.quality !== null) {
    facts.push({
      id: "generalization",
      text: `${metric[0].toUpperCase()}${metric.slice(1)} went from ${pct(opt.quality)} on the optimization split to ${pct(w.quality)} on the ${SPLIT_LABEL[split].toLowerCase()} split (${points(w.quality, opt.quality)}).`,
      tone: "neutral",
    });
  }
  return facts;
}

function constraintFact(checks: ConstraintCheck[], split: SplitId): Fact {
  const where = `on the ${SPLIT_LABEL[split].toLowerCase()} split`;
  if (!checks.length) return { id: "constraints", text: `No hard constraints were set.`, tone: "neutral" };
  const failed = checks.filter((c) => !c.ok);
  if (!failed.length) return { id: "constraints", text: `Met all ${checks.length} hard constraints ${where}.`, tone: "better" };
  const names = failed.map((c) => `${CONSTRAINT_LABEL[c.key].toLowerCase()}${c.observed === null ? " (not measured)" : ""}`);
  return { id: "constraints", text: `Broke ${names.join(", ")} ${where}.`, tone: "worse" };
}

function tone(delta: number, epsilon: number): FactTone {
  if (Math.abs(delta) <= epsilon) return "same";
  return delta > 0 ? "better" : "worse";
}
