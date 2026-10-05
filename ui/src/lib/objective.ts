import type { BalancedWeights, Objective } from "@/api";
import { costPer1k, pct, secs } from "./format";

/** The value an objective's learning curve plots, and how to print it. */
export function curveMetric(objective: Objective, metricName: string) {
  if (objective === "cost") return { label: "cost per 1k examples", format: (v: number) => costPer1k(v), lowerIsBetter: true, bounded: false };
  if (objective === "latency") return { label: "mean latency", format: (v: number) => secs(v), lowerIsBetter: true, bounded: false };
  if (objective === "balanced") return { label: "balanced utility", format: (v: number) => v.toFixed(3), lowerIsBetter: false, bounded: false };
  return { label: metricName, format: (v: number) => pct(v), lowerIsBetter: false, bounded: true };
}

/** "0.7 × quality − 0.2 × cost ÷ $0.40/1k − 0.1 × latency ÷ 2 s" */
export function balancedFormula(b: BalancedWeights): string {
  const terms = [`${(b.quality / 100).toFixed(2)} × quality`];
  if (b.cost > 0 && b.costScale) terms.push(`${(b.cost / 100).toFixed(2)} × cost ÷ ${costPer1k(b.costScale)}`);
  if (b.latency > 0 && b.latencyScale) terms.push(`${(b.latency / 100).toFixed(2)} × mean latency ÷ ${secs(b.latencyScale)}`);
  if (b.tokens > 0 && b.tokensScale) terms.push(`${(b.tokens / 100).toFixed(2)} × tokens ÷ ${b.tokensScale.toLocaleString("en-US")}`);
  return terms.join(" − ");
}
