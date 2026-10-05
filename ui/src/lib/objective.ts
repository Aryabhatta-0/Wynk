import type { Objective } from "@/api";
import { ms, pct, usd } from "./format";

/** The metric an objective's learning curve plots, and how to print it. */
export function curveMetric(objective: Objective, metricName: string) {
  if (objective === "cost") return { label: "cost per 1k examples", format: usd, lowerIsBetter: true, bounded: false };
  if (objective === "latency") return { label: "p95 latency", format: ms, lowerIsBetter: true, bounded: false };
  return { label: metricName, format: (v: number) => pct(v), lowerIsBetter: false, bounded: true };
}
