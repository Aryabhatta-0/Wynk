import { ArrowDown, ArrowUp } from "@phosphor-icons/react";
import { useMemo, useState } from "react";
import type { Candidate } from "@/api";
import { CONSTRAINT_LABEL, CONSTRAINT_SHORT, constraintValue, costPer1kNumber, NOT_MEASURED, pct, secs } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Check, StateBadge } from "./ui";
import { WorkflowChain } from "./WorkflowChain";

type SortKey = "rank" | "state" | "generation" | "quality" | "validation" | "cost" | "latency" | "p95";

const STATE_ORDER = { champion: 0, validated: 1, candidate: 2 } as const;
const HIGHER_FIRST: SortKey[] = ["quality", "validation"];

interface Props {
  candidates: Candidate[];
  bestId: string | null;
  metric: string;
  /** rows shown before "Show all" */
  initialRows?: number;
}

export function CandidateTable({ candidates, bestId, metric, initialRows = 12 }: Props) {
  // default: the adapter's rank (hard limits first, then the objective)
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "rank", desc: false });
  const [all, setAll] = useState(false);

  const rows = useMemo(() => {
    const value = (c: Candidate): number | null => {
      switch (sort.key) {
        case "rank":
          return c.rank;
        case "state":
          return STATE_ORDER[c.state];
        case "generation":
          return c.generation;
        case "quality":
          return c.optimization.quality;
        case "validation":
          return c.validation?.quality ?? null;
        case "cost":
          return c.optimization.costPerExample;
        case "latency":
          return c.optimization.meanLatencyS;
        case "p95":
          return c.optimization.p95LatencyS;
      }
    };
    const cmp = (a: Candidate, b: Candidate) => {
      const va = value(a);
      const vb = value(b);
      if (va === null || vb === null) return va === vb ? 0 : va === null ? 1 : -1; // unmeasured last
      return sort.desc ? vb - va : va - vb;
    };
    // infeasible candidates sink below feasible ones whatever the column
    return [...candidates].sort((a, b) => Number(b.feasible) - Number(a.feasible) || cmp(a, b));
  }, [candidates, sort]);

  const shown = all ? rows : rows.slice(0, initialRows);
  const head = (key: SortKey, label: string, num = true) => {
    const active = sort.key === key;
    return (
      <th className={cn(num && "num")} aria-sort={active ? (sort.desc ? "descending" : "ascending") : "none"}>
        <button
          type="button"
          className={cn("inline-flex items-center gap-1 uppercase hover:text-ink", active && "text-ink")}
          onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : HIGHER_FIRST.includes(key) }))}
        >
          {label}
          {active && (sort.desc ? <ArrowDown size={10} weight="bold" aria-hidden="true" /> : <ArrowUp size={10} weight="bold" aria-hidden="true" />)}
        </button>
      </th>
    );
  };

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="data-table min-w-[960px]" aria-label="Candidate workflows">
          <thead>
            <tr>
              {head("rank", "#")}
              {head("state", "State", false)}
              <th>Workflow</th>
              {head("generation", "Gen")}
              {head("quality", `Opt. ${metric}`)}
              {head("validation", `Val. ${metric}`)}
              {head("cost", "Cost / 1k")}
              {head("latency", "Mean")}
              {head("p95", "p95")}
              <th>Constraints</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => {
              const broken = c.constraints.filter((k) => !k.ok);
              return (
                <tr
                  key={c.id}
                  className={cn(c.state === "champion" && "bg-magenta-wash/50", !c.feasible && "text-ink-soft")}
                  data-testid="candidate-row"
                >
                  <td className="num text-ink-soft">{c.rank}</td>
                  <td>
                    <div className="flex flex-col items-start gap-1">
                      <StateBadge state={c.state} />
                      {c.id === bestId && c.state !== "champion" && <span className="text-[11px] text-magenta-ink">current best</span>}
                    </div>
                  </td>
                  <td>
                    <WorkflowChain stages={c.stages} model={c.model} />
                    <div className="mt-1 font-mono text-[10px] text-ink-soft">{c.genomeHash}</div>
                  </td>
                  <td className="num">{c.generation}</td>
                  <td className="num">{pct(c.optimization.quality)}</td>
                  <td className="num">{c.validation ? pct(c.validation.quality) : <span className="text-ink-soft">{NOT_MEASURED}</span>}</td>
                  <td className="num">{c.optimization.costPerExample === null ? NOT_MEASURED : costPer1kNumber(c.optimization.costPerExample)}</td>
                  <td className="num">{secs(c.optimization.meanLatencyS)}</td>
                  <td className="num">{secs(c.optimization.p95LatencyS)}</td>
                  <td>
                    {broken.length ? (
                      <Check ok={false}>
                        <span
                          title={broken
                            .map(
                              (k) => `${CONSTRAINT_LABEL[k.key]}: ${constraintValue(k.key, k.observed)} vs limit ${constraintValue(k.key, k.limit)}`,
                            )
                            .join("\n")}
                        >
                          {broken.map((k) => CONSTRAINT_SHORT[k.key]).join(", ")}
                        </span>
                      </Check>
                    ) : (
                      <Check ok>All met</Check>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > initialRows && (
        <div className="border-t border-line px-3 pt-3">
          <button type="button" className="btn btn-text text-sm" onClick={() => setAll((v) => !v)}>
            {all ? "Show fewer" : `Show all ${rows.length} candidates`}
          </button>
        </div>
      )}
    </div>
  );
}
