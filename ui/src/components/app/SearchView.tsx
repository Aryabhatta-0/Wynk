import { ArrowRight, CaretDown, Minus, TrendDown, TrendUp } from "@phosphor-icons/react";
import { useId, useState } from "react";
import type { EdgeSignal, GenerationSummary } from "@/api";
import { STAGE_LABEL, optionLabel, pct } from "@/lib/format";
import type { StageKind } from "@/api";
import { cn } from "@/lib/utils";

interface Props {
  generations: GenerationSummary[];
  edges: EdgeSignal[];
  maxGenerations: number;
  defaultOpen?: boolean;
}

/*
  What the colony is doing, summarised: per-generation quality and agreement, and the
  transitions the search keeps reinforcing. Edge strengths are relative to the strongest edge;
  raw pheromone values are deliberately not shown.
*/
export function SearchView({ generations, edges, maxGenerations, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const first = generations[0];
  const last = generations.at(-1);

  return (
    <section className="panel min-w-0">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left"
      >
        <span className="flex min-w-0 flex-wrap items-baseline gap-x-2">
          <span className="text-sm font-semibold">Search (ACO)</span>
          <span className="text-xs text-ink-soft">
            {last
              ? `Generation ${last.generation} of ${maxGenerations} · ${pct(last.agreement, 0)} decision agreement`
              : "Waiting for the first generation"}
          </span>
        </span>
        <CaretDown size={14} className={cn("shrink-0 transition-transform duration-200", open && "rotate-180")} aria-hidden="true" />
      </button>

      {open && (
        <div id={id} className="grid gap-5 border-t border-line p-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <div className="min-w-0">
            <h3 className="kicker">Generations</h3>
            {first && last && generations.length > 1 && (
              <p className="mt-1 text-xs leading-relaxed text-ink-soft">
                Decision agreement went from {pct(first.agreement, 0)} in generation 1 to {pct(last.agreement, 0)} in generation {last.generation};
                best quality in a generation from {pct(first.bestQuality)} to {pct(last.bestQuality)}.
              </p>
            )}
            {generations.length ? (
              <div className="mt-2 max-h-72 overflow-y-auto">
                <table className="data-table" aria-label="Generations">
                  <thead>
                    <tr>
                      <th className="num">Gen</th>
                      <th className="num">New</th>
                      <th className="num">Best</th>
                      <th className="num">Mean</th>
                      <th>Agreement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {generations.map((g) => (
                      <tr key={g.generation}>
                        <td className="num">{g.generation}</td>
                        <td className="num">{g.novel}</td>
                        <td className="num">{pct(g.bestQuality)}</td>
                        <td className="num">{pct(g.meanQuality)}</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-line" aria-hidden="true">
                              <div className="h-full rounded-full bg-magenta-ink" style={{ width: `${g.agreement * 100}%` }} />
                            </div>
                            <span className="text-xs tnum">{pct(g.agreement, 0)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="mt-2 text-sm text-ink-soft">No generation has finished yet.</p>
            )}
            <p className="mt-2 text-[11px] leading-snug text-ink-soft">
              New: candidates never proposed before. Agreement: for each stage decision and the model, the share of proposals that made the most
              common choice, averaged. It rises as the colony converges.
            </p>
          </div>

          <div className="min-w-0">
            <h3 className="kicker">Most reinforced transitions</h3>
            {edges.length ? (
              <ol className="mt-2 space-y-2" aria-label="Most reinforced transitions">
                {edges.map((e) => (
                  <li key={`${e.from}-${e.to}`} className="rounded-lg border border-line px-3 py-2">
                    <div className="flex flex-wrap items-center gap-1.5 text-xs">
                      <Node label={e.from} />
                      <ArrowRight size={11} className="text-ink-soft" aria-label="then" />
                      <Node label={e.to} />
                      <Trend trend={e.trend} />
                    </div>
                    <div className="mt-1.5 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line" aria-hidden="true">
                        <div className="h-full rounded-full bg-magenta-ink" style={{ width: `${e.strength * 100}%` }} />
                      </div>
                      <span className="w-10 text-right text-xs tnum">{pct(e.strength, 0)}</span>
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="mt-2 text-sm text-ink-soft">Transitions appear after the first generation.</p>
            )}
            <p className="mt-2 text-[11px] leading-snug text-ink-soft">
              Strength is relative to the strongest transition. It shows where the search concentrates, not why a step helps.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

function Node({ label }: { label: string }) {
  const [kind, option] = label.split(" · ");
  return (
    <span className="rounded-md border border-line bg-white px-1.5 py-0.5">
      <span className="font-semibold">{STAGE_LABEL[kind as StageKind] ?? kind}</span>
      {option && <span className="ml-1 font-mono text-[11px] text-ink-soft">{optionLabel(option)}</span>}
    </span>
  );
}

function Trend({ trend }: { trend: EdgeSignal["trend"] }) {
  const t = {
    rising: { icon: TrendUp, label: "rising", cls: "text-magenta-ink" },
    steady: { icon: Minus, label: "steady", cls: "text-ink-soft" },
    falling: { icon: TrendDown, label: "falling", cls: "text-ink-soft" },
  }[trend];
  return (
    <span className={cn("ml-auto inline-flex items-center gap-1 text-[11px]", t.cls)}>
      <t.icon size={12} aria-hidden="true" />
      {t.label}
    </span>
  );
}
