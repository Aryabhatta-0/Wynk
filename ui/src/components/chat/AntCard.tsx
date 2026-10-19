import { CheckCircle, CircleNotch, Crown, MinusCircle, XCircle } from "@phosphor-icons/react";
import { gatherSourceLabel, isLiveSource, STAGE_LABEL, type AntState, type Stage } from "@/lib/chat";
import { cn } from "@/lib/utils";
import { BlobCard } from "../ui/BlobCard";
import { AntTrail } from "./AntTrail";

interface Props {
  ant: AntState;
  picked: number | undefined;
  focused: boolean;
  onFocus: () => void;
}

/** One ant of the colony: ACO at work in the header, its stages and score below. */
export function AntCard({ ant, picked, focused, onFocus }: Props) {
  const running = ant.status.includes("running");
  const failed = ant.status.includes("failed");
  const chosen = picked === ant.id;
  const lost = picked !== undefined && !chosen;

  return (
    <button
      type="button"
      onClick={onFocus}
      aria-pressed={focused}
      aria-label={`Ant ${ant.id}, show its workflow`}
      className={cn(
        "block w-full cursor-pointer rounded-[22px] text-left transition-[opacity,transform] duration-300 ease-out active:scale-[0.99]",
        lost && "opacity-55 saturate-50",
        focused && "outline-2 outline-offset-4 outline-magenta",
      )}
    >
      <BlobCard
        header={
          <div>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-display text-2xl leading-none font-bold tracking-tight text-ink">Ant {ant.id}</p>
                <p className="mt-1.5 text-xs font-medium text-ink/80">{antLabel(ant)}</p>
              </div>
              {chosen ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-magenta-ink shadow-sm">
                  <Crown size={12} weight="fill" aria-hidden="true" /> Chosen
                </span>
              ) : (
                <span className="tnum rounded-full bg-white/85 px-2.5 py-1 font-mono text-xs text-ink">trail {ant.trail.toFixed(2)}</span>
              )}
            </div>
            <div className="mt-4">
              <AntTrail antId={ant.id} stages={ant.stages} status={ant.status} trail={ant.trail} chosen={chosen} />
            </div>
          </div>
        }
      >
        <ol className="space-y-1 px-5 pt-1 pb-3">
          {ant.stages.map((s, i) => {
            const st = ant.status[i];
            const source = ant.sources[i];
            return (
              <li key={i} className={cn("flex min-w-0 items-center gap-2 text-[13px]", st === "skipped" && "opacity-45")}>
                <StatusIcon status={st} />
                <span className="font-medium">{STAGE_LABEL[s.kind]}</span>
                <span className="truncate text-xs text-ink-soft">{optionsLabel(s, source)}</span>
                {isLiveSource(source) && (
                  <span className="shrink-0 rounded-full bg-ink px-1.5 py-px text-[10px] font-semibold tracking-wide text-white uppercase">live</span>
                )}
                {ant.toolCalls[i] > 0 && (
                  <span className="tnum ml-auto shrink-0 font-mono text-[11px] text-ink-soft">
                    {ant.toolCalls[i]} {ant.toolCalls[i] === 1 ? "tool call" : "tool calls"}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
        <div className="flex min-h-11 items-center justify-between gap-3 border-t border-line px-5 py-2.5 text-xs">
          {ant.score ? (
            <>
              <span className="min-w-0 truncate text-ink-soft">
                {!ant.score.completed
                  ? (ant.messages.find(Boolean) ?? "did not finish")
                  : ant.score.evidence
                    ? `quotes verified, ${ant.tokens} tokens`
                    : "finished, quotes not found"}
              </span>
              <span className="tnum font-mono text-sm font-semibold text-ink">{ant.score.score.toFixed(3)}</span>
            </>
          ) : failed ? (
            <span className="text-ink-soft">{ant.messages.find(Boolean)}</span>
          ) : (
            <span className="text-ink-soft">{running ? "walking its workflow" : "waiting"}</span>
          )}
        </div>
      </BlobCard>
    </button>
  );
}

/** A stage's options in plain words; a GATHER source is named for what it reads. */
export function optionsLabel(stage: Stage, source: string | undefined): string {
  if (stage.kind !== "GATHER" || source === undefined) return stage.options.join(", ");
  return [gatherSourceLabel(source), ...stage.options.slice(1)].join(", ");
}

export function antLabel(ant: Pick<AntState, "role" | "elite" | "elite_task">): string {
  if (ant.role !== "elite") return "Explorer: sampled from the trail";
  if (ant.elite === "task") return `Task elite: best validated workflow for ${ant.elite_task ?? "this fact sheet"}`;
  return "Suite elite: best recorded workflow";
}

function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <CircleNotch size={16} weight="bold" className="animate-spin text-magenta-ink motion-reduce:animate-none" aria-label="running" />;
  if (status === "ok") return <CheckCircle size={16} weight="fill" className="text-magenta" aria-label="done" />;
  if (status === "failed") return <XCircle size={16} weight="fill" className="text-ink" aria-label="failed" />;
  if (status === "skipped") return <MinusCircle size={16} className="text-ink-soft" aria-label="not reached" />;
  return <span className="mx-[3px] size-2.5 rounded-full border border-line-strong" aria-label="waiting" />;
}
