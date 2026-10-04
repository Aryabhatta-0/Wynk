import { CheckCircle, CircleNotch, Crown, MinusCircle, XCircle } from "@phosphor-icons/react";
import { STAGE_LABEL, type AntState } from "../../lib/chat";
import { cn } from "../../lib/cn";
import { BlobCard } from "../ui/BlobCard";

const BRAND = ["#ef75d3", "#e8227a", "#c23a9f", "#ff85b3"];
const MUTED = ["#e4d3df", "#f0e3ec", "#d9c6d4", "#f5ecf2"];
const GLOW = ["#ff96a9", "#e8b4f0", "#ffb3c6", "#d44d8a", "#ff96a9"];

interface Props {
  ant: AntState;
  picked: number | undefined;
  focused: boolean;
  onFocus: () => void;
}

/** One ant of the colony: the workflow it walked, live, and how its result scored. */
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
        "block w-full cursor-pointer rounded-[22px] text-left transition-[opacity,transform] duration-200 ease-out active:scale-[0.99]",
        lost && "opacity-60",
        focused && "ring-2 ring-magenta-ink ring-offset-2 ring-offset-wash",
      )}
    >
      <BlobCard
        headerHeight={104}
        lightColors={failed || lost ? MUTED : BRAND}
        glowColors={GLOW}
        glow={chosen || (running && picked === undefined)}
        header={
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-display text-xl font-bold tracking-tight text-ink">Ant {ant.id}</p>
              <p className="text-xs font-medium text-ink/75">{ant.role === "elite" ? "Elite: best recorded workflow" : "Explorer: sampled from the trail"}</p>
            </div>
            {chosen ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-plum px-2.5 py-1 text-xs font-semibold text-on-plum">
                <Crown size={12} weight="fill" aria-hidden="true" /> Chosen
              </span>
            ) : (
              <span className="tnum rounded-full bg-paper/80 px-2.5 py-1 font-mono text-xs text-ink">trail {ant.trail.toFixed(2)}</span>
            )}
          </div>
        }
      >
        <ol className="space-y-1.5 px-5 pt-1 pb-4">
          {ant.stages.map((s, i) => {
            const st = ant.status[i];
            return (
              <li key={i} className={cn("flex items-center gap-2.5 text-sm", st === "skipped" && "opacity-45")}>
                <StatusIcon status={st} />
                <span className="font-medium">{STAGE_LABEL[s.kind]}</span>
                <span className="truncate text-xs text-ink-soft">{s.options.join(", ")}</span>
              </li>
            );
          })}
        </ol>
        <div className="flex min-h-12 items-center justify-between gap-3 border-t border-line px-5 py-3 text-xs">
          {ant.score ? (
            <>
              <span className="text-ink-soft">
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

function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <CircleNotch size={16} weight="bold" className="animate-spin text-magenta-ink motion-reduce:animate-none" aria-label="running" />;
  if (status === "ok") return <CheckCircle size={16} weight="fill" className="text-magenta-ink" aria-label="done" />;
  if (status === "failed") return <XCircle size={16} weight="fill" className="text-ink" aria-label="failed" />;
  if (status === "skipped") return <MinusCircle size={16} className="text-ink-soft" aria-label="not reached" />;
  return <span className="mx-[3px] size-2.5 rounded-full border border-line-strong" aria-label="waiting" />;
}
